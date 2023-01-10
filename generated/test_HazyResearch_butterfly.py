import sys
_module = sys.modules[__name__]
del sys
butterfly = _module
benchmark = _module
benchmark_linear = _module
butterfly = _module
butterfly_multiply = _module
complex_utils = _module
setup = _module
setup = _module
permutation = _module
permutation_multiply = _module
utils = _module
benchmark_cnn = _module
cifar_experiment = _module
dataset_utils = _module
distill_cov_experiment = _module
distill_experiment = _module
distributed = _module
dataloaders = _module
logger = _module
mixup = _module
resnet = _module
smoothing = _module
training = _module
utils = _module
imagenet_amp = _module
imagenet_analysis = _module
imagenet_experiment = _module
imagenet_finetune = _module
imagenet_main = _module
imagenet_model_surgery = _module
mobilenet_imagenet = _module
model_utils = _module
models = _module
butterfly_conv = _module
circulant1x1conv = _module
densenet = _module
dpn = _module
googlenet = _module
layers = _module
lenet = _module
low_rank_conv = _module
mobilenet = _module
mobilenetv2 = _module
pnasnet = _module
preact_resnet = _module
presnet = _module
resnet = _module
resnet_imagenet = _module
resnet_original = _module
resnext = _module
senet = _module
shufflenet = _module
shufflenetv2 = _module
squeezenet = _module
toeplitzlike1x1conv = _module
vgg = _module
wide_resnet = _module
multiproc = _module
pdataset_utils = _module
permutation_utils = _module
permuted_experiment = _module
profile_perm = _module
shufflenet_imagenet = _module
teacher = _module
teacher_covariance = _module
train_utils = _module
transformer_analysis = _module
visualize_perm = _module
datamodules = _module
cifar = _module
lr_schedulers = _module
butterflenet = _module
cnn5 = _module
cnn5_butterfly = _module
kops = _module
lenet = _module
lops = _module
resnet = _module
resnet_cifar = _module
test_kops = _module
pl_runner = _module
ray_runner = _module
tasks = _module
tee = _module
train = _module
utils = _module
val_format = _module
my_sinkhorn_eval = _module
my_sinkhorn_ops = _module
my_sorting_model = _module
my_sorting_train = _module
learning_transforms = _module
baselines = _module
butterfly_factor = _module
butterfly_old = _module
fft_hadamard_analysis = _module
fisher = _module
heatmap = _module
hstack_diag = _module
inference = _module
learning_circulant = _module
learning_fft = _module
learning_hadamard = _module
learning_legendre = _module
learning_ops = _module
learning_transforms = _module
learning_vandermonde = _module
learning_fft_old = _module
ops = _module
permutation_factor = _module
polish = _module
print_results = _module
profile = _module
robust_pca = _module
semantic_loss = _module
sparsemax = _module
speed_plot = _module
speed_test = _module
speed_test_training = _module
speed_training_plot = _module
target_matrix = _module
test_factor_multiply = _module
training = _module
tune = _module
vandermonde = _module
setup = _module
test_butterfly = _module
test_butterfly_base4 = _module
test_combine = _module
test_complex_utils = _module
test_multiply = _module
test_multiply_base4 = _module
test_permutation = _module
test_special = _module
test_butterfly = _module
test_butterfly_multiply = _module
test_permutation = _module
test_permutation_multiply = _module
torch_butterfly = _module
benchmark_utils = _module
butterfly = _module
butterfly_base4 = _module
combine = _module
complex_utils = _module
diagonal = _module
input_padding_benchmark = _module
multiply = _module
multiply_base4 = _module
permutation = _module
special = _module
dynamic_conv_experiment = _module

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


import time


import math


from torch import nn


import torch.nn.functional as F


import numpy as np


from torch.utils.dlpack import to_dlpack


from torch.utils.dlpack import from_dlpack


import torch.cuda


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDA_HOME


import random


from torch import optim


import torchvision


import torchvision.transforms as transforms


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import torch.distributed as dist


from torch.nn.modules import Module


import torchvision.datasets as datasets


import torch.nn as nn


import torchvision.models as models


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import warnings


from torch.autograd import Variable


import logging


import torchvision.models as torch_models


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


import torch.nn.init as init


import matplotlib


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torchvision import transforms


from torchvision import datasets


import torch.fft


from collections.abc import Sequence


from functools import partial


from torch.nn import functional as F


import numpy


from scipy.optimize import linear_sum_assignment


from scipy.stats import kendalltau


import functools


from scipy.linalg import circulant


from scipy.linalg import hadamard


from numpy.polynomial import legendre


from numpy.polynomial import chebyshev


from scipy.fftpack import fft


from scipy.fftpack import dct


from scipy.fftpack import dst


import scipy.linalg as LA


from scipy.fftpack import fft2


import scipy.sparse as sparse


import copy


import itertools


from scipy import linalg as la


import scipy.fft


import numbers


from typing import Tuple


from typing import Optional


from typing import List


from typing import Union


import scipy.linalg


from functools import reduce


import re


def complex_reshape(x, *shape):
    if not x.is_complex():
        return x.reshape(*shape)
    else:
        return torch.view_as_complex(torch.view_as_real(x).reshape(*shape, 2))


real_dtype_to_complex = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def twiddle_base2_to_base4(twiddle, increasing_stride=True):
    nstacks, nblocks, log_n = twiddle.shape[:3]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    twiddle2 = twiddle[:, :, -1:] if log_n % 2 == 1 else torch.empty(nstacks, nblocks, 0, n // 2, 2, 2, dtype=twiddle.dtype, device=twiddle.device)
    twiddle4 = torch.empty(nstacks, nblocks, log_n // 2, n // 4, 4, 4, dtype=twiddle.dtype, device=twiddle.device)
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n // 2):
            log2_stride = 2 * idx if cur_increasing_stride else log_n - 2 - 2 * idx
            stride = 1 << log2_stride
            even = twiddle[:, block, 2 * idx].view(nstacks, n // (4 * stride), 2, stride, 2, 2).transpose(-3, -4)
            odd = twiddle[:, block, 2 * idx + 1].view(nstacks, n // (4 * stride), 2, stride, 2, 2).transpose(-3, -4)
            if cur_increasing_stride:
                prod = odd.transpose(-2, -3).unsqueeze(-1) * even.transpose(-2, -3).unsqueeze(-4)
            else:
                prod = odd.unsqueeze(-2) * even.permute(0, 1, 2, 4, 5, 3).unsqueeze(-3)
            prod = prod.reshape(nstacks, n // 4, 4, 4)
            twiddle4[:, block, idx].copy_(prod)
        cur_increasing_stride = not cur_increasing_stride
    return twiddle4, twiddle2


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: a torch.Tensor, or 'randn', 'ortho', 'identity', 'fft_no_br', or 'ifft_no_br'.
            Whether the weight matrix should be initialized to from randn twiddle, or to be
            randomly orthogonal/unitary, or to be the identity matrix, or the normalized FFT/iFFT
            twiddle (without the bit-reversal permutation).
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False, increasing_stride=True, init='randn', nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = self.nstacks, nblocks, log_n, n // 2, 2, 2
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['empty', 'randn', 'ortho', 'identity', 'fft_no_br', 'ifft_no_br']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)
        twiddle = self.twiddle
        if self.init is None or self.init == 'empty':
            return
        elif self.init == 'randn':
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            if not self.complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((det * c, -det * s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2))
            else:
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.rand((3,) + twiddle_core_shape) * math.pi * 2
                A = torch.exp(1.0j * (alpha + psi)) * c
                B = torch.exp(1.0j * (alpha + chi)) * s
                C = -torch.exp(1.0j * (alpha - chi)) * s
                D = torch.exp(1.0j * (alpha - psi)) * c
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((A, B), dim=-1), torch.stack((C, D), dim=-1)), dim=-2))
        elif self.init == 'identity':
            twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_eye = twiddle_eye.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_eye)
        elif self.init in ['fft_no_br', 'ifft_no_br']:
            assert self.complex, 'fft_no_br/ifft_no_br init requires Butterfly to be complex'
            special_fn = torch_butterfly.special.fft if self.init == 'fft_no_br' else torch_butterfly.special.ifft
            b_fft = special_fn(self.n, normalized=True, br_first=self.increasing_stride, with_br_perm=False)
            with torch.no_grad():
                twiddle[:, 0] = b_fft.twiddle
            if self.nblocks > 1:
                twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
                twiddle_eye = twiddle_eye.expand(*twiddle[:, 1:].shape).contiguous()
                with torch.no_grad():
                    twiddle[:, 1:] = twiddle_eye

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = twiddle[:, :, :log_n, :n // 2] if self.increasing_stride else twiddle[:, :, -log_n:, :n // 2]
            output_size = None
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input):
        input_size = input.size(-1)
        output = complex_reshape(input, -1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:
            output = output[:, :out_size]
        if self.bias is not None:
            output = output + self.bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        self.twiddle *= scale ** (1.0 / self.twiddle.shape[1] / self.twiddle.shape[2])
        return self

    def diagonal_multiply_(self, diagonal, diag_first):
        """ Combine a Butterfly and a diagonal into another Butterfly.
        Only support nstacks==1 for now.
        Parameters:
            diagonal: size (in_size,) if diag_first, else (out_size,). Should be of type complex
                if butterfly.complex == True.
            diag_first: If True, the map is input -> diagonal -> butterfly.
                If False, the map is input -> butterfly -> diagonal.
        """
        return torch_butterfly.combine.diagonal_butterfly(self, diagonal, diag_first, inplace=True)

    def to_base4(self):
        with torch.no_grad():
            twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        new = torch_butterfly.ButterflyBase4(self.in_size, self.out_size, self.bias is not None, self.complex, self.increasing_stride, init=(twiddle4, twiddle2), nblocks=self.nblocks)
        if new.bias is not None:
            with torch.no_grad():
                new.bias.copy_(self.bias)
        return new

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'.format(self.in_size, self.out_size, self.bias is not None, self.complex, self.increasing_stride, self.init, self.nblocks)
        return s


class ButterflyBmm(Butterfly):
    """Same as Butterfly, but performs batched matrix multiply.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        matrix_batch: how many butterfly matrices
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, matrix_batch=1, bias=True, complex=False, increasing_stride=True, init='randn', nblocks=1):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.matrix_batch = matrix_batch
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = self.matrix_batch * self.nstacks, nblocks, log_n, n // 2, 2, 2
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['randn', 'ortho', 'identity', 'fft_no_br', 'ifft_no_br']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.matrix_batch, out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True
        self.reset_parameters()

    def forward(self, input, transpose=False, conjugate=False):
        """
        Parameters:
            input: (batch, *, matrix_batch, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
        Return:
            output: (batch, *, matrix_batch, out_size)
        """
        return super().forward(input, transpose, conjugate, subtwiddle=False)

    def pre_process(self, input):
        input_size = input.size(-1)
        assert input.size(-2) == self.matrix_batch
        output = complex_reshape(input, -1, self.matrix_batch, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(2).expand(batch, self.matrix_batch, self.nstacks, input_size)
        output = output.reshape(batch, self.matrix_batch * self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.matrix_batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:
            output = output[:, :, :out_size]
        if self.bias is not None:
            output = output + self.bias[:, :out_size]
        return output.view(*input.size()[:-2], self.matrix_batch, self.out_size)
    to_base4 = None

    def extra_repr(self):
        s = 'in_size={}, out_size={}, matrix_batch={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'.format(self.in_size, self.out_size, self.matrix_batch, self.bias is not None, self.complex, self.increasing_stride, self.init, self.nblocks)
        return s


class Permutation(nn.Module):

    def forward(self, x, samples=1):
        soft_perms = self.sample_soft_perm((samples, x.size(0)))
        return x.unsqueeze(0) @ soft_perms

    def mean_perm(self):
        pass

    def sample_soft_perm(self, sample_shape=()):
        """ Return soft permutation of shape sample_shape + (size, size) """
        pass


class IndexLastDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, permutation):
        ctx.save_for_backward(permutation)
        return X[..., permutation]

    @staticmethod
    def backward(ctx, grad):
        permutation, = ctx.saved_tensors
        output = torch.empty_like(grad)
        output[..., permutation] = grad
        return output, None


index_last_dim = IndexLastDim.apply


def bitreversal_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in FFT.
    By default, the permutation is stored in numpy array.
    Parameter:
        n: integer, must be a power of 2.
        pytorch_format: whether the permutation is stored as numpy array or pytorch tensor.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(log_n):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    perm = perm.squeeze(0)
    return perm if not pytorch_format else torch.tensor(perm)


class Node:

    def __init__(self, value):
        self.value = value
        self.in_edges = []
        self.out_edges = []


def half_balance(v: np.ndarray, return_swap_locations: bool=False) ->Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
    """Return the permutation vector that makes the permutation vector v
    n//2-balanced. Directly follows the proof of Lemma G.2.
    Parameters:
        v: the permutation as a vector, stored in right-multiplication format.
    """
    n = len(v)
    assert n % 2 == 0
    nh = n // 2
    nodes = [Node(i) for i in range(nh)]
    for i in range(nh):
        s, t = nodes[v[i] % nh], nodes[v[i + nh] % nh]
        s.out_edges.append((t, i))
        t.in_edges.append((s, i + nh))
    assert all(len(node.in_edges) + len(node.out_edges) == 2 for node in nodes)
    swap_low_locs = []
    swap_high_locs = []
    while len(nodes):
        start_node, start_loc = nodes[-1], n - 1
        next_node = None
        while next_node != start_node:
            if next_node is None:
                next_node, next_loc = start_node, start_loc
            old_node, old_loc = next_node, next_loc
            if old_node.out_edges:
                next_node, old_loc = old_node.out_edges.pop()
                next_loc = old_loc + nh
                next_node.in_edges.remove((old_node, next_loc))
            else:
                next_node, old_loc = old_node.in_edges.pop()
                next_loc = old_loc - nh
                next_node.out_edges.remove((old_node, next_loc))
                swap_low_locs.append(next_loc)
                swap_high_locs.append(old_loc)
            nodes.remove(old_node)
    perm = np.arange(n, dtype=int)
    perm[swap_low_locs], perm[swap_high_locs] = swap_high_locs, swap_low_locs
    if not return_swap_locations:
        return perm, v[perm]
    else:
        return swap_low_locs, v[perm]


def invert(perm: Union[np.ndarray, torch.Tensor]) ->Union[np.ndarray, torch.Tensor]:
    """Get the inverse of a given permutation vector.
    Equivalent to converting a permutation vector from left-multiplication format to right
    multiplication format.
    Work with both numpy array and Pytorch Tensor.
    """
    assert isinstance(perm, (np.ndarray, torch.Tensor))
    n = perm.shape[-1]
    if isinstance(perm, np.ndarray):
        result = np.empty(n, dtype=int)
        result[perm] = np.arange(n, dtype=int)
    else:
        result = torch.empty(n, dtype=int, device=perm.device)
        result[perm] = torch.arange(n, dtype=int)
    return result


def swap_locations_to_twiddle_factor(n: int, swap_locations: np.ndarray) ->torch.Tensor:
    twiddle = torch.eye(2).expand(n // 2, 2, 2).contiguous()
    swap_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    twiddle[swap_locations] = swap_matrix.unsqueeze(0)
    return twiddle


def outer_twiddle_factors(v: np.ndarray) ->Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Decompose the permutations v to get the right/right twiddle factor, and new permutations
    that only permute elements that are size//2 indices apart.
    Parameters:
        v: (batch_size, size), each is a permutation vector of size @size, in left-multiplication
            format.
    Return:
        twiddle_right_factor: (batch_size * size // 2, 2, 2)
        twiddle_left_factor: (batch_size * size // 2, 2, 2)
        new_v: (batch_size * 2, size // 2)
    """
    batch_size, size = v.shape
    assert size >= 2
    v_right = np.vstack([invert(chunk) for chunk in v])
    half_balance_results = [half_balance(chunk, return_swap_locations=True) for chunk in v_right]
    twiddle_right_factor = torch.cat([swap_locations_to_twiddle_factor(size, swap_low_locs) for swap_low_locs, _ in half_balance_results])
    v_right = np.vstack([v_permuted for _, v_permuted in half_balance_results])
    v_left = np.vstack([invert(perm) for perm in v_right])
    size_half = size // 2
    swap_low_x, swap_low_y = np.nonzero(v_left[:, :size_half] // size_half == 1)
    swap_low_locs_flat = swap_low_y + swap_low_x * size // 2
    twiddle_left_factor = swap_locations_to_twiddle_factor(batch_size * size, swap_low_locs_flat)
    v_left[swap_low_x, swap_low_y], v_left[swap_low_x, swap_low_y + size_half] = v_left[swap_low_x, swap_low_y + size // 2], v_left[swap_low_x, swap_low_y]
    new_v = (v_left % size_half).reshape(batch_size * 2, size // 2)
    assert np.allclose(np.sort(new_v), np.arange(size // 2))
    return twiddle_right_factor, twiddle_left_factor, new_v


class Real2ComplexFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return X

    @staticmethod
    def backward(ctx, grad):
        return grad.real


real2complex = Real2ComplexFn.apply


def perm2butterfly(v: Union[np.ndarray, torch.Tensor], complex: bool=False, increasing_stride: bool=False) ->Butterfly:
    """
    Parameter:
        v: a permutation, stored as a vector, in left-multiplication format.
            (i.e., applying v to a vector x is equivalent to x[p])
        complex: whether the Butterfly is complex or real.
        increasing_stride: whether the returned Butterfly should have increasing_stride=False or
            True. False corresponds to Lemma G.3 and True corresponds to Lemma G.6.
    Return:
        b: a Butterfly that performs the same permutation as v.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    n = len(v)
    log_n = int(math.ceil(math.log2(n)))
    if n < 1 << log_n:
        v = np.concatenate([v, np.arange(n, 1 << log_n)])
    if increasing_stride:
        br = bitreversal_permutation(1 << log_n)
        b = perm2butterfly(br[v[br]], complex=complex, increasing_stride=False)
        b.increasing_stride = True
        br_half = bitreversal_permutation((1 << log_n) // 2, pytorch_format=True)
        with torch.no_grad():
            b.twiddle.copy_(b.twiddle[:, :, :, br_half])
        b.in_size = b.out_size = n
        return b
    v = v[None]
    twiddle_right_factors, twiddle_left_factors = [], []
    for _ in range(log_n):
        right_factor, left_factor, v = outer_twiddle_factors(v)
        twiddle_right_factors.append(right_factor)
        twiddle_left_factors.append(left_factor)
    twiddle = torch.stack([torch.stack(twiddle_right_factors), torch.stack(twiddle_left_factors).flip([0])]).unsqueeze(0)
    b = Butterfly(n, n, bias=False, complex=complex, increasing_stride=False, init=twiddle if not complex else real2complex(twiddle), nblocks=2)
    return b


class FixedPermutation(nn.Module):

    def __init__(self, permutation: torch.Tensor) ->None:
        """Fixed permutation.
        Parameter:
            permutation: (n, ) tensor of ints
        """
        super().__init__()
        self.register_buffer('permutation', permutation)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        return index_last_dim(input, self.permutation)

    def to_butterfly(self, complex=False, increasing_stride=False):
        return perm2butterfly(self.permutation, complex, increasing_stride)


class PermutationFactorEvenOddMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        ctx.save_for_backward(p, input)
        return permutation_factor_even_odd_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_even_odd_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_even_odd_mult = PermutationFactorEvenOddMult.apply


class PermutationFactorReverseMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        ctx.save_for_backward(p, input)
        return permutation_factor_reverse_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_reverse_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_reverse_mult = PermutationFactorReverseMult.apply


def permutation_mult_single_factor(prob, input):
    """Multiply by a single permutation factor, parameterized by the probabilities.
    Parameters:
        prob: (3, ), where prob[0] is the probability of separating the even and odd indices,
            and prob[1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, 'size must be a power of 2'
    assert prob.shape == (3,)
    output = input.contiguous()
    output = permutation_factor_even_odd_mult(prob[:1], output)
    output = permutation_factor_reverse_mult(prob[1:], output)
    return output


def permutation_mult_single_factor_torch(prob, input):
    """Multiply by a single permutation factor.
    Parameters:
        prob: (3, ), where prob[0] is the probability of separating the even and odd indices,
            and prob[1:3] are the probabilities of reversing the 1st and 2nd halves respectively.
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, 'size must be a power of 2'
    assert prob.shape == (3,)
    output = input.contiguous()
    if input.dim() == 2:
        stride = n // 2
        output = (1 - prob[0]) * output.view(-1, 2, stride) + prob[0] * output.view(-1, stride, 2).transpose(-1, -2)
        output = (1 - prob[1:]).unsqueeze(-1) * output + prob[1:].unsqueeze(-1) * output.flip(-1)
        return output.view(batch_size, n)
    else:
        stride = n // 2
        output = (1 - prob[0]) * output.view(-1, 2, stride, 2) + prob[0] * output.view(-1, stride, 2, 2).transpose(-2, -3)
        output = (1 - prob[1:]).unsqueeze(-1).unsqueeze(-1) * output + prob[1:].unsqueeze(-1).unsqueeze(-1) * output.flip(-2)
        return output.view(batch_size, n, 2)


use_extension = True


permutation_mult_single = permutation_mult_single_factor if use_extension else permutation_mult_single_factor_torch


class PermutationFactor(nn.Module):
    """A single permutation factor.

    Parameters:
        size: size of input (and of output)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        m = int(math.ceil(math.log2(size)))
        assert size == 1 << m, 'size must be a power of 2'
        self.logit = nn.Parameter(torch.randn(3))

    def forward(self, input):
        """
        Parameters:
            input: (batch, size) if real or (batch, size, 2) if complex
        Return:
            output: (batch, size) if real or (batch, size, 2) if complex
        """
        prob = torch.sigmoid(self.logit)
        return permutation_mult_single(prob, input)

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        prob = torch.sigmoid(self.logit).round()
        input = torch.arange(self.size, dtype=prob.dtype, device=self.logit.device).unsqueeze(0)
        return permutation_mult_single(prob, input).squeeze(0).round().long()

    def extra_repr(self):
        return 'size={}'.format(self.size)


def flat_dist_call(tensors, call, extra_args=None):
    flat_dist_call.warn_on_half = True
    buckets = {}
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    if flat_dist_call.warn_on_half:
        if torch.HalfTensor in buckets:
            None
            flat_dist_call.warn_on_half = False
    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        coalesced /= dist.get_world_size()
        for buf, synced in zip(bucket, _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)


class NLLMultiLabelSmooth(nn.Module):

    def __init__(self, smoothing=0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class KnowledgeDistillationLoss(nn.Module):
    """
    Loss with knowledge distillation.
    """

    def __init__(self, original_loss, temperature=1.0, alpha_ce=0.5):
        super().__init__()
        self.original_loss = original_loss
        self.temperature = temperature
        self.alpha_ce = alpha_ce

    def forward(self, s_logit, t_logit, target):
        loss_kd = F.kl_div(F.log_softmax(s_logit / self.temperature, dim=-1), F.softmax(t_logit / self.temperature, dim=-1), reduction='batchmean') * self.temperature ** 2
        loss_og = self.original_loss(s_logit, target)
        return (1 - self.alpha_ce) * loss_og + self.alpha_ce * loss_kd


class Block(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * group_width))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MobileNet(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10, num_structured_layers=0, structure_type='B', nblocks=0, param='regular'):
        assert structure_type in ['B', 'LR', 'Circulant', 'Toeplitzlike']
        assert num_structured_layers <= len(self.cfg)
        super(MobileNet, self).__init__()
        self.structure_type = structure_type
        self.param = param
        self.nblocks = nblocks
        self.is_structured = [False] * (len(self.cfg) - num_structured_layers) + [True] * num_structured_layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x, is_structured in zip(self.cfg, self.is_structured):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, is_structured, structure_type=self.structure_type, param=self.param, nblocks=self.nblocks))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ShuffleNet(nn.Module):

    def __init__(self, num_classes=1000, groups=8, width_mult=1.0, shuffle='P', preact=False):
        super(ShuffleNet, self).__init__()
        num_blocks = [4, 8, 4]
        groups_to_outplanes = {(1): [144, 288, 576], (2): [200, 400, 800], (3): [240, 480, 960], (4): [272, 544, 1088], (8): [384, 768, 1536]}
        out_planes = groups_to_outplanes[groups]
        out_planes = [_make_divisible(p * width_mult, groups) for p in out_planes]
        input_channel = _make_divisible(24 * width_mult, groups)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.in_planes = input_channel
        self.stage2 = self._make_layer(out_planes[0], num_blocks[0], groups, grouped_conv_1st_layer=False, shuffle=shuffle, preact=preact)
        self.stage3 = self._make_layer(out_planes[1], num_blocks[1], groups, shuffle=shuffle, preact=preact)
        self.stage4 = self._make_layer(out_planes[2], num_blocks[2], groups, shuffle=shuffle, preact=preact)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups, grouped_conv_1st_layer=True, shuffle='P', preact=False):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups, grouped_conv_1st_layer=grouped_conv_1st_layer, shuffle=shuffle, preact=preact))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = out.mean([2, 3])
        out = self.linear(out)
        return out


class ModelAndLoss(nn.Module):

    def __init__(self, arch, loss, pretrained_weights=None, cuda=True, fp16=False, width=1.0, n_struct_layers=0, struct='D', softmax_struct='D', sm_pooling=1, groups=8, shuffle='P'):
        super(ModelAndLoss, self).__init__()
        self.arch = arch
        None
        if arch == 'mobilenetv1':
            model = MobileNet(width_mult=width, structure=[struct] * n_struct_layers, softmax_structure=softmax_struct, sm_pooling=sm_pooling)
        elif arch == 'shufflenetv1':
            model = ShuffleNet(width_mult=width, groups=groups, shuffle=shuffle)
        else:
            model = models.__dict__[arch]()
        if pretrained_weights is not None:
            None
            model.load_state_dict(pretrained_weights)
        if cuda:
            model = model
        if fp16:
            model = network_to_half(model)
        criterion = loss()
        if cuda:
            criterion = criterion
        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        if hasattr(self, '_teacher_model'):
            with torch.no_grad():
                teacher_output = self._teacher_model(data)
            loss = self.loss(output, teacher_output, target)
        else:
            loss = self.loss(output, target)
        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


class Butterfly1x1Conv(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    """

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)


class BbtMultConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding):
        """
        Parameters:
            twiddle: (nstack, nblocks * 2 * log n, n/2, 2, 2) where n = c_in
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels
            padding: amount of zero-padding around border of input
        Returns:
            output: (b_in * h_out * w_out, nstack, c_in)
        """
        output = bbt_conv2d(twiddle, input, kernel_size, padding)
        ctx.save_for_backward(twiddle, input)
        ctx._kernel_size = kernel_size
        ctx._padding = padding
        ctx._input_size = input.size()
        ctx._b_in = input.size(0)
        ctx._c_in = input.size(1)
        ctx._h_in = input.size(2)
        ctx._w_in = input.size(3)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (b_in * h_out * w_out, cin/cout * nstack, c_out)
            twiddle: (nstack, nblocks * 2 * log n, n / 2, 2, 2) where n = c_in
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (b_in, c_in, h_in, w_in)
        """
        twiddle, input = ctx.saved_tensors
        d_coefficients, d_input = bbt_conv2d_forward_backward(twiddle, input, grad, ctx._kernel_size, ctx._padding)
        return d_coefficients, d_input, None, None


class ButterflyMultConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels
            padding: amount of zero-padding around border of input
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                    decreasing stride (e.g., n/2, n/4, ..., 1).
                    Note that this only changes the order of multiplication, not how twiddle is stored.
                    In other words, twiddle[@log_stride] always stores the twiddle for @stride.
            return_intermediates: whether to return all the intermediate values computed
        Returns:
            output: (b_in * h_out * w_out, nstack, c_in)
        """
        output = butterfly_conv2d(twiddle, input, kernel_size, padding, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._kernel_size = kernel_size
        ctx._padding = padding
        ctx._increasing_stride = increasing_stride
        ctx._input_size = input.size()
        ctx._b_in = input.size(0)
        ctx._c_in = input.size(1)
        ctx._h_in = input.size(2)
        ctx._w_in = input.size(3)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (b_in * h_out * w_out, cin/cout * nstack, c_out)
            twiddle: (nstack, log n, n / 2, 2, 2) where n = c_in
            output + intermediate values for backward: (log n + 1, b_in * h_out * w_out,
                                                cin/cout * nstack, c_out)
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (b_in, c_in, h_in, w_in)
        """
        twiddle, input = ctx.saved_tensors
        d_coefficients, d_input = butterfly_conv2d_forward_backward(twiddle, input, grad, ctx._kernel_size, ctx._padding, ctx._increasing_stride)
        return d_coefficients, d_input, None, None, None


def complex_mul_torch(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack((X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1], X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]), dim=-1)


def conjugate_torch(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def cupy2torch(tensor):
    return from_dlpack(tensor.toDlpack())


def torch2cupy(tensor):
    return cp.fromDlpack(to_dlpack(tensor))


def torch2numpy(X):
    """Convert a torch float32 tensor to a numpy array, sharing the same memory.
    """
    return X.detach().numpy()


use_cupy = False


class Conjugate(torch.autograd.Function):
    """X is a complex64 tensors but stored as float32 tensors, with last dimension = 2.
    """

    @staticmethod
    def forward(ctx, X):
        assert X.shape[-1] == 2, 'Last dimension must be 2'
        if X.is_cuda:
            if use_cupy:
                return cupy2torch(torch2cupy(X).view('complex64').conj().view('float32'))
            else:
                return conjugate_torch(X)
        else:
            return torch.from_numpy(np.ascontiguousarray(torch2numpy(X)).view('complex64').conj().view('float32'))

    @staticmethod
    def backward(ctx, grad):
        return Conjugate.apply(grad)


conjugate = Conjugate.apply


class ComplexMul(torch.autograd.Function):
    """X and Y are complex64 tensors but stored as float32 tensors, with last dimension = 2.
    """

    @staticmethod
    def forward(ctx, X, Y):
        assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
        ctx.save_for_backward(X, Y)
        if X.is_cuda:
            if use_cupy:
                return cupy2torch((torch2cupy(X).view('complex64') * torch2cupy(Y).view('complex64')).view('float32'))
            else:
                return complex_mul_torch(X, Y)
        else:
            X_np = np.ascontiguousarray(torch2numpy(X)).view('complex64')
            Y_np = np.ascontiguousarray(torch2numpy(Y)).view('complex64')
            return torch.from_numpy((X_np * Y_np).view('float32'))

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = ComplexMul.apply(grad, conjugate(Y)), ComplexMul.apply(grad, conjugate(X))
        dims_to_sum_X = [(-i) for i in range(1, X.dim() + 1) if X.shape[-i] != grad.shape[-i]]
        dims_to_sum_Y = [(-i) for i in range(1, Y.dim() + 1) if Y.shape[-i] != grad.shape[-i]]
        if dims_to_sum_X:
            grad_X = grad_X.sum(dim=dims_to_sum_X, keepdim=True)
        if dims_to_sum_Y:
            grad_Y = grad_Y.sum(dim=dims_to_sum_Y, keepdim=True)
        if grad.dim() > X.dim():
            grad_X = grad_X.sum(tuple(range(grad.dim() - X.dim())))
        if grad.dim() > Y.dim():
            grad_Y = grad_Y.sum(tuple(range(grad.dim() - Y.dim())))
        return grad_X, grad_Y


complex_mul = ComplexMul.apply


def butterfly_mult_untied_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
        input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, nstack, n = input.shape[:3]
    m = int(math.log2(n))
    assert n == 1 << m, 'size must be a power of 2'
    assert twiddle.shape == (nstack, m, n // 2, 2, 2) if input.dim() == 3 else (nstack, m, n // 2, 2, 2, 2)
    if input.dim() == 3:
        output = input.contiguous()
        intermediates = [output]
        for log_stride in (range(m) if increasing_stride else range(m)[::-1]):
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n) for intermediate in intermediates])
    else:
        output = input.contiguous()
        intermediates = [output]
        for log_stride in (range(m) if increasing_stride else range(m)[::-1]):
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2, 2).permute(0, 1, 3, 4, 2, 5)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t, output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n, 2) for intermediate in intermediates])


def butterfly_mult_conv2d_torch(twiddle, input, kernel_size, padding, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
        input: (b_in, c_in, h_in, w_in)
        kernel_size: int, size of convolution kernel, currently only supports square kernels
        padding: amount of zero-padding around border of input
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed
    Returns:
        output: (b_in * h_out * w_out, nstack, c_in)
    """
    b_in, c_in, h_in, w_in = input.shape
    c_out = twiddle.size(0) // (kernel_size * kernel_size) * c_in
    assert c_in == 1 << int(math.log2(c_in)), 'currently requires c_in to be a power of 2'
    assert c_out == 1 << int(math.log2(c_out)), 'currently requires c_out to be a power of 2'
    h_out = h_in + 2 * padding - (kernel_size - 1)
    w_out = w_in + 2 * padding - (kernel_size - 1)
    matrix_batch = kernel_size * kernel_size
    c_out_ratio = c_out // c_in
    assert c_out_ratio >= 1, 'only tested for c_out >= c_in'
    input_patches = F.unfold(input, kernel_size=kernel_size, dilation=1, padding=padding, stride=1).view(b_in, c_in, kernel_size * kernel_size, h_out * w_out)
    input_reshape = input_patches.permute(0, 3, 2, 1).reshape(b_in * h_out * w_out, matrix_batch, c_in)
    input_reshape = input_reshape.unsqueeze(2).expand(b_in * h_out * w_out, matrix_batch, c_out_ratio, c_in)
    input_reshape = input_reshape.reshape(b_in * h_out * w_out, matrix_batch * c_out_ratio, c_in)
    return butterfly_mult_untied_torch(twiddle, input_reshape, increasing_stride, return_intermediates)


butterfly_mult_conv2d = ButterflyMultConv2d.apply if use_extension else butterfly_mult_conv2d_torch


class ButterflyMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True, is_training=True, fast=False):
        """
        Parameters:
            twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        if not is_training and not input.is_cuda and input.dim() == 3 and input.dtype == torch.float and input.shape[-1] > 8:
            output = butterfly_multiply_untied_eval(twiddle, input, increasing_stride)
        elif not fast:
            output = butterfly_multiply_untied(twiddle, input, increasing_stride, False)
        else:
            output = butterfly_multiply_untied_forward_fast(twiddle, input, increasing_stride)
        ctx.save_for_backward(twiddle, input)
        ctx._increasing_stride = increasing_stride
        ctx._fast = fast
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        twiddle, input = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        fast = ctx._fast
        n = input.shape[2]
        if input.dim() == 3 and n <= 1024 and input.is_cuda:
            if not fast:
                d_coefficients, d_input = butterfly_multiply_untied_forward_backward(twiddle, input, grad, increasing_stride)
            else:
                d_coefficients, d_input = butterfly_multiply_untied_forward_backward_fast(twiddle, input, grad, increasing_stride)
        else:
            output_and_intermediate = butterfly_multiply_untied(twiddle, input, increasing_stride, True)
            d_coefficients, d_input = butterfly_multiply_untied_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None, None, None


butterfly_mult_untied = ButterflyMultUntied.apply if use_extension else butterfly_mult_untied_torch


def bbt_mult_conv2d(twiddle, input, kernel_size, padding):
    n = input.shape[1]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    assert nblocks * 2 * m == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2 * log n, n / 2, 2, 2)'
    if n <= 1024 and input.is_cuda and nblocks <= 14:
        return BbtMultConv2d.apply(twiddle, input, kernel_size, padding)
    else:
        output = input
        reverse_idx = torch.arange(m - 1, -1, -1, device=twiddle.device)
        first = True
        for t in twiddle.chunk(nblocks, dim=1):
            if first:
                output = butterfly_mult_conv2d(t[:, reverse_idx], output, kernel_size, padding, False)
                first = False
            else:
                output = butterfly_mult_untied(t[:, reverse_idx], output, False, True, False)
            output = butterfly_mult_untied(t[:, m:], output, True, True, False)
        return output


class ButterflyConv2d(ButterflyBmm):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        **kwargs: args to ButterflyBmm, see Butterfly class
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, fused_unfold=False, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.fused_unfold = fused_unfold
        super().__init__(in_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], complex=False, **kwargs)
        if self.bias is not None:
            self.bias_conv = nn.Parameter(self.bias[0].clone())
            self.bias = None

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        if True:
            input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
            input = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
            output = super().forward(input)
        else:
            batch_out = batch * h_out * w_out
            if self.param == 'regular':
                if self.nblocks == 0:
                    output = butterfly_mult_conv2d(self.twiddle, input, self.kernel_size[0], self.padding[0], self.increasing_stride)
                else:
                    output = bbt_mult_conv2d(self.twiddle, input, self.kernel_size[0], self.padding[0])
            elif self.param == 'ortho':
                c, s = torch.cos(self.twiddle), torch.sin(self.twiddle)
                twiddle = torch.stack((torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2)
                output = butterfly_mult_conv2d(self.twiddle, input, self.kernel_size[0], self.padding[0], self.increasing_stride)
            elif self.param == 'svd':
                with torch.no_grad():
                    self.twiddle[..., 1, :].clamp_(min=1 / self.max_gain_per_factor, max=self.max_gain_per_factor)
                output = butterfly_mult_conv2d_svd(self.twiddle, input, self.kernel_size[0], self.padding[0], self.increasing_stride)
            output = super().post_process(input, output)
        output = output.mean(dim=1)
        if hasattr(self, 'bias_conv'):
            output = output + self.bias_conv
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)


class ButterflyConv2dBBT(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        nblocks: number of BBT blocks in the product
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular' or 'ortho' or 'svd'.
            'ortho' and 'svd' only support real, not complex.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole BB^T matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, tied_weight=True, nblocks=1, ortho_init=False, param='regular', max_gain=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.nblocks = nblocks
        max_gain_per_block = max_gain ** (1 / (2 * nblocks))
        layers = []
        for i in range(nblocks):
            layers.append(ButterflyBmm(in_channels if i == 0 else out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain_per_block))
            layers.append(ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, bias if i == nblocks - 1 else False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain_per_block))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
        output = self.layers(input_reshape).mean(dim=1)
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)


class ButterflyConv2dBBTBBT(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular' or 'ortho' or 'svd'.
            'ortho' and 'svd' only support real, not complex.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole BB^T BB^T matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, tied_weight=True, ortho_init=False, param='regular', max_gain=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.layers = nn.Sequential(ButterflyBmm(in_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)), ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)), ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)), ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], bias, False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)))

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
        output = self.layers(input_reshape).mean(dim=1)
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)


class CirculantLinear(nn.Module):

    def __init__(self, size, nstack=1):
        super().__init__()
        self.size = size
        self.nstack = nstack
        init_stddev = math.sqrt(1.0 / self.size)
        c = torch.randn(nstack, size) * init_stddev
        self.c_f = nn.Parameter(torch.rfft(c, 1))
        self.c_f._is_structured = True

    def forward(self, input):
        """
        Parameters:
            input: (batch, size)
        Return:
            output: (batch, nstack * size)
        """
        batch = input.shape[0]
        input_f = torch.rfft(input, 1)
        prod = complex_mul(self.c_f, input_f.unsqueeze(1))
        return torch.irfft(prod, 1, signal_sizes=(self.size,)).view(batch, self.nstack * self.size)


class Circulant1x1Conv(CirculantLinear):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, nstack * c, h, w)
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DPN(nn.Module):

    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1), nn.BatchNorm2d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1), nn.BatchNorm2d(n3x3red), nn.ReLU(True), nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), nn.BatchNorm2d(n3x3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1), nn.BatchNorm2d(n5x5red), nn.ReLU(True), nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True), nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_planes, pool_planes, kernel_size=1), nn.BatchNorm2d(pool_planes), nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class LeNet(nn.Module):
    name = 'lenet'

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class MLP(nn.Module):

    def __init__(self, method='linear', **kwargs):
        super().__init__()
        if method == 'linear':
            make_layer = lambda name: self.add_module(name, nn.Linear(1024, 1024, bias=True))
        elif method == 'butterfly':
            make_layer = lambda name: self.add_module(name, Butterfly(1024, 1024, bias=True, **kwargs))
        elif method == 'low-rank':
            make_layer = lambda name: self.add_module(name, nn.Sequential(nn.Linear(1024, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], 1024, bias=True)))
        elif method == 'toeplitz':
            make_layer = lambda name: self.add_module(name, sl.ToeplitzLikeC(layer_size=1024, bias=True, **kwargs))
        else:
            assert False, f'method {method} not supported'
        make_layer('fc10')
        make_layer('fc11')
        make_layer('fc12')
        make_layer('fc2')
        make_layer('fc3')
        self.logits = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3, 1024)
        x = self.fc10(x[:, 0, :]) + self.fc11(x[:, 1, :]) + self.fc12(x[:, 2, :])
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.logits(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, dropout=False, method='linear', tied_weight=False, **kwargs):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.dropout = nn.Dropout() if dropout else nn.Identity()
        self.features_size = 256 * 4 * 4
        self.fc1 = nn.Linear(self.features_size, self.features_size)
        if method == 'linear':
            self.fc = nn.Linear(self.features_size, self.features_size, bias=False)
        elif method == 'butterfly':
            self.fc = Butterfly(self.features_size, self.features_size, tied_weight=tied_weight, bias=False, **kwargs)
        elif method == 'low-rank':
            self.fc = nn.Sequential(nn.Linear(self.features_size, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], self.features_size, bias=False))
        else:
            assert False, f'method {method} not supported'
        self.bias = nn.Parameter(torch.zeros(self.features_size))
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Sequential(self.dropout, self.fc2, nn.ReLU(), nn.Linear(self.features_size, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = nn.ReLU(self.fc1(x) + self.bias)
        x = self.classifier(x)
        return x


class LowRankConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, rank=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.rank = rank
        self.G = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1], self.rank, self.in_channels))
        self.H = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1], self.out_channels, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in, fan_out = self.in_channels, self.out_channels
        nn.init.uniform_(self.G, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        nn.init.uniform_(self.H, -1 / math.sqrt(self.rank), 1 / math.sqrt(self.rank))
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        M = torch.bmm(self.H, self.G).permute(1, 2, 0).reshape(self.out_channels, self.in_channels, *self.kernel_size)
        return F.conv2d(x, M, self.bias, self.stride, self.padding, self.dilation)


class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SepConv(nn.Module):
    """Separable Convolution."""

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1 + y2)


class CellB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(CellB, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(2 * out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride == 2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        b1 = F.relu(y1 + y2)
        b2 = F.relu(y3 + y4)
        y = torch.cat([b1, b2], 1)
        return F.relu(self.bn2(self.conv2(y)))


class PNASNet(nn.Module):

    def __init__(self, cell_type, num_cells, num_planes):
        super(PNASNet, self).__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes * 2)
        self.layer3 = self._make_layer(num_planes * 2, num_cells=6)
        self.layer4 = self._downsample(num_planes * 4)
        self.layer5 = self._make_layer(num_planes * 4, num_cells=6)
        self.linear = nn.Linear(num_planes * 4, 10)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out


class PreActBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        out = out * w
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ButterflyPermutation(Permutation):

    def __init__(self, size, sig='BT1', param='ortho2', stochastic=False, temp=1.0, samples=1, sample_method='gumbel', hard=False):
        super().__init__()
        self.size = size
        self.sig = sig
        self.param = param
        self.stochastic = stochastic
        self.temp = temp
        self.samples = samples
        self.sample_method = sample_method
        self.hard = hard
        self.m = int(math.ceil(math.log2(size)))
        assert size == 1 << self.m, 'ButterflyPermutation: Only power of 2 supported.'
        if self.stochastic:
            self.mean_temp = 1.0
            self.sample_temp = temp
            if hard:
                self.generate_fn = self.sample_hard_perm
            else:
                self.generate_fn = self.sample_soft_perm
        else:
            self.mean_temp = temp
            self.generate_fn = self.mean_perm
        self.hard_temp = 0.02
        self.hard_iters = int(1.0 / self.hard_temp)
        if sig[:2] == 'BT' and sig[2:].isdigit():
            depth = int(sig[2:])
            self.twiddle_core_shape = 2 * depth, 1, self.m, self.size // 2
            self.strides = [0, 1] * depth
        elif sig[0] == 'B' and sig[1:].isdigit():
            depth = int(sig[1:])
            self.twiddle_core_shape = depth, 1, self.m, self.size // 2
            self.strides = [1] * depth
        elif sig[0] == 'T' and sig[1:].isdigit():
            depth = int(sig[1:])
            self.twiddle_core_shape = depth, 1, self.m, self.size // 2
            self.strides = [0] * depth
        else:
            assert False, f'ButterflyPermutation: signature {sig} not supported.'
        self.depth = self.twiddle_core_shape[0]
        margin = 0.001
        init = (1 - 2 * margin) * torch.rand(self.twiddle_core_shape) + margin
        if self.param == 'ds':
            self.twiddle = nn.Parameter(init)
        elif self.param == 'logit':
            init = perm.sample_gumbel(self.twiddle_core_shape) - perm.sample_gumbel(self.twiddle_core_shape)
            init_temp = 1.0 / self.depth
            self.twiddle = nn.Parameter(init / init_temp)
        elif param == 'ortho2':
            self.twiddle = nn.Parameter(torch.acos(torch.sqrt(init)))
        else:
            assert False, f'ButterflyPermutation: Parameter type {self.param} not supported.'
        self.twiddle._is_perm_param = True

    def entropy(self, p=None):
        """ TODO: How does this compare to the matrix entropy of the expanded mean matrix? """
        if p == 'logit':
            assert self.param == 'logit'

            def binary_ent(p):
                eps = 1e-10
                return -(p * torch.log2(eps + p) + (1 - p) * torch.log2(1 - p + eps))
            _twiddle = self.map_twiddle(self.twiddle)
            ent1 = torch.sum(binary_ent(_twiddle))
            return ent1
            x = torch.exp(-self.twiddle)
            ent2 = torch.log2(1.0 + x) + self.twiddle * (x / (1.0 + x))
            ent2 = torch.sum(ent2)
            None
            return ent2
        if p is None:
            perms = self.generate_perm()
        elif p == 'mean':
            perms = self.mean_perm()
        elif p == 'mle':
            perms = self.mle_perm()
        elif p == 'sample':
            perms = self.sample_perm()
        else:
            assert False, f'Permutation type {p} not supported.'
        return perm.entropy(perms, reduction='mean')

    def generate_perm(self):
        """ Generate (a batch of) permutations for training """
        return self.generate_fn()

    def map_twiddle(self, twiddle):
        if self.param == 'ds':
            return twiddle
        elif self.param == 'logit':
            return 1.0 / (1.0 + torch.exp(-twiddle))
        elif self.param == 'ortho2':
            return torch.cos(twiddle) ** 2
        else:
            assert False, f'Unreachable'

    def compute_perm(self, twiddle, strides, squeeze=True):
        """
        # twiddle: (depth, 1, log n, n/2)
        twiddle: (depth, samples, log n, n/2)
        strides: (depth,) bool

        Returns: (samples, n, n)
        """
        samples = twiddle.size(1)
        P = torch.eye(self.size, device=twiddle.device).unsqueeze(1).repeat((1, samples, 1))
        for t, stride in zip(twiddle, strides):
            twiddle_factor_mat = torch.stack((torch.stack((t, 1 - t), dim=-1), torch.stack((1 - t, t), dim=-1)), dim=-2)
            P = butterfly_mult_untied(twiddle_factor_mat, P, stride, self.training)
        P = P.transpose(0, 1)
        return P.squeeze() if squeeze else P

    def mean_perm(self):
        _twiddle = self.map_twiddle(self.twiddle)
        p = self.compute_perm(_twiddle, self.strides)
        return p

    def mle_perm(self):
        _twiddle = self.map_twiddle(self.twiddle)
        hard_twiddle = torch.where(_twiddle > 0.5, torch.tensor(1.0, device=_twiddle.device), torch.tensor(0.0, device=_twiddle.device))
        p = self.compute_perm(hard_twiddle, self.strides)
        return p

    def sample_perm(self, sample_shape=()):
        if self.stochastic:
            return self.sample_soft_perm()
        else:
            return self.sample_hard_perm()

    def sample_soft_perm(self, sample_shape=()):
        sample_shape = self.samples,
        if self.param == 'logit':
            logits = torch.stack((self.twiddle, torch.zeros_like(self.twiddle)), dim=-1)
            shape = logits.size()
            noise = perm.sample_gumbel((logits.size(0), self.samples) + logits.size()[2:], device=logits.device)
            logits_noise = logits + noise
            sample_twiddle = torch.softmax(logits_noise / self.sample_temp, dim=-1)[..., 0]
            perms = self.compute_perm(sample_twiddle, self.strides, squeeze=False)
            return perms
        else:
            _twiddle = self.map_twiddle(self.twiddle)
            if self.sample_method == 'gumbel':
                logits = torch.stack((torch.log(_twiddle), torch.log(1.0 - _twiddle)), dim=-1)
                logits_noise = perm.add_gumbel_noise(logits, sample_shape)
                sample_twiddle = torch.softmax(logits_noise / self.sample_temp, dim=-1)[..., 0]
            elif self.sample_method == 'uniform':
                r = torch.rand(_twiddle.size())
                _twiddle = _twiddle - r
                sample_twiddle = 1.0 / (1.0 + torch.exp(-_twiddle / self.sample_temp))
            else:
                assert False, 'sample_method {self.sample_method} not supported'
        perms = torch.stack([self.compute_perm(twiddle, self.strides) for twiddle in sample_twiddle], dim=0)
        return perms

    def sample_hard_perm(self, sample_shape=()):
        sample_shape = self.samples,
        _twiddle = self.map_twiddle(self.twiddle)
        r = torch.rand(_twiddle.size(), device=_twiddle.device)
        _twiddle = _twiddle - r
        sample_twiddle = _twiddle.repeat(*sample_shape, *([1] * _twiddle.dim()))
        hard_twiddle = torch.where(sample_twiddle > 0, torch.ones_like(sample_twiddle), torch.zeros_like(sample_twiddle))
        sample_twiddle.data = hard_twiddle
        if self.training:
            assert sample_twiddle.requires_grad
        perms = torch.stack([self.compute_perm(twiddle, self.strides) for twiddle in sample_twiddle], dim=0)
        return perms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IdentityPermutation(Permutation):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def generate_perm(self):
        return torch.eye(self.size, device=device)

    def mean_perm(self):
        return torch.eye(self.size, device=device)

    def mle_perm(self):
        return torch.eye(self.size, device=device)

    def sample_perm(self):
        return torch.eye(self.size, device=device)


class LinearPermutation(Permutation):

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.W = nn.Parameter(torch.empty(size, size))
        self.W.is_perm_param = True
        nn.init.kaiming_uniform_(self.W)

    def generate_perm(self):
        return self.W

    def mean_perm(self):
        return self.W

    def mle_perm(self):
        return self.W

    def sample_perm(self):
        return self.W


class TensorPermutation(nn.Module):

    def __init__(self, w, h, method='identity', rank=2, train=True, **kwargs):
        super().__init__()
        self.w = w
        self.h = h
        if method == 'linear':
            self.perm_type = LinearPermutation
        elif method == 'butterfly':
            self.perm_type = ButterflyPermutation
        elif method == 'identity':
            self.perm_type = IdentityPermutation
        else:
            assert False, f'Permutation method {method} not supported.'
        self.rank = rank
        if self.rank == 1:
            self.permute = nn.ModuleList([self.perm_type(w * h, **kwargs)])
        elif self.rank == 2:
            self.permute = nn.ModuleList([self.perm_type(w, **kwargs), self.perm_type(h, **kwargs)])
        else:
            assert False, 'prank must be 1 or 2'
        if train == False:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x, perm=None):
        if perm is None:
            perm_fn = self.perm_type.generate_perm
        elif perm == 'mean':
            perm_fn = self.perm_type.mean_perm
        elif perm == 'mle':
            perm_fn = self.perm_type.mle_perm
        elif perm == 'sample':
            perm_fn = self.perm_type.sample_perm
        else:
            assert False, f'Permutation type {perm} not supported.'
        if self.rank == 1:
            perm = perm_fn(self.permute[0])
            x = x.view(-1, self.w * self.h)
            x = x @ perm
            x = x.view(-1, 3, self.w, self.h)
        elif self.rank == 2:
            x = x.transpose(-1, -2)
            perm2 = perm_fn(self.permute[1])
            x = x @ perm2.unsqueeze(-3).unsqueeze(-3)
            x = x.transpose(-1, -2)
            perm1 = perm_fn(self.permute[0])
            x = x @ perm1.unsqueeze(-3).unsqueeze(-3)
            x = x.view(-1, 3, self.w, self.h)
        return x

    def get_permutations(self, perm=None):
        if perm is None:
            perm_fn = self.perm_type.generate_perm
        elif perm == 'mean':
            perm_fn = self.perm_type.mean_perm
        elif perm == 'mle':
            perm_fn = self.perm_type.mle_perm
        elif perm == 'sample':
            perm_fn = self.perm_type.sample_perm
        else:
            assert False, f'Permutation type {perm} not supported.'
        perms = torch.stack([perm_fn(p) for p in self.permute], dim=0)
        return perms

    def entropy(self, p):
        ents = torch.stack([perm.entropy(p) for perm in self.permute], dim=0)
        return torch.mean(ents)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10, zero_init_residual=False, **perm_args):
        super().__init__()
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.zero_init_residual = zero_init_residual
        self.permute = TensorPermutation(32, 32, **perm_args)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = nn.Linear(512 * self.block.expansion, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch = x.size(0)
        x = self.permute(x)
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.logits(x)
        return x


def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class BottleneckFinal(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.bn3(out)
        out = self.relu(out)
        return out


class BottleneckZero(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4, init_zero=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockOriginal(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockOriginal, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNetOriginal paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetOriginal(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetOriginal, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ShuffleBlock(nn.Module):

    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class SplitBlock(nn.Module):

    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(self.bn2(self.conv2(out1)))
        out2 = F.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = F.relu(self.bn5(self.conv5(out2)))
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out


configs = {(0.5): {'out_channels': (48, 96, 192, 1024), 'num_blocks': (3, 7, 3)}, (1): {'out_channels': (116, 232, 464, 1024), 'num_blocks': (3, 7, 3)}, (1.5): {'out_channels': (176, 352, 704, 1024), 'num_blocks': (3, 7, 3)}, (2): {'out_channels': (224, 488, 976, 2048), 'num_blocks': (3, 7, 3)}}


class ShuffleNetV2(nn.Module):

    def __init__(self, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class ToeplitzlikeLinear(nn.Module):

    def __init__(self, in_size, out_size, rank=4, bias=True, corner=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.nstack = int(math.ceil(out_size / self.in_size))
        self.rank = rank
        assert not corner, 'corner not currently supported'
        self.corner = corner
        init_stddev = math.sqrt(1.0 / (rank * in_size))
        self.G = nn.Parameter(torch.randn(self.nstack, rank, in_size) * init_stddev)
        self.H = nn.Parameter(torch.randn(self.nstack, rank, in_size) * init_stddev)
        self.G._is_structured = True
        self.H._is_structured = True
        self.register_buffer('reverse_idx', torch.arange(in_size - 1, -1, -1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        u = input.view(np.prod(input.size()[:-1]), input.size(-1))
        batch = u.shape[0]
        n = self.in_size
        v = self.H
        u_f = torch.rfft(torch.cat((u[:, self.reverse_idx], torch.zeros_like(u)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        uv_f = complex_mul(u_f.unsqueeze(1).unsqueeze(1), v_f)
        transpose_out = torch.irfft(uv_f, 1, signal_sizes=(2 * n,))[..., self.reverse_idx]
        v = self.G
        w = transpose_out
        w_f = torch.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        wv_sum_f = complex_mul(w_f, v_f).sum(dim=2)
        output = torch.irfft(wv_sum_f, 1, signal_sizes=(2 * n,))[..., :n]
        output = output.reshape(batch, self.nstack * self.in_size)[:, :self.out_size]
        if self.bias is not None:
            output = output + self.bias
        return output.view(*input.size()[:-1], self.out_size)

    def extra_repr(self):
        return 'in_size={}, out_size={}, bias={}, rank={}, corner={}'.format(self.in_size, self.out_size, self.bias is not None, self.rank, self.corner)


class Toeplitzlike1x1Conv(ToeplitzlikeLinear):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, nstack * c, h, w)
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1, structure_type=None, **kwargs):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if structure_type == 'B':
            self.conv1 = ButterflyConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, ortho_init=True, **kwargs)
        elif structure_type == 'LR':
            rank = kwargs.get('rank', 1)
            self.conv1 = LowRankConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, rank=rank)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        if structure_type == 'B':
            self.conv2 = ButterflyConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True, ortho_init=True, **kwargs)
        elif structure_type == 'LR':
            rank = kwargs.get('rank', 1)
            self.conv2 = LowRankConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True, rank=rank)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if structure_type == 'B':
                conv = ButterflyConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, ortho_init=True, **kwargs)
            elif structure_type == 'LR':
                rank = kwargs.get('rank', 1)
                conv = LowRankConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, rank=rank)
            else:
                conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            self.shortcut = nn.Sequential(conv)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, structure_type=None, **kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, structure_type=structure_type, **kwargs)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, structure_type=None, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, structure_type, **kwargs))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class HadamardTransformCuda(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    """

    @staticmethod
    def forward(ctx, twiddle, x):
        ctx.save_for_backward(twiddle)
        return butterfly_multiply_untied_forward_fast(twiddle, x, True)

    @staticmethod
    def backward(ctx, grad):
        twiddle, = ctx.saved_tensors
        return None, HadamardTransformCuda.apply(twiddle, grad)


hadamard_transform_cuda = HadamardTransformCuda.apply


def twiddle_normal_to_fast_format(twiddle):
    """Convert twiddle stored in the normal format to the fast format.
    Parameters:
        twiddle: (nstack, log_n, n / 2, 2, 2)
    Returns:
        twiddle_fast: (nstack, log_n, 2, n)
    """
    twiddle = twiddle.clone()
    nstack = twiddle.shape[0]
    n = twiddle.shape[2] * 2
    m = int(math.log2(n))
    twiddle[:, :, :, 1] = twiddle[:, :, :, 1, [1, 0]]
    twiddle_list = []
    for i in range(m):
        stride = 1 << i
        new_twiddle = twiddle[:, i]
        new_twiddle = new_twiddle.reshape(nstack, n // 2 // stride, stride, 2, 2)
        new_twiddle = new_twiddle.permute(0, 1, 3, 2, 4)
        new_twiddle = new_twiddle.reshape(nstack, n, 2).transpose(1, 2)
        twiddle_list.append(new_twiddle)
    result = torch.stack(twiddle_list, dim=1)
    return result


class Hadamard(nn.Module):

    def __init__(self, n):
        super().__init__()
        m = int(math.ceil(math.log2(n)))
        self.n = n
        self.extended_n = 1 << m
        with torch.no_grad():
            twiddle = torch.tensor([[1, 1], [1, -1]], dtype=torch.float) / math.sqrt(2)
            twiddle = twiddle.reshape(1, 1, 1, 2, 2).expand((1, m, self.extended_n // 2, 2, 2))
            twiddle = twiddle_normal_to_fast_format(twiddle)
        self.register_buffer('twiddle', twiddle)

    def forward(self, x):
        if self.n < self.extended_n:
            x = F.pad(x, (0, self.extended_n - self.n))
        output = hadamard_transform_cuda(self.twiddle, x.unsqueeze(1)).squeeze(1)
        if self.n < self.extended_n:
            output = output[:, :self.n]
        return output


class Hadamard1x1Conv(Hadamard):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, c, h, w)
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super(Hadamard1x1Conv, self).forward(input_reshape)
        return output.view(batch, h * w, c).transpose(1, 2).view(batch, c, h, w)


class CNN5(nn.Module):
    name = 'cnn5'

    def __init__(self, num_channels=32, num_classes=10):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels * 4)
        self.fc1 = nn.Linear(4 * 4 * num_channels * 4, num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(num_channels * 4)
        self.fc2 = nn.Linear(num_channels * 4, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bn3(self.conv3(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 4 * 4 * self.num_channels * 4)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class Complex2Real(nn.Module):

    def forward(self, input):
        return input.real


class Real2Complex(nn.Module):

    def forward(self, input):
        return real2complex(input)


class TensorProduct(nn.Module):

    def __init__(self, map1, map2) ->None:
        """Perform map1 on the last dimension of the input and then map2 on the next
        to last dimension.
        """
        super().__init__()
        self.map1 = map1
        self.map2 = map2

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        """
        Parameter:
            input: (*, n2, n1)
        Return:
            output: (*, n2, n1)
        """
        out = self.map1(input)
        return self.map2(out.transpose(-1, -2)).transpose(-1, -2)


def complex_matmul_torch(X, Y):
    return torch.view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag, X.real @ Y.imag + X.imag @ Y.real], dim=-1))


def cp2torch(tensor):
    return torch.view_as_complex(from_dlpack(cp.ascontiguousarray(tensor)[..., None].view(complex_np_dtype_to_real[tensor.dtype]).toDlpack()))


complex_torch_dtype_to_np = {torch.complex64: np.complex64, torch.complex128: np.complex128}


def torch2cp(tensor):
    return cp.fromDlpack(to_dlpack(torch.view_as_real(tensor.contiguous()))).view(complex_torch_dtype_to_np[tensor.dtype]).squeeze(-1)


class ComplexMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        if not X.is_cuda:
            return X @ Y
        else:
            return cp2torch(torch2cp(X) @ torch2cp(Y)) if use_cupy else complex_matmul_torch(X, Y)

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            Y_t = Y.transpose(-1, -2)
            if not Y.is_cuda:
                grad_X = (grad @ Y_t.conj()).sum_to_size(*X.shape)
            else:
                grad_X = (cp2torch(torch2cp(grad) @ torch2cp(Y_t.conj())) if use_cupy else complex_matmul_torch(grad, Y_t.conj())).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            X_t = X.transpose(-1, -2)
            if not X.is_cuda:
                grad_Y = (X_t.conj() @ grad).sum_to_size(*Y.shape)
            else:
                grad_Y = (cp2torch(torch2cp(X_t.conj()) @ torch2cp(grad)) if use_cupy else complex_matmul_torch(X_t.conj(), grad)).sum_to_size(*Y.shape)
        return grad_X, grad_Y


def complex_matmul(X, Y):
    return X @ Y if not X.is_complex() else ComplexMatmul.apply(X, Y)


class KOP2d(nn.Module):

    def __init__(self, in_size, in_ch, out_ch, kernel_size, complex=True, init='ortho', nblocks=1, base=2, zero_pad=True):
        super().__init__()
        self.in_size = in_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.complex = complex
        assert init in ['ortho', 'fft']
        if init == 'fft':
            assert self.complex, 'fft init requires complex=True'
        self.init = init
        self.nblocks = nblocks
        assert base in [2, 4]
        self.base = base
        self.zero_pad = zero_pad
        if isinstance(self.in_size, int):
            self.in_size = self.in_size, self.in_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = self.kernel_size, self.kernel_size
        self.padding = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.weight = nn.Parameter(nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, padding=self.padding, bias=False).weight.flip([-1, -2]))
        increasing_strides = [False, False, True]
        inits = ['ortho'] * 3 if self.init == 'ortho' else ['fft_no_br', 'fft_no_br', 'ifft_no_br']
        self.Kd, self.K1, self.K2 = [TensorProduct(Butterfly(self.in_size[-1], self.in_size[-1], bias=False, complex=complex, increasing_stride=incstride, init=i, nblocks=nblocks), Butterfly(self.in_size[-2], self.in_size[-2], bias=False, complex=complex, increasing_stride=incstride, init=i, nblocks=nblocks)) for incstride, i in zip(increasing_strides, inits)]
        with torch.no_grad():
            self.Kd.map1 *= math.sqrt(self.in_size[-1])
            self.Kd.map2 *= math.sqrt(self.in_size[-2])
        if self.zero_pad and self.complex:
            with torch.no_grad():
                n1, n2 = self.Kd.map1.n, self.Kd.map2.n
                device = self.Kd.map1.twiddle.device
                br1 = bitreversal_permutation(n1, pytorch_format=True)
                br2 = bitreversal_permutation(n2, pytorch_format=True)
                diagonal1 = torch.exp(1.0j * 2 * math.pi / n1 * self.padding[-1] * torch.arange(n1, device=device))[br1]
                diagonal2 = torch.exp(1.0j * 2 * math.pi / n2 * self.padding[-2] * torch.arange(n2, device=device))[br2]
                self.Kd.map1.twiddle[:, 0, -1, :, 0, :] *= diagonal1[::2].unsqueeze(-1)
                self.Kd.map1.twiddle[:, 0, -1, :, 1, :] *= diagonal1[1::2].unsqueeze(-1)
                self.Kd.map2.twiddle[:, 0, -1, :, 0, :] *= diagonal2[::2].unsqueeze(-1)
                self.Kd.map2.twiddle[:, 0, -1, :, 1, :] *= diagonal2[1::2].unsqueeze(-1)
        if base == 4:
            self.Kd.map1, self.Kd.map2 = self.Kd.map1.to_base4(), self.Kd.map2.to_base4()
            self.K1.map1, self.K1.map2 = self.K1.map1.to_base4(), self.K1.map2.to_base4()
            self.K2.map1, self.K2.map2 = self.K2.map1.to_base4(), self.K2.map2.to_base4()
        if complex:
            self.Kd = nn.Sequential(Real2Complex(), self.Kd)
            self.K1 = nn.Sequential(Real2Complex(), self.K1)
            self.K2 = nn.Sequential(self.K2, Complex2Real())

    def forward(self, x):
        x_f = self.K1(x)
        w_f = self.Kd(self.weight)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = self.K2(prod)
        return out


class CNN5Butterfly(CNN5):
    name = 'cnn5butterfly'

    def __init__(self, num_channels=32, num_classes=10, **kwargs):
        nn.Module.__init__(self)
        self.num_channels = num_channels
        in_size = 32
        self.conv1 = KOP2d(in_size, 3, num_channels, 3, **kwargs)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = KOP2d(in_size // 2, num_channels, num_channels * 2, 3, **kwargs)
        self.bn2 = nn.BatchNorm2d(num_channels * 2)
        self.conv3 = KOP2d(in_size // 4, num_channels * 2, num_channels * 4, 3, **kwargs)
        self.bn3 = nn.BatchNorm2d(num_channels * 4)
        self.fc1 = nn.Linear(4 * 4 * num_channels * 4, num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(num_channels * 4)
        self.fc2 = nn.Linear(num_channels * 4, num_classes)


class LeNetPadded(nn.Module):
    name = 'lenetpadded'

    def __init__(self, num_classes=10, padding_mode='circular', pooling_mode='avg'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2, padding_mode=padding_mode)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        assert pooling_mode in ['avg', 'max']
        self.pool2d = F.avg_pool2d if pooling_mode == 'avg' else F.max_pool2d

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class ComplexLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True) ->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            weight /= math.sqrt(2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        output = complex_reshape(input, -1, input.size(-1))
        output = complex_matmul(output, self.weight.t())
        output = output.reshape(*input.shape[:-1], output.shape[-1])
        return output if self.bias is None else output + self.bias

    def extra_repr(self) ->str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class LOP2d(nn.Module):
    """Similar to KOP2d, but we use nn.Linear instead of Butterfly.
    """

    def __init__(self, in_size, in_ch, out_ch, kernel_size, complex=True, init='random'):
        super().__init__()
        self.in_size = in_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.complex = complex
        assert init in ['random', 'fft']
        if init == 'fft':
            assert self.complex, 'fft init requires complex=True'
        self.init = init
        if isinstance(self.in_size, int):
            self.in_size = self.in_size, self.in_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = self.kernel_size, self.kernel_size
        self.padding = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.weight = nn.Parameter(nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, padding=self.padding, bias=False).weight.flip([-1, -2]))
        linear_cls = nn.Linear if not complex else ComplexLinear
        self.Kd, self.K1, self.K2 = [TensorProduct(linear_cls(self.in_size[-1], self.in_size[-1], bias=False), linear_cls(self.in_size[-2], self.in_size[-2], bias=False)) for _ in range(3)]
        if init == 'fft':
            eye1 = torch.eye(self.in_size[-1], dtype=torch.complex64)
            eye2 = torch.eye(self.in_size[-2], dtype=torch.complex64)
            fft_mat1 = torch.fft.fft(eye1, norm='ortho')
            fft_mat2 = torch.fft.fft(eye2, norm='ortho')
            ifft_mat1 = torch.fft.ifft(eye1, norm='ortho')
            ifft_mat2 = torch.fft.ifft(eye2, norm='ortho')
            with torch.no_grad():
                self.Kd.map1.weight.copy_(fft_mat1)
                self.Kd.map2.weight.copy_(fft_mat2)
                self.K1.map1.weight.copy_(fft_mat1)
                self.K1.map2.weight.copy_(fft_mat2)
                self.K2.map1.weight.copy_(ifft_mat1)
                self.K2.map2.weight.copy_(ifft_mat2)
        with torch.no_grad():
            self.Kd.map1.weight *= math.sqrt(self.in_size[-1])
            self.Kd.map2.weight *= math.sqrt(self.in_size[-2])
        self.Kd.map1.weight._is_structured = True
        self.Kd.map2.weight._is_structured = True
        self.K1.map1.weight._is_structured = True
        self.K1.map2.weight._is_structured = True
        self.K2.map1.weight._is_structured = True
        self.K2.map2.weight._is_structured = True
        if complex:
            self.Kd = nn.Sequential(Real2Complex(), self.Kd)
            self.K1 = nn.Sequential(Real2Complex(), self.K1)
            self.K2 = nn.Sequential(self.K2, Complex2Real())

    def forward(self, x):
        w = F.pad(self.weight, (0, self.in_size[-1] - self.kernel_size[-1])).roll(-self.padding[-1], dims=-1)
        w = F.pad(w, (0, 0, 0, self.in_size[-2] - self.kernel_size[-2])).roll(-self.padding[-2], dims=-2)
        x_f = self.K1(x)
        w_f = self.Kd(w)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = self.K2(prod)
        return out


class ResNet18(ResNet):
    name = 'resnet18'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [2, 2, 2, 2], num_classes)


class ResNet34(ResNet):
    name = 'resnet34'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [3, 4, 6, 3], num_classes)


class ResNet50(ResNet):
    name = 'resnet50'

    def __init__(self, num_classes=10):
        return super().__init__(Bottleneck, [3, 4, 6, 3], num_classes)


class ResNet101(ResNet):
    name = 'resnet101'

    def __init__(self, num_classes=10):
        return super().__init__(Bottleneck, [3, 4, 23, 3], num_classes)


class ResNet152(ResNet):
    name = 'resnet101'

    def __init__(self, num_classes=10):
        return super().__init__(Bottleneck, [3, 8, 36, 3], num_classes)


class ResNet8(ResNet):
    name = 'resnet8'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [1, 1, 1], num_classes)


class ResNet14(ResNet):
    name = 'resnet14'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [2, 2, 2], num_classes)


class ResNet20(ResNet):
    name = 'resnet20'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [3, 3, 3], num_classes)


class ResNet32(ResNet):
    name = 'resnet32'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [5, 5, 5], num_classes)


class ResNet44(ResNet):
    name = 'resnet44'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [7, 7, 7], num_classes)


class ResNet56(ResNet):
    name = 'resnet56'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [9, 9, 9], num_classes)


class ResNet110(ResNet):
    name = 'resnet110'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [18, 18, 18], num_classes)


class ResNet1202(ResNet):
    name = 'resnet1202'

    def __init__(self, num_classes=10):
        return super().__init__(BasicBlock, [200, 200, 200], num_classes)


class Features(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        This Feature extractor class takes an input and constructs a feature vector. It can be applied independently to all elements of the input sequence

        in_flattened_vector: input flattened vector
        latent_dim: number of neurons in latent layer
        output_dim: dimension of log alpha square matrix
        """
        super().__init__()
        self.linear1 = nn.Linear(1, latent_dim)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.d2 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.

        x: Tensor of shape (batch_size, 1)
        """
        x = self.d1(self.relu1(self.linear1(x)))
        x = self.d2(self.linear2(x))
        return x


class Sinkhorn_Net(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        super().__init__()
        self.output_dim = output_dim
        self.features = Features(latent_dim, output_dim, dropout_prob)

    def forward(self, x):
        """
        x: Tensor of length (batch, sequence_length)
        Note that output_dim should correspond to the intended sequence length
        """
        x = x.view(-1, 1)
        x = self.features(x)
        x = x.reshape(-1, self.output_dim, self.output_dim)
        return x


def project_simplex(v, z=1.0):
    """Project a vector v onto the simplex.
    That is, return argmin_w ||w - v||^2 where w >= 0 elementwise and sum(w) = z.
    Parameters:
        v: Tensor of shape (batch_size, n)
        z: real number
    Return:
        Projection of v on the simplex, along the last dimension: (batch_size, n)
    """
    v_sorted, _ = v.sort(dim=-1, descending=True)
    range_ = torch.arange(1.0, 1 + v.shape[-1])
    cumsum_divided = (v_sorted.cumsum(dim=-1) - z) / range_
    cond = (v_sorted - cumsum_divided > 0).type(v.dtype)
    rho = (cond * range_).argmax(dim=-1)
    tau = cumsum_divided[range(v.shape[0]), rho]
    return torch.clamp(v - tau.unsqueeze(-1), min=0)


def sparsemax_grad(output, grad):
    support = output > 0
    support_f = support.type(grad.dtype)
    s = (grad * support_f).sum(dim=-1) / support_f.sum(dim=-1)
    return support_f * (grad - s.unsqueeze(-1))


class Sparsemax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v):
        output = project_simplex(v)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        output, = ctx.saved_tensors
        return sparsemax_grad(output, grad)


sparsemax = Sparsemax.apply


class MatrixProduct(nn.Module):
    """Product of matrices. The order are chosen by softmaxes, which are learnable.
    Each factor matrix must implement .matrix() function.
    """

    def __init__(self, factors, n_terms=None, complex=False, fixed_order=False, softmax_fn='softmax'):
        super().__init__()
        self.factors = nn.ModuleList(factors)
        if n_terms is None:
            n_terms = len(factors)
        self.n_terms = n_terms
        self.complex = complex
        self.matmul_op = complex_matmul if complex else operator.matmul
        self.fixed_order = fixed_order
        if not self.fixed_order:
            assert softmax_fn in ['softmax', 'sparsemax']
            self.logit = nn.Parameter(torch.randn((self.n_terms, len(factors))))
            if softmax_fn == 'softmax':
                self.softmax_fn = lambda logit: nn.functional.softmax(logit, dim=-1)
            else:
                self.softmax_fn = sparsemax

    def matrix(self, temperature=1.0):
        if self.fixed_order:
            matrices = [factor.matrix() for factor in self.factors]
            return functools.reduce(self.matmul_op, matrices)
        else:
            prob = self.softmax_fn(self.logit / temperature)
            stack = torch.stack([factor.matrix() for factor in self.factors])
            matrices = (prob @ stack.reshape(stack.shape[0], -1)).reshape((-1,) + stack.shape[1:])
            return functools.reduce(self.matmul_op, matrices)

    def forward(self, input, temperature=1.0):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if self.fixed_order:
            output = input
            for factor in self.factors[::-1]:
                output = factor(output)
            return output
        else:
            prob = self.softmax_fn(self.logit / temperature)
            output = input
            for i in range(self.n_terms)[::-1]:
                stack = torch.stack([factor(output) for factor in self.factors])
                output = (prob[i:i + 1] @ stack.reshape(stack.shape[0], -1)).reshape(stack.shape[1:])
            return output


def sinkhorn(logit, n_iters=5):
    """Sinkhorn iterations.
    Parameters:
        logit: (..., n, n)
        n_iters: integer
    Return:
        (..., n, n) matrix that's close to a doubly stochastic matrix.
    """
    assert logit.dim() >= 2, 'logit must be at least a 2D tensor'
    assert logit.shape[-2] == logit.shape[-1], 'logit must be a square matrix'
    for _ in range(n_iters):
        logit = logit - torch.logsumexp(logit, dim=-1, keepdim=True)
        logit = logit - torch.logsumexp(logit, dim=-2, keepdim=True)
    return torch.exp(logit)


class ButterflyProduct(MatrixProduct):
    """Product of butterfly matrices. The order are chosen by softmaxes, which
    are learnable.
    """

    def __init__(self, size, n_terms=None, complex=False, fixed_order=False, softmax_fn='softmax', learn_perm=False):
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        self.size = size
        factors = [Butterfly(size, diagonal=1 << i, complex=complex) for i in range(m)[::-1]]
        super().__init__(factors, n_terms, complex, fixed_order, softmax_fn)
        self.learn_perm = learn_perm
        if learn_perm:
            self.perm_logit = nn.Parameter(torch.randn((size, size)))

    def matrix(self, temperature=1.0):
        matrix = super().matrix(temperature)
        if self.learn_perm:
            perm = sinkhorn(self.perm_logit / temperature)
            if not self.complex:
                matrix = matrix @ perm
            else:
                matrix = (matrix.transpose(-1, -2) @ perm).transpose(-1, -2)
        return matrix

    def forward(self, input, temperature=1.0):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if self.learn_perm:
            perm = sinkhorn(self.perm_logit / temperature)
            if not self.complex:
                input = input @ perm.t()
            else:
                input = (input.transpose(-1, -2) @ perm.t()).transpose(-1, -2)
        return super().forward(input, temperature)


class ButterflyFactorMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, input):
        ctx.save_for_backward(coefficients, input)
        return butterfly_factor_multiply(coefficients, input)

    @staticmethod
    def backward(ctx, grad):
        coefficients, input = ctx.saved_tensors
        d_coefficients, d_input = butterfly_factor_multiply_backward(grad, coefficients, input)
        return d_coefficients, d_input


butterfly_factor_mult = ButterflyFactorMult.apply


class Block2x2Diag(nn.Module):
    """Block matrix of size n x n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, complex=False, ABCD=None, ortho_init=False):
        """
        Parameters:
            size: size of butterfly matrix
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (2, 2, size//2) if real or (2, 2, size//2, 2) if complex
            ortho_init: whether the twiddle factors are initialized to be orthogonal (real) or unitary (complex)
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        ABCD_shape = (2, 2, size // 2) if not complex else (2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            if not ortho_init:
                self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
            elif not complex:
                theta = torch.rand(size // 2) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, (size // 2,), dtype=c.dtype) * 2 - 1
                self.ABCD = nn.Parameter(torch.stack((torch.stack((det * c, -det * s)), torch.stack((s, c)))))
            else:
                phi = torch.asin(torch.sqrt(torch.rand(size // 2)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.randn(3, size // 2) * math.pi * 2
                A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
                B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
                C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
                D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
                self.ABCD = nn.Parameter(torch.stack((torch.stack((A, B)), torch.stack((C, D)))))
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if not self.complex:
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2)).view(input.shape)
        else:
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2, 2)).view(input.shape)


class Block2x2DiagProduct(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, complex=False, decreasing_size=True, ortho_init=False):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        self.size = size
        self.complex = complex
        sizes = [(size >> i) for i in range(m)] if decreasing_size else [(size >> i) for i in range(m)[::-1]]
        self.factors = nn.ModuleList([Block2x2Diag(size_, complex=complex, ortho_init=ortho_init) for size_ in sizes])

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.contiguous()
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output


class ButterflyFactorMultIntermediate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        output = butterfly_multiply_intermediate(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output[-1]

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_multiply_intermediate_backward(grad, twiddle, output)
        return d_coefficients, d_input


butterfly_factor_mult_intermediate = ButterflyFactorMultIntermediate.apply


class Block2x2DiagProductAllinOne(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, rank=1, complex=False, twiddle=None, ortho_init=False):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        self.size = size
        self.rank = rank
        self.complex = complex
        twiddle_shape = (rank, size - 1, 2, 2) if not complex else (rank, size - 1, 2, 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if twiddle is None:
            if not ortho_init:
                self.twiddle = nn.Parameter(torch.randn(twiddle_shape) * scaling)
            elif not complex:
                theta = torch.rand(rank, size - 1) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, (rank, size - 1), dtype=c.dtype) * 2 - 1
                self.twiddle = nn.Parameter(torch.stack((torch.stack((det * c, -det * s), dim=-1), torch.stack((s, c), dim=-1)), dim=-1))
            else:
                phi = torch.asin(torch.sqrt(torch.rand(rank, size - 1)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.randn(3, rank, size - 1) * math.pi * 2
                A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
                B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
                C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
                D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
                self.twiddle = nn.Parameter(torch.stack((torch.stack((A, B), dim=-1), torch.stack((C, D), dim=-1)), dim=-1))
        else:
            assert twiddle.shape == twiddle_shape, f'twiddle must have shape {twiddle_shape}'
            self.twiddle = twiddle

    def forward(self, input):
        """
        Parameters:
            input: (batch, size) if real or (batch, size, 2) if complex
        Return:
            output: (batch, rank * size) if real or (batch, rank * size, 2) if complex
        """
        output_shape = (input.shape[0], self.rank * input.shape[1]) if self.real else (input.shape[0], self.rank * input.shape[1], 2)
        return butterfly_factor_mult_intermediate(self.twiddle, input).view(output_shape)


class Block2x2DiagRectangular(nn.Module):
    """Block matrix of size k n x k n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, stack=1, complex=False, ABCD=None, n_blocks=1, tied_weight=True):
        """
        Parameters:
            size: input has shape (stack, ..., size)
            stack: number of stacked components, output has shape (stack, ..., size)
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (stack, 2, 2, size//2) if real or (stack, 2, 2, size//2, 2) if complex
            n_blocks: number of such blocks of ABCD
            tied_weight: whether the weights ABCD at different blocks are tied to be the same.
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.stack = stack
        self.complex = complex
        self.n_blocks = n_blocks
        self.tied_weight = tied_weight
        if tied_weight:
            ABCD_shape = (stack, 2, 2, size // 2) if not complex else (stack, 2, 2, size // 2, 2)
        else:
            ABCD_shape = (stack, n_blocks, 2, 2, size // 2) if not complex else (stack, n_blocks, 2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (stack, ..., size) if real or (stack, ..., size, 2) if complex
            if not tied_weight: (stack, n_blocks, ..., size) if real or (stack, n_blocks, ..., size, 2) if complex
        Return:
            output: (stack, ..., size) if real or (stack, ..., size, 2) if complex
            if not tied_weight: (stack, n_blocks, ..., size) if real or (stack, n_blocks, ..., size, 2) if complex
        """
        if self.tied_weight:
            if not self.complex:
                return (self.ABCD.unsqueeze(1) * input.view(self.stack, -1, 1, 2, self.size // 2)).sum(dim=-2).view(input.shape)
            else:
                return complex_mul(self.ABCD.unsqueeze(1), input.view(self.stack, -1, 1, 2, self.size // 2, 2)).sum(dim=-3).view(input.shape)
        elif not self.complex:
            return (self.ABCD.unsqueeze(2) * input.view(self.stack, self.n_blocks, -1, 1, 2, self.size // 2)).sum(dim=-2).view(input.shape)
        else:
            return complex_mul(self.ABCD.unsqueeze(2), input.view(self.stack, self.n_blocks, -1, 1, 2, self.size // 2, 2)).sum(dim=-3).view(input.shape)


class Block2x2DiagProductRectangular(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, in_size, out_size, complex=False, decreasing_size=True, tied_weight=True, bias=True):
        super().__init__()
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        self.in_size_extended = 1 << m
        self.out_size = out_size
        self.stack = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.tied_weight = tied_weight
        in_sizes = [(self.in_size_extended >> i) for i in range(m)] if decreasing_size else [(self.in_size_extended >> i) for i in range(m)[::-1]]
        if tied_weight:
            self.factors = nn.ModuleList([Block2x2DiagRectangular(in_size_, stack=self.stack, complex=complex) for in_size_ in in_sizes])
        else:
            self.factors = nn.ModuleList([Block2x2DiagRectangular(in_size_, stack=self.stack, complex=complex, n_blocks=self.in_size_extended // in_size_, tied_weight=tied_weight) for in_size_ in in_sizes])
        if bias:
            if not self.complex:
                self.bias = nn.Parameter(torch.Tensor(out_size))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_size, 2))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if hasattr(self, 'bias'):
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (..., in_size) if real or (..., in_size, 2) if complex
        Return:
            output: (..., out_size) if real or (..., out_size, 2) if complex
        """
        output = input.contiguous()
        if self.in_size != self.in_size_extended:
            if not self.complex:
                output = torch.cat((output, torch.zeros(output.shape[:-1] + (self.in_size_extended - self.in_size,), dtype=output.dtype, device=output.device)), dim=-1)
            else:
                output = torch.cat((output, torch.zeros(output.shape[:-2] + (self.in_size_extended - self.in_size, 2), dtype=output.dtype, device=output.device)), dim=-2)
        output = output.unsqueeze(0).expand((self.stack,) + output.shape)
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        if not self.complex:
            output = output.permute(tuple(range(1, output.dim() - 1)) + (0, -1)).reshape(input.shape[:-1] + (self.stack * self.in_size_extended,))[..., :self.out_size]
        else:
            output = output.permute(tuple(range(1, output.dim() - 2)) + (0, -2, -1)).reshape(input.shape[:-2] + (self.stack * self.in_size_extended, 2))[..., :self.out_size, :]
        if hasattr(self, 'bias'):
            output += self.bias
        return output


class Block2x2DiagBmm(nn.Module):
    """Block matrix of size n x n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, complex=False, ABCD=None):
        """
        Parameters:
            size: size of butterfly matrix
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (2, 2, size//2) if real or (2, 2, size//2, 2) if complex
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        ABCD_shape = (size // 2, 2, 2) if not complex else (2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (size, batch_size) if real or (size, batch_size, 2) if complex
        Return:
            output: (size, batch_size) if real or (size, batch_size, 2) if complex
        """
        if not self.complex:
            return (self.ABCD @ input.view(self.size // 2, 2, -1)).view(input.shape)
        else:
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2, 2)).view(input.shape)


class Block2x2DiagProductBmm(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, complex=False, decreasing_size=True):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        self.size = size
        self.complex = complex
        sizes = [(size >> i) for i in range(m)] if decreasing_size else [(size >> i) for i in range(m)[::-1]]
        self.factors = nn.ModuleList([Block2x2DiagBmm(size_, complex=complex) for size_ in sizes])
        self.br_perm = bitreversal_permutation(size)

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.t()[self.br_perm]
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view((factor.size, -1))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output[self.br_perm].t()


class BlockPerm(nn.Module):
    """Block permutation matrix of size n x n.
    """

    def __init__(self, size, logit=None, complex=False):
        """
        Parameters:
            size: size of permutation matrix
            complex: real of complex input
            logit: (3, ) nn.Parameter, containing logits for probability of
                   separating even and odd (logit[0]), probability of reversing
                   the first half (logit[1]), and probability of reversing the
                   second half (logit[2]).
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        if logit is None:
            self.logit = nn.Parameter(torch.randn(3))
        else:
            self.logit = logit
        self.reverse_perm = nn.Parameter(torch.arange(self.size // 2 - 1, -1, -1), requires_grad=False)

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        prob = torch.sigmoid(self.logit)
        output = input
        if not self.complex:
            output = permutation_factor_even_odd_mult(prob[:1], output.view(-1, self.size))
            output = permutation_factor_reverse_mult(prob[1:], output)
        else:
            output = permutation_factor_even_odd_mult(prob[:1], output.view(-1, self.size))
            output = permutation_factor_reverse_mult(prob[1:], output)
        return output.view(input.shape)

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        logit = nn.Parameter(torch.where(self.logit >= 0, torch.tensor(float('inf'), device=self.logit.device), torch.tensor(float('-inf'), device=self.logit.device)))
        argmax_instance = self.__class__(self.size, logit, complex=False)
        p = argmax_instance.forward(torch.arange(self.size, dtype=torch.float, device=self.logit.device)).round().long()
        return p


class BlockPermProduct(nn.Module):
    """Product of block permutation matrices.
    """

    def __init__(self, size, complex=False, share_logit=False, increasing_size=True):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        self.size = size
        self.complex = complex
        self.share_logit = share_logit
        sizes = [(size >> i) for i in range(m - 1)[::-1]] if increasing_size else [(size >> i) for i in range(m - 1)]
        if share_logit:
            self.logit = nn.Parameter(torch.randn(3))
            self.factors = nn.ModuleList([BlockPerm(size_, self.logit, complex=complex) for size_ in sizes])
        else:
            self.factors = nn.ModuleList([BlockPerm(size_, complex=complex) for size_ in sizes])

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.contiguous()
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        p = torch.arange(self.size, device=self.factors[0].logit.device)
        for factor in self.factors[::-1]:
            p = p.reshape(-1, factor.size)[:, factor.argmax()].reshape(self.size)
        return p


def polymatmul(A, B):
    """Batch-multiply two matrices of polynomials
    Parameters:
        A: (N, batch_size, n, m, d1)
        B: (batch_size, m, p, d2)
    Returns:
        AB: (N, batch_size, n, p, d1 + d2 - 1)
    """
    unsqueezed = False
    if A.dim() == 4:
        unsqueezed = True
        A = A.unsqueeze(0)
    N, batch_size, n, m, d1 = A.shape
    batch_size_, m_, p, d2 = B.shape
    assert batch_size == batch_size_
    assert m == m_
    result = F.conv1d(A.transpose(1, 2).reshape(N * n, batch_size * m, d1), B.transpose(1, 2).reshape(batch_size * p, m, d2).flip(-1), padding=d2 - 1, groups=batch_size).reshape(N, n, batch_size, p, d1 + d2 - 1).transpose(1, 2)
    return result.squeeze(0) if unsqueezed else result


class HstackDiag(nn.Module):
    """Horizontally stacked diagonal matrices of size n x 2n. Each entry in a 2x2
    matrix of polynomials.
    """

    def __init__(self, size, deg=0, diag1=None, diag2=None):
        """
        Parameters:
            size: size of diagonal matrix
            deg: degree of the polynomials
            diag1: initialization for the diagonal, should be n x 2 x 2 x (d + 1), where d is the degree of the polynomials
            diag2: initialization for the diagonal, should be n x 2 x 2 x (d + 1), where d is the degree of the polynomials
        """
        super().__init__()
        self.size = size
        self.diag1 = diag1 or nn.Parameter(torch.randn(size, 2, 2, deg + 1))
        self.diag2 = diag2 or nn.Parameter(torch.randn(size, 2, 2, deg + 1))
        assert self.diag1.shape == self.diag2.shape, 'The two diagonals must have the same shape'
        self.deg = self.diag1.shape[-1] - 1

    def forward(self, input_):
        """
        Parameters:
            input_: (b, 2 * size, 2, 2, d1)
        Return:
            output: (b, size, 2, 2, d1 + self.deg - 1)
        """
        output = polymatmul(input_[:, :self.size], self.diag1) + polymatmul(input_[:, self.size:], self.diag2)
        return output


class HstackDiagProduct(nn.Module):
    """Product of HstackDiag matrices.
    """

    def __init__(self, size):
        m = int(math.log2(size))
        assert size == 1 << m, 'size must be a power of 2'
        super().__init__()
        self.size = size
        self.factors = nn.ModuleList([HstackDiag(size >> i + 1, deg=1 << i) for i in range(m)[::-1]])
        self.P_init = nn.Parameter(torch.randn(1, 2, 1, 2))

    def forward(self, input_):
        """
        Parameters:
            input_: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input_
        for factor in self.factors[::-1]:
            output = factor(output)
        result = polymatmul(output[:, :, [1], :, :-1], self.P_init).squeeze(1).squeeze(1).squeeze(1)
        return result


class ButterflyUnitary(Butterfly):
    """Same as Butterfly, but constrained to be unitary
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, increasing_stride=True, nblocks=1):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = True
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        complex_dtype = real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = self.nstacks, nblocks, log_n, n // 2, 4
        self.init = 'ortho'
        self.twiddle = nn.Parameter(torch.empty(twiddle_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=complex_dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        twiddle_core_shape = self.twiddle.shape[:-1]
        phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
        alpha, psi, chi = torch.rand((3,) + twiddle_core_shape) * math.pi * 2
        with torch.no_grad():
            self.twiddle.copy_(torch.stack([phi, alpha, psi, chi], dim=-1))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        phi, alpha, psi, chi = torch.unbind(self.twiddle, -1)
        c, s = torch.cos(phi), torch.sin(phi)
        A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
        B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
        C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
        D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
        twiddle = torch.stack([torch.stack([A, B], dim=-2), torch.stack([C, D], dim=-2)], dim=-3)
        twiddle = torch.view_as_complex(twiddle)
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = twiddle[:, :, :log_n, :n // 2] if self.increasing_stride else twiddle[:, :, -log_n:, :n // 2]
            output_size = None
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))
    __imul__ = None
    to_base4 = None

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, increasing_stride={}, nblocks={}'.format(self.in_size, self.out_size, self.bias is not None, self.increasing_stride, self.nblocks)
        return s


def butterfly_multiply_base4_torch(twiddle4, twiddle2, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle4.shape[1]
    log_n = twiddle4.shape[2] * 2 + twiddle2.shape[2]
    n = 1 << log_n
    if log_n // 2 > 0:
        assert twiddle4.shape == (nstacks, nblocks, log_n // 2, n // 4, 4, 4)
    if log_n % 2 == 1:
        assert twiddle2.shape == (nstacks, nblocks, 1, n // 2, 2, 2)
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    output_size = n if output_size is None else output_size
    assert output_size <= n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n // 2):
            log2_stride = 2 * idx if cur_increasing_stride else log_n - 2 - 2 * idx
            stride = 1 << log2_stride
            t = twiddle4[:, block, idx].view(nstacks, n // (4 * stride), stride, 4, 4).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstacks, n // (4 * stride), 1, 4, stride)
            output = (t * output_reshape).sum(dim=4)
        if log_n % 2 == 1:
            log2_stride = log_n - 1 if cur_increasing_stride else 0
            stride = 1 << log2_stride
            t = twiddle2[:, block, 0].view(nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)[:, :, :output_size]


class ButterflyBase4(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, *args, **kwargs):
        init = kwargs.get('init', None)
        if isinstance(init, tuple) and len(init) == 2 and isinstance(init[0], torch.Tensor) and isinstance(init[1], torch.Tensor):
            twiddle4, twiddle2 = init[0].clone(), init[1].clone()
            kwargs['init'] = 'empty'
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        del self.twiddle
        self.twiddle4 = nn.Parameter(twiddle4)
        self.twiddle2 = nn.Parameter(twiddle2)
        self.twiddle4._is_structured = True
        self.twiddle2._is_structured = True

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        output = butterfly_multiply_base4_torch(self.twiddle4, self.twiddle2, output, self.increasing_stride, output_size)
        return self.post_process(input, output)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        scale_per_entry = scale ** (1.0 / self.nblocks / self.log_n)
        self.twiddle4 *= scale_per_entry ** 2
        self.twiddle2 *= scale_per_entry
        return self


class Diagonal(nn.Module):

    def __init__(self, size=None, complex=False, diagonal_init=None):
        """Multiply by diagonal matrix
        Parameter:
            size: int
            diagonal_init: (n, )
        """
        super().__init__()
        if diagonal_init is not None:
            self.size = diagonal_init.shape
            self.diagonal = nn.Parameter(diagonal_init.detach().clone())
            self.complex = self.diagonal.is_complex()
        else:
            assert size is not None
            self.size = size
            dtype = torch.get_default_dtype() if not complex else real_dtype_to_complex[torch.get_default_dtype()]
            self.diagonal = nn.Parameter(torch.randn(size, dtype=dtype))
            self.complex = complex

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        return input * self.diagonal


class DiagonalMultiplySum(nn.Module):

    def __init__(self, diagonal_init):
        """
        Parameters:
            diagonal_init: (out_channels, in_channels, size)
        """
        super().__init__()
        self.diagonal = nn.Parameter(diagonal_init.detach().clone())
        self.complex = self.diagonal.is_complex()

    def forward(self, input):
        """
        Parameters:
            input: (batch, in_channels, size)
        Return:
            output: (batch, out_channels, size)
        """
        return (input.unsqueeze(1) * self.diagonal).sum(dim=2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveConcatPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockOriginal,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block2x2DiagProductRectangular,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Block2x2DiagRectangular,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ButterflyBase4,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CNN5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CellA,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellB,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Complex2Real,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1': 4, 'n3x3red': 4, 'n3x3': 4, 'n5x5red': 4, 'n5x5': 4, 'pool_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KnowledgeDistillationLoss,
     lambda: ([], {'original_loss': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LowRankConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 1024])], {}),
     True),
    (NLLMultiLabelSmooth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Real2Complex,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet110,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet1202,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet14,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet18,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet20,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet32,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet44,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet56,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SepConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleBlock,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sinkhorn_Net,
     lambda: ([], {'latent_dim': 4, 'output_dim': 4, 'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SplitBlock,
     lambda: ([], {'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TensorProduct,
     lambda: ([], {'map1': _mock_layer(), 'map2': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (wide_basic,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_HazyResearch_butterfly(_paritybench_base):
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

