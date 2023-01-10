import sys
_module = sys.modules[__name__]
del sys
conf = _module
falkon = _module
benchmarks = _module
common = _module
benchmark_utils = _module
create_weather_dataset = _module
datasets = _module
error_metrics = _module
summary = _module
benchmark_runner = _module
models = _module
gpflow_model = _module
gpytorch_sgpr = _module
gpytorch_variational_models = _module
lauum_timings = _module
mmv_timings = _module
potrf_timings = _module
time_improvements = _module
center_selection = _module
gsc_losses = _module
hopt = _module
benchmarking = _module
benchmark_cli = _module
runner_gd = _module
runner_gridsearch = _module
objectives = _module
exact_objectives = _module
compreg = _module
gcv = _module
holdout = _module
loocv = _module
new_compreg = _module
sgpr = _module
utils = _module
objectives = _module
stoch_objectives = _module
stoch_new_compreg = _module
utils = _module
transforms = _module
optimization = _module
gd_train = _module
grid_search = _module
models = _module
reporting = _module
utils = _module
kernels = _module
diff_kernel = _module
distance_kernel = _module
dot_prod_kernel = _module
keops_helpers = _module
kernel = _module
la_helpers = _module
cpu_trsm = _module
cuda_trsm = _module
wrapper = _module
mkl_bindings = _module
mkl_bind = _module
mmv_ops = _module
batch_mmv = _module
fmm = _module
fmmv = _module
fmmv_incore = _module
keops = _module
utils = _module
falkon = _module
incore_falkon = _module
logistic_falkon = _module
model_utils = _module
ooc_ops = _module
ooc_lauum = _module
ooc_potrf = _module
ooc_utils = _module
parallel_lauum = _module
optim = _module
conjgrad = _module
options = _module
preconditioner = _module
flk_preconditioner = _module
logistic_preconditioner = _module
pc_utils = _module
sparse = _module
sparse_ops = _module
sparse_tensor = _module
tests = _module
conftest = _module
gen_random = _module
helpers = _module
naive_kernels = _module
test_batch_mmv = _module
test_chol_prec = _module
test_conjgrad = _module
test_cyblas = _module
test_device_copy = _module
test_dim_selectors = _module
test_falkon = _module
test_gsc_losses = _module
test_hopt = _module
test_kernels = _module
test_kernels_sparse = _module
test_logistic_falkon = _module
test_mkl = _module
test_nysel = _module
test_ooc_lauum = _module
test_ooc_potrf = _module
test_sparse = _module
test_stress_multi_core = _module
test_trsm_wrapper = _module
test_util = _module
device_copy = _module
devices = _module
fake_queue = _module
helpers = _module
stream_utils = _module
switches = _module
tensor_helpers = _module
threading = _module
tictoc = _module
uci_datasets_download = _module
setup = _module

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


from enum import Enum


from abc import abstractmethod


from abc import ABC


from typing import Union


from typing import Tuple


import numpy as np


import scipy.io as scio


import scipy.sparse


from scipy.sparse import load_npz


from sklearn.datasets import load_svmlight_file


from torch.utils.tensorboard import SummaryWriter


import functools


import time


from typing import Optional


from typing import List


from typing import Dict


from typing import Any


from scipy.linalg.lapack import slauum


from scipy.linalg.lapack import dlauum


from scipy.linalg.lapack import spotrf


from scipy.linalg.lapack import dpotrf


import warnings


import math


from functools import partial


import pandas as pd


import abc


from torch.distributions.transforms import identity_transform


from typing import Sequence


import torch.distributions.constraints as constraints


import torch.nn.functional as F


from functools import reduce


from typing import Iterator


from copy import deepcopy


from torch import nn


from collections.abc import Callable


from numpy.ctypeslib import as_array


import torch.cuda as tcd


import torch.cuda.comm


from typing import Callable


from sklearn import base


import scipy.linalg.blas as sclb


from scipy.linalg import lapack as scll


from scipy.spatial.distance import cdist


import scipy


import random


from sklearn import datasets


import scipy.linalg.lapack as scll


import torch.cuda


from typing import Type


import torch.multiprocessing


from typing import Generator


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


class PositiveTransform(torch.distributions.transforms.Transform):
    _cache_size = 0
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self, lower_bound=0.0):
        super().__init__()
        self.lower_bound = lower_bound

    def __eq__(self, other):
        if not isinstance(other, PositiveTransform):
            return False
        return other.lower_bound == self.lower_bound

    def _call(self, x):
        y = F.softplus(x)
        y = y + self.lower_bound
        return y

    def _inverse(self, y):
        x = y - self.lower_bound
        threshold = torch.log(torch.tensor(torch.finfo(y.dtype).eps, dtype=y.dtype)) + torch.tensor(2.0, dtype=y.dtype)
        is_too_small = x < torch.exp(threshold)
        is_too_large = x > -threshold
        too_small_val = torch.log(x)
        too_large_val = x
        x = torch.where(is_too_small | is_too_large, torch.tensor(1.0, dtype=y.dtype, device=y.device), x)
        x = x + torch.log(-torch.expm1(-x))
        return torch.where(is_too_small, too_small_val, torch.where(is_too_large, too_large_val, x))


EPS = 5e-05


class Optimizer(object):
    """Base class for optimizers. This is an empty shell at the moment.
    """

    def __init__(self):
        pass


class StopOptimizationException(Exception):

    def __init__(self, message):
        super().__init__()
        self.message = message


class TicToc:
    __t_start = {}

    def __init__(self, title='', debug=True):
        self.title = title
        self.should_print = debug

    def tic(self, _print=False):
        mp_name = self.mp_name
        times = TicToc.__t_start.setdefault(mp_name, [])
        if _print and self.should_print:
            indent_level = len(times)
            indent_str = self._get_indent_str(indent_level)
            None
        times.append(time.time())

    def toc(self):
        mp_name = self.mp_name
        times = TicToc.__t_start[mp_name]
        t_elapsed = time.time() - times.pop()
        indent_level = len(times)
        indent_str = self._get_indent_str(indent_level)
        if self.should_print:
            None

    def toc_val(self):
        mp_name = self.mp_name
        times = TicToc.__t_start.setdefault(mp_name, [])
        return time.time() - times.pop()

    @property
    def mp_name(self):
        return '%s.%s' % (mpr.current_process().name, thr.current_thread().name)

    @staticmethod
    def _get_indent_str(level):
        return '--' * level

    def __enter__(self):
        self.tic(_print=True)

    def __exit__(self, type, value, traceback):
        self.toc()


def _ccontig_strides(sizes) ->Tuple[int, ...]:
    if len(sizes) == 0:
        return tuple()
    return tuple(np.cumprod(sizes[1:][::-1])[::-1].tolist() + [1])


def _new_strided_tensor(size: Tuple[int], stride: Tuple[int], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool) ->torch.Tensor:
    if not torch.cuda.is_available():
        pin_memory = False
    elif isinstance(device, torch.device):
        pin_memory &= device.type == 'cpu'
    else:
        pin_memory &= device.lower() == 'cpu'
    return torch.empty_strided(size=size, stride=stride, dtype=dtype, device=device, requires_grad=False, pin_memory=pin_memory)


def create_C(size: Tuple[int, ...], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool=False) ->torch.Tensor:
    """Allocates an empty, row-contiguous 1 or 2-dimensional tensor

    Parameters
    -----------
    size : tuple of integers
        Must be a tuple of length 1 or 2 indicating the shape of the
        created tensor.
    dtype : torch.dtype
        The type of the new tensor.
    device : str or torch.device
        The device on which the tensor should be allocated (e.g. 'cpu', 'cuda:0')
    pin_memory : bool
        Whether a CPU tensor should be allocated in pinned memory or
        not. If allocating a GPU tensor this flag has no effect.

    Returns
    --------
    t : torch.Tensor
        The allocated tensor
    """
    strides = _ccontig_strides(size)
    return _new_strided_tensor(size, strides, dtype, device, pin_memory)


def _fcontig_strides(sizes) ->Tuple[int, ...]:
    if len(sizes) == 0:
        return tuple()
    return tuple([1] + np.cumprod(sizes)[:-1].tolist())


def create_fortran(size: Tuple[int, ...], dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool=False) ->torch.Tensor:
    """Allocates an empty, column-contiguous 1 or 2-dimensional tensor

    Parameters
    -----------
    size : tuple of integers
        Must be a tuple of length 1 or 2 indicating the shape of the
        created tensor.
    dtype : torch.dtype
        The type of the new tensor.
    device : str or torch.device
        The device on which the tensor should be allocated (e.g. 'cpu', 'cuda:0')
    pin_memory : bool
        Whether a CPU tensor should be allocated in pinned memory or
        not. If allocating a GPU tensor this flag has no effect.

    Returns
    --------
    t : torch.Tensor
        The allocated tensor
    """
    strides = _fcontig_strides(size)
    return _new_strided_tensor(size, strides, dtype, device, pin_memory)


def is_contig(tensor: torch.Tensor) ->bool:
    stride = tensor.stride()
    for s in stride:
        if s == 1:
            return True
    return False


def is_f_contig(tensor: torch.Tensor, strict: bool=False) ->bool:
    """Check if a pytorch Tensor is column-contiguous (Fortran order)

    Column-contiguity means that the stride of the first dimension (of
    a 2D tensor) must be equal to 1.
    In case of 1D tensors we just check contiguity

    Parameters
    -----------
    tensor : torch.Tensor
        1 or 2-dimensional tensor whose stride should be checked.
    strict : bool
        For 1D arrays there is no difference for row and column contiguity.
        2D arrays where one of the dimensions is of size 1 can be either
        treated like 1D arrays (`strict=False`) or like 2D arrays
        (`strict=True`).

    Returns
    --------
    fortran : bool
        Whether the input tensor is column-contiguous
    """
    strides = tensor.stride()
    sizes = tensor.shape
    if len(sizes) == 0:
        return True
    if len(sizes) == 1:
        return strides[0] == 1
    if sizes[-2] == 1:
        if strict:
            return strides[-2] == 1
        return strides[-1] == 1 or strides[-2] == 1
    if sizes[-1] == 1:
        if strict:
            return strides[-2] == 1 and strides[-1] >= sizes[-2]
        return strides[-2] == 1
    if strides[-2] != 1 or strides[-1] < strides[-2]:
        return False
    return True


def create_same_stride(size: Tuple[int, ...], other: torch.Tensor, dtype: torch.dtype, device: Union[str, torch.device], pin_memory: bool=False) ->torch.Tensor:
    if is_f_contig(other, strict=True):
        return create_fortran(size=size, dtype=dtype, device=device, pin_memory=pin_memory)
    elif is_contig(other):
        return create_C(size=size, dtype=dtype, device=device, pin_memory=pin_memory)
    else:
        raise ValueError('Desired stride is not contiguous, cannot create.')


def copy_same_stride(tensor: torch.Tensor, pin_memory: bool=False) ->torch.Tensor:
    new = create_same_stride(tensor.shape, tensor, tensor.dtype, tensor.device, pin_memory)
    new.copy_(tensor)
    return new


class SparseType(Enum):
    """Whether a `SparseTensor` is in CSC or CSR format.
    """
    CSR = 'csr'
    CSC = 'csc'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


class SparseTensor:
    """Wrapper class to represent sparse 2D matrices in CSR or CSC format.

    The wrapper holds three 1D torch tensors which give the sparse representation
    (an index pointer, an index and the non-zero values of the matrix).
    It supports some of the common torch tensor management functions (e.g. `pin_memory`, `device`,
    `size`) and conversion to and from the corresponding scipy sparse matrix representation.
    It does **not** define any mathematical function on sparse matrices, which are
    instead defined separately (see :func:`~falkon.sparse.sparse_matmul` for example).

    Parameters
    ----------
    indexptr : torch.Tensor
        Array of row (or column for CSC data) pointers into the
        `index` and `data` arrays. Should be either of type long or int.
    index : torch.Tensor
        Array of column (or row for CSC data) indices for non-zero elements.
        Should be either of type long or int.
    data : torch.Tensor
        Array of the non-zero elements for the sparse matrix.
    size : Tuple[int, int]
        Shape of the 2D tensor (rows, columns).
    sparse_type: str or falkon.sparse.sparse_tensor.SparseType
        Whether the matrix should be interpreted as CSR or CSC format.
    """

    def __init__(self, indexptr: torch.Tensor, index: torch.Tensor, data: torch.Tensor, size: Tuple[int, int], sparse_type: Union[str, SparseType]=SparseType.CSR):
        if isinstance(sparse_type, str):
            sparse_type = SparseType(sparse_type)
        if sparse_type == SparseType.CSR:
            if indexptr.shape[0] - 1 != size[0]:
                raise ValueError('Data is not in correct csr format. Incorrect indexptr size.')
        elif sparse_type == SparseType.CSC:
            if indexptr.shape[0] - 1 != size[1]:
                raise ValueError('Data is not in correct csc format. Incorrect indexptr size.')
        else:
            raise ValueError('Sparse type %s not valid.' % sparse_type)
        if index.shape[0] != data.shape[0]:
            raise ValueError('Data is not in correct format. Different sizes for index and values.')
        dev = data.device
        if index.device != dev or indexptr.device != dev:
            raise ValueError('Cannot create SparseTensor with components on different devices.')
        self.indexptr = indexptr
        self.index = index
        self.data = data
        self.sparse_type = sparse_type
        self._size = size

    @property
    def shape(self):
        return self._size

    def size(self, dim: Optional[int]=None):
        if dim is None:
            return self._size
        return self._size[dim]

    @property
    def dtype(self) ->torch.dtype:
        return self.data.dtype

    @property
    def is_csc(self):
        return self.sparse_type == SparseType.CSC

    @property
    def is_csr(self):
        return self.sparse_type == SparseType.CSR

    @property
    def device(self) ->torch.device:
        return self.data.device

    @property
    def is_cuda(self) ->bool:
        return self.data.is_cuda

    def nnz(self):
        return self.data.numel()

    @property
    def density(self):
        return self.nnz() / (self._size[0] * self._size[1])

    def dim(self):
        return len(self._size)

    def narrow_rows(self, start: Optional[int], length: Optional[int]) ->'SparseTensor':
        """Select a subset of contiguous rows from the sparse matrix.
        If this is a CSC sparse matrix, instead of taking contiguous rows we take contiguous
        columns.

        Parameters
        ----------
        start: int or None
            The index of the first row to select. If None will be assumed to be 0.
        length: int or None
            The number of rows to select. If None will be assumed to be all rows after `start`.

        Returns
        --------
        SparseTensor
            A new :class:`~falkon.sparse.sparse_tensor.SparseTensor` object with `length` rows.

        Notes
        ------
        The output matrix will share storage with the original matrix whenever possible.
        """
        if start is None:
            start = 0
        elif start > self.shape[0]:
            raise IndexError('Start is greater than the length of the array')
        if length is None:
            length = self.shape[0] - start
        elif length + start > self.shape[0]:
            raise IndexError('End larger than array')
        end = start + length
        startptr = self.indexptr[start]
        endptr = self.indexptr[end]
        new_indexptr = self.indexptr[start:end + 1]
        new_index = self.index[startptr:endptr]
        new_data = self.data[startptr:endptr]
        if start > 0:
            new_indexptr = new_indexptr.clone().detach()
            new_indexptr.sub_(startptr)
        return SparseTensor(indexptr=new_indexptr, index=new_index, data=new_data, size=(length, self.size(1)))

    def to(self, dtype=None, device=None, non_blocking=False) ->'SparseTensor':
        new_data = self.data
        new_indexptr = self.indexptr
        new_index = self.index
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        change_dtype = dtype != self.dtype
        change_device = device != self.device
        if change_dtype or change_device:
            new_data = self.data
        if change_device:
            new_indexptr = self.indexptr
            new_index = self.index
        return SparseTensor(indexptr=new_indexptr, index=new_index, data=new_data, size=self.shape, sparse_type=self.sparse_type)

    def cuda(self) ->'SparseTensor':
        return SparseTensor(indexptr=self.indexptr, index=self.index, data=self.data, size=self.shape, sparse_type=self.sparse_type)

    def index_to_int_(self):
        self.indexptr = self.indexptr
        self.index = self.index

    def index_to_int(self):
        new_index = self.index
        new_indexptr = self.indexptr
        return SparseTensor(indexptr=new_indexptr, index=new_index, data=self.data, size=self.shape, sparse_type=self.sparse_type)

    def index_to_long_(self):
        self.indexptr = self.indexptr
        self.index = self.index

    def index_to(self, dtype: torch.dtype):
        new_index = self.index
        new_indexptr = self.indexptr
        return SparseTensor(indexptr=new_indexptr, index=new_index, data=self.data, size=self.shape, sparse_type=self.sparse_type)

    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.indexptr = self.indexptr.pin_memory()
        self.index = self.index.pin_memory()
        return self

    def transpose_csc(self):
        if self.is_csc:
            raise RuntimeError('Cannot transpose_csc since data is already in csc format')
        new_size = self.shape[1], self.shape[0]
        return SparseTensor(indexptr=self.indexptr, index=self.index, data=self.data, size=new_size, sparse_type=SparseType.CSC)

    def transpose_csr(self):
        if self.is_csr:
            raise RuntimeError('Cannot transpose_csr since data is already in csr format')
        new_size = self.shape[1], self.shape[0]
        return SparseTensor(indexptr=self.indexptr, index=self.index, data=self.data, size=new_size, sparse_type=SparseType.CSR)

    @staticmethod
    def from_scipy(mat: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]) ->'SparseTensor':
        if isinstance(mat, scipy.sparse.csr_matrix):
            return SparseTensor(indexptr=torch.from_numpy(mat.indptr), index=torch.from_numpy(mat.indices), data=torch.from_numpy(mat.data), size=mat.shape[:2], sparse_type=SparseType.CSR)
        elif isinstance(mat, scipy.sparse.csc_matrix):
            return SparseTensor(indexptr=torch.from_numpy(mat.indptr), index=torch.from_numpy(mat.indices), data=torch.from_numpy(mat.data), size=mat.shape[:2], sparse_type=SparseType.CSC)
        else:
            raise NotImplementedError('Cannot convert type %s to SparseTensor. Please use the CSR or CSC formats' % type(mat))

    def to_scipy(self, copy: bool=False) ->Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]:
        if self.is_cuda:
            return self.to_scipy(copy=copy)
        if self.is_csr:
            return scipy.sparse.csr_matrix((self.data, self.index, self.indexptr), shape=self.shape, copy=copy)
        elif self.is_csc:
            return scipy.sparse.csc_matrix((self.data, self.index, self.indexptr), shape=self.shape, copy=copy)
        else:
            raise NotImplementedError('Cannot convert %s matrix to scipy' % self.sparse_type)


def check_same_device(*args: Union[None, torch.Tensor, SparseTensor]) ->bool:
    dev = None
    for t in args:
        if t is None:
            continue
        t_dev = t.device
        if dev is None:
            dev = t_dev
        elif t_dev != dev:
            return False
    return True


def check_same_dtype(*args: Optional[Union[torch.Tensor, SparseTensor]]) ->bool:
    dt = None
    all_equal = True
    for a in args:
        if a is None:
            continue
        if dt is None:
            dt = a.dtype
        else:
            all_equal &= a.dtype == dt
    return all_equal


class Preconditioner(ABC):
    """Generic preconditioner class, used to accelerate solutions to linear systems.

    Given a system of equations :math:`H\\beta = Y`, where :math:`H` typically contains in some
    form our data matrix `X` and `Y` contains the targets. We can use matrix :math:`B` to
    create an equivalent linear system which will have lower condition number:

    .. math::

        BB^\\top H \\beta = Y

    where :math:`BB^\\top \\approx H^{-1}` in order to make the preconditioner effective, but not
    too expensive to compute. Then, in order to use the preconditioner in an algorithm based
    on matrix-vector products (such as conjugate gradient descent), we must be able to "apply" the
    matrix :math:`B` and its transpose :math:`B^	op` to any vector.

    For this reason, this class exposes abstract methods `apply` and `apply_t` which should
    be overridden in concrete preconditioner implementations

    See Also
    --------
    :class:`falkon.preconditioner.FalkonPreconditioner` :
        for an actual preconditioner implementation
    """

    def __init__(self):
        pass

    @abstractmethod
    def apply(self, v):
        pass

    @abstractmethod
    def apply_t(self, v):
        pass


def check_init(*none_check):

    def _checker(fun):

        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            is_init = True
            for el in none_check:
                if getattr(self, el, None) is None:
                    is_init = False
                    break
            if not is_init:
                raise RuntimeError('FALKON preconditioner is not initialized. Please run `init` before any other method on the preconditioner.')
            return fun(self, *args, **kwargs)
        return wrapper
    return _checker


arr_type = Union[torch.Tensor, np.ndarray]


def copy_triang(mat: arr_type, upper: bool) ->arr_type:
    """Copy one triangle of `mat` to the other, making it symmetric.

    The input is a square matrix: CUDA and CPU tensors as well as numpy arrays are supported.
    This operation runs in-place.

    Parameters
    ----------
    mat
        The input square tensor, or numpy array. This can also be a CUDA tensor.
    upper
        If `upper=True` the upper triangle will be copied into the lower triangle of `mat`,
        otherwise the lower triangle of `mat` will be copied into its upper triangle.

    Returns
    -------
    mat
        The same tensor, or numpy array as was passed as a parameter, with the desired
        operation performed on it. The output matrix will be symmetric.
    """
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            return copy_triang(mat, upper=upper)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_copy_triang(mat, upper=upper)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def inplace_add_diag_th(A: torch.Tensor, k: float) ->torch.Tensor:
    A.diagonal().add_(k)
    return A


def inplace_set_diag_th(A: torch.Tensor, k: torch.Tensor) ->torch.Tensor:
    A.diagonal().copy_(k)
    return A


def choose_fn(dtype, f64_fn, f32_fn, fn_name):
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            return f64_fn
        if dtype == torch.float32:
            return f32_fn
    if dtype == np.float64:
        return f64_fn
    if dtype == np.float32:
        return f32_fn
    raise TypeError('No %s function exists for data type %s.' % (fn_name, dtype))


def mul_triang(mat: arr_type, upper: bool, preserve_diag: bool, multiplier: float) ->arr_type:
    """Multiply a triangular matrix by a scalar.

    The input is a square matrix, and parameters determine what exactly is the triangular
    part which will be multiplied. CUDA and CPU tensors as well as numpy arrays
    are supported.
    This operation runs in-place.

    Parameters
    ----------
    mat
        The input square tensor, or numpy array. This can also be a CUDA tensor.
    upper
        Whether to consider the upper, or the lower triangular part of `mat`.
    preserve_diag
        Whether the diagonal of `mat` will be multiplied. If `preserve_diag=True`, then the
        diagonal will not be multiplied.
    multiplier
        The scalar by which the triangular input matrix will be multiplied.

    Returns
    -------
    mat
        The same tensor, or numpy array as was passed as a parameter, with the desired
        operation performed on it.
    """
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            return mul_triang(mat, upper=upper, preserve_diag=preserve_diag, multiplier=multiplier)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_mul_triang(mat, upper=upper, preserve_diag=int(preserve_diag), multiplier=multiplier)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def potrf(mat: arr_type, upper: bool, clean: bool, overwrite: bool, cuda: bool) ->arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda or cuda:
            raise NotImplementedError("'potrf' is only implemented for CPU tensors. See the ooc_ops module for CUDA implementations.")
        else:
            out_torch_convert = True
            mat = mat.numpy()
    out = c_potrf(mat, upper=upper, clean=clean, overwrite=overwrite)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


def cpu_trsm(A: np.ndarray, v: np.ndarray, alpha: float, lower: int, transpose: int) ->np.ndarray:
    trsm_fn = choose_fn(A.dtype, sclb.dtrsm, sclb.strsm, 'TRSM')
    vF = np.copy(v, order='F')
    trsm_fn(alpha, A, vF, side=0, lower=lower, trans_a=transpose, overwrite_b=1)
    if not v.flags.f_contiguous:
        vF = np.copy(vF, order='C')
    return vF


def trsm(v: arr_type, A: arr_type, alpha: float, lower: int=0, transpose: int=0) ->arr_type:
    out_torch_convert = False
    if isinstance(A, torch.Tensor):
        if isinstance(v, torch.Tensor):
            if not check_same_device(A, v):
                raise ValueError('A and v must be on the same device.')
            if A.is_cuda and v.is_cuda:
                return cuda_trsm(A, v, alpha, bool(lower), bool(transpose))
            else:
                out_torch_convert = True
                A = A.numpy()
                v = v.numpy()
        elif A.is_cuda:
            raise ValueError('A and v must be on the same device.')
        else:
            out_torch_convert = True
            A = A.numpy()
    vout = cpu_trsm(A, v, alpha, lower, transpose)
    if out_torch_convert:
        return torch.from_numpy(vout)
    return vout


def vec_mul_triang(mat: arr_type, multipliers: arr_type, upper: bool, side: int) ->arr_type:
    out_torch_convert = False
    if isinstance(mat, torch.Tensor):
        if mat.is_cuda:
            multipliers = multipliers.reshape(-1)
            return vec_mul_triang(mat, multipliers, upper, side)
        else:
            out_torch_convert = True
            mat = mat.numpy()
    if isinstance(multipliers, torch.Tensor):
        multipliers = multipliers.numpy().reshape(-1)
    out = c_vec_mul_triang(mat, multiplier=multipliers, upper=upper, side=side)
    if out_torch_convert:
        return torch.from_numpy(out)
    return out


class Timer:

    def __init__(self, time_list: List[float]):
        self.times = time_list
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.start_time)


def calc_deff_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term):
    """Nystrom effective dimension backward"""
    out_deff_bwd = 2 * zy_knm_solve_zy[:t].mean() - zy_solve_knm_knm_solve_zy[:t].mean()
    if include_kmm_term:
        out_deff_bwd -= pen_n * zy_solve_kmm_solve_zy[:t].mean()
    return out_deff_bwd


def calc_dfit_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term):
    """Nystrom regularized data-fit backward"""
    dfit_bwd = -(2 * zy_knm_solve_zy[t:].sum() - zy_solve_knm_knm_solve_zy[t:].sum())
    if include_kmm_term:
        dfit_bwd += pen_n * zy_solve_kmm_solve_zy[t:].sum()
    return dfit_bwd


def calc_grads_tensors(inputs: Sequence[torch.Tensor], inputs_need_grad: Sequence[bool], num_nondiff_inputs: int, output: torch.Tensor, retain_graph: bool, allow_unused: bool) ->Tuple[Optional[torch.Tensor], ...]:
    """

    Parameters
    ----------
    inputs
        Sequence of tensors with respect to which the gradient needs computing
    inputs_need_grad
        Sequence of booleans, stating whether the inputs need the gradient computation.
        This sequence corresponds to ctx.needs_input_grad hence it includes all inputs
        to some nn.Function, not just the differentiable inputs (which are passed in the `inputs`
        parameter).
        Hence `len(inputs_need_grad) != len(inputs)`. To make the code work, the inputs to the
        nn.Function we are dealing with must be organized such that the non-differentiable inputs
        come before the potentially differentiable inputs!
    num_nondiff_inputs: int
        The number of non-differentiable inputs to the nn.Function.
    output
        output of the differentiated function
    retain_graph
        See corresponding option in `torch.autograd.grad`
    allow_unused
        See corresponding option in `torch.autograd.grad`

    Returns
    -------
    The gradients of `output` with respect to the sequence of inputs. If an input does not require
    gradient, the corresponding gradient in the result will be set to `None`.
    """
    assert len(inputs) <= len(inputs_need_grad)
    saved_idx = 0
    needs_grad = []
    for i, i_grad in enumerate(inputs_need_grad):
        if i_grad:
            needs_grad.append(inputs[saved_idx])
        if i >= num_nondiff_inputs:
            saved_idx += 1
    grads = torch.autograd.grad(output, needs_grad, retain_graph=retain_graph, allow_unused=allow_unused)
    grads_idx = 0
    results = []
    for i, i_grad in enumerate(inputs_need_grad):
        if i_grad:
            results.append(grads[grads_idx])
            grads_idx += 1
        else:
            results.append(None)
    return tuple(results)


def calc_trace_bwd(k_mn: Optional[torch.Tensor], k_mn_zy: Optional[torch.Tensor], solve2: torch.Tensor, kmm: torch.Tensor, X: Optional[torch.Tensor], t: Optional[int], trace_type: str):
    """Nystrom kernel trace backward pass"""
    if trace_type == 'ste':
        assert k_mn_zy is not None and t is not None, 'Incorrect arguments to trace_bwd'
        return -(2 * k_mn_zy[:, :t].mul(solve2).sum(0).mean() - (solve2 * (kmm @ solve2)).sum(0).mean())
    elif trace_type == 'direct':
        assert k_mn is not None, 'Incorrect arguments to trace_bwd'
        return -(2 * k_mn.mul(solve2).sum() - (solve2 * (kmm @ solve2)).sum())
    elif trace_type == 'fast':
        assert k_mn_zy is not None and t is not None and X is not None, 'Incorrect arguments to trace_bwd'
        k_subs = k_mn_zy
        norm = X.shape[0] / t
        return -norm * (2 * k_subs.mul(solve2).sum() - (solve2 * (kmm @ solve2)).sum())


def calc_trace_fwd(init_val: torch.Tensor, k_mn: Optional[torch.Tensor], k_mn_zy: Optional[torch.Tensor], kmm_chol: torch.Tensor, X: Optional[torch.Tensor], t: Optional[int], trace_type: str):
    """ Nystrom kernel trace forward """
    if trace_type == 'ste':
        assert k_mn_zy is not None and t is not None, 'Incorrect arguments to trace_fwd'
        solve1 = torch.triangular_solve(k_mn_zy[:, :t], kmm_chol, upper=False, transpose=False).solution
        solve2 = torch.triangular_solve(solve1, kmm_chol, upper=False, transpose=True).solution.contiguous()
        init_val -= solve1.square_().sum(0).mean()
    elif trace_type == 'direct':
        assert k_mn is not None, 'Incorrect arguments to trace_fwd'
        solve1 = trsm(k_mn, kmm_chol, 1.0, lower=True, transpose=False)
        solve2 = trsm(solve1, kmm_chol, 1.0, lower=True, transpose=True)
        init_val -= solve1.square_().sum()
    elif trace_type == 'fast':
        assert k_mn_zy is not None and t is not None, 'Incorrect arguments to trace_fwd'
        k_subs = k_mn_zy
        assert k_subs.shape == (kmm_chol.shape[0], t), 'Shape incorrect'
        solve1 = torch.triangular_solve(k_subs, kmm_chol, upper=False, transpose=False).solution
        solve2 = torch.triangular_solve(solve1, kmm_chol, upper=False, transpose=True).solution.contiguous()
        norm = X.shape[0] / t
        init_val -= solve1.square_().sum() * norm
    else:
        raise ValueError('Trace-type %s unknown' % trace_type)
    return init_val, solve2


def cholesky(M, upper=False, check_errors=True):
    if upper:
        U, info = torch.linalg.cholesky_ex(M.transpose(-2, -1).conj())
        if check_errors:
            if info > 0:
                raise RuntimeError('Cholesky failed on row %d' % info)
        return U.transpose(-2, -1).conj()
    else:
        L, info = torch.linalg.cholesky_ex(M, check_errors=False)
        if check_errors:
            if info > 0:
                raise RuntimeError('Cholesky failed on row %d' % info)
        return L


def init_random_vecs(n, t, dtype, device, gaussian_random: bool):
    if gaussian_random:
        Z = torch.randn(n, t, dtype=dtype, device=device)
    else:
        Z = torch.empty(n, t, dtype=dtype, device=device).bernoulli_().mul_(2).sub_(1)
    return Z


def sizeof_dtype(dtype: Union[torch.dtype, np.dtype, Type]) ->int:
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            return 8
        if dtype == torch.float32:
            return 4
    if dtype == np.float64:
        return 8
    if dtype == np.float32:
        return 4
    raise TypeError('Dtype %s not valid' % dtype)


def get_scalar(t: Union[torch.Tensor, float]) ->float:
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            return deepcopy(t.detach().cpu().item())
        return deepcopy(torch.flatten(t)[0].detach().cpu().item())
    return t


def stochastic_nystrom_compreg(kernel, penalty, centers, X, Y, num_estimators, deterministic, solve_options, solve_maxiter, gaussian_random, warm_start=True, trace_type='ste'):
    return NystromCompRegFn.apply(kernel, deterministic, solve_options, solve_maxiter, gaussian_random, warm_start, trace_type, num_estimators, X, Y, penalty, centers, *kernel.diff_params.values())


def check_sparse(*args: Union[torch.Tensor, SparseTensor]) ->List[bool]:
    out = []
    for t in args:
        out.append(isinstance(t, SparseTensor))
    return out


class FakeQueue:

    def __init__(self):
        self.lst = []

    def get(self):
        return self.lst.pop(0)

    def put(self, obj):
        self.lst.append(obj)

    def __len__(self):
        return len(self.lst)


def _call_direct(target, arg):
    args_queue = FakeQueue()
    args_queue.put(arg[0])
    new_args_tuple = -1, args_queue, arg[1]
    return target(*new_args_tuple)


def _check_contiguity(*args: Tuple[Optional[torch.Tensor], str]) ->None:
    for tensor, name in args:
        if tensor is not None and not is_contig(tensor):
            raise ValueError(f"Tensor '{name}' must be memory contiguous")


def _start_wait_processes(target, args) ->List[Any]:
    processes, outputs = [], []
    for i, a in enumerate(args):
        args_queue = FakeQueue()
        args_queue.put(a[0])
        new_args_tuple = i, args_queue, a[1]
        process = PropagatingThread(target=target, name=f'GPU-{a[1]}', args=new_args_tuple)
        processes.append(process)
    for p in processes:
        p.start()
    for p in processes:
        outputs.append(p.join())
    return outputs


def calc_gpu_block_sizes(device_info, tot_size):
    gpu_speed = np.array([g.speed for g in device_info])
    speed_frac = np.array(gpu_speed) / np.sum(gpu_speed)
    block_sizes = np.cumsum(np.concatenate(([0], speed_frac))) * tot_size
    block_sizes[0] = 0
    block_sizes[-1] = tot_size
    return np.floor(block_sizes).astype(np.int64).tolist()


def create_output_mat(out: Optional[torch.Tensor], data_devs: Sequence[torch.device], is_sparse: bool, shape: Tuple[int, int], dtype: torch.dtype, comp_dev_type: str, other_mat: torch.Tensor, output_stride: Optional[str]=None) ->torch.Tensor:
    if out is not None:
        return out
    out_dev = torch.device('cpu')
    for ddev in data_devs:
        if ddev.type == 'cuda':
            out_dev = ddev
            break
    if is_sparse:
        output_stride = 'F'
    if output_stride is None:
        out = create_same_stride(shape, other_mat, dtype, device=out_dev, pin_memory=out_dev.type != 'cuda' and comp_dev_type == 'cuda')
    elif output_stride == 'F':
        out = create_fortran(shape, dtype, device=out_dev, pin_memory=out_dev.type != 'cuda' and comp_dev_type == 'cuda')
    else:
        out = create_C(shape, dtype, device=out_dev, pin_memory=out_dev.type != 'cuda' and comp_dev_type == 'cuda')
    return out


def select_dim_over_n(max_n, m, d, coef_nm, coef_nd, coef_md, coef_n, coef_m, coef_d, rest, max_mem):
    """
    n * (m * coef_nm + d * coef_nd + coef_n) + rest <= max_mem
    """
    n_coef = m * coef_nm + d * coef_nd + coef_n
    rest_mem = rest + coef_md * m * d + coef_m * m + coef_d * d
    v_n = (max_mem - rest_mem) / n_coef
    out_n = int(min(v_n, max_n))
    if out_n <= 0:
        raise MemoryError('Available memory %.2fMB is not enough.' % (max_mem / 2 ** 20))
    return out_n


def _dense_dmmv_blk_sizes(n, d, m, t, avail_mem: float, extra_mem: dict, m1_ic: bool, m2_ic: bool, v_ic: bool, out_ic: bool) ->Tuple[int, int]:
    coef_nd, coef_md, coef_mt = 0, 0, 0
    coef_nm = 1
    coef_nt = 1
    if not m1_ic:
        coef_nd += 1
    if not m2_ic:
        coef_md += 1
    if not v_ic:
        coef_mt += 1
    if not out_ic:
        coef_mt += 1
    blk_n = select_dim_over_n(max_n=n, m=m, d=d, max_mem=avail_mem, coef_nm=coef_nm + extra_mem.get('nm', 0), coef_nd=coef_nd + extra_mem.get('nd', 0), coef_md=coef_md + extra_mem.get('md', 0), coef_n=coef_nt * t + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0), coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0), coef_d=extra_mem.get('d', 0), rest=0)
    mem_needed = blk_n * m
    mem_needed += blk_n * t * coef_nt
    mem_needed += blk_n * d * coef_nd
    mem_needed += m * d * coef_md
    mem_needed += m * t * coef_mt
    return blk_n, mem_needed


def _dev_from_id(device_id: int) ->torch.device:
    if device_id < 0:
        return torch.device('cpu')
    return torch.device('cuda:%d' % device_id)


def _is_incore(computation_device: torch.device, data_device: torch.device) ->bool:
    return computation_device.type == data_device.type


def _sparse_dmmv_blk_sizes(n, d, m, t, avail_mem, extra_mem: dict, incore: bool, dev_out_exists: bool, m1_density: float, m2_density: float):
    coef_nm = 3
    coef_nd, coef_md, coef_nt, coef_mt = 0, 0, 0, 0
    coef_nt += 1
    if not incore:
        coef_nd += 2 * m1_density
        coef_md += 2 * m2_density
        coef_mt += 1
        if not dev_out_exists:
            coef_mt += 1
    blk_n = select_dim_over_n(max_n=n, m=m, d=d, max_mem=avail_mem, coef_nm=coef_nm + extra_mem.get('nm', 0), coef_nd=coef_nd + extra_mem.get('nd', 0), coef_md=coef_md + extra_mem.get('md', 0), coef_n=coef_nt * t + 2 + extra_mem.get('n', 0) + t * extra_mem.get('nt', 0), coef_m=coef_mt * t + extra_mem.get('m', 0) + t * extra_mem.get('mt', 0), coef_d=1 + extra_mem.get('d', 0), rest=0)
    mem_needed = blk_n * m
    mem_needed += blk_n * t
    if not incore:
        mem_needed += m * t
        if not dev_out_exists:
            mem_needed += m * t
    return blk_n, mem_needed


def extract_C(from_tns: torch.Tensor, size: Tuple[int, ...], offset: int) ->torch.Tensor:
    strides = _ccontig_strides(size)
    return from_tns.as_strided(size=size, stride=strides, storage_offset=int(offset))


def extract_fortran(from_tns: torch.Tensor, size: Tuple[int, ...], offset: int) ->torch.Tensor:
    strides = _fcontig_strides(size)
    return from_tns.as_strided(size=size, stride=strides, storage_offset=int(offset))


def extract_same_stride(from_tns: torch.Tensor, size: Tuple[int, ...], other: torch.Tensor, offset: int=0) ->torch.Tensor:
    if is_f_contig(other, strict=True):
        return extract_fortran(from_tns, size, offset)
    elif is_contig(other):
        return extract_C(from_tns, size, offset)
    else:
        raise ValueError('Desired stride is not contiguous, cannot extract.')


def _extract_flat(flat_tn, size, other, offset):
    struct_tn = extract_same_stride(flat_tn, size=size, other=other, offset=offset)
    offset += np.prod(struct_tn.shape)
    return struct_tn, offset


def dmmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: torch.Tensor, w: Optional[torch.Tensor], out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, mem_needed: int, dev: torch.device, tid: int):
    m1_ic, m2_ic, v_ic, out_ic = _is_incore(dev, m1.device), _is_incore(dev, m2.device), _is_incore(dev, v.device), _is_incore(dev, out.device)
    N, D = m1.shape
    M, T = v.shape
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, M), other=out, offset=flat_offset)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
    if m1_ic:
        dev_m1 = None
    else:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
    if m2_ic:
        dev_m2 = m2
    else:
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(M, D), other=m2, offset=flat_offset)
    if v_ic:
        dev_v = v
    else:
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    if out_ic:
        dev_out = out
    else:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)
    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)
        dev_out.fill_(0.0)
        if not m2_ic:
            copy(m2, dev_m2, non_blocking=True)
        if not v_ic:
            with ExitStack() as stack2:
                if s2 is not None:
                    stack2.enter_context(tcd.stream(s2))
                copy(v, dev_v, non_blocking=True)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if m1_ic:
                c_dev_m1 = m1[i:i + leni, :]
            else:
                c_dev_m1 = copy(m1[i:i + leni, :], dev_m1[:leni, :], non_blocking=True)
            if w is not None:
                with ExitStack() as stack2:
                    if s2 is not None:
                        stack2.enter_context(tcd.stream(s2))
                    c_dev_w = copy(w[i:i + leni, :], dev_w[:leni, :], non_blocking=True)
            else:
                c_dev_w = dev_w[:leni, :].fill_(0.0)
            c_dev_ker = dev_ker[:leni, :].fill_(0.0)
            c_dev_ker = kernel.compute(c_dev_m1, dev_m2, c_dev_ker, diag=False)
            if s2 is not None:
                s2.synchronize()
            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
            if s1 is not None:
                s1.synchronize()
        if not out_ic:
            copy(dev_out, out, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


def sparse_dmmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor, w: Optional[torch.Tensor], out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, mem_needed: int, dev: torch.device, tid: int):
    incore = _is_incore(dev, m1.device)
    dev_out_exists = out.device == dev
    N, D = m1.shape
    M, T = v.shape
    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, M), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_w, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=v if w is None else w, offset=flat_offset)
    dev_out, dev_v, dev_m2 = out, v, m2
    if not incore:
        if not dev_out_exists:
            dev_out, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(M, T), other=v, offset=flat_offset)
    with ExitStack() as stack, torch.inference_mode():
        s1 = None
        if dev.type == 'cuda':
            s1 = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(s1))
        dev_out.fill_(0.0)
        if not incore:
            copy(v, dev_v, non_blocking=True)
            dev_m2 = SparseTensor.from_scipy(m2.transpose_csc().to_scipy().tocsr(copy=False)).index_to_int()
        else:
            dev_m2 = m2.transpose_csc()
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            c_m1 = m1.narrow_rows(i, leni)
            if incore:
                c_dev_m1 = c_m1
            else:
                c_dev_m1 = c_m1.index_to_int()
            if w is None:
                c_dev_w = dev_w[:leni, :].fill_(0.0)
            else:
                c_dev_w = copy(w[i:i + leni, :], dev_w[:leni, :], non_blocking=True)
            c_dev_ker = ker_gpu[:leni].fill_(0.0)
            c_dev_ker = kernel.compute_sparse(c_dev_m1, dev_m2, c_dev_ker, diag=False, X1_csr=c_m1, X2_csr=m2)
            c_dev_w.addmm_(c_dev_ker, dev_v)
            dev_out.addmm_(c_dev_ker.T, c_dev_w)
        if not incore and not dev_out_exists:
            copy(dev_out, out, non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


def dmmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, w, out = a.X1, a.X2, a.v, a.w, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    assert not a.differentiable, 'D-MMV not implemented for differentiable outputs'
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    dev_out_exists = out.device == dev
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape
    if is_sparse:
        blk_n, mem_needed = _sparse_dmmv_blk_sizes(n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore, dev_out_exists=dev_out_exists, m1_density=X1.density, m2_density=X2.density)
        sparse_dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev, tid=proc_idx)
    else:
        m1_ic, m2_ic, v_ic, out_ic = _is_incore(dev, X1.device), _is_incore(dev, X2.device), _is_incore(dev, v.device), _is_incore(dev, out.device)
        blk_n, mem_needed = _dense_dmmv_blk_sizes(n=n, d=d, m=m, t=t, avail_mem=avail_mem, extra_mem=extra_mem, m1_ic=m1_ic, m2_ic=m2_ic, v_ic=v_ic, out_ic=out_ic)
        dmmv_run_thread(X1, X2, v, w, out, kernel, blk_n, mem_needed, dev, tid=proc_idx)


def mm_diff_run_thread(m1: torch.Tensor, m2: torch.Tensor, out: torch.Tensor, kernel: 'falkon.kernels.Kernel', n: int, m: int, comp_dt: torch.dtype, dev: torch.device, tid: int):
    N, D = m1.shape
    M = m2.shape[0]
    """ Run splitting along N, M """
    bwd_out = torch.tensor(0.0, dtype=torch.float64, device=out.device)
    with ExitStack() as stack:
        stream = None
        if dev.type == 'cuda':
            stream = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(stream))
        for i in range(0, N, n):
            leni = min(n, N - i)
            c_dev_m1 = m1[i:i + leni, :]
            for j in range(0, M, m):
                lenj = min(m, M - j)
                c_dev_m2 = m2[j:j + lenj, :]
                c_dev_out = kernel.compute_diff(c_dev_m1, c_dev_m2, diag=False)
                c_out = c_dev_out
                bwd_out = bwd_out + c_out.mul(out[i:i + leni, j:j + lenj]).sum()
        if tid != -1 and stream is not None:
            stream.synchronize()
    return bwd_out


def mm_run_thread(m1: torch.Tensor, m2: torch.Tensor, out: torch.Tensor, kernel: 'falkon.kernels.Kernel', n: int, m: int, comp_dt: torch.dtype, dev: torch.device, tid: int):
    is_ooc = dev.type != m1.device.type
    change_dtype = comp_dt != m1.dtype
    N, D = m1.shape
    M = m2.shape[0]
    """ Initialize extra buffers """
    flat_offset = 0
    total_memory = 0
    has_gpu_bufs = is_ooc or change_dtype
    if has_gpu_bufs:
        total_memory += n * m + n * D + m * D
    flat_dev_t = torch.empty(size=(total_memory,), dtype=comp_dt, device=dev)
    dev_nm, dev_m1, dev_m2 = None, None, None
    if has_gpu_bufs:
        dev_nm, flat_offset = _extract_flat(flat_dev_t, size=(n, m), other=out, offset=flat_offset)
        dev_m1, flat_offset = _extract_flat(flat_dev_t, size=(n, D), other=m1, offset=flat_offset)
        dev_m2, flat_offset = _extract_flat(flat_dev_t, size=(m, D), other=m2, offset=flat_offset)
    """ Run splitting along N, M """
    with ExitStack() as stack, torch.inference_mode():
        stream = None
        if dev.type == 'cuda':
            stream = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(stream))
        for i in range(0, N, n):
            leni = min(n, N - i)
            if has_gpu_bufs:
                c_dev_m1 = copy(m1[i:i + leni, :], dev_m1[:leni, :], non_blocking=True, allow_dtype_change=True)
            else:
                c_dev_m1 = m1[i:i + leni, :]
            for j in range(0, M, m):
                lenj = min(m, M - j)
                if has_gpu_bufs:
                    c_dev_m2 = copy(m2[j:j + lenj, :], dev_m2[:lenj, :], non_blocking=True, allow_dtype_change=True)
                    c_dev_out = dev_nm[:leni, :lenj]
                else:
                    c_dev_m2 = m2[j:j + lenj, :]
                    c_dev_out = out[i:i + leni, j:j + lenj]
                c_dev_out.fill_(0.0)
                kernel.compute(c_dev_m1, c_dev_m2, c_dev_out, diag=False)
                if has_gpu_bufs:
                    copy(c_dev_out, out[i:i + leni, j:j + lenj], non_blocking=True, allow_dtype_change=True)
        if tid != -1 and stream is not None:
            stream.synchronize()
    return out


def solve_lin(b, c):
    return -c / b


def solve_quad(a, b, c):
    if a == 0:
        return float('inf')
    return (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def select_dim_over_nm(max_n, max_m, d, coef_nd, coef_md, coef_nm, coef_n, coef_m, rest, max_mem):
    """Finds the optimal values for `n` and `m` to fit in available memory.

    This function should be called for problems where the GPU needs to hold
    two blocks of data (one of size m, one of size n) and one kernel block
    (of size n x m).

    Parameters
    -----------
    max_n : int
        The maximum value for n (the first dimension of the problem)
    max_m : int
        The maximum value for m (the second dimension of the problem)
    d : int
        The dimensionality of the data
    coef_nd : float
        How many n*d blocks need to be held in memory
    coef_md : float
        How many m*d blocks need to be held in memory
    coef_nm : float
        How many m*n blocks need to be held in memory
    coef_n : float
        How many n-dimensional vectors need to be held in memory
    coef_m : float
        How many m-dimensional vectors need to be held in memory
    rest : float
        additional bytes to be kept in memory
    max_mem : float
        The amount of available memory in bytes. This is the main problem constraint

    Returns
    -------
    out_n : int
        The dimension n to use in order to fit in available memory
    out_m : int
        The dimension m to use in order to fit in available memory

    Notes
    ------
    The equation gives a hyperbola. We intersect the hyperbola
    with a line from the origin, with the slope given by the ratio
    of max_m and max_n. We then solve a quadratic equation to find
    the intersection point.
    """
    fac = max_m / max_n
    if coef_nm == 0 and (coef_nd == 0 and coef_md == 0 and coef_n == 0 and coef_m == 0):
        v_n = max_n
    elif coef_nm == 0:
        v_n = solve_lin(b=d * (coef_nd + fac * coef_md) + coef_n + coef_m * fac, c=rest - max_mem)
    else:
        v_n = solve_quad(a=fac * coef_nm, b=d * (fac * coef_md + coef_nd) + fac * coef_m + coef_n, c=rest - max_mem)
    v_m = fac * v_n
    out_n = int(min(v_n, max_n))
    out_m = int(min(v_m, max_m))
    if out_n <= 0 or out_m <= 0:
        raise MemoryError('Available memory %.2fMB is not enough.' % (max_mem / 2 ** 20))
    return out_n, out_m


def sparse_mm_run_thread(m1: SparseTensor, m2: SparseTensor, out: torch.Tensor, kernel: 'falkon.kernels.Kernel', n: int, m: int, comp_dt: torch.dtype, dev: torch.device, tid: int):
    """Inner loop to compute (part of) a kernel matrix for two sparse input tensors

    Parameters
    ----------
    m1
        Left input tensor for computing the kernel
    m2
        Right input tensor for computing the kernel
    out
        Output dense matrix in which to store the result
    kernel
        Kernel object, used for computing the kernel. This must implement the
        :meth:`falkon.kernels.kernel.Kernel.compute_sparse` method.
    n
        Block size for the first axis of `m1`
    m
        Block size for the first ais of `m2`
    comp_dt
        Data-type in which to run the actual calculations (may be different from the data-type
        of `m1` or `m2`).
    dev
        Device on which to run the calculations
    tid
        Thread ID. If on the main thread this will be -1

    Returns
    -------
    out : torch.Tensor
        The kernel matrix. Should use the same underlying storage as the parameter `out`.
    """
    is_ooc = dev.type != m1.device.type
    change_dtype = comp_dt != m1.dtype
    N, D = m1.shape
    M = m2.shape[0]
    """ Initialize extra buffers """
    has_gpu_bufs = is_ooc or change_dtype
    dev_nm = None
    if has_gpu_bufs:
        dev_nm = create_same_stride((n, m), out, comp_dt, dev)
    """ Run splitting along N, M """
    with ExitStack() as stack, torch.inference_mode():
        stream = None
        if dev.type == 'cuda':
            stream = tcd.current_stream(dev) if tid == -1 else tcd.Stream(dev)
            stack.enter_context(tcd.device(dev))
            stack.enter_context(tcd.stream(stream))
        for j in range(0, M, m):
            lenj = min(m, M - j)
            c_m2 = m2.narrow_rows(j, lenj)
            if dev.type == 'cuda':
                c_dev_m2 = SparseTensor.from_scipy(c_m2.transpose_csc().to_scipy().tocsr(copy=False)).index_to_int()
            else:
                c_dev_m2 = c_m2.transpose_csc()
            for i in range(0, N, n):
                leni = min(n, N - i)
                c_m1 = m1.narrow_rows(i, leni)
                if dev.type == 'cuda':
                    c_dev_m1 = c_m1.index_to_int()
                else:
                    c_dev_m1 = c_m1
                if has_gpu_bufs:
                    c_dev_out = dev_nm[:leni, :lenj]
                else:
                    c_dev_out = out[i:i + leni, j:j + lenj]
                c_dev_out.fill_(0.0)
                c_dev_out = kernel.compute_sparse(c_dev_m1, c_dev_m2, c_dev_out, diag=False, X1_csr=c_m1, X2_csr=c_m2)
                if has_gpu_bufs:
                    copy(c_dev_out, out[i:i + leni, j:j + lenj], non_blocking=True, allow_dtype_change=True)
            if tid != -1 and stream is not None:
                stream.synchronize()
    return out


def mm_run_starter(proc_idx, queue, device_id):
    a: ArgsFmm = queue.get()
    X1, X2, out = a.X1, a.X2, a.out
    kernel, computation_dtype = a.kernel, a.gpu_dtype
    differentiable = a.differentiable
    max_mem = a.max_mem
    if device_id < 0:
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda:%d' % device_id)
    is_ooc = dev.type != X1.device.type
    change_dtype = computation_dtype != X1.dtype
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)
    avail_mem = max_mem / sizeof_dtype(computation_dtype)
    extra_mem = kernel.extra_mem()
    if differentiable:
        diff_coef_nm = 10
        assert not is_sparse, 'Sparse + differentiable mmvs are not supported'
        n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1], coef_nd=extra_mem.get('nd', 0) + 1, coef_md=extra_mem.get('md', 0) + 1, coef_nm=(extra_mem.get('nm', 0) + 1) * diff_coef_nm, coef_n=extra_mem.get('n', 0), coef_m=extra_mem.get('m', 0), rest=extra_mem.get('d', 0), max_mem=avail_mem)
    elif is_sparse:
        if is_ooc or change_dtype:
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1], coef_nd=2 * X1.density, coef_md=2 * X2.density, coef_nm=3, coef_n=0, coef_m=0, rest=0, max_mem=avail_mem)
        else:
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1], coef_nd=0, coef_md=0, coef_nm=0, coef_n=0, coef_m=0, rest=0, max_mem=avail_mem)
    elif is_ooc or change_dtype:
        n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1], coef_nd=extra_mem.get('nd', 0) + 1, coef_md=extra_mem.get('md', 0) + 1, coef_nm=extra_mem.get('nm', 0) + 1, coef_n=extra_mem.get('n', 0), coef_m=extra_mem.get('m', 0), rest=extra_mem.get('d', 0), max_mem=avail_mem)
    else:
        n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1], coef_nd=extra_mem.get('nd', 0), coef_md=extra_mem.get('md', 0), coef_nm=extra_mem.get('nm', 0), coef_n=extra_mem.get('n', 0), coef_m=extra_mem.get('m', 0), rest=extra_mem.get('d', 0), max_mem=avail_mem)
    if differentiable:
        return mm_diff_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev, tid=proc_idx)
    elif is_sparse:
        return sparse_mm_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev, tid=proc_idx)
    else:
        return mm_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev, tid=proc_idx)


def select_dim_over_nd(max_n, max_d, coef_nd, coef_n, coef_d, rest, max_mem):
    """
    solves the problem, max n*d such that n <= maxN, d <= maxD and
    coef_nd*nd + coef_n*n + coef_d*d + rest <= tot
    """
    if coef_nd == 0 and (coef_n == 0 or coef_d == 0):
        if coef_d == coef_n:
            n, d = max_n, max_d
        elif coef_n == 0:
            n = max_n
            d = (max_mem - rest) / coef_d
        else:
            n = (max_mem - rest) / coef_n
            d = max_d
    else:
        if coef_nd == 0:
            x = solve_lin(b=coef_n + coef_d, c=rest - max_mem)
        else:
            try:
                x = solve_quad(a=coef_nd, b=coef_n + coef_d, c=rest - max_mem)
            except ValueError:
                x = -1
        n = math.floor(min(max_n, x))
        d = math.floor(min(max_d, x))
        if d == max_d and n < max_n:
            n = (max_mem - rest - coef_d * d) / (coef_nd * d + coef_n)
        elif d < max_d and n == max_n:
            d = (max_mem - rest - coef_n * n) / (coef_nd * n + coef_d)
    n = int(min(max_n, n))
    d = int(min(max_d, d))
    if n <= 0 or d <= 0:
        raise MemoryError('Available memory %.2fMB is not enough.' % (max_mem / 2 ** 20))
    return n, d


def select_dim_over_nm_v2(max_n, max_m, coef_nm, coef_n, coef_m, rest, max_mem):
    """
    solves the problem, max n*m such that n <= maxN, m <= maxM and
    coef_nm*nm + coef_n*n + coef_m*m <= tot
    """
    return select_dim_over_nd(max_n=max_n, max_d=max_m, coef_nd=coef_nm, coef_n=coef_n, coef_d=coef_m, rest=rest, max_mem=max_mem)


def _dense_mmv_blk_sizes(n: int, d: int, m: int, t: int, avail_mem: float, extra_mem: Dict[str, float], m1_ic: bool, m2_ic: bool, v_ic: bool, out_ic: bool) ->Tuple[int, int, int]:
    coef_nm = 1
    coef_n = d if not m1_ic else 0
    coef_m = d if not m2_ic else 0
    coef_n = coef_n + t if not out_ic else coef_n
    coef_m = coef_m + t if not v_ic else coef_m
    blk_n, blk_m = select_dim_over_nm_v2(max_n=n, max_m=m, max_mem=avail_mem, coef_nm=coef_nm + extra_mem.get('nm', 0), coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d, coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d, rest=extra_mem.get('d', 0))
    mem_needed = blk_m * blk_n
    mem_needed += blk_n * coef_n
    mem_needed += blk_m * coef_m
    return blk_n, blk_m, mem_needed


def _sparse_mmv_blk_sizes(n, d, m, t, avail_mem, extra_mem, incore: bool, m1_density: float, m2_density: float):
    coef_nm = 3
    coef_n, coef_m, coef_rest = 0, 0, 0
    if not incore:
        coef_n += 2 + 2 * d * m1_density + t
        coef_m += 2 * d * m2_density + t
        coef_rest = d
    blk_n, blk_m = select_dim_over_nm_v2(max_n=n, max_m=m, max_mem=avail_mem, coef_nm=coef_nm + extra_mem.get('nm', 0), coef_n=coef_n + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d, coef_m=coef_m + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d, rest=coef_rest + extra_mem.get('d', 0))
    mem_needed = blk_m * blk_n
    if not incore:
        mem_needed += (blk_n + blk_m) * t
    return blk_n, blk_m, mem_needed


def mmv_diff_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: Optional[torch.Tensor], out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, blk_m: int, dev: torch.device, tid: int):
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape
    inputs = [m1, m2, v] + list(kernel.diff_params.values())
    grads = []
    for ipt in inputs:
        if ipt.requires_grad:
            grads.append(torch.zeros_like(ipt))
        else:
            grads.append(None)
    inputs_need_grad, input_idxs = zip(*[(ipt, idx) for idx, ipt in enumerate(inputs) if ipt.requires_grad])
    with ExitStack() as stack:
        s1, s2 = _init_two_streams(stack, dev, tid)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            c_dev_m1 = m1[i:i + leni, :]
            c_dev_m1_g = None if grads[0] is None else grads[0][i:i + leni, :]
            c_dev_out = out[i:i + leni, :]
            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                c_dev_m2 = m2[j:j + lenj, :]
                c_dev_m2_g = None if grads[1] is None else grads[1][j:j + lenj, :]
                with ExitStack() as stack_s2:
                    if not incore:
                        stack_s2.enter_context(tcd.stream(s2))
                    c_dev_v = v[j:j + lenj, :]
                    c_dev_v_g = None if grads[2] is None else grads[2][j:j + lenj, :]
                c_dev_ker = kernel.compute_diff(c_dev_m1, c_dev_m2, diag=False)
                if not incore:
                    s2.synchronize()
                c_dev_mmv = c_dev_ker @ c_dev_v
                c_inputs = [c_dev_m1, c_dev_m2, c_dev_v] + list(kernel.diff_params.values())
                c_dev_grads_old = [c_dev_m1_g, c_dev_m2_g, c_dev_v_g] + grads[3:]
                c_dev_grads = torch.autograd.grad(c_dev_mmv, [c_inputs[idx] for idx in input_idxs], grad_outputs=c_dev_out)
                for c_grad, c_idx in zip(c_dev_grads, input_idxs):
                    c_dev_grads_old[c_idx].add_(c_grad)
                if grads[1] is not None:
                    grads[1][j:j + lenj, :] = c_dev_m2_g
                if grads[2] is not None:
                    grads[2][j:j + lenj, :] = c_dev_v_g
                if not incore:
                    s1.synchronize()
            if grads[0] is not None:
                grads[0][i:i + leni, :] = c_dev_m1_g
        if tid != -1 and s1 is not None:
            s1.synchronize()
    return grads


def mmv_run_thread(m1: torch.Tensor, m2: torch.Tensor, v: Optional[torch.Tensor], out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, blk_m: int, mem_needed: int, dev: torch.device, tid: int):
    m1_ic, m2_ic, v_ic, out_ic = _is_incore(dev, m1.device), _is_incore(dev, m2.device), _is_incore(dev, v.device), _is_incore(dev, out.device)
    incore = all((m1_ic, m2_ic, v_ic, out_ic))
    N, D = m1.shape
    M, T = v.shape
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    dev_ker, flat_offset = _extract_flat(flat_gpu, size=(blk_n, blk_m), other=out, offset=flat_offset)
    if m1_ic:
        dev_m1 = None
    else:
        dev_m1, flat_offset = _extract_flat(flat_gpu, size=(blk_n, D), other=m1, offset=flat_offset)
    if m2_ic:
        dev_m2 = None
    else:
        dev_m2, flat_offset = _extract_flat(flat_gpu, size=(blk_m, D), other=m2, offset=flat_offset)
    if v_ic:
        dev_v = None
    else:
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)
    if out_ic:
        dev_out = None
    else:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)
    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            if m1_ic:
                c_dev_m1 = m1[i:i + leni, :]
            else:
                c_dev_m1 = copy(m1[i:i + leni, :], dev_m1[:leni, :], non_blocking=True)
            if out_ic:
                c_dev_out = out[i:i + leni]
            else:
                c_dev_out = dev_out[:leni]
            c_dev_out.fill_(0.0)
            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                if m2_ic:
                    c_dev_m2 = m2[j:j + lenj, :]
                else:
                    c_dev_m2 = copy(m2[j:j + lenj, :], dev_m2[:lenj, :], non_blocking=True)
                if v_ic:
                    c_dev_v = v[j:j + lenj, :]
                else:
                    with ExitStack() as stack2:
                        if dev.type == 'cuda':
                            stack2.enter_context(tcd.stream(s2))
                        c_dev_v = copy(v[j:j + lenj, :], dev_v[:lenj, :], non_blocking=True)
                c_dev_ker = dev_ker[:leni, :lenj].fill_(0.0)
                c_dev_ker = kernel.compute(c_dev_m1, c_dev_m2, c_dev_ker, diag=False)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)
                if not incore:
                    s1.synchronize()
            if not out_ic:
                copy(c_dev_out, out[i:i + leni], non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


def sparse_mmv_run_thread(m1: SparseTensor, m2: SparseTensor, v: torch.Tensor, out: torch.Tensor, kernel: 'falkon.kernels.Kernel', blk_n: int, blk_m: int, mem_needed: int, dev: torch.device, tid: int):
    """Inner loop to compute (part of) a kernel-vector product for sparse input matrices.

    Parameters
    ----------
    m1
        Left input tensor for computing the kernel
    m2
        Right input tensor for computing the kernel
    v
        Dense vector to be multiplied by the kernel matrix
    out
        Dense output vector which should store the result of the kernel vector product on exit
        from this function.
    kernel
        Kernel object, used for computing the kernel. This must implement the
        :meth:`falkon.kernels.kernel.Kernel.compute_sparse` method.
    blk_n
        Block size for the first axis of `m1`
    blk_m
        Block size for the first ais of `m2`
    mem_needed
        Memory needed for pre-allocations
    dev
        Device on which to run the calculations
    tid
        Thread ID or -1 if on main thread

    Returns
    -------
    out : torch.Tensor
        The kernel matrix. Should use the same underlying storage as the parameter `out`.
    """
    incore = _is_incore(dev, m1.device)
    N, D = m1.shape
    M, T = v.shape
    """ Initialize extra buffers """
    flat_gpu = torch.empty(size=(mem_needed,), dtype=m1.dtype, device=dev)
    flat_offset = 0
    ker_gpu = extract_fortran(flat_gpu, size=(blk_n, blk_m), offset=flat_offset)
    flat_offset += np.prod(ker_gpu.shape)
    dev_v, dev_out = None, None
    if not incore:
        dev_out, flat_offset = _extract_flat(flat_gpu, size=(blk_n, T), other=out, offset=flat_offset)
        dev_v, flat_offset = _extract_flat(flat_gpu, size=(blk_m, T), other=v, offset=flat_offset)
    with ExitStack() as stack, torch.inference_mode():
        s1, s2 = _init_two_streams(stack, dev, tid)
        for i in range(0, N, blk_n):
            leni = min(blk_n, N - i)
            c_m1 = m1.narrow_rows(i, leni)
            if incore:
                c_dev_out = out[i:i + leni]
                c_dev_m1 = c_m1
            else:
                c_dev_out = dev_out[:leni]
                c_dev_m1 = c_m1.index_to_int()
            c_dev_out.fill_(0.0)
            for j in range(0, M, blk_m):
                lenj = min(blk_m, M - j)
                c_m2 = m2.narrow_rows(j, lenj)
                if incore:
                    c_dev_m2 = c_m2.transpose_csc()
                    c_dev_v = v[j:j + lenj]
                else:
                    c_dev_m2 = SparseTensor.from_scipy(c_m2.transpose_csc().to_scipy().tocsr(copy=False)).index_to_int()
                    with ExitStack() as stack2:
                        if dev.type == 'cuda':
                            stack2.enter_context(tcd.stream(s2))
                        c_dev_v = copy(v[j:j + lenj], dev_v[:lenj], non_blocking=True)
                c_dev_ker = ker_gpu[:leni, :lenj].fill_(0.0)
                c_dev_ker = kernel.compute_sparse(c_dev_m1, c_dev_m2, c_dev_ker, diag=False, X1_csr=c_m1, X2_csr=c_m2)
                if not incore:
                    s2.synchronize()
                c_dev_out.addmm_(c_dev_ker, c_dev_v)
                if not incore:
                    copy(c_dev_out, out[i:i + leni], non_blocking=True)
        if tid != -1 and s1 is not None:
            s1.synchronize()


def mmv_run_starter(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1, X2, v, out = a.X1, a.X2, a.v, a.out
    kernel = a.kernel
    max_mem = a.max_mem
    differentiable = a.differentiable
    dev = _dev_from_id(device_id)
    incore = _is_incore(dev, X1.device)
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)
    avail_mem = max_mem / sizeof_dtype(X1.dtype)
    extra_mem = kernel.extra_mem()
    n, d = X1.shape
    m, t = v.shape
    if differentiable:
        diff_coef_nm = 4
        assert not is_sparse, 'Sparse + differentiable mmvs are not supported'
        blk_n, blk_m = select_dim_over_nm_v2(max_n=n, max_m=m, max_mem=avail_mem, coef_nm=diff_coef_nm + extra_mem.get('nm', 0), coef_n=2 * (d + t + extra_mem.get('n', 0) + extra_mem.get('nd', 0) * d), coef_m=2 * (d + t + extra_mem.get('m', 0) + extra_mem.get('md', 0) * d), rest=extra_mem.get('d', 0))
        return mmv_diff_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, dev, tid=proc_idx)
    if is_sparse:
        blk_n, blk_m, mem_needed = _sparse_mmv_blk_sizes(n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, incore=incore, m1_density=X1.density, m2_density=X2.density)
        return sparse_mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev, tid=proc_idx)
    else:
        m1_ic, m2_ic, v_ic, out_ic = _is_incore(dev, X1.device), _is_incore(dev, X2.device), _is_incore(dev, v.device), _is_incore(dev, out.device)
        blk_n, blk_m, mem_needed = _dense_mmv_blk_sizes(n=n, m=m, d=d, t=t, avail_mem=avail_mem, extra_mem=extra_mem, m1_ic=m1_ic, m2_ic=m2_ic, v_ic=v_ic, out_ic=out_ic)
        return mmv_run_thread(X1, X2, v, out, kernel, blk_n, blk_m, mem_needed, dev, tid=proc_idx)

