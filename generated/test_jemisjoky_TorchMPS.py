import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
torchmps = _module
contractables = _module
embeddings = _module
mps_base = _module
prob_mps = _module
tests = _module
benchmarks = _module
test_benchmark_mps_base = _module
test_benchmark_prob_mps = _module
custom_feature_map = _module
dynamic_mps_basic = _module
mat_region_open_periodic_bcs = _module
static_mps_basic = _module
svd_flex = _module
ti_mps_basic = _module
test_embeddings = _module
test_mps_base = _module
test_prob_mps = _module
test_utils2 = _module
utils_for_tests = _module
torchmps = _module
utils = _module
utils2 = _module
train_script = _module

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


from math import sqrt


from math import pi


from functools import partial


from typing import Union


from typing import Optional


from typing import Callable


from torch import nn


import warnings


from itertools import repeat


from typing import Sequence


from typing import Tuple


from torch import Tensor


from random import randint


import numpy as np


import torch.nn as nn


from math import log


import time


from torchvision import transforms


from torchvision import datasets


class DataDomain:
    """
    Defines a domain for input data to a probabilistic model

    DataDomain supports both continuous and discrete domains, with the
    latter always associated with indices of the form `0, 1, ..., max_val-1`.
    For continuous domains, real intervals of the form `[min_val, max_val]`
    can be defined.

    Args:
        continuous (bool): Whether data domain is continuous or discrete
        max_val (int or float): For discrete domains, this is the number of
            indices to use, with the maximum index being max_val - 1. For
            continuous domains, this is the endpoint of the real interval.
        min_val (float): Only used for continuous domains, this is the
            startpoint of the real interval.
    """

    def __init__(self, continuous: bool, max_val: Union[int, float], min_val: Optional[float]=None):
        if continuous:
            assert max_val > min_val
            self.min_val = min_val
        else:
            assert max_val >= 0
        self.max_val = max_val
        self.continuous = continuous


class TrainableEmbedding(nn.Module):
    """
    Framework for trainable embedding function converting data to vectors

    This acts as a wrapper for a user-specified `torch.nn.Module` instance,
    whose parameters are trained jointly with those of the MPS using it.

    Args:
        emb_fun (torch.nn.Module): Initialized function arbitrary tensors of
            values and returning tensor of embedded vectors, which has one
            additional axis in the last position. These values must be either
            integers, for discrete data domains, or reals, for continuous
            data domains.
        data_domain (DataDomain): Object which specifies the domain on which
            the data fed to the embedding function is defined.
    """

    def __init__(self, emb_fun: Callable, data_domain: DataDomain):
        super().__init__()
        assert isinstance(emb_fun, nn.Module)
        self.domain = data_domain
        self.emb_fun = lambda x: emb_fun(x[..., None])

    def make_lambda(self, num_points: int=1000):
        """
        Compute the lambda matrix used for normalization
        """
        if self.domain.continuous:
            points = torch.linspace(self.domain.min_val, self.domain.max_val, steps=num_points)
            self.num_points = num_points
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == num_points
            emb_mats = einsum('bi,bj->bij', emb_vecs, emb_vecs.conj())
            lamb_mat = torch.trapz(emb_mats, points, dim=0)
        else:
            points = torch.arange(self.domain.max_val).long()
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == self.domain.max_val
            emb_mats = einsum('bi,bj->bij', emb_vecs, emb_vecs.conj())
            lamb_mat = torch.sum(emb_mats, dim=0)
        assert lamb_mat.ndim == 2
        assert lamb_mat.shape[0] == lamb_mat.shape[1]
        self.lamb_mat = lamb_mat
        if torch.allclose(lamb_mat.diag().diag(), lamb_mat):
            lamb_mat = lamb_mat.diag()
            if torch.allclose(lamb_mat.mean(), lamb_mat):
                self.lamb_mat = lamb_mat.mean()
            else:
                self.lamb_mat = lamb_mat

    def forward(self, input_data):
        """
        Embed input data via the user-specified embedding function
        """
        self.make_lambda()
        return self.emb_fun(input_data)


class FixedEmbedding(nn.Module):
    """
    Framework for fixed embedding function converting data to vectors

    Args:
        emb_fun (function): Function taking arbitrary tensors of values and
            returning tensor of embedded vectors, which has one additional
            axis in the last position. These values must be either integers,
            for discrete data domains, or reals, for continuous data domains.
        data_domain (DataDomain): Object which specifies the domain on which
            the data fed to the embedding function is defined.
    """

    def __init__(self, emb_fun: Callable, data_domain: DataDomain):
        super().__init__()
        assert hasattr(emb_fun, '__call__')
        self.domain = data_domain
        self.emb_fun = emb_fun
        self.make_lambda()

    @torch.no_grad()
    def make_lambda(self, num_points: int=1000):
        """
        Compute the lambda matrix used for normalization
        """
        if self.domain.continuous:
            points = torch.linspace(self.domain.min_val, self.domain.max_val, steps=num_points)
            self.num_points = num_points
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == num_points
            emb_mats = einsum('bi,bj->bij', emb_vecs, emb_vecs.conj())
            lamb_mat = torch.trapz(emb_mats, points, dim=0)
        else:
            points = torch.arange(self.domain.max_val).long()
            emb_vecs = self.emb_fun(points)
            assert emb_vecs.ndim == 2
            self.emb_dim = emb_vecs.shape[1]
            assert emb_vecs.shape[0] == self.domain.max_val
            emb_mats = einsum('bi,bj->bij', emb_vecs, emb_vecs.conj())
            lamb_mat = torch.sum(emb_mats, dim=0)
        assert lamb_mat.ndim == 2
        assert lamb_mat.shape[0] == lamb_mat.shape[1]
        self.lamb_mat = lamb_mat
        if torch.allclose(lamb_mat.diag().diag(), lamb_mat):
            lamb_mat = lamb_mat.diag()
            if torch.allclose(lamb_mat.mean(), lamb_mat):
                self.lamb_mat = lamb_mat.mean()
            else:
                self.lamb_mat = lamb_mat

    def forward(self, input_data):
        """
        Embed input data via the user-specified embedding function
        """
        return self.emb_fun(input_data)


TensorSeq = Union[Tensor, Sequence[Tensor]]


def shape_broadcast(shape_list: Sequence[tuple]):
    """
    Predict shape of broadcasted tensors with given input shapes

    Code based on Stack Overflow post `here <https://stackoverflow.com/question
    s/54859286/is-there-a-function-that-can-apply-numpys-broadcasting-rules-to-
    a-list-of-shape/>`_

    Args:
        shape_list: Sequence of shapes, each input as a tuple

    Returns:
        b_shape: Broadcasted shape of those input in `shape_list`
    """
    max_shp = max(shape_list, key=len)
    out = list(max_shp)
    for shp in shape_list:
        if shp is max_shp:
            continue
        for i, x in enumerate(shp, -len(shp)):
            if x != 1 and x != out[i]:
                if out[i] != 1:
                    raise ValueError
                out[i] = x
    return tuple(out)


def batch_broadcast(tens_list: Sequence[Tensor], num_nonbatch: Sequence[int]):
    """
    Broadcast collection of tensors to have matching batch indices

    Broadcasting behavior is identical to standard PyTorch/NumPy but with
    broadcasting only performed on batch indices, which are always assumed
    to be the left-most indices. The separation between batch and non-batch
    indices is set by `num_nonbatch`, which gives the number of non-batch
    indices in each tensor.

    Args:
        tens_list: Sequence of tensors whose batch indices are being
            broadcast together. If the shape of batch indices cannot be
            broadcast, then `batch_broadcast` will throw an error
        num_nonbatch: Sequence of integers describing the number of
            non-batch indices in each of the tensors in `tens_list`. These
            non-batch indices are assumed to be the right-most indices of
            each respective tensor

    Returns:
        out_list: Sequence of tensors, which are broadcasted versions of
            those input in `tens_list`
    """
    assert not isinstance(tens_list, Tensor)
    assert len(tens_list) == len(num_nonbatch)
    assert all(i >= 0 for i in num_nonbatch)
    assert all(t.ndim >= nnb for t, nnb in zip(tens_list, num_nonbatch))
    if len(tens_list) < 2:
        return tens_list
    b_shapes = [t.shape[:t.ndim - nnb] for t, nnb in zip(tens_list, num_nonbatch)]
    try:
        full_batch = shape_broadcast(b_shapes)
        bdims = len(full_batch)
    except ValueError:
        raise ValueError(f"Following batch shapes couldn't be broadcast: {tuple(b_shapes)}")

    def safe_expand(t, shp):
        return t if len(shp) == 0 else t.expand(*shp)
    tens_list = [t[(None,) * (bdims + nnb - t.ndim)] for t, nnb in zip(tens_list, num_nonbatch)]
    shapes = [(full_batch + t.shape[bdims:]) for t in tens_list]
    out_list = tuple(safe_expand(t, shp) for t, shp in zip(tens_list, shapes))
    return out_list


def bundle_tensors(tensors: TensorSeq, dim: int=0) ->TensorSeq:
    """
    When possible, converts a sequence of tensors into single batch tensor

    When all input tensors have the same shape or only one tensor is input,
    a batch tensor is produced with a new batch index. Collections of
    tensors with inhomogeneous shapes are returned unchanged.

    Args:
        tensors: Sequence of tensors
        dim: Location of the new batch dimension

    Returns:
        out_tens: Single batched tensor, when possible, or unchanged input
    """
    if isinstance(tensors, Tensor):
        return tensors
    if len(set(t.shape for t in tensors)) > 1 or len(tensors) == 0:
        return tensors
    else:
        return torch.stack(tensors, dim=dim)


def mat_reduce_par(matrices: Tensor) ->Tuple[Tensor, Tensor]:
    """
    Contract sequence of square matrices with parallel mat-mat multiplies

    Args:
        matrices: Sequence of matrices to multiply, specified as a single
            tensor with shape `(batch, seq_len, bond_dim, bond_dim)`.

    Returns:
        prod_mat: Product of input matrices.
        log_scale: Vector with shape `(batch, 1, 1)` containing the logarithms
            of positive-valued corrections to the matrices in `prod_mat`, so
            that the actual values are `prod_mat * exp(log_scale)`.
    """
    assert matrices.ndim >= 3
    s_dim = -3
    n_mats = matrices.shape[s_dim]
    log_scale = torch.zeros(matrices.shape[:-3])[..., None, None]
    if n_mats == 0:
        eye = torch.eye(matrices.shape[-1], dtype=matrices.dtype)
        matrices, _ = batch_broadcast((eye, matrices), (2, 3))
        return matrices, log_scale
    elif n_mats == 1:
        return matrices.squeeze(dim=s_dim), log_scale
    assert matrices.shape[-2] == matrices.shape[-1]
    bond_dim = matrices.shape[-1]
    while n_mats > 1:
        half_n = n_mats // 2
        floor_n = half_n * 2
        even_mats = matrices[..., 0:floor_n:2, :, :]
        odd_mats = matrices[..., 1:floor_n:2, :, :]
        leftover = matrices[..., floor_n:, :, :]
        matrices = even_mats @ odd_mats
        matrices = torch.cat((matrices, leftover), dim=s_dim)
        n_mats = matrices.shape[s_dim]
        rescales = matrices.abs().sum(dim=(-2, -1), keepdim=True) / bond_dim
        log_scale = log_scale + rescales.log().sum(dim=-3)
        matrices = matrices / rescales
    return matrices.squeeze(dim=s_dim), log_scale


def mat_reduce_seq(matrices: Sequence[Tensor]) ->Tensor:
    """
    Multiply sequence of matrices sequentially, from left to right

    Args:
        matrices: Sequence of matrices to multiply.

    Returns:
        prod_mat: Product of input matrices.
        log_scale: Logarithms of positive-valued corrections to the matrices
        in `prod_mat`, so that actual values are `prod_mat * exp(log_scale)`.
    """
    r2l = matrices[-1].size(-1) < matrices[0].size(-2)
    if r2l:
        matrices = tuple(m.transpose(-2, -1) for m in matrices[::-1])
    product, matrices = matrices[0], matrices[1:]
    log_scale = torch.zeros(product.shape[:-2])[..., None, None]
    for mat in matrices:
        product = torch.matmul(product, mat)
        target_norm = sqrt(product.shape[-2] * product.shape[-1])
        rescale = product.abs().sum(dim=(-2, -1), keepdim=True) / target_norm
        log_scale = log_scale + torch.log(rescale)
        product = product / rescale
    product = product.transpose(-2, -1) if r2l else product
    return product, log_scale


def contract_matseq(matrices: TensorSeq, left_vec: Optional[Tensor]=None, right_vec: Optional[Tensor]=None, parallel_eval: bool=False, log_format: bool=False) ->Tensor:
    """
    Matrix-multiply sequence of matrices with optional boundary vectors

    The output is a single matrix, or a vector/scalar if one/both boundary
    vectors are given. In the latter case, the first vector is treated as
    a row vector and the last as a column vector and put at the beginning
    or end of the sequence of matrices to reduce to a vector/scalar.

    Parallel matrix-matrix multiplications can be used to make the
    computation more GPU-friendly, at the cost of a larger overall compute
    cost. By default, this method is only used when an output matrix is
    desired, but can be forced by setting parallel_eval to True.

    When matrices or boundary vectors contain additional batch indices
    (assumed to be left-most indices), then batch matrix multiplication is
    carried out over all batch indices, which are broadcast together.
    Shapes described below neglect these additional batch indices, which will
    be possessed by all outputs whenever they are present in input `matrices`.

    Args:
        matrices: Single tensor of shape `(L, D, D)`, or sequence of
            matrices with compatible shapes :math:`(D_i, D_{i+1})`, for
            :math:`i = 0, 1, \\ldots, L`.
        left_vec: Left boundary vector with shape `(D_0,)`, or None if no
            left boundary is present.
        right_vec: Left boundary vector with shape `(D_L,)`, or None if no
            right boundary is present.
        parallel_eval: Whether or not to force parallel evaluation in
            matrix contraction, which requires all input matrices to have
            same shape.
            Default: ``False``
        log_format: Whether or not to return the output as two objects, a
            contraction output and a logarithm scale correction for each
            product of matrices in the original input.
            Default: ``False``

    Returns:
        contraction: Single scalar, vector, or matrix, equal to the
            sequential contraction of the input matrices with (resp.)
            two, one, or zero boundary vectors.
        log_scale: Real scalar corrections to the magnitude of `contraction`,
            so that the real output is `contraction * exp(log_scale)`. Only
            present when log_format is True.
    """
    bnd_vecs = [left_vec, right_vec]
    real_vec = [(v is not None) for v in bnd_vecs]
    num_vecs = sum(real_vec)
    assert all(v is None or isinstance(v, Tensor) for v in bnd_vecs)
    assert num_vecs <= 2
    same_shape = isinstance(matrices, Tensor)
    if not same_shape:
        matrices = bundle_tensors(matrices, dim=-3)
        same_shape = isinstance(matrices, Tensor)
    num_mats = matrices.shape[-3] if same_shape else len(matrices)
    use_parallel = same_shape and (parallel_eval or num_vecs == 0)
    if num_vecs == 0 and not same_shape:
        matrices = batch_broadcast(matrices, (2,) * num_mats)
    elif num_vecs == 1:
        v_ind = real_vec.index(True)
        vec = bnd_vecs[v_ind]
        if same_shape:
            vec, matrices = batch_broadcast((vec, matrices), (1, 3))
        else:
            outs = batch_broadcast((vec,) + tuple(matrices), (1,) + (2,) * num_mats)
            vec, matrices = outs[0], outs[1:]
        bnd_vecs[v_ind] = vec
    elif num_vecs == 2 and same_shape:
        outs = batch_broadcast(bnd_vecs + [matrices], (1, 1, 3))
        bnd_vecs, matrices = outs[:2], outs[2]
    elif num_vecs == 2 and not same_shape:
        outs = batch_broadcast(bnd_vecs + list(matrices), (1, 1) + (2,) * num_mats)
        bnd_vecs, matrices = outs[:2], outs[2:]
    if use_parallel:
        product, log_scale = mat_reduce_par(matrices)
        if real_vec[0]:
            product = torch.matmul(bnd_vecs[0][..., None, :], product)
        if real_vec[1]:
            product = torch.matmul(product, bnd_vecs[1][..., None])
    else:
        if num_vecs == 0 and len(matrices) == 0:
            raise ValueError('Must input at least one matrix or boundary vector to contract_matseq')
        if same_shape:
            matrices = [matrices[..., i, :, :] for i in range(num_mats)]
        else:
            matrices = list(matrices)
        if real_vec[0]:
            matrices = [bnd_vecs[0][..., None, :]] + matrices
        if real_vec[1]:
            matrices.append(bnd_vecs[1][..., None])
        product, log_scale = mat_reduce_seq(matrices)
    if real_vec[0]:
        product.squeeze_(-2)
        log_scale.squeeze_(-2)
    if real_vec[1]:
        product.squeeze_(-1)
        log_scale.squeeze_(-1)
    if log_format:
        return product, log_scale
    else:
        return product * torch.exp(log_scale)


def realify(tensor: Tensor) ->Tensor:
    """
    Convert approximately real complex tensor to real tensor

    Input must be approximately real, `realify` will raise error if not
    """
    if tensor.is_complex():
        assert torch.allclose(tensor.imag, torch.zeros(()), atol=0.0001)
        return tensor.real
    else:
        return tensor


def hermitian_trace(tensor: Tensor) ->Tensor:
    """
    Same as `torch.trace` for Hermitian matrices, ensures real output
    """
    if tensor.is_complex():
        return realify(torch.trace(tensor))
    else:
        return torch.trace(tensor)


def get_log_norm(core_tensor: Tensor, boundary_vecs: Tensor, length: Optional[int]=None, lamb_mat: Optional[Tensor]=None) ->Tensor:
    """
    Compute the log (squared) L2 norm of tensor described by MPS model

    Uses iterated tensor contraction to compute :math:`\\log(|\\psi|^2)`,
    where :math:`\\psi` is the n'th order tensor arising from contracting
    all MPS core tensors and boundary vectors together. In the Born machine
    paradigm this is equivalently :math:`\\log(Z)`, with :math:`Z` the
    normalization constant for the probability.

    Can be used to compute fixed-len log norms as well as arbitrary-len
    log norms, for the case of uniform MPS. The latter case is the log of
    the sum of all len-n squared norms, for :math:`n = 0, 1, \\ldots`.
    Non-convergent core tensors

    Args:
        core_tensor: Either single core tensor of shape `(input_dim, D, D)`
            or core tensor sequence of shape `(seq_len, input_dim, D, D)`,
            for `D` the bond dimension of the MPS.
        boundary_vecs: Matrix of shape `(2, D)` whose 0th/1st columns give
            the left/right boundary vectors of the MPS.
        length: In case of single core tensor, specifies the length for
            which the log L2 norm is computed. Non-negative values give
            fixed-len log norms, while -1 gives arbitrary-len log norm.
        lamb_mat: Positive semidefinite matrix :math:`\\Lambda` which is used
            in the presence of an input embedding to marginalize over all all
            values of the input domain.

    Returns:
        l_norm: Scalar value giving the log squared L2 norm of the
            probability amplitude tensor described by the MPS.
    """
    uniform = core_tensor.ndim == 3
    assert core_tensor.ndim in (3, 4)
    assert boundary_vecs.shape[-1] == core_tensor.shape[-1]
    assert uniform or length in (None, core_tensor.shape[0])
    assert lamb_mat is None or lamb_mat.ndim <= 2
    assert not (uniform and length is None)
    fixed_len = not uniform or length != -1
    uniform = core_tensor.ndim == 3
    left_vec, r_vec = boundary_vecs
    dens_mat = left_vec[:, None] @ left_vec[None, :].conj()
    log_norm = torch.zeros(())
    if uniform and fixed_len:
        core_tensor = (core_tensor,) * length
    if lamb_mat is None:
        t_op = lambda dm, ct: einsum('ilr,lp,ipq->rq', ct, dm, ct.conj())
    elif lamb_mat.ndim == 0:
        t_op = lambda dm, ct: lamb_mat * einsum('ilr,lp,ipq->rq', ct, dm, ct.conj())
    elif lamb_mat.ndim == 1:
        t_op = lambda dm, ct: einsum('i,ilr,lp,ipq->rq', lamb_mat, ct, dm, ct.conj())
    elif lamb_mat.ndim == 2:
        t_op = lambda dm, ct: einsum('ij,ilr,lp,jpq->rq', lamb_mat, ct, dm, ct.conj())

    def transfer_op(core_t, d_mat, l_norm, l_sum=None):
        d_mat = t_op(d_mat, core_t)
        trace = hermitian_trace(d_mat)
        l_norm = l_norm + torch.log(trace)
        d_mat = d_mat / trace
        if fixed_len:
            return d_mat, l_norm
        else:
            this_norm = einsum('r,rq,q->', r_vec, d_mat, r_vec.conj())
            new_lnorm = torch.log(this_norm) + l_norm
            l_sum = torch.logsumexp(torch.stack((l_sum, new_lnorm)))
            return d_mat, l_norm, l_sum
    if fixed_len:
        for core in core_tensor:
            dens_mat, log_norm = transfer_op(core, dens_mat, log_norm)
        correction = einsum('r,rq,q->', r_vec, dens_mat, r_vec.conj())
        return log_norm + torch.log(realify(correction))
    else:
        raise NotImplementedError


class CIndex:
    """Wrapper class that allows complex tensors to be indexed"""

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        if self.tensor.is_complex():
            t_r, t_i = self.tensor.real, self.tensor.imag
            return t_r[index] + 1.0j * t_i[index]
        else:
            return self.tensor[index]


def get_mat_slices(seq_input: Tensor, core_tensor: Tensor) ->Tensor:
    """
    Use sequential input and core tensor to get sequence of matrix slices

    Args:
        seq_input: Tensor with shape `(batch, seq_len)` for discrete input
            sequences, or `(batch, seq_len, input_dim)` for vector input
            sequences.
        core_tensor: Tensor with shape `(input_dim, bond_dim, bond_dim)`
            for uniform MPS or `(seq_len, input_dim, bond_dim, bond_dim)`
            for fixed-length MPS.

    Returns:
        mat_slices: Tensor with shape `(batch, seq_len, bond_dim, bond_dim)`,
            containing all transition matrix core slices of `core_tensor`
            relative to data in `seq_input`.
    """
    assert seq_input.ndim in (2, 3)
    assert core_tensor.ndim in (3, 4)
    indexing = seq_input.ndim == 2
    uniform = core_tensor.ndim == 3
    input_dim, bond_dim = core_tensor.shape[-3:-1]
    seq_len = seq_input.shape[1]
    if core_tensor.is_complex() and seq_input.is_floating_point():
        seq_input = seq_input
    if indexing and uniform:
        mat_slices = core_tensor[seq_input]
    elif indexing and not uniform:
        mat_slices = CIndex(core_tensor)[torch.arange(seq_len)[None], seq_input]
    elif not indexing and uniform:
        mat_slices = einsum('bti,ide->btde', seq_input, core_tensor)
    elif not indexing and not uniform:
        mat_slices = einsum('bti,tide->btde', seq_input, core_tensor)
    return mat_slices


def batch_to(tensor: Tensor, batch_shape: tuple, num_nonbatch: int):
    """
    Expand give tensor via broadcasting to have given batch dimensions

    Args:
        tensor: Tensor whose batch indices are being expanded.
        batch_shape: Shape to which the batch indices of `tensor` will be
            expanded to. If the batch indices of `tensor` is incompatible
            with `batch_shape`, then `batch_to` will throw an error.
        num_nonbatch: Integer describing the number of non-batch indices
            in `tensor`. Non-batch indices are assumed to be the
            right-most indices `tensor`.

    Returns:
        out_tensor: Broadcasted version of input tensor.
    """
    batch_ref = torch.empty(batch_shape)
    out_tensor, _ = batch_broadcast((tensor, batch_ref), (num_nonbatch, 0))
    return out_tensor


def phaseify(tensor: Tensor) ->Tensor:
    """
    Convert real tensor into complex one with random complex phases
    """
    return tensor * torch.exp(2.0j * pi * torch.rand_like(tensor))


def near_eye_init(shape: tuple, is_complex: bool=False, noise: float=0.001) ->Tensor:
    """
    Initialize an MPS core tensor with all slices close to identity matrix

    Args:
        shape: Shape of the core tensor being initialized.
        is_complex: Whether to initialize a complex core tensor.
            Default: False
        noise: Normalized noise value setting stdev around identity matrix.
            Default: 1e-3

    Returns:
        core_tensor: Randomly initialized near-identity core tensor.
    """
    assert len(shape) >= 3
    if shape[-2] != shape[-1]:
        if torch.prod(torch.tensor(shape[:-3])) != 1:
            raise ValueError("Batch core tensor with non-square matrix slices requested, pretty sure this isn't what you wanted")
        else:
            warnings.warn('Core tensor with non-square matrix slices requested, is this really what you wanted?')
    eye_core = batch_to(torch.eye(*shape[-2:]), shape[:-2], 2)
    noise = noise / torch.sqrt(torch.prod(torch.tensor(shape[-2:])).float())
    delta = noise * torch.randn(shape)
    if is_complex:
        delta = phaseify(delta)
    return eye_core + delta


def normal_init(shape: tuple, is_complex: bool=False, rel_std: float=1.0) ->Tensor:
    """
    Initialize an MPS core tensor with all slices normally distributed

    Args:
        shape: Shape of the core tensor being initialized.
        is_complex: Whether to initialize a complex core tensor.
            Default: False
        rel_std: Relative standard deviation of entries of matrix slices,
            scaled by a factor of the bond dimension of the MPS.
            Default: 1.0

    Returns:
        core_tensor: Normally distributed near-identity core tensor.
    """
    assert len(shape) >= 3
    if shape[-2] != shape[-1]:
        if torch.prod(torch.tensor(shape[:-3])) != 1:
            raise ValueError("Batch core tensor with non-square matrix slices requested, pretty sure this isn't what you wanted")
        else:
            warnings.warn('Core tensor with non-square matrix slices requested, is this really what you wanted?')
    std = 1 / shape[-1]
    core_tensor = std * torch.randn(shape)
    if is_complex:
        core_tensor = phaseify(core_tensor)
    return core_tensor


def slim_eval_fun(seq_input: Tensor, core_tensor: Tensor, bound_vecs: Tensor) ->Tensor:
    """
    Evaluate MPS tensor elements relative to a batch of sequential inputs.

    Args:
        seq_input: Tensor with shape `(batch, seq_len)` for discrete input
            sequences, or `(batch, seq_len, input_dim)` for vector input
            sequences.
        core_tensor: Tensor with shape `(input_dim, bond_dim, bond_dim)`
            for uniform MPS or `(seq_len, input_dim, bond_dim, bond_dim)`
            for fixed-length MPS.
        bound_vecs: Left and right boundary vectors expressed in matrix
            with shape `(2, bond_dim)`.

    Returns:
        contraction: Vector with shape `(batch,)` containing the elements
            of the MPS parameterized by `core_tensor` and `bound_vecs`,
            relative to the inputs in `seq_input`.
        log_scale: Vector with shape `(batch,)` containing (the logarithms of)
            positive-valued corrections to the scalar outputs in contraction,
            so that the actual values are `contraction * exp(log_scale)`.
    """
    seq_input = seq_input.transpose(0, 1)
    seq_len = len(seq_input)
    batch = seq_input.shape[1]
    bond_dim = core_tensor.shape[-1]
    assert bound_vecs.ndim == 2
    assert seq_input.ndim in (2, 3)
    assert core_tensor.ndim in (3, 4)
    assert bound_vecs.shape[0] == 2
    assert bound_vecs.shape[-1] == core_tensor.shape[-1]
    uniform = core_tensor.ndim == 3
    vec_input = seq_input.ndim == 3
    if not uniform:
        assert len(core_tensor) == seq_len
    if vec_input:
        assert seq_input.shape[-1] == core_tensor.shape[-3]
    if core_tensor.is_complex() and seq_input.is_floating_point():
        seq_input = seq_input
    if uniform:
        all_cores = repeat(core_tensor, seq_len)
    else:
        all_cores = core_tensor
    if vec_input:
        slice_fun = lambda inps, core: einsum('bi,ide->bde', inps, core)
    else:
        slice_fun = lambda inps, core: core[inps]
    log_scale = torch.zeros(batch)
    vecs = bound_vecs[0][None, None]
    for inps, core in zip(seq_input, all_cores):
        mats = slice_fun(inps, core)
        vecs = torch.matmul(vecs, mats)
        rescale = vecs.abs().sum(dim=-1, keepdim=True) / bond_dim
        log_scale = log_scale + rescale.log()[:, 0, 0]
        vecs = vecs / rescale
    contraction = torch.matmul(vecs.squeeze(dim=1), bound_vecs[1][:, None])
    assert contraction.shape == (batch, 1)
    return contraction.squeeze(dim=1), log_scale


class ProbMPS(nn.Module):
    """
    Fixed-length MPS model using L2 probabilities for generative modeling

    Probabilities of fixed-length inputs are obtained via the Born rule of
    quantum mechanics, making ProbMPS a "Born machine" model. For a model
    acting on length-n inputs, the probability assigned to the sequence
    :math:`x = x_1 x_2 \\dots x_n` is :math:`P(x) = |h_n^T \\omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A_t[x_t] + B) h_{t-1},

    with :math:`h_0 := \\alpha` (for :math:`\\alpha, \\omega` trainable
    parameter vectors), :math:`A_t[i]` the i'th matrix slice of a
    third-order core tensor for the t'th input, and :math:`B` an optional
    bias matrix.

    Note that calling a :class:`ProbMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbMPS`.

    Args:
        seq_len: Length of fixed-length discrete sequence inputs. Inputs
            can be either batches of discrete sequences, with a shape of
            `(input_len, batch)`, or batches of vector sequences, with a
            shape of `(input_len, batch, input_dim)`.
        input_dim: Dimension of the inputs to each core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking adjacent MPS cores,
            which are assumed to be equal everywhere.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
        init_method: String specifying how to initialize the MPS core tensors.
            Giving "near_eye" initializes all core slices to near the identity,
            while "normal" has all core elements be normally distributed.
            Default: ``"near_eye"``
        embed_fun: Function which embeds discrete or continous scalar values
            into vectors of dimension `input_dim`. Must be able to take in a
            tensor of any order `n` and output a tensor of order `n+1`, where
            the scalar values of the input are represented as vectors in the
            *last* axis of the output.
            Default: ``None`` (no embedding function)
        domain: Instance of the `DataDomain` class, which specifies whether
            the input data domain is continuous vs. discrete, and what range
            of values the domain takes.
            Default: ``None``
    """

    def __init__(self, seq_len: int, input_dim: int, bond_dim: int, complex_params: bool=False, use_bias: bool=False, init_method: str='near_eye', embed_fun: Optional[Callable]=None, domain: Optional[DataDomain]=None) ->None:
        super().__init__()
        assert min(seq_len, input_dim, bond_dim) > 0
        assert init_method in ('near_eye', 'normal')
        init_fun = near_eye_init if init_method == 'near_eye' else normal_init
        core_tensors = init_fun((seq_len, input_dim, bond_dim, bond_dim), is_complex=complex_params)
        rand_vec = torch.randn(bond_dim) / sqrt(bond_dim)
        edge_vecs = torch.stack((rand_vec,) * 2)
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)
        self.complex_params = complex_params
        self.embedding = None
        if isinstance(embed_fun, (FixedEmbedding, TrainableEmbedding)):
            self.embedding = embed_fun
            if hasattr(embed_fun, 'emb_dim'):
                assert self.embedding.emb_dim == input_dim
        elif embed_fun is not None:
            assert domain is not None
            self.embedding = FixedEmbedding(embed_fun, domain)
            assert self.embedding.emb_dim == input_dim

    def forward(self, input_data: Tensor, slim_eval: bool=False, parallel_eval: bool=False) ->Tensor:
        """
        Get the log probabilities of batch of input data

        Args:
            input_data: Sequential with shape `(batch, seq_len)`, for
                discrete inputs, or shape `(batch, seq_len, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            log_probs: Vector with shape `(batch,)` giving the natural
                logarithm of the probability of each input sequence.
        """
        if self.embedding is not None:
            input_data = self.embedding(input_data)
        if slim_eval:
            if self.use_bias:
                raise ValueError('Bias matrices not supported for slim_eval')
            psi_vals, log_scales = slim_eval_fun(input_data, self.core_tensors, self.edge_vecs)
        else:
            mat_slices = get_mat_slices(input_data, self.core_tensors)
            if self.use_bias:
                mat_slices = mat_slices + self.bias_mat[None, None]
            psi_vals, log_scales = contract_matseq(mat_slices, self.edge_vecs[0], self.edge_vecs[1], parallel_eval, log_format=True)
        log_norm = self.log_norm()
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())
        log_uprobs = torch.log(torch.abs(psi_vals)) + log_scales
        return 2 * log_uprobs - log_norm

    def loss(self, input_data: Tensor, slim_eval: bool=False, parallel_eval: bool=False) ->Tensor:
        """
        Get the negative log likelihood loss for batch of input data

        Args:
            input_data: Sequential with shape `(seq_len, batch)`, for
                discrete inputs, or shape `(seq_len, batch, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            loss_val: Scalar value giving average of the negative log
                likelihood loss of all sequences in input batch.
        """
        return -torch.mean(self.forward(input_data, slim_eval=slim_eval, parallel_eval=parallel_eval))

    def log_norm(self) ->Tensor:
        """
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\\log(|\\psi|^2)`,
        where :math:`\\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None, None]
        else:
            core_tensors = self.core_tensors
        lamb_mat = None if self.embedding is None else self.embedding.lamb_mat
        return get_log_norm(core_tensors, self.edge_vecs, lamb_mat=lamb_mat)

    @property
    def seq_len(self):
        return self.core_tensors.shape[0]

    @property
    def input_dim(self):
        return self.core_tensors.shape[1]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[2]

    @property
    def use_bias(self):
        return hasattr(self, 'bias_mat')


class ProbUnifMPS(ProbMPS):
    """
    Uniform MPS model using L2 probabilities for generative modeling

    Probabilities of sequential inputs are obtained via the Born rule of
    quantum mechanics, making ProbUnifMPS a "Born machine" model. Given an
    input sequence of length n, the probability assigned to the sequence
    :math:`x = x_1 x_2 \\dots x_n` is :math:`P(x) = |h_n^T \\omega|^2 / Z`,
    where :math:`Z` is a normalization constant and the hidden state
    vectors :math:`h_t` are updated according to:

    .. math::
        h_t = (A[x_t] + B) h_{t-1},

    with :math:`h_0 := \\alpha` (for :math:`\\alpha, \\omega` trainable
    parameter vectors), :math:`A[i]` the i'th matrix slice of the
    third-order MPS core tensor, and :math:`B` an optional bias matrix.

    Note that calling a :class:`ProbUnifMPS` instance with given input will
    return the **logarithm** of the input probabilities, to avoid underflow
    in the case of longer sequences. To get the negative log likelihood
    loss for a batch of inputs, use the :attr:`loss` function of the
    :class:`ProbUnifMPS`.

    Args:
        input_dim: Dimension of the inputs to the uMPS core. For vector
            sequence inputs this is the dimension of the input vectors,
            while for discrete sequence inputs this is the size of the
            discrete alphabet.
        bond_dim: Dimension of the bond spaces linking copies of uMPS core.
        complex_params: Whether model parameters are complex or real. The
            former allows more expressivity, but is less common in Pytorch.
            Default: ``False``
        use_bias: Whether to use a trainable bias matrix in evaluation.
            Default: ``False``
        init_method: String specifying how to initialize the MPS core tensors.
            Giving "near_eye" initializes all core slices to near the identity,
            while "normal" has all core elements be normally distributed.
            Default: ``"near_eye"``
        embed_fun: Function which embeds discrete or continous scalar values
            into vectors of dimension `input_dim`. Must be able to take in a
            tensor of any order `n` and output a tensor of order `n+1`, where
            the scalar values of the input are represented as vectors in the
            *last* axis of the output.
            Default: ``None`` (no embedding function)
        domain: Instance of the `DataDomain` class, which specifies whether
            the input data domain is continuous vs. discrete, and what range
            of values the domain takes.
            Default: ``None``
    """

    def __init__(self, input_dim: int, bond_dim: int, complex_params: bool=False, use_bias: bool=False, init_method: str='near_eye', embed_fun: Optional[Callable]=None, domain: Optional[DataDomain]=None) ->None:
        super(ProbMPS, self).__init__()
        assert min(input_dim, bond_dim) > 0
        assert init_method in ('near_eye', 'normal')
        init_fun = near_eye_init if init_method == 'near_eye' else normal_init
        core_tensors = init_fun((input_dim, bond_dim, bond_dim), is_complex=complex_params)
        rand_vec = torch.randn(bond_dim) / sqrt(bond_dim)
        edge_vecs = torch.stack((rand_vec, rand_vec.conj()))
        if complex_params:
            edge_vecs = phaseify(edge_vecs)
        self.core_tensors = nn.Parameter(core_tensors)
        self.edge_vecs = nn.Parameter(edge_vecs)
        if use_bias:
            bias_mat = torch.zeros(bond_dim, bond_dim)
            if complex_params:
                bias_mat = phaseify(bias_mat)
            self.bias_mat = nn.Parameter(bias_mat)
        self.complex_params = complex_params
        self.embedding = None
        if isinstance(embed_fun, (FixedEmbedding, TrainableEmbedding)):
            self.embedding = embed_fun
            if hasattr(embed_fun, 'emb_dim'):
                assert self.embedding.emb_dim == input_dim
        elif embed_fun is not None:
            assert domain is not None
            self.embedding = FixedEmbedding(embed_fun, domain)
            assert self.embedding.emb_dim == input_dim

    def forward(self, input_data: Tensor, slim_eval: bool=False, parallel_eval: bool=False) ->Tensor:
        """
        Get the log probabilities of batch of input data

        Args:
            input_data: Sequential with shape `(batch, seq_len)`, for
                discrete inputs, or shape `(batch, seq_len, input_dim)`,
                for vector inputs.
            slim_eval: Whether to use a less memory intensive MPS
                evaluation function, useful for larger inputs.
                Default: ``False``
            parallel_eval: Whether to use a more memory intensive parallel
                MPS evaluation function, useful for smaller models.
                Overrides `slim_eval` when both are requested.
                Default: ``False``

        Returns:
            log_probs: Vector with shape `(batch,)` giving the natural
                logarithm of the probability of each input sequence.
        """
        batch, seq_len = input_data.shape[:2]
        if self.embedding is not None:
            input_data = self.embedding(input_data)
        if slim_eval:
            if self.use_bias:
                raise ValueError('Bias matrices not supported for slim_eval')
            psi_vals, log_scales = slim_eval_fun(input_data, self.core_tensors, self.edge_vecs)
        else:
            mat_slices = get_mat_slices(input_data, self.core_tensors)
            if self.use_bias:
                mat_slices = mat_slices + self.bias_mat[None]
            psi_vals, log_scales = contract_matseq(mat_slices, self.edge_vecs[0], self.edge_vecs[1], parallel_eval, log_format=True)
        log_norm = self.log_norm(seq_len)
        assert log_norm.isfinite()
        assert torch.all(psi_vals.isfinite())
        log_uprobs = torch.log(torch.abs(psi_vals)) + log_scales
        assert log_uprobs.shape == (batch,)
        return 2 * log_uprobs - log_norm

    def log_norm(self, data_len) ->Tensor:
        """
        Compute the log normalization of the MPS for its fixed-size input

        Uses iterated tensor contraction to compute :math:`\\log(|\\psi|^2)`,
        where :math:`\\psi` is the n'th order tensor described by the
        contraction of MPS parameter cores. In the Born machine paradigm,
        this is also :math:`\\log(Z)`, for :math:`Z` the normalization
        constant for the probability.

        Returns:
            l_norm: Scalar value giving the log squared L2 norm of the
                n'th order prob. amp. tensor described by the MPS.
        """
        if self.use_bias:
            core_tensors = self.core_tensors + self.bias_mat[None]
        else:
            core_tensors = self.core_tensors
        lamb_mat = None if self.embedding is None else self.embedding.lamb_mat
        return get_log_norm(core_tensors, self.edge_vecs, lamb_mat=lamb_mat, length=data_len)

    @property
    def input_dim(self):
        return self.core_tensors.shape[0]

    @property
    def bond_dim(self):
        return self.core_tensors.shape[1]

    @property
    def use_bias(self):
        return hasattr(self, 'bias_mat')


class Contractable:
    """
    Container for tensors with labeled indices and a global batch size

    The labels for our indices give some high-level knowledge of the tensor
    layout, and permit the contraction of pairs of indices in a more
    systematic manner. However, much of the actual heavy lifting is done
    through specific contraction routines in different subclasses

    Attributes:
        tensor (Tensor):    A Pytorch tensor whose first index is a batch
                            index. Sub-classes of Contractable may put other
                            restrictions on tensor
        bond_str (str):     A string whose letters each label a separate mode
                            of our tensor, and whose length equals the order
                            (number of modes) of our tensor
        global_bs (int):    The batch size associated with all Contractables.
                            This is shared between all Contractable instances
                            and allows for automatic expanding of tensors
    """
    global_bs = None

    def __init__(self, tensor, bond_str):
        shape = list(tensor.shape)
        num_dim = len(shape)
        str_len = len(bond_str)
        global_bs = Contractable.global_bs
        batch_dim = tensor.size(0)
        if 'b' not in bond_str and str_len == num_dim or 'b' == bond_str[0] and str_len == num_dim + 1:
            if global_bs is not None:
                tensor = tensor.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError('No batch size given and no previous batch size set')
            if bond_str[0] != 'b':
                bond_str = 'b' + bond_str
        elif bond_str[0] != 'b' or str_len != num_dim:
            raise ValueError(f"Length of bond string '{{bond_str}}' ({len(bond_str)}) must match order of tensor ({len(shape)})")
        elif global_bs is None or global_bs != batch_dim:
            Contractable.global_bs = batch_dim
        elif global_bs != batch_dim:
            raise RuntimeError(f'Batch size previously set to {global_bs}, but input tensor has batch size {batch_dim}')
        self.tensor = tensor
        self.bond_str = bond_str

    def __mul__(self, contractable, rmul=False):
        """
        Multiply with another contractable along a linear index

        The default behavior is to multiply the 'r' index of this instance
        with the 'l' index of contractable, matching the batch ('b')
        index of both, and take the outer product of other indices.
        If rmul is True, contractable is instead multiplied on the right.
        """
        if isinstance(contractable, Scalar) or not hasattr(contractable, 'tensor') or type(contractable) is MatRegion:
            return NotImplemented
        tensors = [self.tensor, contractable.tensor]
        bond_strs = [list(self.bond_str), list(contractable.bond_str)]
        lowercases = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        if rmul:
            tensors = tensors[::-1]
            bond_strs = bond_strs[::-1]
        for i, bs in enumerate(bond_strs):
            assert bs[0] == 'b'
            assert len(set(bs)) == len(bs)
            assert all([(c in lowercases) for c in bs])
            assert i == 0 and 'r' in bs or i == 1 and 'l' in bs
        used_chars = set(bond_strs[0]).union(bond_strs[1])
        free_chars = [c for c in lowercases if c not in used_chars]
        specials = ['b', 'l', 'r']
        for i, c in enumerate(bond_strs[1]):
            if c in bond_strs[0] and c not in specials:
                bond_strs[1][i] = free_chars.pop()
        sum_char = free_chars.pop()
        bond_strs[0][bond_strs[0].index('r')] = sum_char
        bond_strs[1][bond_strs[1].index('l')] = sum_char
        specials.append(sum_char)
        out_str = ['b']
        for bs in bond_strs:
            out_str.extend([c for c in bs if c not in specials])
        out_str.append('l' if 'l' in bond_strs[0] else '')
        out_str.append('r' if 'r' in bond_strs[1] else '')
        bond_strs = [''.join(bs) for bs in bond_strs]
        out_str = ''.join(out_str)
        ein_str = f'{bond_strs[0]},{bond_strs[1]}->{out_str}'
        out_tensor = torch.einsum(ein_str, [tensors[0], tensors[1]])
        if out_str == 'br':
            return EdgeVec(out_tensor, is_left_vec=True)
        elif out_str == 'bl':
            return EdgeVec(out_tensor, is_left_vec=False)
        elif out_str == 'blr':
            return SingleMat(out_tensor)
        elif out_str == 'bolr':
            return OutputCore(out_tensor)
        else:
            return Contractable(out_tensor, out_str)

    def __rmul__(self, contractable):
        """
        Multiply with another contractable along a linear index
        """
        return self.__mul__(contractable, rmul=True)

    def reduce(self):
        """
        Return the contractable without any modification

        reduce() can be any method which returns a contractable. This is
        trivially possible for any contractable by returning itself
        """
        return self


class ContractableList(Contractable):
    """
    A list of contractables which can all be multiplied together in order

    Calling reduce on a ContractableList instance will first reduce every item
    to a linear contractable, and then contract everything together
    """

    def __init__(self, contractable_list):
        if not isinstance(contractable_list, list) or contractable_list is []:
            raise ValueError('Input to ContractableList must be nonempty list')
        for i, item in enumerate(contractable_list):
            if not isinstance(item, Contractable):
                raise ValueError(f'Input items to ContractableList must be Contractable instances, but item {i} is not')
        self.contractable_list = contractable_list

    def __mul__(self, contractable, rmul=False):
        """
        Multiply a contractable by everything in ContractableList in order
        """
        assert hasattr(contractable, 'tensor')
        output = contractable.tensor
        if rmul:
            for item in self.contractable_list:
                output = item * output
        else:
            for item in self.contractable_list[::-1]:
                output = output * item
        return output

    def __rmul__(self, contractable):
        """
        Multiply another contractable by everything in ContractableList
        """
        return self.__mul__(contractable, rmul=True)

    def reduce(self, parallel_eval=False):
        """
        Reduce all the contractables in list before multiplying them together
        """
        c_list = self.contractable_list
        if parallel_eval:
            c_list = [item.reduce() for item in c_list]
        while len(c_list) > 1:
            try:
                c_list[-2] = c_list[-2] * c_list[-1]
                del c_list[-1]
            except TypeError:
                c_list[1] = c_list[0] * c_list[1]
                del c_list[0]
        return c_list[0]


class InitialVector(nn.Module):
    """
    Vector of ones and zeros to act as initial vector within the MPS

    By default the initial vector is chosen to be all ones, but if fill_dim is
    specified then only the first fill_dim entries are set to one, with the
    rest zero.

    If fixed_vec is False, then the initial vector will be registered as a
    trainable model parameter.
    """

    def __init__(self, bond_dim, fill_dim=None, fixed_vec=True, is_left_vec=True):
        super().__init__()
        vec = torch.ones(bond_dim)
        if fill_dim is not None:
            assert fill_dim >= 0 and fill_dim <= bond_dim
            vec[fill_dim:] = 0
        if fixed_vec:
            vec.requires_grad = False
            self.register_buffer(name='vec', tensor=vec)
        else:
            vec.requires_grad = True
            self.register_parameter(name='vec', param=nn.Parameter(vec))
        assert isinstance(is_left_vec, bool)
        self.is_left_vec = is_left_vec

    def forward(self):
        """
        Return our initial vector wrapped as an EdgeVec contractable
        """
        return EdgeVec(self.vec, self.is_left_vec)

    def core_len(self):
        return 1

    def __len__(self):
        return 0


class InputSite(nn.Module):
    """
    A single MPS core which takes in a single input datum, bond_str = 'lri'
    """

    def __init__(self, tensor):
        super().__init__()
        self.register_parameter(name='tensor', param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Contract input with MPS core and return result as a SingleMat

        Args:
            input_data (Tensor): Input with shape [batch_size, feature_dim]
        """
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(2)
        mat = torch.einsum('lri,bi->blr', [tensor, input_data])
        return SingleMat(mat)

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]
        self.tensor *= scale

    def core_len(self):
        return 1

    def __len__(self):
        return 1


def svd_flex(tensor, svd_string, max_D=None, cutoff=1e-10, sv_right=True, sv_vec=None):
    """
    Split an input tensor into two pieces using a SVD across some partition

    Args:
        tensor (Tensor):    Pytorch tensor with at least two indices

        svd_string (str):   String of the form 'init_str->left_str,right_str',
                            where init_str describes the indices of tensor, and
                            left_str/right_str describe those of the left and
                            right output tensors. The characters of left_str
                            and right_str form a partition of the characters in
                            init_str, but each contain one additional character
                            representing the new bond which comes from the SVD

                            Reversing the terms in svd_string to the left and
                            right of '->' gives an ein_string which can be used
                            to multiply both output tensors to give a (low rank
                            approximation) of the input tensor

        cutoff (float):     A truncation threshold which eliminates any
                            singular values which are strictly less than cutoff

        max_D (int):        A maximum allowed value for the new bond. If max_D
                            is specified, the returned tensors

        sv_right (bool):    The SVD gives two orthogonal matrices and a matrix
                            of singular values. sv_right=True merges the SV
                            matrix with the right output, while sv_right=False
                            merges it with the left output

        sv_vec (Tensor):    Pytorch vector with length max_D, which is modified
                            in place to return the vector of singular values

    Returns:
        left_tensor (Tensor),
        right_tensor (Tensor):  Tensors whose indices are described by the
                                left_str and right_str parts of svd_string

        bond_dim:               The dimension of the new bond appearing from
                                the cutoff in our SVD. Note that this generally
                                won't match the dimension of left_/right_tensor
                                at this mode, which is padded with zeros
                                whenever max_D is specified
    """

    def prod(int_list):
        output = 1
        for num in int_list:
            output *= num
        return output
    with torch.no_grad():
        svd_string = svd_string.replace(' ', '')
        init_str, post_str = svd_string.split('->')
        left_str, right_str = post_str.split(',')
        assert all([c.islower() for c in init_str + left_str + right_str])
        assert len(set(init_str + left_str + right_str)) == len(init_str) + 1
        assert len(set(init_str)) + len(set(left_str)) + len(set(right_str)) == len(init_str) + len(left_str) + len(right_str)
        bond_char = set(left_str).intersection(set(right_str)).pop()
        left_part = left_str.replace(bond_char, '')
        right_part = right_str.replace(bond_char, '')
        ein_str = f'{init_str}->{left_part + right_part}'
        tensor = torch.einsum(ein_str, [tensor]).contiguous()
        left_shape = list(tensor.shape[:len(left_part)])
        right_shape = list(tensor.shape[len(left_part):])
        left_dim, right_dim = prod(left_shape), prod(right_shape)
        tensor = tensor.view([left_dim, right_dim])
        left_mat, svs, right_mat = torch.svd(tensor)
        svs, _ = torch.sort(svs, descending=True)
        right_mat = torch.t(right_mat)
        if max_D and len(svs) > max_D:
            svs = svs[:max_D]
            left_mat = left_mat[:, :max_D]
            right_mat = right_mat[:max_D]
        elif max_D and len(svs) < max_D:
            copy_svs = torch.zeros([max_D])
            copy_svs[:len(svs)] = svs
            copy_left = torch.zeros([left_mat.size(0), max_D])
            copy_left[:, :left_mat.size(1)] = left_mat
            copy_right = torch.zeros([max_D, right_mat.size(1)])
            copy_right[:right_mat.size(0)] = right_mat
            svs, left_mat, right_mat = copy_svs, copy_left, copy_right
        if sv_vec is not None and svs.shape == sv_vec.shape:
            sv_vec[:] = svs
        elif sv_vec is not None and svs.shape != sv_vec.shape:
            raise TypeError(f'sv_vec.shape must be {list(svs.shape)}, but is currently {list(sv_vec.shape)}')
        truncation = 0
        for s in svs:
            if s < cutoff:
                break
            truncation += 1
        if truncation == 0:
            raise RuntimeError('SVD cutoff too large, attempted to truncate tensor to bond dimension 0')
        if max_D:
            svs[truncation:] = 0
            left_mat[:, truncation:] = 0
            right_mat[truncation:] = 0
        else:
            max_D = truncation
            svs = svs[:truncation]
            left_mat = left_mat[:, :truncation]
            right_mat = right_mat[:truncation]
        if sv_right:
            right_mat = torch.einsum('l,lr->lr', [svs, right_mat])
        else:
            left_mat = torch.einsum('lr,r->lr', [left_mat, svs])
        left_tensor = left_mat.view(left_shape + [max_D])
        right_tensor = right_mat.view([max_D] + right_shape)
        if left_str != left_part + bond_char:
            left_tensor = torch.einsum(f'{left_part + bond_char}->{left_str}', [left_tensor])
        if right_str != bond_char + right_part:
            right_tensor = torch.einsum(f'{bond_char + right_part}->{right_str}', [right_tensor])
        return left_tensor, right_tensor, truncation


class MergedInput(nn.Module):
    """
    Contiguous region of merged MPS cores, each taking in a pair of input data

    Since MergedInput arises after contracting together existing input cores,
    a merged input tensor is required for initialization
    """

    def __init__(self, tensor):
        shape = tensor.shape
        assert len(shape) == 5
        assert shape[1] == shape[2]
        assert shape[3] == shape[4]
        super().__init__()
        self.register_parameter(name='tensor', param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Contract input with merged MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                 feature_dim], where input_dim must be even
                                 (each merged core takes 2 inputs)
        """
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)
        assert input_data.size(1) % 2 == 0
        inputs = [input_data[:, 0::2], input_data[:, 1::2]]
        tensor = torch.einsum('slrij,bsj->bslri', [tensor, inputs[1]])
        mats = torch.einsum('bslri,bsi->bslr', [tensor, inputs[0]])
        return MatRegion(mats)

    def _unmerge(self, cutoff=1e-10):
        """
        Separate the cores in our MergedInput and return an InputRegion

        The length of the resultant InputRegion will be identical to our
        original MergedInput (same number of inputs), but its core_len will
        be doubled (twice as many individual cores)
        """
        tensor = self.tensor
        svd_string = 'lrij->lui,urj'
        max_D = tensor.size(1)
        core_list, bond_list, sv_list = [], [None], [None]
        for merged_core in tensor:
            sv_vec = torch.empty(max_D)
            left_core, right_core, bond_dim = svd_flex(merged_core, svd_string, max_D, cutoff, sv_vec=sv_vec)
            core_list += [left_core, right_core]
            bond_list += [bond_dim, None]
            sv_list += [sv_vec, None]
        tensor = torch.stack(core_list)
        return [InputRegion(tensor)], bond_list, sv_list

    def get_norm(self):
        """
        Returns list of the norm of each core in MergedInput
        """
        return [torch.norm(core) for core in self.tensor]

    @torch.no_grad()
    def rescale_norm(self, scale_list):
        """
        Rescales the norm of each core by an amount specified in scale_list

        For the i'th tensor defining a core in MergedInput, we rescale as
        tensor_i <- scale_i * tensor_i, where scale_i = scale_list[i]
        """
        assert len(scale_list) == len(self.tensor)
        for core, scale in zip(self.tensor, scale_list):
            core *= scale

    def core_len(self):
        return len(self)

    def __len__(self):
        """
        Returns the number of input sites, which is twice the number of cores
        """
        return 2 * self.tensor.size(0)


class InputRegion(nn.Module):
    """
    Contiguous region of MPS input cores, associated with bond_str = 'slri'
    """

    def __init__(self, tensor, use_bias=True, fixed_bias=True, bias_mat=None, ephemeral=False):
        super().__init__()
        assert len(tensor.shape) == 4
        assert tensor.size(1) == tensor.size(2)
        bond_dim = tensor.size(1)
        if use_bias:
            assert bias_mat is None or isinstance(bias_mat, torch.Tensor)
            bias_mat = torch.eye(bond_dim).unsqueeze(0) if bias_mat is None else bias_mat
            bias_modes = len(list(bias_mat.shape))
            assert bias_modes in [2, 3]
            if bias_modes == 2:
                bias_mat = bias_mat.unsqueeze(0)
        if ephemeral:
            self.register_buffer(name='tensor', tensor=tensor.contiguous())
            self.register_buffer(name='bias_mat', tensor=bias_mat)
        else:
            self.register_parameter(name='tensor', param=nn.Parameter(tensor.contiguous()))
            if fixed_bias:
                self.register_buffer(name='bias_mat', tensor=bias_mat)
            else:
                self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))
        self.use_bias = use_bias
        self.fixed_bias = fixed_bias

    def forward(self, input_data):
        """
        Contract input with MPS cores and return result as a MatRegion

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        tensor = self.tensor
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)
        mats = torch.einsum('slri,bsi->bslr', [tensor, input_data])
        if self.use_bias:
            bias_mat = self.bias_mat.unsqueeze(0)
            mats = mats + bias_mat.expand_as(mats)
        return MatRegion(mats)

    def _merge(self, offset):
        """
        Merge all pairs of neighboring cores and return a new list of cores

        offset is either 0 or 1, which gives the first core at which we start
        our merging. Depending on the length of our InputRegion, the output of
        merge may have 1, 2, or 3 entries, with the majority of sites ending in
        a MergedInput instance
        """
        assert offset in [0, 1]
        num_sites = self.core_len()
        parity = num_sites % 2
        if num_sites == 0:
            return [None]
        if (offset, parity) == (1, 1):
            out_list = [self[0], self[1:]._merge(offset=0)[0]]
        elif (offset, parity) == (1, 0):
            out_list = [self[0], self[1:-1]._merge(offset=0)[0], self[-1]]
        elif (offset, parity) == (0, 1):
            out_list = [self[:-1]._merge(offset=0)[0], self[-1]]
        else:
            tensor = self.tensor
            even_cores, odd_cores = tensor[0::2], tensor[1::2]
            assert len(even_cores) == len(odd_cores)
            merged_cores = torch.einsum('slui,surj->slrij', [even_cores, odd_cores])
            out_list = [MergedInput(merged_cores)]
        return [x for x in out_list if x is not None]

    def __getitem__(self, key):
        """
        Returns an InputRegion instance sliced along the site index
        """
        assert isinstance(key, int) or isinstance(key, slice)
        if isinstance(key, slice):
            return InputRegion(self.tensor[key])
        else:
            return InputSite(self.tensor[key])

    def get_norm(self):
        """
        Returns list of the norms of each core in InputRegion
        """
        return [torch.norm(core) for core in self.tensor]

    @torch.no_grad()
    def rescale_norm(self, scale_list):
        """
        Rescales the norm of each core by an amount specified in scale_list

        For the i'th tensor defining a core in InputRegion, we rescale as
        tensor_i <- scale_i * tensor_i, where scale_i = scale_list[i]
        """
        assert len(scale_list) == len(self.tensor)
        for core, scale in zip(self.tensor, scale_list):
            core *= scale

    def core_len(self):
        return len(self)

    def __len__(self):
        return self.tensor.size(0)


class OutputMat(Contractable):
    """
    An output core associated with an edge of our MPS
    """

    def __init__(self, mat, is_left_mat):
        if len(mat.shape) not in [2, 3]:
            raise ValueError('OutputMat tensors must have shape [batch_size, D, output_dim], or else [D, output_dim] if batch size has already been set')
        bond_str = 'b' + ('r' if is_left_mat else 'l') + 'o'
        super().__init__(mat, bond_str=bond_str)

    def __mul__(self, edge_vec, rmul=False):
        """
        Multiply with an edge vector along the shared linear index
        """
        if not isinstance(edge_vec, EdgeVec):
            raise NotImplemented
        else:
            return super().__mul__(edge_vec, rmul)

    def __rmul__(self, edge_vec):
        return self.__mul__(edge_vec, rmul=True)


class TerminalOutput(nn.Module):
    """
    Output matrix at end of chain to transmute virtual state into output vector

    By default, a fixed rectangular identity matrix with shape
    [bond_dim, output_dim] will be used as a state transducer. If fixed_mat is
    False, then the matrix will be registered as a trainable model parameter.
    """

    def __init__(self, bond_dim, output_dim, fixed_mat=False, is_left_mat=False):
        super().__init__()
        if fixed_mat and output_dim > bond_dim:
            raise ValueError(f'With fixed_mat=True, TerminalOutput currently only supports initialization for bond_dim >= output_dim, but here bond_dim={bond_dim} and output_dim={output_dim}')
        mat = torch.eye(bond_dim, output_dim)
        if fixed_mat:
            mat.requires_grad = False
            self.register_buffer(name='mat', tensor=mat)
        else:
            mat = mat + torch.randn_like(mat) / bond_dim
            mat.requires_grad = True
            self.register_parameter(name='mat', param=nn.Parameter(mat))
        assert isinstance(is_left_mat, bool)
        self.is_left_mat = is_left_mat

    def forward(self):
        """
        Return our terminal matrix wrapped as an OutputMat contractable
        """
        return OutputMat(self.mat, self.is_left_mat)

    def core_len(self):
        return 1

    def __len__(self):
        return 0


def init_tensor(shape, bond_str, init_method):
    """
    Initialize a tensor with a given shape

    Args:
        shape:       The shape of our output parameter tensor.

        bond_str:    The bond string describing our output parameter tensor,
                     which is used in 'random_eye' initialization method.
                     The characters 'l' and 'r' are used to refer to the
                     left or right virtual indices of our tensor, and are
                     both required to be present for the random_eye and
                     min_random_eye initialization methods.

        init_method: The method used to initialize the entries of our tensor.
                     This can be either a string, or else a tuple whose first
                     entry is an initialization method and whose remaining
                     entries are specific to that method. In each case, std
                     will always refer to a standard deviation for a random
                     normal random component of each entry of the tensor.

                     Allowed options are:
                        * ('random_eye', std): Initialize each tensor input
                            slice close to the identity
                        * ('random_zero', std): Initialize each tensor input
                            slice close to the zero matrix
                        * ('min_random_eye', std, init_dim): Initialize each
                            tensor input slice close to a truncated identity
                            matrix, whose truncation leaves init_dim unit
                            entries on the diagonal. If init_dim is larger
                            than either of the bond dimensions, then init_dim
                            is capped at the smaller bond dimension.
    """
    if not isinstance(init_method, str):
        init_str = init_method[0]
        std = init_method[1]
        if init_str == 'min_random_eye':
            init_dim = init_method[2]
        init_method = init_str
    else:
        std = 1e-09
    assert len(shape) == len(bond_str)
    assert len(set(bond_str)) == len(bond_str)
    if init_method not in ['random_eye', 'min_random_eye', 'random_zero']:
        raise ValueError(f'Unknown initialization method: {init_method}')
    if init_method in ['random_eye', 'min_random_eye']:
        bond_chars = ['l', 'r']
        assert all([(c in bond_str) for c in bond_chars])
        if init_method == 'min_random_eye':
            bond_dims = [shape[bond_str.index(c)] for c in bond_chars]
            if all([(init_dim <= full_dim) for full_dim in bond_dims]):
                bond_dims = [init_dim, init_dim]
            else:
                init_dim = min(bond_dims)
            eye_shape = [(init_dim if c in bond_chars else 1) for c in bond_str]
            expand_shape = [(init_dim if c in bond_chars else shape[i]) for i, c in enumerate(bond_str)]
        elif init_method == 'random_eye':
            eye_shape = [(shape[i] if c in bond_chars else 1) for i, c in enumerate(bond_str)]
            expand_shape = shape
            bond_dims = [shape[bond_str.index(c)] for c in bond_chars]
        eye_tensor = torch.eye(bond_dims[0], bond_dims[1]).view(eye_shape)
        eye_tensor = eye_tensor.expand(expand_shape)
        tensor = torch.zeros(shape)
        tensor[[slice(dim) for dim in expand_shape]] = eye_tensor
        tensor += std * torch.randn(shape)
    elif init_method == 'random_zero':
        tensor = std * torch.randn(shape)
    return tensor


class TI_MPS(nn.Module):
    """
    Sequence MPS which converts input of arbitrary length to a single output vector
    """

    def __init__(self, output_dim, bond_dim, feature_dim=2, parallel_eval=False, fixed_ends=False, init_std=1e-09, use_bias=True, fixed_bias=True):
        super().__init__()
        tensor = init_tensor(bond_str='lri', shape=[bond_dim, bond_dim, feature_dim], init_method=('random_zero', init_std))
        self.register_parameter(name='core_tensor', param=nn.Parameter(tensor))
        assert isinstance(fixed_ends, bool)
        self.init_vector = InitialVector(bond_dim, fixed_vec=fixed_ends)
        self.terminal_mat = TerminalOutput(bond_dim, output_dim, fixed_mat=fixed_ends)
        if use_bias:
            if fixed_bias:
                bias_mat = torch.eye(bond_dim)
                self.register_buffer(name='bias_mat', tensor=bias_mat)
            else:
                bias_mat = init_tensor(bond_str='lr', shape=[bond_dim, bond_dim], init_method=('random_eye', init_std))
                self.register_parameter(name='bias_mat', param=nn.Parameter(bias_mat))
        else:
            self.bias_mat = None
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.parallel_eval = parallel_eval
        self.use_bias = use_bias
        self.fixed_bias = fixed_bias
        self.feature_map = None

    def forward(self, input_data):
        """
        Converts batch input tensor into a batch output tensor

        Args:
            input_data: A tensor of shape [batch_size, length, feature_dim].
        """
        batch_input = self.format_input(input_data)
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(1)
        expanded_core = self.core_tensor.expand([seq_len, self.bond_dim, self.bond_dim, self.feature_dim])
        input_region = InputRegion(expanded_core, use_bias=self.use_bias, fixed_bias=self.fixed_bias, bias_mat=self.bias_mat, ephemeral=True)
        contractable_list = [input_region(batch_input)]
        contractable_list = [self.init_vector()] + contractable_list
        contractable_list.append(self.terminal_mat())
        contractable_list = ContractableList(contractable_list)
        output = contractable_list.reduce(parallel_eval=self.parallel_eval)
        batch_output = output.tensor
        assert output.bond_str == 'bo'
        assert batch_output.size(0) == batch_size
        assert batch_output.size(1) == self.output_dim
        return batch_output

    def format_input(self, input_data):
        """
        Converts input list of sequences into a single batch sequence tensor.

        If input is already a batch tensor, it is returned unchanged. Otherwise,
        convert input list into a batch sequence with shape [batch_size, length,
        feature_dim].

        If self.use_bias = self.fixed_bias = True, then sequences of different
        lengths can be used, in which case shorter sequences are padded with
        zeros at the end, making the batch tensor length equal to the length
        of the longest input sequence.

        Args:
            input_data: A tensor of shape [batch_size, length] or
            [batch_size, length, feature_dim], or a list of length batch_size,
            whose i'th item is a tensor of shape [length_i, feature_dim] or
            [length_i]. If self.use_bias or self.fixed_bias are False, then
            length_i must be the same for all i.
        """
        feature_dim = self.feature_dim
        if isinstance(input_data, torch.Tensor):
            if len(input_data.shape) == 2:
                input_data = self.embed_input(input_data)
            shape = input_data.shape
            assert len(shape) == 3
            assert shape[2] == feature_dim
            return input_data
        elif isinstance(input_data, list):
            num_modes = len(input_data[0].shape)
            assert num_modes in [1, 2]
            assert all([(isinstance(s, torch.Tensor) and len(s.shape) == num_modes) for s in input_data])
            assert num_modes == 1 or all([(s.size(1) == feature_dim) for s in input_data])
            max_len = max([s.size(0) for s in input_data])
            can_pad = self.use_bias and self.fixed_bias
            if not can_pad and any([(s.size(0) != max_len) for s in input_data]):
                raise ValueError(f'To process input_data as list of sequences with different lengths, must have self.use_bias=self.fixed_bias=True (currently self.use_bias={self.use_bias}, self.fixed_bias={self.fixed_bias})')
            if can_pad:
                batch_size = len(input_data)
                full_size = [batch_size, max_len, feature_dim]
                batch_input = torch.zeros(full_size[:num_modes + 1])
                for i, seq in enumerate(input_data):
                    batch_input[i, :seq.size(0)] = seq
            else:
                batch_input = torch.stack(input_data)
            if len(batch_input.shape) == 2:
                batch_input = self.embed_input(batch_input)
            return batch_input
        else:
            raise ValueError('input_data must either be Tensor with shape[batch_size, length] or [batch_size, length, feature_dim], or list of Tensors with shapes [length_i, feature_dim] or [length_i]')

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate local feature spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, length].

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_dim, feature_dim]
        """
        assert len(input_data.shape) == 2
        batch_dim, length = input_data.shape
        feature_dim = self.feature_dim
        embedded_shape = [batch_dim, length, feature_dim]
        if self.feature_map is not None:
            f_map = self.feature_map
            embedded_data = torch.stack([torch.stack([f_map(x) for x in batch]) for batch in input_data])
            assert list(embedded_data.shape) == embedded_shape
        else:
            if self.feature_dim != 2:
                raise RuntimeError(f'self.feature_dim = {feature_dim}, but default feature_map requires self.feature_dim = 2')
            embedded_data = torch.empty(embedded_shape)
            embedded_data[:, :, 0] = input_data
            embedded_data[:, :, 1] = 1 - input_data
        return embedded_data

    def register_feature_map(self, feature_map):
        """
        Register a custom feature map to be used for embedding input data

        Args:
            feature_map (function): Takes a single scalar input datum and
                                    returns an embedded representation of the
                                    image. The output size of the function must
                                    match self.feature_dim. If feature_map=None,
                                    then the feature map will be reset to a
                                    simple default linear embedding
        """
        if feature_map is not None:
            test_out = feature_map(torch.tensor(0))
            assert isinstance(test_out, torch.Tensor)
            out_shape, needed_shape = list(test_out.shape), [self.feature_dim]
            if out_shape != needed_shape:
                raise ValueError(f'Given feature_map returns values with shape {list(out_shape)}, but should return values of size {list(needed_shape)}')
        self.feature_map = feature_map


class LinearRegion(nn.Module):
    """
    List of modules which feeds input to each module and returns reduced output
    """

    def __init__(self, module_list, periodic_bc=False, parallel_eval=False, module_states=None):
        if not isinstance(module_list, list) or module_list is []:
            raise ValueError('Input to LinearRegion must be nonempty list')
        for i, item in enumerate(module_list):
            if not isinstance(item, nn.Module):
                raise ValueError(f'Input items to LinearRegion must be PyTorch Module instances, but item {i} is not')
        super().__init__()
        self.module_list = nn.ModuleList(module_list)
        self.periodic_bc = periodic_bc
        self.parallel_eval = parallel_eval

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        periodic_bc = self.periodic_bc
        parallel_eval = self.parallel_eval
        lin_bonds = ['l', 'r']
        to_cuda = input_data.is_cuda
        device = f'cuda:{input_data.get_device()}' if to_cuda else 'cpu'
        ind = 0
        contractable_list = []
        for module in self.module_list:
            mod_len = len(module)
            if mod_len == 1:
                mod_input = input_data[:, ind]
            else:
                mod_input = input_data[:, ind:ind + mod_len]
            ind += mod_len
            contractable_list.append(module(mod_input))
        if periodic_bc:
            contractable_list = ContractableList(contractable_list)
            contractable = contractable_list.reduce(parallel_eval=True)
            tensor, bond_str = contractable.tensor, contractable.bond_str
            assert all(c in bond_str for c in lin_bonds)
            in_str, out_str = '', ''
            for c in bond_str:
                if c in lin_bonds:
                    in_str += 'l'
                else:
                    in_str += c
                    out_str += c
            ein_str = in_str + '->' + out_str
            return torch.einsum(ein_str, [tensor])
        else:
            end_items = [contractable_list[i] for i in [0, -1]]
            bond_strs = [item.bond_str for item in end_items]
            bond_inds = [bs.index(c) for bs, c in zip(bond_strs, lin_bonds)]
            bond_dims = [item.tensor.size(ind) for item, ind in zip(end_items, bond_inds)]
            end_vecs = [torch.zeros(dim) for dim in bond_dims]
            for vec in end_vecs:
                vec[0] = 1
            contractable_list.insert(0, EdgeVec(end_vecs[0], is_left_vec=True))
            contractable_list.append(EdgeVec(end_vecs[1], is_left_vec=False))
            contractable_list = ContractableList(contractable_list)
            output = contractable_list.reduce(parallel_eval=parallel_eval)
            return output.tensor

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])


class OutputSite(nn.Module):
    """
    A single MPS core with no input and a single output index, bond_str = 'olr'
    """

    def __init__(self, tensor):
        super().__init__()
        self.register_parameter(name='tensor', param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        """
        Return the OutputSite wrapped as an OutputCore contractable
        """
        return OutputCore(self.tensor)

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]
        self.tensor *= scale

    def core_len(self):
        return 1

    def __len__(self):
        return 0


class MergedOutput(nn.Module):
    """
    Merged MPS core taking in one input datum and returning an output vector

    Since MergedOutput arises after contracting together an existing input and
    output core, an already-merged tensor is required for initialization

    Args:
        tensor (Tensor):    Value that our merged core is initialized to
        left_output (bool): Specifies if the output core is on the left side of
                            the input core (True), or on the right (False)
    """

    def __init__(self, tensor, left_output):
        assert len(tensor.shape) == 4
        super().__init__()
        self.register_parameter(name='tensor', param=nn.Parameter(tensor.contiguous()))
        self.left_output = left_output

    def forward(self, input_data):
        """
        Contract input with input index of core and return an OutputCore

        Args:
            input_data (Tensor): Input with shape [batch_size, feature_dim]
        """
        tensor = self.tensor
        assert len(input_data.shape) == 2
        assert input_data.size(1) == tensor.size(3)
        tensor = torch.einsum('olri,bi->bolr', [tensor, input_data])
        return OutputCore(tensor)

    def _unmerge(self, cutoff=1e-10):
        """
        Split our MergedOutput into an OutputSite and an InputSite

        The non-zero entries of our tensors are dynamically sized according to
        the SVD cutoff, but will generally be padded with zeros to give the
        new index a regular size.
        """
        tensor = self.tensor
        left_output = self.left_output
        if left_output:
            svd_string = 'olri->olu,uri'
            max_D = tensor.size(2)
            sv_vec = torch.empty(max_D)
            output_core, input_core, bond_dim = svd_flex(tensor, svd_string, max_D, cutoff, sv_vec=sv_vec)
            return [OutputSite(output_core), InputSite(input_core)], [None, bond_dim, None], [None, sv_vec, None]
        else:
            svd_string = 'olri->our,lui'
            max_D = tensor.size(1)
            sv_vec = torch.empty(max_D)
            output_core, input_core, bond_dim = svd_flex(tensor, svd_string, max_D, cutoff, sv_vec=sv_vec)
            return [InputSite(input_core), OutputSite(output_core)], [None, bond_dim, None], [None, sv_vec, None]

    def get_norm(self):
        """
        Returns the norm of our core tensor, wrapped as a singleton list
        """
        return [torch.norm(self.tensor)]

    @torch.no_grad()
    def rescale_norm(self, scale):
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]
        self.tensor *= scale

    def core_len(self):
        return 2

    def __len__(self):
        return 1


class MergedLinearRegion(LinearRegion):
    """
    Dynamic variant of LinearRegion that periodically rearranges its submodules
    """

    def __init__(self, module_list, periodic_bc=False, parallel_eval=False, cutoff=1e-10, merge_threshold=2000):
        super().__init__(module_list, periodic_bc, parallel_eval)
        self.offset = 0
        self._merge(offset=self.offset)
        self._merge(offset=(self.offset + 1) % 2)
        self.module_list = getattr(self, f'module_list_{self.offset}')
        self.input_counter = 0
        self.merge_threshold = merge_threshold
        self.cutoff = cutoff

    def forward(self, input_data):
        """
        Contract input with list of MPS cores and return result as contractable

        MergedLinearRegion keeps an input counter of the number of inputs, and
        when this exceeds its merge threshold, triggers an unmerging and
        remerging of its parameter tensors.

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim,
                                                   feature_dim]
        """
        if self.input_counter >= self.merge_threshold:
            bond_list, sv_list = self._unmerge(cutoff=self.cutoff)
            self.offset = (self.offset + 1) % 2
            self._merge(offset=self.offset)
            self.input_counter -= self.merge_threshold
            self.module_list = getattr(self, f'module_list_{self.offset}')
        else:
            bond_list, sv_list = None, None
        self.input_counter += input_data.size(0)
        output = super().forward(input_data)
        if bond_list:
            return output, bond_list, sv_list
        else:
            return output

    @torch.no_grad()
    def _merge(self, offset):
        """
        Convert unmerged modules in self.module_list to merged counterparts

        Calling _merge (or _unmerge) directly can cause undefined behavior,
        but see MergedLinearRegion.forward for intended use

        This proceeds by first merging all unmerged cores internally, then
        merging lone cores when possible during a second sweep
        """
        assert offset in [0, 1]
        unmerged_list = self.module_list
        site_num = offset
        merged_list = []
        for core in unmerged_list:
            assert not isinstance(core, MergedInput)
            assert not isinstance(core, MergedOutput)
            if hasattr(core, '_merge'):
                merged_list.extend(core._merge(offset=site_num % 2))
            else:
                merged_list.append(core)
            site_num += core.core_len()
        while True:
            mod_num, site_num = 0, 0
            combined_list = []
            while mod_num < len(merged_list) - 1:
                left_core, right_core = merged_list[mod_num:mod_num + 2]
                new_core = self.combine(left_core, right_core, merging=True)
                if new_core is None or offset != site_num % 2:
                    combined_list.append(left_core)
                    mod_num += 1
                    site_num += left_core.core_len()
                else:
                    assert new_core.core_len() == left_core.core_len() + right_core.core_len()
                    combined_list.append(new_core)
                    mod_num += 2
                    site_num += new_core.core_len()
                if mod_num == len(merged_list) - 1:
                    combined_list.append(merged_list[mod_num])
                    mod_num += 1
            if len(combined_list) == len(merged_list):
                break
            else:
                merged_list = combined_list
        list_name = f'module_list_{offset}'
        if not hasattr(self, list_name):
            setattr(self, list_name, nn.ModuleList(merged_list))
        else:
            module_list = getattr(self, list_name)
            assert len(module_list) == len(merged_list)
            for i in range(len(module_list)):
                assert module_list[i].tensor.shape == merged_list[i].tensor.shape
                module_list[i].tensor[:] = merged_list[i].tensor

    @torch.no_grad()
    def _unmerge(self, cutoff=1e-10):
        """
        Convert merged modules to unmerged counterparts

        Calling _unmerge (or _merge) directly can cause undefined behavior,
        but see MergedLinearRegion.forward for intended use

        This proceeds by first unmerging all merged cores internally, then
        combining lone cores where possible
        """
        list_name = f'module_list_{self.offset}'
        merged_list = getattr(self, list_name)
        unmerged_list, bond_list, sv_list = [], [None], [None]
        for core in merged_list:
            if hasattr(core, '_unmerge'):
                new_cores, new_bonds, new_svs = core._unmerge(cutoff)
                unmerged_list.extend(new_cores)
                bond_list.extend(new_bonds[1:])
                sv_list.extend(new_svs[1:])
            else:
                assert not isinstance(core, InputRegion)
                unmerged_list.append(core)
                bond_list.append(None)
                sv_list.append(None)
        while True:
            mod_num = 0
            combined_list = []
            while mod_num < len(unmerged_list) - 1:
                left_core, right_core = unmerged_list[mod_num:mod_num + 2]
                new_core = self.combine(left_core, right_core, merging=False)
                if new_core is None:
                    combined_list.append(left_core)
                    mod_num += 1
                else:
                    combined_list.append(new_core)
                    mod_num += 2
                if mod_num == len(unmerged_list) - 1:
                    combined_list.append(unmerged_list[mod_num])
                    mod_num += 1
            if len(combined_list) == len(unmerged_list):
                break
            else:
                unmerged_list = combined_list
        log_norms = []
        for core in unmerged_list:
            log_norms.append([torch.log(norm) for norm in core.get_norm()])
        log_scale = sum([sum(ns) for ns in log_norms])
        log_scale /= sum([len(ns) for ns in log_norms])
        scales = [[torch.exp(log_scale - n) for n in ns] for ns in log_norms]
        for core, these_scales in zip(unmerged_list, scales):
            core.rescale_norm(these_scales)
        self.module_list = nn.ModuleList(unmerged_list)
        return bond_list, sv_list

    def combine(self, left_core, right_core, merging):
        """
        Combine a pair of cores into a new core using context-dependent rules

        Depending on the types of left_core and right_core, along with whether
        we're currently merging (merging=True) or unmerging (merging=False),
        either return a new core, or None if no rule exists for this context
        """
        if merging and (isinstance(left_core, OutputSite) and isinstance(right_core, InputSite) or isinstance(left_core, InputSite) and isinstance(right_core, OutputSite)):
            left_site = isinstance(left_core, InputSite)
            if left_site:
                new_tensor = torch.einsum('lui,our->olri', [left_core.tensor, right_core.tensor])
            else:
                new_tensor = torch.einsum('olu,uri->olri', [left_core.tensor, right_core.tensor])
            return MergedOutput(new_tensor, left_output=not left_site)
        elif not merging and (isinstance(left_core, InputRegion) and isinstance(right_core, InputSite) or isinstance(left_core, InputSite) and isinstance(right_core, InputRegion)):
            left_site = isinstance(left_core, InputSite)
            if left_site:
                left_tensor = left_core.tensor.unsqueeze(0)
                right_tensor = right_core.tensor
            else:
                left_tensor = left_core.tensor
                right_tensor = right_core.tensor.unsqueeze(0)
            assert left_tensor.shape[1:] == right_tensor.shape[1:]
            new_tensor = torch.cat([left_tensor, right_tensor])
            return InputRegion(new_tensor)
        else:
            return None

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return sum([len(module) for module in self.module_list])


class MPS(nn.Module):
    """
    Tunable MPS model giving mapping from fixed-size data to output vector

    Model works by first converting each 'pixel' (local data) to feature
    vector via a simple embedding, then contracting embeddings with inputs
    to each MPS cores. The resulting transition matrices are contracted
    together along bond dimensions (i.e. hidden state spaces), with output
    produced via an uncontracted edge of an additional output core.

    MPS model permits many customizable behaviors, including custom
    'routing' of MPS through the input, choice of boundary conditions
    (meaning the model can act as a tensor train or a tensor ring),
    GPU-friendly parallel evaluation, and an experimental mode to support
    adaptive choice of bond dimensions based on singular value spectrum.

    Args:
        input_dim:       Number of 'pixels' in the input to the MPS
        output_dim:      Size of the vectors output by MPS via output core
        bond_dim:        Dimension of the 'bonds' connecting adjacent MPS
                         cores, which act as hidden state spaces of the
                         model. In adaptive mode, bond_dim instead
                         specifies the maximum allowed bond dimension
        feature_dim:     Size of the local feature spaces each pixel is
                         embedded into (default: 2)
        periodic_bc:     Whether MPS has periodic boundary conditions (i.e.
                         is a tensor ring) or open boundary conditions
                         (i.e. is a tensor train) (default: False)
        parallel_eval:   Whether contraction of tensors is performed in a
                         serial or parallel fashion. The former is less
                         expensive for open boundary conditions, but
                         parallelizes more poorly (default: False)
        label_site:      Location in the MPS chain where output is placed
                         (default: input_dim // 2)
        path:            List specifying a path through the input data
                         which MPS is 'routed' along. For example, choosing
                         path=[0, 1, ..., input_dim-1] gives a standard
                         in-order traversal (behavior when path=None), while
                         path=[0, 2, ..., input_dim-1] specifies an MPS
                         accepting input only from even-valued input pixels
                         (default: None)
        init_std:        Size of the Gaussian noise used in default
                         near-identity initialization (default: 1e-9)
        initializer:     Pytorch initializer for custom initialization of
                         MPS cores, with None specifying default
                         near-identity initialization (default: None)
        use_bias:        Whether to use trainable bias matrices in MPS
                         cores, which are initialized near the zero matrix
                         (default: False)
        adaptive_mode:   Whether MPS is trained with experimental adaptive
                         bond dimensions selection (default: False)
        cutoff:          Singular value cutoff controlling bond dimension
                         adaptive selection (default: 1e-9)
        merge_threshold: Number of inputs before adaptive MPS shifts its
                         merge state once, with two shifts leading to the
                         update of all bond dimensions (default: 2000)
    """

    def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, periodic_bc=False, parallel_eval=False, label_site=None, path=None, init_std=1e-09, initializer=None, use_bias=True, adaptive_mode=False, cutoff=1e-10, merge_threshold=2000):
        super().__init__()
        if label_site is None:
            label_site = input_dim // 2
        assert label_site >= 0 and label_site <= input_dim
        if adaptive_mode:
            use_bias = False
        module_list = []
        init_args = {'bond_str': 'slri', 'shape': [label_site, bond_dim, bond_dim, feature_dim], 'init_method': ('min_random_eye' if adaptive_mode else 'random_zero', init_std, output_dim)}
        if label_site > 0:
            tensor = init_tensor(**init_args)
            module_list.append(InputRegion(tensor, use_bias=use_bias, fixed_bias=False))
        tensor = init_tensor(shape=[output_dim, bond_dim, bond_dim], bond_str='olr', init_method=('min_random_eye' if adaptive_mode else 'random_eye', init_std, output_dim))
        module_list.append(OutputSite(tensor))
        if label_site < input_dim:
            init_args['shape'] = [input_dim - label_site, bond_dim, bond_dim, feature_dim]
            tensor = init_tensor(**init_args)
            module_list.append(InputRegion(tensor, use_bias=use_bias, fixed_bias=False))
        if adaptive_mode:
            self.linear_region = MergedLinearRegion(module_list=module_list, periodic_bc=periodic_bc, parallel_eval=parallel_eval, cutoff=cutoff, merge_threshold=merge_threshold)
            self.bond_list = bond_dim * torch.ones(input_dim + 2, dtype=torch.long)
            if not periodic_bc:
                self.bond_list[0], self.bond_list[-1] = 1, 1
            self.sv_list = -1.0 * torch.ones([input_dim + 2, bond_dim])
        else:
            self.linear_region = LinearRegion(module_list=module_list, periodic_bc=periodic_bc, parallel_eval=parallel_eval)
        assert len(self.linear_region) == input_dim
        if path:
            assert isinstance(path, (list, torch.Tensor))
            assert len(path) == input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.feature_dim = feature_dim
        self.periodic_bc = periodic_bc
        self.adaptive_mode = adaptive_mode
        self.label_site = label_site
        self.path = path
        self.use_bias = use_bias
        self.cutoff = cutoff
        self.merge_threshold = merge_threshold
        self.feature_map = None

    def forward(self, input_data):
        """
        Embed our data and pass it to an MPS with a single output site

        Args:
            input_data (Tensor): Input with shape [batch_size, input_dim] or
                                 [batch_size, input_dim, feature_dim]. In the
                                 former case, the data points are turned into
                                 2D vectors using a default linear feature map.

                                 When using a user-specified path, the size of
                                 the second tensor mode need not exactly equal
                                 input_dim, since the path variable is used to
                                 slice a certain subregion of input_data. This
                                 can be used to define multiple MPS 'strings',
                                 which act on different parts of the input.
        """
        if self.path:
            path_inputs = []
            for site_num in self.path:
                path_inputs.append(input_data[:, site_num])
            input_data = torch.stack(path_inputs, dim=1)
        input_data = self.embed_input(input_data)
        output = self.linear_region(input_data)
        if isinstance(output, tuple):
            output, new_bonds, new_svs = output
            assert len(new_bonds) == len(self.bond_list)
            assert len(new_bonds) == len(new_svs)
            for i, bond_dim in enumerate(new_bonds):
                if bond_dim is not None:
                    assert new_svs[i] is not None
                    self.bond_list[i] = bond_dim
                    self.sv_list[i] = new_svs[i]
        return output

    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate local feature spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, input_dim], or
                                    [batch_size, input_dim, feature_dim]. In the
                                    latter case, the data is assumed to already
                                    be embedded, and is returned unchanged.

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                    [batch_size, input_dim, feature_dim]
        """
        assert len(input_data.shape) in [2, 3]
        assert input_data.size(1) == self.input_dim
        if len(input_data.shape) == 3:
            if input_data.size(2) != self.feature_dim:
                raise ValueError(f'input_data has wrong shape to be unembedded or pre-embedded data (input_data.shape = {list(input_data.shape)}, feature_dim = {self.feature_dim})')
            return input_data
        if self.feature_map is not None:
            f_map = self.feature_map
            embedded_data = torch.stack([torch.stack([f_map(x) for x in batch]) for batch in input_data])
            assert embedded_data.shape == torch.Size([input_data.size(0), self.input_dim, self.feature_dim])
        else:
            if self.feature_dim != 2:
                raise RuntimeError(f'self.feature_dim = {self.feature_dim}, but default feature_map requires self.feature_dim = 2')
            embedded_data = torch.stack([input_data, 1 - input_data], dim=2)
        return embedded_data

    def register_feature_map(self, feature_map):
        """
        Register a custom feature map to be used for embedding input data

        Args:
            feature_map (function): Takes a single scalar input datum and
                                    returns an embedded representation of the
                                    image. The output size of the function must
                                    match self.feature_dim. If feature_map=None,
                                    then the feature map will be reset to a
                                    simple default linear embedding
        """
        if feature_map is not None:
            out_shape = feature_map(torch.tensor(0)).shape
            needed_shape = torch.Size([self.feature_dim])
            if out_shape != needed_shape:
                raise ValueError(f'Given feature_map returns values of size {list(out_shape)}, but should return values of size {list(needed_shape)}')
        self.feature_map = feature_map

    def core_len(self):
        """
        Returns the number of cores, which is at least the required input size
        """
        return self.linear_region.core_len()

    def __len__(self):
        """
        Returns the number of input sites, which equals the input size
        """
        return self.input_dim

