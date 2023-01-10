import sys
_module = sys.modules[__name__]
del sys
conf = _module
mrisensesim = _module
profile_torchkbnufft = _module
setup = _module
tests = _module
conftest = _module
create_old_data = _module
test_dcomp = _module
test_interp = _module
test_math = _module
test_nufft = _module
test_sense_nufft = _module
test_toep = _module
torchkbnufft = _module
_autograd = _module
interp = _module
_math = _module
_nufft = _module
dcomp = _module
fft = _module
interp = _module
spmat = _module
toep = _module
utils = _module
functional = _module
interp = _module
nufft = _module
modules = _module
_kbmodule = _module
kbinterp = _module
kbnufft = _module

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


import numpy as np


import torch


from torch.autograd import Function


from torch import Tensor


from typing import Optional


from typing import Sequence


from typing import Union


from typing import List


import torch.fft


import torch.nn.functional as F


from typing import Tuple


import itertools


from scipy import special


from scipy.sparse import coo_matrix


import torch.nn as nn


DTYPE_MAP = [(torch.complex128, torch.float64), (torch.complex64, torch.float32)]


def build_numpy_spmatrix(omega: np.ndarray, numpoints: Sequence[int], im_size: Sequence[int], grid_size: Sequence[int], n_shift: Sequence[int], order: Sequence[float], alpha: Sequence[float]) ->coo_matrix:
    """Builds a sparse matrix with the interpolation coefficients.

    Args:
        omega: An array of coordinates to interpolate to (radians/voxel).
        numpoints: Number of points to use for interpolation in each dimension.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A scipy sparse interpolation matrix.
    """
    spmat = -1
    ndims = omega.shape[0]
    klength = omega.shape[1]

    def interp_coeff(om, npts, grdsz, alpha, order):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        Jvec = np.reshape(np.array(range(1, npts + 1)), (1, npts))
        kern_in = -1 * Jvec + np.expand_dims(interp_dist, 1)
        cur_coeff = np.zeros(shape=kern_in.shape, dtype=np.complex128)
        indices = np.absolute(kern_in) < npts / 2
        bess_arg = np.sqrt(1 - (kern_in[indices] / (npts / 2)) ** 2)
        denom = special.iv(order, alpha)
        cur_coeff[indices] = special.iv(order, alpha * bess_arg) / denom
        cur_coeff = np.real(cur_coeff)
        return cur_coeff, kern_in
    full_coef = []
    kd = []
    for it_om, it_im_size, it_grid_size, it_numpoints, it_om, it_alpha, it_order in zip(omega, im_size, grid_size, numpoints, omega, alpha, order):
        coef, kern_in = interp_coeff(it_om, it_numpoints, it_grid_size, it_alpha, it_order)
        gam = 2 * np.pi / it_grid_size
        phase_scale = 1.0j * gam * (it_im_size - 1) / 2
        phase = np.exp(phase_scale * kern_in)
        full_coef.append(phase * coef)
        koff = np.expand_dims(np.floor(it_om / gam - it_numpoints / 2), 1)
        Jvec = np.reshape(np.arange(1, it_numpoints + 1), (1, it_numpoints))
        kd.append(np.mod(Jvec + koff, it_grid_size) + 1)
    for i in range(len(kd)):
        kd[i] = (kd[i] - 1) * np.prod(grid_size[i + 1:])
    kk = kd[0]
    spmat_coef = full_coef[0]
    for i in range(1, ndims):
        Jprod = int(np.prod(numpoints[:i + 1]))
        kk = np.reshape(np.expand_dims(kk, 1) + np.expand_dims(kd[i], 2), (klength, Jprod))
        spmat_coef = np.reshape(np.expand_dims(spmat_coef, 1) * np.expand_dims(full_coef[i], 2), (klength, Jprod))
    phase = np.exp(1.0j * np.dot(np.transpose(omega), np.expand_dims(n_shift, 1)))
    spmat_coef = np.conj(spmat_coef) * phase
    trajind = np.expand_dims(np.arange(klength), 1)
    trajind = np.repeat(trajind, int(np.prod(numpoints)), axis=1)
    spmat = coo_matrix((spmat_coef.flatten(), (trajind.flatten(), kk.flatten())), shape=(klength, np.prod(grid_size)))
    return spmat


def build_table(im_size: Sequence[int], grid_size: Sequence[int], numpoints: Sequence[int], table_oversamp: Sequence[int], order: Sequence[float], alpha: Sequence[float]) ->List[Tensor]:
    """Builds an interpolation table.

    Args:
        numpoints: Number of points to use for interpolation in each dimension.
        table_oversamp: Table oversampling factor.
        grid_size: Size of the grid to interpolate from.
        im_size: Size of base image.
        ndims: Number of image dimensions.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A list of tables for each dimension.
    """
    table = []
    for it_im_size, it_grid_size, it_numpoints, it_table_oversamp, it_order, it_alpha in zip(im_size, grid_size, numpoints, table_oversamp, order, alpha):
        t1 = it_numpoints / 2 - 1 + np.array(range(it_table_oversamp)) / it_table_oversamp
        om1 = t1 * 2 * np.pi / it_grid_size
        s1 = build_numpy_spmatrix(np.expand_dims(om1, 0), numpoints=(it_numpoints,), im_size=(it_im_size,), grid_size=(it_grid_size,), n_shift=(0,), order=(it_order,), alpha=(it_alpha,))
        h = np.array(s1.getcol(it_numpoints - 1).todense())
        for col in range(it_numpoints - 2, -1, -1):
            h = np.concatenate((h, np.array(s1.getcol(col).todense())), axis=0)
        h = np.concatenate((h.flatten(), np.array([0])))
        table.append(torch.tensor(h))
    return table


def validate_args(im_size: Sequence[int], grid_size: Optional[Sequence[int]]=None, numpoints: Union[int, Sequence[int]]=6, n_shift: Optional[Sequence[int]]=None, table_oversamp: Union[int, Sequence[int]]=2 ** 10, kbwidth: float=2.34, order: Union[float, Sequence[float]]=0.0, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None) ->Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[float], Sequence[float], torch.dtype, torch.device]:
    im_size = tuple(im_size)
    if grid_size is None:
        grid_size = tuple([(dim * 2) for dim in im_size])
    else:
        grid_size = tuple(grid_size)
    if isinstance(numpoints, int):
        numpoints = tuple([numpoints for _ in range(len(grid_size))])
    else:
        numpoints = tuple(numpoints)
    if n_shift is None:
        n_shift = tuple([(dim // 2) for dim in im_size])
    else:
        n_shift = tuple(n_shift)
    if isinstance(table_oversamp, int):
        table_oversamp = tuple(table_oversamp for _ in range(len(grid_size)))
    else:
        table_oversamp = tuple(table_oversamp)
    alpha = tuple(kbwidth * numpoint for numpoint in numpoints)
    if isinstance(order, float):
        order = tuple(order for _ in range(len(grid_size)))
    else:
        order = tuple(order)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    assert len(grid_size) == len(im_size)
    assert len(n_shift) == len(im_size)
    assert len(numpoints) == len(im_size)
    assert len(alpha) == len(im_size)
    assert len(order) == len(im_size)
    assert len(table_oversamp) == len(im_size)
    return im_size, grid_size, numpoints, n_shift, table_oversamp, order, alpha, dtype, device


def init_fn(im_size: Sequence[int], grid_size: Optional[Sequence[int]]=None, numpoints: Union[int, Sequence[int]]=6, n_shift: Optional[Sequence[int]]=None, table_oversamp: Union[int, Sequence[int]]=2 ** 10, kbwidth: float=2.34, order: Union[float, Sequence[float]]=0.0, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None) ->Tuple[List[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Initialization function for NUFFT objects.

    Args:
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers.
        device: Which device to create tensors on.

    Returns:
        Tuple containing all variables recast as Tensors:
            tables (List of tensors)
            im_size
            grid_size
            n_shift
            numpoints
            offset_list
            table_oversamp
            order
            alpha
    """
    im_size, grid_size, numpoints, n_shift, table_oversamp, order, alpha, dtype, device = validate_args(im_size, grid_size, numpoints, n_shift, table_oversamp, kbwidth, order, dtype, device)
    tables = build_table(numpoints=numpoints, table_oversamp=table_oversamp, grid_size=grid_size, im_size=im_size, order=order, alpha=alpha)
    assert len(tables) == len(im_size)
    offset_list = list(itertools.product(*[range(numpoint) for numpoint in numpoints]))
    if dtype.is_floating_point:
        real_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[1] == real_dtype:
                complex_dtype = pair[0]
                break
    elif dtype.is_complex:
        complex_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[0] == complex_dtype:
                real_dtype = pair[1]
                break
    else:
        raise TypeError('Unrecognized dtype.')
    tables = [table for table in tables]
    return tables, torch.tensor(im_size, dtype=torch.long, device=device), torch.tensor(grid_size, dtype=torch.long, device=device), torch.tensor(n_shift, dtype=real_dtype, device=device), torch.tensor(numpoints, dtype=torch.long, device=device), torch.tensor(offset_list, dtype=torch.long, device=device), torch.tensor(table_oversamp, dtype=torch.long, device=device), torch.tensor(order, dtype=real_dtype, device=device), torch.tensor(alpha, dtype=real_dtype, device=device)


class KbModule(nn.Module):
    """Parent class for torchkbnufft modules.

    This class handles initialization of NUFFT precomputations and registers
    the resulting tensors as buffers.
    """

    def __init__(self, im_size: Sequence[int], grid_size: Optional[Sequence[int]]=None, numpoints: Union[int, Sequence[int]]=6, n_shift: Optional[Sequence[int]]=None, table_oversamp: Union[int, Sequence[int]]=2 ** 10, kbwidth: float=2.34, order: Union[float, Sequence[float]]=0.0, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__()
        tables, im_size_t, grid_size_t, n_shift_t, numpoints_t, offsets_t, table_oversamp_t, order_t, alpha_t = init_fn(im_size=im_size, grid_size=grid_size, numpoints=numpoints, n_shift=n_shift, table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, dtype=dtype, device=device)
        for i, table in enumerate(tables):
            self.register_buffer(f'table_{i}', table)
        self.register_buffer('im_size', im_size_t)
        self.register_buffer('grid_size', grid_size_t)
        self.register_buffer('n_shift', n_shift_t)
        self.register_buffer('numpoints', numpoints_t)
        self.register_buffer('offsets', offsets_t)
        self.register_buffer('table_oversamp', table_oversamp_t)
        self.register_buffer('order', order_t)
        self.register_buffer('alpha', alpha_t)

    def to(self, *args, **kwargs):
        """Rewrite nn.Module.to to support complex floats."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError('KbModule.to only accepts floating point or complex dtypes, but got desired dtype={}'.format(dtype))
            if dtype.is_complex:
                complex_dtype = dtype
                for pair in DTYPE_MAP:
                    if pair[0] == complex_dtype:
                        real_dtype = pair[1]
                        break
            elif dtype.is_floating_point:
                real_dtype = dtype
                for pair in DTYPE_MAP:
                    if pair[1] == real_dtype:
                        complex_dtype = pair[0]
                        break
            else:
                raise TypeError('Unrecognized type.')

        def convert(t):
            if t.is_floating_point() and dtype is not None:
                cur_dtype = real_dtype
            elif t.is_complex() and dtype is not None:
                cur_dtype = complex_dtype
            else:
                cur_dtype = None
            if convert_to_format is not None and t.dim() == 4:
                return t
            return t
        return self._apply(convert)

    def __repr__(self):
        out = '\n{}\n'.format(self.__class__.__name__)
        out = out + '----------------------------------------\n'
        out = out + 'buffers\n'
        for buf, val in self.__dict__['_buffers'].items():
            out = out + f'\ttensor: {buf}, shape: {tuple(val.shape)}\n'
        return out


class KbInterpModule(KbModule):
    """Parent class for KbInterp classes.

    See subclasses for an explanation of inputs.
    """

    def __init__(self, im_size: Sequence[int], grid_size: Optional[Sequence[int]]=None, numpoints: Union[int, Sequence[int]]=6, n_shift: Optional[Sequence[int]]=None, table_oversamp: Union[int, Sequence[int]]=2 ** 10, kbwidth: float=2.34, order: Union[float, Sequence[float]]=0.0, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__(im_size=im_size, grid_size=grid_size, numpoints=numpoints, n_shift=n_shift, table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, dtype=dtype, device=device)


class KbInterp(KbInterpModule):
    """Non-uniform Kaiser-Bessel interpolation layer.

    This object interpolates a grid of Fourier data to off-grid locations
    using a Kaiser-Bessel kernel. Mathematically, in one dimension it estimates
    :math:`Y_m, m \\in [0, ..., M-1]` at frequency locations :math:`\\omega_m`
    from :math:`X_k, k \\in [0, ..., K-1]`, the oversampled DFT of
    :math:`x_n, n \\in [0, ..., N-1]`. To perform the estimate, this layer
    applies

    .. math::
        Y_m = \\sum_{j=1}^J X_{\\{k_m+j\\}_K}u^*_j(\\omega_m),

    where :math:`u` is the Kaiser-Bessel kernel, :math:`k_m` is the index of
    the root offset of nearest samples of :math:`X` to frequency location
    :math:`\\omega_m`, and :math:`J` is the number of nearest neighbors to use
    from  :math:`X_k`. Multiple dimensions are handled separably. For a
    detailed description of the notation see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://doi.org/10.1109/TSP.2002.807005>`_.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image, analagous to :math:`N`
      (used for calculating the kernel but not for the actual operation).

    * :attr:`grid_size` is the size of the grid prior to interpolation,
      analogous to :math:`K`. To reduce errors, NUFFT operations are done on an
      oversampled grid to reduce interpolation distances. This will typically
      be 1.25 to 2 times :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest to use for interpolation,
      i.e., :math:`J`.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for ``fftshift``. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``

    Examples:

        >>> image = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> kb_ob = tkbn.KbInterp(im_size=(8, 8), grid_size=(8, 8))
        >>> data = kb_ob(image, omega)
    """

    def forward(self, image: Tensor, omega: Tensor, interp_mats: Optional[Tuple[Tensor, Tensor]]=None) ->Tensor:
        """Interpolate from gridded data to scattered data.

        Input tensors should be of shape ``(N, C) + grid_size``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size
        ``(len(grid_size), klength)`` or ``(N, len(grid_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        Note:

            If the batch dimension is included in ``omega``, the interpolator
            will parallelize over the batch dimension. This is efficient for
            many small trajectories that might occur in dynamic imaging
            settings.

        If your tensors are real-valued, ensure that 2 is the size of the last
        dimension.

        Args:
            image: Gridded data to be interpolated to scattered data.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix KB interpolation (overrides default table
                interpolation).

        Returns:
            ``image`` calculated at Fourier frequencies specified by ``omega``.
        """
        if interp_mats is not None:
            output = tkbnF.kb_spmat_interp(image=image, interp_mats=interp_mats)
        else:
            tables = []
            for i in range(len(self.im_size)):
                tables.append(getattr(self, f'table_{i}'))
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)
            output = tkbnF.kb_table_interp(image=image, omega=omega, tables=tables, n_shift=self.n_shift, numpoints=self.numpoints, table_oversamp=self.table_oversamp, offsets=self.offsets)
        return output


class KbInterpAdjoint(KbInterpModule):
    """Non-uniform Kaiser-Bessel interpolation adjoint layer.

    This object interpolates off-grid Fourier data to on-grid locations using a
    Kaiser-Bessel kernel. Mathematically, in one dimension it estimates
    :math:`X_k, k \\in [0, ..., K-1]`, the oversampled DFT of
    :math:`x_n, n \\in [0, ..., N-1]`, from a signal
    :math:`Y_m, m \\in [0, ..., M-1]` at frequency locations :math:`\\omega_m`.
    To perform the estimate, this layer applies

    .. math::
        X_k = \\sum_{j=1}^J \\sum_{m=0}^{M-1} Y_m u_j(\\omega_m)
        \\mathbb{1}_{\\{\\{k_m+j\\}_K=k\\}},

    where :math:`u` is the Kaiser-Bessel kernel, :math:`k_m` is the index of
    the root offset of nearest samples of :math:`X` to frequency location
    :math:`\\omega_m`, :math:`\\mathbb{1}` is an indicator function, and
    :math:`J` is the number of nearest neighbors to use from :math:`X_k`.
    Multiple dimensions are handled separably. For a detailed description of
    the notation see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://doi.org/10.1109/TSP.2002.807005>`_.

    Note:

        This function is not the inverse of :py:class:`KbInterp`; it is the
        adjoint.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image, analagous to :math:`N`
      (used for calculating the kernel but not for the actual operation).

    * :attr:`grid_size` is the size of the grid after adjoint interpolation,
      analogous to :math:`K`. To reduce errors, NUFFT operations are done on an
      oversampled grid to reduce interpolation distances. This will typically
      be 1.25 to 2 times :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors to use for
      interpolation, i.e., :math:`J`.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for ``fftshift``. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``

    Examples:

        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> adjkb_ob = tkbn.KbInterpAdjoint(im_size=(8, 8), grid_size=(8, 8))
        >>> image = adjkb_ob(data, omega)
    """

    def forward(self, data: Tensor, omega: Tensor, interp_mats: Optional[Tuple[Tensor, Tensor]]=None, grid_size: Optional[Tensor]=None) ->Tensor:
        """Interpolate from scattered data to gridded data.

        Input tensors should be of shape ``(N, C) + klength``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size
        ``(len(grid_size), klength)`` or ``(N, len(grid_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        Note:

            If the batch dimension is included in ``omega``, the interpolator
            will parallelize over the batch dimension. This is efficient for
            many small trajectories that might occur in dynamic imaging
            settings.

        If your tensors are real-valued, ensure that 2 is the size of the last
        dimension.

        Args:
            data: Data to be gridded.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix KB interpolation (overrides default table
                interpolation).

        Returns:
            ``data`` interpolated to the grid.
        """
        if grid_size is None:
            assert isinstance(self.grid_size, Tensor)
            grid_size = self.grid_size
        if interp_mats is not None:
            output = tkbnF.kb_spmat_interp_adjoint(data=data, interp_mats=interp_mats, grid_size=grid_size)
        else:
            tables = []
            for i in range(len(self.im_size)):
                tables.append(getattr(self, f'table_{i}'))
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)
            output = tkbnF.kb_table_interp_adjoint(data=data, omega=omega, tables=tables, n_shift=self.n_shift, numpoints=self.numpoints, table_oversamp=self.table_oversamp, offsets=self.offsets, grid_size=grid_size)
        return output


def kaiser_bessel_ft(omega: np.ndarray, numpoints: int, alpha: float, order: float, d: int) ->np.ndarray:
    """Computes FT of KB function for scaling in image domain.

    Args:
        omega: An array of coordinates to interpolate to.
        numpoints: Number of points to use for interpolation in each dimension.
        alpha: KB parameter.
        order: Order of Kaiser-Bessel kernel.
        d (int):  # TODO: find what d is

    Returns:
        The scaling coefficients.
    """
    z = np.sqrt((2 * np.pi * (numpoints / 2) * omega) ** 2 - alpha ** 2 + 0.0j)
    nu = d / 2 + order
    scaling_coef = (2 * np.pi) ** (d / 2) * (numpoints / 2) ** d * alpha ** order / special.iv(order, alpha) * special.jv(nu, z) / z ** nu
    scaling_coef = np.real(scaling_coef)
    return scaling_coef


def compute_scaling_coefs(im_size: Sequence[int], grid_size: Sequence[int], numpoints: Sequence[int], alpha: Sequence[float], order: Sequence[float]) ->Tensor:
    """Computes scaling coefficients for NUFFT operation.

    Args:
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        numpoints: Number of points to use for interpolation in each dimension.
        alpha: KB parameter.
        order: Order of Kaiser-Bessel kernel.

    Returns:
        The scaling coefficients.
    """
    num_coefs = np.array(range(im_size[0])) - (im_size[0] - 1) / 2
    scaling_coef = 1 / kaiser_bessel_ft(num_coefs / grid_size[0], numpoints[0], alpha[0], order[0], 1)
    if numpoints[0] == 1:
        scaling_coef = np.ones(scaling_coef.shape)
    for i in range(1, len(im_size)):
        indlist = np.array(range(im_size[i])) - (im_size[i] - 1) / 2
        scaling_coef = np.expand_dims(scaling_coef, axis=-1)
        tmp = 1 / kaiser_bessel_ft(indlist / grid_size[i], numpoints[i], alpha[i], order[i], 1)
        for _ in range(i):
            tmp = tmp[np.newaxis]
        if numpoints[i] == 1:
            tmp = np.ones(tmp.shape)
        scaling_coef = scaling_coef * tmp
    return torch.tensor(scaling_coef)


class KbNufftModule(KbModule):
    """Parent class for KbNufft classes.

    See subclasses for an explanation of inputs.
    """

    def __init__(self, im_size: Sequence[int], grid_size: Optional[Sequence[int]]=None, numpoints: Union[int, Sequence[int]]=6, n_shift: Optional[Sequence[int]]=None, table_oversamp: Union[int, Sequence[int]]=2 ** 10, kbwidth: float=2.34, order: Union[float, Sequence[float]]=0.0, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__(im_size=im_size, grid_size=grid_size, numpoints=numpoints, n_shift=n_shift, table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, dtype=dtype, device=device)
        scaling_coef = compute_scaling_coefs(im_size=self.im_size.tolist(), grid_size=self.grid_size.tolist(), numpoints=self.numpoints.tolist(), alpha=self.alpha.tolist(), order=self.order.tolist())
        self.register_buffer('scaling_coef', scaling_coef)


class KbNufft(KbNufftModule):
    """Non-uniform FFT layer.

    This object applies the FFT and interpolates a grid of Fourier data to
    off-grid locations using a Kaiser-Bessel kernel. Mathematically, in one
    dimension it estimates :math:`Y_m, m \\in [0, ..., M-1]` at frequency
    locations :math:`\\omega_m` from :math:`X_k, k \\in [0, ..., K-1]`, the
    oversampled DFT of :math:`x_n, n \\in [0, ..., N-1]`. To perform the
    estimate, this layer applies

    .. math::
        X_k = \\sum_{n=0}^{N-1} s_n x_n e^{-i \\gamma k n}
    .. math::
        Y_m = \\sum_{j=1}^J X_{\\{k_m+j\\}_K} u^*_j(\\omega_m)

    In the first step, an image-domain signal :math:`x_n` is converted to a
    gridded, oversampled frequency-domain signal, :math:`X_k`. The scaling
    coefficeints :math:`s_n` are multiplied to precompensate for NUFFT
    interpolation errors. The oversampling coefficient is
    :math:`\\gamma = 2\\pi / K, K >= N`.

    In the second step, :math:`u`, the Kaiser-Bessel kernel, is used to
    estimate :math:`X_k` at off-grid frequency locations :math:`\\omega_m`.
    :math:`k_m` is the index of the root offset of nearest samples of :math:`X`
    to frequency location :math:`\\omega_m`, and :math:`J` is the number of
    nearest neighbors to use from :math:`X_k`. Multiple dimensions are handled
    separably. For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://doi.org/10.1109/TSP.2002.807005>`_.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image, analagous to :math:`N`.

    * :attr:`grid_size` is the size of the grid after forward FFT, analogous
      to :math:`K`. To reduce errors, NUFFT operations are done on an
      oversampled grid to reduce interpolation distances. This will typically
      be 1.25 to 2 times :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors to use
      for interpolation, i.e., :math:`J`.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for ``fftshift``. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``

    Examples:

        >>> image = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> kb_ob = tkbn.KbNufft(im_size=(8, 8))
        >>> data = kb_ob(image, omega)
    """

    def forward(self, image: Tensor, omega: Tensor, interp_mats: Optional[Tuple[Tensor, Tensor]]=None, smaps: Optional[Tensor]=None, norm: Optional[str]=None) ->Tensor:
        """Apply FFT and interpolate from gridded data to scattered data.

        Input tensors should be of shape ``(N, C) + im_size``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(grid_size), klength)``
        or ``(N, len(grid_size), klength)``, where ``klength`` is the length of
        the k-space trajectory.

        Note:

            If the batch dimension is included in ``omega``, the interpolator
            will parallelize over the batch dimension. This is efficient for
            many small trajectories that might occur in dynamic imaging
            settings.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            image: Object to calculate off-grid Fourier samples from.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix NUFFT interpolation (overrides default table
                interpolation).
            smaps: Sensitivity maps. If input, these will be multiplied before
                the forward NUFFT.
            norm: Whether to apply normalization with the FFT operation.
                Options are ``"ortho"`` or ``None``.

        Returns:
            ``image`` calculated at Fourier frequencies specified by ``omega``.
        """
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError('image dtype does not match smaps dtype.')
        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError('For real inputs, last dimension must be size 2.')
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError('For real inputs, last dimension must be size 2.')
                smaps = torch.view_as_complex(smaps)
            is_complex = False
            image = torch.view_as_complex(image)
        if smaps is not None:
            image = image * smaps
        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            output = tkbnF.kb_spmat_nufft(image=image, scaling_coef=self.scaling_coef, im_size=self.im_size, grid_size=self.grid_size, interp_mats=interp_mats, norm=norm)
        else:
            tables = []
            for i in range(len(self.im_size)):
                tables.append(getattr(self, f'table_{i}'))
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)
            output = tkbnF.kb_table_nufft(image=image, scaling_coef=self.scaling_coef, im_size=self.im_size, grid_size=self.grid_size, omega=omega, tables=tables, n_shift=self.n_shift, numpoints=self.numpoints, table_oversamp=self.table_oversamp, offsets=self.offsets, norm=norm)
        if not is_complex:
            output = torch.view_as_real(output)
        return output


class KbNufftAdjoint(KbNufftModule):
    """Non-uniform FFT adjoint layer.

    This object interpolates off-grid Fourier data to on-grid locations
    using a Kaiser-Bessel kernel prior to inverse DFT. Mathematically, in one
    dimension it estimates :math:`x_n, n \\in [0, ..., N-1]` from a off-grid
    signal :math:`Y_m, m \\in [0, ..., M-1]` where the off-grid frequency
    locations are :math:`\\omega_m`. To perform the estimate, this layer applies

    .. math::
        X_k = \\sum_{j=1}^J \\sum_{m=0}^{M-1} Y_m u_j(\\omega_m)
        \\mathbb{1}_{\\{\\{k_m+j\\}_K=k\\}},
    .. math::
        x_n = s_n^* \\sum_{k=0}^{K-1} X_k e^{i \\gamma k n}

    In the first step, :math:`u`, the Kaiser-Bessel kernel, is used to
    estimate :math:`Y` at on-grid frequency locations from locations at
    :math:`\\omega`. :math:`k_m` is the index of the root offset of nearest
    samples of :math:`X` to frequency location :math:`\\omega_m`,
    :math:`\\mathbb{1}` is an indicator function, and :math:`J` is the number of
    nearest neighbors to use from :math:`X_k, k \\in [0, ..., K-1]`.

    In the second step, an image-domain signal :math:`x_n` is estimated from a
    gridded, oversampled frequency-domain signal, :math:`X_k` by applying the
    inverse FFT, after which the complex conjugate scaling coefficients
    :math:`s_n` are multiplied. The oversampling coefficient is
    :math:`\\gamma = 2\\pi / K, K >= N`. Multiple dimensions are handled
    separably. For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://doi.org/10.1109/TSP.2002.807005>`_.

    Note:

        This function is not the inverse of :py:class:`KbNufft`; it is the
        adjoint.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image, analagous to :math:`N`.

    * :attr:`grid_size` is the size of the grid after adjoint interpolation,
      analogous to :math:`K`. To reduce errors, NUFFT operations are done on an
      oversampled grid to reduce interpolation distances. This will typically
      be 1.25 to 2 times :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors to use for
      interpolation, i.e., :math:`J`.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for ``fftshift``. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``

    Examples:

        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
        >>> image = adjkb_ob(data, omega)
    """

    def forward(self, data: Tensor, omega: Tensor, interp_mats: Optional[Tuple[Tensor, Tensor]]=None, smaps: Optional[Tensor]=None, norm: Optional[str]=None) ->Tensor:
        """Interpolate from scattered data to gridded data and then iFFT.

        Input tensors should be of shape ``(N, C) + klength``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(grid_size), klength)``
        or ``(N, len(grid_size), klength)``, where ``klength`` is the length of
        the k-space trajectory.

        Note:

            If the batch dimension is included in ``omega``, the interpolator
            will parallelize over the batch dimension. This is efficient for
            many small trajectories that might occur in dynamic imaging
            settings.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            data: Data to be gridded and then inverse FFT'd.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix NUFFT interpolation (overrides default table
                interpolation).
            smaps: Sensitivity maps. If input, these will be multiplied before
                the forward NUFFT.
            norm: Whether to apply normalization with the FFT operation.
                Options are ``"ortho"`` or ``None``.

        Returns:
            ``data`` transformed to the image domain.
        """
        if smaps is not None:
            if not smaps.dtype == data.dtype:
                raise TypeError('data dtype does not match smaps dtype.')
        is_complex = True
        if not data.is_complex():
            if not data.shape[-1] == 2:
                raise ValueError('For real inputs, last dimension must be size 2.')
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError('For real inputs, last dimension must be size 2.')
                smaps = torch.view_as_complex(smaps)
            is_complex = False
            data = torch.view_as_complex(data)
        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            output = tkbnF.kb_spmat_nufft_adjoint(data=data, scaling_coef=self.scaling_coef, im_size=self.im_size, grid_size=self.grid_size, interp_mats=interp_mats, norm=norm)
        else:
            tables = []
            for i in range(len(self.im_size)):
                tables.append(getattr(self, f'table_{i}'))
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)
            output = tkbnF.kb_table_nufft_adjoint(data=data, scaling_coef=self.scaling_coef, im_size=self.im_size, grid_size=self.grid_size, omega=omega, tables=tables, n_shift=self.n_shift, numpoints=self.numpoints, table_oversamp=self.table_oversamp, offsets=self.offsets, norm=norm)
        if smaps is not None:
            output = torch.sum(output * smaps.conj(), dim=1, keepdim=True)
        if not is_complex:
            output = torch.view_as_real(output)
        return output


class ToepNufft(torch.nn.Module):
    """Forward/backward NUFFT with Toeplitz embedding.

    This module applies :math:`Tx`, where :math:`T` is a matrix such that
    :math:`T \\approx A'A`, where :math:`A` is a NUFFT matrix. Using Toeplitz
    embedding, this module approximates the :math:`A'A` operation without
    interpolations, which is extremely fast.

    The module is intended to be used in combination with an FFT kernel
    computed as frequency response of an embedded Toeplitz matrix.
    You can use :py:meth:`~torchkbnufft.calc_toeplitz_kernel` to calculate the
    kernel.

    The FFT kernel should be passed to this module's forward operation, which
    applies a (zero-padded) FFT filter using the kernel.

    Examples:

        >>> image = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> toep_ob = tkbn.ToepNufft()
        >>> kernel = tkbn.calc_toeplitz_kernel(omega, im_size=(8, 8))
        >>> image = toep_ob(image, kernel)
    """

    def __init__(self):
        super().__init__()

    def toep_batch_loop(self, image: Tensor, smaps: Tensor, kernel: Tensor, norm: Optional[str]) ->Tensor:
        output = []
        if len(kernel.shape) > len(image.shape[2:]):
            if smaps.shape[0] == 1:
                for mini_image, mini_kernel in zip(image, kernel):
                    mini_image = mini_image.unsqueeze(0) * smaps
                    mini_image = tkbnF.fft_filter(image=mini_image, kernel=mini_kernel, norm=norm)
                    mini_image = torch.sum(mini_image * smaps.conj(), dim=1, keepdim=True)
                    output.append(mini_image.squeeze(0))
            else:
                for mini_image, smap, mini_kernel in zip(image, smaps, kernel):
                    mini_image = mini_image.unsqueeze(0) * smap.unsqueeze(0)
                    mini_image = tkbnF.fft_filter(image=mini_image, kernel=mini_kernel, norm=norm)
                    mini_image = torch.sum(mini_image * smap.unsqueeze(0).conj(), dim=1, keepdim=True)
                    output.append(mini_image.squeeze(0))
        else:
            for mini_image, smap in zip(image, smaps):
                mini_image = mini_image.unsqueeze(0) * smap.unsqueeze(0)
                mini_image = tkbnF.fft_filter(image=mini_image, kernel=kernel, norm=norm)
                mini_image = torch.sum(mini_image * smap.unsqueeze(0).conj(), dim=1, keepdim=True)
                output.append(mini_image.squeeze(0))
        return torch.stack(output)

    def forward(self, image: Tensor, kernel: Tensor, smaps: Optional[Tensor]=None, norm: Optional[str]=None) ->Tensor:
        """Toeplitz NUFFT forward function.

        Args:
            image: The image to apply the forward/backward Toeplitz-embedded
                NUFFT to.
            kernel: The filter response taking into account Toeplitz embedding.
            norm: Whether to apply normalization with the FFT operation.
                Options are ``"ortho"`` or ``None``.

        Returns:
            ``image`` after applying the Toeplitz forward/backward NUFFT.
        """
        if not kernel.dtype == image.dtype:
            raise TypeError('kernel and image must have same dtype.')
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError('image dtype does not match smaps dtype.')
        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError('For real inputs, last dimension must be size 2.')
            if not kernel.shape[-1] == 2:
                raise ValueError('For real inputs, last dimension must be size 2.')
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError('For real inputs, last dimension must be size 2.')
                smaps = torch.view_as_complex(smaps)
            is_complex = False
            image = torch.view_as_complex(image)
            kernel = torch.view_as_complex(kernel)
        if len(kernel.shape) > len(image.shape[2:]):
            if kernel.shape[0] == 1:
                kernel = kernel[0]
            elif not kernel.shape[0] == image.shape[0]:
                raise ValueError('If using batch dimension, kernel must have same batch size as image')
        if smaps is None:
            output = tkbnF.fft_filter(image=image, kernel=kernel, norm=norm)
        else:
            output = self.toep_batch_loop(image=image, smaps=smaps, kernel=kernel, norm=norm)
        if not is_complex:
            output = torch.view_as_real(output)
        return output

