import sys
_module = sys.modules[__name__]
del sys
conf = _module
pytorch_wavelets = _module
_version = _module
dtcwt = _module
coeffs = _module
data = _module
lowlevel = _module
lowlevel2 = _module
transform2d = _module
transform_funcs = _module
dwt = _module
lowlevel = _module
swt_inverse = _module
transform2d = _module
scatternet = _module
layers = _module
lowlevel = _module
utils = _module
save = _module
setup = _module
Transform2d_np = _module
datasets = _module
near_sym_a2 = _module
parser = _module
profile = _module
profile2 = _module
test_coldfilt = _module
test_colfilter = _module
test_dtcwt = _module
test_dtcwt_grad = _module
test_dwt = _module
test_dwt_grad = _module
test_rowdfilt = _module
test_rowfilter = _module
test_scatnet_bwd = _module
test_scatnet_fwd = _module

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


import torch


import torch.nn.functional as F


import numpy as np


import torch.nn as nn


from numpy import ndarray


from numpy import sqrt


from torch import tensor


from torch.autograd import Function


import re


from torch.autograd import gradcheck


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
        """

    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 2:
            h0_col, h1_col = wave[0], wave[1]
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 4:
            h0_col, h1_col = wave[0], wave[1]
            h0_row, h1_row = wave[2], wave[3]
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)
        for j in range(self.J):
            ll, high = lowlevel.AFB2D.apply(ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)
        return ll, yh


COEFF_CACHE = {}


def _load_from_file(basename, varnames):
    try:
        mat = COEFF_CACHE[basename]
    except KeyError:
        with resource_stream('pytorch_wavelets.dtcwt.data', basename + '.npz') as f:
            mat = dict(load(f))
        COEFF_CACHE[basename] = mat
    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(varnames)))


def level1(name, compact=False):
    """Load level 1 wavelet by name.

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    =============  ============================================
    Name           Wavelet
    =============  ============================================
    antonini       Antonini 9,7 tap filters.
    farras         Farras 8,8 tap filters
    legall         LeGall 5,3 tap filters.
    near_sym_a     Near-Symmetric 5,7 tap filters.
    near_sym_b     Near-Symmetric 13,19 tap filters.
    near_sym_b_bp  Near-Symmetric 13,19 tap filters + BP filter
    =============  ============================================

    Return a tuple whose elements are a vector specifying the h0o, g0o, h1o and
    g1o coefficients.

    See :ref:`rot-symm-wavelets` for an explanation of the ``near_sym_b_bp``
    wavelet filters.

    :raises IOError: if name does not correspond to a set of wavelets known to
        the library.
    :raises ValueError: if name doesn't specify
        :py:func:`pytorch_wavelets.dtcwt.coeffs.qshift` wavelet.

    """
    if compact:
        if name == 'near_sym_b_bp':
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o', 'h2o', 'g2o'))
        else:
            return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b'))


def pm(a, b):
    u = (a + b) / sqrt(2)
    v = (a - b) / sqrt(2)
    return u, v


class DTCWTForward2(nn.Module):
    """ DTCWT based on 4 DWTs. Still works, but the above implementation is
    faster """

    def __init__(self, biort='farras', qshift='qshift_a', J=3, mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.J = J
        if isinstance(biort, str):
            biort = level1(biort)
        assert len(biort) == 8
        h0a1, h0b1, _, _, h1a1, h1b1, _, _ = biort
        DWTaa1 = DWTForward(J=1, wave=(h0a1, h1a1, h0a1, h1a1), mode=mode)
        DWTab1 = DWTForward(J=1, wave=(h0a1, h1a1, h0b1, h1b1), mode=mode)
        DWTba1 = DWTForward(J=1, wave=(h0b1, h1b1, h0a1, h1a1), mode=mode)
        DWTbb1 = DWTForward(J=1, wave=(h0b1, h1b1, h0b1, h1b1), mode=mode)
        self.level1 = nn.ModuleList([DWTaa1, DWTab1, DWTba1, DWTbb1])
        if J > 1:
            if isinstance(qshift, str):
                qshift = _qshift(qshift)
            assert len(qshift) == 8
            h0a, h0b, _, _, h1a, h1b, _, _ = qshift
            DWTaa = DWTForward(J - 1, (h0a, h1a, h0a, h1a), mode=mode)
            DWTab = DWTForward(J - 1, (h0a, h1a, h0b, h1b), mode=mode)
            DWTba = DWTForward(J - 1, (h0b, h1b, h0a, h1a), mode=mode)
            DWTbb = DWTForward(J - 1, (h0b, h1b, h0b, h1b), mode=mode)
            self.level2 = nn.ModuleList([DWTaa, DWTab, DWTba, DWTbb])

    def forward(self, x):
        x = x / 2
        J = self.J
        w = [[[None for _ in range(2)] for _ in range(2)] for j in range(J)]
        lows = [[None for _ in range(2)] for _ in range(2)]
        for m in range(2):
            for n in range(2):
                ll, (w[0][m][n],) = self.level1[m * 2 + n](x)
                if J > 1:
                    ll, bands = self.level2[m * 2 + n](ll)
                    for j in range(1, J):
                        w[j][m][n] = bands[j - 1]
                lows[m][n] = ll
        yh = [None] * J
        for j in range(J):
            deg75r, deg105i = pm(w[j][0][0][:, :, (1)], w[j][1][1][:, :, (1)])
            deg105r, deg75i = pm(w[j][0][1][:, :, (1)], w[j][1][0][:, :, (1)])
            deg15r, deg165i = pm(w[j][0][0][:, :, (0)], w[j][1][1][:, :, (0)])
            deg165r, deg15i = pm(w[j][0][1][:, :, (0)], w[j][1][0][:, :, (0)])
            deg135r, deg45i = pm(w[j][0][0][:, :, (2)], w[j][1][1][:, :, (2)])
            deg45r, deg135i = pm(w[j][0][1][:, :, (2)], w[j][1][0][:, :, (2)])
            w[j] = None
            yhr = torch.stack((deg15r, deg45r, deg75r, deg105r, deg135r, deg165r), dim=1)
            yhi = torch.stack((deg15i, deg45i, deg75i, deg105i, deg135i, deg165i), dim=1)
            yh[j] = torch.stack((yhr, yhi), dim=-1)
        return lows, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """

    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 2:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 4:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = wave[2], wave[3]
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.g0_col = nn.Parameter(filts[0], requires_grad=False)
        self.g1_col = nn.Parameter(filts[1], requires_grad=False)
        self.g0_row = nn.Parameter(filts[2], requires_grad=False)
        self.g1_row = nn.Parameter(filts[3], requires_grad=False)
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[(...), :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[(...), :-1]
            ll = lowlevel.SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll


class DTCWTInverse2(nn.Module):

    def __init__(self, biort='farras', qshift='qshift_a', mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        if isinstance(biort, str):
            biort = level1(biort)
        assert len(biort) == 8
        _, _, g0a1, g0b1, _, _, g1a1, g1b1 = biort
        IWTaa1 = DWTInverse(wave=(g0a1, g1a1, g0a1, g1a1), mode=mode)
        IWTab1 = DWTInverse(wave=(g0a1, g1a1, g0b1, g1b1), mode=mode)
        IWTba1 = DWTInverse(wave=(g0b1, g1b1, g0a1, g1a1), mode=mode)
        IWTbb1 = DWTInverse(wave=(g0b1, g1b1, g0b1, g1b1), mode=mode)
        self.level1 = nn.ModuleList([IWTaa1, IWTab1, IWTba1, IWTbb1])
        if isinstance(qshift, str):
            qshift = _qshift(qshift)
        assert len(qshift) == 8
        _, _, g0a, g0b, _, _, g1a, g1b = qshift
        IWTaa = DWTInverse(wave=(g0a, g1a, g0a, g1a), mode=mode)
        IWTab = DWTInverse(wave=(g0a, g1a, g0b, g1b), mode=mode)
        IWTba = DWTInverse(wave=(g0b, g1b, g0a, g1a), mode=mode)
        IWTbb = DWTInverse(wave=(g0b, g1b, g0b, g1b), mode=mode)
        self.level2 = nn.ModuleList([IWTaa, IWTab, IWTba, IWTbb])

    def forward(self, x):
        yl, yh = x
        J = len(yh)
        w = [[[[None for band in range(3)] for j in range(J)] for m in range(2)] for n in range(2)]
        for j in range(J):
            w[0][0][j][0], w[1][1][j][0] = pm(yh[j][:, (2), :, :, :, (0)], yh[j][:, (3), :, :, :, (1)])
            w[0][1][j][0], w[1][0][j][0] = pm(yh[j][:, (3), :, :, :, (0)], yh[j][:, (2), :, :, :, (1)])
            w[0][0][j][1], w[1][1][j][1] = pm(yh[j][:, (0), :, :, :, (0)], yh[j][:, (5), :, :, :, (1)])
            w[0][1][j][1], w[1][0][j][1] = pm(yh[j][:, (5), :, :, :, (0)], yh[j][:, (0), :, :, :, (1)])
            w[0][0][j][2], w[1][1][j][2] = pm(yh[j][:, (1), :, :, :, (0)], yh[j][:, (4), :, :, :, (1)])
            w[0][1][j][2], w[1][0][j][2] = pm(yh[j][:, (4), :, :, :, (0)], yh[j][:, (1), :, :, :, (1)])
            w[0][0][j] = torch.stack(w[0][0][j], dim=2)
            w[0][1][j] = torch.stack(w[0][1][j], dim=2)
            w[1][0][j] = torch.stack(w[1][0][j], dim=2)
            w[1][1][j] = torch.stack(w[1][1][j], dim=2)
        y = None
        for m in range(2):
            for n in range(2):
                lo = yl[m][n]
                if J > 1:
                    lo = self.level2[m * 2 + n]((lo, w[m][n][1:]))
                lo = self.level1[m * 2 + n]((lo, (w[m][n][0],)))
                if y is None:
                    y = lo
                else:
                    y = y + lo
        y = y / 2
        return y


def colfilter(X, h, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    if mode == 'symmetric':
        xe = symm_pad(row, m)
        X = F.conv2d(X[:, :, (xe)], h.repeat(ch, 1, 1, 1), groups=ch)
    else:
        X = F.conv2d(X, h.repeat(ch, 1, 1, 1), groups=ch, padding=(m, 0))
    return X


def q2c(y, dim=-1):
    """
    Convert from quads in y to complex numbers in z.
    """
    y = y / np.sqrt(2)
    a, b = y[:, :, 0::2, 0::2], y[:, :, 0::2, 1::2]
    c, d = y[:, :, 1::2, 0::2], y[:, :, 1::2, 1::2]
    return (a - d, b + c), (a + d, b - c)


def highs_to_orientations(lh, hl, hh, o_dim):
    (deg15r, deg15i), (deg165r, deg165i) = q2c(lh)
    (deg45r, deg45i), (deg135r, deg135i) = q2c(hh)
    (deg75r, deg75i), (deg105r, deg105i) = q2c(hl)
    reals = torch.stack([deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=o_dim)
    imags = torch.stack([deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=o_dim)
    return reals, imags


def rowfilter(X, h, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    b, ch, row, col = X.shape
    m = h.shape[2] // 2
    h = h.transpose(2, 3).contiguous()
    if mode == 'symmetric':
        xe = symm_pad(col, m)
        X = F.conv2d(X[:, :, :, (xe)], h.repeat(ch, 1, 1, 1), groups=ch)
    else:
        X = F.conv2d(X, h.repeat(ch, 1, 1, 1), groups=ch, padding=(0, m))
    return X


def fwd_j1(x, h0, h1, skip_hps, o_dim, mode):
    """ Level 1 forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = rowfilter(x, h0, mode)
        hi = rowfilter(x, h1, mode)
        ll = colfilter(lo, h0, mode)
        lh = colfilter(lo, h1, mode)
        del lo
        hl = colfilter(hi, h0, mode)
        hh = colfilter(hi, h1, mode)
        del hi
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = rowfilter(x, h0, mode)
        ll = colfilter(ll, h0, mode)
        highr = x.new_zeros([])
        highi = x.new_zeros([])
    return ll, highr, highi


def get_dimensions5(o_dim, ri_dim):
    """ Get the orientation, height and width dimensions after the real and
    imaginary parts have been popped off (5 dimensional tensor)."""
    o_dim = o_dim % 6
    ri_dim = ri_dim % 6
    if ri_dim < o_dim:
        o_dim -= 1
    if o_dim == 4:
        h_dim = 2
        w_dim = 3
    elif o_dim == 3:
        h_dim = 2
        w_dim = 4
    else:
        h_dim = 3
        w_dim = 4
    return o_dim, ri_dim, h_dim, w_dim


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError('Unkown pad type: {}'.format(mode))


def c2q(w1, w2):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """
    w1r, w1i = w1
    w2r, w2i = w2
    x1 = w1r + w2r
    x2 = w1i + w2i
    x3 = w1i - w2i
    x4 = -w1r + w2r
    b, ch, r, c = w1r.shape
    y = w1r.new_zeros((b, ch, r * 2, c * 2), requires_grad=w1r.requires_grad)
    y[:, :, ::2, ::2] = x1
    y[:, :, ::2, 1::2] = x2
    y[:, :, 1::2, ::2] = x3
    y[:, :, 1::2, 1::2] = x4
    y /= np.sqrt(2)
    return y


def orientations_to_highs(reals, imags, o_dim):
    dev = reals.device
    horiz = torch.index_select(reals, o_dim, tensor([0, 5], device=dev))
    diag = torch.index_select(reals, o_dim, tensor([1, 4], device=dev))
    vertic = torch.index_select(reals, o_dim, tensor([2, 3], device=dev))
    deg15r, deg165r = torch.unbind(horiz, dim=o_dim)
    deg45r, deg135r = torch.unbind(diag, dim=o_dim)
    deg75r, deg105r = torch.unbind(vertic, dim=o_dim)
    dev = imags.device
    horiz = torch.index_select(imags, o_dim, tensor([0, 5], device=dev))
    diag = torch.index_select(imags, o_dim, tensor([1, 4], device=dev))
    vertic = torch.index_select(imags, o_dim, tensor([2, 3], device=dev))
    deg15i, deg165i = torch.unbind(horiz, dim=o_dim)
    deg45i, deg135i = torch.unbind(diag, dim=o_dim)
    deg75i, deg105i = torch.unbind(vertic, dim=o_dim)
    lh = c2q((deg15r, deg15i), (deg165r, deg165i))
    hl = c2q((deg75r, deg75i), (deg105r, deg105i))
    hh = c2q((deg45r, deg45i), (deg135r, deg135i))
    return lh, hl, hh


def inv_j1(ll, highr, highi, g0, g1, o_dim, h_dim, w_dim, mode):
    """ Level1 inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)
        if ll is None or ll.shape == torch.Size([]):
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode)
            del lh, hh, hl
        else:
            r, c = ll.shape[2:]
            r1, c1 = highr.shape[h_dim], highr.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:, :, 1:-1]
            if c != c1 * 2:
                ll = ll[:, :, :, 1:-1]
            hi = colfilter(hh, g1, mode) + colfilter(hl, g0, mode)
            lo = colfilter(lh, g1, mode) + colfilter(ll, g0, mode)
            del lh, hl, hh
        y = rowfilter(hi, g1, mode) + rowfilter(lo, g0, mode)
    return y


class FWD_J1(Function):
    """ Differentiable function doing 1 level forward DTCWT """

    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, o_dim, ri_dim, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(h0, h1)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        ll, highr, highi = fwd_j1(x, h0, h1, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = ll.new_zeros([])
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0, h1 = ctx.saved_tensors
        mode = ctx.mode
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j1(dl, dhr, dhi, h0, h1, o_dim, h_dim, w_dim, mode)
        return dx, None, None, None, None, None, None


def coldfilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    batch, ch, r, c = X.shape
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' + 'X was {}'.format(X.shape))
    if mode == 'symmetric':
        m = ha.shape[2]
        xe = symm_pad(r, m)
        X = torch.cat((X[:, :, (xe[2::2])], X[:, :, (xe[3::2])]), dim=1)
        h = torch.cat((ha.repeat(ch, 1, 1, 1), hb.repeat(ch, 1, 1, 1)), dim=0)
        X = F.conv2d(X, h, stride=(2, 1), groups=ch * 2)
    else:
        raise NotImplementedError()
    if highpass:
        X = torch.stack((X[:, ch:], X[:, :ch]), dim=-2).view(batch, ch, r2, c)
    else:
        X = torch.stack((X[:, :ch], X[:, ch:]), dim=-2).view(batch, ch, r2, c)
    return X


def rowdfilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    batch, ch, r, c = X.shape
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of cols in X must be a multiple of 4\n' + 'X was {}'.format(X.shape))
    if mode == 'symmetric':
        m = ha.shape[2]
        xe = symm_pad(c, m)
        X = torch.cat((X[:, :, :, (xe[2::2])], X[:, :, :, (xe[3::2])]), dim=1)
        h = torch.cat((ha.reshape(1, 1, 1, m).repeat(ch, 1, 1, 1), hb.reshape(1, 1, 1, m).repeat(ch, 1, 1, 1)), dim=0)
        X = F.conv2d(X, h, stride=(1, 2), groups=ch * 2)
    else:
        raise NotImplementedError()
    if highpass:
        Y = torch.stack((X[:, ch:], X[:, :ch]), dim=-1).view(batch, ch, r, c2)
    else:
        Y = torch.stack((X[:, :ch], X[:, ch:]), dim=-1).view(batch, ch, r, c2)
    return Y


def fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode):
    """ Level 2 plus forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = rowdfilt(x, h0b, h0a, False, mode)
        hi = rowdfilt(x, h1b, h1a, True, mode)
        ll = coldfilt(lo, h0b, h0a, False, mode)
        lh = coldfilt(lo, h1b, h1a, True, mode)
        hl = coldfilt(hi, h0b, h0a, False, mode)
        hh = coldfilt(hi, h1b, h1a, True, mode)
        del lo, hi
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = rowdfilt(x, h0b, h0a, False, mode)
        ll = coldfilt(ll, h0b, h0a, False, mode)
        highr = None
        highi = None
    return ll, highr, highi


def colifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    m = ha.shape[2]
    m2 = m // 2
    hao = ha[:, :, 1::2]
    hae = ha[:, :, ::2]
    hbo = hb[:, :, 1::2]
    hbe = hb[:, :, ::2]
    batch, ch, r, c = X.shape
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2.\n' + 'X was {}'.format(X.shape))
    xe = symm_pad(r, m2)
    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:, :, (xe[1:-2:2])], X[:, :, (xe[:-2:2])], X[:, :, (xe[3::2])], X[:, :, (xe[2::2])]), dim=1)
        else:
            X = torch.cat((X[:, :, (xe[:-2:2])], X[:, :, (xe[1:-2:2])], X[:, :, (xe[2::2])], X[:, :, (xe[3::2])]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:, :, (xe[2:-1:2])], X[:, :, (xe[1:-1:2])], X[:, :, (xe[2:-1:2])], X[:, :, (xe[1:-1:2])]), dim=1)
        else:
            X = torch.cat((X[:, :, (xe[1:-1:2])], X[:, :, (xe[2:-1:2])], X[:, :, (xe[1:-1:2])], X[:, :, (xe[2:-1:2])]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1), h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)), dim=0)
    X = F.conv2d(X, h, groups=4 * ch)
    X = torch.stack([X[:, :ch], X[:, ch:2 * ch], X[:, 2 * ch:3 * ch], X[:, 3 * ch:]], dim=3).view(batch, ch, r * 2, c)
    return X


def rowifilt(X, ha, hb, highpass=False, mode='symmetric'):
    if X is None or X.shape == torch.Size([]):
        return torch.zeros(1, 1, 1, 1, device=X.device)
    m = ha.shape[2]
    m2 = m // 2
    hao = ha[:, :, 1::2]
    hae = ha[:, :, ::2]
    hbo = hb[:, :, 1::2]
    hbe = hb[:, :, ::2]
    batch, ch, r, c = X.shape
    if c % 2 != 0:
        raise ValueError('No. of cols in X must be a multiple of 2.\n' + 'X was {}'.format(X.shape))
    xe = symm_pad(c, m2)
    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        if highpass:
            X = torch.cat((X[:, :, :, (xe[1:-2:2])], X[:, :, :, (xe[:-2:2])], X[:, :, :, (xe[3::2])], X[:, :, :, (xe[2::2])]), dim=1)
        else:
            X = torch.cat((X[:, :, :, (xe[:-2:2])], X[:, :, :, (xe[1:-2:2])], X[:, :, :, (xe[2::2])], X[:, :, :, (xe[3::2])]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        if highpass:
            X = torch.cat((X[:, :, :, (xe[2:-1:2])], X[:, :, :, (xe[1:-1:2])], X[:, :, :, (xe[2:-1:2])], X[:, :, :, (xe[1:-1:2])]), dim=1)
        else:
            X = torch.cat((X[:, :, :, (xe[1:-1:2])], X[:, :, :, (xe[2:-1:2])], X[:, :, :, (xe[1:-1:2])], X[:, :, :, (xe[2:-1:2])]), dim=1)
    h = torch.cat((h1.repeat(ch, 1, 1, 1), h2.repeat(ch, 1, 1, 1), h3.repeat(ch, 1, 1, 1), h4.repeat(ch, 1, 1, 1)), dim=0).reshape(4 * ch, 1, 1, m2)
    X = F.conv2d(X, h, groups=4 * ch)
    X = torch.stack([X[:, :ch], X[:, ch:2 * ch], X[:, 2 * ch:3 * ch], X[:, 3 * ch:]], dim=4).view(batch, ch, r, c * 2)
    return X


def inv_j2plus(ll, highr, highi, g0a, g1a, g0b, g1b, o_dim, h_dim, w_dim, mode):
    """ Level2+ inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowifilt(colifilt(ll, g0b, g0a, False, mode), g0b, g0a, False, mode)
    else:
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)
        if ll is None or ll.shape == torch.Size([]):
            hi = colifilt(hh, g1b, g1a, True, mode) + colifilt(hl, g0b, g0a, False, mode)
            lo = colifilt(lh, g1b, g1a, True, mode)
            del lh, hh, hl
        else:
            hi = colifilt(hh, g1b, g1a, True, mode) + colifilt(hl, g0b, g0a, False, mode)
            lo = colifilt(lh, g1b, g1a, True, mode) + colifilt(ll, g0b, g0a, False, mode)
            del lh, hl, hh
        y = rowifilt(hi, g1b, g1a, True, mode) + rowifilt(lo, g0b, g0a, False, mode)
    return y


class FWD_J2PLUS(Function):
    """ Differentiable function doing second level forward DTCWT """

    @staticmethod
    def forward(ctx, x, h0a, h1a, h0b, h1b, skip_hps, o_dim, ri_dim, mode):
        mode = 'symmetric'
        ctx.mode = mode
        ctx.save_for_backward(h0a, h1a, h0b, h1b)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        ll, highr, highi = fwd_j2plus(x, h0a, h1a, h0b, h1b, skip_hps, o_dim, mode)
        if not skip_hps:
            highs = torch.stack((highr, highi), dim=ri_dim)
        else:
            highs = ll.new_zeros([])
        return ll, highs

    @staticmethod
    def backward(ctx, dl, dh):
        h0a, h1a, h0b, h1b = ctx.saved_tensors
        mode = ctx.mode
        h0a, h0b = h0b, h0a
        h1a, h1b = h1b, h1a
        dx = None
        if ctx.needs_input_grad[0]:
            o_dim, ri_dim, h_dim, w_dim = ctx.dims
            if dh is not None and dh.shape != torch.Size([]):
                dhr, dhi = torch.unbind(dh, dim=ri_dim)
            else:
                dhr = dl.new_zeros([])
                dhi = dl.new_zeros([])
            dx = inv_j2plus(dl, dhr, dhi, h0a, h1a, h0b, h1b, o_dim, h_dim, w_dim, mode)
        return dx, None, None, None, None, None, None, None, None


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError('Unkown pad type: {}'.format(mode))


def _as_col_vector(v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def prep_filt(h, c, transpose=False):
    """ Prepares an array to be of the correct format for pytorch.
    Can also specify whether to make it a row filter (set tranpose=True)"""
    h = _as_col_vector(h)[::-1]
    h = h[(None), (None), :]
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0, 1, 3, 2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.get_default_dtype())


class DTCWTForward(nn.Module):
    """ Performs a 2d DTCWT Forward decomposition of an image

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters. Can also
            give a two tuple for the low and highpass filters directly.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters. Can
            also give a 4-tuple for the low tree a, low tree b, high tree a and
            high tree b filters directly.
        J (int): Number of levels of decomposition
        skip_hps (bools): List of bools of length J which specify whether or
            not to calculate the bandpass outputs at the given scale.
            skip_hps[0] is for the first scale. Can be a single bool in which
            case that is applied to all scales.
        include_scale (bool): If true, return the bandpass outputs. Can also be
            a list of length J specifying which lowpasses to return. I.e. if
            [False, True, True], the forward call will return the second and
            third lowpass outputs, but discard the lowpass from the first level
            transform.
        o_dim (int): Which dimension to put the orientations in
        ri_dim (int): which dimension to put the real and imaginary parts
    """

    def __init__(self, biort='near_sym_a', qshift='qshift_a', J=3, skip_hps=False, include_scale=False, o_dim=2, ri_dim=-1, mode='symmetric'):
        super().__init__()
        if o_dim == ri_dim:
            raise ValueError('Orientations and real/imaginary parts must be in different dimensions.')
        self.biort = biort
        self.qshift = qshift
        self.J = J
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        self.mode = mode
        if isinstance(biort, str):
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        else:
            self.h0o = torch.nn.Parameter(prep_filt(biort[0], 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(biort[1], 1), False)
        if isinstance(qshift, str):
            h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)
        else:
            self.h0a = torch.nn.Parameter(prep_filt(qshift[0], 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(qshift[1], 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(qshift[2], 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(qshift[3], 1), False)
        if isinstance(skip_hps, (list, tuple, ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps] * self.J
        if isinstance(include_scale, (list, tuple, ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [include_scale] * self.J

    def forward(self, x):
        """ Forward Dual Tree Complex Wavelet Transform

        Args:
            x (tensor): Input to transform. Should be of shape
                :math:`(N, C_{in}, H_{in}, W_{in})`.

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                If include_scale was true, yl will be a list of lowpass
                coefficients, otherwise will be just the final lowpass
                coefficient of shape :math:`(N, C_{in}, H_{in}', W_{in}')`. Yh
                will be a list of the complex bandpass coefficients of shape
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or similar
                shape depending on o_dim and ri_dim

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` are the shapes of a
            DTCWT pyramid.
        """
        scales = [x.new_zeros([])] * self.J
        highs = [x.new_zeros([])] * self.J
        mode = mode_to_int(self.mode)
        if self.J == 0:
            return x, None
        r, c = x.shape[2:]
        if r % 2 != 0:
            x = torch.cat((x, x[:, :, -1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:, :, :, -1:]), dim=3)
        low, h = FWD_J1.apply(x, self.h0o, self.h1o, self.skip_hps[0], self.o_dim, self.ri_dim, mode)
        highs[0] = h
        if self.include_scale[0]:
            scales[0] = low
        for j in range(1, self.J):
            r, c = low.shape[2:]
            if r % 4 != 0:
                low = torch.cat((low[:, :, 0:1], low, low[:, :, -1:]), dim=2)
            if c % 4 != 0:
                low = torch.cat((low[:, :, :, 0:1], low, low[:, :, :, -1:]), dim=3)
            low, h = FWD_J2PLUS.apply(low, self.h0a, self.h1a, self.h0b, self.h1b, self.skip_hps[j], self.o_dim, self.ri_dim, mode)
            highs[j] = h
            if self.include_scale[j]:
                scales[j] = low
        if True in self.include_scale:
            return scales, highs
        else:
            return low, highs


class INV_J1(Function):
    """ Differentiable function doing 1 level inverse DTCWT """

    @staticmethod
    def forward(ctx, lows, highs, g0, g1, o_dim, ri_dim, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0, g1)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        if highs is not None and highs.shape != torch.Size([]):
            highr, highi = torch.unbind(highs, dim=ri_dim)
        else:
            highr = lows.new_zeros([])
            highi = lows.new_zeros([])
        y = inv_j1(lows, highr, highi, g0, g1, o_dim, h_dim, w_dim, mode)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0, g1 = ctx.saved_tensors
        dl = None
        dh = None
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        mode = ctx.mode
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _, _ = fwd_j1(dy, g0, g1, True, o_dim, mode)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dhr, dhi = fwd_j1(dy, g0, g1, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dhr, dhi = fwd_j1(dy, g0, g1, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        return dl, dh, None, None, None, None, None


class INV_J2PLUS(Function):
    """ Differentiable function doing level 2 onwards inverse DTCWT """

    @staticmethod
    def forward(ctx, lows, highs, g0a, g1a, g0b, g1b, o_dim, ri_dim, mode):
        mode = 'symmetric'
        ctx.mode = mode
        ctx.save_for_backward(g0a, g1a, g0b, g1b)
        ctx.dims = get_dimensions5(o_dim, ri_dim)
        o_dim, ri_dim, h_dim, w_dim = ctx.dims
        if highs is not None and highs.shape != torch.Size([]):
            highr, highi = torch.unbind(highs, dim=ri_dim)
        else:
            highr = lows.new_zeros([])
            highi = lows.new_zeros([])
        y = inv_j2plus(lows, highr, highi, g0a, g1a, g0b, g1b, o_dim, h_dim, w_dim, mode)
        return y

    @staticmethod
    def backward(ctx, dy):
        g0a, g1a, g0b, g1b = ctx.saved_tensors
        g0a, g0b = g0b, g0a
        g1a, g1b = g1b, g1a
        o_dim, ri_dim = ctx.dims[0], ctx.dims[1]
        mode = ctx.mode
        dl = None
        dh = None
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            dl, _, _ = fwd_j2plus(dy, g0a, g1a, g0b, g1b, True, o_dim, mode)
        elif ctx.needs_input_grad[1] and not ctx.needs_input_grad[0]:
            _, dhr, dhi = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            dl, dhr, dhi = fwd_j2plus(dy, g0a, g1a, g0b, g1b, False, o_dim, mode)
            dh = torch.stack((dhr, dhi), dim=ri_dim)
        return dl, dh, None, None, None, None, None, None, None


def get_dimensions6(o_dim, ri_dim):
    """ Get the orientation, real/imag, height and width dimensions
    for the full tensor (6 dimensions)."""
    o_dim = o_dim % 6
    ri_dim = ri_dim % 6
    if ri_dim < o_dim:
        o_dim -= 1
    if o_dim >= 3 and ri_dim >= 3:
        h_dim = 2
    elif o_dim >= 4 or ri_dim >= 4:
        h_dim = 3
    else:
        h_dim = 4
    if o_dim >= 4 and ri_dim >= 4:
        w_dim = 3
    elif o_dim >= 4 or ri_dim >= 4:
        w_dim = 4
    else:
        w_dim = 5
    return o_dim, ri_dim, h_dim, w_dim


class DTCWTInverse(nn.Module):
    """ 2d DTCWT Inverse

    Args:
        biort (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first level biorthogonal wavelet filters. Can also
            give a two tuple for the low and highpass filters directly.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second level quarter shift filters. Can
            also give a 4-tuple for the low tree a, low tree b, high tree a and
            high tree b filters directly.
        J (int): Number of levels of decomposition.
        o_dim (int):which dimension the orientations are in
        ri_dim (int): which dimension to put th real and imaginary parts in
    """

    def __init__(self, biort='near_sym_a', qshift='qshift_a', o_dim=2, ri_dim=-1, mode='symmetric'):
        super().__init__()
        self.biort = biort
        self.qshift = qshift
        self.o_dim = o_dim
        self.ri_dim = ri_dim
        self.mode = mode
        if isinstance(biort, str):
            _, g0o, _, g1o = _biort(biort)
            self.g0o = torch.nn.Parameter(prep_filt(g0o, 1), False)
            self.g1o = torch.nn.Parameter(prep_filt(g1o, 1), False)
        else:
            self.g0o = torch.nn.Parameter(prep_filt(biort[0], 1), False)
            self.g1o = torch.nn.Parameter(prep_filt(biort[1], 1), False)
        if isinstance(qshift, str):
            _, _, g0a, g0b, _, _, g1a, g1b = _qshift(qshift)
            self.g0a = torch.nn.Parameter(prep_filt(g0a, 1), False)
            self.g0b = torch.nn.Parameter(prep_filt(g0b, 1), False)
            self.g1a = torch.nn.Parameter(prep_filt(g1a, 1), False)
            self.g1b = torch.nn.Parameter(prep_filt(g1b, 1), False)
        else:
            self.g0a = torch.nn.Parameter(prep_filt(qshift[0], 1), False)
            self.g0b = torch.nn.Parameter(prep_filt(qshift[1], 1), False)
            self.g1a = torch.nn.Parameter(prep_filt(qshift[2], 1), False)
            self.g1b = torch.nn.Parameter(prep_filt(qshift[3], 1), False)

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
                yl is a tensor of shape :math:`(N, C_{in}, H_{in}', W_{in}')`
                and yh is a list of  the complex bandpass coefficients of shape
                :math:`list(N, C_{in}, 6, H_{in}'', W_{in}'', 2)`, or similar
                depending on o_dim and ri_dim

        Returns:
            Reconstructed output

        Note:
            Can accept Nones or an empty tensor (torch.tensor([])) for the
            lowpass or bandpass inputs. In this cases, an array of zeros
            replaces that input.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` are the shapes of a
            DTCWT pyramid.

        Note:
            If include_scale was true for the forward pass, you should provide
            only the final lowpass output here, as normal for an inverse wavelet
            transform.
        """
        low, highs = coeffs
        J = len(highs)
        mode = mode_to_int(self.mode)
        _, _, h_dim, w_dim = get_dimensions6(self.o_dim, self.ri_dim)
        for j, s in zip(range(J - 1, 0, -1), highs[1:][::-1]):
            if s is not None and s.shape != torch.Size([]):
                assert s.shape[self.o_dim] == 6, 'Inverse transform must have input with 6 orientations'
                assert len(s.shape) == 6, 'Bandpass inputs must have 6 dimensions'
                assert s.shape[self.ri_dim] == 2, 'Inputs must be complex with real and imaginary parts in the ri dimension'
                r, c = low.shape[2:]
                r1, c1 = s.shape[h_dim], s.shape[w_dim]
                if r != r1 * 2:
                    low = low[:, :, 1:-1]
                if c != c1 * 2:
                    low = low[:, :, :, 1:-1]
            low = INV_J2PLUS.apply(low, s, self.g0a, self.g1a, self.g0b, self.g1b, self.o_dim, self.ri_dim, mode)
        if highs[0] is not None and highs[0].shape != torch.Size([]):
            r, c = low.shape[2:]
            r1, c1 = highs[0].shape[h_dim], highs[0].shape[w_dim]
            if r != r1 * 2:
                low = low[:, :, 1:-1]
            if c != c1 * 2:
                low = low[:, :, :, 1:-1]
        low = INV_J1.apply(low, highs[0], self.g0o, self.g1o, self.o_dim, self.ri_dim, mode)
        return low


class SWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """

    def __init__(self, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 2:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 4:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = wave[2], wave[3]
        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.g0_col = nn.Parameter(filts[0], requires_grad=False)
            self.g1_col = nn.Parameter(filts[1], requires_grad=False)
            self.g0_row = nn.Parameter(filts[2], requires_grad=False)
            self.g1_row = nn.Parameter(filts[3], requires_grad=False)
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row, g1_row)
            self.h = nn.Parameter(filts, requires_grad=False)
        self.mode = mode
        self.separable = separable

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[(...), :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[(...), :-1]
            if self.separable:
                lh, hl, hh = torch.unbind(h, dim=2)
                filts = self.g0_col, self.g1_col, self.g0_row, self.g1_row
                ll = lowlevel.sfb2d(ll, lh, hl, hh, filts, mode=self.mode)
            else:
                c = torch.cat((ll[:, :, (None)], h), dim=2)
                ll = lowlevel.sfb2d_nonsep(c, self.h, mode=self.mode)
        return ll


class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """

    def __init__(self, J=1, wave='db1', mode='periodization'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 2:
            h0_col, h1_col = wave[0], wave[1]
            h0_row, h1_row = h0_col, h1_col
        elif len(wave) == 4:
            h0_col, h1_col = wave[0], wave[1]
            h0_row, h1_row = wave[2], wave[3]
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        ll = x
        coeffs = []
        filts = self.h0_col, self.h1_col, self.h0_row, self.h1_row
        for j in range(self.J):
            y = lowlevel.afb2d_atrous(ll, filts, self.mode, 2 ** j)
            coeffs.append(y)
            ll = y[:, :, (0)]
        return coeffs


class ScatLayerj1_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o, mode, bias, combine_colour):
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 2 == c % 2 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour
        ll, reals, imags = fwd_j1(x, h0o, h1o, False, 1, mode)
        ll = F.avg_pool2d(ll, 2)
        if combine_colour:
            r = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            r = r[:, :, (None)]
        else:
            r = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
        if x.requires_grad:
            drdx = reals / r
            drdy = imags / r
            ctx.save_for_backward(h0o, h1o, drdx, drdy)
        else:
            z = x.new_zeros(1)
            ctx.save_for_backward(h0o, h1o, z, z)
        r = r - bias
        del reals, imags
        if combine_colour:
            Z = torch.cat((ll, r[:, :, (0)]), dim=1)
        else:
            Z = torch.cat((ll[:, (None)], r), dim=1)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode
        if ctx.needs_input_grad[0]:
            h0o, h1o, drdx, drdy = ctx.saved_tensors
            h0o_t = h0o
            h1o_t = h1o
            if ctx.combine_colour:
                dYl, dr = dZ[:, :3], dZ[:, 3:]
                dr = dr[:, :, (None)]
            else:
                dYl, dr = dZ[:, (0)], dZ[:, 1:]
            ll = 1 / 4 * F.interpolate(dYl, scale_factor=2, mode='nearest')
            reals = dr * drdx
            imags = dr * drdy
            dX = inv_j1(ll, reals, imags, h0o_t, h1o_t, 1, 3, 4, mode)
        return (dX,) + (None,) * 5


def fwd_j1_rot(x, h0, h1, h2, skip_hps, o_dim, mode):
    """ Level 1 forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = rowfilter(x, h0, mode)
        hi = rowfilter(x, h1, mode)
        ba = rowfilter(x, h2, mode)
        lh = colfilter(lo, h1, mode)
        hl = colfilter(hi, h0, mode)
        hh = colfilter(ba, h2, mode)
        ll = colfilter(lo, h0, mode)
        del lo, hi, ba
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = rowfilter(x, h0, mode)
        ll = colfilter(ll, h0, mode)
        highr = x.new_zeros([])
        highi = x.new_zeros([])
    return ll, highr, highi


def inv_j1_rot(ll, highr, highi, g0, g1, g2, o_dim, h_dim, w_dim, mode):
    """ Level1 inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowfilter(colfilter(ll, g0), g0)
    else:
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)
        if ll is None or ll.shape == torch.Size([]):
            lo = colfilter(lh, g1, mode)
            hi = colfilter(hl, g0, mode)
            ba = colfilter(hh, g2, mode)
            del lh, hh, hl
        else:
            r, c = ll.shape[2:]
            r1, c1 = highr.shape[h_dim], highr.shape[w_dim]
            if r != r1 * 2:
                ll = ll[:, :, 1:-1]
            if c != c1 * 2:
                ll = ll[:, :, :, 1:-1]
            lo = colfilter(lh, g1, mode) + colfilter(ll, g0, mode)
            hi = colfilter(hl, g0, mode)
            ba = colfilter(hh, g2, mode)
            del lh, hl, hh
        y = rowfilter(hi, g1, mode) + rowfilter(lo, g0, mode) + rowfilter(ba, g2, mode)
    return y


class ScatLayerj1_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. Uses the rotationally symmetric
    filters, i.e. a slightly more expensive operation."""

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o, mode, bias, combine_colour):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.in_shape = x.shape
        ctx.combine_colour = combine_colour
        batch, ch, r, c = x.shape
        assert r % 2 == c % 2 == 0
        ll, reals, imags = fwd_j1_rot(x, h0o, h1o, h2o, False, 1, mode)
        ll = F.avg_pool2d(ll, 2)
        if combine_colour:
            r = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            r = r[:, :, (None)]
        else:
            r = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
        if x.requires_grad:
            drdx = reals / r
            drdy = imags / r
            ctx.save_for_backward(h0o, h1o, h2o, drdx, drdy)
        else:
            z = x.new_zeros(1)
            ctx.save_for_backward(h0o, h1o, h2o, z, z)
        r = r - bias
        del reals, imags
        if combine_colour:
            Z = torch.cat((ll, r[:, :, (0)]), dim=1)
        else:
            Z = torch.cat((ll[:, (None)], r), dim=1)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode
        if ctx.needs_input_grad[0]:
            h0o, h1o, h2o, drdx, drdy = ctx.saved_tensors
            if ctx.combine_colour:
                dYl, dr = dZ[:, :3], dZ[:, 3:]
                dr = dr[:, :, (None)]
            else:
                dYl, dr = dZ[:, (0)], dZ[:, 1:]
            ll = 1 / 4 * F.interpolate(dYl, scale_factor=2, mode='nearest')
            reals = dr * drdx
            imags = dr * drdy
            dX = inv_j1_rot(ll, reals, imags, h0o, h1o, h2o, 1, 3, 4, mode)
        return (dX,) + (None,) * 6


class ScatLayer(nn.Module):
    """ Does one order of scattering at a single scale. Can be made into a
    second order scatternet by stacking two of these layers.
    Inputs:
        biort (str): the biorthogonal filters to use. if 'near_sym_b_bp' will
            use the rotationally symmetric filters. These have 13 and 19 taps
            so are quite long. They also require 7 1D convolutions instead of 6.
        x (torch.tensor): Input of shape (N, C, H, W)
        mode (str): padding mode. Can be 'symmetric' or 'zero'
        magbias (float): the magnitude bias to use for smoothing
        combine_colour (bool): if true, will only have colour lowpass and have
            greyscale bandpass
    Returns:
        y (torch.tensor): y has the lowpass and invariant U terms stacked along
            the channel dimension, and so has shape (N, 7*C, H/2, W/2). Where
            the first C channels are the lowpass outputs, and the next 6C are
            the magnitude highpass outputs.
    """

    def __init__(self, biort='near_sym_a', mode='symmetric', magbias=0.01, combine_colour=False):
        super().__init__()
        self.biort = biort
        self.mode_str = mode
        self.mode = mode_to_int(mode)
        self.magbias = magbias
        self.combine_colour = combine_colour
        if biort == 'near_sym_b_bp':
            self.bandpass_diag = True
            h0o, _, h1o, _, h2o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            self.h2o = torch.nn.Parameter(prep_filt(h2o, 1), False)
        else:
            self.bandpass_diag = False
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)

    def forward(self, x):
        _, ch, r, c = x.shape
        if r % 2 != 0:
            x = torch.cat((x, x[:, :, -1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:, :, :, -1:]), dim=3)
        if self.combine_colour:
            assert ch == 3
        if self.bandpass_diag:
            Z = ScatLayerj1_rot_f.apply(x, self.h0o, self.h1o, self.h2o, self.mode, self.magbias, self.combine_colour)
        else:
            Z = ScatLayerj1_f.apply(x, self.h0o, self.h1o, self.mode, self.magbias, self.combine_colour)
        if not self.combine_colour:
            b, _, c, h, w = Z.shape
            Z = Z.view(b, 7 * c, h, w)
        return Z

    def extra_repr(self):
        return "biort='{}', mode='{}', magbias={}".format(self.biort, self.mode_str, self.magbias)


class ScatLayerj2_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o, h0a, h0b, h1a, h1b, mode, bias, combine_colour):
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 8 == c % 8 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour
        s0, reals, imags = fwd_j1(x, h0o, h1o, False, 1, mode)
        if combine_colour:
            s1_j1 = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            s1_j1 = s1_j1[:, :, (None)]
            if x.requires_grad:
                dsdx1 = reals / s1_j1
                dsdy1 = imags / s1_j1
            s1_j1 = s1_j1 - bias
            s0, reals, imags = fwd_j2plus(s0, h0a, h1a, h0b, h1b, False, 1, mode)
            s1_j2 = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            s1_j2 = s1_j2[:, :, (None)]
            if x.requires_grad:
                dsdx2 = reals / s1_j2
                dsdy2 = imags / s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)
            s1_j1 = s1_j1[:, :, (0)]
            s1_j1, reals, imags = fwd_j1(s1_j1, h0o, h1o, False, 1, mode)
            s2_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2_1 = reals / s2_j1
                dsdy2_1 = imags / s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b, z, z, z, z, z, z)
            del reals, imags
            Z = torch.cat((s0, s1_j1, s1_j2[:, :, (0)], s2_j1), dim=1)
        else:
            s1_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx1 = reals / s1_j1
                dsdy1 = imags / s1_j1
            s1_j1 = s1_j1 - bias
            s0, reals, imags = fwd_j2plus(s0, h0a, h1a, h0b, h1b, False, 1, mode)
            s1_j2 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2 = reals / s1_j2
                dsdy2 = imags / s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)
            p = s1_j1.shape
            s1_j1 = s1_j1.view(p[0], 6 * p[2], p[3], p[4])
            s1_j1, reals, imags = fwd_j1(s1_j1, h0o, h1o, False, 1, mode)
            s2_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2_1 = reals / s2_j1
                dsdy2_1 = imags / s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[2] // 6, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            s1_j1 = s1_j1.view(p[0], 6, p[2], p[3] // 2, p[4] // 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b, z, z, z, z, z, z)
            del reals, imags
            Z = torch.cat((s0[:, (None)], s1_j1, s1_j2, s2_j1), dim=1)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode
        if ctx.needs_input_grad[0]:
            o_dim = 1
            h_dim = 3
            w_dim = 4
            h0o, h1o, h0a, h0b, h1a, h1b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1 = ctx.saved_tensors
            h0o_t = h0o
            h1o_t = h1o
            h0a_t = h0b
            h0b_t = h0a
            h1a_t = h1b
            h1b_t = h1a
            if ctx.combine_colour:
                ds0, ds1_j1, ds1_j2, ds2_j1 = dZ[:, :3], dZ[:, 3:9], dZ[:, 9:15], dZ[:, 15:]
                ds1_j2 = ds1_j2[:, :, (None)]
                ds1_j1 = 1 / 4 * F.interpolate(ds1_j1, scale_factor=2, mode='nearest')
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, 6, q[2], q[3])
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1(ds1_j1, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1[:, :, (None)]
                ds0 = 1 / 4 * F.interpolate(ds0, scale_factor=2, mode='nearest')
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus(ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, o_dim, h_dim, w_dim, mode)
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1(ds0, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
            else:
                ds0, ds1_j1, ds1_j2, ds2_j1 = dZ[:, (0)], dZ[:, 1:7], dZ[:, 7:13], dZ[:, 13:]
                p = ds1_j1.shape
                ds1_j1 = ds1_j1.view(p[0], p[2] * 6, p[3], p[4])
                ds1_j1 = 1 / 4 * F.interpolate(ds1_j1, scale_factor=2, mode='nearest')
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, q[2] * 6, q[3], q[4])
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1(ds1_j1, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1.view(p[0], 6, p[2], p[3] * 2, p[4] * 2)
                ds0 = 1 / 4 * F.interpolate(ds0, scale_factor=2, mode='nearest')
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus(ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, o_dim, h_dim, w_dim, mode)
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1(ds0, reals, imags, h0o_t, h1o_t, o_dim, h_dim, w_dim, mode)
        return (dX,) + (None,) * 9


def fwd_j2plus_rot(x, h0a, h1a, h0b, h1b, h2a, h2b, skip_hps, o_dim, mode):
    """ Level 2 plus forward dtcwt.

    Have it as a separate function as can be used by
    the forward pass of the forward transform and the backward pass of the
    inverse transform.
    """
    if not skip_hps:
        lo = rowdfilt(x, h0b, h0a, False, mode)
        hi = rowdfilt(x, h1b, h1a, True, mode)
        ba = rowdfilt(x, h2b, h2a, True, mode)
        lh = coldfilt(lo, h1b, h1a, True, mode)
        hl = coldfilt(hi, h0b, h0a, False, mode)
        hh = coldfilt(ba, h2b, h2a, True, mode)
        ll = coldfilt(lo, h0b, h0a, False, mode)
        del lo, hi, ba
        highr, highi = highs_to_orientations(lh, hl, hh, o_dim)
    else:
        ll = rowdfilt(x, h0b, h0a, False, mode)
        ll = coldfilt(ll, h0b, h0a, False, mode)
        highr = None
        highi = None
    return ll, highr, highi


def inv_j2plus_rot(ll, highr, highi, g0a, g1a, g0b, g1b, g2a, g2b, o_dim, h_dim, w_dim, mode):
    """ Level2+ inverse dtcwt.

    Have it as a separate function as can be used by the forward pass of the
    inverse transform and the backward pass of the forward transform.
    """
    if highr is None or highr.shape == torch.Size([]):
        y = rowifilt(colifilt(ll, g0b, g0a, False, mode), g0b, g0a, False, mode)
    else:
        lh, hl, hh = orientations_to_highs(highr, highi, o_dim)
        if ll is None or ll.shape == torch.Size([]):
            lo = colifilt(lh, g1b, g1a, True, mode)
            hi = colifilt(hl, g0b, g0a, False, mode)
            ba = colifilt(hh, g2b, g2a, True, mode)
            del lh, hh, hl
        else:
            lo = colifilt(lh, g1b, g1a, True, mode) + colifilt(ll, g0b, g0a, False, mode)
            hi = colifilt(hl, g0b, g0a, False, mode)
            ba = colifilt(hh, g2b, g2a, True, mode)
            del lh, hl, hh
        y = rowifilt(hi, g1b, g1a, True, mode) + rowifilt(lo, g0b, g0a, False, mode) + rowifilt(ba, g2b, g2a, True, mode)
    return y


class ScatLayerj2_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT bandpass biorthogonal and qshift filters . """

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, mode, bias, combine_colour):
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        assert r % 8 == c % 8 == 0
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.combine_colour = combine_colour
        s0, reals, imags = fwd_j1_rot(x, h0o, h1o, h2o, False, 1, mode)
        if combine_colour:
            s1_j1 = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            s1_j1 = s1_j1[:, :, (None)]
            if x.requires_grad:
                dsdx1 = reals / s1_j1
                dsdy1 = imags / s1_j1
            s1_j1 = s1_j1 - bias
            s0, reals, imags = fwd_j2plus_rot(s0, h0a, h1a, h0b, h1b, h2a, h2b, False, 1, mode)
            s1_j2 = torch.sqrt(reals[:, :, (0)] ** 2 + imags[:, :, (0)] ** 2 + reals[:, :, (1)] ** 2 + imags[:, :, (1)] ** 2 + reals[:, :, (2)] ** 2 + imags[:, :, (2)] ** 2 + bias ** 2)
            s1_j2 = s1_j2[:, :, (None)]
            if x.requires_grad:
                dsdx2 = reals / s1_j2
                dsdy2 = imags / s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)
            s1_j1 = s1_j1[:, :, (0)]
            s1_j1, reals, imags = fwd_j1_rot(s1_j1, h0o, h1o, h2o, False, 1, mode)
            s2_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2_1 = reals / s2_j1
                dsdy2_1 = imags / s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, z, z, z, z, z, z)
            del reals, imags
            Z = torch.cat((s0, s1_j1, s1_j2[:, :, (0)], s2_j1), dim=1)
        else:
            s1_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx1 = reals / s1_j1
                dsdy1 = imags / s1_j1
            s1_j1 = s1_j1 - bias
            s0, reals, imags = fwd_j2plus_rot(s0, h0a, h1a, h0b, h1b, h2a, h2b, False, 1, mode)
            s1_j2 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2 = reals / s1_j2
                dsdy2 = imags / s1_j2
            s1_j2 = s1_j2 - bias
            s0 = F.avg_pool2d(s0, 2)
            p = s1_j1.shape
            s1_j1 = s1_j1.view(p[0], 6 * p[2], p[3], p[4])
            s1_j1, reals, imags = fwd_j1_rot(s1_j1, h0o, h1o, h2o, False, 1, mode)
            s2_j1 = torch.sqrt(reals ** 2 + imags ** 2 + bias ** 2)
            if x.requires_grad:
                dsdx2_1 = reals / s2_j1
                dsdy2_1 = imags / s2_j1
            q = s2_j1.shape
            s2_j1 = s2_j1.view(q[0], 36, q[2] // 6, q[3], q[4])
            s2_j1 = s2_j1 - bias
            s1_j1 = F.avg_pool2d(s1_j1, 2)
            s1_j1 = s1_j1.view(p[0], 6, p[2], p[3] // 2, p[4] // 2)
            if x.requires_grad:
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1)
            else:
                z = x.new_zeros(1)
                ctx.save_for_backward(h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, z, z, z, z, z, z)
            del reals, imags
            Z = torch.cat((s0[:, (None)], s1_j1, s1_j2, s2_j1), dim=1)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode
        if ctx.needs_input_grad[0]:
            o_dim = 1
            h_dim = 3
            w_dim = 4
            h0o, h1o, h2o, h0a, h0b, h1a, h1b, h2a, h2b, dsdx1, dsdy1, dsdx2, dsdy2, dsdx2_1, dsdy2_1 = ctx.saved_tensors
            h0o_t = h0o
            h1o_t = h1o
            h2o_t = h2o
            h0a_t = h0b
            h0b_t = h0a
            h1a_t = h1b
            h1b_t = h1a
            h2a_t = h2b
            h2b_t = h2a
            if ctx.combine_colour:
                ds0, ds1_j1, ds1_j2, ds2_j1 = dZ[:, :3], dZ[:, 3:9], dZ[:, 9:15], dZ[:, 15:]
                ds1_j2 = ds1_j2[:, :, (None)]
                ds1_j1 = 1 / 4 * F.interpolate(ds1_j1, scale_factor=2, mode='nearest')
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, 6, q[2], q[3])
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1_rot(ds1_j1, reals, imags, h0o_t, h1o_t, h2o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1[:, :, (None)]
                ds0 = 1 / 4 * F.interpolate(ds0, scale_factor=2, mode='nearest')
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus_rot(ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, h2a_t, h2b_t, o_dim, h_dim, w_dim, mode)
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1_rot(ds0, reals, imags, h0o_t, h1o_t, h2o_t, o_dim, h_dim, w_dim, mode)
            else:
                ds0, ds1_j1, ds1_j2, ds2_j1 = dZ[:, (0)], dZ[:, 1:7], dZ[:, 7:13], dZ[:, 13:]
                p = ds1_j1.shape
                ds1_j1 = ds1_j1.view(p[0], p[2] * 6, p[3], p[4])
                ds1_j1 = 1 / 4 * F.interpolate(ds1_j1, scale_factor=2, mode='nearest')
                q = ds2_j1.shape
                ds2_j1 = ds2_j1.view(q[0], 6, q[2] * 6, q[3], q[4])
                reals = ds2_j1 * dsdx2_1
                imags = ds2_j1 * dsdy2_1
                ds1_j1 = inv_j1_rot(ds1_j1, reals, imags, h0o_t, h1o_t, h2o_t, o_dim, h_dim, w_dim, mode)
                ds1_j1 = ds1_j1.view(p[0], 6, p[2], p[3] * 2, p[4] * 2)
                ds0 = 1 / 4 * F.interpolate(ds0, scale_factor=2, mode='nearest')
                reals = ds1_j2 * dsdx2
                imags = ds1_j2 * dsdy2
                ds0 = inv_j2plus_rot(ds0, reals, imags, h0a_t, h1a_t, h0b_t, h1b_t, h2a_t, h2b_t, o_dim, h_dim, w_dim, mode)
                reals = ds1_j1 * dsdx1
                imags = ds1_j1 * dsdy1
                dX = inv_j1_rot(ds0, reals, imags, h0o_t, h1o_t, h2o_t, o_dim, h_dim, w_dim, mode)
        return (dX,) + (None,) * 12


class ScatLayerj2(nn.Module):
    """ Does second order scattering for two scales. Uses correct dtcwt first
    and second level filters compared to ScatLayer which only uses biorthogonal
    filters.

    Inputs:
        biort (str): the biorthogonal filters to use. if 'near_sym_b_bp' will
            use the rotationally symmetric filters. These have 13 and 19 taps
            so are quite long. They also require 7 1D convolutions instead of 6.
        x (torch.tensor): Input of shape (N, C, H, W)
        mode (str): padding mode. Can be 'symmetric' or 'zero'
    Returns:
        y (torch.tensor): y has the lowpass and invariant U terms stacked along
            the channel dimension, and so has shape (N, 7*C, H/2, W/2). Where
            the first C channels are the lowpass outputs, and the next 6C are
            the magnitude highpass outputs.
    """

    def __init__(self, biort='near_sym_a', qshift='qshift_a', mode='symmetric', magbias=0.01, combine_colour=False):
        super().__init__()
        self.biort = biort
        self.qshift = biort
        self.mode_str = mode
        self.mode = mode_to_int(mode)
        self.magbias = magbias
        self.combine_colour = combine_colour
        if biort == 'near_sym_b_bp':
            assert qshift == 'qshift_b_bp'
            self.bandpass_diag = True
            h0o, _, h1o, _, h2o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            self.h2o = torch.nn.Parameter(prep_filt(h2o, 1), False)
            h0a, h0b, _, _, h1a, h1b, _, _, h2a, h2b, _, _ = _qshift('qshift_b_bp')
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)
            self.h2a = torch.nn.Parameter(prep_filt(h2a, 1), False)
            self.h2b = torch.nn.Parameter(prep_filt(h2b, 1), False)
        else:
            self.bandpass_diag = False
            h0o, _, h1o, _ = _biort(biort)
            self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
            self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
            h0a, h0b, _, _, h1a, h1b, _, _ = _qshift(qshift)
            self.h0a = torch.nn.Parameter(prep_filt(h0a, 1), False)
            self.h0b = torch.nn.Parameter(prep_filt(h0b, 1), False)
            self.h1a = torch.nn.Parameter(prep_filt(h1a, 1), False)
            self.h1b = torch.nn.Parameter(prep_filt(h1b, 1), False)

    def forward(self, x):
        ch, r, c = x.shape[1:]
        rem = r % 8
        if rem != 0:
            rows_after = (9 - rem) // 2
            rows_before = (8 - rem) // 2
            x = torch.cat((x[:, :, :rows_before], x, x[:, :, -rows_after:]), dim=2)
        rem = c % 8
        if rem != 0:
            cols_after = (9 - rem) // 2
            cols_before = (8 - rem) // 2
            x = torch.cat((x[:, :, :, :cols_before], x, x[:, :, :, -cols_after:]), dim=3)
        if self.combine_colour:
            assert ch == 3
        if self.bandpass_diag:
            pass
            Z = ScatLayerj2_rot_f.apply(x, self.h0o, self.h1o, self.h2o, self.h0a, self.h0b, self.h1a, self.h1b, self.h2a, self.h2b, self.mode, self.magbias, self.combine_colour)
        else:
            Z = ScatLayerj2_f.apply(x, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a, self.h1b, self.mode, self.magbias, self.combine_colour)
        if not self.combine_colour:
            b, _, c, h, w = Z.shape
            Z = Z.view(b, 49 * c, h, w)
        return Z

    def extra_repr(self):
        return "biort='{}', mode='{}', magbias={}".format(self.biort, self.mode_str, self.magbias)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.xfm = DTCWTForward(J=3, C=3)
        self.ifm = DTCWTInverse(J=3, C=3)
        self.sparsify = SparsifyWaveCoeffs2(3, 3)

    def forward(self, x):
        coeffs = self.xfm(x)
        coeffs = self.sparsify(coeffs)
        y = self.ifm(coeffs)
        return y

