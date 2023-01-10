import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
openunmix = _module
cli = _module
data = _module
evaluate = _module
filtering = _module
model = _module
predict = _module
transforms = _module
utils = _module
train = _module
setup = _module
tests = _module
create_vectors = _module
test_augmentations = _module
test_datasets = _module
test_io = _module
test_jit = _module
test_model = _module
test_regression = _module
test_transforms = _module
test_utils = _module
test_wiener = _module

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


import torch.hub


import torch


import torchaudio


import numpy as np


import random


from typing import Optional


from typing import Union


from typing import Tuple


from typing import List


from typing import Any


from typing import Callable


import torch.utils.data


import functools


import torch.nn as nn


from torch import Tensor


from torch.utils.data import DataLoader


from typing import Mapping


import torch.nn.functional as F


from torch.nn import LSTM


from torch.nn import BatchNorm1d


from torch.nn import Linear


from torch.nn import Parameter


import warnings


import time


import sklearn.preprocessing


import copy


import torch.onnx


from torch.testing._internal.jit_utils import JitTestCase


class OpenUnmix(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(self, nb_bins: int=4096, nb_channels: int=2, hidden_size: int=512, nb_layers: int=3, unidirectional: bool=False, input_mean: Optional[np.ndarray]=None, input_scale: Optional[np.ndarray]=None, max_bin: Optional[int]=None):
        super(OpenUnmix, self).__init__()
        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        self.hidden_size = hidden_size
        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)
        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2
        self.lstm = LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=nb_layers, bidirectional=not unidirectional, batch_first=False, dropout=0.4 if nb_layers > 1 else 0)
        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)
        self.fc3 = Linear(in_features=hidden_size, out_features=self.nb_output_bins * nb_channels, bias=False)
        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[:self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)
        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[:self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)
        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)
        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) ->Tensor:
        """
        Args:
            x: input spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        mix = x.detach().clone()
        x = x[..., :self.nb_bins]
        x = x + self.input_mean
        x = x * self.input_scale
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)
        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x *= self.output_scale
        x += self.output_mean
        x = F.relu(x) * mix
        return x.permute(1, 2, 3, 0)


class ComplexNorm(nn.Module):
    """Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, mono: bool=False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec: Tensor) ->Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        spec = torch.abs(torch.view_as_complex(spec))
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)
        return spec


class AsteroidISTFT(nn.Module):

    def __init__(self, fb):
        super(AsteroidISTFT, self).__init__()
        self.dec = Decoder(fb)

    def forward(self, X: Tensor, length: Optional[int]=None) ->Tensor:
        aux = from_torchaudio(X)
        return self.dec(aux, length=length)


class AsteroidSTFT(nn.Module):

    def __init__(self, fb):
        super(AsteroidSTFT, self).__init__()
        self.enc = Encoder(fb)

    def forward(self, x):
        aux = self.enc(x)
        return to_torchaudio(aux)


class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """

    def __init__(self, n_fft: int=4096, n_hop: int=1024, center: bool=False, sample_rate: float=44100.0, window: Optional[nn.Parameter]=None) ->None:
        super(TorchISTFT, self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

    def forward(self, X: Tensor, length: Optional[int]=None) ->Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        y = torch.istft(torch.view_as_complex(X), n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=self.center, normalized=False, onesided=True, length=length)
        y = y.reshape(shape[:-3] + y.shape[-1:])
        return y


class TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform
    uses hard coded hann_window.
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        window (nn.Parameter, optional): window function
    """

    def __init__(self, n_fft: int=4096, n_hop: int=1024, center: bool=False, window: Optional[nn.Parameter]=None):
        super(TorchSTFT, self).__init__()
        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x: Tensor) ->Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """
        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape
        x = x.view(-1, shape[-1])
        complex_stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=self.center, normalized=False, onesided=True, pad_mode='reflect', return_complex=True)
        stft_f = torch.view_as_real(complex_stft)
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
        return stft_f


def make_filterbanks(n_fft=4096, n_hop=1024, center=False, sample_rate=44100.0, method='torch'):
    window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
    if method == 'torch':
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == 'asteroid':
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(n_fft=n_fft, hop_length=n_hop, win_length=n_fft, window=window, center=center, sample_rate=sample_rate)
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return encoder, decoder


def _norm(x: torch.Tensor) ->torch.Tensor:
    """Computes the norm value of a torch Tensor, assuming that it
    comes as real and imaginary part in its last dimension.

    Args:
        x (Tensor): Input Tensor of shape [shape=(..., 2)]

    Returns:
        Tensor: shape as x excluding the last dimension.
    """
    return torch.abs(x[..., 0]) ** 2 + torch.abs(x[..., 1]) ** 2


def atan2(y, x):
    """Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


def _conj(z, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts.
    can work in place in case out is z"""
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]
    return out


def _mul_add(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""
    target_shape = torch.Size([max(sa, sb) for sa, sb in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out


def _covariance(y_j):
    """
    Compute the empirical covariance for a source.

    Args:
        y_j (Tensor): complex stft of the source.
            [shape=(nb_frames, nb_bins, nb_channels, 2)].

    Returns:
        Cj (Tensor): [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
            just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    nb_frames, nb_bins, nb_channels = y_j.shape[:-1]
    Cj = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels, 2), dtype=y_j.dtype, device=y_j.device)
    indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels))
    for index in indices:
        Cj[:, :, index[0], index[1], :] = _mul_add(y_j[:, :, index[0], :], _conj(y_j[:, :, index[1], :]), Cj[:, :, index[0], index[1], :])
    return Cj


def _inv(z: torch.Tensor, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts.
    can work in place in case out is z"""
    ez = _norm(z)
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez
    return out


def _mul(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts
    can work in place in case out is a only"""
    target_shape = torch.Size([max(sa, sb) for sa, sb in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return out


def _invert(M: torch.Tensor, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Invert 1x1 or 2x2 matrices

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Args:
        M (Tensor): [shape=(..., nb_channels, nb_channels, 2)]
            matrices to invert: must be square along dimensions -3 and -2

    Returns:
        invM (Tensor): [shape=M.shape]
            inverses of M
    """
    nb_channels = M.shape[-2]
    if out is None or out.shape != M.shape:
        out = torch.empty_like(M)
    if nb_channels == 1:
        out = _inv(M, out)
    elif nb_channels == 2:
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        invDet = _inv(det)
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else:
        raise Exception('Only 2 channels are supported for the torch version.')
    return out


def expectation_maximization(y: torch.Tensor, x: torch.Tensor, iterations: int=2, eps: float=1e-10, batch_size: int=200):
    """Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.

    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices

    Notes:
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.
        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning:
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.
    """
    nb_frames, nb_bins, nb_channels = x.shape[:-1]
    nb_sources = y.shape[-1]
    regularization = torch.cat((torch.eye(nb_channels, dtype=x.dtype, device=x.device)[..., None], torch.zeros((nb_channels, nb_channels, 1), dtype=x.dtype, device=x.device)), dim=2)
    regularization = torch.sqrt(torch.as_tensor(eps)) * regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1))
    R = [torch.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype, device=x.device) for j in range(nb_sources)]
    weight: torch.Tensor = torch.zeros((nb_bins,), dtype=x.dtype, device=x.device)
    v: torch.Tensor = torch.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype, device=x.device)
    for it in range(iterations):
        v = torch.mean(torch.abs(y[..., 0, :]) ** 2 + torch.abs(y[..., 1, :]) ** 2, dim=-2)
        for j in range(nb_sources):
            R[j] = torch.tensor(0.0, device=x.device)
            weight = torch.tensor(eps, device=x.device)
            pos: int = 0
            batch_size = batch_size if batch_size else nb_frames
            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1
                R[j] = R[j] + torch.sum(_covariance(y[t, ..., j]), dim=0)
                weight = weight + torch.sum(v[t, ..., j], dim=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)
        if y.requires_grad:
            y = y.clone()
        pos = 0
        while pos < nb_frames:
            t = torch.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1
            y[t, ...] = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + v[t, ..., j, None, None, None] * R[j][None, ...].clone()
            inv_Cxx = _invert(Cxx)
            for j in range(nb_sources):
                gain = torch.zeros_like(inv_Cxx)
                indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels), torch.arange(nb_channels))
                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(R[j][None, :, index[0], index[2], :].clone(), inv_Cxx[:, :, index[2], index[1], :], gain[:, :, index[0], index[1], :])
                gain = gain * v[t, ..., None, None, None, j]
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])
    return y, v, R


def wiener(targets_spectrograms: torch.Tensor, mix_stft: torch.Tensor, iterations: int=1, softmask: bool=False, residual: bool=False, scale_factor: float=10.0, eps: float=1e-10):
    """Wiener-based separation for multichannel audio.

    The method uses the (possibly multichannel) spectrograms  of the
    sources to separate the (complex) Short Term Fourier Transform  of the
    mix. Separation is done in a sequential way by:

    * Getting an initial estimate. This can be done in two ways: either by
      directly using the spectrograms with the mixture phase, or
      by using a softmasking strategy. This initial phase is controlled
      by the `softmask` flag.

    * If required, adding an additional residual target as the mix minus
      all targets.

    * Refinining these initial estimates through a call to
      :func:`expectation_maximization` if the number of iterations is nonzero.

    This implementation also allows to specify the epsilon value used for
    regularization. It is based on [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [4] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        targets_spectrograms (Tensor): spectrograms of the sources
            [shape=(nb_frames, nb_bins, nb_channels, nb_sources)].
            This is a nonnegative tensor that is
            usually the output of the actual separation method of the user. The
            spectrograms may be mono, but they need to be 4-dimensional in all
            cases.
        mix_stft (Tensor): [shape=(nb_frames, nb_bins, nb_channels, complex=2)]
            STFT of the mixture signal.
        iterations (int): [scalar]
            number of iterations for the EM algorithm
        softmask (bool): Describes how the initial estimates are obtained.
            * if `False`, then the mixture phase will directly be used with the
            spectrogram as initial estimates.
            * if `True`, initial estimates are obtained by multiplying the
            complex mix element-wise with the ratio of each target spectrogram
            with the sum of them all. This strategy is better if the model are
            not really good, and worse otherwise.
        residual (bool): if `True`, an additional target is created, which is
            equal to the mixture minus the other targets, before application of
            expectation maximization
        eps (float): Epsilon value to use for computing the separations.
            This is used whenever division with a model energy is
            performed, i.e. when softmasking and when iterating the EM.
            It can be understood as the energy of the additional white noise
            that is taken out when separating.

    Returns:
        Tensor: shape=(nb_frames, nb_bins, nb_channels, complex=2, nb_sources)
            STFT of estimated sources

    Notes:
        * Be careful that you need *magnitude spectrogram estimates* for the
        case `softmask==False`.
        * `softmask=False` is recommended
        * The epsilon value will have a huge impact on performance. If it's
        large, only the parts of the signal with a significant energy will
        be kept in the sources. This epsilon then directly controls the
        energy of the reconstruction error.

    Warning:
        As in :func:`expectation_maximization`, we recommend converting the
        mixture `x` to double precision `torch.float64` *before* calling
        :func:`wiener`.
    """
    if softmask:
        y = mix_stft[..., None] * (targets_spectrograms / (eps + torch.sum(targets_spectrograms, dim=-1, keepdim=True)))[..., None, :]
    else:
        angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]
        nb_sources = targets_spectrograms.shape[-1]
        y = torch.zeros(mix_stft.shape + (nb_sources,), dtype=mix_stft.dtype, device=mix_stft.device)
        y[..., 0, :] = targets_spectrograms * torch.cos(angle)
        y[..., 1, :] = targets_spectrograms * torch.sin(angle)
    if residual:
        y = torch.cat([y, mix_stft[..., None] - y.sum(dim=-1, keepdim=True)], dim=-1)
    if iterations == 0:
        return y
    max_abs = torch.max(torch.as_tensor(1.0, dtype=mix_stft.dtype, device=mix_stft.device), torch.sqrt(_norm(mix_stft)).max() / scale_factor)
    mix_stft = mix_stft / max_abs
    y = y / max_abs
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]
    y = y * max_abs
    return y


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Args:
        targets (dict of str: nn.Module): dictionary of target models
            the spectrogram models to be used by the Separator.
        niter (int): Number of EM steps for refining initial estimates in a
            post-processing stage. Zeroed if only one target is estimated.
            defaults to `1`.
        residual (bool): adds an additional residual target, obtained by
            subtracting the other estimated targets from the mixture,
            before any potential EM post-processing.
            Defaults to `False`.
        wiener_win_len (int or None): The size of the excerpts
            (number of frames) on which to apply filtering
            independently. This means assuming time varying stereo models and
            localization of sources.
            None means not batching but using the whole signal. It comes at the
            price of a much larger memory usage.
        filterbank (str): filterbank implementation method.
            Supported are `['torch', 'asteroid']`. `torch` is about 30% faster
            compared to `asteroid` on large FFT sizes such as 4096. However,
            asteroids stft can be exported to onnx, which makes is practical
            for deployment.
    """

    def __init__(self, target_models: Mapping[str, nn.Module], niter: int=0, softmask: bool=False, residual: bool=False, sample_rate: float=44100.0, n_fft: int=4096, n_hop: int=1024, nb_channels: int=2, wiener_win_len: Optional[int]=300, filterbank: str='torch'):
        super(Separator, self).__init__()
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len
        self.stft, self.istft = make_filterbanks(n_fft=n_fft, n_hop=n_hop, center=True, method=filterbank, sample_rate=sample_rate)
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.target_models = nn.ModuleDict(target_models)
        self.nb_targets = len(self.target_models)
        self.register_buffer('sample_rate', torch.as_tensor(sample_rate))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) ->Tensor:
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)
        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            target_spectrogram = target_module(X.detach().clone())
            spectrograms[..., j] = target_spectrogram
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)
        if self.residual:
            nb_sources += 1
        if nb_sources == 1 and self.niter > 0:
            raise Exception('Cannot use EM if only one target is estimated.Provide two targets or create an additional one with `--residual`')
        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device)
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1
                targets_stft[sample, cur_frame] = wiener(spectrograms[sample, cur_frame], mix_stft[sample, cur_frame], self.niter, softmask=self.softmask, residual=self.residual)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
        estimates = self.istft(targets_stft, length=audio.shape[2])
        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict]=None) ->dict:
        """Convert estimates as stacked tensor to dictionary

        Args:
            estimates (Tensor): separated targets of shape
                (nb_samples, nb_targets, nb_channels, nb_timesteps)
            aggregate_dict (dict or None)

        Returns:
            (dict of str: Tensor):
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]
        if self.residual:
            estimates_dict['residual'] = estimates[:, -1, ...]
        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict

