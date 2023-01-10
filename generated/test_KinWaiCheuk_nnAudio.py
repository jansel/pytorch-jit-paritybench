import sys
_module = sys.modules[__name__]
del sys
Spectrogram = _module
nnAudio = _module
features = _module
cfp = _module
cqt = _module
gammatone = _module
griffin_lim = _module
mel = _module
stft = _module
vqt = _module
librosa_functions = _module
utils = _module
setup = _module
parameters = _module
test_cfp = _module
test_cqt = _module
test_stft = _module
test_vqt = _module
tests_mel = _module
conf = _module

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


import numpy as np


import torch


import torch.nn as nn


import time


import scipy


from time import time


from torch.nn.functional import conv1d


from torch.nn.functional import conv2d


from torch.nn.functional import fold


from scipy.signal import get_window


import math


from scipy import signal


from scipy.fftpack import fft


import warnings


from scipy.signal import chirp


from scipy.signal import sweep_poly


epsilon = 1e-08


__TORCH_GTE_1_7 = False


def rfft_fn(x, n=None, onesided=False):
    if __TORCH_GTE_1_7:
        y = torch.fft.fft(x)
        return torch.view_as_real(y)
    else:
        return torch.rfft(x, n, onesided=onesided)


class Combined_Frequency_Periodicity(nn.Module):
    """
    Vectorized version of the code in https://github.com/leo-so/VocalMelodyExtPatchCNN/blob/master/MelodyExt.py.
    This feature is described in 'Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic Music'
    https://ieeexplore.ieee.org/document/7118691

    Parameters
    ----------
    fr : int
        Frequency resolution. The higher the number, the lower the resolution is.
        Maximum frequency resolution occurs when ``fr=1``. The default value is ``2``

    fs : int
        Sample rate of the input audio clips. The default value is ``16000``

    hop_length : int
        The hop (or stride) size. The default value is ``320``.

    window_size : str
        It is same as ``n_fft`` in other Spectrogram classes. The default value is ``2049``

    fc : int
        Starting frequency. For example, ``fc=80`` means that `Z` starts at 80Hz.
        The default value is ``80``.

    tc : int
        Inverse of ending frequency. For example ``tc=1/8000`` means that `Z` ends at 8000Hz.
        The default value is ``1/8000``.

    g: list
        Coefficients for non-linear activation function. ``len(g)`` should be the number of activation layers.
        Each element in ``g`` is the activation coefficient, for example ``[0.24, 0.6, 1]``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    Returns
    -------
    Z : torch.tensor
        The Combined Frequency and Period Feature. It is equivalent to ``tfrLF * tfrLQ``

    tfrL0: torch.tensor
        STFT output

    tfrLF: torch.tensor
        Frequency Feature

    tfrLQ: torch.tensor
        Period Feature

    Examples
    --------
    >>> spec_layer = Spectrogram.Combined_Frequency_Periodicity()
    >>> Z, tfrL0, tfrLF, tfrLQ = spec_layer(x)

    """

    def __init__(self, fr=2, fs=16000, hop_length=320, window_size=2049, fc=80, tc=1 / 1000, g=[0.24, 0.6, 1], NumPerOct=48):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length
        self.N = int(fs / float(fr))
        self.f = fs * np.linspace(0, 0.5, np.round(self.N // 2), endpoint=True)
        self.pad_value = self.N - window_size
        h = scipy.signal.blackmanharris(window_size)
        self.register_buffer('h', torch.tensor(h).float())
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(fs * tc)
        self.fc_idx = round(fc / fr)
        self.HighFreqIdx = int(round(1 / tc / fr) + 1)
        self.HighQuefIdx = int(round(fs / fc) + 1)
        self.f = self.f[:self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx) / float(fs)
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(self.f, self.q, fr, fc, tc, NumPerOct, fs)
        self.register_buffer('freq2logfreq_matrix', torch.tensor(freq2logfreq_matrix).float())
        self.register_buffer('quef2logfreq_matrix', torch.tensor(quef2logfreq_matrix).float())

    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])
        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = rfft_fn(spec, 1, onesided=False)[:, :, :, 0] / np.sqrt(self.N)
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = rfft_fn(ceps, 1, onesided=False)[:, :, :, 0] / np.sqrt(self.N)
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)
        return spec, ceps

    def forward(self, x):
        tfr0 = torch.stft(x, self.N, hop_length=self.hop_length, win_length=self.window_size, window=self.h, onesided=False, pad_mode='constant')
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1)) / torch.norm(self.h)
        tfr0 = tfr0.transpose(1, 2)[:, 1:-1]
        tfr, ceps = self._CFP(tfr0)
        tfr0 = tfr0[:, :, :int(round(self.N / 2))]
        tfr = tfr[:, :, :int(round(self.N / 2))]
        ceps = ceps[:, :, :int(round(self.N / 2))]
        tfr0 = tfr0[:, :, :self.HighFreqIdx]
        tfr = tfr[:, :, :self.HighFreqIdx]
        ceps = ceps[:, :, :self.HighQuefIdx]
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1, 2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1, 2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1, 2))
        Z = tfrLF * tfrLQ
        self.t = np.arange(self.hop_length, np.ceil(len(x) / float(self.hop_length)) * self.hop_length, self.hop_length)
        return Z, tfrL0, tfrLF, tfrLQ

    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g != 0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else:
            X = torch.log(X.relu() + epsilon)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X

    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1 / tc
        Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
        central_freq = []
        for i in range(0, Nest):
            CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break
        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            l = int(round(central_freq[i - 1] / fr))
            r = int(round(central_freq[i + 1] / fr) + 1)
            if l >= r - 1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                        freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
        f = 1 / q
        quef_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
        return freq_band_transformation, quef_band_transformation


class CFP(nn.Module):
    """
    This is the modified version of :func:`~nnAudio.Spectrogram.Combined_Frequency_Periodicity`. This version different from the original version by returnning only ``Z`` and the number of timesteps fits with other classes.

    Parameters
    ----------
    fr : int
        Frequency resolution. The higher the number, the lower the resolution is.
        Maximum frequency resolution occurs when ``fr=1``. The default value is ``2``

    fs : int
        Sample rate of the input audio clips. The default value is ``16000``

    hop_length : int
        The hop (or stride) size. The default value is ``320``.

    window_size : str
        It is same as ``n_fft`` in other Spectrogram classes. The default value is ``2049``

    fc : int
        Starting frequency. For example, ``fc=80`` means that `Z` starts at 80Hz.
        The default value is ``80``.

    tc : int
        Inverse of ending frequency. For example ``tc=1/8000`` means that `Z` ends at 8000Hz.
        The default value is ``1/8000``.

    g: list
        Coefficients for non-linear activation function. ``len(g)`` should be the number of activation layers.
        Each element in ``g`` is the activation coefficient, for example ``[0.24, 0.6, 1]``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    Returns
    -------
    Z : torch.tensor
        The Combined Frequency and Period Feature. It is equivalent to ``tfrLF * tfrLQ``

    tfrL0: torch.tensor
        STFT output

    tfrLF: torch.tensor
        Frequency Feature

    tfrLQ: torch.tensor
        Period Feature

    Examples
    --------
    >>> spec_layer = Spectrogram.Combined_Frequency_Periodicity()
    >>> Z, tfrL0, tfrLF, tfrLQ = spec_layer(x)

    """

    def __init__(self, fr=2, fs=16000, hop_length=320, window_size=2049, fc=80, tc=1 / 1000, g=[0.24, 0.6, 1], NumPerOct=48):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length
        self.N = int(fs / float(fr))
        self.f = fs * np.linspace(0, 0.5, np.round(self.N // 2), endpoint=True)
        self.pad_value = self.N - window_size
        h = scipy.signal.blackmanharris(window_size)
        self.register_buffer('h', torch.tensor(h).float())
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(fs * tc)
        self.fc_idx = round(fc / fr)
        self.HighFreqIdx = int(round(1 / tc / fr) + 1)
        self.HighQuefIdx = int(round(fs / fc) + 1)
        self.f = self.f[:self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx) / float(fs)
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(self.f, self.q, fr, fc, tc, NumPerOct, fs)
        self.register_buffer('freq2logfreq_matrix', torch.tensor(freq2logfreq_matrix).float())
        self.register_buffer('quef2logfreq_matrix', torch.tensor(quef2logfreq_matrix).float())

    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])
        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = rfft_fn(spec, 1, onesided=False)[:, :, :, 0] / np.sqrt(self.N)
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = rfft_fn(ceps, 1, onesided=False)[:, :, :, 0] / np.sqrt(self.N)
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)
        return spec, ceps

    def forward(self, x):
        tfr0 = torch.stft(x, self.N, hop_length=self.hop_length, win_length=self.window_size, window=self.h, onesided=False, pad_mode='constant')
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1)) / torch.norm(self.h)
        tfr0 = tfr0.transpose(1, 2)
        tfr, ceps = self._CFP(tfr0)
        tfr0 = tfr0[:, :, :int(round(self.N / 2))]
        tfr = tfr[:, :, :int(round(self.N / 2))]
        ceps = ceps[:, :, :int(round(self.N / 2))]
        tfr0 = tfr0[:, :, :self.HighFreqIdx]
        tfr = tfr[:, :, :self.HighFreqIdx]
        ceps = ceps[:, :, :self.HighQuefIdx]
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1, 2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1, 2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1, 2))
        Z = tfrLF * tfrLQ
        self.t = np.arange(self.hop_length, np.ceil(len(x) / float(self.hop_length)) * self.hop_length, self.hop_length)
        return Z

    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g != 0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else:
            X = torch.log(X.relu() + epsilon)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X

    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1 / tc
        Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
        central_freq = []
        for i in range(0, Nest):
            CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break
        Nest = len(central_freq)
        freq_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            l = int(round(central_freq[i - 1] / fr))
            r = int(round(central_freq[i + 1] / fr) + 1)
            if l >= r - 1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                        freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
        f = 1 / q
        quef_band_transformation = np.zeros((Nest - 1, len(f)), dtype=np.float)
        for i in range(1, Nest - 1):
            for j in range(int(round(fs / central_freq[i + 1])), int(round(fs / central_freq[i - 1]) + 1)):
                if f[j] > central_freq[i - 1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i - 1]) / (central_freq[i] - central_freq[i - 1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i + 1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
        return freq_band_transformation, quef_band_transformation


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """
    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError('Only support input with shape = (batch, len) or shape = (len)')
    return x


def complex_mul(cqt_filter, stft):
    """Since PyTorch does not support complex numbers and its operation.
    We need to write our own complex multiplication function. This one is specially
    designed for CQT usage.

    Parameters
    ----------
    cqt_filter : tuple of torch.Tensor
        The tuple is in the format of ``(real_torch_tensor, imag_torch_tensor)``

    Returns
    -------
    tuple of torch.Tensor
        The output is in the format of ``(real_torch_tensor, imag_torch_tensor)``
    """
    cqt_filter_real = cqt_filter[0]
    cqt_filter_imag = cqt_filter[1]
    fourier_real = stft[0]
    fourier_imag = stft[1]
    CQT_real = torch.matmul(cqt_filter_real, fourier_real) - torch.matmul(cqt_filter_imag, fourier_imag)
    CQT_imag = torch.matmul(cqt_filter_real, fourier_imag) + torch.matmul(cqt_filter_imag, fourier_real)
    return CQT_real, CQT_imag


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == 'gaussian':
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(('gaussian', sigma), N, fftbins=fftbins)
        else:
            Warning('Tuple windows may have undesired behaviour regarding Q factor')
    elif isinstance(window, float):
        Warning('You are using Kaiser window with beta factor ' + str(window) + '. Correct behaviour not checked.')
    else:
        raise Exception('The function get_window from scipy only supports strings, tuples and floats.')


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """
    return int(np.ceil(np.log2(A)))


def create_cqt_kernels(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1, window='hann', fmax=None, topbin_check=True, gamma=0, pad_fft=True):
    """
    Automatically create CQT kernels in time domain
    """
    fftLen = 2 ** nextpow2(np.ceil(Q * fs / fmin))
    if fmax != None and n_bins == None:
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    elif fmax == None and n_bins != None:
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    else:
        warnings.warn('If fmax is given, n_bins will be ignored', SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
    if np.max(freqs) > fs / 2 and topbin_check == True:
        raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency,                           please reduce the n_bins'.format(np.max(freqs)))
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))
    max_len = int(max(lengths))
    fftLen = int(2 ** np.ceil(np.log2(max_len)))
    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]
        if l % 2 == 1:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))
        window_dispatch = get_window_dispatch(window, int(l), fftbins=True)
        sig = window_dispatch * np.exp(np.r_[-l // 2:l // 2] * 1.0j * 2 * np.pi * freq / fs) / l
        if norm:
            tempKernel[k, start:start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start:start + int(l)] = sig
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


def pad_center(data, size, axis=-1, **kwargs):
    """Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    """
    kwargs.setdefault('mode', 'constant')
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = lpad, int(size - n - lpad)
    if lpad < 0:
        raise ParameterError('Target size ({:d}) must be at least input size ({:d})'.format(size, n))
    return np.pad(data, lengths, **kwargs)


def create_fourier_kernels(n_fft, win_length=None, freq_bins=None, fmin=50, fmax=6000, sr=44100, freq_scale='linear', window='hann', verbose=True):
    """This function creates the Fourier Kernel for STFT, Melspectrogram and CQT.
    Most of the parameters follow librosa conventions. Part of the code comes from
    pytorch_musicnet. https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    n_fft : int
        The window size

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    fmin : int
        The starting frequency for the lowest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    freq_scale: 'linear', 'log', 'log2', or 'no'
        Determine the spacing between each frequency bin.
        When 'linear', 'log' or 'log2' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``.
        If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    bins2freq : list
        Mapping each frequency bin to frequency in Hz.

    binslist : list
        The normalized frequency ``k`` in digital domain.
        This ``k`` is in the Discrete Fourier Transform equation $$

    """
    if freq_bins == None:
        freq_bins = n_fft // 2 + 1
    if win_length == None:
        win_length = n_fft
    s = np.arange(0, n_fft, 1.0)
    wsin = np.empty((freq_bins, 1, n_fft))
    wcos = np.empty((freq_bins, 1, n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []
    window_mask = get_window(window, int(win_length), fftbins=True)
    window_mask = pad_center(window_mask, n_fft)
    if freq_scale == 'linear':
        if verbose == True:
            None
        start_bin = start_freq * n_fft / sr
        scaling_ind = (end_freq - start_freq) * (n_fft / sr) / freq_bins
        for k in range(freq_bins):
            bins2freq.append((k * scaling_ind + start_bin) * sr / n_fft)
            binslist.append(k * scaling_ind + start_bin)
            wsin[k, 0, :] = np.sin(2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft)
    elif freq_scale == 'log':
        if verbose == True:
            None
        start_bin = start_freq * n_fft / sr
        scaling_ind = np.log(end_freq / start_freq) / freq_bins
        for k in range(freq_bins):
            bins2freq.append(np.exp(k * scaling_ind) * start_bin * sr / n_fft)
            binslist.append(np.exp(k * scaling_ind) * start_bin)
            wsin[k, 0, :] = np.sin(2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft)
    elif freq_scale == 'log2':
        if verbose == True:
            None
        start_bin = start_freq * n_fft / sr
        scaling_ind = np.log2(end_freq / start_freq) / freq_bins
        for k in range(freq_bins):
            bins2freq.append(2 ** (k * scaling_ind) * start_bin * sr / n_fft)
            binslist.append(2 ** (k * scaling_ind) * start_bin)
            wsin[k, 0, :] = np.sin(2 * np.pi * (2 ** (k * scaling_ind) * start_bin) * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * (2 ** (k * scaling_ind) * start_bin) * s / n_fft)
    elif freq_scale == 'no':
        for k in range(freq_bins):
            bins2freq.append(k * sr / n_fft)
            binslist.append(k)
            wsin[k, 0, :] = np.sin(2 * np.pi * k * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * k * s / n_fft)
    else:
        None
    return wsin.astype(np.float32), wcos.astype(np.float32), bins2freq, binslist, window_mask.astype(np.float32)


class CQT1992(nn.Module):
    """
    This alogrithm uses the method proposed in [1], which would run extremely slow if low frequencies (below 220Hz)
    are included in the frequency bins.
    Please refer to :func:`~nnAudio.Spectrogram.CQT1992v2` for a more
    computational and memory efficient version.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.



    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    trainable_STFT : bool
        Determine if the time to frequency domain transformation kernel for the input audio is trainable or not.
        Default is ``False``

    trainable_CQT : bool
        Determine if the frequency domain CQT kernel is trainable or not.
        Default is ``False``

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

     output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84, trainable_STFT=False, trainable_CQT=False, bins_per_octave=12, filter_scale=1, output_format='Magnitude', norm=1, window='hann', center=True, pad_mode='reflect'):
        super().__init__()
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.output_format = output_format
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        None
        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        self.register_buffer('lenghts', lenghts)
        self.frequencies = freqs
        cqt_kernels = fft(cqt_kernels)[:, :self.kernel_width // 2 + 1]
        None
        None
        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(self.kernel_width, window='ones', freq_scale='no')
        wsin = torch.tensor(kernel_sin * window)
        wcos = torch.tensor(kernel_cos * window)
        cqt_kernels_real = torch.tensor(cqt_kernels.real)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag)
        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
        if trainable_CQT:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable_CQT)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable_CQT)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)
        None

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width // 2)
            x = padding(x)
        fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)
        CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag), (fourier_real, fourier_imag))
        CQT = torch.stack((CQT_real, -CQT_imag), -1)
        if normalization_type == 'librosa':
            CQT *= torch.sqrt(self.lenghts.view(-1, 1, 1)) / self.kernel_width
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2 / self.kernel_width
        else:
            raise ValueError('The normalization_type %r is not part of our current options.' % normalization_type)
        if output_format == 'Magnitude':
            return torch.sqrt(CQT.pow(2).sum(-1))
        elif output_format == 'Complex':
            return CQT
        elif output_format == 'Phase':
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)

    def extra_repr(self) ->str:
        return 'STFT kernel size = {}, CQT kernel size = {}'.format((*self.wcos.shape,), (*self.cqt_kernels_real.shape,))


def create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03):
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through.
    Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """
    passbandMax = band_center / (1 + transitionBandwidth)
    stopbandMin = band_center * (1 + transitionBandwidth)
    keyFrequencies = [0.0, passbandMax, stopbandMin, 1.0]
    gainAtKeyFrequencies = [1.0, 1.0, 0.0, 0.0]
    filterKernel = signal.firwin2(kernelLength, keyFrequencies, gainAtKeyFrequencies)
    return filterKernel.astype(np.float32)


def downsampling_by_2(x, filterKernel):
    """A helper function that downsamples the audio by half. It is used in CQT2010 and CQT2010v2

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_2(x, filterKernel)
    """
    x = conv1d(x, filterKernel, stride=2, padding=(filterKernel.shape[-1] - 1) // 2)
    return x


def downsampling_by_n(x, filterKernel, n):
    """A helper function that downsamples the audio by a arbitary factor n.
    It is used in CQT2010 and CQT2010v2.

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    n : int
        The downsampling factor

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_n(x, filterKernel)
    """
    x = conv1d(x, filterKernel, stride=n, padding=(filterKernel.shape[-1] - 1) // 2)
    return x


def get_cqt_complex2(x, cqt_kernels_real, cqt_kernels_imag, hop_length, padding, wcos=None, wsin=None):
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""
    try:
        x = padding(x)
    except:
        warnings.warn(f'\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\npadding with reflection mode might not be the best choice, try using constant padding', UserWarning)
        x = torch.nn.functional.pad(x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2))
    if wcos == None or wsin == None:
        CQT_real = conv1d(x, cqt_kernels_real, stride=hop_length)
        CQT_imag = -conv1d(x, cqt_kernels_imag, stride=hop_length)
    else:
        fourier_real = conv1d(x, wcos, stride=hop_length)
        fourier_imag = conv1d(x, wsin, stride=hop_length)
        CQT_real, CQT_imag = complex_mul((cqt_kernels_real, cqt_kernels_imag), (fourier_real, fourier_imag))
    return torch.stack((CQT_real, CQT_imag), -1)


def early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """Compute the number of early downsampling operations"""
    downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist / filter_cutoff)) - 1) - 1)
    num_twos = nextpow2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    return min(downsample_count1, downsample_count2)


def early_downsample(sr, hop_length, n_octaves, nyquist, filter_cutoff):
    """Return new sampling rate and hop length after early dowansampling"""
    downsample_count = early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    downsample_factor = 2 ** downsample_count
    hop_length //= downsample_factor
    new_sr = sr / float(downsample_factor)
    sr = new_sr
    return sr, hop_length, downsample_factor


def get_early_downsample_params(sr, hop_length, fmax_t, Q, n_octaves, verbose):
    """Used in CQT2010 and CQT2010v2"""
    window_bandwidth = 1.5
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    sr, hop_length, downsample_factor = early_downsample(sr, hop_length, n_octaves, sr // 2, filter_cutoff)
    if downsample_factor != 1:
        if verbose == True:
            None
        earlydownsample = True
        early_downsample_filter = create_lowpass_filter(band_center=1 / downsample_factor, kernelLength=256, transitionBandwidth=0.03)
        early_downsample_filter = torch.tensor(early_downsample_filter)[None, None, :]
    else:
        if verbose == True:
            None
        early_downsample_filter = None
        earlydownsample = False
    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample


class CQT2010(nn.Module):
    """
    This algorithm is using the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave.
    Then we keep downsampling the input audio by a factor of 2 to convoluting it with the
    small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled
    input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code
    from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', trainable_STFT=False, filter_scale=1, trainable_CQT=False, output_format='Magnitude', earlydownsample=True, verbose=True):
        super().__init__()
        self.norm = norm
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.output_format = output_format
        self.earlydownsample = earlydownsample
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        if verbose == True:
            None
        start = time()
        lowpass_filter = torch.tensor(create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.001))
        self.register_buffer('lowpass_filter', lowpass_filter[None, None, :])
        if verbose == True:
            None
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        if remainder == 0:
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
        self.fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
        if fmax_t > sr / 2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency,                               please reduce the n_bins'.format(fmax_t))
        if self.earlydownsample == True:
            if verbose == True:
                None
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, self.earlydownsample = get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves, verbose)
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            if verbose == True:
                None
        else:
            self.downsample_factor = 1.0
        if verbose == True:
            None
        start = time()
        basis, self.n_fft, _, _ = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)
        self.basis = basis
        fft_basis = fft(basis)[:, :self.n_fft // 2 + 1]
        cqt_kernels_real = torch.tensor(fft_basis.real)
        cqt_kernels_imag = torch.tensor(fft_basis.imag)
        if verbose == True:
            None
        if verbose == True:
            None
        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(self.n_fft, window='ones', freq_scale='no')
        wsin = kernel_sin * window
        wcos = kernel_cos * window
        wsin = torch.tensor(wsin)
        wcos = torch.tensor(wcos)
        if verbose == True:
            None
        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
        if trainable_CQT:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable_CQT)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable_CQT)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        if self.earlydownsample == True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = get_cqt_complex2(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding, wcos=self.wcos, wsin=self.wsin)
        x_down = x
        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = get_cqt_complex2(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding, wcos=self.wcos, wsin=self.wsin)
            CQT = torch.cat((CQT1, CQT), 1)
        CQT = CQT[:, -self.n_bins:, :]
        if normalization_type == 'librosa':
            CQT *= torch.sqrt(self.lenghts.view(-1, 1, 1)) / self.n_fft
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2 / self.n_fft
        else:
            raise ValueError('The normalization_type %r is not part of our current options.' % normalization_type)
        if output_format == 'Magnitude':
            return torch.sqrt(CQT.pow(2).sum(-1))
        elif output_format == 'Complex':
            return CQT
        elif output_format == 'Phase':
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)

    def extra_repr(self) ->str:
        return 'STFT kernel size = {}, CQT kernel size = {}'.format((*self.wcos.shape,), (*self.cqt_kernels_real.shape,))


class CQT1992v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it runs faster
    than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis, analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window should be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

    output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, bins_per_octave=12, filter_scale=1, norm=1, window='hann', center=True, pad_mode='reflect', trainable=False, output_format='Magnitude', verbose=True):
        super().__init__()
        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        if verbose == True:
            None
        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax)
        self.register_buffer('lenghts', lenghts)
        self.frequencies = freqs
        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)
        if trainable:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)
        if verbose == True:
            None

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        normalization_type : str
            Type of the normalisation. The possible options are: 

            'librosa' : the output fits the librosa one 

            'convolutional' : the output conserves the convolutional inequalities of the wavelet transform:

            for all p ϵ [1, inf] 

                - || CQT ||_p <= || f ||_p || g ||_1 

                - || CQT ||_p <= || f ||_1 || g ||_p 

                - || CQT ||_2 = || f ||_2 || g ||_2 

            'wrap' : wraps positive and negative frequencies into positive frequencies. This means that the CQT of a
            sinus (or a cosinus) with a constant amplitude equal to 1 will have the value 1 in the bin corresponding to
            its frequency.
        """
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width // 2)
            x = padding(x)
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = -conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)
        if normalization_type == 'librosa':
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError('The normalization_type %r is not part of our current options.' % normalization_type)
        if output_format == 'Magnitude':
            if self.trainable == False:
                CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2))
            else:
                CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2) + 1e-08)
            return CQT
        elif output_format == 'Complex':
            return torch.stack((CQT_real, CQT_imag), -1)
        elif output_format == 'Phase':
            phase_real = torch.cos(torch.atan2(CQT_imag, CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag, CQT_real))
            return torch.stack((phase_real, phase_imag), -1)

    def forward_manual(self, x):
        """
        Method for debugging
        """
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width // 2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width // 2)
            x = padding(x)
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)
        CQT = torch.sqrt(CQT_real.pow(2) + CQT_imag.pow(2))
        return CQT * torch.sqrt(self.lenghts.view(-1, 1))


def get_cqt_complex(x, cqt_kernels_real, cqt_kernels_imag, hop_length, padding):
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""
    try:
        x = padding(x)
    except:
        warnings.warn(f'\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\npadding with reflection mode might not be the best choice, try using constant padding', UserWarning)
        x = torch.nn.functional.pad(x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2))
    CQT_real = conv1d(x, cqt_kernels_real, stride=hop_length)
    CQT_imag = -conv1d(x, cqt_kernels_imag, stride=hop_length)
    return torch.stack((CQT_real, CQT_imag), -1)


class CQT2010v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    norm : bool
        Normalization for the CQT result.

    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, filter_scale=1, bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect', earlydownsample=True, trainable=False, output_format='Magnitude', verbose=True):
        super().__init__()
        self.norm = norm
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        if verbose == True:
            None
        start = time()
        lowpass_filter = torch.tensor(create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.001))
        self.register_buffer('lowpass_filter', lowpass_filter[None, None, :])
        if verbose == True:
            None
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose == True:
            None
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        if remainder == 0:
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
        self.fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
        if fmax_t > sr / 2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency,                             please reduce the n_bins'.format(fmax_t))
        if self.earlydownsample == True:
            if verbose == True:
                None
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, self.earlydownsample = get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves, verbose)
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            if verbose == True:
                None
        else:
            self.downsample_factor = 1.0
        if verbose == True:
            None
        start = time()
        basis, self.n_fft, lenghts, _ = create_cqt_kernels(Q, sr, self.fmin_t, n_filters, bins_per_octave, norm=basis_norm, topbin_check=False)
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)
        self.basis = basis
        cqt_kernels_real = torch.tensor(basis.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(basis.imag).unsqueeze(1)
        if trainable:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)
        if verbose == True:
            None
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        if self.earlydownsample == True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
        x_down = x
        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
            CQT = torch.cat((CQT1, CQT), 1)
        CQT = CQT[:, -self.n_bins:, :]
        CQT = CQT * self.downsample_factor
        if normalization_type == 'librosa':
            CQT = CQT * torch.sqrt(self.lenghts.view(-1, 1, 1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2
        else:
            raise ValueError('The normalization_type %r is not part of our current options.' % normalization_type)
        if output_format == 'Magnitude':
            if self.trainable == False:
                return torch.sqrt(CQT.pow(2).sum(-1))
            else:
                return torch.sqrt(CQT.pow(2).sum(-1) + 1e-08)
        elif output_format == 'Complex':
            return CQT
        elif output_format == 'Phase':
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)


class CQT(CQT1992v2):
    """An abbreviation for :func:`~nnAudio.Spectrogram.CQT1992v2`. Please refer to the :func:`~nnAudio.Spectrogram.CQT1992v2` documentation"""
    pass


def fft2gammatonemx(sr=20000, n_fft=2048, n_bins=64, width=1.0, fmin=0.0, fmax=11025, maxlen=1024):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """
    wts = np.zeros([n_bins, n_fft], dtype=np.float32)
    EarQ = 9.26449
    minBW = 24.7
    order = 1
    nFr = np.array(range(n_bins)) + 1
    em = EarQ * minBW
    cfreqs = (fmax + em) * np.exp(nFr * (-np.log(fmax + em) + np.log(fmin + em)) / n_bins) - em
    cfreqs = cfreqs[::-1]
    GTord = 4
    ucircArray = np.array(range(int(n_fft / 2 + 1)))
    ucirc = np.exp(1.0j * 2 * np.pi * ucircArray / n_fft)
    ERB = width * np.power(np.power(cfreqs / EarQ, order) + np.power(minBW, order), 1 / order)
    B = 1.019 * 2 * np.pi * ERB
    r = np.exp(-B / sr)
    theta = 2 * np.pi * cfreqs / sr
    pole = r * np.exp(1.0j * theta)
    T = 1 / sr
    ebt = np.exp(B * T)
    cpt = 2 * cfreqs * np.pi * T
    ccpt = 2 * T * np.cos(cpt)
    scpt = 2 * T * np.sin(cpt)
    A11 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2)
    A12 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2)
    A13 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2)
    A14 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2)
    zros = -np.array([A11, A12, A13, A14]) / T
    wIdx = range(int(n_fft / 2 + 1))
    gain = np.abs((-2 * np.exp(4 * 1.0j * cfreqs * np.pi * T) * T + 2 * np.exp(-(B * T) + 2 * 1.0j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) * (-2 * np.exp(4 * 1.0j * cfreqs * np.pi * T) * T + 2 * np.exp(-(B * T) + 2 * 1.0j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) * (-2 * np.exp(4 * 1.0j * cfreqs * np.pi * T) * T + 2 * np.exp(-(B * T) + 2 * 1.0j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) * (-2 * np.exp(4 * 1.0j * cfreqs * np.pi * T) * T + 2 * np.exp(-(B * T) + 2 * 1.0j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) / (-2 / np.exp(2 * B * T) - 2 * np.exp(4 * 1.0j * cfreqs * np.pi * T) + 2 * (1 + np.exp(4 * 1.0j * cfreqs * np.pi * T)) / np.exp(B * T)) ** 4)
    wts[:, wIdx] = T ** 4 / np.reshape(gain, (n_bins, 1)) * np.abs(ucirc - np.reshape(zros[0], (n_bins, 1))) * np.abs(ucirc - np.reshape(zros[1], (n_bins, 1))) * np.abs(ucirc - np.reshape(zros[2], (n_bins, 1))) * np.abs(ucirc - np.reshape(zros[3], (n_bins, 1))) * np.abs(np.power(np.multiply(np.reshape(pole, (n_bins, 1)) - ucirc, np.conj(np.reshape(pole, (n_bins, 1))) - ucirc), -GTord))
    wts = wts[:, range(maxlen)]
    return wts, cfreqs


def get_gammatone(sr, n_fft, n_bins=64, fmin=20.0, fmax=None, htk=False, norm=1, dtype=np.float32):
    """Create a Filterbank matrix to combine FFT bins into Gammatone bins
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_bins    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    G         : np.ndarray [shape=(n_bins, 1 + n_fft/2)]
        Gammatone transform matrix
    """
    if fmax is None:
        fmax = float(sr) / 2
    n_bins = int(n_bins)
    weights, _ = fft2gammatonemx(sr=sr, n_fft=n_fft, n_bins=n_bins, fmin=fmin, fmax=fmax, maxlen=int(n_fft // 2 + 1))
    return 1 / n_fft * weights


class Gammatonegram(nn.Module):
    """
    This function is to calculate the Gammatonegram of the input signal.
    
    Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    n_fft : int
        The window size for the STFT. Default value is 2048
    n_mels : int
        The number of Gammatonegram filter banks. The filter banks maps the n_fft to Gammatone bins. Default value is 64

    hop_length : int
        The hop (or stride) size. Default value is 512.
    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'
    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time index is the beginning of the STFT kernel, if ``True``, the time index is the center of the STFT kernel. Default value if ``True``.
    pad_mode : str
        The padding method. Default value is 'reflect'.
    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the Mel scale is logarithmic. The default value is ``False``

    fmin : int
        The starting frequency for the lowest Gammatone filter bank
    fmax : int
        The ending frequency for the highest Gammatone filter bank
    trainable_mel : bool
        Determine if the Gammatone filter banks are trainable or not. If ``True``, the gradients for Mel filter banks will also be caluclated and the Mel filter banks will be updated during model training. Default value is ``False``
    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT kernels will also be caluclated and the STFT kernels will be updated during model training. Default value is ``False``

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.Gammatonegram()
    >>> specs = spec_layer(x)
    
    """

    def __init__(self, sr=44100, n_fft=2048, n_bins=64, hop_length=512, window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, fmin=20.0, fmax=None, norm=1, trainable_bins=False, trainable_STFT=False, verbose=True):
        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power
        start = time()
        wsin, wcos, self.bins2freq, _, _ = create_fourier_kernels(n_fft, freq_bins=None, window=window, freq_scale='no', sr=sr)
        wsin = torch.tensor(wsin, dtype=torch.float)
        wcos = torch.tensor(wcos, dtype=torch.float)
        if trainable_STFT:
            wsin = nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
        start = time()
        gammatone_basis = get_gammatone(sr, n_fft, n_bins, fmin, fmax)
        gammatone_basis = torch.tensor(gammatone_basis)
        if verbose == True:
            None
            None
        else:
            pass
        if trainable_bins:
            gammatone_basis = nn.Parameter(gammatone_basis, requires_grad=trainable_bins)
            self.register_parameter('gammatone_basis', gammatone_basis)
        else:
            self.register_buffer('gammatone_basis', gammatone_basis)

    def forward(self, x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft // 2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft // 2)
            x = padding(x)
        spec = torch.sqrt(conv1d(x, self.wsin, stride=self.stride).pow(2) + conv1d(x, self.wcos, stride=self.stride).pow(2)) ** self.power
        gammatonespec = torch.matmul(self.gammatone_basis, spec)
        return gammatonespec


class Griffin_Lim(nn.Module):
    """
    Converting Magnitude spectrograms back to waveforms based on the "fast Griffin-Lim"[1].
    This Griffin Lim is a direct clone from librosa.griffinlim.

    [1] Perraudin, N., Balazs, P., & Søndergaard, P. L. “A fast Griffin-Lim algorithm,”
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4), Oct. 2013.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    n_iter=32 : int
        The number of iterations for Griffin-Lim. The default value is ``32``

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``.
        Please make sure the value is the same as the forward STFT.

    momentum : float
        The momentum for the update rule. The default value is ``0.99``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'

    """

    def __init__(self, n_fft, n_iter=32, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', momentum=0.99, device='cpu'):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.n_iter = n_iter
        self.center = center
        self.pad_mode = pad_mode
        self.momentum = momentum
        self.device = device
        if win_length == None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        if hop_length == None:
            self.hop_length = n_fft // 4
        else:
            self.hop_length = hop_length
        self.w = torch.tensor(get_window(window, int(self.win_length), fftbins=True), device=device).float()

    def forward(self, S):
        """
        Convert a batch of magnitude spectrograms to waveforms.

        Parameters
        ----------
        S : torch tensor
            Spectrogram of the shape ``(batch, n_fft//2+1, timesteps)``
        """
        assert S.dim() == 3, 'Please make sure your input is in the shape of (batch, freq_bins, timesteps)'
        rand_phase = torch.randn(*S.shape, device=self.device)
        angles = torch.empty((*S.shape, 2), device=self.device)
        angles[:, :, :, 0] = torch.cos(2 * np.pi * rand_phase)
        angles[:, :, :, 1] = torch.sin(2 * np.pi * rand_phase)
        rebuilt = torch.zeros(*angles.shape, device=self.device)
        for _ in range(self.n_iter):
            tprev = rebuilt
            inverse = torch.istft(S.unsqueeze(-1) * angles, self.n_fft, self.hop_length, win_length=self.win_length, window=self.w, center=self.center)
            rebuilt = torch.stft(inverse, self.n_fft, self.hop_length, win_length=self.win_length, window=self.w, pad_mode=self.pad_mode)
            angles[:, :, :] = rebuilt[:, :, :] - self.momentum / (1 + self.momentum) * tprev[:, :, :]
            angles = angles.div(torch.sqrt(angles.pow(2).sum(-1)).unsqueeze(-1) + 1e-16)
        inverse = torch.istft(S.unsqueeze(-1) * angles, self.n_fft, self.hop_length, win_length=self.win_length, window=self.w, center=self.center)
        return inverse


def extend_fbins(X):
    """Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
    reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
    X_upper = X[:, 1:-1].flip(1)
    X_upper[:, :, :, 1] = -X_upper[:, :, :, 1]
    return torch.cat((X[:, :, :], X_upper), 1)


def overlap_add(X, stride):
    n_fft = X.shape[1]
    output_len = n_fft + stride * (X.shape[2] - 1)
    return fold(X, (1, output_len), kernel_size=(1, n_fft), stride=stride).flatten(1)


def torch_window_sumsquare(w, n_frames, stride, n_fft, power=2):
    w_stacks = w.unsqueeze(-1).repeat((1, n_frames)).unsqueeze(0)
    output_len = w_stacks.shape[1] + stride * (w_stacks.shape[2] - 1)
    return fold(w_stacks ** power, (1, output_len), kernel_size=(1, n_fft), stride=stride)


class STFTBase(nn.Module):
    """
    STFT and iSTFT share the same `inverse_stft` function
    """

    def inverse_stft(self, X, kernel_cos, kernel_sin, onesided=True, length=None, refresh_win=True):
        if onesided:
            X = extend_fbins(X)
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = conv2d(X_real_bc, kernel_cos, stride=(1, 1))
        b2 = conv2d(X_imag_bc, kernel_sin, stride=(1, 1))
        real = a1 - b2
        real = real.squeeze(-2) * self.window_mask
        real /= self.n_fft
        real = overlap_add(real, self.stride)
        if hasattr(self, 'w_sum') == False or refresh_win == True:
            self.w_sum = torch_window_sumsquare(self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft).flatten()
            self.nonzero_indices = self.w_sum > 1e-10
        else:
            pass
        real[:, self.nonzero_indices] = real[:, self.nonzero_indices].div(self.w_sum[self.nonzero_indices])
        if length is None:
            if self.center:
                real = real[:, self.pad_amount:-self.pad_amount]
        elif self.center:
            real = real[:, self.pad_amount:self.pad_amount + length]
        else:
            real = real[:, :length]
        return real


class STFT(STFTBase):
    """This function is to calculate the short-time Fourier transform (STFT) of the input signal.
    Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    n_fft : int
        Size of Fourier transform. Default value is 2048.

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    freq_scale : 'linear', 'log', 'log2' or 'no'
        Determine the spacing between each frequency bin. When `linear`, 'log' or `log2` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the STFT kernel, if ``True``, the time index is the center of
        the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    iSTFT : bool
        To activate the iSTFT module or not. By default, it is False to save GPU memory.
        Note: The iSTFT kernel is not trainable. If you want
        a trainable iSTFT, use the iSTFT module.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Control the spectrogram output type, either ``Magnitude``, ``Complex``, or ``Phase``.
        The output_format can also be changed during the ``forward`` method.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.
        ``shape = (num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
        ``shape = (num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.STFT()
    >>> specs = spec_layer(x)
    """

    def __init__(self, n_fft=2048, win_length=None, freq_bins=None, hop_length=None, window='hann', freq_scale='no', center=True, pad_mode='reflect', iSTFT=False, fmin=50, fmax=6000, sr=22050, trainable=False, output_format='Complex', verbose=True):
        super().__init__()
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)
        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.iSTFT = iSTFT
        self.trainable = trainable
        start = time()
        kernel_sin, kernel_cos, self.bins2freq, self.bin_list, window_mask = create_fourier_kernels(n_fft, win_length=win_length, freq_bins=freq_bins, window=window, freq_scale=freq_scale, fmin=fmin, fmax=fmax, sr=sr, verbose=verbose)
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)
        kernel_sin_inv = torch.cat((kernel_sin, -kernel_sin[1:-1].flip(0)), 0)
        kernel_cos_inv = torch.cat((kernel_cos, kernel_cos[1:-1].flip(0)), 0)
        if iSTFT:
            self.register_buffer('kernel_sin_inv', kernel_sin_inv.unsqueeze(-1))
            self.register_buffer('kernel_cos_inv', kernel_cos_inv.unsqueeze(-1))
        window_mask = torch.tensor(window_mask)
        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask
        if self.trainable == False:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
        if self.trainable == True:
            wsin = nn.Parameter(wsin, requires_grad=self.trainable)
            wcos = nn.Parameter(wcos, requires_grad=self.trainable)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)
        self.register_buffer('window_mask', window_mask.unsqueeze(0).unsqueeze(-1))
        if verbose == True:
            None
        else:
            pass

    def forward(self, x, output_format=None):
        """
        Convert a batch of waveforms to spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        output_format : str
            Control the type of spectrogram to be return. Can be either ``Magnitude`` or ``Complex`` or ``Phase``.
            Default value is ``Complex``.

        """
        output_format = output_format or self.output_format
        self.num_samples = x.shape[-1]
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.pad_amount, 0)
            elif self.pad_mode == 'reflect':
                if self.num_samples < self.pad_amount:
                    raise AssertionError('Signal length shorter than reflect padding length (n_fft // 2).')
                padding = nn.ReflectionPad1d(self.pad_amount)
            x = padding(x)
        spec_imag = conv1d(x, self.wsin, stride=self.stride)
        spec_real = conv1d(x, self.wcos, stride=self.stride)
        spec_real = spec_real[:, :self.freq_bins, :]
        spec_imag = spec_imag[:, :self.freq_bins, :]
        if output_format == 'Magnitude':
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable == True:
                return torch.sqrt(spec + 1e-08)
            else:
                return torch.sqrt(spec)
        elif output_format == 'Complex':
            return torch.stack((spec_real, -spec_imag), -1)
        elif output_format == 'Phase':
            return torch.atan2(-spec_imag + 0.0, spec_real)

    def inverse(self, X, onesided=True, length=None, refresh_win=True):
        """
        This function is same as the :func:`~nnAudio.Spectrogram.iSTFT` class,
        which is to convert spectrograms back to waveforms.
        It only works for the complex value spectrograms. If you have the magnitude spectrograms,
        please use :func:`~nnAudio.Spectrogram.Griffin_Lim`.

        Parameters
        ----------
        onesided : bool
            If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
            else use ``onesided=False``

        length : int
            To make sure the inverse STFT has the same output length of the original waveform, please
            set `length` as your intended waveform length. By default, ``length=None``,
            which will remove ``n_fft//2`` samples from the start and the end of the output.

        refresh_win : bool
            Recalculating the window sum square. If you have an input with fixed number of timesteps,
            you can increase the speed by setting ``refresh_win=False``. Else please keep ``refresh_win=True``


        """
        if hasattr(self, 'kernel_sin_inv') != True or hasattr(self, 'kernel_cos_inv') != True:
            raise NameError('Please activate the iSTFT module by setting `iSTFT=True` if you want to use `inverse`')
        assert X.dim() == 4, 'Inverse iSTFT only works for complex number,make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2).\nIf you have a magnitude spectrogram, please consider using Griffin-Lim.'
        return self.inverse_stft(X, self.kernel_cos_inv, self.kernel_sin_inv, onesided, length, refresh_win)

    def extra_repr(self) ->str:
        return 'n_fft={}, Fourier Kernel size={}, iSTFT={}, trainable={}'.format(self.n_fft, (*self.wsin.shape,), self.iSTFT, self.trainable)


def fft_frequencies(sr=22050, n_fft=2048):
    """Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    """
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])
    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.
    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])
    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """
    This function is cloned from librosa 0.7.
    Please refer to the original
    `documentation <https://librosa.org/doc/latest/generated/librosa.mel_frequencies.html?highlight=mel_frequencies#librosa.mel_frequencies>`__
    for more info.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.

    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).

    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).

    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])
    """
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def get_mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1, dtype=np.float32):
    """
    This function is cloned from librosa 0.7.
    Please refer to the original
    `documentation <https://librosa.org/doc/latest/generated/librosa.filters.mel.html>`__
    for more info.
    Create a Filterbank matrix to combine FFT bins into Mel-frequency bins


    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
    Clip the maximum frequency to 8KHz
    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if fmax is None:
        fmax = float(sr) / 2
    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    if norm == 1:
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        warnings.warn('Empty filters detected in mel frequency basis. Some channels will produce empty responses. Try increasing your sampling rate (and fmax) or reducing n_mels.')
    return weights


class MelSpectrogram(nn.Module):
    """This function is to calculate the Melspectrogram of the input signal.
    Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.
        It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    n_fft : int
        The window size for the STFT. Default value is 2048

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    n_mels : int
        The number of Mel filter banks. The filter banks maps the n_fft to mel bins.
        Default value is 128.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``,
        the time index is the beginning of the STFT kernel, if ``True``, the time index is the
        center of the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the
        Mel scale is logarithmic. The default value is ``False``.

    fmin : int
        The starting frequency for the lowest Mel filter bank.

    fmax : int
        The ending frequency for the highest Mel filter bank.

    norm :
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization, AKA 'slaney' default in librosa).
        Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    trainable_mel : bool
        Determine if the Mel filter banks are trainable or not. If ``True``, the gradients for Mel
        filter banks will also be calculated and the Mel filter banks will be updated during model
        training. Default value is ``False``.

    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MelSpectrogram()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, n_fft=2048, win_length=None, n_mels=128, hop_length=512, window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False, verbose=True, **kwargs):
        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power
        self.trainable_mel = trainable_mel
        self.trainable_STFT = trainable_STFT
        self.stft = STFT(n_fft=n_fft, win_length=win_length, freq_bins=None, hop_length=hop_length, window=window, freq_scale='no', center=center, pad_mode=pad_mode, sr=sr, trainable=trainable_STFT, output_format='Magnitude', verbose=verbose, **kwargs)
        start = time()
        start = time()
        mel_basis = get_mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
        mel_basis = torch.tensor(mel_basis)
        if verbose == True:
            None
            None
        else:
            pass
        if trainable_mel:
            mel_basis = nn.Parameter(mel_basis, requires_grad=trainable_mel)
            self.register_parameter('mel_basis', mel_basis)
        else:
            self.register_buffer('mel_basis', mel_basis)

    def forward(self, x):
        """
        Convert a batch of waveforms to Mel spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        x = broadcast_dim(x)
        spec = self.stft(x, output_format='Magnitude') ** self.power
        melspec = torch.matmul(self.mel_basis, spec)
        return melspec

    def extra_repr(self) ->str:
        return 'Mel filter banks size = {}, trainable_mel={}'.format((*self.mel_basis.shape,), self.trainable_mel, self.trainable_STFT)


class MFCC(nn.Module):
    """This function is to calculate the Mel-frequency cepstral coefficients (MFCCs) of the input signal.
    This algorithm first extracts Mel spectrograms from the audio clips,
    then the discrete cosine transform is calcuated to obtain the final MFCCs.
    Therefore, the Mel spectrogram part can be made trainable using
    ``trainable_mel`` and ``trainable_STFT``.
    It only support type-II DCT at the moment. Input signal should be in either of the following shapes.

    1. ``(len_audio)``

    2. ``(num_audio, len_audio)``

    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.  It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    n_mfcc : int
        The number of Mel-frequency cepstral coefficients

    norm : string
        The default value is 'ortho'. Normalization for DCT basis

    **kwargs
        Other arguments for Melspectrogram such as n_fft, n_mels, hop_length, and window

    Returns
    -------
    MFCCs : torch.tensor
        It returns a tensor of MFCCs.  shape = ``(num_samples, n_mfcc, time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MFCC()
    >>> mfcc = spec_layer(x)
    """

    def __init__(self, sr=22050, n_mfcc=20, norm='ortho', verbose=True, ref=1.0, amin=1e-10, top_db=80.0, **kwargs):
        super().__init__()
        self.melspec_layer = MelSpectrogram(sr=sr, verbose=verbose, **kwargs)
        self.m_mfcc = n_mfcc
        if amin <= 0:
            raise ParameterError('amin must be strictly positive')
        amin = torch.tensor([amin])
        ref = torch.abs(torch.tensor([ref]))
        self.register_buffer('amin', amin)
        self.register_buffer('ref', ref)
        self.top_db = top_db
        self.n_mfcc = n_mfcc

    def _power_to_db(self, S):
        """
        Refer to https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db
        for the original implmentation.
        """
        log_spec = 10.0 * torch.log10(torch.max(S, self.amin))
        log_spec -= 10.0 * torch.log10(torch.max(self.amin, self.ref))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1)
            log_spec = torch.max(log_spec, batch_wise_max - self.top_db)
        return log_spec

    def _dct(self, x, norm=None):
        """
        Refer to https://github.com/zh217/torch-dct for the original implmentation.
        """
        x = x.permute(0, 2, 1)
        x_shape = x.shape
        N = x_shape[-1]
        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip([2])], dim=2)
        Vc = rfft_fn(v, 1, onesided=False)
        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        V = Vc[:, :, :, 0] * W_r - Vc[:, :, :, 1] * W_i
        if norm == 'ortho':
            V[:, :, 0] /= np.sqrt(N) * 2
            V[:, :, 1:] /= np.sqrt(N / 2) * 2
        V = 2 * V
        return V.permute(0, 2, 1)

    def forward(self, x):
        """
        Convert a batch of waveforms to MFCC.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        x = self.melspec_layer(x)
        x = self._power_to_db(x)
        x = self._dct(x, norm='ortho')[:, :self.m_mfcc, :]
        return x

    def extra_repr(self) ->str:
        return 'n_mfcc = {}'.format(self.n_mfcc)


class iSTFT(STFTBase):
    """This class is to convert spectrograms back to waveforms. It only works for the complex value spectrograms.
    If you have the magnitude spectrograms, please use :func:`~nnAudio.Spectrogram.Griffin_Lim`.
    The parameters (e.g. n_fft, window) need to be the same as the STFT in order to obtain the correct inverse.
    If trainability is not required, it is recommended to use the ``inverse`` method under the ``STFT`` class
    to save GPU/RAM memory.

    When ``trainable=True`` and ``freq_scale!='no'``, there is no guarantee that the inverse is perfect, please
    use with extra care.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins
        Please make sure the value is the same as the forward STFT.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``.
        Please make sure the value is the same as the forward STFT.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable_kernels : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.

    trainable_window : bool
        Determine if the window function is trainable or not.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
        It returns a batch of waveforms.

    Examples
    --------
    >>> spec_layer = Spectrogram.iSTFT()
    >>> specs = spec_layer(x)
    """

    def __init__(self, n_fft=2048, win_length=None, freq_bins=None, hop_length=None, window='hann', freq_scale='no', center=True, fmin=50, fmax=6000, sr=22050, trainable_kernels=False, trainable_window=False, verbose=True, refresh_win=True):
        super().__init__()
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)
        self.n_fft = n_fft
        self.win_length = win_length
        self.stride = hop_length
        self.center = center
        self.pad_amount = self.n_fft // 2
        self.refresh_win = refresh_win
        start = time()
        kernel_sin, kernel_cos, _, _, window_mask = create_fourier_kernels(n_fft, win_length=win_length, freq_bins=n_fft, window=window, freq_scale=freq_scale, fmin=fmin, fmax=fmax, sr=sr, verbose=False)
        window_mask = get_window(window, int(win_length), fftbins=True)
        window_mask = torch.tensor(window_mask).unsqueeze(0).unsqueeze(-1)
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float).unsqueeze(-1)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float).unsqueeze(-1)
        if trainable_kernels:
            kernel_sin = nn.Parameter(kernel_sin, requires_grad=trainable_kernels)
            kernel_cos = nn.Parameter(kernel_cos, requires_grad=trainable_kernels)
            self.register_parameter('kernel_sin', kernel_sin)
            self.register_parameter('kernel_cos', kernel_cos)
        else:
            self.register_buffer('kernel_sin', kernel_sin)
            self.register_buffer('kernel_cos', kernel_cos)
        if trainable_window:
            window_mask = nn.Parameter(window_mask, requires_grad=trainable_window)
            self.register_parameter('window_mask', window_mask)
        else:
            self.register_buffer('window_mask', window_mask)
        if verbose == True:
            None
        else:
            pass

    def forward(self, X, onesided=False, length=None, refresh_win=None):
        """
        If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
        else use ``onesided=False``
        To make sure the inverse STFT has the same output length of the original waveform, please
        set `length` as your intended waveform length. By default, ``length=None``,
        which will remove ``n_fft//2`` samples from the start and the end of the output.
        If your input spectrograms X are of the same length, please use ``refresh_win=None`` to increase
        computational speed.
        """
        if refresh_win == None:
            refresh_win = self.refresh_win
        assert X.dim() == 4, 'Inverse iSTFT only works for complex number,make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)'
        return self.inverse_stft(X, self.kernel_cos, self.kernel_sin, onesided, length, refresh_win)


class VQT(torch.nn.Module):

    def __init__(self, sr=22050, hop_length=512, fmin=32.7, fmax=None, n_bins=84, filter_scale=1, bins_per_octave=12, norm=True, basis_norm=1, gamma=0, window='hann', pad_mode='reflect', earlydownsample=True, trainable=False, output_format='Magnitude', verbose=True):
        super().__init__()
        self.norm = norm
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.sr = sr
        self.gamma = gamma
        self.basis_norm = basis_norm
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)
        if verbose == True:
            None
        start = time()
        lowpass_filter = torch.tensor(create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.001))
        self.register_buffer('lowpass_filter', lowpass_filter[None, None, :])
        if verbose == True:
            None
        n_filters = min(bins_per_octave, n_bins)
        self.n_filters = n_filters
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose == True:
            None
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        if remainder == 0:
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)
        self.fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
        if fmax_t > sr / 2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency,                             please reduce the n_bins'.format(fmax_t))
        if self.earlydownsample == True:
            if verbose == True:
                None
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, self.earlydownsample = get_early_downsample_params(sr, hop_length, fmax_t, Q, self.n_octaves, verbose)
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            if verbose == True:
                None
        else:
            self.downsample_factor = 1.0
        alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        lenghts = np.ceil(Q * sr / (freqs + gamma / alpha))
        max_len = int(max(lenghts))
        self.n_fft = int(2 ** np.ceil(np.log2(max_len)))
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)
        my_sr = self.sr
        for i in range(self.n_octaves):
            if i > 0:
                my_sr /= 2
            Q = float(self.filter_scale) / (2 ** (1 / self.bins_per_octave) - 1)
            basis, self.n_fft, lengths, _ = create_cqt_kernels(Q, my_sr, self.fmin_t * 2 ** -i, self.n_filters, self.bins_per_octave, norm=self.basis_norm, topbin_check=False, gamma=self.gamma)
            cqt_kernels_real = torch.tensor(basis.real.astype(np.float32)).unsqueeze(1)
            cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32)).unsqueeze(1)
            self.register_buffer('cqt_kernels_real_{}'.format(i), cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag_{}'.format(i), cqt_kernels_imag)

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to VQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.

            1. ``(len_audio)``

            2. ``(num_audio, len_audio)``

            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        if self.earlydownsample == True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        vqt = []
        x_down = x
        my_sr = self.sr
        for i in range(self.n_octaves):
            if i > 0:
                x_down = downsampling_by_2(x_down, self.lowpass_filter)
                hop //= 2
            else:
                x_down = x
            if self.pad_mode == 'constant':
                my_padding = nn.ConstantPad1d(getattr(self, 'cqt_kernels_real_{}'.format(i)).shape[-1] // 2, 0)
            elif self.pad_mode == 'reflect':
                my_padding = nn.ReflectionPad1d(getattr(self, 'cqt_kernels_real_{}'.format(i)).shape[-1] // 2)
            cur_vqt = get_cqt_complex(x_down, getattr(self, 'cqt_kernels_real_{}'.format(i)), getattr(self, 'cqt_kernels_imag_{}'.format(i)), hop, my_padding)
            vqt.insert(0, cur_vqt)
        vqt = torch.cat(vqt, dim=1)
        vqt = vqt[:, -self.n_bins:, :]
        vqt = vqt * self.downsample_factor
        if normalization_type == 'librosa':
            vqt = vqt * torch.sqrt(self.lenghts.view(-1, 1, 1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            vqt *= 2
        else:
            raise ValueError('The normalization_type %r is not part of our current options.' % normalization_type)
        if output_format == 'Magnitude':
            if self.trainable == False:
                return torch.sqrt(vqt.pow(2).sum(-1))
            else:
                return torch.sqrt(vqt.pow(2).sum(-1) + 1e-08)
        elif output_format == 'Complex':
            return vqt
        elif output_format == 'Phase':
            phase_real = torch.cos(torch.atan2(vqt[:, :, :, 1], vqt[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(vqt[:, :, :, 1], vqt[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)

