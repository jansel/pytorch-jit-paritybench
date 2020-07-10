import sys
_module = sys.modules[__name__]
del sys
asteroid = _module
data = _module
dns_dataset = _module
librimix_dataset = _module
musdb18_dataset = _module
wham_dataset = _module
whamr_dataset = _module
wsj0_mix = _module
deprecation_utils = _module
engine = _module
optimizers = _module
system = _module
filterbanks = _module
analytic_free_fb = _module
enc_dec = _module
free_fb = _module
griffin_lim = _module
inputs_and_masks = _module
multiphase_gammatone_fb = _module
param_sinc_fb = _module
stft_fb = _module
transforms = _module
losses = _module
cluster = _module
mse = _module
multi_scale_spectral = _module
pit_wrapper = _module
pmsqe = _module
sdr = _module
stoi = _module
masknn = _module
activations = _module
blocks = _module
consistency = _module
convolutional = _module
norms = _module
recurrent = _module
metrics = _module
models = _module
base_models = _module
conv_tasnet = _module
dprnn_tasnet = _module
torch_utils = _module
utils = _module
conf = _module
denoise = _module
eval_on_synthetic = _module
preprocess_dns = _module
model = _module
train = _module
eval = _module
create_local_metadata = _module
model = _module
train = _module
test_dataloader = _module
start_evaluation = _module
eval = _module
preprocess_wham = _module
model = _module
train = _module
eval = _module
model = _module
train = _module
eval = _module
augmented_wham = _module
resample_dataset = _module
model = _module
train = _module
get_training_stats = _module
model = _module
train = _module
eval = _module
model = _module
system = _module
train = _module
eval = _module
preprocess_whamr = _module
model = _module
train = _module
eval = _module
preprocess_wsj0mix = _module
model = _module
train = _module
hubconf = _module
setup = _module
optimizers_test = _module
system_test = _module
filterbanks_test = _module
stft_test = _module
transforms_test = _module
loss_functions_test = _module
pit_wrapper_test = _module
activations_test = _module
blocks_test = _module
consistency_test = _module
norms_test = _module
metrics_test = _module
models_test = _module
torch_utils_test = _module
utils_test = _module

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
xrange = range
wraps = functools.wraps


import torch


from torch.utils import data


import numpy as np


from torch.utils.data.dataset import Dataset


import random as random


import torch.utils.data


import random


from torch.optim.optimizer import Optimizer


from torch.optim import Adam


from torch.optim import RMSprop


from torch.optim import SGD


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import Adamax


from torch.optim import AdamW


from torch.optim import ASGD


import torch.nn as nn


import warnings


from torch import nn


from torch.nn import functional as F


import math


from torch.nn.modules.loss import _Loss


from numpy import VisibleDeprecationWarning


from itertools import permutations


from torch import tensor


from scipy.io import loadmat


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import fold


from torch.nn.functional import unfold


from collections import OrderedDict


import collections


import inspect


from functools import partial


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torch.utils.data import Dataset


import torch.nn.functional as F


from sklearn.cluster import KMeans


from torch import optim


from torch.testing import assert_allclose


from torch import testing


from scipy.signal import get_window


import itertools


class Filterbank(nn.Module):
    """ Base Filterbank class.
    Each subclass has to implement a `filters` property.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the conv or transposed conv. (Hop size).
            If None (default), set to ``kernel_size // 2``.

    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(self, n_filters, kernel_size, stride=None):
        super(Filterbank, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        self.n_feats_out = n_filters

    @property
    def filters(self):
        """ Abstract method for filters. """
        raise NotImplementedError

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class. """
        config = {'fb_name': self.__class__.__name__, 'n_filters': self.n_filters, 'kernel_size': self.kernel_size, 'stride': self.stride}
        return config


class _EncDec(nn.Module):
    """ Base private class for Encoder and Decoder.

    Common parameters and methods.

    Args:
        filterbank (:class:`Filterbank`): Filterbank instance. The filterbank
            to use as an encoder or a decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.

    Attributes:
        filterbank (:class:`Filterbank`)
        stride (int)
        is_pinv (bool)
    """

    def __init__(self, filterbank, is_pinv=False):
        super(_EncDec, self).__init__()
        self.filterbank = filterbank
        self.stride = self.filterbank.stride
        self.is_pinv = is_pinv

    @property
    def filters(self):
        return self.filterbank.filters

    def compute_filter_pinv(self, filters):
        """ Computes pseudo inverse filterbank of given filters."""
        scale = self.filterbank.stride / self.filterbank.kernel_size
        shape = filters.shape
        ifilt = torch.pinverse(filters.squeeze()).transpose(-1, -2).view(shape)
        return ifilt * scale

    def get_filters(self):
        """ Returns filters or pinv filters depending on `is_pinv` attribute """
        if self.is_pinv:
            return self.compute_filter_pinv(self.filters)
        else:
            return self.filters

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {'is_pinv': self.is_pinv}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(_EncDec):
    """ Decoder class.
    
    Add decoding methods to Filterbank classes.
    Not intended to be subclassed.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use as an decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        padding (int): Zero-padding added to both sides of the input.
        output_padding (int): Additional size added to one side of the
            output shape.

    Notes
        `padding` and `output_padding` arguments are directly passed to
        F.conv_transpose1d.
    """

    def __init__(self, filterbank, is_pinv=False, padding=0, output_padding=0):
        super().__init__(filterbank, is_pinv=is_pinv)
        self.padding = padding
        self.output_padding = output_padding

    @classmethod
    def pinv_of(cls, filterbank):
        """ Returns an Decoder, pseudo inverse of a filterbank or Encoder."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True)
        elif isinstance(filterbank, Encoder):
            return cls(filterbank.filterbank, is_pinv=True)

    def forward(self, spec):
        """ Applies transposed convolution to a TF representation.

        This is equivalent to overlap-add.

        Args:
            spec (:class:`torch.Tensor`): 3D or 4D Tensor. The TF
                representation. (Output of :func:`Encoder.forward`).
        Returns:
            :class:`torch.Tensor`: The corresponding time domain signal.
        """
        filters = self.get_filters()
        if spec.ndim == 2:
            return F.conv_transpose1d(spec.unsqueeze(0), filters, stride=self.stride, padding=self.padding, output_padding=self.output_padding).squeeze()
        if spec.ndim == 3:
            return F.conv_transpose1d(spec, filters, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        elif spec.ndim > 3:
            view_as = (-1,) + spec.shape[-2:]
            out = F.conv_transpose1d(spec.view(view_as), filters, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            return out.view(spec.shape[:-2] + (-1,))


class Encoder(_EncDec):
    """ Encoder class.

    Add encoding methods to Filterbank classes.
    Not intended to be subclassed.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use
            as an encoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        as_conv1d (bool): Whether to behave like nn.Conv1d.
            If True (default), forwarding input with shape (batch, 1, time)
            will output a tensor of shape (batch, freq, conv_time).
            If False, will output a tensor of shape (batch, 1, freq, conv_time).
        padding (int): Zero-padding added to both sides of the input.

    Notes:
        (time, ) --> (freq, conv_time)
        (batch, time) --> (batch, freq, conv_time)  # Avoid
        if as_conv1d:
            (batch, 1, time) --> (batch, freq, conv_time)
            (batch, chan, time) --> (batch, chan, freq, conv_time)
        else:
            (batch, chan, time) --> (batch, chan, freq, conv_time)
        (batch, any, dim, time) --> (batch, any, dim, freq, conv_time)

    """

    def __init__(self, filterbank, is_pinv=False, as_conv1d=True, padding=0):
        super(Encoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.padding = padding

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        """ Returns an :class:`~.Encoder`, pseudo inverse of a
        :class:`~.Filterbank` or :class:`~.Decoder`."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True, **kwargs)
        elif isinstance(filterbank, Decoder):
            return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def forward(self, waveform):
        """ Convolve 1D torch.Tensor with the filters from a filterbank."""
        filters = self.get_filters()
        if waveform.ndim == 1:
            return F.conv1d(waveform[None, None], filters, stride=self.stride, padding=self.padding).squeeze()
        elif waveform.ndim == 2:
            warnings.warn('Input tensor was 2D. Applying the corresponding Decoder to the current output will result in a 3D tensor. This behaviours was introduced to match Conv1D and ConvTranspose1D, please use 3D inputs to avoid it. For example, this can be done with input_tensor.unsqueeze(1).')
            return F.conv1d(waveform.unsqueeze(1), filters, stride=self.stride, padding=self.padding)
        elif waveform.ndim == 3:
            batch, channels, time_len = waveform.shape
            if channels == 1 and self.as_conv1d:
                return F.conv1d(waveform, filters, stride=self.stride, padding=self.padding)
            else:
                return self.batch_1d_conv(waveform, filters)
        else:
            return self.batch_1d_conv(waveform, filters)

    def batch_1d_conv(self, inp, filters):
        batched_conv = F.conv1d(inp.view(-1, 1, inp.shape[-1]), filters, stride=self.stride, padding=self.padding)
        output_shape = inp.shape[:-1] + batched_conv.shape[-2:]
        return batched_conv.view(output_shape)


class FreeFB(Filterbank):
    """ Free filterbank without any constraints. Equivalent to
    :class:`nn.Conv1d`.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.

    Attributes:
        n_feats_out (int): Number of output filters.

    References:
        [1] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, stride=None, **kwargs):
        super(FreeFB, self).__init__(n_filters, kernel_size, stride=stride)
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    @property
    def filters(self):
        return self._filters


def erb_scale_2_freq_hz(freq_erb):
    """ Convert frequency on ERB scale to frequency in Hertz """
    freq_hz = (np.exp(freq_erb / 9.265) - 1) * 24.7 * 9.265
    return freq_hz


def freq_hz_2_erb_scale(freq_hz):
    """ Convert frequency in Hertz to frequency on ERB scale """
    freq_erb = 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
    return freq_erb


def gammatone_impulse_response(samplerate_hz, len_sec, center_freq_hz, phase_shift):
    """ Generate single parametrized gammatone filter """
    p = 2
    erb = 24.7 + 0.108 * center_freq_hz
    divisor = np.pi * np.math.factorial(2 * p - 2) * np.power(2, float(-(2 * p - 2))) / np.square(np.math.factorial(p - 1))
    b = erb / divisor
    a = 1.0
    len_sample = int(np.floor(samplerate_hz * len_sec))
    t = np.linspace(1.0 / samplerate_hz, len_sec, len_sample)
    gammatone_ir = a * np.power(t, p - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * center_freq_hz * t + phase_shift)
    return gammatone_ir


def normalize_filters(filterbank):
    """ Normalizes a filterbank such that all filters
    have the same root mean square (RMS). """
    rms_per_filter = np.sqrt(np.mean(np.square(filterbank), axis=1))
    rms_normalization_values = 1.0 / (rms_per_filter / np.amax(rms_per_filter))
    normalized_filterbank = filterbank * rms_normalization_values[:, (np.newaxis)]
    return normalized_filterbank


def generate_mpgtf(samplerate_hz, len_sec, n_filters):
    center_freq_hz_min = 100
    n_center_freqs = 24
    len_sample = int(np.floor(samplerate_hz * len_sec))
    index = 0
    filterbank = np.zeros((n_filters, len_sample))
    current_center_freq_hz = center_freq_hz_min
    phase_pair_count = (np.ones(n_center_freqs) * np.floor(n_filters / 2 / n_center_freqs)).astype(int)
    remaining_phase_pairs = ((n_filters - np.sum(phase_pair_count) * 2) / 2).astype(int)
    if remaining_phase_pairs > 0:
        phase_pair_count[:remaining_phase_pairs] = phase_pair_count[:remaining_phase_pairs] + 1
    for i in range(n_center_freqs):
        for phase_index in range(phase_pair_count[i]):
            current_phase_shift = np.float(phase_index) / phase_pair_count[i] * np.pi
            filterbank[(index), :] = gammatone_impulse_response(samplerate_hz, len_sec, current_center_freq_hz, current_phase_shift)
            index = index + 1
        filterbank[index:index + phase_pair_count[i], :] = -filterbank[index - phase_pair_count[i]:index, :]
        index = index + phase_pair_count[i]
        current_center_freq_hz = erb_scale_2_freq_hz(freq_hz_2_erb_scale(current_center_freq_hz) + 1)
    filterbank = normalize_filters(filterbank)
    return filterbank


class MultiphaseGammatoneFB(Filterbank):
    """ Multi-Phase Gammatone Filterbank as described in [1].
    Please cite [1] whenever using this.
    Original code repository: `<https://github.com/sp-uhh/mp-gtf>`

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        sample_rate (int, optional): The sample rate (used for initialization).
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.

    References:
    [1] David Ditter, Timo Gerkmann, "A Multi-Phase Gammatone Filterbank for
        Speech Separation via TasNet", ICASSP 2020
        Available: `<https://ieeexplore.ieee.org/document/9053602/>`
    """

    def __init__(self, n_filters=128, kernel_size=16, sample_rate=8000, stride=None, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.n_feats_out = n_filters
        length_in_seconds = kernel_size / sample_rate
        mpgtf = generate_mpgtf(sample_rate, length_in_seconds, n_filters)
        filters = torch.from_numpy(mpgtf).unsqueeze(1).float()
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters


class ParamSincFB(Filterbank):
    """Extension of the parameterized filterbank from [1] proposed in [2].
    Modified and extended from from `<https://github.com/mravanelli/SincNet>`__

    Args:
        n_filters (int): Number of filters. Half of `n_filters` (the real
            parts) will have parameters, the other half will correspond to the
            imaginary parts. `n_filters` should be even.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution. If None (default),
            set to ``kernel_size // 2``.
        sample_rate (int, optional): The sample rate (used for initialization).
        min_low_hz (int, optional): Lowest low frequency allowed (Hz).
        min_band_hz (int, optional): Lowest band frequency allowed (Hz).

    Attributes:
        n_feats_out (int): Number of output filters.

    References:
        [1] : "Speaker Recognition from raw waveform with SincNet". SLT 2018.
        Mirco Ravanelli, Yoshua Bengio.  https://arxiv.org/abs/1808.00158

        [2] : "Filterbank design for end-to-end speech separation".
        Submitted to ICASSP 2020. Manuel Pariente, Samuele Cornell,
        Antoine Deleforge, Emmanuel Vincent. https://arxiv.org/abs/1910.10400
    """

    def __init__(self, n_filters, kernel_size, stride=None, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        if kernel_size % 2 == 0:
            None
            kernel_size += 1
        super(ParamSincFB, self).__init__(n_filters, kernel_size, stride=stride)
        self.sample_rate = sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        self.half_kernel = self.kernel_size // 2
        self.cutoff = int(n_filters // 2)
        self.n_feats_out = 2 * self.cutoff
        self._initialize_filters()
        if n_filters % 2 != 0:
            None
        window_ = np.hamming(self.kernel_size)[:self.half_kernel]
        n_ = 2 * np.pi * (torch.arange(-self.half_kernel, 0.0).view(1, -1) / self.sample_rate)
        self.register_buffer('window_', torch.from_numpy(window_).float())
        self.register_buffer('n_', n_)

    def _initialize_filters(self):
        """ Filter Initialization along the Mel scale"""
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.n_filters // 2 + 1, dtype='float32')
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).view(-1, 1))

    @property
    def filters(self):
        """ Compute filters from parameters """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        cos_filters = self.make_filters(low, high, filt_type='cos')
        sin_filters = self.make_filters(low, high, filt_type='sin')
        return torch.cat([cos_filters, sin_filters], dim=0)

    def make_filters(self, low, high, filt_type='cos'):
        band = (high - low)[:, (0)]
        ft_low = torch.matmul(low, self.n_)
        ft_high = torch.matmul(high, self.n_)
        if filt_type == 'cos':
            bp_left = (torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2) * self.window_
            bp_center = 2 * band.view(-1, 1)
            bp_right = torch.flip(bp_left, dims=[1])
        elif filt_type == 'sin':
            bp_left = (torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2) * self.window_
            bp_center = torch.zeros_like(band.view(-1, 1))
            bp_right = -torch.flip(bp_left, dims=[1])
        else:
            raise ValueError('Invalid filter type {}'.format(filt_type))
        band_pass = torch.cat([bp_left, bp_center, bp_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        return band_pass.view(self.n_filters // 2, 1, self.kernel_size)

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {'sample_rate': self.sample_rate, 'min_low_hz': self.min_low_hz, 'min_band_hz': self.min_band_hz}
        base_config = super(ParamSincFB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class STFTFB(Filterbank):
    """ STFT filterbank.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.

    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(self, n_filters, kernel_size, stride=None, window=None, **kwargs):
        super(STFTFB, self).__init__(n_filters, kernel_size, stride=stride)
        assert n_filters >= kernel_size
        self.cutoff = int(n_filters / 2 + 1)
        self.n_feats_out = 2 * self.cutoff
        if window is None:
            self.window = np.hanning(kernel_size + 1)[:-1] ** 0.5
        else:
            ws = window.size
            if not ws == kernel_size:
                raise AssertionError('Expected window of size {}.Received window of size {} instead.'.format(kernel_size, ws))
            self.window = window
        filters = np.fft.fft(np.eye(n_filters))
        filters /= 0.5 * np.sqrt(kernel_size * n_filters / self.stride)
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        indexes = list(range(lpad, n_filters - rpad))
        filters = np.vstack([np.real(filters[:self.cutoff, (indexes)]), np.imag(filters[:self.cutoff, (indexes)])])
        filters[(0), :] /= np.sqrt(2)
        filters[(n_filters // 2), :] /= np.sqrt(2)
        filters = torch.from_numpy(filters * self.window).unsqueeze(1).float()
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters


class PairwiseMSE(_Loss):
    """ Measure pairwise mean square error on a batch.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, nsrc, *].
            The batch of training targets

    Returns:
        :class:`torch.Tensor`: with shape [batch, nsrc, nsrc]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(PairwiseMSE(), pit_from='pairwise')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        targets = targets.unsqueeze(1)
        est_targets = est_targets.unsqueeze(2)
        pw_loss = (targets - est_targets) ** 2
        mean_over = list(range(3, pw_loss.ndim))
        return pw_loss.mean(dim=mean_over)


class SingleSrcMSE(_Loss):
    """ Measure mean square error on a batch.
    Supports both tensors with and without source axis.

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, *].
            The batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, *].
            The batch of training targets.

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples:

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # singlesrc_mse / multisrc_mse support both 'pw_pt' and 'perm_avg'.
        >>> loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)
    """

    def forward(self, est_targets, targets):
        loss = (targets - est_targets) ** 2
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)


class DeprecationMixin:
    """ Deprecation mixin. Example to come """

    def warn_deprecated(self):
        warnings.warn('{} is deprecated since v0.1.0, it will be removed in v0.2.0. Please use {} instead.'.format(self.__class__.__name__, self.__class__.__bases__[0].__name__), VisibleDeprecationWarning)


class NoSrcMSE(SingleSrcMSE, DeprecationMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


EPS = 1e-08


def check_complex(tensor, dim=-2):
    """ Assert tensor in complex-like in a given dimension.

    Args:
        tensor (torch.Tensor): tensor to be checked.
        dim(int): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.

    Raises:
        AssertionError if dimension is not even in the specified dimension

    """
    if tensor.shape[dim] % 2 != 0:
        raise AssertionError('Could not equally chunk the tensor (shape {}) along the given dimension ({}). Dim axis is probably wrong')


def take_mag(x, dim=-2):
    """ Takes the magnitude of a complex tensor.

    The operands is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.

    Returns:
        :class:`torch.Tensor`: The magnitude of x.
    """
    check_complex(x, dim=dim)
    power = torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)
    power = power + EPS
    return power.pow(0.5)


class SingleSrcMultiScaleSpectral(_Loss):
    """ Measure multi-scale spectral loss as described in [1]

    Args:
        n_filters (list): list containing the number of filter desired for
            each STFT
        windows_size (list): list containing the size of the window desired for
            each STFT
        hops_size (list): list containing the size of the hop desired for
            each STFT

    Shape:
        est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of target estimates.
        targets (:class:`torch.Tensor`): Expected shape [batch, time].
            Batch of training targets.
        alpha (float) : Weighting factor for the log term

    Returns:
        :class:`torch.Tensor`: with shape [batch]

    Examples:
        >>> import torch
        >>> targets = torch.randn(10, 32000)
        >>> est_targets = torch.randn(10, 32000)
        >>> # Using it by itself on a pair of source/estimate
        >>> loss_func = SingleSrcMultiScaleSpectral()
        >>> loss = loss_func(est_targets, targets)

        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> targets = torch.randn(10, 2, 32000)
        >>> est_targets = torch.randn(10, 2, 32000)
        >>> # Using it with PITLossWrapper with sets of source/estimates
        >>> loss_func = PITLossWrapper(SingleSrcMultiScaleSpectral(),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(est_targets, targets)

    References:
        [1] Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and
        Adam Roberts DDSP: Differentiable Digital Signal Processing
        International Conference on Learning Representations ICLR 2020 $
    """

    def __init__(self, n_filters=None, windows_size=None, hops_size=None, alpha=1.0):
        super().__init__()
        if windows_size is None:
            windows_size = [2048, 1024, 512, 256, 128, 64, 32]
        if n_filters is None:
            n_filters = [2048, 1024, 512, 256, 128, 64, 32]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128, 64, 32, 16]
        self.windows_size = windows_size
        self.n_filters = n_filters
        self.hops_size = hops_size
        self.alpha = alpha
        self.encoders = nn.ModuleList(Encoder(STFTFB(n_filters[i], windows_size[i], hops_size[i])) for i in range(len(self.n_filters)))

    def forward(self, est_target, target):
        batch_size = est_target.shape[0]
        est_target = est_target.unsqueeze(1)
        target = target.unsqueeze(1)
        loss = torch.zeros(batch_size, device=est_target.device)
        for encoder in self.encoders:
            loss += self.compute_spectral_loss(encoder, est_target, target)
        return loss

    def compute_spectral_loss(self, encoder, est_target, target):
        batch_size = est_target.shape[0]
        spect_est_target = take_mag(encoder(est_target)).view(batch_size, -1)
        spect_target = take_mag(encoder(target)).view(batch_size, -1)
        linear_loss = self.norm1(spect_est_target - spect_target)
        log_loss = self.norm1(torch.log(spect_est_target + EPS) - torch.log(spect_target + EPS))
        return linear_loss + self.alpha * log_loss

    @staticmethod
    def norm1(a):
        return torch.norm(a, p=1, dim=1)


class PITLossWrapper(nn.Module):
    """ Permutation invariant loss wrapper.

    Args:
        loss_func: function with signature (targets, est_targets, **kwargs).
        mode (str): Determines how PIT is applied (deprecated,
            use `expects` instead.)
        pit_from (str): Determines how PIT is applied.

            * ``'pw_mtx'`` (pairwise matrix): `loss_func` computes pairwise
              losses and returns a torch.Tensor of shape
              :math:`(batch, n\\_src, n\\_src)`. Each element
              :math:`[batch, i, j]` corresponds to the loss between
              :math:`targets[:, i]` and :math:`est\\_targets[:, j]`
            * ``'pw_pt'`` (pairwise point): `loss_func` computes the loss for
              a batch of single source and single estimates (tensors won't
              have the source axis). Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.get_pw_losses`.
            * ``'perm_avg'``(permutation average): `loss_func` computes the
              average loss for a given permutations of the sources and
              estimates. Output shape : :math:`(batch)`.
              See :meth:`~PITLossWrapper.best_perm_from_perm_avg_loss`.

            In terms of efficiency, ``'perm_avg'`` is the least efficicient.

        perm_reduce (Callable): torch function to reduce permutation losses.
            Defaults to None (equivalent to mean). Signature of the func
            (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!).
            `perm_reduce` can receive **kwargs during forward using the
            `reduce_kwargs` argument (dict). If those argument are static,
            consider defining a small function or using `functools.partial`.
            Only used in `'pw_mtx'` and `'pw_pt'` `pit_from` modes.

    For each of these modes, the best permutation and reordering will be
    automatically computed.

    Examples:
        >>> import torch
        >>> from asteroid.losses import pairwise_neg_sisdr
        >>> sources = torch.randn(10, 3, 16000)
        >>> est_sources = torch.randn(10, 3, 16000)
        >>> # Compute PIT loss based on pairwise losses
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        >>> loss_val = loss_func(est_sources, sources)
        >>>
        >>> # Using reduce
        >>> def reduce(perm_loss, src):
        >>>     weighted = perm_loss * src.norm(dim=-1, keepdim=True)
        >>>     return torch.mean(weighted, dim=-1)
        >>>
        >>> loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx',
        >>>                            perm_reduce=reduce)
        >>> reduce_kwargs = {'src': sources}
        >>> loss_val = loss_func(est_sources, sources,
        >>>                      reduce_kwargs=reduce_kwargs)
    """

    def __init__(self, loss_func, pit_from='pw_mtx', mode=None, perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.mode = mode
        self.perm_reduce = perm_reduce
        if self.mode is not None:
            warnings.warn('`mode` argument is deprecated since v0.1.0 andwill be remove in v0.2.0. Use argument `pit_from`instead', VisibleDeprecationWarning)
            mapping = dict(pairwise='pw_mtx', wo_src='pw_pt', w_src='perm_avg')
            self.pit_from = mapping.get(mode, None)
        if self.pit_from not in ['pw_mtx', 'pw_pt', 'perm_avg']:
            raise ValueError('Unsupported loss function type for now. Expectedone of [`pw_mtx`, `pw_pt`, `perm_avg`]')

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        """ Find the best permutation and return the loss.

        Args:
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets
            return_est: Boolean. Whether to return the reordered targets
                estimates (To compute metrics or to save example).
            reduce_kwargs (dict or None): kwargs that will be passed to the
                pairwise losses reduce function (`perm_reduce`).
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            - Best permutation loss for each batch sample, average over
                the batch. torch.Tensor(loss_value)
            - The reordered targets estimates if return_est is True.
                torch.Tensor of shape [batch, nsrc, *].
        """
        n_src = targets.shape[1]
        assert n_src < 10, f'Expected source axis along dim 1, found {n_src}'
        if self.pit_from == 'pw_mtx':
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == 'pw_pt':
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == 'perm_avg':
            min_loss, min_loss_idx = self.best_perm_from_perm_avg_loss(self.loss_func, est_targets, targets, **kwargs)
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
            return mean_loss, reordered
        else:
            return
        assert pw_losses.ndim == 3, 'Something went wrong with the loss function, please read the docs.'
        assert pw_losses.shape[0] == targets.shape[0], 'PIT loss needs same batch dim as input'
        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, min_loss_idx = self.find_best_perm(pw_losses, n_src, perm_reduce=self.perm_reduce, **reduce_kwargs)
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        """ Get pair-wise losses between the training targets and its estimate
        for a given loss function.

        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function to get pair-wise losses from.
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            torch.Tensor or size [batch, nsrc, nsrc], losses computed for
            all permutations of the targets and est_targets.

        This function can be called on a loss function which returns a tensor
        of size [batch]. There are more efficient ways to compute pair-wise
        losses using broadcasting.
        """
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, (est_idx), (target_idx)] = loss_func(est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def best_perm_from_perm_avg_loss(loss_func, est_targets, targets, **kwargs):
        """ Find best permutation from loss function with source axis.

        Args:
            loss_func: function with signature (targets, est_targets, **kwargs)
                The loss function batch losses from.
            est_targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of target estimates.
            targets: torch.Tensor. Expected shape [batch, nsrc, *].
                The batch of training targets.
            **kwargs: additional keyword argument that will be passed to the
                loss function.

        Returns:
            tuple:
                :class:`torch.Tensor`: The loss corresponding to the best
                permutation of size (batch,).

                :class:`torch.LongTensor`: The indexes of the best permutations.
        """
        n_src = targets.shape[1]
        perms = list(permutations(range(n_src)))
        loss_set = torch.stack([loss_func(est_targets[:, (perm)], targets, **kwargs) for perm in perms], dim=1)
        min_loss, min_loss_idx = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx[:, (0)]

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src, perm_reduce=None, **kwargs):
        """Find the best permutation, given the pair-wise losses.

        Args:
            pair_wise_losses (:class:`torch.Tensor`):
                Tensor of shape [batch, n_src, n_src]. Pairwise losses.
            n_src (int): Number of sources.
            perm_reduce (Callable): torch function to reduce permutation losses.
                Defaults to None (equivalent to mean). Signature of the func
                (pwl_set, **kwargs) : (B, n_src!, n_src) --> (B, n_src!)
            **kwargs: additional keyword argument that will be passed to the
                permutation reduce function.

        Returns:
            tuple:
                :class:`torch.Tensor`: The loss corresponding to the best
                permutation of size (batch,).

                :class:`torch.LongTensor`: The indexes of the best permutations.

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        idx = torch.unsqueeze(perms, 2)
        if perm_reduce is None:
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            batch = pwl.shape[0]
            n_perm = idx.shape[0]
            pwl_set = pwl[:, (torch.arange(n_src)), (idx.squeeze(-1))]
            loss_set = perm_reduce(pwl_set, **kwargs)
        min_loss_idx = torch.argmin(loss_set, dim=1)
        min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx

    @staticmethod
    def reorder_source(source, n_src, min_loss_idx):
        """ Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape [batch, n_src, time]
            n_src (int): Number of sources.
            min_loss_idx (torch.LongTensor): Tensor of shape [batch],
                each item is in [0, n_src!).

        Returns:
            :class:`torch.Tensor`:
                Reordered sources of shape [batch, n_src, time].

        MIT Copyright (c) 2018 Kaituo XU.
        See `Original code
        <https://github.com/kaituoxu/Conv-TasNet/blob/master>`__ and `License
        <https://github.com/kaituoxu/Conv-TasNet/blob/master/LICENSE>`__.
        """
        perms = source.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        min_loss_perm = torch.index_select(perms, dim=0, index=min_loss_idx)
        reordered_sources = torch.zeros_like(source)
        for b in range(source.shape[0]):
            for c in range(n_src):
                reordered_sources[b, c] = source[b, min_loss_perm[b][c]]
        return reordered_sources


class SingleSrcPMSQE(nn.Module):
    """ Computes the Perceptual Metric for Speech Quality Evaluation (PMSQE)
    as described in [1].
    This version is only designed for 16 kHz (512 length DFT).
    Adaptation to 8 kHz could be done by changing the parameters of the
    class (see Tensorflow implementation).
    The SLL, frequency and gain equalization are applied in each
    sequence independently.

    Parameters:
        window_name (str): Select the used window function for the correct
            factor to be applied. Defaults to sqrt hanning window.
            Among ['rect', 'hann', 'sqrt_hann', 'hamming', 'flatTop'].
        window_weight (float, optional): Correction to the window factor
            applied.
        bark_eq (bool, optional): Whether to apply bark equalization.
        gain_eq (bool, optional): Whether to apply gain equalization.
        sample_rate (int): Sample rate of the input audio.

    References:
        [1] J.M.Martin, A.M.Gomez, J.A.Gonzalez, A.M.Peinado 'A Deep Learning
        Loss Function based on the Perceptual Evaluation of the
        Speech Quality', IEEE Signal Processing Letters, 2018.
        Implemented by Juan M. Martin. Contact: mdjuamart@ugr.es
        Copyright 2019: University of Granada, Signal Processing, Multimedia
        Transmission and Speech/Audio Technologies (SigMAT) Group.

    Notes:
        Inspired on the Perceptual Evaluation of the Speech Quality (PESQ)
        algorithm, this function consists of two regularization factors :
        the symmetrical and asymmetrical distortion in the loudness domain.

    Examples:
        >>> import torch
        >>> from asteroid.filterbanks import STFTFB, Encoder, transforms
        >>> from asteroid.losses import PITLossWrapper, SingleSrcPMSQE
        >>> stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
        >>> # Usage by itself
        >>> ref, est = torch.randn(2, 1, 16000), torch.randn(2, 1, 16000)
        >>> ref_spec = transforms.take_mag(stft(ref))
        >>> est_spec = transforms.take_mag(stft(est))
        >>> loss_func = SingleSrcPMSQE()
        >>> loss_value = loss_func(est_spec, ref_spec)
        >>> # Usage with PITLossWrapper
        >>> loss_func = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_pt')
        >>> ref, est = torch.randn(2, 3, 16000), torch.randn(2, 3, 16000)
        >>> ref_spec = transforms.take_mag(stft(ref))
        >>> est_spec = transforms.take_mag(stft(est))
        >>> loss_value = loss_func(ref_spec, est_spec)
    """

    def __init__(self, window_name='sqrt_hann', window_weight=1.0, bark_eq=True, gain_eq=True, sample_rate=16000):
        super().__init__()
        self.window_name = window_name
        self.window_weight = window_weight
        self.bark_eq = bark_eq
        self.gain_eq = gain_eq
        if sample_rate not in [16000, 8000]:
            raise ValueError('Unsupported sample rate {}'.format(sample_rate))
        self.sample_rate = sample_rate
        if sample_rate == 16000:
            self.Sp = 6.910853e-06
            self.Sl = 0.1866055
            self.nbins = 512
            self.nbark = 49
        else:
            self.Sp = 2.764344e-05
            self.Sl = 0.1866055
            self.nbins = 256
            self.nbark = 42
        self.alpha = 0.1
        self.beta = 0.309 * self.alpha
        pow_correc_factor = self.get_correction_factor(window_name)
        self.pow_correc_factor = pow_correc_factor * self.window_weight
        self.abs_thresh_power = None
        self.modified_zwicker_power = None
        self.width_of_band_bark = None
        self.bark_matrix = None
        self.mask_sll = None
        self.populate_constants(self.sample_rate)
        self.sqrt_total_width = torch.sqrt(torch.sum(self.width_of_band_bark))

    def forward(self, est_targets, targets, pad_mask=None):
        """
        Args
            est_targets (torch.Tensor): Dimensions (B, T, F).
                Padded degraded power spectrum in time-frequency domain.
            targets (torch.Tensor): Dimensions (B, T, F).
                Zero-Padded reference power spectrum in time-frequency domain.
            pad_mask (torch.Tensor, optional):  Dimensions (B, T, 1). Mask
                to indicate the padding frames. Defaults to all ones.

        Dimensions
            B: Number of sequences in the batch.
            T: Number of time frames.
            F: Number of frequency bins.

        Returns
            torch.tensor of shape (B, ), wD + 0.309 * wDA

        Notes
            Dimensions (B, F, T) are also supported by SingleSrcPMSQE but are
            less efficient because input tensors are transposed (not inplace).

        Examples

        """
        assert est_targets.shape == targets.shape
        try:
            freq_idx = est_targets.shape.index(self.nbins // 2 + 1)
        except ValueError:
            raise ValueError('Could not find dimension with {} elements in input tensors, verify your inputs'.format(self.nbins // 2 + 1))
        if freq_idx == 1:
            est_targets = est_targets.transpose(1, 2)
            targets = targets.transpose(1, 2)
        if pad_mask is not None:
            pad_mask = pad_mask.transpose(1, 2) if freq_idx == 1 else pad_mask
        else:
            pad_mask = torch.ones(est_targets.shape[0], est_targets.shape[1], 1)
        ref_spectra = self.magnitude_at_sll(targets, pad_mask)
        deg_spectra = self.magnitude_at_sll(est_targets, pad_mask)
        ref_bark_spectra = self.bark_computation(ref_spectra)
        deg_bark_spectra = self.bark_computation(deg_spectra)
        if self.bark_eq:
            deg_bark_spectra = self.bark_freq_equalization(ref_bark_spectra, deg_bark_spectra)
        if self.gain_eq:
            deg_bark_spectra = self.bark_gain_equalization(ref_bark_spectra, deg_bark_spectra)
        sym_d, asym_d = self.compute_distortion_tensors(ref_bark_spectra, deg_bark_spectra)
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        wd_frame, wda_frame = self.per_frame_distortion(sym_d, asym_d, audible_power_ref)
        dims = [-1, -2]
        pmsqe_frame = (self.alpha * wd_frame + self.beta * wda_frame) * pad_mask
        pmsqe = torch.sum(pmsqe_frame, dim=dims) / pad_mask.sum(dims)
        return pmsqe

    def magnitude_at_sll(self, spectra, pad_mask):
        masked_spectra = spectra * pad_mask * self.mask_sll
        freq_mean_masked_spectra = torch.mean(masked_spectra, dim=-1, keepdim=True)
        sum_spectra = torch.sum(freq_mean_masked_spectra, dim=-2, keepdim=True)
        seq_len = torch.sum(pad_mask, dim=-2, keepdim=True)
        mean_pow = sum_spectra / seq_len
        return 10000000.0 * spectra / mean_pow

    def bark_computation(self, spectra):
        return self.Sp * torch.matmul(spectra, self.bark_matrix)

    def compute_audible_power(self, bark_spectra, factor=1.0):
        thr_bark = torch.where(bark_spectra > self.abs_thresh_power * factor, bark_spectra, torch.zeros_like(bark_spectra))
        return torch.sum(thr_bark, dim=-1, keepdim=True)

    def bark_gain_equalization(self, ref_bark_spectra, deg_bark_spectra):
        audible_power_ref = self.compute_audible_power(ref_bark_spectra, 1.0)
        audible_power_deg = self.compute_audible_power(deg_bark_spectra, 1.0)
        gain = (audible_power_ref + 5000.0) / (audible_power_deg + 5000.0)
        limited_gain = torch.min(gain, 5.0 * torch.ones_like(gain))
        limited_gain = torch.max(limited_gain, 0.0003 * torch.ones_like(limited_gain))
        return limited_gain * deg_bark_spectra

    def bark_freq_equalization(self, ref_bark_spectra, deg_bark_spectra):
        """This version is applied in the degraded directly."""
        audible_power_x100 = self.compute_audible_power(ref_bark_spectra, 100.0)
        not_silent = audible_power_x100 >= 10000000.0
        cond_thr = ref_bark_spectra >= self.abs_thresh_power * 100.0
        ref_thresholded = torch.where(cond_thr, ref_bark_spectra, torch.zeros_like(ref_bark_spectra))
        deg_thresholded = torch.where(cond_thr, deg_bark_spectra, torch.zeros_like(deg_bark_spectra))
        avg_ppb_ref = torch.sum(torch.where(not_silent, ref_thresholded, torch.zeros_like(ref_thresholded)), dim=-2, keepdim=True)
        avg_ppb_deg = torch.sum(torch.where(not_silent, deg_thresholded, torch.zeros_like(deg_thresholded)), dim=-2, keepdim=True)
        equalizer = (avg_ppb_ref + 1000.0) / (avg_ppb_deg + 1000.0)
        equalizer = torch.min(equalizer, 100.0 * torch.ones_like(equalizer))
        equalizer = torch.max(equalizer, 0.01 * torch.ones_like(equalizer))
        return equalizer * deg_bark_spectra

    def loudness_computation(self, bark_spectra):
        aterm = torch.pow(self.abs_thresh_power / 0.5, self.modified_zwicker_power)
        bterm = torch.pow(0.5 + 0.5 * bark_spectra / self.abs_thresh_power, self.modified_zwicker_power) - 1.0
        loudness_dens = self.Sl * aterm * bterm
        cond = bark_spectra < self.abs_thresh_power
        return torch.where(cond, torch.zeros_like(loudness_dens), loudness_dens)

    def compute_distortion_tensors(self, ref_bark_spec, deg_bark_spec):
        original_loudness = self.loudness_computation(ref_bark_spec)
        distorted_loudness = self.loudness_computation(deg_bark_spec)
        r = torch.abs(distorted_loudness - original_loudness)
        m = 0.25 * torch.min(original_loudness, distorted_loudness)
        sym_d = torch.max(r - m, torch.zeros_like(r))
        asym = torch.pow((deg_bark_spec + 50.0) / (ref_bark_spec + 50.0), 1.2)
        cond = asym < 3.0 * torch.ones_like(asym)
        asym_factor = torch.where(cond, torch.zeros_like(asym), torch.min(asym, 12.0 * torch.ones_like(asym)))
        asym_d = asym_factor * sym_d
        return sym_d, asym_d

    def per_frame_distortion(self, sym_d, asym_d, total_power_ref):
        d_frame = torch.sum(torch.pow(sym_d * self.width_of_band_bark, 2.0), dim=-1, keepdim=True)
        d_frame = torch.sqrt(d_frame) * self.sqrt_total_width
        da_frame = torch.sum(asym_d * self.width_of_band_bark, dim=-1, keepdim=True)
        weights = torch.pow((total_power_ref + 100000.0) / 10000000.0, 0.04)
        wd_frame = torch.min(d_frame / weights, 45.0 * torch.ones_like(d_frame))
        wda_frame = torch.min(da_frame / weights, 45.0 * torch.ones_like(da_frame))
        return wd_frame, wda_frame

    @staticmethod
    def get_correction_factor(window_name):
        """ Returns the power correction factor depending on the window. """
        if window_name == 'rect':
            return 1.0
        elif window_name == 'hann':
            return 2.666666666666754
        elif window_name == 'sqrt_hann':
            return 2.0
        elif window_name == 'hamming':
            return 2.51635879188799
        elif window_name == 'flatTop':
            return 5.70713295690759
        else:
            raise ValueError('Unexpected window type {}'.format(window_name))

    def populate_constants(self, sample_rate):
        if sample_rate == 8000:
            self.register_8k_constants()
        elif sample_rate == 16000:
            self.register_16k_constants()
        mask_sll = np.zeros(shape=[self.nbins // 2 + 1], dtype=np.float32)
        mask_sll[11] = 0.5 * 25.0 / 31.25
        mask_sll[12:104] = 1.0
        mask_sll[104] = 0.5
        correction = self.pow_correc_factor * (self.nbins + 2.0) / self.nbins ** 2
        mask_sll = mask_sll * correction
        self.mask_sll = nn.Parameter(tensor(mask_sll), requires_grad=False)

    def register_16k_constants(self):
        abs_thresh_power = [51286152.0, 2454709.5, 70794.59375, 4897.788574, 1174.897705, 389.045166, 104.71286, 45.70882, 17.782795, 9.772372, 4.897789, 3.090296, 1.905461, 1.258925, 0.977237, 0.724436, 0.562341, 0.457088, 0.389045, 0.331131, 0.295121, 0.269153, 0.25704, 0.251189, 0.251189, 0.251189, 0.251189, 0.263027, 0.288403, 0.30903, 0.338844, 0.371535, 0.398107, 0.436516, 0.467735, 0.489779, 0.501187, 0.501187, 0.512861, 0.524807, 0.524807, 0.524807, 0.512861, 0.47863, 0.42658, 0.371535, 0.363078, 0.416869, 0.537032]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        modif_zwicker_power = [0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25168783742879913, 0.2480666573186961, 0.244767379124259, 0.24173800119368227, 0.23893798876066405, 0.23633516221479894, 0.23390360348392067, 0.23162209128929445, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        width_of_band_bark = [0.157344, 0.317994, 0.322441, 0.326934, 0.331474, 0.336061, 0.340697, 0.345381, 0.350114, 0.354897, 0.359729, 0.364611, 0.369544, 0.374529, 0.379565, 0.384653, 0.389794, 0.394989, 0.400236, 0.405538, 0.410894, 0.416306, 0.421773, 0.427297, 0.432877, 0.438514, 0.444209, 0.449962, 0.455774, 0.461645, 0.467577, 0.473569, 0.479621, 0.485736, 0.491912, 0.498151, 0.504454, 0.510819, 0.51725, 0.523745, 0.530308, 0.536934, 0.543629, 0.55039, 0.55722, 0.564119, 0.571085, 0.578125, 0.585232]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, 'bark_matrix_16k.mat')
        bark_matrix = loadmat(bark_path)['Bark_matrix_16k'].astype('float32')
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)

    def register_8k_constants(self):
        abs_thresh_power = [51286152, 2454709.5, 70794.59375, 4897.788574, 1174.897705, 389.045166, 104.71286, 45.70882, 17.782795, 9.772372, 4.897789, 3.090296, 1.905461, 1.258925, 0.977237, 0.724436, 0.562341, 0.457088, 0.389045, 0.331131, 0.295121, 0.269153, 0.25704, 0.251189, 0.251189, 0.251189, 0.251189, 0.263027, 0.288403, 0.30903, 0.338844, 0.371535, 0.398107, 0.436516, 0.467735, 0.489779, 0.501187, 0.501187, 0.512861, 0.524807, 0.524807, 0.524807]
        self.abs_thresh_power = nn.Parameter(tensor(abs_thresh_power), requires_grad=False)
        modif_zwicker_power = [0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25520097857560436, 0.25168783742879913, 0.2480666573186961, 0.244767379124259, 0.24173800119368227, 0.23893798876066405, 0.23633516221479894, 0.23390360348392067, 0.23162209128929445, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]
        self.modified_zwicker_power = nn.Parameter(tensor(modif_zwicker_power), requires_grad=False)
        width_of_band_bark = [0.157344, 0.317994, 0.322441, 0.326934, 0.331474, 0.336061, 0.340697, 0.345381, 0.350114, 0.354897, 0.359729, 0.364611, 0.369544, 0.374529, 0.379565, 0.384653, 0.389794, 0.394989, 0.400236, 0.405538, 0.410894, 0.416306, 0.421773, 0.427297, 0.432877, 0.438514, 0.444209, 0.449962, 0.455774, 0.461645, 0.467577, 0.473569, 0.479621, 0.485736, 0.491912, 0.498151, 0.504454, 0.510819, 0.51725, 0.523745, 0.530308, 0.536934]
        self.width_of_band_bark = nn.Parameter(tensor(width_of_band_bark), requires_grad=False)
        local_path = pathlib.Path(__file__).parent.absolute()
        bark_path = os.path.join(local_path, 'bark_matrix_8k.mat')
        bark_matrix = loadmat(bark_path)['Bark_matrix_8k'].astype('float32')
        self.bark_matrix = nn.Parameter(tensor(bark_matrix), requires_grad=False)


class PairwiseNegSDR(_Loss):
    """ Base class for pairwise negative SI-SDR, SD-SDR and SNR on a batch.

        Args:
            sdr_type (str): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape
                [batch, n_src, time]. Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch, n_src, n_src].
            Pairwise losses.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"),
            >>>                            pit_from='pairwise')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)
        if self.sdr_type in ['sisdr', 'sdsdr']:
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return -pair_wise_sdr


class SingleSrcNegSDR(_Loss):
    """ Base class for single-source negative SI-SDR, SD-SDR and SNR.

        Args:
            sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target and
                estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.
            reduction (string, optional): Specifies the reduction to apply to
                the output:
            ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else
                [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
            >>>                            pit_from='pw_pt')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.
        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True, reduction='none'):
        assert reduction != 'sum', NotImplementedError
        super().__init__(reduction=reduction)
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_target, target):
        assert target.size() == est_target.size()
        if self.zero_mean:
            mean_source = torch.mean(target, dim=1, keepdim=True)
            mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
            target = target - mean_source
            est_target = est_target - mean_estimate
        if self.sdr_type in ['sisdr', 'sdsdr']:
            dot = torch.sum(est_target * target, dim=1, keepdim=True)
            s_target_energy = torch.sum(target ** 2, dim=1, keepdim=True) + EPS
            scaled_target = dot * target / s_target_energy
        else:
            scaled_target = target
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = est_target - target
        else:
            e_noise = est_target - scaled_target
        losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + EPS)
        losses = losses.mean() if self.reduction == 'mean' else losses
        return -losses


class MultiSrcNegSDR(_Loss):
    """ Base class for computing negative SI-SDR, SD-SDR and SNR for a given
        permutation of source and their estimates.

        Args:
            sdr_type (string): choose between "snr" for plain SNR, "sisdr" for
                SI-SDR and "sdsdr" for SD-SDR [1].
            zero_mean (bool, optional): by default it zero mean the target
                and estimate before computing the loss.
            take_log (bool, optional): by default the log10 of sdr is returned.

        Shape:
            est_targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of target estimates.
            targets (:class:`torch.Tensor`): Expected shape [batch, time].
                Batch of training targets.

        Returns:
            :class:`torch.Tensor`: with shape [batch] if reduction='none' else
                [] scalar if reduction='mean'.

        Examples:

            >>> import torch
            >>> from asteroid.losses import PITLossWrapper
            >>> targets = torch.randn(10, 2, 32000)
            >>> est_targets = torch.randn(10, 2, 32000)
            >>> loss_func = PITLossWrapper(MultiSrcNegSDR("sisdr"),
            >>>                            pit_from='perm_avg')
            >>> loss = loss_func(est_targets, targets)

        References:
            [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
            International Conference on Acoustics, Speech and Signal
            Processing (ICASSP) 2019.

        """

    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super().__init__()
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        if self.sdr_type in ['sisdr', 'sdsdr']:
            pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + EPS
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            scaled_targets = targets
        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = est_targets - targets
        else:
            e_noise = est_targets - scaled_targets
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
        return -torch.mean(pair_wise_sdr, dim=-1)


class NonPitSDR(MultiSrcNegSDR, DeprecationMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


class NoSrcSDR(SingleSrcNegSDR, DeprecationMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warn_deprecated()


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm

    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(self, in_chan, hid_chan, skip_out_chan, kernel_size, padding, dilation, norm_type='gLN'):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(), conv_norm(hid_chan), depth_conv1d, nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


def has_arg(fn, name):
    """ Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if `fn` can be called with `name` as a keyword
            argument.

    Returns:
        bool: whether `fn` accepts a `name` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class TDConvNet(nn.Module):
    """ Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.

    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type='gLN', mask_act='relu', kernel_size=None):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        if kernel_size is not None:
            warnings.warn('`kernel_size` argument is deprecated since v0.2.1 and will be remove in v0.3.0. Use argument `conv_kernel_size` instead', VisibleDeprecationWarning)
            conv_kernel_size = kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan, conv_kernel_size, padding=padding, dilation=2 ** x, norm_type=norm_type))
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]

        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = 0.0
        for i in range(len(self.TCN)):
            tcn_out = self.TCN[i](output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {'in_chan': self.in_chan, 'out_chan': self.out_chan, 'bn_chan': self.bn_chan, 'hid_chan': self.hid_chan, 'skip_chan': self.skip_chan, 'conv_kernel_size': self.conv_kernel_size, 'n_blocks': self.n_blocks, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'mask_act': self.mask_act}
        return config


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.
        
        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class ChanLN(_LayerNorm):
    """Channel-wise Layer Normalization (chanLN)."""

    def forward(self, x):
        """ Applies forward pass.
        
        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: chanLN_x `[batch, chan, *]`
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    """Cumulative Global layer normalization(cumLN)."""

    def forward(self, x):
        """

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, channels, length]`
        Returns:
             :class:`torch.Tensor`: cumLN_x `[batch, channels, length]`
        """
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError('expected 4D or 3D input (got {}D input)'.format(input.dim()))


class SingleRNN(nn.Module):
    """ Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ['RNN', 'LSTM', 'GRU']
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True, bidirectional=bool(bidirectional))

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class StackedResidualRNN(nn.Module):
    """ Stacked RNN with builtin residual connection.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=False):
        super(StackedResidualRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is False, 'Bidirectional not supported yet'
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(SingleRNN(rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """ Builtin residual connections + dropout applied before residual.
            Input shape : [batch, time_axis, feat_axis]
        """
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class DPRNNBlock(nn.Module):
    """ Dual-Path RNN Block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_size (int): Number of hidden neurons in the RNNs.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN.
        rnn_type (str, optional): Type of RNN used. Choose from ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers used in each RNN.
        dropout (float, optional): Dropout ratio. Must be in [0, 1].

    References:
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(self, in_chan, hid_size, norm_type='gLN', bidirectional=True, rnn_type='LSTM', num_layers=1, dropout=0):
        super(DPRNNBlock, self).__init__()
        self.intra_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True)
        self.intra_linear = nn.Linear(hid_size * 2, in_chan)
        self.intra_norm = norms.get(norm_type)(in_chan)
        self.inter_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        num_direction = int(bidirectional) + 1
        self.inter_linear = nn.Linear(hid_size * num_direction, in_chan)
        self.inter_norm = norms.get(norm_type)(in_chan)

    def forward(self, x):
        """ Input shape : [batch, feats, chunk_size, num_chunks] """
        B, N, K, L = x.size()
        output = x
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1)
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    """ Dual-path RNN Network for Single-Channel Source Separation
        introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References:
        [1] "Dual-path RNN: efficient long sequence modeling for
            time-domain single-channel speech separation", Yi Luo, Zhuo Chen
            and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(self, in_chan, n_src, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type='gLN', mask_act='relu', bidirectional=True, rnn_type='LSTM', num_layers=1, dropout=0):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        net = []
        for x in range(self.n_repeats):
            net += [DPRNNBlock(bn_chan, hid_size, norm_type=norm_type, bidirectional=bidirectional, rnn_type=rnn_type, num_layers=num_layers, dropout=dropout)]
        self.net = nn.Sequential(*net)
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)
        mask_nl_class = activations.get(mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        n_chunks = output.size(-1)
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        output = self.net(output)
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(output.reshape(batch * self.n_src, to_unfold, n_chunks), (n_frames, 1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask

    def get_config(self):
        config = {'in_chan': self.in_chan, 'out_chan': self.out_chan, 'bn_chan': self.bn_chan, 'hid_size': self.hid_size, 'chunk_size': self.chunk_size, 'hop_size': self.hop_size, 'n_repeats': self.n_repeats, 'n_src': self.n_src, 'norm_type': self.norm_type, 'mask_act': self.mask_act, 'bidirectional': self.bidirectional, 'rnn_type': self.rnn_type, 'num_layers': self.num_layers, 'dropout': self.dropout}
        return config


class BaseTasNet(nn.Module):
    """ Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
    """

    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, wav):
        """ Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        was_one_d = False
        if wav.ndim == 1:
            was_one_d = True
            wav = wav.unsqueeze(0).unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        tf_rep = self.encoder(wav)
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        out_wavs = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), wav)
        if was_one_d:
            return out_wavs.squeeze(0)
        return out_wavs

    def separate(self, wav):
        """ Infer separated sources from input waveforms.

        Args:
            wav (Union[torch.Tensor, numpy.ndarray]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            Union[torch.Tensor, numpy.ndarray], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.
        """
        return self._separate(wav)

    def _separate(self, wav):
        """ Hidden separation method

        Args:
            wav (Union[torch.Tensor, numpy.ndarray]): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.

        Returns:
            Union[torch.Tensor, numpy.ndarray], the estimated sources.
                (batch, n_src, time) or (n_src, time) w/o batch dim.
        """
        was_numpy = False
        if isinstance(wav, np.ndarray):
            was_numpy = True
            wav = torch.from_numpy(wav)
        input_device = wav.device
        model_device = next(self.parameters()).device
        wav = wav
        out_wavs = self.forward(wav)
        out_wavs = out_wavs
        if was_numpy:
            return out_wavs.cpu().data.numpy()
        return out_wavs

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """ Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.

        Returns:
            Instance of BaseTasNet

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_args` and `state_dict`.
        """
        if isinstance(pretrained_model_conf_or_path, str):
            conf = torch.load(pretrained_model_conf_or_path, map_location='cpu')
        else:
            conf = pretrained_model_conf_or_path
        if 'model_args' not in conf.keys():
            raise ValueError('Expected config dictionary to have field model_args`. Found only: {}'.format(conf.keys()))
        if 'state_dict' not in conf.keys():
            raise ValueError('Expected config dictionary to have field state_dict`. Found only: {}'.format(conf.keys()))
        model = cls(*args, **conf['model_args'], **kwargs)
        model.load_state_dict(conf['state_dict'])
        return model

    def serialize(self):
        """ Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        model_conf = dict()
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError('Filterbank and Mask network config sharecommon keys. Merging them is not safe.')
        model_conf['model_args'] = {**fb_config, **masknet_config}
        model_conf['state_dict'] = self.state_dict()
        return model_conf


def apply_mag_mask(tf_rep, mask, dim=-2):
    """ Applies a real-valued mask to a complex-valued representation.

    If `tf_rep` has 2N elements along `dim`, `mask` has N elements, `mask` is
    duplicated along `dim` to apply the same mask to both the Re and Im.

    `tf_rep` is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix

    .. code::

        [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    is interpreted as

    .. code::

        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]

    where `j` is such that `j * j = -1`.

    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to. Re and Im are concatenated along `dim`.
        mask (:class:`torch.Tensor`): The real-valued mask to be applied.
        dim (int): The frequency (or equivalent) dimension of both `tf_rep` and
            `mask` along which real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    check_complex(tf_rep, dim=dim)
    mask = torch.cat([mask, mask], dim=dim)
    return tf_rep * mask


def ebased_vad(mag_spec, th_db=40):
    """ Compute energy-based VAD from a magnitude spectrogram (or equivalent).

    Args:
        mag_spec (torch.Tensor): the spectrogram to perform VAD on.
            Expected shape (batch, *, freq, time).
            The VAD mask will be computed independently for all the leading
            dimensions until the last two. Independent of the ordering of the
            last two dimensions.
        th_db (int): The threshold in dB from which a TF-bin is considered
            silent.

    Returns:
        torch.BoolTensor, the VAD mask.


    Examples:
        >>> import torch
        >>> mag_spec = torch.abs(torch.randn(10, 2, 65, 16))
        >>> batch_src_mask = ebased_vad(mag_spec)
    """
    log_mag = 20 * torch.log10(mag_spec)
    to_view = list(mag_spec.shape[:-2]) + [1, -1]
    max_log_mag = torch.max(log_mag.view(to_view), -1, keepdim=True)[0]
    return log_mag > max_log_mag - th_db


def pad_x_to_y(x, y, axis=-1):
    """  Pad first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad x to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, x padded to match y's shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.size(axis)
    output_len = x.size(axis)
    return nn.functional.pad(x, [0, inp_len - output_len])


class Model(nn.Module):

    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        final_proj, mask_out = self.masker(take_mag(tf_rep))
        return final_proj, mask_out

    def separate(self, x):
        """ Separate with mask-inference head, output waveforms """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        proj, mask_out = self.masker(take_mag(tf_rep))
        masked = apply_mag_mask(tf_rep.unsqueeze(1), mask_out)
        wavs = pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out

    def dc_head_separate(self, x):
        """ Cluster embeddings to produce binary masks, output waveforms """
        kmeans = KMeans(n_clusters=self.masker.n_src)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        mag_spec = take_mag(tf_rep)
        proj, mask_out = self.masker(mag_spec)
        active_bins = ebased_vad(mag_spec)
        active_proj = proj[active_bins.view(1, -1)]
        bin_clusters = kmeans.fit_predict(active_proj.cpu().data.numpy())
        est_mask_list = []
        for i in range(self.masker.n_src):
            mask = ~active_bins
            mask[active_bins] = torch.from_numpy(bin_clusters == i)
            est_mask_list.append(mask.float())
        est_masks = torch.stack(est_mask_list, dim=1)
        masked = apply_mag_mask(tf_rep, est_masks)
        wavs = pad_x_to_y(self.decoder(masked), x)
        dic_out = dict(tfrep=tf_rep, mask=mask_out, masked_tfrep=masked, proj=proj)
        return wavs, dic_out


class SimpleModel(nn.Module):
    """ Simple recurrent model for the DNS challenge.

    Args:
        input_size (int): input size along the features dimension
        hidden_size (int): hidden size in the recurrent net
        output_size (int): output size, defaults to `:attr:` input_size
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can also
            be passed in lowercase letters.
        n_layers (int): Number of recurrent layers.
        dropout (float): dropout value between recurrent layers.
    """

    def __init__(self, input_size, hidden_size, output_size=None, rnn_type='gru', n_layers=3, dropout=0.3):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size if output_size is None else output_size
        self.output_size = output_size
        self.in_proj_layer = nn.Linear(input_size, hidden_size)
        self.residual_rec = blocks.StackedResidualRNN(rnn_type, hidden_size, n_layers=n_layers, dropout=dropout)
        self.out_proj_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """ Mask estimator's forward pass. Expects [batch, time, input_size]"""
        out_rec = self.residual_rec(torch.relu(self.in_proj_layer(x)))
        return torch.relu(self.out_proj_layer(out_rec))


class SeparableDilatedConv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1] without skip
        output. As used in the two step approach [2]. This block uses the
        groupnorm across features and also produces always a padded output.

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        dilation (int): Dilation of the depth-wise convolution.

    References:
        [1]: "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
             for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
             https://arxiv.org/abs/1809.07454
        [2]: Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self, in_chan=256, hid_chan=512, kernel_size=3, dilation=1):
        super(SeparableDilatedConv1DBlock, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(in_channels=in_chan, out_channels=hid_chan, kernel_size=1), nn.PReLU(), nn.GroupNorm(1, hid_chan, eps=1e-08), nn.Conv1d(in_channels=hid_chan, out_channels=hid_chan, kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation, groups=hid_chan), nn.PReLU(), nn.GroupNorm(1, hid_chan, eps=1e-08), nn.Conv1d(in_channels=hid_chan, out_channels=in_chan, kernel_size=1))

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        y = x.clone()
        return x + self.module(y)


class TwoStepTDCN(nn.Module):
    """
        A time-dilated convolutional network (TDCN) similar to the initial
        ConvTasNet architecture where the encoder and decoder have been
        pre-trained separately. The TwoStepTDCN infers masks directly on the
        latent space and works using an signal to distortion ratio (SDR) loss
        directly on the ideal latent masks.
        Adaptive basis encoder and decoder with inference of ideal masks.
        Copied from: https://github.com/etzinis/two_step_mask_learning/

        Args:
            pretrained_filterbank: A pretrained encoder decoder like the one
                implemented in asteroid.filterbanks.simple_adaptive
            n_sources (int, optional): Number of masks to estimate.
            n_blocks (int, optional): Number of convolutional blocks in each
                repeat. Defaults to 8.
            n_repeats (int, optional): Number of repeats. Defaults to 4.
            bn_chan (int, optional): Number of channels after the bottleneck.
            hid_chan (int, optional): Number of channels in the convolutional
                blocks.
            kernel_size (int, optional): Kernel size in convolutional blocks.
                n_sources: The number of sources
        References:
            Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self, pretrained_filterbank, bn_chan=256, hid_chan=512, kernel_size=3, n_blocks=8, n_repeats=4, n_sources=2):
        super(TwoStepTDCN, self).__init__()
        try:
            self.pretrained_filterbank = pretrained_filterbank
            self.encoder = self.pretrained_filterbank.mix_encoder
            self.decoder = self.pretrained_filterbank.decoder
            self.fbank_basis = self.encoder.conv.out_channels
            self.fbank_kernel_size = self.encoder.conv.kernel_size[0]
            self.encoder.conv.weight.requires_grad = False
            self.encoder.conv.bias.requires_grad = False
            self.decoder.deconv.weight.requires_grad = False
            self.decoder.deconv.bias.requires_grad = False
        except Exception as e:
            None
            raise ValueError('Could not load features form the pretrained adaptive filterbank.')
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.n_sources = n_sources
        self.ln_in = nn.BatchNorm1d(self.fbank_basis)
        self.l1 = nn.Conv1d(in_channels=self.fbank_basis, out_channels=self.bn_chan, kernel_size=1)
        self.separator = nn.Sequential(*[SeparableDilatedConv1DBlock(in_chan=self.bn_chan, hid_chan=self.hid_chan, kernel_size=self.kernel_size, dilation=2 ** d) for _ in range(self.n_blocks) for d in range(self.n_repeats)])
        self.mask_layer = nn.Conv2d(in_channels=1, out_channels=self.n_sources, kernel_size=(self.fbank_basis + 1, 1), padding=(self.fbank_basis - self.fbank_basis // 2, 0))
        if self.bn_chan != self.fbank_basis:
            self.out_reshape = nn.Conv1d(in_channels=self.bn_chan, out_channels=self.fbank_basis, kernel_size=1)
        self.ln_mask_in = nn.BatchNorm1d(self.fbank_basis)

    def forward(self, x):
        x = self.encoder(x)
        encoded_mixture = x.clone()
        x = self.ln_in(x)
        x = self.l1(x)
        x = self.separator(x)
        if self.bn_chan != self.fbank_basis:
            x = self.out_reshape(x)
        x = self.ln_mask_in(x)
        x = nn.functional.relu(x)
        x = self.mask_layer(x.unsqueeze(1))
        masks = nn.functional.softmax(x, dim=1)
        return masks * encoded_mixture.unsqueeze(1)

    def infer_source_signals(self, mixture_wav):
        adfe_sources = self.forward(mixture_wav)
        rec_wavs = self.decoder(adfe_sources.view(adfe_sources.shape[0], -1, adfe_sources.shape[-1]))
        return rec_wavs


class AdaptiveEncoder1D(nn.Module):
    """
        A 1D convolutional block that transforms signal in wave form into higher
        dimension.

        Args:
            input shape: [batch, 1, n_samples]
            output shape: [batch, freq_res, n_samples//sample_res]
            freq_res: number of output frequencies for the encoding convolution
            sample_res: int, length of the encoding filter
    """

    def __init__(self, freq_res, sample_res):
        super().__init__()
        self.conv = nn.Conv1d(1, freq_res, sample_res, stride=sample_res // 2, padding=sample_res // 2)

    def forward(self, s):
        return F.relu(self.conv(s))


class AdaptiveDecoder1D(nn.Module):
    """ A 1D deconvolutional block that transforms encoded representation
    into wave form.
    input shape: [batch, freq_res, sample_res]
    output shape: [batch, 1, sample_res*n_samples]
    freq_res: number of output frequencies for the encoding convolution
    sample_res: length of the encoding filter
    """

    def __init__(self, freq_res, sample_res, n_sources):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_sources * freq_res, n_sources, sample_res, padding=sample_res // 2, stride=sample_res // 2, groups=n_sources, output_padding=sample_res // 2 - 1)

    def forward(self, x):
        return self.deconv(x)


class AdaptiveEncoderDecoder(nn.Module):
    """
        Adaptive basis encoder and decoder with inference of ideal masks.
        Copied from: https://github.com/etzinis/two_step_mask_learning/

        Args:
            freq_res: The number of frequency like representations
            sample_res: The number of samples in kernel 1D convolutions
            n_sources: The number of sources
        References:
            Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
            Smaragdis, P., "Two-Step Sound Source Separation:
            Training on Learned Latent Targets." In Acoustics, Speech
            and Signal Processing (ICASSP), 2020 IEEE International Conference.
            https://arxiv.org/abs/1910.09804
    """

    def __init__(self, freq_res=256, sample_res=21, n_sources=2):
        super().__init__()
        self.freq_res = freq_res
        self.sample_res = sample_res
        self.mix_encoder = AdaptiveEncoder1D(freq_res, sample_res)
        self.decoder = AdaptiveDecoder1D(freq_res, sample_res, n_sources)
        self.n_sources = n_sources

    def get_target_masks(self, clean_sources):
        """
        Get target masks for the given clean sources
        :param clean_sources: [batch, n_sources, time_samples]
        :return: Ideal masks for the given sources:
        [batch, n_sources, time_samples//(sample_res // 2)]
        """
        enc_mask_list = [self.mix_encoder(clean_sources[:, (i), :].unsqueeze(1)) for i in range(self.n_sources)]
        total_mask = torch.stack(enc_mask_list, dim=1)
        return F.softmax(total_mask, dim=1)

    def reconstruct(self, mixture):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        return self.decoder(enc_mixture)

    def get_encoded_sources(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        return s_recon_enc

    def forward(self, mixture, clean_sources):
        enc_mixture = self.mix_encoder(mixture.unsqueeze(1))
        enc_masks = self.get_target_masks(clean_sources)
        s_recon_enc = enc_masks * enc_mixture.unsqueeze(1)
        recon_sources = self.decoder(s_recon_enc.view(s_recon_enc.shape[0], -1, s_recon_enc.shape[-1]))
        return recon_sources, enc_masks


class TasNet(nn.Module):
    """ Some kind of TasNet, but not the original one
    Differences:
        - Overlap-add support (strided convolutions)
        - No frame-wise normalization on the wavs
        - GlobLN as bottleneck layer.
        - No skip connection.

    Args:
        fb_conf (dict): see local/conf.yml
        mask_conf (dict): see local/conf.yml
    """

    def __init__(self, fb_conf, mask_conf):
        super().__init__()
        self.n_src = mask_conf['n_src']
        self.n_filters = fb_conf['n_filters']
        self.encoder_sig = Encoder(FreeFB(**fb_conf))
        self.encoder_relu = Encoder(FreeFB(**fb_conf))
        self.decoder = Decoder(FreeFB(**fb_conf))
        self.bn_layer = GlobLN(fb_conf['n_filters'])
        self.masker = nn.Sequential(SingleRNN('lstm', fb_conf['n_filters'], hidden_size=mask_conf['n_units'], n_layers=mask_conf['n_layers'], bidirectional=True, dropout=mask_conf['dropout']), nn.Linear(2 * mask_conf['n_units'], self.n_src * self.n_filters), nn.Sigmoid())

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encode(x)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks
        return pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


class Chimera(nn.Module):

    def __init__(self, in_chan, n_src, rnn_type='lstm', n_layers=2, hidden_size=600, bidirectional=True, dropout=0.3, embedding_dim=20, take_log=False):
        super().__init__()
        self.input_dim = in_chan
        self.n_src = n_src
        self.take_log = take_log
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(rnn_type, in_chan, hidden_size, n_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * self.n_src)
        self.mask_act = nn.Sigmoid()
        self.embedding_layer = nn.Linear(rnn_out_dim, in_chan * embedding_dim)
        self.embedding_act = nn.Tanh()

    def forward(self, input_data):
        batch, _, n_frames = input_data.shape
        if self.take_log:
            input_data = torch.log(input_data + EPS)
        out = self.rnn(input_data.permute(0, 2, 1))
        out = self.dropout(out)
        proj = self.embedding_layer(out)
        proj = self.embedding_act(proj)
        proj = proj.view(batch, n_frames, -1, self.embedding_dim).transpose(1, 2)
        proj = proj.reshape(batch, -1, self.embedding_dim)
        proj_norm = torch.norm(proj, p=2, dim=-1, keepdim=True)
        projection_final = proj / (proj_norm + EPS)
        mask_out = self.mask_layer(out).view(batch, n_frames, self.n_src, self.input_dim)
        mask_out = mask_out.permute(0, 2, 3, 1)
        mask_out = self.mask_act(mask_out)
        return projection_final, mask_out


def batch_matrix_norm(matrix, norm_order=2):
    """ Normalize a matrix according to `norm_order`

    Args:
        matrix (torch.Tensor): Expected shape [batch, *]
        norm_order (int): Norm order.

    Returns:
        torch.Tensor, normed matrix of shape [batch]
    """
    keep_batch = list(range(1, matrix.ndim))
    return torch.norm(matrix, p=norm_order, dim=keep_batch) ** norm_order


def deep_clustering_loss(embedding, tgt_index, binary_mask=None):
    """ Compute the deep clustering loss defined in [1].

    Args:
        embedding (torch.Tensor): Estimated embeddings.
            Expected shape  (batch, frequency x frame, embedding_dim)
        tgt_index (torch.Tensor): Dominating source index in each TF bin.
            Expected shape: [batch, frequency, frame]
        binary_mask (torch.Tensor): VAD in TF plane. Bool or Float.
            See asteroid.filterbanks.transforms.ebased_vad.

    Returns:
         `torch.Tensor`. Deep clustering loss for every batch sample.

    Examples:
        >>> import torch
        >>> from asteroid.losses.cluster import deep_clustering_loss
        >>> spk_cnt = 3
        >>> embedding = torch.randn(10, 5*400, 20)
        >>> targets = torch.LongTensor([10, 400, 5]).random_(0, spk_cnt)
        >>> loss = deep_clustering_loss(embedding, targets)

    Reference:
        [1] Zhong-Qiu Wang, Jonathan Le Roux, John R. Hershey
            "ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING"

    Notes:
        Be careful in viewing the embedding tensors. The target indices
        `tgt_index` are of shape (batch, freq, frames). Even if the embedding
        is of shape (batch, freq*frames, emb), the underlying view should be
        (batch, freq, frames, emb) and not (batch, frames, freq, emb).
    """
    spk_cnt = len(tgt_index.unique())
    batch, bins, frames = tgt_index.shape
    if binary_mask is None:
        binary_mask = torch.ones(batch, bins * frames, 1)
    binary_mask = binary_mask.float()
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.view(batch, bins * frames, 1)
    binary_mask = binary_mask
    tgt_embedding = torch.zeros(batch, bins * frames, spk_cnt, device=tgt_index.device)
    tgt_embedding.scatter_(2, tgt_index.view(batch, bins * frames, 1), 1)
    tgt_embedding = tgt_embedding * binary_mask
    embedding = embedding * binary_mask
    est_proj = torch.einsum('ijk,ijl->ikl', embedding, embedding)
    true_proj = torch.einsum('ijk,ijl->ikl', tgt_embedding, tgt_embedding)
    true_est_proj = torch.einsum('ijk,ijl->ikl', embedding, tgt_embedding)
    cost = batch_matrix_norm(est_proj) + batch_matrix_norm(true_proj)
    cost = cost - 2 * batch_matrix_norm(true_est_proj)
    return cost / torch.sum(binary_mask, dim=[1, 2])


pairwise_mse = PairwiseMSE()


class ChimeraLoss(nn.Module):
    """ Combines Deep clustering loss and mask inference loss for ChimeraNet.

    Args:
        alpha (float): loss weight. Total loss will be :
            `alpha` * dc_loss + (1 - `alpha`) * mask_mse_loss.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha >= 0, "Negative alpha values don't make sense."
        assert alpha <= 1, "Alpha values above 1 don't make sense."
        self.src_mse = PITLossWrapper(pairwise_mse, pit_from='pw_mtx')
        self.alpha = alpha

    def forward(self, est_embeddings, target_indices, est_src=None, target_src=None, mix_spec=None):
        """

        Args:
            est_embeddings (torch.Tensor): Estimated embedding from the DC head.
            target_indices (torch.Tensor): Target indices that'll be passed to
                the DC loss.
            est_src (torch.Tensor): Estimated magnitude spectrograms (or masks).
            target_src (torch.Tensor): Target magnitude spectrograms (or masks).
            mix_spec (torch.Tensor): The magnitude spectrogram of the mixture
                from which VAD will be computed. If None, no VAD is used.

        Returns:
            torch.Tensor, the total loss, averaged over the batch.
            dict with `dc_loss` and `pit_loss` keys, unweighted losses.
        """
        if self.alpha != 0 and (est_src is None or target_src is None):
            raise ValueError('Expected target and estimated spectrograms to compute the PIT loss, found None.')
        binary_mask = None
        if mix_spec is not None:
            binary_mask = ebased_vad(mix_spec)
        dc_loss = deep_clustering_loss(embedding=est_embeddings, tgt_index=target_indices, binary_mask=binary_mask)
        src_pit_loss = self.src_mse(est_src, target_src)
        tot = self.alpha * dc_loss.mean() + (1 - self.alpha) * src_pit_loss
        loss_dict = dict(dc_loss=dc_loss.mean(), pit_loss=src_pit_loss)
        return tot, loss_dict


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveDecoder1D,
     lambda: ([], {'freq_res': 4, 'sample_res': 4, 'n_sources': 4}),
     lambda: ([torch.rand([4, 16, 64])], {}),
     True),
    (AdaptiveEncoder1D,
     lambda: ([], {'freq_res': 4, 'sample_res': 4}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     True),
    (BaseTasNet,
     lambda: ([], {'encoder': _mock_layer(), 'masker': _mock_layer(), 'decoder': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChanLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Chimera,
     lambda: ([], {'in_chan': 4, 'n_src': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CumLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GlobLN,
     lambda: ([], {'channel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiSrcNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoSrcMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoSrcSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NonPitSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PairwiseMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PairwiseNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SeparableDilatedConv1DBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64])], {}),
     True),
    (SimpleModel,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SingleRNN,
     lambda: ([], {'rnn_type': 'gru', 'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SingleSrcMSE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SingleSrcNegSDR,
     lambda: ([], {'sdr_type': 'snr'}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (StackedResidualRNN,
     lambda: ([], {'rnn_type': 'gru', 'n_units': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_mpariente_asteroid(_paritybench_base):
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

