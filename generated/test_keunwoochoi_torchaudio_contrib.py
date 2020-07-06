import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_functional = _module
test_layers = _module
torchaudio_contrib = _module
beta_hpss = _module
functional = _module
layers = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


def hpss(mag_specgrams, kernel_size=31, power=2.0, hard=False, mask_only=False):
    """
    A function that performs harmonic-percussive source separation.
    Original method is by Derry Fitzgerald
    (https://www.researchgate.net/publication/254583990_HarmonicPercussive_Separation_using_Median_Filtering).

    Args:
        mag_specgrams (Tensor): any magnitude spectrograms in batch, (not in a decibel scale!)
            in a shape of (batch, ch, freq, time)

        kernel_size (int or (int, int)): odd-numbered
            if tuple,
                1st: width of percussive-enhancing filter (one along freq axis)
                2nd: width of harmonic-enhancing filter (one along time axis)
            if int,
                it's applied for both perc/harm filters

        power (float): to which the enhanced spectrograms are used in computing soft masks.

        hard (bool): whether the mask will be binarized (True) or not

        mask_only (bool): if true, returns the masks only.

    Returns:
        ret (Tuple): A tuple of four

            ret[0]: magnitude spectrograms - harmonic parts (Tensor, in same size with `mag_specgrams`)
            ret[1]: magnitude spectrograms - percussive parts (Tensor, in same size with `mag_specgrams`)
            ret[2]: harmonic mask (Tensor, in same size with `mag_specgrams`)
            ret[3]: percussive mask (Tensor, in same size with `mag_specgrams`)
    """

    def _enhance_either_hpss(mag_specgrams_padded, out, kernel_size, power, which, offset):
        """
        A helper function for HPSS

        Args:
            mag_specgrams_padded (Tensor): one that median filtering can be directly applied

            out (Tensor): The tensor to store the result

            kernel_size (int): The kernel size of median filter

            power (float): to which the enhanced spectrograms are used in computing soft masks.

            which (str): either 'harm' or 'perc'

            offset (int): the padded length

        """
        if which == 'harm':
            for t in range(out.shape[3]):
                out[:, :, :, (t)] = torch.median(mag_specgrams_padded[:, :, offset:-offset, t:t + kernel_size], dim=3)[0]
        elif which == 'perc':
            for f in range(out.shape[2]):
                out[:, :, (f), :] = torch.median(mag_specgrams_padded[:, :, f:f + kernel_size, offset:-offset], dim=2)[0]
        else:
            raise NotImplementedError('it should be either but you passed which={}'.format(which))
        if power != 1.0:
            out.pow_(power)
    eps = 1e-06
    if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int)):
        raise TypeError('kernel_size is expected to be either tuple of input, but it is: %s' % type(kernel_size))
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    pad = kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2
    harm, perc, ret = torch.empty_like(mag_specgrams), torch.empty_like(mag_specgrams), torch.empty_like(mag_specgrams)
    mag_specgrams_padded = F.pad(mag_specgrams, pad=pad, mode='reflect')
    _enhance_either_hpss(mag_specgrams_padded, out=perc, kernel_size=kernel_size[0], power=power, which='perc', offset=kernel_size[1] // 2)
    _enhance_either_hpss(mag_specgrams_padded, out=harm, kernel_size=kernel_size[1], power=power, which='harm', offset=kernel_size[0] // 2)
    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)
    if mask_only:
        return None, None, mask_harm, mask_perc
    return mag_specgrams * mask_harm, mag_specgrams * mask_perc, mask_harm, mask_perc


class HPSS(nn.Module):
    """
    Wrap hpss.

    Args and Returns --> see `hpss`.
    """

    def __init__(self, kernel_size=31, power=2.0, hard=False, mask_only=False):
        super(HPSS, self).__init__()
        self.kernel_size = kernel_size
        self.power = power
        self.hard = hard
        self.mask_only = mask_only

    def forward(self, mag_specgrams):
        return hpss(mag_specgrams, self.kernel_size, self.power, self.hard, self.mask_only)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={}, power={}, hard={}, mask_only={})'.format(self.kernel_size, self.power, self.hard, self.mask_only)


class _ModuleNoStateBuffers(nn.Module):
    """
    Extension of nn.Module that removes buffers
    from state_dict.
    """

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(_ModuleNoStateBuffers, self).state_dict(destination, prefix, keep_vars)
        for k in self._buffers:
            del ret[prefix + k]
        return ret

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffers = self._buffers
        self._buffers = {}
        result = super(_ModuleNoStateBuffers, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result


def stft(waveforms, fft_length, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True):
    """Compute a short-time Fourier transform of the input waveform(s).
    It wraps `torch.stft` but after reshaping the input audio
    to allow for `waveforms` that `.dim()` >= 3.
    It follows most of the `torch.stft` default value, but for `window`,
    if it's not specified (`None`), it uses hann window.

    Args:
        waveforms (Tensor): Tensor of audio signal
            of size `(*, channel, time)`
        fft_length (int): FFT size [sample]
        hop_length (int): Hop size [sample] between STFT frames.
            Defaults to `fft_length // 4` (75%-overlapping windows)
            by `torch.stft`.
        win_length (int): Size of STFT window.
            Defaults to `fft_length` by `torch.stft`.
        window (Tensor): 1-D Tensor.
            Defaults to Hann Window of size `win_length`
            *unlike* `torch.stft`.
        center (bool): Whether to pad `waveforms` on both sides so that the
            `t`-th frame is centered at time `t * hop_length`.
            Defaults to `True` by `torch.stft`.
        pad_mode (str): padding method (see `torch.nn.functional.pad`).
            Defaults to `'reflect'` by `torch.stft`.
        normalized (bool): Whether the results are normalized.
            Defaults to `False` by `torch.stft`.
        onesided (bool): Whether the half + 1 frequency bins
            are returned to removethe symmetric part of STFT
            of real-valued signal. Defaults to `True`
            by `torch.stft`.

    Returns:
        complex_specgrams (Tensor): `(*, channel, num_freqs, time, complex=2)`

    Example:
        >>> waveforms = torch.randn(16, 2, 10000)  # (batch, channel, time)
        >>> x = stft(waveforms, 2048, 512)
        >>> x.shape
        torch.Size([16, 2, 1025, 20])
    """
    leading_dims = waveforms.shape[:-1]
    waveforms = waveforms.reshape(-1, waveforms.size(-1))
    if window is None:
        if win_length is None:
            window = torch.hann_window(fft_length)
        else:
            window = torch.hann_window(win_length)
    complex_specgrams = torch.stft(waveforms, n_fft=fft_length, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided)
    complex_specgrams = complex_specgrams.reshape(leading_dims + complex_specgrams.shape[1:])
    return complex_specgrams


class STFT(_ModuleNoStateBuffers):
    """Compute a short-time Fourier transform of the input waveform(s).
    It essentially wraps `torch.stft` but after reshaping the input audio
    to allow for `waveforms` that `.dim()` >= 3.
    It follows most of the `torch.stft` default value, but for `window`,
    if it's not specified (`None`), it uses hann window.

    Args:
        fft_length (int): FFT size [sample]
        hop_length (int): Hop size [sample] between STFT frames.
            Defaults to `fft_length // 4` (75%-overlapping windows) by `torch.stft`.
        win_length (int): Size of STFT window.
            Defaults to `fft_length` by `torch.stft`.
        window (Tensor): 1-D Tensor.
            Defaults to Hann Window of size `win_length` *unlike* `torch.stft`.
        center (bool): Whether to pad `waveforms` on both sides so that the
            `t`-th frame is centered at time `t * hop_length`.
            Defaults to `True` by `torch.stft`.
        pad_mode (str): padding method (see `torch.nn.functional.pad`).
            Defaults to `'reflect'` by `torch.stft`.
        normalized (bool): Whether the results are normalized.
            Defaults to `False` by `torch.stft`.
        onesided (bool): Whether the half + 1 frequency bins are returned to remove
            the symmetric part of STFT of real-valued signal.
            Defaults to `True` by `torch.stft`.
    """

    def __init__(self, fft_length, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True):
        super(STFT, self).__init__()
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        if window is None:
            if win_length is None:
                window = torch.hann_window(fft_length)
            else:
                window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, waveforms):
        """
        Args:
            waveforms (Tensor): Tensor of audio signal of size `(*, channel, time)`

        Returns:
            complex_specgrams (Tensor): `(*, channel, num_freqs, time, complex=2)`
        """
        complex_specgrams = stft(waveforms, self.fft_length, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=self.center, pad_mode=self.pad_mode, normalized=self.normalized, onesided=self.onesided)
        return complex_specgrams

    def __repr__(self):
        param_str1 = '(fft_length={}, hop_length={}, win_length={})'.format(self.fft_length, self.hop_length, self.win_length)
        param_str2 = '(center={}, pad_mode={}, normalized={}, onesided={})'.format(self.center, self.pad_mode, self.normalized, self.onesided)
        return self.__class__.__name__ + param_str1 + param_str2


def complex_norm(complex_tensor, power=1.0):
    """Compute the norm of complex tensor input

    Args:
        complex_tensor (Tensor): Tensor shape of `(*, complex=2)`
        power (float): Power of the norm. Defaults to `1.0`.

    Returns:
        Tensor: power of the normed input tensor, shape of `(*, )`
    """
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


class ComplexNorm(nn.Module):
    """Compute the norm of complex tensor input

    Args:
        power (float): Power of the norm. Defaults to `1.0`.

    """

    def __init__(self, power=1.0):
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, complex_tensor):
        """
        Args:
            complex_tensor (Tensor): Tensor shape of `(*, complex=2)`

        Returns:
            Tensor: norm of the input tensor, shape of `(*, )`
        """
        return complex_norm(complex_tensor, self.power)

    def __repr__(self):
        return self.__class__.__name__ + '(power={})'.format(self.power)


def apply_filterbank(mag_specgrams, filterbank):
    """
    Transform spectrogram given a filterbank matrix.

    Args:
        mag_specgrams (Tensor): (batch, channel, num_freqs, time)
        filterbank (Tensor): (num_freqs, num_bands)

    Returns:
        (Tensor): (batch, channel, num_bands, time)
    """
    return torch.matmul(mag_specgrams.transpose(-2, -1), filterbank).transpose(-2, -1)


class ApplyFilterbank(_ModuleNoStateBuffers):
    """
    Applies a filterbank transform.
    """

    def __init__(self, filterbank):
        super(ApplyFilterbank, self).__init__()
        self.register_buffer('filterbank', filterbank)

    def forward(self, mag_specgrams):
        """
        Args:
            mag_specgrams (Tensor): (channel, time, freq) or (batch, channel, time, freq).

        Returns:
            (Tensor): freq -> filterbank.size(0)
        """
        return apply_filterbank(mag_specgrams, self.filterbank)


def angle(complex_tensor):
    """
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])


def phase_vocoder(complex_specgrams, rate, phase_advance):
    """
    Phase vocoder. Given a STFT tensor, speed up in time
    without modifying pitch by a factor of `rate`.

    Args:
        complex_specgrams (Tensor):
            (*, channel, num_freqs, time, complex=2)
        rate (float): Speed-up factor.
        phase_advance (Tensor): Expected phase advance in
            each bin. (num_freqs, 1).

    Returns:
        complex_specgrams_stretch (Tensor):
            (*, channel, num_freqs, ceil(time/rate), complex=2).

    Example:
        >>> num_freqs, hop_length = 1025, 512
        >>> # (batch, channel, num_freqs, time, complex=2)
        >>> complex_specgrams = torch.randn(16, 1, num_freqs, 300, 2)
        >>> rate = 1.3 # Slow down by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, num_freqs)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([16, 1, 1025, 231, 2])
    """
    ndim = complex_specgrams.dim()
    time_slice = [slice(None)] * (ndim - 2)
    time_steps = torch.arange(0, complex_specgrams.size(-2), rate, device=complex_specgrams.device)
    alphas = torch.remainder(time_steps, torch.tensor(1.0, device=complex_specgrams.device))
    phase_0 = angle(complex_specgrams[time_slice + [slice(1)]])
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 0, 0, 2])
    complex_specgrams_0 = complex_specgrams[time_slice + [time_steps.long()]]
    complex_specgrams_1 = complex_specgrams[time_slice + [(time_steps + 1).long()]]
    angle_0 = angle(complex_specgrams_0)
    angle_1 = angle(complex_specgrams_1)
    norm_0 = torch.norm(complex_specgrams_0, dim=-1)
    norm_1 = torch.norm(complex_specgrams_1, dim=-1)
    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[time_slice + [slice(-1)]]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)
    mag = alphas * norm_1 + (1 - alphas) * norm_0
    real_stretch = mag * torch.cos(phase_acc)
    imag_stretch = mag * torch.sin(phase_acc)
    complex_specgrams_stretch = torch.stack([real_stretch, imag_stretch], dim=-1)
    return complex_specgrams_stretch


class TimeStretch(_ModuleNoStateBuffers):
    """
    Stretch stft in time without modifying pitch for a given rate.

    Args:

        hop_length (int): Number audio of frames between STFT columns.
        num_freqs (int, optional): number of filter banks from stft.
        fixed_rate (float): rate to speed up or slow down by.
            Defaults to None (in which case a rate must be
            passed to the forward method per batch).
    """

    def __init__(self, hop_length, num_freqs, fixed_rate=None):
        super(TimeStretch, self).__init__()
        self.fixed_rate = fixed_rate
        phase_advance = torch.linspace(0, math.pi * hop_length, num_freqs)[..., None]
        self.register_buffer('phase_advance', phase_advance)

    def forward(self, complex_specgrams, overriding_rate=None):
        """

        Args:
            complex_specgrams (Tensor): complex spectrogram
                (*, channel, freq, time, complex=2)
            overriding_rate (float or None): speed up to apply to this batch.
                If no rate is passed, use self.fixed_rate.

        Returns:
            (Tensor): (*, channel, num_freqs, ceil(time/rate), complex=2)
        """
        if overriding_rate is None:
            rate = self.fixed_rate
            if rate is None:
                raise ValueError('If no fixed_rate is specified, must pass a valid rate to the forward method.')
        else:
            rate = overriding_rate
        if rate == 1.0:
            return complex_specgrams
        return phase_vocoder(complex_specgrams, rate, self.phase_advance)

    def __repr__(self):
        param_str = '(fixed_rate={})'.format(self.fixed_rate)
        return self.__class__.__name__ + param_str


def amplitude_to_db(x, ref=1.0, amin=1e-07):
    """
    Amplitude-to-decibel conversion (logarithmic mapping with base=10)
    By using `amin=1e-7`, it assumes 32-bit floating point input. If the
    data precision differs, use approproate `amin` accordingly.

    Args:
        x (Tensor): Input amplitude
        ref (float): Amplitude value that is equivalent to 0 decibel
        amin (float): Minimum amplitude. Any input that is smaller than `amin` is
            clamped to `amin`.
    Returns:
        (Tensor): same size of x, after conversion
    """
    x = x.pow(2.0)
    x = torch.clamp(x, min=amin)
    return 10.0 * (torch.log10(x) - torch.log10(torch.tensor(ref, device=x.device, requires_grad=False, dtype=x.dtype)))


class AmplitudeToDb(_ModuleNoStateBuffers):
    """
    Amplitude-to-decibel conversion (logarithmic mapping with base=10)
    By using `amin=1e-7`, it assumes 32-bit floating point input. If the
    data precision differs, use approproate `amin` accordingly.

    Args:
        ref (float): Amplitude value that is equivalent to 0 decibel
        amin (float): Minimum amplitude. Any input that is smaller than `amin` is
            clamped to `amin`.
    """

    def __init__(self, ref=1.0, amin=1e-07):
        super(AmplitudeToDb, self).__init__()
        self.ref = ref
        self.amin = amin
        assert ref > amin, 'Reference value is expected to be bigger than amin, but I haveref:{} and amin:{}'.format(ref, amin)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input amplitude

        Returns:
            (Tensor): same size of x, after conversion
        """
        return amplitude_to_db(x, ref=self.ref, amin=self.amin)

    def __repr__(self):
        param_str = '(ref={}, amin={})'.format(self.ref, self.amin)
        return self.__class__.__name__ + param_str


def db_to_amplitude(x, ref=1.0):
    """
    Decibel-to-amplitude conversion (exponential mapping with base=10)

    Args:
        x (Tensor): Input in decibel to be converted
        ref (float): Amplitude value that is equivalent to 0 decibel

    Returns:
        (Tensor): same size of x, after conversion
    """
    power_spec = torch.pow(10.0, x / 10.0 + torch.log10(torch.tensor(ref, device=x.device, requires_grad=False, dtype=x.dtype)))
    return power_spec.pow(0.5)


class DbToAmplitude(_ModuleNoStateBuffers):
    """
    Decibel-to-amplitude conversion (exponential mapping with base=10)

    Args:
        x (Tensor): Input in decibel to be converted
        ref (float): Amplitude value that is equivalent to 0 decibel

    Returns:
        (Tensor): same size of x, after conversion
    """

    def __init__(self, ref=1.0):
        super(DbToAmplitude, self).__init__()
        self.ref = ref

    def forward(self, x):
        """
        Args:
            x (Tensor): Input in decibel to be converted

        Returns:
            (Tensor): same size of x, after conversion
        """
        return db_to_amplitude(x, ref=self.ref)

    def __repr__(self):
        param_str = '(ref={})'.format(self.ref)
        return self.__class__.__name__ + param_str


def mu_law_encoding(x, n_quantize=256):
    """Apply mu-law encoding to the input tensor.
    Usually applied to waveforms

    Args:
        x (Tensor): input value
        n_quantize (int): quantization level. For 8-bit encoding, set 256 (2 ** 8).

    Returns:
        (Tensor): same size of x, after encoding

    """
    if not x.dtype.is_floating_point:
        x = x
    mu = torch.tensor(n_quantize - 1, dtype=x.dtype, requires_grad=False)
    x_mu = x.sign() * torch.log1p(mu * x.abs()) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
    return x_mu


class MuLawEncoding(_ModuleNoStateBuffers):
    """Apply mu-law encoding to the input tensor.
    Usually applied to waveforms

    Args:
        n_quantize (int): quantization level. For 8-bit encoding, set 256 (2 ** 8).

    """

    def __init__(self, n_quantize=256):
        super(MuLawEncoding, self).__init__()
        self.n_quantize = n_quantize

    def forward(self, x):
        """
        Args:
            x (Tensor): input value

        Returns:
            (Tensor): same size of x, after encoding
        """
        return mu_law_encoding(x, self.n_quantize)

    def __repr__(self):
        param_str = '(n_quantize={})'.format(self.n_quantize)
        return self.__class__.__name__ + param_str


def mu_law_decoding(x_mu, n_quantize=256, dtype=torch.get_default_dtype()):
    """Apply mu-law decoding (expansion) to the input tensor.

    Args:
        x_mu (Tensor): mu-law encoded input
        n_quantize (int): quantization level. For 8-bit decoding, set 256 (2 ** 8).
        dtype: specifies `dtype` for the decoded value. Default: `torch.get_default_dtype()`

    Returns:
        (Tensor): mu-law decoded tensor
    """
    if not x_mu.dtype.is_floating_point:
        x_mu = x_mu
    mu = torch.tensor(n_quantize - 1, dtype=x_mu.dtype, requires_grad=False)
    x = x_mu / mu * 2 - 1.0
    x = x.sign() * (torch.exp(x.abs() * torch.log1p(mu)) - 1.0) / mu
    return x


class MuLawDecoding(_ModuleNoStateBuffers):
    """Apply mu-law decoding (expansion) to the input tensor.
    Usually applied to waveforms

    Args:
        n_quantize (int): quantization level. For 8-bit decoding, set 256 (2 ** 8).
    """

    def __init__(self, n_quantize=256):
        super(MuLawDecoding, self).__init__()
        self.n_quantize = n_quantize

    def forward(self, x_mu):
        """
        Args:
            x_mu (Tensor): mu-law encoded input

        Returns:
            (Tensor): mu-law decoded tensor
        """
        return mu_law_decoding(x_mu, self.n_quantize)

    def __repr__(self):
        param_str = '(n_quantize={})'.format(self.n_quantize)
        return self.__class__.__name__ + param_str


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AmplitudeToDb,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ComplexNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DbToAmplitude,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HPSS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MuLawDecoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MuLawEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (STFT,
     lambda: ([], {'fft_length': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_keunwoochoi_torchaudio_contrib(_paritybench_base):
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

