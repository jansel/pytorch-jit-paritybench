import sys
_module = sys.modules[__name__]
del sys
DCUNet = _module
complex_nn = _module
constant = _module
criterion = _module
inference = _module
metric = _module
noisedataset = _module
sedataset = _module
source_separator = _module
unet = _module
utils = _module
estimate_directory = _module
evaluation = _module
train = _module
train_dcunet = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch.functional as F


from torch.utils.data import Dataset


import numpy as np


import torch.optim as optim


from torch.utils.data import DataLoader


class ComplexConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, **kwargs)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


def realimag(mag, phase):
    """
    Combine a magnitude spectrogram and a phase spectrogram to a complex-valued spectrogram with shape (*, 2)
    """
    spec_real = mag * torch.cos(phase)
    spec_imag = mag * torch.sin(phase)
    spec = torch.stack([spec_real, spec_imag], dim=-1)
    return spec


class ApplyMask(nn.Module):

    def __init__(self, complex=True, log_amp=False):
        super().__init__()
        self.amp2db = audio_nn.DbToAmplitude()
        self.complex = complex
        self.log_amp = log_amp

    def forward(self, bd):
        if not self.complex:
            Y_hat = bd['mag_X'] * bd['M_hat']
            Y_hat = realimag(Y_hat, bd['phase_X'])
            if self.log_amp:
                raise NotImplementedError
        else:
            Y_hat = bd['X'] * bd['M_hat']
            if self.log_amp:
                Y_hat = self.amp2db(Y_hat)
        return Y_hat


SAMPLE_RATE = 16000


HOP_LENGTH = SAMPLE_RATE * 16 // 1000


N_FFT = SAMPLE_RATE * 64 // 1000


def istft(stft_matrix, hop_length=None, win_length=None, window='hann', center=True, normalized=False, onesided=True, length=None):
    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == 'hann'
    assert center == True
    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)
    batch = stft_matrix.shape[0]
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    istft_window = torch.hann_window(n_fft).view(1, -1)
    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))
        ytmp = istft_window * iffted
        y[:, sample:sample + n_fft] += ytmp
    y = y[:, n_fft // 2:]
    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = F.pad(y, (0, length - y.shape[1]))
    coeff = n_fft / float(hop_length) / 2.0
    return y / coeff


class ISTFT(nn.Module):

    def __init__(self, complex=True, log_amp=False, length=16384):
        super().__init__()
        self.amp2db = audio_nn.DbToAmplitude()
        self.complex = complex
        self.log_amp = log_amp
        self.length = length

    def forward(self, Y_hat):
        num_batch = Y_hat.shape[0]
        num_channel = Y_hat.shape[1]
        Y_hat = Y_hat.view(Y_hat.shape[0] * Y_hat.shape[1], Y_hat.shape[2], Y_hat.shape[3], Y_hat.shape[4])
        y_hat = istft(Y_hat, hop_length=HOP_LENGTH, win_length=N_FFT, length=self.length)
        y_hat = y_hat.view(num_batch, num_channel, -1)
        return y_hat


class STFT(nn.Module):

    def __init__(self, complex=True, log_amp=False):
        super(self.__class__, self).__init__()
        self.stft = audio_nn.STFT(fft_length=N_FFT, hop_length=HOP_LENGTH)
        self.amp2db = audio_nn.AmplitudeToDb()
        self.complex = complex
        self.log_amp = log_amp
        window = torch.hann_window(N_FFT)
        self.register_buffer('window', window)

    def forward(self, bd):
        with torch.no_grad():
            bd['X'] = self.stft(bd['x'])
            if not self.complex:
                bd['mag_X'], bd['phase_X'] = audio_nn.magphase(bd['X'], power=1.0)
            if self.log_amp:
                bd['X'] = self.amp2db(bd['X'])
        return bd


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode='zeros'):
        super().__init__()
        if padding is None:
            padding = [((i - 1) // 2) for i in kernel_size]
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):

    def __init__(self, input_channels=1, complex=False, model_complexity=45, model_depth=20, padding_mode='zeros'):
        super().__init__()
        if complex:
            model_complexity = int(model_complexity // 1.414)
        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2
        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i], stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module('encoder{}'.format(i), module)
            self.encoders.append(module)
        self.decoders = []
        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i], stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
            self.add_module('decoder{}'.format(i), module)
            self.decoders.append(module)
        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d
        linear = conv(self.dec_channels[-1], 1, 1)
        self.add_module('linear', linear)
        self.complex = complex
        self.padding_mode = padding_mode
        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, bd):
        if self.complex:
            x = bd['X']
        else:
            x = bd['mag_X']
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)
        mask = self.linear(p)
        mask = torch.tanh(mask)
        bd['M_hat'] = mask
        return bd

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels, model_complexity, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2]
            self.enc_kernel_sizes = [(7, 5), (7, 5), (5, 3), (5, 3), (5, 3)]
            self.enc_strides = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]
            self.enc_paddings = [(2, 1), None, None, None, None]
            self.dec_channels = [0, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2]
            self.dec_kernel_sizes = [(4, 3), (4, 4), (6, 4), (6, 4), (7, 5)]
            self.dec_strides = [(2, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
            self.dec_paddings = [(1, 1), (1, 1), (2, 1), (2, 1), (2, 1)]
        elif model_depth == 20:
            self.enc_channels = [input_channels, model_complexity, model_complexity, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, 128]
            self.enc_kernel_sizes = [(7, 1), (1, 7), (6, 4), (7, 5), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]
            self.enc_strides = [(1, 1), (1, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)]
            self.enc_paddings = [(3, 0), (0, 3), None, None, None, None, None, None, None, None]
            self.dec_channels = [0, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2, model_complexity * 2]
            self.dec_kernel_sizes = [(4, 3), (4, 2), (4, 3), (4, 2), (4, 3), (4, 2), (6, 3), (7, 5), (1, 7), (7, 1)]
            self.dec_strides = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (1, 1), (1, 1)]
            self.dec_paddings = [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1), (1, 0), (2, 1), (2, 1), (0, 3), (3, 0)]
        else:
            raise ValueError('Unknown model depth : {}'.format(model_depth))


def cut_padding(y, required_length, random_state, deterministic=False):
    if isinstance(y, list):
        audio_length = y[0].shape[-1]
    else:
        audio_length = y.shape[-1]
    if audio_length < required_length:
        if deterministic:
            pad_left = 0
        else:
            pad_left = random_state.randint(required_length - audio_length + 1)
        pad_right = required_length - audio_length - pad_left
        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = F.pad(y[i], (pad_left, pad_right))
            audio_length = y[0].shape[-1]
        else:
            y = F.pad(y, (pad_left, pad_right))
            audio_length = y.shape[-1]
    if deterministic:
        audio_begin = 0
    else:
        audio_begin = random_state.randint(audio_length - required_length + 1)
    audio_end = required_length + audio_begin
    if isinstance(y, list):
        for i in range(len(y)):
            y[i] = y[i][..., audio_begin:audio_end]
    else:
        y = y[..., audio_begin:audio_end]
    return y


class SourceSeparator(nn.Module):

    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode):
        """
        :param complex: Whether to use complex networks.
        :param model_complexity:
        :param model_depth: Only two options are available : 10, 20
        :param log_amp: Whether to use log amplitude to estimate signals
        :param padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
        """
        super().__init__()
        self.net = nn.Sequential(STFT(complex=complex, log_amp=log_amp), UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode), ApplyMask(complex=complex, log_amp=log_amp), ISTFT(complex=complex, log_amp=log_amp))

    def forward(self, x, istft=True):
        if istft:
            return self.net(x)
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            return x

    def inference_one_audio(self, audio, normalize=True):
        """
        :param audio: channel x samples (tensor, float) 
        :return: 
        """
        audict = SourceSeparator.preprocess_audio(audio, sequence_length=16384)
        with torch.no_grad():
            for k, v in audict.items():
                audict[k] = v.unsqueeze(1)
            Y_hat = self.forward(audict, istft=False).squeeze(1)
            y_hat = istft(Y_hat, HOP_LENGTH, length=audio.shape[-1])
            if normalize:
                mx = y_hat.max(dim=-1)[0].view(y_hat.shape[0], -1)
                mn = y_hat.min(dim=-1)[0].view(y_hat.shape[0], -1)
                y_hat = 2 * (y_hat - mn) / (mx - mn) - 1.0
        return y_hat

    @staticmethod
    def preprocess_audio(x, sequence_length=None):
        assert sequence_length is not None
        audio_length = x.shape[-1]
        if sequence_length is not None:
            if audio_length % sequence_length > 0:
                target_length = (audio_length // sequence_length + 1) * sequence_length
            else:
                target_length = audio_length
            x = cut_padding(x, target_length, np.random.RandomState(0), deterministic=True)
        x_max = x.max(dim=-1)[0].view(x.shape[0], -1)
        x_min = x.min(dim=-1)[0].view(x.shape[0], -1)
        x = 2 * (x - x_min) / (x_max - x_min) - 1.0
        rt = dict(x=x, x_max=x_max, x_min=x_min)
        return rt


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ComplexConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ComplexConvTranspose2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4], 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sweetcocoa_DeepComplexUNetPyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

