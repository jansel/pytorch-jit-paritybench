import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
hubconf = _module
inference = _module
discriminator = _module
generator = _module
identity = _module
multiscale = _module
res_stack = _module
preprocess = _module
trainer = _module
audio_processing = _module
hparams = _module
plotting = _module
stft = _module
train = _module
utils = _module
validation = _module
writer = _module

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


import random


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from scipy.io.wavfile import write


import torch.nn as nn


import torch.nn.functional as F


from scipy.signal import get_window


from torch.autograd import Variable


import math


import itertools


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.ModuleList([nn.Sequential(nn.ReflectionPad1d(7), nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)), nn.LeakyReLU(0.2, inplace=True)), nn.Sequential(nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)), nn.LeakyReLU(0.2, inplace=True)), nn.Sequential(nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)), nn.LeakyReLU(0.2, inplace=True)), nn.Sequential(nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)), nn.LeakyReLU(0.2, inplace=True)), nn.Sequential(nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)), nn.LeakyReLU(0.2, inplace=True)), nn.Sequential(nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)), nn.LeakyReLU(0.2, inplace=True)), nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))])

    def forward(self, x):
        """
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        """
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


MAX_WAV_VALUE = 32768.0


class ResStack(nn.Module):

    def __init__(self, channel):
        super(ResStack, self).__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(3 ** i), nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3 ** i)), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))) for i in range(3)])
        self.shortcuts = nn.ModuleList([nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)) for i in range(3)])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class Generator(nn.Module):

    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.generator = nn.Sequential(nn.ReflectionPad1d(3), nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1)), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)), ResStack(256), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)), ResStack(128), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)), ResStack(64), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)), ResStack(32), nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)), nn.Tanh())

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        zero = torch.full((1, self.mel_channel, 10), -11.5129)
        mel = torch.cat((mel, zero), dim=2)
        audio = self.forward(mel)
        audio = audio.squeeze()
        audio = audio[:-(hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()
        return audio


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([Discriminator() for _ in range(3)])
        self.pooling = nn.ModuleList([Identity()] + [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)])

    def forward(self, x):
        ret = list()
        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))
        return ret


def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, (None), :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, (None), :])
        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0).cpu()
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, (approx_nonzero_indices)] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'mel_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResStack,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
]

class Test_seungwonpark_melgan(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

