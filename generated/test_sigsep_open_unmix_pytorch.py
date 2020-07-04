import sys
_module = sys.modules[__name__]
del sys
data = _module
eval = _module
hubconf = _module
model = _module
test = _module
tests = _module
test_datasets = _module
test_inference = _module
test_model = _module
test_regression = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch.nn import LSTM


from torch.nn import Linear


from torch.nn import BatchNorm1d


from torch.nn import Parameter


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


import numpy as np


import random


import copy


class NoOp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):

    def __init__(self, n_fft=4096, n_hop=1024, center=False):
        super(STFT, self).__init__()
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=
            False)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        nb_samples, nb_channels, nb_timesteps = x.size()
        x = x.reshape(nb_samples * nb_channels, -1)
        stft_f = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center, normalized=False,
            onesided=True, pad_mode='reflect')
        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, self.
            n_fft // 2 + 1, -1, 2)
        return stft_f


class Spectrogram(nn.Module):

    def __init__(self, power=1, mono=True):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)
        return stft_f.permute(2, 0, 1, 3)


class OpenUnmix(nn.Module):

    def __init__(self, n_fft=4096, n_hop=1024, input_is_spectrogram=False,
        hidden_size=512, nb_channels=2, sample_rate=44100, nb_layers=3,
        input_mean=None, input_scale=None, max_bin=None, unidirectional=
        False, power=1):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        super(OpenUnmix, self).__init__()
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins
        self.hidden_size = hidden_size
        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=nb_channels == 1)
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)
        self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)
        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2
        self.lstm = LSTM(input_size=hidden_size, hidden_size=
            lstm_hidden_size, num_layers=nb_layers, bidirectional=not
            unidirectional, batch_first=False, dropout=0.4)
        self.fc2 = Linear(in_features=hidden_size * 2, out_features=
            hidden_size, bias=False)
        self.bn2 = BatchNorm1d(hidden_size)
        self.fc3 = Linear(in_features=hidden_size, out_features=self.
            nb_output_bins * nb_channels, bias=False)
        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[:self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)
        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[:self.nb_bins]
                ).float()
        else:
            input_scale = torch.ones(self.nb_bins)
        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)
        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def forward(self, x):
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        mix = x.detach().clone()
        x = x[(...), :self.nb_bins]
        x += self.input_mean
        x *= self.input_scale
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
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sigsep_open_unmix_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(NoOp(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

