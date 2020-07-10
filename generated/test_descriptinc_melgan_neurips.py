import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
mel2wav = _module
dataset = _module
interface = _module
modules = _module
utils = _module
generate_from_folder = _module
train = _module

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


import torch.utils.data


import torch.nn.functional as F


import numpy as np


import random


import torch.nn as nn


from torch.nn.utils import weight_norm


import scipy.io.wavfile


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import time


class Audio2Mel(nn.Module):

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None):
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window', window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), 'reflect').squeeze(1)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-05))
        return log_mel_spec


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(dilation), WNConv1d(dim, dim, kernel_size=3, dilation=dilation), nn.LeakyReLU(0.2), WNConv1d(dim, dim, kernel_size=1))
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))
        model = [nn.ReflectionPad1d(3), WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0)]
        for i, r in enumerate(ratios):
            model += [nn.LeakyReLU(0.2), WNConvTranspose1d(mult * ngf, mult * ngf // 2, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)]
            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]
            mult //= 2
        model += [nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), WNConv1d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):

    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()
        model['layer_0'] = nn.Sequential(nn.ReflectionPad1d(7), WNConv1d(1, ndf, kernel_size=15), nn.LeakyReLU(0.2, True))
        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)
            model['layer_%d' % n] = nn.Sequential(WNConv1d(nf_prev, nf, kernel_size=stride * 10 + 1, stride=stride, padding=stride * 5, groups=nf_prev // 4), nn.LeakyReLU(0.2, True))
        nf = min(nf * 2, 1024)
        model['layer_%d' % (n_layers + 1)] = nn.Sequential(WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(0.2, True))
        model['layer_%d' % (n_layers + 2)] = WNConv1d(nf, 1, kernel_size=3, stride=1, padding=1)
        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):

    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f'disc_{i}'] = NLayerDiscriminator(ndf, n_layers, downsampling_factor)
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'input_size': 4, 'ngf': 4, 'n_residual_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
]

class Test_descriptinc_melgan_neurips(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

