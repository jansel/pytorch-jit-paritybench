import sys
_module = sys.modules[__name__]
del sys
dataset = _module
discriminator = _module
evalution = _module
generator = _module
inference = _module
stft_loss = _module
train = _module
utils = _module

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


import math


import random


import torch


import torch.utils.data


import numpy as np


from scipy.io.wavfile import read


import torch.nn.functional as F


import torch.nn as nn


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


from torch import nn


from torch.nn import functional as F


from scipy.io.wavfile import write


import warnings


import itertools


import time


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DistributedSampler


from torch.utils.data import DataLoader


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


import matplotlib


import matplotlib.pylab as plt


class MelganDiscriminator(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(MelganDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([MelganDiscriminator(use_spectral_norm=True), MelganDiscriminator(), MelganDiscriminator()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()
        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x2 = self.c2(x)
        x2 = self.n2(x2)
        x3 = x1 * torch.sigmoid(x2)
        return x3


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self):
        super(SpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([Down2d(1, 32, (3, 9), (1, 2), (1, 4)), Down2d(32, 32, (3, 8), (1, 2), (1, 3)), Down2d(32, 32, (3, 8), (1, 2), (1, 3)), Down2d(32, 32, (3, 6), (1, 2), (1, 2))])
        self.conv = nn.Conv2d(32, 1, (32, 5), (32, 1), (0, 2))
        self.pool = nn.AvgPool2d((1, 2))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_r = []
        fmap_g = []
        fmap_rs = []
        fmap_gs = []
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y_hat = d(y_hat)
            fmap_r.append(y)
            fmap_g.append(y_hat)
        y = self.conv(y)
        fmap_r.append(y)
        y = self.pool(y)
        y_d_rs.append(torch.flatten(y, 1, -1))
        y_hat = self.conv(y_hat)
        fmap_g.append(y_hat)
        y_hat = self.pool(y_hat)
        y_d_gs.append(torch.flatten(y_hat, 1, -1))
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(self, idim: int, odim: int, n_layers: int=5, n_chans: int=512, n_filts: int=5, dropout_rate: float=0.5, use_batch_norm: bool=True):
        """Initialize postnet module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.BatchNorm1d(ochans), torch.nn.Tanh(), torch.nn.Dropout(dropout_rate))]
            else:
                self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.Tanh(), torch.nn.Dropout(dropout_rate))]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, odim, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.BatchNorm1d(odim), torch.nn.Dropout(dropout_rate))]
        else:
            self.postnet += [torch.nn.Sequential(torch.nn.Conv1d(ichans, odim, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False), torch.nn.Dropout(dropout_rate))]

    def forward(self, xs):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).
        """
        for postnet in self.postnet:
            xs = postnet(xs)
        return xs


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit
    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 - 0.95, padding=None, dilation=1, bias=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(residual_channels, gate_channels, kernel_size, *args, padding=padding, dilation=dilation, bias=bias, **kwargs)
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1d(gate_out_channels, residual_channels, 1, bias=bias)
        self.conv1x1_skip = nn.Conv1d(gate_out_channels, skip_out_channels, 1, bias=bias)

    def forward(self, x):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        splitdim = 1
        x = self.conv(x)
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        x = torch.tanh(a) * torch.sigmoid(b)
        s = self.conv1x1_skip(x)
        x = self.conv1x1_out(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, s


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels=1, bias=False, num_layers=20, num_stacks=2, kernel_size=3, residual_channels=128, gate_channels=128, skip_out_channels=128, postnet_layers=12, postnet_filts=32, use_batch_norm=False, postnet_dropout_rate=0.5):
        super().__init__()
        assert num_layers % num_stacks == 0
        num_layers_per_stack = num_layers // num_stacks
        self.first_conv = nn.Conv1d(in_channels, residual_channels, 3, padding=1, bias=bias)
        self.conv_layers = nn.ModuleList()
        for n_layer in range(num_layers):
            dilation = 2 ** (n_layer % num_layers_per_stack)
            conv = ResidualConv1dGLU(residual_channels, gate_channels, skip_out_channels=skip_out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, dropout=1 - 0.95)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.Sequential(nn.ReLU(True), nn.Conv1d(skip_out_channels, skip_out_channels, 1, bias=True), nn.ReLU(True), nn.Conv1d(skip_out_channels, out_channels, 1, bias=True))
        self.postnet = None if postnet_layers == 0 else Postnet(idim=in_channels, odim=out_channels, n_layers=postnet_layers, n_chans=residual_channels, n_filts=postnet_filts, use_batch_norm=use_batch_norm, dropout_rate=postnet_dropout_rate)

    def forward(self, x, with_postnet=False):
        x = self.first_conv(x)
        skips = 0
        for conv in self.conv_layers:
            x, h = conv(x)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        x = self.last_conv_layers(x)
        if not with_postnet:
            return x, None
        else:
            after_x = self.postnet(x)
            return x, after_x


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-07)).transpose(2, 1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window'):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Down2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Generator,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LogSTFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MelganDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {}),
     False),
    (Postnet,
     lambda: ([], {'idim': 4, 'odim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ResidualConv1dGLU,
     lambda: ([], {'residual_channels': 4, 'gate_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 2])], {}),
     True),
    (SpectralConvergengeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_rishikksh20_hifigan_denoiser(_paritybench_base):
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

