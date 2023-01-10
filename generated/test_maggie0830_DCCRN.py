import sys
_module = sys.modules[__name__]
del sys
complex_progress = _module
module = _module
net_config = _module
si_snr = _module
test = _module
train = _module
train_utils = _module
utils = _module
wav_loader = _module

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


from torch.utils.data import DataLoader


import matplotlib.pyplot as plt


import numpy as np


import time


import torch.nn.functional as F


from torch.utils.data import Dataset


class ComplexConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, device, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm_re = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        batch_size = x.size(1)
        h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        real_real, (h_real, c_real) = self.lstm_re(x[..., 0], (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x[..., 1], (h_imag, c_imag))
        real = real_real - imag_imag
        h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        imag_real, (h_real, c_real) = self.lstm_re(x[..., 1], (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x[..., 0], (h_imag, c_imag))
        imaginary = imag_real + real_imag
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexDense(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear_read = nn.Linear(in_channel, out_channel)
        self.linear_imag = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        real = self.linear_read(real)
        imag = self.linear_imag(imag)
        out = torch.stack((real, imag), dim=-1)
        return out


class ComplexBatchNormal(nn.Module):

    def __init__(self, C, H, W, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.epsilon = 1e-05
        self.running_mean_real = None
        self.running_mean_imag = None
        self.Vrr = None
        self.Vri = None
        self.Vii = None

    def forward(self, x, train=True):
        B, C, H, W, D = x.size()
        real = x[..., 0]
        imaginary = x[..., 1]
        if train:
            mu_real = torch.mean(real, dim=0)
            mu_imag = torch.mean(imaginary, dim=0)
            broadcast_mu_real = mu_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = mu_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            Vrr = torch.mean(real_centred * real_centred, 0) + self.epsilon
            Vii = torch.mean(imag_centred * imag_centred, 0) + self.epsilon
            Vri = torch.mean(real_centred * imag_centred, 0)
            if self.Vrr is None:
                self.running_mean_real = mu_real
                self.running_mean_imag = mu_imag
                self.Vrr = Vrr
                self.Vri = Vri
                self.Vii = Vii
            else:
                self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, B)
        else:
            broadcast_mu_real = self.running_mean_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = self.running_mean_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            return self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, B)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, B):
        tau = Vrr + Vii
        delta = Vrr * Vii - Vri ** 2
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)
        Wrr = ((Vii + s) * inverse_st).repeat(B, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).repeat(B, 1, 1, 1)
        Wri = (-Vri * inverse_st).repeat(B, 1, 1, 1)
        n_real = Wrr * real_centred + Wri * imag_centred
        n_imag = Wii * imag_centred + Wri * real_centred
        broadcast_gamma_rr = self.gamma_rr.repeat(B, 1, 1, 1)
        broadcast_gamma_ri = self.gamma_ri.repeat(B, 1, 1, 1)
        broadcast_gamma_ii = self.gamma_ii.repeat(B, 1, 1, 1)
        broadcast_beta = self.beta.repeat(B, 1, 1, 1)
        bn_real = broadcast_gamma_rr * n_real + broadcast_gamma_ri * n_imag + broadcast_beta
        bn_imag = broadcast_gamma_ri * n_real + broadcast_gamma_ii * n_imag + broadcast_beta
        return torch.stack((bn_real, bn_imag), dim=-1)


class ComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class STFT(nn.Module):

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.stft = audio_nn.STFT(fft_length=self.n_fft, hop_length=self.hop_length, win_length=win_length)

    def forward(self, signal):
        with torch.no_grad():
            x = self.stft(signal)
            mag, phase = audio_nn.magphase(x, power=1.0)
        mix = torch.stack((mag, phase), dim=-1)
        return mix.unsqueeze(1)


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

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length

    def forward(self, x):
        B, C, F, T, D = x.shape
        x = x.view(B, F, T, D)
        x_istft = istft(x, hop_length=self.hop_length, length=600)
        return x_istft.view(B, C, -1)


class Encoder(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train):
        x = self.conv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train=True):
        x = self.transconv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class DCCRN(nn.Module):

    def __init__(self, net_params, device, batch_size=36):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params['dense'][0], net_params['dense'][1])
        self.decoders = []
        en_channels = net_params['encoder_channels']
        en_ker_size = net_params['encoder_kernel_sizes']
        en_strides = net_params['encoder_strides']
        en_padding = net_params['encoder_paddings']
        encoder_chw = net_params['encoder_chw']
        decoder_chw = net_params['decoder_chw']
        for index in range(len(en_channels) - 1):
            model = Encoder(in_channel=en_channels[index], out_channel=en_channels[index + 1], kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index], chw=encoder_chw[index])
            self.add_module('encoder{%d}' % index, model)
            self.encoders.append(model)
        lstm_dims = net_params['lstm_dim']
        for index in range(len(net_params['lstm_dim']) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_dims[index + 1], num_layers=net_params['lstm_layer_num'], device=self.device)
            self.lstms.append(model)
            self.add_module('lstm{%d}' % index, model)
        de_channels = net_params['decoder_channels']
        de_ker_size = net_params['decoder_kernel_sizes']
        de_strides = net_params['decoder_strides']
        de_padding = net_params['decoder_paddings']
        for index in range(len(de_channels) - 1):
            model = Decoder(in_channel=de_channels[index] + en_channels[len(self.encoders) - index], out_channel=de_channels[index + 1], kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index], chw=decoder_chw[index])
            self.add_module('decoder{%d}' % index, model)
            self.decoders.append(model)
        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        self.decoders = nn.ModuleList(self.decoders)
        self.linear = ComplexConv2d(in_channel=2, out_channel=1, kernel_size=1, stride=1)

    def forward(self, x, train=True):
        skiper = []
        for index, encoder in enumerate(self.encoders):
            skiper.append(x)
            x = encoder(x, train)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D)
        lstm_ = lstm_.permute(2, 0, 1, 3)
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_)
        lstm_ = lstm_.permute(1, 0, 2, 3)
        lstm_out = lstm_.reshape(B * T, -1, D)
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4)
        for index, decoder in enumerate(self.decoders):
            p = decoder(p, train)
            p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
        mask = torch.tanh(self.linear(p))
        return mask


class DCCRN_(nn.Module):

    def __init__(self, n_fft, hop_len, net_params, batch_size, device, win_length):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_length=win_length)
        self.DCCRN = DCCRN(net_params, device=device, batch_size=batch_size)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length)

    def forward(self, signal, train=True):
        stft = self.stft(signal)
        mask_predict = self.DCCRN(stft, train=train)
        predict = stft * mask_predict
        clean = self.istft(predict)
        return clean


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ComplexBatchNormal,
     lambda: ([], {'C': 4, 'H': 4, 'W': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (ComplexConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ComplexConvTranspose2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ComplexDense,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ComplexLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_maggie0830_DCCRN(_paritybench_base):
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

