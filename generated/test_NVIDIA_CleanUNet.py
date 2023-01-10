import sys
_module = sys.modules[__name__]
del sys
dataset = _module
denoise = _module
distributed = _module
network = _module
python_eval = _module
stft_loss = _module
train = _module
util = _module

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


from scipy.io.wavfile import read as wavread


import warnings


import torch


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


import random


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import torchaudio


from copy import deepcopy


import torch.nn as nn


from scipy.io.wavfile import write as wavwrite


import time


import torch.distributed as dist


from torch.autograd import Variable


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import functools


from math import cos


from math import pi


from math import floor


from math import sin


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64, d_model=512, d_inner=2048, dropout=0.1, n_position=624, scale_emb=False):
        super().__init__()
        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        enc_output = src_seq
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


def padding(x, D, K, S):
    """padding zeroes to x so that denoised audio has the same length"""
    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)
    for _ in range(D):
        L = (L - 1) * S + K
    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


class CleanUNet(nn.Module):
    """ CleanUNet architecture. """

    def __init__(self, channels_input=1, channels_output=1, channels_H=64, max_H=768, encoder_n_layers=8, kernel_size=4, stride=2, tsfm_n_layers=3, tsfm_n_head=8, tsfm_d_model=512, tsfm_d_inner=2048):
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """
        super(CleanUNet, self).__init__()
        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(encoder_n_layers):
            self.encoder.append(nn.Sequential(nn.Conv1d(channels_input, channels_H, kernel_size, stride), nn.ReLU(), nn.Conv1d(channels_H, channels_H * 2, 1), nn.GLU(dim=1)))
            channels_input = channels_H
            if i == 0:
                self.decoder.append(nn.Sequential(nn.Conv1d(channels_H, channels_H * 2, 1), nn.GLU(dim=1), nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride)))
            else:
                self.decoder.insert(0, nn.Sequential(nn.Conv1d(channels_H, channels_H * 2, 1), nn.GLU(dim=1), nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride), nn.ReLU()))
            channels_output = channels_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)
        self.tsfm_conv1 = nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(d_word_vec=tsfm_d_model, n_layers=tsfm_n_layers, n_head=tsfm_n_head, d_k=tsfm_d_model // tsfm_n_head, d_v=tsfm_d_model // tsfm_n_head, d_model=tsfm_d_model, d_inner=tsfm_d_inner, dropout=0.0, n_position=0, scale_emb=False)
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1
        std = noisy_audio.std(dim=2, keepdim=True) + 0.001
        noisy_audio /= std
        x = padding(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]
        len_s = x.shape[-1]
        attn_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()
        x = self.tsfm_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x += skip_i[:, :, :x.shape[-1]]
            x = upsampling_block(x)
        x = x[:, :, :L] * std
        return x


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

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
    if is_pytorch_17plus:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-07)).transpose(2, 1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window', band='full'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.register_buffer('window', getattr(torch, window)(win_length))

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
        if self.band == 'high':
            freq_mask_ind = x_mag.shape[1] // 2
            sc_loss = self.spectral_convergence_loss(x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :])
            mag_loss = self.log_stft_magnitude_loss(x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :])
        elif self.band == 'full':
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        else:
            raise NotImplementedError
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window', sc_lambda=0.1, mag_lambda=0.1, band='full'):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            *_lambda (float): a balancing factor across different losses.
            band (str): high-band or full-band loss

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, band)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))
            y = y.view(-1, y.size(2))
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss *= self.sc_lambda
        sc_loss /= len(self.stft_losses)
        mag_loss *= self.mag_lambda
        mag_loss /= len(self.stft_losses)
        return sc_loss, mag_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CleanUNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'd_inner': 4, 'n_head': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LogSTFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_model': 4, 'd_k': 4, 'd_v': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_in': 4, 'd_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpectralConvergenceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 512]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_NVIDIA_CleanUNet(_paritybench_base):
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

