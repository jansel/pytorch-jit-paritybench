import sys
_module = sys.modules[__name__]
del sys
attentions = _module
audio_processing = _module
commons = _module
data_utils = _module
inference_waveglow_vocoder = _module
models = _module
modules = _module
monotonic_align = _module
setup = _module
stft = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
train = _module
unet = _module
utils = _module
convert_model = _module
denoiser = _module
distributed = _module
glow = _module
glow_old = _module
inference = _module
mel2samp = _module
train = _module

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


import copy


import math


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


from scipy.signal import get_window


import random


import torch.utils.data


import matplotlib.pyplot as plt


import scipy


import torch.nn.functional as F


from torch.autograd import Variable


from torch import optim


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import torch.multiprocessing as mp


import torch.distributed as dist


from torch import einsum


from inspect import isfunction


from functools import partial


import logging


from scipy.io.wavfile import read


import time


from scipy.io.wavfile import write


from torch.utils.data.distributed import DistributedSampler


class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == 'gelu':
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=0.0001):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0.0, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = *key.size(), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, 'Relative attention is only available for self-attention.'
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, 'Proximal bias is only available for self-attention.'
            scores = scores + self._attention_bias_proximal(t_s)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -10000.0)
            if self.block_length is not None:
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -10000.0 * (1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(relative_embeddings, commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
        batch, heads, length, _ = x.size()
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
        batch, heads, length, _ = x.size()
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class Encoder(nn.Module):

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=None, block_length=None, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class CouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale
        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.wn = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        b, c, t = x.size()
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)
        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-06 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


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
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
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
        if input_data.device.type == 'cuda':
            input_data = input_data.view(num_batches, 1, num_samples)
            input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
            input_data = input_data.squeeze(1)
            forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, padding=0)
            cutoff = int(self.filter_length / 2 + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]
        else:
            x = input_data.detach().numpy()
            real_part = []
            imag_part = []
            for y in x:
                y_ = stft(y, self.filter_length, self.hop_length, self.win_length, self.window)
                real_part.append(y_.real[None, :, :])
                imag_part.append(y_.imag[None, :, :])
            real_part = np.concatenate(real_part, 0)
            imag_part = np.concatenate(imag_part, 0)
            real_part = torch.from_numpy(real_part)
            imag_part = torch.from_numpy(imag_part)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part.data, real_part.data)
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        if magnitude.device.type == 'cuda':
            inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, self.inverse_basis, stride=self.hop_length, padding=0)
            if self.window is not None:
                window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
                approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
                window_sum = torch.from_numpy(window_sum)
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
                inverse_transform *= float(self.filter_length) / self.hop_length
            inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
            inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
            inverse_transform = inverse_transform.squeeze(1)
        else:
            x_org = recombine_magnitude_phase.detach().numpy()
            n_b, n_f, n_t = x_org.shape
            x = np.empty([n_b, n_f // 2, n_t], dtype=np.complex64)
            x.real = x_org[:, :n_f // 2]
            x.imag = x_org[:, n_f // 2:]
            inverse_transform = []
            for y in x:
                y_ = istft(y, self.hop_length, self.win_length, self.window)
                inverse_transform.append(y_[None, :])
            inverse_transform = np.concatenate(inverse_transform, 0)
            inverse_transform = torch.from_numpy(inverse_transform)
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


class TacotronSTFT(nn.Module):

    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0):
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


class DurationPredictor(nn.Module):

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, filter_channels_dp, n_heads, n_layers, kernel_size, p_dropout, window_size=None, block_length=None, mean_only=False, prenet=False, gin_channels=0):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        if prenet:
            self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
        self.encoder = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size=window_size, block_length=block_length)
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1)
        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)
        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)
        logw = self.proj_w(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask


class DiffusionDecoder(nn.Module):

    def __init__(self, unet_channels=64, unet_in_channels=2, unet_out_channels=1, dim_mults=(1, 2, 4), groups=8, with_time_emb=True, beta_0=0.05, beta_1=20, N=1000, T=1):
        super().__init__()
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.N = N
        self.T = T
        self.delta_t = T * 1.0 / N
        self.discrete_betas = torch.linspace(beta_0, beta_1, N)
        self.unet = unet.Unet(dim=unet_channels, out_dim=unet_out_channels, dim_mults=dim_mults, groups=groups, channels=unet_in_channels, with_time_emb=with_time_emb)

    def marginal_prob(self, mu, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None]) * x + (1 - torch.exp(log_mean_coeff[:, None, None])) * mu
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def cal_loss(self, x, mu, t, z, std, g=None):
        time_steps = t * (self.N - 1)
        if g:
            x = torch.stack([x, mu, g], 1)
        else:
            x = torch.stack([x, mu], 1)
        grad = self.unet(x, time_steps)
        loss = torch.square(grad + z / std[:, None, None]) * torch.square(std[:, None, None])
        return loss

    def forward(self, mu, y=None, g=None, gen=False):
        if not gen:
            t = torch.FloatTensor(y.shape[0]).uniform_(0, self.T - self.delta_t) + self.delta_t
            mean, std = self.marginal_prob(mu, y, t)
            z = torch.randn_like(y)
            x = mean + std[:, None, None] * z
            loss = self.cal_loss(x, mu, t, z, std, g)
            return loss
        else:
            with torch.no_grad():
                y_T = torch.randn_like(mu) + mu
                y_t_plus_one = y_T
                y_t = None
                for n in tqdm(range(self.N - 1, 0, -1)):
                    t = torch.FloatTensor(1).fill_(n)
                    if g:
                        x = torch.stack([y_t_plus_one, mu, g], 1)
                    else:
                        x = torch.stack([y_t_plus_one, mu], 1)
                    grad = self.unet(x, t)
                    y_t = y_t_plus_one - 0.5 * self.delta_t * self.discrete_betas[n] * (mu - y_t_plus_one - grad)
                    y_t_plus_one = y_t
            return y_t


class DiffusionGenerator(nn.Module):

    def __init__(self, n_vocab, hidden_channels, filter_channels, filter_channels_dp, enc_out_channels, kernel_size=3, n_heads=2, n_layers_enc=6, p_dropout=0.0, n_speakers=0, gin_channels=0, window_size=None, block_length=None, mean_only=False, hidden_channels_enc=None, hidden_channels_dec=None, prenet=False, dec_unet_channels=64, dec_dim_mults=(1, 2, 4), dec_groups=8, dec_unet_in_channels=2, dec_unet_out_channels=1, dec_with_time_emb=True, beta_0=0.05, beta_1=20, N=1000, T=1, **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.enc_out_channels = enc_out_channels
        self.dec_in_channels = enc_out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_speakers = n_speakers
        self.gin_channels = enc_out_channels
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet
        self.dec_unet_channels = dec_unet_channels
        self.dec_unet_in_channels = dec_unet_in_channels if self.n_speakers < 1 else dec_unet_in_channels + 1
        self.dec_unet_out_channels = dec_unet_out_channels
        self.dec_dim_mults = dec_dim_mults
        self.dec_groups = dec_groups
        self.dec_with_time_emb = dec_with_time_emb
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.N = N
        self.T = T
        self.encoder = TextEncoder(n_vocab, enc_out_channels, hidden_channels_enc or hidden_channels, filter_channels, filter_channels_dp, n_heads, n_layers_enc, kernel_size, p_dropout, window_size=window_size, block_length=block_length, mean_only=mean_only, prenet=prenet, gin_channels=gin_channels)
        self.decoder = DiffusionDecoder(unet_channels=self.dec_unet_channels, unet_in_channels=self.dec_unet_in_channels, unet_out_channels=self.dec_unet_out_channels, dim_mults=self.dec_dim_mults, groups=self.dec_groups, with_time_emb=self.dec_with_time_emb, beta_0=self.beta_0, beta_1=self.beta_1, N=self.N, T=self.T)
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1.0, length_scale=1.0):
        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)
        x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)
        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        if gen:
            attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2)
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2)
            logw_ = torch.log(1e-08 + torch.sum(attn, -1)) * x_mask
            y = self.decoder(z_m, gen=True)
            return (y, z_m, z_logs, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
        else:
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)
                logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * y ** 2)
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), y)
                logp4 = torch.sum(-0.5 * x_m ** 2 * x_s_sq_r, [1]).unsqueeze(-1)
                logp = logp1 + logp2 + logp3 + logp4
                attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2)
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2)
            logw_ = torch.log(1e-08 + torch.sum(attn, -1)) * x_mask
            grad_loss = self.decoder(mu=z_m, y=y, g=g, gen=False).mean()
            return grad_loss, (z_m, z_logs, z_mask), (attn, logw, logw_)


class ConvReluNorm(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, 'Number of layers should be larger than 0.'
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels, kernel_size):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()
        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](audio), self.cond_layers[i](spect), torch.IntTensor([self.n_channels]))
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels:, :]
            else:
                skip_acts = res_skip_acts
            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        return self.end(output)


class ActNorm(nn.Module):

    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2))
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True
        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len
        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - m ** 2
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-06))
            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape)
            logs_init = (-logs).view(*self.logs.shape)
            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):

    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian
        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)
        if reverse:
            if hasattr(self, 'weight_inv'):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float())
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)
        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(self.weight.float())


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x):
        return self.block(x)


def exists(x):
    return x is not None


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Unet(nn.Module):

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), groups=8, channels=2, with_time_emb=True):
        super().__init__()
        self.channels = channels
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        else:
            time_dim = None
            self.time_mlp = None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim), ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Downsample(dim_out) if not is_last else nn.Identity()]))
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(nn.ModuleList([ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim), ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Upsample(dim_in) if not is_last else nn.Identity()]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time):
        length = x.shape[-1]
        x = F.pad(x, (0, math.ceil(length / 4) * 4 - length), 'constant', 0)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)
        return torch.squeeze(self.final_conv(x), 1)[:, :, :length]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda : torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda : torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):

    def __init__(self, denoise_fn, *, image_size, channels=3, timesteps=1000, loss_type='l1', betas=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)
        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4, win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length, hop_length=int(filter_length / n_overlap), win_length=win_length)
        if mode == 'zeros':
            mel_input = torch.zeros((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)
        else:
            raise Exception('Mode {} if not supported'.format(mode))
        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)
        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


class WaveGlowLoss(torch.nn.Module):

    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]
        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list


class WaveGlow(torch.nn.Module):

    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every, n_early_size, WN_config):
        super(WaveGlow, self).__init__()
        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        n_half = int(n_group / 2)
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        return None
        """
        forward_input[0] = audio: batch x time
        forward_input[1] = upsamp_spectrogram:  batch x n_cond_channels x time
        """
        """
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        s_list = []
        s_conv_list = []

        for k in range(self.n_flows):
            if k%4 == 0 and k > 0:
                output_audio.append(audio[:,:self.n_multi,:])
                audio = audio[:,self.n_multi:,:]

            # project to new basis
            audio, s = self.convinv[k](audio)
            s_conv_list.append(s)

            n_half = int(audio.size(1)/2)
            if k%2 == 0:
                audio_0 = audio[:,:n_half,:]
                audio_1 = audio[:,n_half:,:]
            else:
                audio_1 = audio[:,:n_half,:]
                audio_0 = audio[:,n_half:,:]

            output = self.nn[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(s)*audio_1 + b
            s_list.append(s)

            if k%2 == 0:
                audio = torch.cat([audio[:,:n_half,:], audio_1],1)
            else:
                audio = torch.cat([audio_1, audio[:,n_half:,:]], 1)
        output_audio.append(audio)
        return torch.cat(output_audio,1), s_list, s_conv_list
        """

    def infer(self, spect, sigma=1.0):
        spect = self.upsample(spect)
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.HalfTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()
        else:
            audio = torch.FloatTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()
        audio = torch.autograd.Variable(sigma * audio)
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            if k % 2 == 0:
                audio_0 = audio[:, :n_half, :]
                audio_1 = audio[:, n_half:, :]
            else:
                audio_1 = audio[:, :n_half, :]
                audio_0 = audio[:, n_half:, :]
            output = self.WN[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            if k % 2 == 0:
                audio = torch.cat([audio[:, :n_half, :], audio_1], 1)
            else:
                audio = torch.cat([audio_1, audio[:, n_half:, :]], 1)
            audio = self.convinv[k](audio, reverse=True)
            if k % 4 == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma * z, audio), 1)
        return audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'hidden_channels': 4, 'filter_channels': 4, 'n_heads': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (InvConvNear,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Invertible1x1Conv,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'channels': 4, 'out_channels': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WaveGlowLoss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
]

class Test_WelkinYang_GradTTS(_paritybench_base):
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

