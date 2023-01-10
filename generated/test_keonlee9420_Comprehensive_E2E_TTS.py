import sys
_module = sys.modules[__name__]
del sys
audio = _module
audio_processing = _module
stft = _module
tools = _module
dataset = _module
audio_ds = _module
batcher = _module
constants = _module
conv_models = _module
embedding = _module
utils = _module
evaluate = _module
E2ETTS = _module
model = _module
blocks = _module
loss = _module
modules = _module
speaker_embedder = _module
preprocess = _module
ljspeech = _module
vctk = _module
synthesize = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
pinyin = _module
symbols = _module
train = _module
model = _module
pitch_tools = _module
tools = _module

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


import numpy as np


from scipy.signal import get_window


import torch.nn.functional as F


import torchaudio


from scipy.io.wavfile import write


import math


from scipy.io.wavfile import read


from torch.utils.data import Dataset


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.nn import Conv1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import spectral_norm


from torch.nn.utils import remove_weight_norm


from torch.nn import functional as F


import copy


from collections import OrderedDict


from torch.nn import ConvTranspose1d


import random


import re


from scipy.stats import betabinom


from sklearn.preprocessing import StandardScaler


from string import punctuation


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DistributedSampler


from torch.distributed import init_process_group


from torch.nn.parallel import DistributedDataParallel


from torch.cuda import amp


import itertools


from scipy.interpolate import interp1d


import matplotlib


from scipy.io import wavfile


from matplotlib import pyplot as plt


from sklearn.manifold import TSNE


def window_sumsquare(window, n_frames, hop_length, win_length, n_fft, dtype=np.float32, norm=None):
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

    def __init__(self, filter_length, hop_length, win_length, window='hann'):
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
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, torch.autograd.Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0).cpu()
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, torch.autograd.Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
        if self.window is not None:
            window_sum = window_sumsquare(self.window, magnitude.size(-1), hop_length=self.hop_length, win_length=self.win_length, n_fft=self.filter_length, dtype=np.float32)
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            inverse_transform *= float(self.filter_length) / self.hop_length
        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2)]
        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


class TacotronSTFT(torch.nn.Module):

    def __init__(self, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax):
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
        energy = torch.norm(magnitudes, dim=1)
        return mel_output, energy


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class TorchSTFT(torch.nn.Module):
    """ Log scaled Mel-Spectrogram """

    def __init__(self, preprocess_config, n_fft=None, f_clamp=True):
        super(TorchSTFT, self).__init__()
        self.n_fft = preprocess_config['preprocessing']['stft']['filter_length'] if n_fft is None else n_fft
        self.n_mel_channels = preprocess_config['preprocessing']['mel']['n_mel_channels']
        self.sampling_rate = preprocess_config['preprocessing']['audio']['sampling_rate']
        self.hop_length = preprocess_config['preprocessing']['stft']['hop_length']
        self.win_length = min(preprocess_config['preprocessing']['stft']['win_length'], self.n_fft)
        self.mel_fmin = preprocess_config['preprocessing']['mel']['mel_fmin']
        self.mel_fmax = preprocess_config['preprocessing']['mel']['mel_fmax'] if f_clamp else None
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, y, center=False, return_complex=False, return_energy=False, mel_fmax=None):
        if torch.min(y) < -1.0:
            None
        if torch.max(y) > 1.0:
            None
        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(self.sampling_rate, self.n_fft, self.n_mel_channels, self.mel_fmin, self.mel_fmax if mel_fmax is None else mel_fmax)
            self.mel_basis[str(self.mel_fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float()
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length)
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)), mode='reflect')
        y = y.squeeze(1)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window[str(y.device)], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=return_complex)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-09)
        magnitudes = spec.data
        energy = torch.norm(magnitudes, dim=1)
        spec = torch.matmul(self.mel_basis[str(self.mel_fmax) + '_' + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
        if return_energy:
            return spec, energy
        else:
            return spec


class BatchNorm1dTBC(nn.Module):

    def __init__(self, c):
        super(BatchNorm1dTBC, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        """

        :param x: [T, B, C]
        :return: [T, B, C]
        """
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        x = x.permute(2, 0, 1)
        return x


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(100000.0)


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.enable_torch_version = False
        if hasattr(F, 'multi_head_attention_forward'):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None, before_softmax=False, need_head_weights=False, enc_dec_attn_constraint_mask=None, reset_attn_weight=None):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if self.enable_torch_version and incremental_state is None and not static_kv and reset_attn_weight is None:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        if incremental_state is not None:
            None
            exit()
        else:
            saved_state = None
        if self.self_attention:
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            None
            exit()
        src_len = k.size(1)
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask[:, None].repeat([1, self.num_heads, 1, 1]).reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(enc_dec_attn_constraint_mask.unsqueeze(2).bool(), -1000000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1000000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        if reset_attn_weight is not None:
            if reset_attn_weight:
                self.last_attn_probs = attn_probs.detach()
            else:
                assert self.last_attn_probs is not None
                attn_probs = self.last_attn_probs
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None
        return attn, (attn_weights, attn_logits)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class CustomSwish(nn.Module):

    def forward(self, input_tensor):
        return Swish.apply(input_tensor)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerFFNLayer(nn.Module):

    def __init__(self, hidden_size, filter_size, padding='SAME', kernel_size=1, dropout=0.0, act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == 'SAME':
            self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            self.ffn_1 = nn.Sequential(nn.ConstantPad1d((kernel_size - 1, 0), 0.0), nn.Conv1d(hidden_size, filter_size, kernel_size))
        self.ffn_2 = Linear(filter_size, hidden_size)
        if self.act == 'swish':
            self.swish_fn = CustomSwish()

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            assert incremental_state is None, 'Nar-generation does not allow this.'
            exit(1)
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5
        if incremental_state is not None:
            x = x[-1:]
        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        if self.act == 'swish':
            x = self.swish_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class EncSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, padding='SAME', norm='ln', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            if norm == 'ln':
                self.layer_norm1 = LayerNorm(c)
            elif norm == 'bn':
                self.layer_norm1 = BatchNorm1dTBC(c)
            self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x)
            x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln', ffn_padding='SAME', ffn_act='gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(hidden_size, num_heads, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=kernel_size, padding=ffn_padding, norm=norm, act=ffn_act)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class FFTBlocks(nn.Module):

    def __init__(self, hidden_size, num_layers, max_seq_len=2000, ffn_kernel_size=9, dropout=None, num_heads=2, use_pos_embed=True, use_last_norm=True, norm='ln', ffn_padding='SAME', ffn_act='gelu', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = max_seq_len
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, init_size=max_seq_len)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(self.hidden_size, self.dropout, kernel_size=ffn_kernel_size, num_heads=num_heads, ffn_padding=ffn_padding, ffn_act=ffn_act) for _ in range(self.num_layers)])
        if self.use_last_norm:
            if norm == 'ln':
                self.layer_norm = nn.LayerNorm(embed_dim)
            elif norm == 'bn':
                self.layer_norm = BatchNorm1dTBC(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, padding_mask=None, attn_mask=None, return_hiddens=False):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        nonpadding_mask_TB = 1 - padding_mask.transpose(0, 1).float()[:, :, None]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)
            x = x.transpose(1, 2)
        else:
            x = x.transpose(0, 1)
        return x, padding_mask


class Decoder(FFTBlocks):

    def __init__(self, config):
        super().__init__(config['transformer']['decoder_hidden'], config['transformer']['decoder_layer'], max_seq_len=config['max_seq_len'] * 2, ffn_kernel_size=config['transformer']['ffn_kernel_size'], dropout=config['transformer']['decoder_dropout'], num_heads=config['transformer']['decoder_head'], ffn_padding=config['transformer']['ffn_padding'], ffn_act=config['transformer']['ffn_act'])


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


_pad = '_'


_punctuation = "!'(),.:;? "


_silences = ['@sp', '@spn', '@sil']


_special = '-'


class TextEncoder(FFTBlocks):

    def __init__(self, config):
        max_seq_len = config['max_seq_len']
        hidden_size = config['transformer']['encoder_hidden']
        super().__init__(hidden_size, config['transformer']['encoder_layer'], max_seq_len=max_seq_len * 2, ffn_kernel_size=config['transformer']['ffn_kernel_size'], dropout=config['transformer']['encoder_dropout'], num_heads=config['transformer']['encoder_head'], use_pos_embed=False, ffn_padding=config['transformer']['ffn_padding'], ffn_act=config['transformer']['ffn_act'])
        self.padding_idx = 0
        self.embed_tokens = Embedding(len(symbols) + 1, hidden_size, self.padding_idx)
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, self.padding_idx, init_size=max_seq_len)

    def forward(self, txt_tokens, encoder_padding_mask):
        """

        :param txt_tokens: [B, T]
        :param encoder_padding_mask: [B, T]
        :return: {
            "encoder_out": [T x B x C]
        }
        """
        x, src_word_emb = self.forward_embedding(txt_tokens)
        x, _ = super(TextEncoder, self).forward(x, encoder_padding_mask)
        return x, src_word_emb

    def forward_embedding(self, txt_tokens):
        txt_embs = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        x = txt_embs + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, txt_embs


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), lrelu_slope=0.1):
        super(ResBlock1, self).__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3), lrelu_slope=0.1):
        super(ResBlock2, self).__init__()
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Upsampler(torch.nn.Module):

    def __init__(self, preprocess_config, model_config, train_config):
        super(Upsampler, self).__init__()
        self.lrelu_slope = model_config['generator']['lrelu_slope']
        in_channels = model_config['transformer']['decoder_hidden']
        resblock_kernel_sizes = model_config['generator']['resblock_kernel_sizes']
        upsample_rates = model_config['generator']['upsample_rates']
        upsample_initial_channel = model_config['generator']['upsample_initial_channel']
        resblock = model_config['generator']['resblock']
        upsample_kernel_sizes = model_config['generator']['upsample_kernel_sizes']
        resblock_dilation_sizes = model_config['generator']['resblock_dilation_sizes']
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, self.lrelu_slope))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear', transpose=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        return x


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, n_mel_channels, n_att_channels, n_text_channels, temperature, multi_speaker):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.key_proj = nn.Sequential(ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'), torch.nn.ReLU(), ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True))
        self.query_proj = nn.Sequential(ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'), torch.nn.ReLU(), ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True), torch.nn.ReLU(), ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True))
        if multi_speaker:
            self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(-1, keys.shape[-1], -1)).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(-1, queries.shape[-1], -1)).transpose(1, 2)
        keys_enc = self.key_proj(keys)
        queries_enc = self.query_proj(queries)
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        attn = -self.temperature * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-08)
        attn_logprob = attn.clone()
        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float('inf'))
        attn = self.softmax(attn)
        return attn, attn_logprob


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The outputs are calculated in log domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME', dur_loss='mse'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dur_loss = dur_loss
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding == 'SAME' else (kernel_size - 1, 0), 0), Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))
        xs = xs * (1 - x_masks.float())[:, :, None]
        if self.dur_loss in ['mse']:
            xs = xs.squeeze(-1)
        return xs


class PitchPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding == 'SAME' else (kernel_size - 1, 0), 0), Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs, squeeze=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(1, -1))
        return xs.squeeze(-1) if squeeze else xs


class EnergyPredictor(PitchPredictor):
    pass


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])
    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), 'constant', 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), 'constant', 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)
        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


def denorm_f0(f0, uv, config, pitch_padding=None, min=None, max=None):
    if config['pitch_norm'] == 'standard':
        f0 = f0 * config['f0_std'] + config['f0_mean']
    if config['pitch_norm'] == 'log':
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and config['use_uv']:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


f0_bin = 256


f0_max = 1100.0


f0_mel_max = 1127 * np.log(1 + f0_max / 700)


f0_min = 50.0


f0_mel_min = 1127 * np.log(1 + f0_min / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        self.preprocess_config = preprocess_config
        self.binarization_start_steps = train_config['duration']['binarization_start_steps']
        self.use_pitch_embed = model_config['variance_embedding']['use_pitch_embed']
        self.use_energy_embed = model_config['variance_embedding']['use_energy_embed']
        self.predictor_grad = model_config['variance_predictor']['predictor_grad']
        self.hidden_size = model_config['transformer']['encoder_hidden']
        self.filter_size = model_config['variance_predictor']['filter_size']
        self.predictor_layers = model_config['variance_predictor']['predictor_layers']
        self.dropout = model_config['variance_predictor']['dropout']
        self.ffn_padding = model_config['transformer']['ffn_padding']
        self.kernel = model_config['variance_predictor']['predictor_kernel']
        self.aligner = AlignmentEncoder(n_mel_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'], n_att_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'], n_text_channels=model_config['transformer']['encoder_hidden'], temperature=model_config['duration_modeling']['aligner_temperature'], multi_speaker=model_config['multi_speaker'])
        self.duration_predictor = DurationPredictor(self.hidden_size, n_chans=self.filter_size, n_layers=model_config['variance_predictor']['dur_predictor_layers'], dropout_rate=self.dropout, padding=self.ffn_padding, kernel_size=model_config['variance_predictor']['dur_predictor_kernel'], dur_loss=train_config['loss']['dur_loss'])
        self.length_regulator = LengthRegulator()
        if self.use_pitch_embed:
            n_bins = model_config['variance_embedding']['pitch_n_bins']
            self.pitch_type = preprocess_config['preprocessing']['pitch']['pitch_type']
            self.use_uv = preprocess_config['preprocessing']['pitch']['use_uv']
            self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=self.filter_size, n_layers=self.predictor_layers, dropout_rate=self.dropout, odim=2 if self.pitch_type == 'frame' else 1, padding=self.ffn_padding, kernel_size=self.kernel)
            self.pitch_embedding = Embedding(n_bins, self.hidden_size, padding_idx=0)
        if self.use_energy_embed:
            energy_quantization = model_config['variance_embedding']['energy_quantization']
            assert energy_quantization in ['linear', 'log']
            n_bins = model_config['variance_embedding']['energy_n_bins']
            with open(os.path.join(preprocess_config['path']['preprocessed_path'], 'stats.json')) as f:
                stats = json.load(f)
                energy_min, energy_max = stats[f'energy'][:2]
            self.energy_predictor = EnergyPredictor(self.hidden_size, n_chans=self.filter_size, n_layers=self.predictor_layers, dropout_rate=self.dropout, odim=1, padding=self.ffn_padding, kernel_size=self.kernel)
            if energy_quantization == 'log':
                self.energy_bins = nn.Parameter(torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)), requires_grad=False)
            else:
                self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False)
            self.energy_embedding = Embedding(n_bins, self.hidden_size, padding_idx=0)

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out)

    def get_pitch_embedding(self, decoder_inp, f0, uv, control):
        decoder_inp = decoder_inp.detach() + self.predictor_grad * (decoder_inp - decoder_inp.detach())
        pitch_pred = self.pitch_predictor(decoder_inp) * control
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
        if self.use_uv and uv is None:
            uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv, self.preprocess_config['preprocessing']['pitch'])
        pitch = f0_to_coarse(f0_denorm)
        pitch_embed = self.pitch_embedding(pitch)
        pitch_pred = {'pitch_pred': pitch_pred, 'f0_denorm': f0_denorm}
        return pitch_pred, pitch_embed

    def get_energy_embedding(self, x, target, control):
        x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.energy_predictor(x, squeeze=True)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding

    def forward(self, x, text_embedding, src_len, max_src_len, src_mask, mel=None, mel_len=None, max_mel_len=None, mel_mask=None, pitch_target=None, energy_target=None, seq_start=None, attn_prior=None, speaker_embedding=None, step=1, p_control=1.0, e_control=1.0, d_control=1.0):
        if speaker_embedding is not None:
            x = x + speaker_embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        log_duration_prediction = self.duration_predictor(x.detach() + self.predictor_grad * (x - x.detach()), src_mask)
        attn_out = None
        if attn_prior is not None:
            attn_soft, attn_logprob = self.aligner(mel.transpose(1, 2), text_embedding.transpose(1, 2), src_mask.unsqueeze(-1), attn_prior.transpose(1, 2), speaker_embedding)
            attn_hard = self.binarize_attention_parallel(attn_soft, src_len, mel_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            attn_out = attn_soft, attn_hard, attn_hard_dur, attn_logprob
        if attn_prior is not None:
            if step < self.binarization_start_steps:
                A_soft = attn_soft.squeeze(1)
                x = torch.bmm(A_soft, x)
            else:
                x, mel_len = self.length_regulator(x, attn_hard_dur, max_mel_len)
            duration_rounded = attn_hard_dur
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction) - 1) * d_control, min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)
            mel_mask = get_mask_from_lengths(mel_len)
        pitch_prediction = energy_prediction = None
        x_temp = x.clone()
        if self.use_pitch_embed:
            if pitch_target is not None:
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target['f0'], pitch_target['uv'], p_control)
            else:
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, None, None, p_control)
            x_temp = x_temp + pitch_embedding
        if self.use_energy_embed:
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, e_control)
            x_temp = x_temp + energy_embedding
        x = x_temp.clone()
        return x, log_duration_prediction, duration_rounded, mel_len, mel_mask, pitch_prediction, energy_prediction, attn_out


class E2ETTS(nn.Module):
    """ End-to-End TTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(E2ETTS, self).__init__()
        self.model_config = model_config
        self.hop_length = preprocess_config['preprocessing']['stft']['hop_length']
        self.segment_length_up = preprocess_config['preprocessing']['audio']['segment_length']
        self.segment_length = self.segment_length_up // self.hop_length
        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.proj = nn.Linear(model_config['transformer']['decoder_hidden'], model_config['transformer']['decoder_hidden'])
        self.upsampler = Upsampler(preprocess_config, model_config, train_config)
        self.speaker_emb = None
        if model_config['multi_speaker']:
            self.embedder_type = preprocess_config['preprocessing']['speaker_embedder']
            if self.embedder_type == 'none':
                with open(os.path.join(preprocess_config['path']['preprocessed_path'], 'speakers.json'), 'r') as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(n_speaker, model_config['transformer']['encoder_hidden'])
            else:
                self.speaker_emb = nn.Linear(model_config['external_speaker_dim'], model_config['transformer']['encoder_hidden'])

    def forward(self, speakers, texts, src_lens, max_src_len, audios=None, audio_lens=None, max_audio_len=None, mels=None, mel_lens=None, max_mel_len=None, p_targets=None, e_targets=None, seq_starts=None, attn_priors=None, spker_embeds=None, step=1, p_control=1.0, e_control=1.0, d_control=1.0, cut=True):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        audio_masks = get_mask_from_lengths(audio_lens, max_audio_len) if audio_lens is not None else None
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        texts, text_embeds = self.encoder(texts, src_masks)
        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == 'none':
                speaker_embeds = self.speaker_emb(speakers)
            else:
                assert spker_embeds is not None, 'Speaker embedding should not be None'
                speaker_embeds = self.speaker_emb(spker_embeds)
        enc_outs, log_d_predictions, d_rounded, mel_lens, mel_masks, p_predictions, e_predictions, attn_outs = self.variance_adaptor(texts, text_embeds, src_lens, max_src_len, src_masks, mels, mel_lens, max_mel_len, mel_masks, p_targets, e_targets, seq_starts, attn_priors, speaker_embeds, step, p_control, e_control, d_control)
        dec_outs, mel_masks = self.decoder(enc_outs, mel_masks)
        if cut:
            dec_out_cuts = torch.zeros(dec_outs.shape[0], self.segment_length, dec_outs.shape[2], dtype=dec_outs.dtype, device=dec_outs.device)
            dec_out_cut_lengths = []
            for i, (dec_out_, seq_start_) in enumerate(zip(dec_outs, seq_starts)):
                dec_out_cut_length_ = self.segment_length + (mel_lens[i] - self.segment_length).clamp(None, 0)
                dec_out_cut_lengths.append(dec_out_cut_length_)
                cut_lower, cut_upper = seq_start_, seq_start_ + dec_out_cut_length_
                dec_out_cuts[i, :dec_out_cut_length_] = dec_out_[cut_lower:cut_upper, :]
            dec_out_cuts = self.proj(dec_out_cuts)
            dec_out_cut_lengths = torch.LongTensor(dec_out_cut_lengths)
            dec_out_cut_masks = get_mask_from_lengths(dec_out_cut_lengths, self.segment_length)
        else:
            dec_out_cuts = self.proj(dec_outs)
            dec_out_cut_lengths = mel_lens
            dec_out_cut_masks = mel_masks
        output = self.upsampler(dec_out_cuts.transpose(1, 2))
        output_masks = get_mask_from_lengths(dec_out_cut_lengths * self.hop_length, output.shape[-1])
        output = output.masked_fill(output_masks.unsqueeze(1), 0)
        return output, dec_out_cuts, p_predictions, e_predictions, log_d_predictions, d_rounded, src_masks, mel_masks, dec_out_cut_masks, src_lens, mel_lens, dec_out_cut_lengths, attn_outs


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorP, self).__init__()
        self.lrelu_slope = lrelu_slope
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, lrelu_slope=0.1):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2, lrelu_slope=lrelu_slope), DiscriminatorP(3, lrelu_slope=lrelu_slope), DiscriminatorP(5, lrelu_slope=lrelu_slope), DiscriminatorP(7, lrelu_slope=lrelu_slope), DiscriminatorP(11, lrelu_slope=lrelu_slope)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self, lrelu_slope=0.1):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True, lrelu_slope=lrelu_slope), DiscriminatorS(lrelu_slope=lrelu_slope), DiscriminatorS(lrelu_slope=lrelu_slope)])
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


class SwishBlock(nn.Module):
    """ Swish Block """

    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(LinearNorm(in_channels, hidden_dim, bias=True), nn.SiLU(), LinearNorm(hidden_dim, out_channels, bias=True), nn.SiLU())

    def forward(self, S, E, V):
        out = torch.cat([S.unsqueeze(-1), E.unsqueeze(-1), V.unsqueeze(1).expand(-1, E.size(1), -1, -1)], dim=-1)
        out = self.layer(out)
        return out


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Sequential(ConvNorm(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(out_channels), activation)
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = F.dropout(self.conv_layer(x), self.dropout, self.training)
        x = self.layer_norm(x.contiguous().transpose(1, 2))
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x


class BinLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()


class ForwardSumLoss(nn.Module):

    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)
        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[:query_lens[bid], :, :key_lens[bid] + 1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(curr_logprob, target_seq, input_lengths=query_lens[bid:bid + 1], target_lengths=key_lens[bid:bid + 1])
            total_loss += loss
        total_loss /= attn_logprob.shape[0]
        return total_loss


def sil_phonemes_ids():
    return [_symbol_to_id[sil] for sil in _silences]


class E2ETTSLoss(nn.Module):
    """ E2ETTS Loss """

    def __init__(self, preprocess_config, model_config, train_config, device):
        super(E2ETTSLoss, self).__init__()
        self.device = device
        self.loss_config = train_config['loss']
        self.fft_sizes = train_config['loss']['fft_sizes']
        self.var_start_steps = train_config['step']['var_start_steps']
        self.pitch_config = preprocess_config['preprocessing']['pitch']
        self.use_pitch_embed = model_config['variance_embedding']['use_pitch_embed']
        self.use_energy_embed = model_config['variance_embedding']['use_energy_embed']
        self.binarization_loss_enable_steps = train_config['duration']['binarization_loss_enable_steps']
        self.binarization_loss_warmup_steps = train_config['duration']['binarization_loss_warmup_steps']
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.sil_ph_ids = sil_phonemes_ids()

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        return loss, r_losses, g_losses

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l
        return loss, gen_losses

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens, losses):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        dur_gt.requires_grad = False
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()
        if self.loss_config['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config['dur_loss'] == 'mog':
            return NotImplementedError
        elif self.loss_config['dur_loss'] == 'crf':
            return NotImplementedError
        losses['pdur'] = losses['pdur'] * self.loss_config['lambda_ph_dur']
        if self.loss_config['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            if not torch.isnan(wdur_loss).all():
                losses['wdur'] = wdur_loss * self.loss_config['lambda_word_dur']
        if self.loss_config['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * self.loss_config['lambda_sent_dur']
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets, losses):
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        f0 = pitch_targets['f0']
        uv = pitch_targets['uv']
        nonpadding = self.mel_masks.float()
        self.add_f0_loss(pitch_predictions['pitch_pred'], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config['use_uv']:
            assert p_pred[..., 1].shape == uv.shape
            losses['uv'] = (F.binary_cross_entropy_with_logits(p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * self.loss_config['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        if self.loss_config['pitch_loss'] in ['l1', 'l2']:
            pitch_loss_fn = F.l1_loss if self.loss_config['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() / nonpadding.sum() * self.loss_config['lambda_f0']
        elif self.loss_config['pitch_loss'] == 'ssim':
            return NotImplementedError

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(self.mel_masks)
        energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss

    def get_init_losses(self, device):
        duration_loss = {'pdur': torch.zeros(1), 'wdur': torch.zeros(1), 'sdur': torch.zeros(1)}
        pitch_loss = {}
        if self.pitch_config['use_uv']:
            pitch_loss['uv'] = torch.zeros(1)
        if self.loss_config['pitch_loss'] in ['l1', 'l2']:
            pitch_loss['f0'] = torch.zeros(1)
        energy_loss = torch.zeros(1)
        return duration_loss, pitch_loss, energy_loss

    def variance_loss(self, inputs, predictions, step):
        texts, _, _, _, _, _, _, _, _, pitch_data, energies, _, attn_priors, spker_embeds = inputs[3:]
        p_predictions, e_predictions, log_d_predictions, _, src_masks, mel_masks, _, src_lens, mel_lens, _, attn_outs = predictions[2:]
        self.src_masks = ~src_masks
        self.mel_masks = ~mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
        duration_loss, pitch_loss, energy_loss = self.get_init_losses(self.device)
        if step >= self.var_start_steps:
            duration_loss = self.get_duration_loss(log_d_predictions, attn_hard_dur, texts, duration_loss)
            if self.use_pitch_embed:
                pitch_loss = self.get_pitch_loss(p_predictions, pitch_data, pitch_loss)
            if self.use_energy_embed:
                energy_loss = self.get_energy_loss(e_predictions, energies)
        ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.0
        else:
            bin_loss_weight = min((step - self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
        total_loss = sum(duration_loss.values()) + sum(pitch_loss.values()) + energy_loss + ctc_loss + bin_loss
        if torch.isnan(total_loss).all():
            ipdb.set_trace()
        return total_loss, pitch_loss, energy_loss, duration_loss, ctc_loss, bin_loss


class PreDefinedEmbedder(nn.Module):
    """ Speaker Embedder Wrapper """

    def __init__(self, config):
        super(PreDefinedEmbedder, self).__init__()
        self.sampling_rate = config['preprocessing']['audio']['sampling_rate']
        self.win_length = config['preprocessing']['stft']['win_length']
        self.embedder_type = config['preprocessing']['speaker_embedder']
        self.embedder_cuda = config['preprocessing']['speaker_embedder_cuda']
        self.embedder = self._get_speaker_embedder()

    def _get_speaker_embedder(self):
        embedder = None
        if self.embedder_type == 'DeepSpeaker':
            embedder = embedding.build_model('./deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5')
        else:
            raise NotImplementedError
        return embedder

    def forward(self, audio):
        if self.embedder_type == 'DeepSpeaker':
            spker_embed = embedding.predict_embedding(self.embedder, audio, self.sampling_rate, self.win_length, self.embedder_cuda)
        return spker_embed


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlignmentEncoder,
     lambda: ([], {'n_mel_channels': 4, 'n_att_channels': 4, 'n_text_channels': 4, 'temperature': 4, 'multi_speaker': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm1dTBC,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BinLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (CustomSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {}),
     False),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (EnergyPredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearNorm,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {}),
     False),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PitchPredictor,
     lambda: ([], {'idim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ResBlock1,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResBlock2,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SinusoidalPositionalEmbedding,
     lambda: ([], {'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerFFNLayer,
     lambda: ([], {'hidden_size': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_keonlee9420_Comprehensive_E2E_TTS(_paritybench_base):
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

