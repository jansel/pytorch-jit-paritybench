import sys
_module = sys.modules[__name__]
del sys
base_binarizer = _module
base_preprocess = _module
align_and_binarize = _module
binarize = _module
preprocess = _module
train_mfa_align = _module
txt_processors = _module
base_text_processor = _module
en = _module
zh = _module
wav_processors = _module
base_processor = _module
common_processors = _module
base_tts_infer = _module
ds = _module
fs = _module
fs2_orig = _module
infer = _module
ps_flow = _module
synta = _module
adapt = _module
mfa = _module
run_mfa_align = _module
conformer = _module
espnet_positional_embedding = _module
espnet_transformer_attn = _module
layers = _module
conv = _module
layers = _module
nar_tts_modules = _module
glow_modules = _module
res_flow = _module
utils = _module
rel_transformer = _module
rnn = _module
transformer = _module
wavenet = _module
align_ops = _module
net = _module
shallow_diffusion_tts = _module
fs = _module
fs2_orig = _module
utils = _module
fvae = _module
portaspeech = _module
portaspeech_flow = _module
multi_window_disc = _module
syntactic_graph_buider = _module
syntactic_graph_encoder = _module
syntaspeech = _module
hifigan = _module
mel_utils = _module
stft_loss = _module
parallel_wavegan = _module
causal_conv = _module
pqmf = _module
residual_block = _module
residual_stack = _module
tf_layers = _module
upsample = _module
losses = _module
stft_loss = _module
models = _module
freq_discriminator = _module
melgan = _module
parallel_wavegan = _module
source = _module
optimizers = _module
radam = _module
stft_loss = _module
run = _module
dataset_utils = _module
diffspeech = _module
fs = _module
fs2_orig = _module
ps = _module
ps_flow = _module
speech_base = _module
synta = _module
tts_utils = _module
vocoder_infer = _module
base_vocoder = _module
hifigan = _module
pwg = _module
dataset_utils = _module
hifigan = _module
vocoder_base = _module
audio = _module
align = _module
cwt = _module
griffin_lim = _module
io = _module
utils = _module
pitch_extractors = _module
rnnoise = _module
vad = _module
base_task = _module
ckpt_utils = _module
dataset_utils = _module
ddp_utils = _module
hparams = _module
indexed_datasets = _module
meters = _module
multiprocess_utils = _module
single_thread_env = _module
tensor_utils = _module
trainer = _module
diagonal_metrics = _module
dtw = _module
laplace_var = _module
pitch_distance = _module
ssim = _module
model_utils = _module
schedulers = _module
seq_utils = _module
os_utils = _module
plot = _module
encoding = _module
text_encoder = _module
text_norm = _module

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


from torch import nn


import math


import numpy


import torch.nn as nn


import torch.nn.functional as F


import scipy


from torch.nn import functional as F


import numpy as np


from torch.nn import Parameter


from torch.nn import Linear


from math import sqrt


import random


from functools import partial


from inspect import isfunction


from copy import deepcopy


import torch.distributions as dist


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


import torch.utils.data


from scipy.io.wavfile import read


from scipy.signal import kaiser


import logging


import torch.nn.functional as torch_nn_func


from torch.optim import *


from torch.optim.optimizer import Optimizer


import torch.optim


import torch.distributions


import pandas as pd


import torch.distributed as dist


import matplotlib.pyplot as plt


from torch.utils.data import DistributedSampler


import re


from scipy.interpolate import interp1d


from torch.utils.tensorboard import SummaryWriter


import types


from functools import wraps


from itertools import chain


from torch.utils.data import ConcatDataset


from torch.nn.parallel import DistributedDataParallel


from torch.nn.parallel.distributed import _find_tensors


import time


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


import copy


import torch.multiprocessing as mp


from torch.autograd import Variable


from math import exp


from collections import defaultdict


import matplotlib


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias)
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


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


class EncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, size, self_attn, feed_forward, feed_forward_macaron, conv_module, dropout_rate, normalize_before=True, concat_after=False):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)
            self.norm_final = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.
        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask


class MultiLayeredConv1d(torch.nn.Module):
    """Multi-layered conv1d for Transformer block.
    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.
        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).
        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, :x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class ConformerLayers(nn.Module):

    def __init__(self, hidden_size, num_layers, kernel_size=9, dropout=0.0, num_heads=4, use_last_norm=True, save_hidden=False):
        super().__init__()
        self.use_last_norm = use_last_norm
        self.layers = nn.ModuleList()
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = hidden_size, hidden_size * 4, 1, dropout
        self.pos_embed = RelPositionalEncoding(hidden_size, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_size, RelPositionMultiHeadedAttention(num_heads, hidden_size, 0.0), positionwise_layer(*positionwise_layer_args), positionwise_layer(*positionwise_layer_args), ConvolutionModule(hidden_size, kernel_size, Swish()), dropout) for _ in range(num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        self.save_hidden = save_hidden
        if save_hidden:
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        """

        :param x: [B, T, H]
        :param padding_mask: [B, T]
        :return: [B, T, H]
        """
        self.hiddens = []
        nonpadding_mask = x.abs().sum(-1) > 0
        x = self.pos_embed(x)
        for l in self.encoder_layers:
            x, mask = l(x, nonpadding_mask[:, None, :])
            if self.save_hidden:
                self.hiddens.append(x[0])
        x = x[0]
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class ConformerEncoder(ConformerLayers):

    def __init__(self, hidden_size, dict_size, num_layers=None):
        conformer_enc_kernel_size = 9
        super().__init__(hidden_size, num_layers, conformer_enc_kernel_size)
        self.embed = Embedding(dict_size, hidden_size, padding_idx=0)

    def forward(self, x):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)
        x = super(ConformerEncoder, self).forward(x)
        return x


class ConformerDecoder(ConformerLayers):

    def __init__(self, hidden_size, num_layers):
        conformer_dec_kernel_size = 9
        super().__init__(hidden_size, num_layers, conformer_dec_kernel_size)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self, kernel_size=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, dropout=0.0, dilation=1, bias=True, use_causal_conv=False):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, s


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, hidden_size, out_dims, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True, is_BTC=True, num_layers=None, post_net_kernel=3):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        if num_layers is not None:
            dilations = [1] * num_layers
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_size, kernel_size, d, n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple, dropout=dropout, ln_eps=ln_eps) for d in dilations])
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(hidden_size)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(hidden_size, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, hidden_size)
        elif norm_type == 'ln':
            norm = LayerNorm(hidden_size, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(hidden_size, out_dims, kernel_size=post_net_kernel, padding=post_net_kernel // 2)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x, nonpadding=None):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class TextConvEncoder(ConvBlocks):

    def __init__(self, dict_size, hidden_size, out_dims, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True, num_layers=None, post_net_kernel=3):
        super().__init__(hidden_size, out_dims, dilations, kernel_size, norm_type, layers_in_block, c_multiple, dropout, ln_eps, init_weights, num_layers=num_layers, post_net_kernel=post_net_kernel)
        self.embed_tokens = Embedding(dict_size, hidden_size, 0)
        self.embed_scale = math.sqrt(hidden_size)

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        return super().forward(x)


class ConditionalConvBlocks(ConvBlocks):

    def __init__(self, hidden_size, c_cond, c_out, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True, is_BTC=True, num_layers=None):
        super().__init__(hidden_size, c_out, dilations, kernel_size, norm_type, layers_in_block, c_multiple, dropout, ln_eps, init_weights, is_BTC=False, num_layers=num_layers)
        self.g_prenet = nn.Conv1d(c_cond, hidden_size, 3, padding=1)
        self.is_BTC_ = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, cond, nonpadding=None):
        if self.is_BTC_:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2)
            if nonpadding is not None:
                nonpadding = nonpadding.transpose(1, 2)
        if nonpadding is None:
            nonpadding = x.abs().sum(1)[:, None]
        x = x + self.g_prenet(cond)
        x = x * nonpadding
        x = super(ConditionalConvBlocks, self).forward(x)
        if self.is_BTC_:
            x = x.transpose(1, 2)
        return x


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Permute(nn.Module):

    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


class DurationPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=kernel_size // 2), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, x, x_padding=None):
        x = x.transpose(1, -1)
        for f in self.conv:
            x = f(x)
            if x_padding is not None:
                x = x * (1 - x_padding.float())[:, None, :]
        x = self.linear(x.transpose(1, -1))
        x = x * (1 - x_padding.float())[:, :, None]
        x = x[..., 0]
        return x


class SyntaDurationPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(SyntaDurationPredictor, self).__init__()
        self.graph_encoder = GraphAuxEnc(in_dim=idim, hid_dim=idim, out_dim=idim)
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=kernel_size // 2), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, x, x_padding=None, ph2word=None, graph_lst=None, etypes_lst=None):
        x = x.transpose(1, -1)
        assert ph2word is not None and graph_lst is not None and etypes_lst is not None
        x_graph = self.graph_encoder(graph_lst, x, ph2word, etypes_lst)
        x = x + x_graph * 1.0
        for f in self.conv:
            x = f(x)
            if x_padding is not None:
                x = x * (1 - x_padding.float())[:, None, :]
        x = self.linear(x.transpose(1, -1))
        x = x * (1 - x_padding.float())[:, :, None]
        x = x[..., 0]
        return x


class LengthRegulator(torch.nn.Module):

    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        assert alpha > 0
        """
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None]
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2token = (token_idx * token_mask.long()).sum(1)
        return mel2token


class PitchPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1):
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, padding=kernel_size // 2), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = x.transpose(1, -1)
        for f in self.conv:
            x = f(x)
        x = self.linear(x.transpose(1, -1))
        return x


class EnergyPredictor(PitchPredictor):
    pass


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
            logdet = torch.sum(-self.logs) * x_len
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

    def __init__(self, channels, n_split=4, no_jacobian=False, lu=True, n_sqz=2, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.no_jacobian = no_jacobian
        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.lu = lu
        if lu:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=float), -1)
            eye = np.eye(*w_init.shape, dtype=float)
            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))
        else:
            self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        x = x.view(b, self.n_sqz, c // self.n_split, self.n_split // self.n_sqz, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)
        if self.lu:
            self.weight, log_s = self._get_weight()
            logdet = log_s.sum()
            logdet = logdet * (c / self.n_split) * x_len
        else:
            logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len
        if reverse:
            if hasattr(self, 'weight_inv'):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float())
            logdet = -logdet
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)
        z = z.view(b, self.n_sqz, self.n_split // self.n_sqz, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def _get_weight(self):
        l, log_s, u = self.l, self.log_s, self.u
        l = l * self.l_mask + self.eye
        u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
        weight = torch.matmul(self.p, torch.matmul(l, u))
        return weight, log_s

    def store_inverse(self):
        weight, _ = self._get_weight()
        self.weight_inv = torch.inverse(weight.float())


class InvConv(nn.Module):

    def __init__(self, channels, no_jacobian=False, lu=True, **kwargs):
        super().__init__()
        w_shape = [channels, channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(float)
        LU_decomposed = lu
        if not LU_decomposed:
            self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=float), -1)
            eye = np.eye(*w_shape, dtype=float)
            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.weight = None

    def get_weight(self, device, reverse):
        w_shape = self.w_shape
        self.p = self.p
        self.sign_s = self.sign_s
        self.l_mask = self.l_mask
        self.eye = self.eye
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = self.log_s.sum()
        if not reverse:
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            l = torch.inverse(l.double()).float()
            u = torch.inverse(u.double()).float()
            w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        """
        log-det = log|abs(|W|)| * pixels
        """
        b, c, t = x.size()
        if x_mask is None:
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        logdet = 0
        if not reverse:
            weight, dlogdet = self.get_weight(x.device, reverse)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet * x_len
            return z, logdet
        else:
            if self.weight is None:
                weight, dlogdet = self.get_weight(x.device, reverse)
            else:
                weight, dlogdet = self.weight, self.dlogdet
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet * x_len
            return z, logdet

    def store_inverse(self):
        self.weight, self.dlogdet = self.get_weight('cuda', reverse=True)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):

    def __init__(self, hidden_size, kernel_size, dilation_rate, n_layers, c_cond=0, p_dropout=0, share_cond_layers=False, is_BTC=False):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_size % 2 == 0
        self.is_BTC = is_BTC
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = c_cond
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if c_cond != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(c_cond, 2 * hidden_size * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_size
            else:
                res_skip_channels = hidden_size
            res_skip_layer = torch.nn.Conv1d(hidden_size, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, nonpadding=None, cond=None):
        if self.is_BTC:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if cond is not None else None
            nonpadding = nonpadding.transpose(1, 2) if nonpadding is not None else None
        if nonpadding is None:
            nonpadding = 1
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_size])
        if cond is not None and not self.share_cond_layers:
            cond = self.cond_layer(cond)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if cond is not None:
                cond_offset = i * 2 * self.hidden_size
                cond_l = cond[:, cond_offset:cond_offset + 2 * self.hidden_size, :]
            else:
                cond_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_size, :]) * nonpadding
                output = output + res_skip_acts[:, self.hidden_size:, :]
            else:
                output = output + res_skip_acts
        output = output * nonpadding
        if self.is_BTC:
            output = output.transpose(1, 2)
        return output

    def remove_weight_norm(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


class CouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False, wn=None):
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
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)
        if wn is not None:
            self.wn.in_layers = wn.in_layers
            self.wn.res_skip_layers = wn.res_skip_layers

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
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
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class Glow(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_blocks, n_layers, p_dropout=0.0, n_split=4, n_sqz=2, sigmoid_scale=False, gin_channels=0, inv_conv_type='near', share_cond_layers=False, share_wn_layers=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels
        self.share_cond_layers = share_cond_layers
        if gin_channels != 0 and share_cond_layers:
            cond_layer = torch.nn.Conv1d(gin_channels * n_sqz, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        wn = None
        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            if inv_conv_type == 'near':
                self.flows.append(InvConvNear(channels=in_channels * n_sqz, n_split=n_split, n_sqz=n_sqz))
            if inv_conv_type == 'invconv':
                self.flows.append(InvConv(channels=in_channels * n_sqz))
            if share_wn_layers > 0:
                if b % share_wn_layers == 0:
                    wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels * n_sqz, p_dropout, share_cond_layers)
            self.flows.append(CouplingBlock(in_channels * n_sqz, hidden_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, n_layers=n_layers, gin_channels=gin_channels * n_sqz, p_dropout=p_dropout, sigmoid_scale=sigmoid_scale, wn=wn))

    def forward(self, x, x_mask=None, g=None, reverse=False, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)
        if return_hiddens:
            hs = []
        if self.n_sqz > 1:
            x, x_mask_ = utils.squeeze(x, x_mask, self.n_sqz)
            if g is not None:
                g, _ = utils.squeeze(g, x_mask, self.n_sqz)
            x_mask = x_mask_
        if self.share_cond_layers and g is not None:
            g = self.cond_layer(g)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=reverse)
            if return_hiddens:
                hs.append(x)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, x_mask = utils.unsqueeze(x, x_mask, self.n_sqz)
        if return_hiddens:
            return x, logdet_tot, hs
        return x, logdet_tot

    def store_inverse(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)
        for f in self.flows:
            f.store_inverse()


class FlipLayer(nn.Module):

    def forward(self, x, nonpadding, cond=None, reverse=False):
        x = torch.flip(x, [1])
        return x


class CouplingLayer(nn.Module):

    def __init__(self, c_in, hidden_size, kernel_size, n_layers, p_dropout=0, c_in_g=0, nn_type='wn'):
        super().__init__()
        self.channels = c_in
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.c_half = c_in // 2
        self.pre = nn.Conv1d(self.c_half, hidden_size, 1)
        if nn_type == 'wn':
            self.enc = WN(hidden_size, kernel_size, 1, n_layers, p_dropout=p_dropout, c_cond=c_in_g)
        elif nn_type == 'conv':
            self.enc = ConditionalConvBlocks(hidden_size, c_in_g, hidden_size, None, kernel_size, layers_in_block=1, is_BTC=False, num_layers=n_layers)
        self.post = nn.Conv1d(hidden_size, self.c_half, 1)

    def forward(self, x, nonpadding, cond=None, reverse=False):
        x0, x1 = x[:, :self.c_half], x[:, self.c_half:]
        x_ = self.pre(x0) * nonpadding
        x_ = self.enc(x_, nonpadding=nonpadding, cond=cond)
        m = self.post(x_)
        x1 = m + x1 if not reverse else x1 - m
        x = torch.cat([x0, x1], 1)
        return x * nonpadding


class ResFlow(nn.Module):

    def __init__(self, c_in, hidden_size, kernel_size, n_flow_layers, n_flow_steps=4, c_cond=0, nn_type='wn'):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flow_steps):
            self.flows.append(CouplingLayer(c_in, hidden_size, kernel_size, n_flow_layers, c_in_g=c_cond, nn_type=nn_type))
            self.flows.append(FlipLayer())

    def forward(self, x, nonpadding, cond=None, reverse=False):
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(x, nonpadding, cond=cond, reverse=reverse)
        return x


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
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, 1)
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


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


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
            padded_relative_embeddings = F.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
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
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
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

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=None, block_length=None, pre_ln=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.pre_ln = pre_ln
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
        if pre_ln:
            self.last_ln = LayerNorm(hidden_channels)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            x_ = x
            if self.pre_ln:
                x = self.norm_layers_1[i](x)
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_1[i](x)
            x_ = x
            if self.pre_ln:
                x = self.norm_layers_2[i](x)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_2[i](x)
        if self.pre_ln:
            x = self.last_ln(x)
        x = x * x_mask
        return x


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


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


class RelTransformerEncoder(nn.Module):

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout=0.0, window_size=4, block_length=None, prenet=True, pre_ln=True):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)
        if prenet:
            self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, window_size=window_size, block_length=block_length, pre_ln=pre_ln)

    def forward(self, x, x_mask=None):
        if self.n_vocab > 0:
            x_lengths = (x > 0).long().sum(-1)
            x = self.emb(x) * math.sqrt(self.hidden_channels)
        else:
            x_lengths = (x.abs().sum(-1) > 0).long().sum(-1)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1)
        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)
        return x.transpose(1, 2)


class PreNet(nn.Module):

    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class HighwayNetwork(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.0)

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1.0 - g) * x
        return y


class BatchNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class CBHG(nn.Module):

    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()
        self._to_flatten = []
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)
        if proj_channels[-1] != channels:
            self.highway_mismatch = True
            self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        else:
            self.highway_mismatch = False
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)
        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self._to_flatten.append(self.rnn)
        self._flatten_parameters()

    def forward(self, x):
        self._flatten_parameters()
        residual = x
        seq_len = x.size(-1)
        conv_bank = []
        for conv in self.conv1d_bank:
            c = conv(x)
            conv_bank.append(c[:, :, :seq_len])
        conv_bank = torch.cat(conv_bank, dim=1)
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = x + residual
        x = x.transpose(1, 2)
        if self.highway_mismatch is True:
            x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)
        x, _ = self.rnn(x)
        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]


class TacotronEncoder(nn.Module):

    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims, embed_dims, embed_dims, dropout=dropout)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels, proj_channels=[cbhg_channels, cbhg_channels], num_highways=num_highways)
        self.proj_out = nn.Linear(cbhg_channels * 2, cbhg_channels)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        x = self.proj_out(x)
        return x


class RNNEncoder(nn.Module):

    def __init__(self, num_chars, embedding_dim, n_convolutions=3, kernel_size=5):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim, padding_idx=0)
        convolutions = []
        for _ in range(n_convolutions):
            conv_layer = nn.Sequential(ConvNorm(embedding_dim, embedding_dim, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), dilation=1, w_init_gain='relu'), nn.BatchNorm1d(embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(embedding_dim, int(embedding_dim / 2), 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        input_lengths = (x > 0).sum(-1)
        input_lengths = input_lengths.cpu().numpy()
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training) + x
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs


class DecoderRNN(torch.nn.Module):

    def __init__(self, hidden_size, decoder_rnn_dim, dropout):
        super(DecoderRNN, self).__init__()
        self.in_conv1d = nn.Sequential(torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, padding=4), torch.nn.ReLU(), torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, padding=4))
        self.ln = nn.LayerNorm(hidden_size)
        if decoder_rnn_dim == 0:
            decoder_rnn_dim = hidden_size * 2
        self.rnn = torch.nn.LSTM(input_size=hidden_size, hidden_size=decoder_rnn_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.rnn.flatten_parameters()
        self.conv1d = torch.nn.Conv1d(in_channels=decoder_rnn_dim * 2, out_channels=hidden_size, kernel_size=3, padding=1)

    def forward(self, x):
        input_masks = x.abs().sum(-1).ne(0).data[:, :, None]
        input_lengths = input_masks.sum([-1, -2])
        input_lengths = input_lengths.cpu().numpy()
        x = self.in_conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.ln(x)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x * input_masks
        pre_mel = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        pre_mel = pre_mel * input_masks
        return pre_mel


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


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda : 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


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

    def forward(self, x, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                prev_input = saved_state['prev_input']
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size:]
            saved_state['prev_input'] = x
            self._set_input_buffer(incremental_state, saved_state)
        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size ** -0.5
        if incremental_state is not None:
            x = x[-1:]
        if self.act == 'gelu':
            x = F.gelu(x)
        if self.act == 'relu':
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'f') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, 'f', buffer)

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_input' in saved_state:
                del saved_state['prev_input']
            self._set_input_buffer(incremental_state, saved_state)


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


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
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
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
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
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
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            if 'prev_key_padding_mask' in saved_state and saved_state['prev_key_padding_mask'] is not None:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
                if static_kv:
                    key_padding_mask = prev_key_padding_mask
                else:
                    key_padding_mask = torch.cat((prev_key_padding_mask, key_padding_mask), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            self._set_input_buffer(incremental_state, saved_state)
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
            attn_weights = attn_weights.masked_fill(enc_dec_attn_constraint_mask.unsqueeze(2).bool(), -100000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -100000000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_logits = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v
        attn_weights_float = softmax(attn_weights, dim=-1)
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

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, 'attn_state') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

    def clear_buffer(self, incremental_state=None):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                del saved_state['prev_key']
            if 'prev_value' in saved_state:
                del saved_state['prev_value']
            self._set_input_buffer(incremental_state, saved_state)


class EncSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, padding='SAME', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            self.layer_norm1 = LayerNorm(c)
            self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = LayerNorm(c)
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


class DecSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c)
        self.self_attn = MultiheadAttention(c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = LayerNorm(c)
        self.encoder_attn = MultiheadAttention(c, num_heads, encoder_decoder_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm3 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout, act=act)

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, incremental_state=None, self_attn_mask=None, self_attn_padding_mask=None, attn_out=None, reset_attn_weight=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, incremental_state=incremental_state, attn_mask=self_attn_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        attn_logits = None
        if encoder_out is not None or attn_out is not None:
            residual = x
            x = self.layer_norm2(x)
        if encoder_out is not None:
            x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask, incremental_state=incremental_state, static_kv=True, enc_dec_attn_constraint_mask=get_incremental_state(self, incremental_state, 'enc_dec_attn_constraint_mask'), reset_attn_weight=reset_attn_weight)
            attn_logits = attn[1]
        elif attn_out is not None:
            x = self.encoder_attn.in_proj_v(attn_out)
        if encoder_out is not None or attn_out is not None:
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x, incremental_state=incremental_state)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        return x, attn_logits

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return set_incremental_state(self, incremental_state, name, tensor)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(hidden_size, num_heads, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = DecSALayer(hidden_size, num_heads, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=kernel_size)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)


DEFAULT_MAX_TARGET_POSITIONS = 2000


class FFTBlocks(nn.Module):

    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=0.0, num_heads=2, use_pos_embed=True, use_last_norm=True, use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(self.hidden_size, self.dropout, kernel_size=ffn_kernel_size, num_heads=num_heads) for _ in range(self.num_layers)])
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
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
        return x


class FastSpeechEncoder(FFTBlocks):

    def __init__(self, dict_size, hidden_size=256, num_layers=4, kernel_size=9, num_heads=2, dropout=0.0):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads, use_pos_embed=False, dropout=dropout)
        self.embed_tokens = Embedding(dict_size, hidden_size, 0)
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS)

    def forward(self, txt_tokens, attn_mask=None):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens)
        if self.num_layers > 0:
            x = super(FastSpeechEncoder, self).forward(x, encoder_padding_mask, attn_mask=attn_mask)
        return x

    def forward_embedding(self, txt_tokens):
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        if self.use_pos_embed:
            positions = self.embed_positions(txt_tokens)
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FastSpeechDecoder(FFTBlocks):

    def __init__(self, hidden_size=256, num_layers=4, kernel_size=9, num_heads=2):
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """

        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffNet(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        in_dims = hparams['audio_num_mel_bins']
        self.encoder_hidden = hparams['hidden_size']
        self.residual_layers = hparams['residual_layers']
        self.residual_channels = hparams['residual_channels']
        self.dilation_cycle_length = hparams['dilation_cycle_length']
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.residual_layers = nn.ModuleList([ResidualBlock(self.encoder_hidden, self.residual_channels, 2 ** (i % self.dilation_cycle_length)) for i in range(self.residual_layers)])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x[:, None, :, :]


FS_DECODERS = {'fft': lambda hp: FastSpeechDecoder(hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']), 'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']), 'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'], hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'], norm_type=hp['enc_dec_norm'], dropout=hp['dropout'], post_net_kernel=hp.get('dec_post_net_kernel', 3)), 'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, dilation_rate=1, n_layers=hp['dec_layers'], is_BTC=True)}


FS_ENCODERS = {'fft': lambda hp, dict_size: FastSpeechEncoder(dict_size, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'], num_heads=hp['num_heads']), 'tacotron': lambda hp, dict_size: TacotronEncoder(hp['hidden_size'], dict_size, hp['hidden_size'], K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']), 'tacotron2': lambda hp, dict_size: RNNEncoder(dict_size, hp['hidden_size']), 'conv': lambda hp, dict_size: TextConvEncoder(dict_size, hp['hidden_size'], hp['hidden_size'], hp['enc_dilations'], hp['enc_kernel_size'], layers_in_block=hp['layers_in_block'], norm_type=hp['enc_dec_norm'], post_net_kernel=hp.get('enc_post_net_kernel', 3)), 'rel_fft': lambda hp, dict_size: RelTransformerEncoder(dict_size, hp['hidden_size'], hp['hidden_size'], hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'], hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln'])}


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    max_frames = mel2token.shape[1] // frames_multiple * frames_multiple
    mel2token = mel2token[:, :max_frames]
    return mel2token


def denorm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100, pitch_padding=None, min=50, max=900):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == 'log':
        f0 = 2 ** f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, a_min=min, a_max=max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)
    return h


def f0_to_coarse(f0, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


class FastSpeech(nn.Module):

    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = hparams['audio_num_mel_bins'] if out_dims is None else out_dims
        self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_id_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['dur_predictor_layers'], dropout_rate=hparams['predictor_dropout'], kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=5, dropout_rate=0.1, odim=2, kernel_size=hparams['predictor_kernel'])
        if hparams['dec_inp_add_noise']:
            self.z_channels = hparams['z_channels']
            self.dec_inp_noise_proj = nn.Linear(self.hidden_size + self.z_channels, self.hidden_size)

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None, f0=None, uv=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(encoder_out, mel2ph)
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels]))
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        return style_embed

    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(pitch_pred[:, :, 0], pitch_pred[:, :, 1] > 0 if use_uv else None, pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding


def inverse_cwt(Wavelet_lf0, scales):
    b = (np.arange(0, len(scales))[None, None, :] + 1 + 2.5) ** -2.5
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdims=True)) / lf0_rec_sum.std(-1, keepdims=True)
    return lf0_rec_sum


def inverse_cwt_torch(Wavelet_lf0, scales):
    import torch
    b = (torch.arange(0, len(scales)).float()[None, None, :] + 1 + 2.5) ** -2.5
    lf0_rec = Wavelet_lf0 * b
    lf0_rec_sum = lf0_rec.sum(-1)
    lf0_rec_sum = (lf0_rec_sum - lf0_rec_sum.mean(-1, keepdim=True)) / lf0_rec_sum.std(-1, keepdim=True)
    return lf0_rec_sum


def cwt2f0(cwt_spec, mean, std, cwt_scales):
    assert len(mean.shape) == 1 and len(std.shape) == 1 and len(cwt_spec.shape) == 3
    import torch
    if isinstance(cwt_spec, torch.Tensor):
        f0 = inverse_cwt_torch(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = f0.exp()
    else:
        f0 = inverse_cwt(cwt_spec, cwt_scales)
        f0 = f0 * std[:, None] + mean[:, None]
        f0 = np.exp(f0)
    return f0


dj = 1


dt = 0.005


def get_lf0_cwt(lf0):
    """
    input:
        signal of shape (N)
    output:
        Wavelet_lf0 of shape(10, N), scales of shape(10)
    """
    mother = wavelet.MexicanHat()
    s0 = dt * 2
    J = 9
    Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales


def norm_f0(f0, uv, pitch_norm='log', f0_mean=400, f0_std=100):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == 'standard':
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == 'log':
        f0 = torch.log2(f0 + 1e-08) if is_torch else np.log2(f0 + 1e-08)
    if uv is not None:
        f0[uv > 0] = 0
    return f0


class FastSpeech2Orig(FastSpeech):

    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(300, self.hidden_size, 0)
            self.energy_predictor = EnergyPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=2, kernel_size=hparams['predictor_kernel'])
        if hparams['pitch_type'] == 'cwt' and hparams['use_pitch_embed']:
            self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=11, kernel_size=hparams['predictor_kernel'])
            self.cwt_stats_layers = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 2))

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)
        if self.hparams['use_energy_embed']:
            energy_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret)
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels]))
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'cwt':
            decoder_inp = decoder_inp.detach() + self.hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
            pitch_padding = mel2ph == 0
            ret['cwt'] = cwt_out = self.pitch_predictor(decoder_inp)
            stats_out = self.cwt_stats_layers(decoder_inp.mean(1))
            mean = ret['f0_mean'] = stats_out[:, 0]
            std = ret['f0_std'] = stats_out[:, 1]
            cwt_spec = cwt_out[:, :, :10]
            if f0 is None:
                std = std * self.hparams['cwt_std_scale']
                f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
                if self.hparams['use_uv']:
                    assert cwt_out.shape[-1] == 11
                    uv = cwt_out[:, :, -1] > 0
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv if self.hparams['use_uv'] else None, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)
            pitch_embed = self.pitch_embed(pitch)
            return pitch_embed
        else:
            return super(FastSpeech2Orig, self).forward_pitch(decoder_inp, f0, uv, mel2ph, ret, encoder_out)

    def forward_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + self.hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy_embed_inp = energy_pred if energy is None else energy
        energy_embed_inp = torch.clamp(energy_embed_inp * 256 // 4, min=0, max=255).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        _, cwt_scales = get_lf0_cwt(np.ones(10))
        f0 = cwt2f0(cwt_spec, mean, std, cwt_scales)
        f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None)
        return f0_norm


class AuxModel(FastSpeech2Orig):

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        style_embed = self.forward_style_embed(spk_embed, spk_id)
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)
        if self.hparams['use_energy_embed']:
            energy_inp = (decoder_inp_ + style_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret)
        ret['decoder_inp'] = decoder_inp = (decoder_inp + style_embed) * tgt_nonpadding
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels]))
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        if kwargs['skip_decoder']:
            return ret
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret


DIFF_DECODERS = {'wavenet': lambda hp: DiffNet(hp)}


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


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, max_beta, timesteps)
    return betas


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda : torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda : torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):

    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = hparams
        out_dims = hparams['audio_num_mel_bins']
        denoise_fn = DIFF_DECODERS[hparams['diff_decoder_type']](hparams)
        timesteps = hparams['timesteps']
        K_step = hparams['K_step']
        loss_type = hparams['diff_loss_type']
        spec_min = hparams['spec_min']
        spec_max = hparams['spec_max']
        self.denoise_fn = denoise_fn
        self.fs2 = AuxModel(dict_size, hparams)
        self.mel_bins = out_dims
        if hparams['schedule_type'] == 'linear':
            betas = linear_beta_schedule(timesteps, hparams['max_beta'])
        else:
            betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
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
        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

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

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)
        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1)).mean()
            else:
                loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, spk_id=None, ref_mels=None, f0=None, uv=None, energy=None, infer=False, **kwargs):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = self.fs2(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id, f0=f0, uv=uv, energy=energy, infer=infer, skip_decoder=not infer, **kwargs)
        cond = ret['decoder_inp'].transpose(1, 2)
        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = ref_mels
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]
            ret['diff_loss'] = self.p_losses(x, t, cond)
            ret['mel_out'] = None
        else:
            ret['fs2_mel'] = ret['mel_out']
            fs2_mels = ret['mel_out']
            t = self.K_step
            fs2_mels = self.norm_spec(fs2_mels)
            fs2_mels = fs2_mels.transpose(1, 2)[:, None, :, :]
            x = self.q_sample(x_start=fs2_mels, t=torch.tensor([t - 1], device=device).long())
            if self.hparams.get('gaussian_start') is not None and self.hparams['gaussian_start']:
                None
                shape = cond.shape[0], 1, self.mel_bins, cond.shape[2]
                x = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x


class FVAEEncoder(nn.Module):

    def __init__(self, c_in, hidden_size, c_latent, kernel_size, n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        if np.prod(strides) == 1:
            self.pre_net = nn.Conv1d(c_in, hidden_size, kernel_size=1)
        else:
            self.pre_net = nn.Sequential(*[(nn.Conv1d(c_in, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2) if i == 0 else nn.Conv1d(hidden_size, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2)) for i, s in enumerate(strides)])
        if nn_type == 'wn':
            self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
        elif nn_type == 'conv':
            self.nn = ConditionalConvBlocks(hidden_size, c_cond, hidden_size, None, kernel_size, layers_in_block=2, is_BTC=False, num_layers=n_layers)
        self.out_proj = nn.Conv1d(hidden_size, c_latent * 2, 1)
        self.latent_channels = c_latent

    def forward(self, x, nonpadding, cond):
        x = self.pre_net(x)
        nonpadding = nonpadding[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * nonpadding
        x = self.nn(x, nonpadding=nonpadding, cond=cond) * nonpadding
        x = self.out_proj(x)
        m, logs = torch.split(x, self.latent_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs, nonpadding


class FVAEDecoder(nn.Module):

    def __init__(self, c_latent, hidden_size, out_channels, kernel_size, n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.pre_net = nn.Sequential(*[(nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=s, stride=s) if i == 0 else nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=s, stride=s)) for i, s in enumerate(strides)])
        if nn_type == 'wn':
            self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
        elif nn_type == 'conv':
            self.nn = ConditionalConvBlocks(hidden_size, c_cond, hidden_size, [1] * n_layers, kernel_size, layers_in_block=2, is_BTC=False)
        self.out_proj = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, x, nonpadding, cond):
        x = self.pre_net(x)
        x = x * nonpadding
        x = self.nn(x, nonpadding=nonpadding, cond=cond) * nonpadding
        x = self.out_proj(x)
        return x


class FVAE(nn.Module):

    def __init__(self, c_in_out, hidden_size, c_latent, kernel_size, enc_n_layers, dec_n_layers, c_cond, strides, use_prior_flow, flow_hidden=None, flow_kernel_size=None, flow_n_steps=None, encoder_type='wn', decoder_type='wn'):
        super(FVAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.latent_size = c_latent
        self.use_prior_flow = use_prior_flow
        if np.prod(strides) == 1:
            self.g_pre_net = nn.Conv1d(c_cond, c_cond, kernel_size=1)
        else:
            self.g_pre_net = nn.Sequential(*[nn.Conv1d(c_cond, c_cond, kernel_size=s * 2, stride=s, padding=s // 2) for i, s in enumerate(strides)])
        self.encoder = FVAEEncoder(c_in_out, hidden_size, c_latent, kernel_size, enc_n_layers, c_cond, strides=strides, nn_type=encoder_type)
        if use_prior_flow:
            self.prior_flow = ResFlow(c_latent, flow_hidden, flow_kernel_size, flow_n_steps, 4, c_cond=c_cond)
        self.decoder = FVAEDecoder(c_latent, hidden_size, c_in_out, kernel_size, dec_n_layers, c_cond, strides=strides, nn_type=decoder_type)
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, nonpadding=None, cond=None, infer=False, noise_scale=1.0):
        """

        :param x: [B, C_in_out, T]
        :param nonpadding: [B, 1, T]
        :param cond: [B, C_g, T]
        :return:
        """
        if nonpadding is None:
            nonpadding = 1
        cond_sqz = self.g_pre_net(cond)
        if not infer:
            z_q, m_q, logs_q, nonpadding_sqz = self.encoder(x, nonpadding, cond_sqz)
            q_dist = dist.Normal(m_q, logs_q.exp())
            if self.use_prior_flow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, nonpadding_sqz, cond_sqz)
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = ((logqx - logpx) * nonpadding_sqz).sum() / nonpadding_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl * nonpadding_sqz).sum() / nonpadding_sqz.sum() / z_q.shape[1]
                z_p = None
            return z_q, loss_kl, z_p, m_q, logs_q
        else:
            latent_shape = [cond_sqz.shape[0], self.latent_size, cond_sqz.shape[2]]
            z_p = torch.randn(latent_shape) * noise_scale
            if self.use_prior_flow:
                z_p = self.prior_flow(z_p, 1, cond_sqz, reverse=True)
            return z_p


def group_hidden_by_segs(h, seg_ids, max_len):
    """

    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs, cnt_gby_segs


class GraphAuxEnc(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_iterations=5, n_edge_types=6):
        super(GraphAuxEnc, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.skip_connect = True
        self.dropout_after_gae = False
        self.ggc_1 = GatedGraphConv(in_feats=in_dim, out_feats=hid_dim, n_steps=n_iterations, n_etypes=n_edge_types)
        self.ggc_2 = GatedGraphConv(in_feats=hid_dim, out_feats=out_dim, n_steps=n_iterations, n_etypes=n_edge_types)
        self.dropout = nn.Dropout(p=0.5)

    @staticmethod
    def ph_encoding_to_word_encoding(ph_encoding, ph2word, word_len):
        """
        ph_encoding: [batch, t_p, hid]
        ph2word: tensor [batch, t_w]
        word_len: tensor [batch]
        """
        word_encoding_for_graph, batch_word_encoding, has_word_row_idx = GraphAuxEnc._process_ph_to_word_encoding(ph_encoding, ph2word, word_len)
        return batch_word_encoding, word_encoding_for_graph

    def pad_word_encoding_to_phoneme(self, word_encoding, ph2word, t_p):
        return self._postprocess_word2ph(word_encoding, ph2word, t_p)

    @staticmethod
    def _process_ph_to_word_encoding(ph_encoding, ph2word, word_len=None):
        """
        ph_encoding: [batch, t_p, hid]
        ph2word: tensor [batch, t_w]
        word_len: tensor [batch]
        """
        word_len = word_len.reshape([-1])
        max_len = max(word_len)
        num_nodes = sum(word_len)
        batch_word_encoding = group_hidden_by_segs(ph_encoding, ph2word, max_len)
        bs, t_p, hid = batch_word_encoding.shape
        has_word_mask = sequence_mask(word_len, max_len)
        word_encoding = batch_word_encoding.reshape([bs * t_p, hid])
        has_word_row_idx = has_word_mask.reshape([-1])
        word_encoding = word_encoding[has_word_row_idx]
        assert word_encoding.shape[0] == num_nodes
        return word_encoding, batch_word_encoding, has_word_row_idx

    @staticmethod
    def _postprocess_word2ph(word_encoding, ph2word, t_p):
        word_encoding = F.pad(word_encoding, [0, 0, 1, 0])
        ph2word_ = ph2word[:, :, None].repeat([1, 1, word_encoding.shape[-1]])
        out = torch.gather(word_encoding, 1, ph2word_)
        return out

    @staticmethod
    def _repeat_one_sequence(x, d, T):
        """Repeat each frame according to duration."""
        if d.sum() == 0:
            d = d.fill_(1)
        hid = x.shape[-1]
        expanded_lst = [x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0]
        expanded = torch.cat(expanded_lst, dim=0)
        if T > expanded.shape[0]:
            expanded = torch.cat([expanded, torch.zeros([T - expanded.shape[0], hid])], dim=0)
        return expanded

    def word_forward(self, graph_lst, word_encoding, etypes_lst):
        """
        word encoding in, word encoding out.
        """
        batched_graph = dgl.batch(graph_lst)
        inp = word_encoding
        batched_etypes = torch.cat(etypes_lst)
        assert batched_graph.num_nodes() == inp.shape[0]
        gcc1_out = self.ggc_1(batched_graph, inp, batched_etypes)
        if self.dropout_after_gae:
            gcc1_out = self.dropout(gcc1_out)
        gcc2_out = self.ggc_2(batched_graph, gcc1_out, batched_etypes)
        if self.dropout_after_gae:
            gcc2_out = self.ggc_2(batched_graph, gcc2_out, batched_etypes)
        if self.skip_connect:
            assert self.in_dim == self.hid_dim and self.hid_dim == self.out_dim
            gcc2_out = inp + gcc1_out + gcc2_out
        word_len = torch.tensor([g.num_nodes() for g in graph_lst]).reshape([-1])
        max_len = max(word_len)
        has_word_mask = sequence_mask(word_len, max_len)
        has_word_row_idx = has_word_mask.reshape([-1])
        bs = len(graph_lst)
        t_w = max([g.num_nodes() for g in graph_lst])
        hid = word_encoding.shape[-1]
        output = torch.zeros([bs * t_w, hid])
        output[has_word_row_idx] = gcc2_out
        output = output.reshape([bs, t_w, hid])
        word_level_output = output
        return torch.transpose(word_level_output, 1, 2)

    def forward(self, graph_lst, ph_encoding, ph2word, etypes_lst, return_word_encoding=False):
        """
        graph_lst: [list of dgl_graph]
        ph_encoding: [batch, hid, t_p]
        ph2word: [list of list[1,2,2,2,3,3,3]]
        etypes_lst: [list of etypes]; etypes: torch.LongTensor
        """
        t_p = ph_encoding.shape[-1]
        ph_encoding = ph_encoding.transpose(1, 2)
        word_len = torch.tensor([g.num_nodes() for g in graph_lst]).reshape([-1])
        batched_graph = dgl.batch(graph_lst)
        inp, batched_word_encoding, has_word_row_idx = self._process_ph_to_word_encoding(ph_encoding, ph2word, word_len=word_len)
        bs, t_w, hid = batched_word_encoding.shape
        batched_etypes = torch.cat(etypes_lst)
        gcc1_out = self.ggc_1(batched_graph, inp, batched_etypes)
        gcc2_out = self.ggc_2(batched_graph, gcc1_out, batched_etypes)
        gcc2_out = inp + gcc1_out + gcc2_out
        output = torch.zeros([bs * t_w, hid])
        output[has_word_row_idx] = gcc2_out
        output = output.reshape([bs, t_w, hid])
        word_level_output = output
        output = self._postprocess_word2ph(word_level_output, ph2word, t_p)
        output = torch.transpose(output, 1, 2)
        if return_word_encoding:
            return output, torch.transpose(word_level_output, 1, 2)
        else:
            return output


class SyntaFVAE(nn.Module):

    def __init__(self, c_in_out, hidden_size, c_latent, kernel_size, enc_n_layers, dec_n_layers, c_cond, strides, use_prior_flow, flow_hidden=None, flow_kernel_size=None, flow_n_steps=None, encoder_type='wn', decoder_type='wn'):
        super(SyntaFVAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.latent_size = c_latent
        self.use_prior_flow = use_prior_flow
        if np.prod(strides) == 1:
            self.g_pre_net = nn.Conv1d(c_cond, c_cond, kernel_size=1)
        else:
            self.g_pre_net = nn.Sequential(*[nn.Conv1d(c_cond, c_cond, kernel_size=s * 2, stride=s, padding=s // 2) for i, s in enumerate(strides)])
        self.encoder = FVAEEncoder(c_in_out, hidden_size, c_latent, kernel_size, enc_n_layers, c_cond, strides=strides, nn_type=encoder_type)
        if use_prior_flow:
            self.prior_flow = ResFlow(c_latent, flow_hidden, flow_kernel_size, flow_n_steps, 4, c_cond=c_cond)
        self.decoder = FVAEDecoder(c_latent, hidden_size, c_in_out, kernel_size, dec_n_layers, c_cond, strides=strides, nn_type=decoder_type)
        self.prior_dist = dist.Normal(0, 1)
        self.graph_encoder = GraphAuxEnc(in_dim=hidden_size, hid_dim=hidden_size, out_dim=hidden_size)

    def forward(self, x=None, nonpadding=None, cond=None, infer=False, noise_scale=1.0, mel2word=None, ph2word=None, graph_lst=None, etypes_lst=None):
        """

        :param x: target mel, [B, C_in_out, T] 
        :param nonpadding: [B, 1, T]
        :param cond: phoneme encoding, [B, C_g, T]
        :return:
        """
        word_len = ph2word.max(dim=1)[0]
        ph_encoding_for_graph = cond.detach() + 0.1 * (cond - cond.detach())
        _, ph_out_word_encoding_for_graph = GraphAuxEnc.ph_encoding_to_word_encoding(ph_encoding_for_graph.transpose(1, 2), mel2word, word_len)
        t_m = mel2word.shape[-1]
        g_graph = self.graph_encoder.word_forward(graph_lst=graph_lst, word_encoding=ph_out_word_encoding_for_graph, etypes_lst=etypes_lst)
        g_graph = g_graph.transpose(1, 2)
        g_graph = GraphAuxEnc._postprocess_word2ph(g_graph, mel2word, t_m)
        g_graph = g_graph.transpose(1, 2)
        cond = cond + g_graph * 1.0
        if nonpadding is None:
            nonpadding = 1
        cond_sqz = self.g_pre_net(cond)
        if not infer:
            z_q, m_q, logs_q, nonpadding_sqz = self.encoder(x, nonpadding, cond_sqz)
            q_dist = dist.Normal(m_q, logs_q.exp())
            if self.use_prior_flow:
                logqx = q_dist.log_prob(z_q)
                z_p = self.prior_flow(z_q, nonpadding_sqz, cond_sqz)
                logpx = self.prior_dist.log_prob(z_p)
                loss_kl = ((logqx - logpx) * nonpadding_sqz).sum() / nonpadding_sqz.sum() / logqx.shape[1]
            else:
                loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
                loss_kl = (loss_kl * nonpadding_sqz).sum() / nonpadding_sqz.sum() / z_q.shape[1]
                z_p = None
            return z_q, loss_kl, z_p, m_q, logs_q
        else:
            latent_shape = [cond_sqz.shape[0], self.latent_size, cond_sqz.shape[2]]
            z_p = torch.randn(latent_shape) * noise_scale
            if self.use_prior_flow:
                z_p = self.prior_flow(z_p, 1, cond_sqz, reverse=True)
            return z_p


def build_word_mask(x2word, y2word):
    return (x2word[:, :, None] == y2word[:, None, :]).long()


def mel2ph_to_mel2word(mel2ph, ph2word):
    mel2word = (ph2word - 1).gather(1, (mel2ph - 1).clamp(min=0)) + 1
    mel2word = mel2word * (mel2ph > 0).long()
    return mel2word


class PortaSpeech(FastSpeech):

    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        if hparams['use_word_encoder']:
            self.word_encoder = RelTransformerEncoder(word_dict_size, self.hidden_size, self.hidden_size, self.hidden_size, 2, hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
        if hparams['dur_level'] == 'word':
            if hparams['word_encoder_type'] == 'rel_fft':
                self.ph2word_encoder = RelTransformerEncoder(0, self.hidden_size, self.hidden_size, self.hidden_size, 2, hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
            if hparams['word_encoder_type'] == 'fft':
                self.ph2word_encoder = FFTBlocks(self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
            self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_query_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_res_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
            if hparams['text_encoder_postnet']:
                self.text_encoder_postnet = ConvBlocks(self.hidden_size, self.hidden_size, [1] * 3, 5, layers_in_block=2)
        else:
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        if hparams['use_fvae']:
            del self.decoder
            del self.mel_out
            self.fvae = FVAE(c_in_out=self.out_dims, hidden_size=hparams['fvae_enc_dec_hidden'], c_latent=hparams['latent_size'], kernel_size=hparams['fvae_kernel_size'], enc_n_layers=hparams['fvae_enc_n_layers'], dec_n_layers=hparams['fvae_dec_n_layers'], c_cond=self.hidden_size, use_prior_flow=hparams['use_prior_flow'], flow_hidden=hparams['prior_flow_hidden'], flow_kernel_size=hparams['prior_flow_kernel_size'], flow_n_steps=hparams['prior_flow_n_blocks'], strides=[hparams['fvae_strides']], encoder_type=hparams['fvae_encoder_type'], decoder_type=hparams['fvae_decoder_type'])
        else:
            self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
            self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if self.hparams['add_word_pos']:
            self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None, global_step=None, *args, **kwargs):
        ret = {}
        style_embed = self.forward_style_embed(spk_embed, spk_id)
        x, tgt_nonpadding = self.run_text_encoder(txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret)
        x = x * tgt_nonpadding
        ret['nonpadding'] = tgt_nonpadding
        if self.hparams['use_pitch_embed']:
            x = x + self.pitch_embed(pitch)
        ret['decoder_inp'] = x
        ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels, global_step)
        return ret

    def run_text_encoder(self, txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret):
        word2word = torch.arange(word_len)[None, :] + 1
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding + style_embed
        if self.hparams['use_word_encoder']:
            word_encoder_out = self.word_encoder(word_tokens) + style_embed
            ph_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)
        if self.hparams['dur_level'] == 'word':
            word_encoder_out = 0
            h_ph_gb_word = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)[0]
            word_encoder_out = word_encoder_out + self.ph2word_encoder(h_ph_gb_word)
            if self.hparams['use_word_encoder']:
                word_encoder_out = word_encoder_out + self.word_encoder(word_tokens)
            mel2word = self.forward_dur(ph_encoder_out, mel2word, ret, ph2word=ph2word, word_len=word_len)
            mel2word = clip_mel2token_to_multiple(mel2word, self.hparams['frames_multiple'])
            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
            enc_pos = self.get_pos_embed(word2word, ph2word)
            dec_pos = self.get_pos_embed(word2word, mel2word)
            dec_word_mask = build_word_mask(mel2word, ph2word)
            x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
            if self.hparams['add_word_pos']:
                x = x + self.word_pos_proj(dec_pos)
            ret['attn'] = weight
        else:
            mel2ph = self.forward_dur(ph_encoder_out, mel2ph, ret)
            mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
            mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
            x = expand_states(ph_encoder_out, mel2ph)
            if self.hparams['add_word_pos']:
                dec_pos = self.get_pos_embed(word2word, mel2word)
                x = x + self.word_pos_proj(dec_pos)
            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        if self.hparams['use_word_encoder']:
            x = x + expand_states(word_encoder_out, mel2word)
        return x, tgt_nonpadding

    def attention(self, ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        word_enc_out_expend = expand_states(word_encoder_out, mel2word)
        word_enc_out_expend = torch.cat([word_enc_out_expend, dec_pos], -1)
        if self.hparams['text_encoder_postnet']:
            word_enc_out_expend = self.dec_res_proj(word_enc_out_expend)
            word_enc_out_expend = self.text_encoder_postnet(word_enc_out_expend)
            dec_q = x_res = word_enc_out_expend
        else:
            dec_q = self.dec_query_proj(word_enc_out_expend)
            x_res = self.dec_res_proj(word_enc_out_expend)
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1000000000.0)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0):
        if not self.hparams['use_fvae']:
            x = self.decoder(x)
            x = self.mel_out(x)
            ret['kl'] = 0
            return x * tgt_nonpadding
        else:
            decoder_inp = x
            x = x.transpose(1, 2)
            tgt_nonpadding_BHT = tgt_nonpadding.transpose(1, 2)
            if infer:
                z = self.fvae(cond=x, infer=True)
            else:
                tgt_mels = tgt_mels.transpose(1, 2)
                z, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = self.fvae(tgt_mels, tgt_nonpadding_BHT, cond=x)
                if global_step < self.hparams['posterior_start_steps']:
                    z = torch.randn_like(z)
            x_recon = self.fvae.decoder(z, nonpadding=tgt_nonpadding_BHT, cond=x).transpose(1, 2)
            ret['pre_mel_out'] = x_recon
            return x_recon

    def forward_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        if self.hparams['dur_level'] == 'word':
            word_len = kwargs['word_len']
            ph2word = kwargs['ph2word']
            B, T_ph = ph2word.shape
            dur = torch.zeros([B, word_len.max() + 1]).scatter_add(1, ph2word, dur)
            dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            mel2word = self.length_regulator(dur).detach()
        return mel2word

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())
        return x_pos

    def store_inverse_all(self):

        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


class PortaSpeechFlow(PortaSpeech):

    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, word_dict_size, hparams, out_dims)
        cond_hs = 80
        if hparams.get('use_txt_cond', True):
            cond_hs = cond_hs + hparams['hidden_size']
        if hparams.get('use_latent_cond', False):
            cond_hs = cond_hs + hparams['latent_size']
        if hparams['use_cond_proj']:
            self.g_proj = nn.Conv1d(cond_hs, 160, 5, padding=2)
            cond_hs = 160
        self.post_flow = Glow(80, hparams['post_glow_hidden'], hparams['post_glow_kernel_size'], 1, hparams['post_glow_n_blocks'], hparams['post_glow_n_block_layers'], n_split=4, n_sqz=2, gin_channels=cond_hs, share_cond_layers=hparams['post_share_cond_layers'], share_wn_layers=hparams['share_wn_layers'], sigmoid_scale=hparams['sigmoid_scale'])
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None, forward_post_glow=True, two_stage=True, global_step=None):
        is_training = self.training
        train_fvae = not (forward_post_glow and two_stage)
        if not train_fvae:
            self.eval()
        with torch.set_grad_enabled(mode=train_fvae):
            ret = super(PortaSpeechFlow, self).forward(txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, spk_embed, spk_id, pitch, infer, tgt_mels, global_step)
        if (forward_post_glow or not two_stage) and self.hparams['use_post_flow']:
            self.run_post_glow(tgt_mels, infer, is_training, ret)
        return ret

    def run_post_glow(self, tgt_mels, infer, is_training, ret):
        x_recon = ret['mel_out'].transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if self.hparams.get('use_txt_cond', True):
            g = torch.cat([g, ret['decoder_inp'].transpose(1, 2)], 1)
        if self.hparams.get('use_latent_cond', False):
            g_z = ret['z_p'][:, :, :, None].repeat(1, 1, 1, 4).reshape(B, -1, T)
            g = torch.cat([g, g_z], 1)
        if self.hparams['use_cond_proj']:
            g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            if is_training:
                self.post_flow.train()
            nonpadding = ret['nonpadding'].transpose(1, 2)
            y_lengths = nonpadding.sum(-1)
            if self.hparams['detach_postflow_input']:
                g = g.detach()
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self.post_flow(tgt_mels, nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            ret['z_pf'], ret['ldj_pf'] = z_postflow, ldj
            ret['postflow'] = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            if torch.isnan(ret['postflow']):
                ret['postflow'] = None
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :])
            z_post = torch.randn(x_recon.shape) * self.hparams['noise_scale']
            x_recon, _ = self.post_flow(z_post, nonpadding, g, reverse=True)
            ret['mel_out'] = x_recon.transpose(1, 2)


class SingleWindowDisc(nn.Module):

    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.model = nn.ModuleList([nn.Sequential(*[nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25), nn.BatchNorm2d(hidden_size, 0.8)]), nn.Sequential(*[nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25), nn.BatchNorm2d(hidden_size, 0.8)]), nn.Sequential(*[nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])])
        ds_size = time_length // 2 ** 3, (freq_length + 7) // 2 ** 3
        self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)

    def forward(self, x):
        """
        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        for l in self.model:
            x = l(x)
            h.append(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        return validity, h


class MultiWindowDiscriminator(nn.Module):

    def __init__(self, time_lengths, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.discriminators = nn.ModuleList()
        for time_length in time_lengths:
            self.discriminators += [SingleWindowDisc(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size)]

    def forward(self, x, x_len, start_frames_wins=None):
        """
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        """
        validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        h = []
        for i, start_frames in zip(range(len(self.discriminators)), start_frames_wins):
            x_clip, start_frames = self.clip(x, x_len, self.win_lengths[i], start_frames)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            x_clip, h_ = self.discriminators[i](x_clip)
            h += h_
            validity.append(x_clip)
        if len(validity) != len(self.discriminators):
            return None, start_frames_wins, h
        validity = sum(validity)
        return validity, start_frames_wins, h

    def clip(self, x, x_len, win_length, start_frames=None):
        """Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        """
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame:start_frame + win_length]
        return x_batch, start_frames


class Discriminator(nn.Module):

    def __init__(self, time_lengths=[32, 64, 128], freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.discriminator = MultiWindowDiscriminator(freq_length=freq_length, time_lengths=time_lengths, kernel=kernel, c_in=c_in, hidden_size=hidden_size)

    def forward(self, x, start_frames_wins=None):
        """

        :param x: [B, T, 80]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])
        ret = {'y_c': None, 'y': None}
        ret['y'], start_frames_wins, ret['h'] = self.discriminator(x, x_len, start_frames_wins=start_frames_wins)
        ret['start_frames_wins'] = start_frames_wins
        return ret


class SyntaSpeech(FastSpeech):

    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        if hparams['num_spk'] > 1:
            self.spk_embed_proj = Embedding(hparams['num_spk'], self.hidden_size)
        if hparams['use_word_encoder']:
            self.word_encoder = RelTransformerEncoder(word_dict_size, self.hidden_size, self.hidden_size, self.hidden_size, 2, hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
        if hparams['dur_level'] == 'word':
            if hparams['word_encoder_type'] == 'rel_fft':
                self.ph2word_encoder = RelTransformerEncoder(0, self.hidden_size, self.hidden_size, self.hidden_size, 2, hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])
            if hparams['word_encoder_type'] == 'fft':
                self.ph2word_encoder = FFTBlocks(self.hidden_size, hparams['word_enc_layers'], 1, num_heads=hparams['num_heads'])
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
            self.enc_pos_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_query_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.dec_res_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.attn = MultiheadAttention(self.hidden_size, 1, encoder_decoder_attention=True, bias=False)
            self.attn.enable_torch_version = False
            if hparams['text_encoder_postnet']:
                self.text_encoder_postnet = ConvBlocks(self.hidden_size, self.hidden_size, [1] * 3, 5, layers_in_block=2)
        else:
            self.sin_pos = SinusoidalPosEmb(self.hidden_size)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = SyntaDurationPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['dur_predictor_layers'], dropout_rate=hparams['predictor_dropout'], kernel_size=hparams['dur_predictor_kernel'])
        if hparams['use_fvae']:
            del self.decoder
            del self.mel_out
            self.fvae = SyntaFVAE(c_in_out=self.out_dims, hidden_size=hparams['fvae_enc_dec_hidden'], c_latent=hparams['latent_size'], kernel_size=hparams['fvae_kernel_size'], enc_n_layers=hparams['fvae_enc_n_layers'], dec_n_layers=hparams['fvae_dec_n_layers'], c_cond=self.hidden_size, use_prior_flow=hparams['use_prior_flow'], flow_hidden=hparams['prior_flow_hidden'], flow_kernel_size=hparams['prior_flow_kernel_size'], flow_n_steps=hparams['prior_flow_n_blocks'], strides=[hparams['fvae_strides']], encoder_type=hparams['fvae_encoder_type'], decoder_type=hparams['fvae_decoder_type'])
        else:
            self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
            self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if self.hparams['add_word_pos']:
            self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None, spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None, global_step=None, graph_lst=None, etypes_lst=None, *args, **kwargs):
        if self.hparams['use_spk_embed']:
            spk_embed = spk_embed
        elif self.hparams['use_spk_id']:
            spk_embed = self.spk_embed_proj(spk_id)[:, None, :]
        else:
            spk_embed = 0
        ret = {}
        style_embed = self.forward_style_embed(spk_embed, spk_id)
        x, tgt_nonpadding = self.run_text_encoder(txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret, graph_lst=graph_lst, etypes_lst=etypes_lst)
        x = x + style_embed
        x = x * tgt_nonpadding
        ret['nonpadding'] = tgt_nonpadding
        if self.hparams['use_pitch_embed']:
            x = x + self.pitch_embed(pitch)
        ret['decoder_inp'] = x
        if infer and (mel2ph is None or mel2word is None):
            mel2word = ret['mel2word']
        ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels, global_step, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
        return ret

    def run_text_encoder(self, txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, style_embed, ret, graph_lst, etypes_lst):
        word2word = torch.arange(word_len)[None, :] + 1
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding + style_embed
        if self.hparams['use_word_encoder']:
            word_encoder_out = self.word_encoder(word_tokens) + style_embed
            ph_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)
        dur_input = ph_encoder_out * src_nonpadding
        if self.hparams['dur_level'] == 'word':
            word_encoder_out = 0
            h_ph_gb_word = group_hidden_by_segs(ph_encoder_out, ph2word, word_len)[0]
            word_encoder_out = word_encoder_out + self.ph2word_encoder(h_ph_gb_word)
            if self.hparams['use_word_encoder']:
                word_encoder_out = word_encoder_out + self.word_encoder(word_tokens)
            mel2word = self.forward_dur(dur_input, mel2word, ret, ph2word=ph2word, word_len=word_len, graph_lst=graph_lst, etypes_lst=etypes_lst)
            mel2word = clip_mel2token_to_multiple(mel2word, self.hparams['frames_multiple'])
            ret['mel2word'] = mel2word
            tgt_nonpadding = (mel2word > 0).float()[:, :, None]
            enc_pos = self.get_pos_embed(word2word, ph2word)
            dec_pos = self.get_pos_embed(word2word, mel2word)
            dec_word_mask = build_word_mask(mel2word, ph2word)
            x, weight = self.attention(ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask)
            if self.hparams['add_word_pos']:
                x = x + self.word_pos_proj(dec_pos)
            ret['attn'] = weight
        else:
            mel2ph = self.forward_dur(dur_input, mel2ph, ret)
            mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
            mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
            x = expand_states(ph_encoder_out, mel2ph)
            if self.hparams['add_word_pos']:
                dec_pos = self.get_pos_embed(word2word, mel2word)
                x = x + self.word_pos_proj(dec_pos)
            tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        if self.hparams['use_word_encoder']:
            x = x + expand_states(word_encoder_out, mel2word)
        return x, tgt_nonpadding

    def attention(self, ph_encoder_out, enc_pos, word_encoder_out, dec_pos, mel2word, dec_word_mask):
        ph_kv = self.enc_pos_proj(torch.cat([ph_encoder_out, enc_pos], -1))
        word_enc_out_expend = expand_states(word_encoder_out, mel2word)
        word_enc_out_expend = torch.cat([word_enc_out_expend, dec_pos], -1)
        if self.hparams['text_encoder_postnet']:
            word_enc_out_expend = self.dec_res_proj(word_enc_out_expend)
            word_enc_out_expend = self.text_encoder_postnet(word_enc_out_expend)
            dec_q = x_res = word_enc_out_expend
        else:
            dec_q = self.dec_query_proj(word_enc_out_expend)
            x_res = self.dec_res_proj(word_enc_out_expend)
        ph_kv, dec_q = ph_kv.transpose(0, 1), dec_q.transpose(0, 1)
        x, (weight, _) = self.attn(dec_q, ph_kv, ph_kv, attn_mask=(1 - dec_word_mask) * -1000000000.0)
        x = x.transpose(0, 1)
        x = x + x_res
        return x, weight

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0, mel2word=None, ph2word=None, graph_lst=None, etypes_lst=None):
        if not self.hparams['use_fvae']:
            x = self.decoder(x)
            x = self.mel_out(x)
            ret['kl'] = 0
            return x * tgt_nonpadding
        else:
            x = x.transpose(1, 2)
            tgt_nonpadding_BHT = tgt_nonpadding.transpose(1, 2)
            if infer:
                z = self.fvae(cond=x, infer=True, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
            else:
                tgt_mels = tgt_mels.transpose(1, 2)
                z, ret['kl'], ret['z_p'], ret['m_q'], ret['logs_q'] = self.fvae(tgt_mels, tgt_nonpadding_BHT, cond=x, mel2word=mel2word, ph2word=ph2word, graph_lst=graph_lst, etypes_lst=etypes_lst)
                if global_step < self.hparams['posterior_start_steps']:
                    z = torch.randn_like(z)
            x_recon = self.fvae.decoder(z, nonpadding=tgt_nonpadding_BHT, cond=x).transpose(1, 2)
            ret['pre_mel_out'] = x_recon
            return x_recon

    def forward_dur(self, dur_input, mel2word, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        word_len = kwargs['word_len']
        ph2word = kwargs['ph2word']
        graph_lst = kwargs['graph_lst']
        etypes_lst = kwargs['etypes_lst']
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding, ph2word, graph_lst, etypes_lst)
        B, T_ph = ph2word.shape
        dur = torch.zeros([B, word_len.max() + 1]).scatter_add(1, ph2word, dur)
        dur = dur[:, 1:]
        ret['dur'] = dur
        if mel2word is None:
            mel2word = self.length_regulator(dur).detach()
        return mel2word

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())
        return x_pos

    def store_inverse_all(self):

        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HifiGanGenerator(torch.nn.Module):

    def __init__(self, h, c_out=1):
        super(HifiGanGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h['resblock_kernel_sizes'])
        self.num_upsamples = len(h['upsample_rates'])
        self.conv_pre = weight_norm(Conv1d(80, h['upsample_initial_channel'], 7, 1, padding=3))
        resblock = ResBlock1 if h['resblock'] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes'])):
            c_cur = h['upsample_initial_channel'] // 2 ** (i + 1)
            self.ups.append(weight_norm(ConvTranspose1d(c_cur * 2, c_cur, k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h['upsample_initial_channel'] // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes'])):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, c_out, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
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


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, use_cond=False, c_in=1):
        super(DiscriminatorP, self).__init__()
        self.use_cond = use_cond
        if use_cond:
            t = hparams['hop_size']
            self.cond_net = torch.nn.ConvTranspose1d(80, 1, t * 2, stride=t, padding=t // 2)
            c_in = 2
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(c_in, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x, mel):
        fmap = []
        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, use_cond=False, c_in=1):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(2, use_cond=use_cond, c_in=c_in), DiscriminatorP(3, use_cond=use_cond, c_in=c_in), DiscriminatorP(5, use_cond=use_cond, c_in=c_in), DiscriminatorP(7, use_cond=use_cond, c_in=c_in), DiscriminatorP(11, use_cond=use_cond, c_in=c_in)])

    def forward(self, y, y_hat, mel=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, mel)
            y_d_g, fmap_g = d(y_hat, mel)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False, use_cond=False, upsample_rates=None, c_in=1):
        super(DiscriminatorS, self).__init__()
        self.use_cond = use_cond
        if use_cond:
            t = np.prod(upsample_rates)
            self.cond_net = torch.nn.ConvTranspose1d(80, 1, t * 2, stride=t, padding=t // 2)
            c_in = 2
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(c_in, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x, mel):
        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self, use_cond=False, c_in=1):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True, use_cond=use_cond, upsample_rates=[4, 4, hparams['hop_size'] // 16], c_in=c_in), DiscriminatorS(use_cond=use_cond, upsample_rates=[4, 4, hparams['hop_size'] // 32], c_in=c_in), DiscriminatorS(use_cond=use_cond, upsample_rates=[4, 4, hparams['hop_size'] // 64], c_in=c_in)])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=1), AvgPool1d(4, 2, padding=1)])

    def forward(self, y, y_hat, mel=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y, mel)
            y_d_g, fmap_g = d(y_hat, mel)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


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

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window', use_mel_loss=False):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.use_mel_loss = use_mel_loss
        self.mel_basis = None

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        if self.window.device != x.device:
            self.window = self.window
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        if self.use_mel_loss:
            if self.mel_basis is None:
                self.mel_basis = torch.from_numpy(librosa.filters.mel(22050, self.fft_size, 80)).T
            x_mag = x_mag @ self.mel_basis
            y_mag = y_mag @ self.mel_basis
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window='hann_window', use_mel_loss=False):
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
            self.stft_losses += [STFTLoss(fs, ss, wl, window, use_mel_loss)]

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


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, pad='ConstantPad1d', pad_params={'value': 0.0}):
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        return self.conv(self.pad(x))[:, :, :x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).

        Returns:
            Tensor: Output tensor (B, out_channels, T_out).

        """
        return self.deconv(x)[:, :, :-self.stride]


def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design prototype filter for PQMF.

    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.

    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).

    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427

    """
    assert taps % 2 == 0, 'The number of taps mush be even number.'
    assert 0.0 < cutoff_ratio < 1.0, 'Cutoff ratio must be > 0.0 and < 1.0.'
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio
    w = kaiser(taps + 1, beta)
    h = h_i * w
    return h


class PQMF(torch.nn.Module):
    """PQMF module.

    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122

    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        """Initilize PQMF module.

        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.

        """
        super(PQMF, self).__init__()
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps - 1) / 2) + (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - (taps - 1) / 2) - (-1) ** k * np.pi / 4)
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)
        self.register_buffer('analysis_filter', analysis_filter)
        self.register_buffer('synthesis_filter', synthesis_filter)
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer('updown_filter', updown_filter)
        self.subbands = subbands
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.

        Args:
            x (Tensor): Input tensor (B, 1, T).

        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).

        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.

        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self, kernel_size=3, channels=32, dilation=1, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_causal_conv=False):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(ResidualStack, self).__init__()
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
            self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params), torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
        else:
            self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), CausalConv1d(channels, channels, kernel_size, dilation=dilation, bias=bias, pad=pad, pad_params=pad_params), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        return self.stack(c) + self.skip_layer(c)


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode='nearest'):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.

        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, C, F, T).

        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),

        """
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, use_causal_conv=False):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]
            assert (freq_axis_kernel_size - 1) % 2 == 0, 'Not support even number freq axis kernel size.'
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = freq_axis_kernel_size, scale * 2 + 1
            if use_causal_conv:
                padding = freq_axis_padding, scale * 2
            else:
                padding = freq_axis_padding, scale
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T).

        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).

        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., :c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, aux_channels=80, aux_context_window=0, use_causal_conv=False):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.

        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales=upsample_scales, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, interpolate_mode=interpolate_mode, freq_axis_kernel_size=freq_axis_kernel_size, use_causal_conv=use_causal_conv)

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T').

        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).

        Note:
            The length of inputs considers the context window size.

        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class BasicDiscriminatorBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(BasicDiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True), nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)), nn.LeakyReLU(0.2, True), nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)), nn.LeakyReLU(0.2, True), nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)))

    def forward(self, x):
        return self.block(x)


class ResDiscriminatorBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResDiscriminatorBlock, self).__init__()
        self.block1 = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True), nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)))
        self.shortcut1 = nn.utils.weight_norm(nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=2))
        self.block2 = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)), nn.LeakyReLU(0.2, True), nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)))
        self.shortcut2 = nn.utils.weight_norm(nn.Conv1d(out_channel, out_channel, kernel_size=1, stride=1))

    def forward(self, x):
        x1 = self.block1(x)
        x1 = x1 + self.shortcut1(x)
        return self.block2(x1) + self.shortcut2(x1)


class ResNet18Discriminator(nn.Module):

    def __init__(self, stft_channel, in_channel=64):
        super(ResNet18Discriminator, self).__init__()
        self.input = nn.Sequential(nn.utils.weight_norm(nn.Conv1d(stft_channel, in_channel, kernel_size=7, stride=2, padding=1)), nn.LeakyReLU(0.2, True))
        self.df1 = BasicDiscriminatorBlock(in_channel, in_channel)
        self.df2 = ResDiscriminatorBlock(in_channel, in_channel * 2)
        self.df3 = ResDiscriminatorBlock(in_channel * 2, in_channel * 4)
        self.df4 = ResDiscriminatorBlock(in_channel * 4, in_channel * 8)

    def forward(self, x):
        x = self.input(x)
        x = self.df1(x)
        x = self.df2(x)
        x = self.df3(x)
        return self.df4(x)


class FrequencyDiscriminator(nn.Module):

    def __init__(self, in_channel=64, fft_size=1024, hop_length=256, win_length=1024, window='hann_window'):
        super(FrequencyDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.stft_channel = fft_size // 2 + 1
        self.resnet_disc = ResNet18Discriminator(self.stft_channel, in_channel)

    def forward(self, x):
        x_stft = torch.stft(x, self.fft_size, self.hop_length, self.win_length, self.window)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        x_real = self.resnet_disc(real)
        x_imag = self.resnet_disc(imag)
        return x_real, x_imag


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = f0_values / self.sampling_rate % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :] < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class PulseGen(torch.nn.Module):
    """ Definition of Pulse train generator

    There are many ways to implement pulse generator.
    Here, PulseGen is based on SinGen. For a perfect
    """

    def __init__(self, samp_rate, pulse_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(PulseGen, self).__init__()
        self.pulse_amp = pulse_amp
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.noise_std = noise_std
        self.l_sinegen = SineGen(self.sampling_rate, harmonic_num=0, sine_amp=self.pulse_amp, noise_std=0, voiced_threshold=self.voiced_threshold, flag_for_pulse=True)

    def forward(self, f0):
        """ Pulse train generator
        pulse_train, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output pulse_train: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)

        Note: self.l_sine doesn't make sure that the initial phase of
        a voiced segment is np.pi, the first pulse in a voiced segment
        may not be at the first time step within a voiced segment
        """
        with torch.no_grad():
            sine_wav, uv, noise = self.l_sinegen(f0)
            pure_sine = sine_wav - noise
            sine_1 = torch.roll(pure_sine, shifts=1, dims=1)
            uv_1 = torch.roll(uv, shifts=1, dims=1)
            uv_1[:, 0, :] = 0
            sine_2 = torch.roll(pure_sine, shifts=-1, dims=1)
            uv_2 = torch.roll(uv, shifts=-1, dims=1)
            uv_2[:, -1, :] = 0
            loc = (pure_sine > sine_1) * (pure_sine > sine_2) * (uv_1 > 0) * (uv_2 > 0) * (uv > 0) + (uv_1 < 1) * (uv > 0)
            pulse_train = pure_sine * loc
            pulse_noise = torch.randn_like(pure_sine) * self.noise_std
            pulse_train += pulse_noise * loc + pulse_noise * (1 - uv)
        return pulse_train, sine_wav, uv, pulse_noise


class SignalsConv1d(torch.nn.Module):
    """ Filtering input signal with time invariant filter
    Note: FIRFilter conducted filtering given fixed FIR weight
          SignalsConv1d convolves two signals
    Note: this is based on torch.nn.functional.conv1d

    """

    def __init__(self):
        super(SignalsConv1d, self).__init__()

    def forward(self, signal, system_ir):
        """ output = forward(signal, system_ir)

        signal:    (batchsize, length1, dim)
        system_ir: (length2, dim)

        output:    (batchsize, length1, dim)
        """
        if signal.shape[-1] != system_ir.shape[-1]:
            None
            None
            None
            None
            None
            sys.exit(1)
        padding_length = system_ir.shape[0] - 1
        groups = signal.shape[-1]
        signal_pad = torch_nn_func.pad(signal.permute(0, 2, 1), (padding_length, 0))
        ir = torch.flip(system_ir.unsqueeze(1).permute(2, 1, 0), dims=[2])
        output = torch_nn_func.conv1d(signal_pad, ir, groups=groups)
        return output.permute(0, 2, 1)


class CyclicNoiseGen_v1(torch.nn.Module):
    """ CyclicnoiseGen_v1
    Cyclic noise with a single parameter of beta.
    Pytorch v1 implementation assumes f_t is also fixed
    """

    def __init__(self, samp_rate, noise_std=0.003, voiced_threshold=0):
        super(CyclicNoiseGen_v1, self).__init__()
        self.samp_rate = samp_rate
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.l_pulse = PulseGen(samp_rate, pulse_amp=1.0, noise_std=noise_std, voiced_threshold=voiced_threshold)
        self.l_conv = SignalsConv1d()

    def noise_decay(self, beta, f0mean):
        """ decayed_noise = noise_decay(beta, f0mean)
        decayed_noise =  n[t]exp(-t * f_mean / beta / samp_rate)

        beta: (dim=1) or (batchsize=1, 1, dim=1)
        f0mean (batchsize=1, 1, dim=1)

        decayed_noise (batchsize=1, length, dim=1)
        """
        with torch.no_grad():
            length = 4.6 * self.samp_rate / f0mean
            length = length.int()
            time_idx = torch.arange(0, length, device=beta.device)
            time_idx = time_idx.unsqueeze(0).unsqueeze(2)
            time_idx = time_idx.repeat(beta.shape[0], 1, beta.shape[2])
        noise = torch.randn(time_idx.shape, device=beta.device)
        decay = torch.exp(-time_idx * f0mean / beta / self.samp_rate)
        return noise * self.noise_std * decay

    def forward(self, f0s, beta):
        """ Producde cyclic-noise
        """
        pulse_train, sine_wav, uv, noise = self.l_pulse(f0s)
        pure_pulse = pulse_train - noise
        if (uv < 1).all():
            cyc_noise = torch.zeros_like(sine_wav)
        else:
            f0mean = f0s[uv > 0].mean()
            decayed_noise = self.noise_decay(beta, f0mean)[0, :, :]
            cyc_noise = self.l_conv(pure_pulse, decayed_noise)
        cyc_noise = cyc_noise + noise * (1.0 - uv)
        return cyc_noise, pulse_train, sine_wav, uv, noise


class SourceModuleCycNoise_v1(torch.nn.Module):
    """ SourceModuleCycNoise_v1
    SourceModule(sampling_rate, noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz

    noise_std: std of Gaussian noise (default: 0.003)
    voiced_threshold: threshold to set U/V given F0 (default: 0)

    cyc, noise, uv = SourceModuleCycNoise_v1(F0_upsampled, beta)
    F0_upsampled (batchsize, length, 1)
    beta (1)
    cyc (batchsize, length, 1)
    noise (batchsize, length, 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, noise_std=0.003, voiced_threshod=0):
        super(SourceModuleCycNoise_v1, self).__init__()
        self.sampling_rate = sampling_rate
        self.noise_std = noise_std
        self.l_cyc_gen = CyclicNoiseGen_v1(sampling_rate, noise_std, voiced_threshod)

    def forward(self, f0_upsamped, beta):
        """
        cyc, noise, uv = SourceModuleCycNoise_v1(F0, beta)
        F0_upsampled (batchsize, length, 1)
        beta (1)
        cyc (batchsize, length, 1)
        noise (batchsize, length, 1)
        uv (batchsize, length, 1)
        """
        cyc, pulse, sine, uv, add_noi = self.l_cyc_gen(f0_upsamped, beta)
        noise = torch.randn_like(uv) * self.noise_std / 3
        return cyc, noise, uv


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self, in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True, upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_final_nonlinear_activation=True, use_weight_norm=True, use_causal_conv=False, use_pitch_embed=False, use_nsf=False, sample_rate=22050, **kwargs):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANGenerator, self).__init__()
        assert channels >= np.prod(upsample_scales)
        assert channels % 2 ** len(upsample_scales) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        layers = []
        if not use_causal_conv:
            layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        else:
            layers += [CausalConv1d(in_channels, channels, kernel_size, bias=bias, pad=pad, pad_params=pad_params)]
        self.use_pitch_embed = use_pitch_embed
        if use_pitch_embed:
            self.pitch_embed = nn.Embedding(300, in_channels, 0)
            self.c_proj = nn.Conv1d(2 * in_channels, in_channels, 1)
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [torch.nn.ConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2, bias=bias)]
            else:
                layers += [CausalConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // 2 ** (i + 1), dilation=stack_kernel_size ** j, bias=bias, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad, pad_params=pad_params, use_causal_conv=use_causal_conv)]
        self.use_nsf = use_nsf
        if use_nsf:
            self.harmonic_num = 8
            hop_size = np.prod(upsample_scales)
            self.f0_upsamp = torch.nn.Upsample(scale_factor=hop_size)
            self.m_source = SourceModuleCycNoise_v1(sample_rate, 0.003)
            self.nsf_conv = nn.Sequential(nn.Conv1d(1, channels // 2 ** (i + 1), 1), torch.nn.Tanh())
        self.melgan_body = torch.nn.Sequential(*layers)
        layers = []
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias)]
        else:
            layers += [CausalConv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias, pad=pad, pad_params=pad_params)]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan_final = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, c, f0=None, pitch=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        if self.use_pitch_embed:
            c = self.c_proj(torch.cat([c, self.pitch_embed(pitch).transpose(1, 2)], 1))
        x = self.melgan_body(c)
        if self.use_nsf:
            f0_upsample = self.f0_upsamp(f0[:, None, :])
            f0_upsample = self.nsf_conv(f0_upsample)
            x = x + f0_upsample
        x = self.melgan_final(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/spec2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[5, 3], channels=16, max_downsample_channels=1024, bias=True, downsample_scales=[4, 4, 4, 4], nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1
        self.layers += [torch.nn.Sequential(getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params), torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_size=downsample_scale * 10 + 1, stride=downsample_scale, padding=downsample_scale * 5, groups=in_chs // 4, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.layers += [torch.nn.Conv1d(out_chs, out_channels, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2, bias=bias)]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]
        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, scales=3, downsample_pooling='AvgPool1d', downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 1, 'count_include_pad': False}, kernel_sizes=[5, 3], channels=16, max_downsample_channels=1024, bias=True, downsample_scales=[4, 4, 4, 4], nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_weight_norm=True, **kwargs):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()
        for _ in range(scales):
            self.discriminators += [MelGANDiscriminator(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_sizes, channels=channels, max_downsample_channels=max_downsample_channels, bias=bias, downsample_scales=downsample_scales, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad, pad_params=pad_params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)
        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/spec2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, bias=True, use_weight_norm=True, use_causal_conv=False, upsample_conditional_features=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}, use_pitch_embed=False, use_nsf=False, sample_rate=22050):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)
        self.aux_context_window = aux_context_window
        if upsample_conditional_features:
            upsample_params.update({'use_causal_conv': use_causal_conv})
            if upsample_net == 'MelGANGenerator':
                assert aux_context_window == 0
                upsample_params.update({'use_weight_norm': False, 'use_final_nonlinear_activation': False})
                self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == 'ConvInUpsampleNetwork':
                    upsample_params.update({'aux_channels': aux_channels, 'aux_context_window': aux_context_window})
                self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=bias, use_causal_conv=use_causal_conv)
            self.conv_layers += [conv]
        self.last_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, skip_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, out_channels, bias=True)])
        self.use_pitch_embed = use_pitch_embed
        if use_pitch_embed:
            self.pitch_embed = nn.Embedding(300, aux_channels, 0)
            self.c_proj = nn.Linear(2 * aux_channels, aux_channels)
        self.use_nsf = use_nsf
        if use_nsf:
            self.harmonic_num = 8
            hop_size = np.prod(upsample_params['upsample_scales'])
            self.f0_upsamp = torch.nn.Upsample(scale_factor=hop_size)
            self.m_source = SourceModuleCycNoise_v1(sample_rate, 0.003)
            self.nsf_conv = nn.Sequential(nn.Conv1d(1, aux_channels, 1), torch.nn.Tanh())
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c=None, pitch=None, f0=None, **kwargs):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, C_in, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            pitch (Tensor): Local conditioning pitch (B, T').

        Returns:
            Tensor: Output tensor (B, C_out, T)

        """
        if c is not None and self.upsample_net is not None:
            if self.use_pitch_embed:
                p = self.pitch_embed(pitch)
                c = self.c_proj(torch.cat([c.transpose(1, 2), p], -1)).transpose(1, 2)
            c = self.upsample_net(c)
            if self.use_nsf:
                f0_upsample = self.f0_upsamp(f0[:, None, :][:, :, self.aux_context_window:-self.aux_context_window])
                f0_upsample = self.nsf_conv(f0_upsample)
                c = c + f0_upsample
            if x is None:
                x = torch.randn([c.size(0), 1, c.size(-1)])
            assert c.size(-1) == x.size(-1), (c.size(-1), x.size(-1))
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)


hparams = {}


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=10, conv_channels=64, dilation_factor=1, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, bias=True, use_weight_norm=True):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        assert dilation_factor > 0, 'Dilation factor must be > 0.'
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [Conv1d(conv_in_channels, conv_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias), getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(conv_in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, cond=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            cond (Tensor): Input noise signal (B, H, T_frame).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        cond_layer_i = len(self.conv_layers) // 2
        for i, f in enumerate(self.conv_layers):
            if i == cond_layer_i and cond is not None:
                aux_context_window = hparams['aux_context_window']
                cond = cond[:, :, aux_context_window:-aux_context_window]
                cond = cond[:, :, :, None].repeat([1, 1, 1, hparams['hop_size']]).reshape(cond.shape[0], cond.shape[1], -1)
                x = x + cond
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)


class ResidualParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, dropout=0.0, bias=True, use_weight_norm=True, use_causal_conv=False, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            nonlinear_activation_params (dict): Nonlinear function parameters

        """
        super(ResidualParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        self.first_conv = torch.nn.Sequential(Conv1d1x1(in_channels, residual_channels, bias=True), getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params))
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=-1, dilation=dilation, dropout=dropout, bias=bias, use_causal_conv=use_causal_conv)
            self.conv_layers += [conv]
        self.last_conv_layers = torch.nn.ModuleList([getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params), Conv1d1x1(skip_channels, skip_channels, bias=True), getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params), Conv1d1x1(skip_channels, out_channels, bias=True)])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, None)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if version.parse(torch.__version__[:6]) < version.parse('1.11'):
            self._sync_params()
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            assert len(self.device_ids) == 1
            if self.module.training:
                output = self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                output = self.module.test_step(*inputs[0], **kwargs[0])
            else:
                output = self.module.validation_step(*inputs[0], **kwargs[0])
            if torch.is_grad_enabled():
                if self.find_unused_parameters:
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
        else:
            from torch.nn.parallel.distributed import logging
            from torch.nn.parallel.distributed import Join
            from torch.nn.parallel.distributed import _DDPSink
            from torch.nn.parallel.distributed import _tree_flatten_with_rref
            from torch.nn.parallel.distributed import _tree_unflatten_with_rref
            with torch.autograd.profiler.record_function('DistributedDataParallel.forward'):
                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.logger.set_runtime_stats_and_log()
                    self.num_iterations += 1
                    self.reducer.prepare_for_forward()
                work = Join.notify_join_context(self)
                if work:
                    self.reducer._set_forward_pass_work_handle(work, self._divide_by_initial_world_size)
                if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                    logging.info('Reducer buckets have been rebuilt in this iteration.')
                    self._has_rebuilt_buckets = True
                buffer_hook_registered = hasattr(self, 'buffer_hook')
                if self._check_sync_bufs_pre_fwd():
                    self._sync_buffers()
                if self._join_config.enable:
                    self._check_global_requires_backward_grad_sync(is_joined_rank=False)
                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
                if self._check_sync_bufs_post_fwd():
                    self._sync_buffers()
                if torch.is_grad_enabled() and self.require_backward_grad_sync:
                    self.require_forward_param_sync = True
                    if self.find_unused_parameters and not self.static_graph:
                        self.reducer.prepare_for_backward(list(_find_tensors(output)))
                    else:
                        self.reducer.prepare_for_backward([])
                else:
                    self.require_forward_param_sync = False
            if self.find_unused_parameters and not self.static_graph or self.static_graph and self.num_iterations == 1:
                state_dict = {'static_graph': self.static_graph, 'num_iterations': self.num_iterations}
                output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(output)
                output_placeholders = [None for _ in range(len(output_tensor_list))]
                for i, output in enumerate(output_tensor_list):
                    if torch.is_tensor(output) and output.grad_fn is None:
                        output_placeholders[i] = output
                passthrough_tensor_list = _DDPSink.apply(self.reducer, state_dict, *output_tensor_list)
                for i in range(len(output_placeholders)):
                    if output_placeholders[i] is None:
                        output_placeholders[i] = passthrough_tensor_list[i]
                output = _tree_unflatten_with_rref(output_placeholders, treespec, output_is_rref)
        return output


class Tee(object):

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern), key=lambda x: -int(re.findall('.*steps\\_(\\d+)\\.ckpt', x)[0]))


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def move_to_cuda(batch, gpu_id=0):
    if callable(getattr(batch, 'cuda', None)):
        return batch
    elif callable(getattr(batch, 'to', None)):
        return batch
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    return batch


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


class Trainer:

    def __init__(self, work_dir, default_save_path=None, accumulate_grad_batches=1, max_updates=160000, print_nan_grads=False, val_check_interval=2000, num_sanity_val_steps=5, amp=False, log_save_interval=100, tb_log_interval=10, monitor_key='val_loss', monitor_mode='min', num_ckpt_keep=5, save_best=True, resume_from_checkpoint=0, seed=1234, debug=False):
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_updates = max_updates
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.default_save_path = default_save_path
        self.resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint > 0 else None
        self.seed = seed
        self.debug = debug
        self.task = None
        self.optimizers = []
        self.testing = False
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0
        self.monitor_key = monitor_key
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.monitor_op = np.less if monitor_mode == 'min' else np.greater
        self.best_val_results = np.Inf if monitor_mode == 'min' else -np.Inf
        self.mode = 'min'
        self.all_gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x != '']
        self.num_gpus = len(self.all_gpu_ids)
        self.on_gpu = self.num_gpus > 0
        self.root_gpu = 0
        logging.info(f'GPU available: {torch.cuda.is_available()}, GPU used: {self.all_gpu_ids}')
        self.use_ddp = self.num_gpus > 1
        self.proc_rank = 0
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.tb_log_interval = tb_log_interval
        self.amp = amp
        self.amp_scalar = GradScaler()

    def test(self, task_cls):
        self.testing = True
        self.fit(task_cls)

    def fit(self, task_cls):
        if len(self.all_gpu_ids) > 1:
            mp.spawn(self.ddp_run, nprocs=self.num_gpus, args=(task_cls, copy.deepcopy(hparams)))
        else:
            self.task = task_cls()
            self.task.trainer = self
            self.run_single_process(self.task)
        return 1

    def ddp_run(self, gpu_idx, task_cls, hparams_):
        hparams.update(hparams_)
        self.proc_rank = gpu_idx
        self.init_ddp_connection(self.proc_rank, self.num_gpus)
        if dist.get_rank() != 0 and not self.debug:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        task = task_cls()
        task.trainer = self
        torch.cuda.set_device(gpu_idx)
        self.root_gpu = gpu_idx
        self.task = task
        self.run_single_process(task)

    def run_single_process(self, task):
        """Sanity check a few things before starting actual training.

        :param task:
        """
        if self.proc_rank == 0:
            self.save_terminal_logs()
            if not self.testing:
                self.save_codes()
        model = task.build_model()
        if model is not None:
            task.model = model
        checkpoint, _ = get_last_checkpoint(self.work_dir, self.resume_from_checkpoint)
        if checkpoint is not None:
            self.restore_weights(checkpoint)
        elif self.on_gpu:
            task
        if not self.testing:
            self.optimizers = task.configure_optimizers()
            self.fisrt_epoch = True
        if checkpoint is not None:
            self.restore_opt_state(checkpoint)
        del checkpoint
        if self.on_gpu:
            torch.cuda.empty_cache()
        if self.use_ddp:
            self.task = self.configure_ddp(self.task)
            dist.barrier()
        task_ref = self.get_task_ref()
        task_ref.trainer = self
        task_ref.testing = self.testing
        if self.proc_rank == 0:
            task_ref.build_tensorboard(save_dir=self.work_dir, name='tb_logs')
        else:
            os.makedirs('tmp', exist_ok=True)
            task_ref.build_tensorboard(save_dir='tmp', name='tb_tmp')
        self.logger = task_ref.logger
        try:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()
        except KeyboardInterrupt as e:
            traceback.print_exc()
            task_ref.on_keyboard_interrupt()

    def run_evaluation(self, test=False):
        eval_results = self.evaluate(self.task, test, tqdm_desc='Valid' if not test else 'test', max_batches=hparams['eval_max_batches'])
        if eval_results is not None and 'tb_log' in eval_results:
            tb_log_output = eval_results['tb_log']
            self.log_metrics_to_tb(tb_log_output)
        if self.proc_rank == 0 and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        if max_batches == -1:
            max_batches = None
        task.zero_grad()
        task.eval()
        torch.set_grad_enabled(False)
        task_ref = self.get_task_ref()
        if test:
            ret = task_ref.test_start()
            if ret == 'EXIT':
                return
        else:
            task_ref.validation_start()
        outputs = []
        dataloader = task_ref.test_dataloader() if test else task_ref.val_dataloader()
        pbar = tqdm.tqdm(dataloader, desc=tqdm_desc, total=max_batches, dynamic_ncols=True, unit='step', disable=self.root_gpu > 0)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            if max_batches is not None and batch_idx >= max_batches:
                break
            if self.on_gpu:
                batch = move_to_cuda(batch, self.root_gpu)
            args = [batch, batch_idx]
            if self.use_ddp:
                output = task(*args)
            elif test:
                output = task_ref.test_step(*args)
            else:
                output = task_ref.validation_step(*args)
            outputs.append(output)
        if test:
            eval_results = task_ref.test_end(outputs)
        else:
            eval_results = task_ref.validation_end(outputs)
        task.train()
        torch.set_grad_enabled(True)
        return eval_results

    def train(self):
        task_ref = self.get_task_ref()
        task_ref.on_train_start()
        if self.num_sanity_val_steps > 0:
            self.evaluate(self.task, False, 'Sanity Val', max_batches=self.num_sanity_val_steps)
        if self.on_gpu:
            torch.cuda.empty_cache()
        dataloader = task_ref.train_dataloader()
        epoch = self.current_epoch
        while True:
            if self.use_ddp and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
            task_ref.current_epoch = epoch
            self.current_epoch = epoch
            self.batch_loss_value = 0
            task_ref.on_epoch_start()
            train_pbar = tqdm.tqdm(dataloader, initial=self.global_step, total=float('inf'), dynamic_ncols=True, unit='step', disable=self.root_gpu > 0)
            for batch_idx, batch in enumerate(train_pbar):
                if self.global_step % self.val_check_interval == 0 and not self.fisrt_epoch:
                    self.run_evaluation()
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                train_pbar.set_postfix(**pbar_metrics)
                self.fisrt_epoch = False
                if (self.global_step + 1) % self.tb_log_interval == 0:
                    self.log_metrics_to_tb(tb_metrics)
                self.global_step += 1
                task_ref.global_step = self.global_step
                if self.global_step > self.max_updates:
                    None
                    break
            task_ref.on_epoch_end()
            epoch += 1
            if self.global_step > self.max_updates:
                break
        task_ref.on_train_end()

    def run_training_batch(self, batch_idx, batch):
        if batch is None:
            return {}
        all_progress_bar_metrics = []
        all_log_metrics = []
        task_ref = self.get_task_ref()
        for opt_idx, optimizer in enumerate(self.optimizers):
            if optimizer is None:
                continue
            if len(self.optimizers) > 1:
                for param in task_ref.parameters():
                    param.requires_grad = False
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.requires_grad = True
            with autocast(enabled=self.amp):
                if self.on_gpu:
                    batch = move_to_cuda(copy.copy(batch), self.root_gpu)
                args = [batch, batch_idx, opt_idx]
                if self.use_ddp:
                    output = self.task(*args)
                else:
                    output = task_ref.training_step(*args)
                loss = output['loss']
                if loss is None:
                    continue
                progress_bar_metrics = output['progress_bar']
                log_metrics = output['tb_log']
                loss = loss / self.accumulate_grad_batches
            if loss.requires_grad:
                if self.amp:
                    self.amp_scalar.scale(loss).backward()
                else:
                    loss.backward()
            all_log_metrics.append(log_metrics)
            all_progress_bar_metrics.append(progress_bar_metrics)
            if loss is None:
                continue
            if self.print_nan_grads:
                has_nan_grad = False
                for name, param in task_ref.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad.float()).any():
                        None
                        has_nan_grad = True
                if has_nan_grad:
                    exit(0)
            if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                task_ref.on_before_optimization(opt_idx)
                if self.amp:
                    self.amp_scalar.step(optimizer)
                    self.amp_scalar.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)
        all_progress_bar_metrics = {k: v for d in all_progress_bar_metrics for k, v in d.items()}
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        return all_progress_bar_metrics, all_log_metrics

    def restore_weights(self, checkpoint):
        task_ref = self.get_task_ref()
        for k, v in checkpoint['state_dict'].items():
            getattr(task_ref, k).load_state_dict(v)
        if self.on_gpu:
            task_ref
        self.best_val_results = checkpoint['checkpoint_callback_best']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        task_ref.global_step = self.global_step
        if self.use_ddp:
            dist.barrier()

    def restore_opt_state(self, checkpoint):
        if self.testing:
            return
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            try:
                optimizer.load_state_dict(opt_state)
                if self.on_gpu:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v
            except ValueError:
                None
        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            None
            return
        did_restore = True
        return did_restore

    def save_checkpoint(self, epoch, logs=None):
        monitor_op = np.less
        ckpt_path = f'{self.work_dir}/model_ckpt_steps_{self.global_step}.ckpt'
        logging.info(f'Epoch {epoch:05d}@{self.global_step}: saving model to {ckpt_path}')
        self._atomic_save(ckpt_path)
        for old_ckpt in get_all_ckpts(self.work_dir)[self.num_ckpt_keep:]:
            remove_file(old_ckpt)
            logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
        current = None
        if logs is not None and self.monitor_key in logs:
            current = logs[self.monitor_key]
        if current is not None and self.save_best:
            if monitor_op(current, self.best_val_results):
                best_filepath = f'{self.work_dir}/model_ckpt_best.pt'
                self.best_val_results = current
                logging.info(f'Epoch {epoch:05d}@{self.global_step}: {self.monitor_key} reached {current:0.5f}. Saving model to {best_filepath}')
                self._atomic_save(best_filepath)

    def _atomic_save(self, filepath):
        checkpoint = self.dump_checkpoint()
        tmp_path = str(filepath) + '.part'
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, filepath)

    def dump_checkpoint(self):
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step, 'checkpoint_callback_best': self.best_val_results}
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                optimizer_states.append(optimizer.state_dict())
        checkpoint['optimizer_states'] = optimizer_states
        task_ref = self.get_task_ref()
        checkpoint['state_dict'] = {k: v.state_dict() for k, v in task_ref.named_children() if len(list(v.parameters())) > 0}
        return checkpoint

    def configure_ddp(self, task):
        task = DDP(task, device_ids=[self.root_gpu], find_unused_parameters=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        return task

    def init_ddp_connection(self, proc_rank, world_size):
        root_node = '127.0.0.1'
        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]
            number = re.sub('[^0-9]', '', number)
            root_node = name + number
        return root_node

    def get_task_ref(self):
        task: BaseTask = self.task.module if isinstance(self.task, DDP) else self.task
        return task

    def log_metrics_to_tb(self, metrics, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        """
        scalar_metrics = self.metrics_to_scalars(metrics)
        step = step if step is not None else self.global_step
        if self.proc_rank == 0:
            self.log_metrics(self.logger, scalar_metrics, step=step)

    @staticmethod
    def log_metrics(logger, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if type(v) is dict:
                v = self.metrics_to_scalars(v)
            new_metrics[k] = v
        return new_metrics

    def save_terminal_logs(self):
        t = datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs(f'{self.work_dir}/terminal_logs', exist_ok=True)
        Tee(f'{self.work_dir}/terminal_logs/log_{t}.txt', 'w')

    def save_codes(self):
        if len(hparams['save_codes']) > 0:
            t = datetime.now().strftime('%Y%m%d%H%M%S')
            code_dir = f'{self.work_dir}/codes/{t}'
            subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
            for c in hparams['save_codes']:
                if os.path.exists(c):
                    subprocess.check_call(f'rsync -aR --include="*.py" --include="*.yaml" --exclude="__pycache__" --include="*/" --exclude="*" "./{c}" "{code_dir}/"', shell=True)
            None


def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """
    wraps(fn)
    attr_name = '_lazy_' + fn.__name__

    def _get_data_loader(self):
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)
            except AttributeError as e:
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)
        return value
    return _get_data_loader


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


class BaseTask(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseTask, self).__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_norm = hparams['clip_grad_norm']
        self.gradient_clip_val = hparams.get('clip_grad_value', 0)
        self.model = None
        self.training_losses_meter = None
        self.logger: SummaryWriter = None

    def build_model(self):
        raise NotImplementedError

    @data_loader
    def train_dataloader(self):
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        return None

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': AvgrageMeter()}

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        None

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        """

        :param sample:
        :param batch_idx:
        :param optimizer_idx:
        :return: {'loss': torch.Tensor, 'progress_bar': dict, 'tb_log': dict}
        """
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())
        if optimizer_idx >= 0:
            log_outputs[f'lr_{optimizer_idx}'] = self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {'loss': total_loss, 'progress_bar': progress_bar_log, 'tb_log': tb_log}

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: {"losses": {...}, "total_loss": float, ...} or (total loss: torch.Tensor, loss_log: dict)
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        all_losses_meter = {'total_loss': AvgrageMeter()}
        for output in outputs:
            if len(output) == 0 or output is None:
                continue
            if isinstance(output, dict):
                assert 'losses' in output, 'Key "losses" should exist in validation output.'
                n = output.pop('nsamples', 1)
                losses = tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                assert len(output) == 2, 'Validation output should only consist of two elements: (total_loss, losses)'
                n = 1
                total_loss, losses = output
                losses = tensors_to_scalars(losses)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)
        loss_output = {k: round(v.avg, 4) for k, v in all_losses_meter.items()}
        None
        return {'tb_log': {f'val/{k}': v for k, v in loss_output.items()}, 'val_loss': loss_output['total_loss']}

    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    @classmethod
    def start(cls):
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        work_dir = hparams['work_dir']
        trainer = Trainer(work_dir=work_dir, val_check_interval=hparams['val_check_interval'], tb_log_interval=hparams['tb_log_interval'], max_updates=hparams['max_updates'], num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000, accumulate_grad_batches=hparams['accumulate_grad_batches'], print_nan_grads=hparams['print_nan_grads'], resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0), amp=hparams['amp'], monitor_key=hparams['valid_monitor_key'], monitor_mode=hparams['valid_monitor_mode'], num_ckpt_keep=hparams['num_ckpt_keep'], save_best=hparams['save_best'], seed=hparams['seed'], debug=hparams['debug'])
        if not hparams['infer']:
            trainer.fit(cls)
        else:
            trainer.test(cls)

    def on_keyboard_interrupt(self):
        pass


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicDiscriminatorBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BatchNormConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CausalConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConformerDecoder,
     lambda: ([], {'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ConformerLayers,
     lambda: ([], {'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv1d1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvolutionModule,
     lambda: ([], {'channels': 4, 'kernel_size': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CyclicNoiseGen_v1,
     lambda: ([], {'samp_rate': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 37, 4])], {}),
     False),
    (DecSALayer,
     lambda: ([], {'c': 4, 'num_heads': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DecoderRNN,
     lambda: ([], {'hidden_size': 4, 'decoder_rnn_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1])], {}),
     False),
    (Encoder,
     lambda: ([], {'hidden_channels': 4, 'filter_channels': 4, 'n_heads': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (FFTBlocks,
     lambda: ([], {'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (FlipLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (HighwayNetwork,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvConv,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (InvConvNear,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LambdaLayer,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LengthRegulator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogSTFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'channels': 4, 'out_channels': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MultiLayeredConv1d,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ParallelWaveGANDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (ParallelWaveGANGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNet,
     lambda: ([], {'in_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PulseGen,
     lambda: ([], {'samp_rate': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RelPositionMultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResDiscriminatorBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResidualParallelWaveGANDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SineGen,
     lambda: ([], {'samp_rate': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SinusoidalPosEmb,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (SinusoidalPositionalEmbedding,
     lambda: ([], {'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SourceModuleCycNoise_v1,
     lambda: ([], {'sampling_rate': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 37, 4])], {}),
     False),
    (SpectralConvergengeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stretch2d,
     lambda: ([], {'x_scale': 1.0, 'y_scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerDecoderLayer,
     lambda: ([], {'hidden_size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransformerFFNLayer,
     lambda: ([], {'hidden_size': 4, 'filter_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_yerfor_SyntaSpeech(_paritybench_base):
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

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

