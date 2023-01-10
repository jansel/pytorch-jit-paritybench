import sys
_module = sys.modules[__name__]
del sys
base_binarizer = _module
base_binarizer_emotion = _module
base_preprocess = _module
binarize = _module
preprocess = _module
train_mfa_align = _module
data_gen_utils = _module
audio = _module
inference = _module
model = _module
params_data = _module
params_model = _module
test_emotion = _module
txt_processors = _module
base_text_processor = _module
en = _module
wav_processors = _module
base_processor = _module
common_processors = _module
pre_align = _module
GenerSpeech = _module
base_tts_infer = _module
infer = _module
generspeech = _module
glow_modules = _module
mixstyle = _module
prosody_util = _module
wavenet = _module
dataset = _module
generspeech = _module
modules = _module
common_layers = _module
espnet_positional_embedding = _module
ssim = _module
fs2 = _module
pe = _module
tts_modules = _module
hifigan = _module
mel_utils = _module
parallel_wavegan = _module
layers = _module
causal_conv = _module
pqmf = _module
residual_block = _module
residual_stack = _module
tf_layers = _module
upsample = _module
losses = _module
stft_loss = _module
models = _module
melgan = _module
parallel_wavegan = _module
source = _module
optimizers = _module
radam = _module
stft_loss = _module
utils = _module
base_task = _module
run = _module
dataset_utils = _module
fs2 = _module
fs2_utils = _module
pe = _module
tts = _module
tts_base = _module
tts_utils = _module
dataset_utils = _module
vocoder_base = _module
utils = _module
ckpt_utils = _module
common_schedulers = _module
cwt = _module
ddp_utils = _module
hparams = _module
indexed_datasets = _module
multiprocess_utils = _module
os_utils = _module
pitch_utils = _module
pl_utils = _module
plot = _module
text_encoder = _module
text_norm = _module
trainer = _module
training_utils = _module
tts_utils = _module
vocoders = _module
base_vocoder = _module
hifigan = _module
pwg = _module
vocoder_utils = _module

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


from collections import Counter


import random


import numpy as np


import pandas as pd


import warnings


from scipy.ndimage.morphology import binary_dilation


import re


from collections import OrderedDict


from matplotlib import cm


import matplotlib.pyplot as plt


from torch.nn.utils import clip_grad_norm_


from scipy.optimize import brentq


from torch import nn


import logging


import math


import numpy


import time


import itertools


from sklearn import metrics


import torch.nn as nn


import torch.nn.functional as F


from functools import partial


from string import punctuation


import torch.distributions as dist


import scipy


from torch.nn import functional as F


import copy


from scipy.cluster.vq import kmeans2


import matplotlib


import torch.optim


import torch.utils.data


import torch.distributions


from torch.nn import Parameter


import torch.onnx.operators


from torch.autograd import Variable


from math import exp


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import AvgPool1d


from torch.nn import Conv2d


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import spectral_norm


from scipy.io.wavfile import read


from scipy.signal import kaiser


import torch.nn.functional as torch_nn_func


from torch.optim import *


from torch.optim.optimizer import Optimizer


from itertools import chain


from torch.utils.data import ConcatDataset


from torch.utils.tensorboard import SummaryWriter


from functools import wraps


import torch.distributed as dist


from torch.utils.data import DistributedSampler


import types


from scipy.interpolate import interp1d


from torch.nn.parallel import DistributedDataParallel


from torch.nn.parallel.distributed import _find_tensors


from torch.nn import DataParallel


from torch.cuda._utils import _get_device_index


import torch.multiprocessing as mp


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from collections import defaultdict


from sklearn.preprocessing import StandardScaler


mel_n_channels = 40


model_embedding_size = 256


model_hidden_size = 256


model_num_layers = 3


class EmotionEncoder(nn.Module):

    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        self.lstm = nn.LSTM(input_size=mel_n_channels, hidden_size=model_hidden_size, num_layers=model_num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, out_features=model_embedding_size)
        self.relu = torch.nn.ReLU()
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))
        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        return embeds

    def inference(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        return hidden[-1]


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1, eps=1e-05):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=eps)
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


class Flip(nn.Module):

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0))
        return x, logdet

    def store_inverse(self):
        pass


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, share_cond_layers=False):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if gin_channels != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None and not self.share_cond_layers:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


class CouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False, share_cond_layers=False, wn=None):
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
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout, share_cond_layers)
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


DEFAULT_MAX_TARGET_POSITIONS = 2000


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


class GroupNorm1DTBC(nn.GroupNorm):

    def forward(self, input):
        return super(GroupNorm1DTBC, self).forward(input.permute(1, 2, 0)).permute(2, 0, 1)


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
        if self.act == 'swish':
            x = self.swish_fn(x)
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
            elif norm == 'gn':
                self.layer_norm1 = GroupNorm1DTBC(8, c)
            self.self_attn = MultiheadAttention(self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'bn':
            self.layer_norm2 = BatchNorm1dTBC(c)
        elif norm == 'gn':
            self.layer_norm2 = GroupNorm1DTBC(8, c)
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


hparams = {}


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, dropout, kernel_size=None, num_heads=2, norm='ln'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(hidden_size, num_heads, dropout=dropout, attention_dropout=0.0, relu_dropout=dropout, kernel_size=kernel_size if kernel_size is not None else hparams['enc_ffn_kernel_size'], padding=hparams['ffn_padding'], norm=norm, act=hparams['ffn_act'])

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class FFTBlocks(nn.Module):

    def __init__(self, hidden_size, num_layers, ffn_kernel_size=9, dropout=None, num_heads=2, use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout if dropout is not None else hparams['dropout']
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
        return x


class GlowFFTBlocks(FFTBlocks):

    def __init__(self, hidden_size=128, gin_channels=256, num_layers=2, ffn_kernel_size=5, dropout=None, num_heads=4, use_pos_embed=True, use_last_norm=True, norm='ln', use_pos_embed_alpha=True):
        super().__init__(hidden_size, num_layers, ffn_kernel_size, dropout, num_heads, use_pos_embed, use_last_norm, norm, use_pos_embed_alpha)
        self.inp_proj = nn.Conv1d(hidden_size + gin_channels, hidden_size, 1)

    def forward(self, x, x_mask=None, g=None):
        """
        :param x: [B, C_x, T]
        :param x_mask: [B, 1, T]
        :param g: [B, C_g, T]
        :return: [B, C_x, T]
        """
        if g is not None:
            x = self.inp_proj(torch.cat([x, g], 1))
        x = x.transpose(1, 2)
        x = super(GlowFFTBlocks, self).forward(x, x_mask[:, 0] == 0)
        x = x.transpose(1, 2)
        return x


class TransformerCouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale
        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        self.start = start
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.fft_blocks = GlowFFTBlocks(hidden_size=hidden_channels, ffn_kernel_size=3, gin_channels=gin_channels, num_layers=n_layers)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = self.start(x_0) * x_mask
        x = self.fft_blocks(x, x_mask, g)
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
        pass


class Permute(nn.Module):

    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(self.args)


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()
    t = t // n_sqz * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)
    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()
    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)
    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz)
    return x_unsqz * x_mask, x_mask


class FreqFFTCouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale
        hs = hidden_channels
        stride = 8
        self.start = torch.nn.Conv2d(3, hs, kernel_size=stride * 2, stride=stride, padding=stride // 2)
        end = nn.ConvTranspose2d(hs, 2, kernel_size=stride, stride=stride)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = nn.Sequential(nn.Conv2d(hs * 3, hs, 3, 1, 1), nn.ReLU(), nn.GroupNorm(4, hs), nn.Conv2d(hs, hs, 3, 1, 1), end)
        self.fft_v = FFTBlocks(hidden_size=hs, ffn_kernel_size=1, num_layers=n_layers)
        self.fft_h = nn.Sequential(nn.Conv1d(hs, hs, 3, 1, 1), nn.ReLU(), nn.Conv1d(hs, hs, 3, 1, 1))
        self.fft_g = nn.Sequential(nn.Conv1d(gin_channels - 160, hs, kernel_size=stride * 2, stride=stride, padding=stride // 2), Permute(0, 2, 1), FFTBlocks(hidden_size=hs, ffn_kernel_size=1, num_layers=n_layers), Permute(0, 2, 1))

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        g_, _ = unsqueeze(g)
        g_mel = g_[:, :80]
        g_txt = g_[:, 80:]
        g_mel, _ = squeeze(g_mel)
        g_txt, _ = squeeze(g_txt)
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = torch.stack([x_0, g_mel[:, :80], g_mel[:, 80:]], 1)
        x = self.start(x)
        B, C, N_bins, T = x.shape
        x_v = self.fft_v(x.permute(0, 3, 2, 1).reshape(B * T, N_bins, C))
        x_v = x_v.reshape(B, T, N_bins, -1).permute(0, 3, 2, 1)
        x_h = self.fft_h(x.permute(0, 2, 1, 3).reshape(B * N_bins, C, T))
        x_h = x_h.reshape(B, N_bins, -1, T).permute(0, 2, 1, 3)
        x_g = self.fft_g(g_txt)[:, :, None, :].repeat(1, 1, 10, 1)
        x = torch.cat([x_v, x_h, x_g], 1)
        out = self.end(x)
        z_0 = x_0
        m = out[:, 0]
        logs = out[:, 1]
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
        pass


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
            self.flows.append(CouplingBlock(in_channels * n_sqz, hidden_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, n_layers=n_layers, gin_channels=gin_channels * n_sqz, p_dropout=p_dropout, sigmoid_scale=sigmoid_scale, share_cond_layers=share_cond_layers, wn=wn))

    def forward(self, x, x_mask=None, g=None, reverse=False, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)
        if return_hiddens:
            hs = []
        if self.n_sqz > 1:
            x, x_mask_ = squeeze(x, x_mask, self.n_sqz)
            if g is not None:
                g, _ = squeeze(g, x_mask, self.n_sqz)
            x_mask = x_mask_
        if self.share_cond_layers and g is not None:
            g = self.cond_layer(g)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=reverse)
            if return_hiddens:
                hs.append(x)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
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


class GlowV2(nn.Module):

    def __init__(self, in_channels=256, hidden_channels=256, kernel_size=3, dilation_rate=1, n_blocks=8, n_layers=4, p_dropout=0.0, n_split=4, n_split_blocks=3, sigmoid_scale=False, gin_channels=0, share_cond_layers=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_split_blocks = n_split_blocks
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels
        self.cond_layers = nn.ModuleList()
        self.share_cond_layers = share_cond_layers
        self.flows = nn.ModuleList()
        in_channels = in_channels * 2
        for l in range(n_split_blocks):
            blocks = nn.ModuleList()
            self.flows.append(blocks)
            gin_channels = gin_channels * 2
            if gin_channels != 0 and share_cond_layers:
                cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
                self.cond_layers.append(torch.nn.utils.weight_norm(cond_layer, name='weight'))
            for b in range(n_blocks):
                blocks.append(ActNorm(channels=in_channels))
                blocks.append(InvConvNear(channels=in_channels, n_split=n_split))
                blocks.append(CouplingBlock(in_channels, hidden_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, n_layers=n_layers, gin_channels=gin_channels, p_dropout=p_dropout, sigmoid_scale=sigmoid_scale, share_cond_layers=share_cond_layers))

    def forward(self, x=None, x_mask=None, g=None, reverse=False, concat_zs=True, noise_scale=0.66, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
            assert x_mask is not None
            zs = []
            if return_hiddens:
                hs = []
            for i, blocks in enumerate(flows):
                x, x_mask = squeeze(x, x_mask)
                g_ = None
                if g is not None:
                    g, _ = squeeze(g)
                    if self.share_cond_layers:
                        g_ = self.cond_layers[i](g)
                    else:
                        g_ = g
                for layer in blocks:
                    x, logdet = layer(x, x_mask=x_mask, g=g_, reverse=reverse)
                    if return_hiddens:
                        hs.append(x)
                    logdet_tot += logdet
                if i == self.n_split_blocks - 1:
                    zs.append(x)
                else:
                    x, z = torch.chunk(x, 2, 1)
                    zs.append(z)
            if concat_zs:
                zs = [z.reshape(x.shape[0], -1) for z in zs]
                zs = torch.cat(zs, 1)
            if return_hiddens:
                return zs, logdet_tot, hs
            return zs, logdet_tot
        else:
            flows = reversed(self.flows)
            if x is not None:
                assert isinstance(x, list)
                zs = x
            else:
                B, _, T = g.shape
                zs = self.get_prior(B, T, g.device, noise_scale)
            zs_ori = zs
            if g is not None:
                g_, g = g, []
                for i in range(len(self.flows)):
                    g_, _ = squeeze(g_)
                    g.append(self.cond_layers[i](g_) if self.share_cond_layers else g_)
            else:
                g = [None for _ in range(len(self.flows))]
            if x_mask is not None:
                x_masks = []
                for i in range(len(self.flows)):
                    x_mask, _ = squeeze(x_mask)
                    x_masks.append(x_mask)
            else:
                x_masks = [None for _ in range(len(self.flows))]
            x_masks = x_masks[::-1]
            g = g[::-1]
            zs = zs[::-1]
            x = None
            for i, blocks in enumerate(flows):
                x = zs[i] if x is None else torch.cat([x, zs[i]], 1)
                for layer in reversed(blocks):
                    x, logdet = layer(x, x_masks=x_masks[i], g=g[i], reverse=reverse)
                    logdet_tot += logdet
                x, _ = unsqueeze(x)
            return x, logdet_tot, zs_ori

    def store_inverse(self):
        for f in self.modules():
            if hasattr(f, 'store_inverse') and f != self:
                f.store_inverse()

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)

    def get_prior(self, B, T, device, noise_scale=0.66):
        C = 80
        zs = []
        for i in range(len(self.flows)):
            C, T = C, T // 2
            if i == self.n_split_blocks - 1:
                zs.append(torch.randn(B, C * 2, T) * noise_scale)
            else:
                zs.append(torch.randn(B, C, T) * noise_scale)
        return zs


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-06, hidden_size=256):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(hidden_size, 2 * hidden_size)

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x, spk_embed):
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        B = x.size(0)
        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        x_normed = (x - mu) / (sig + 1e-06)
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda
        mu1, sig1 = torch.split(self.affine_layer(spk_embed), self.hidden_size, dim=-1)
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]
        mu_mix = mu1 * lmda + mu2 * (1 - lmda)
        sig_mix = sig1 * lmda + sig2 * (1 - lmda)
        return sig_mix * x_normed + mu_mix


class VQEmbeddingEMA(nn.Module):

    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-05, print_vq_prob=False):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.n_embeddings = n_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.print_vq_prob = print_vq_prob
        self.register_buffer('data_initialized', torch.zeros(1))
        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.zeros(n_embeddings))
        self.register_buffer('ema_weight', self.embedding.clone())

    def encode(self, x):
        B, T, _ = x.shape
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) + torch.sum(x_flat ** 2, dim=1, keepdim=True), x_flat, self.embedding.t(), alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return x_flat, quantized, indices

    def forward(self, x):
        """

        :param x: [B, T, D]
        :return: [B, T, D]
        """
        B, T, _ = x.shape
        M, D = self.embedding.size()
        if self.training and self.data_initialized.item() == 0:
            None
            x_flat = x.detach().reshape(-1, D)
            rp = torch.randperm(x_flat.size(0))
            kd = kmeans2(x_flat[rp].data.cpu().numpy(), self.n_embeddings, minit='points')
            self.embedding.copy_(torch.from_numpy(kd[0]))
            x_flat, quantized, indices = self.encode(x)
            encodings = F.one_hot(indices, M).float()
            self.ema_weight.copy_(torch.matmul(encodings.t(), x_flat))
            self.ema_count.copy_(torch.sum(encodings, dim=0))
        x_flat, quantized, indices = self.encode(x)
        encodings = F.one_hot(indices, M).float()
        indices = indices.reshape(B, T)
        if self.training and self.data_initialized.item() != 0:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        self.data_initialized.fill_(1)
        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction='none')
        nonpadding = (x.abs().sum(-1) > 0).float()
        e_latent_loss = (e_latent_loss.mean(-1) * nonpadding).sum() / nonpadding.sum()
        loss = self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.print_vq_prob:
            None
        return quantized, loss, indices, perplexity


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


class CrossAttenLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttenLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, local_emotion, emotion_key_padding_mask=None, forcing=False):
        if forcing:
            maxlength = src.shape[0]
            k = local_emotion.shape[0] / src.shape[0]
            lengths1 = torch.ceil(torch.tensor([i for i in range(maxlength)]) * k) + 1
            lengths2 = torch.floor(torch.tensor([i for i in range(maxlength)]) * k) - 1
            mask1 = sequence_mask(lengths1, local_emotion.shape[0])
            mask2 = sequence_mask(lengths2, local_emotion.shape[0])
            mask = mask1.float() - mask2.float()
            attn_emo = mask.repeat(src.shape[1], 1, 1)
            src2 = torch.matmul(local_emotion.permute(1, 2, 0), attn_emo.float().transpose(1, 2)).permute(2, 0, 1)
        else:
            src2, attn_emo = self.multihead_attn(src, local_emotion, local_emotion, key_padding_mask=emotion_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_emo


def _make_guided_attention_mask(ilen, rilen, olen, rolen, sigma):
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen, device=rilen.device), torch.arange(olen, device=rolen.device))
    grid_x = grid_x.unsqueeze(0).expand(rilen.size(0), -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(rolen.size(0), -1, -1)
    rilen = rilen.unsqueeze(1).unsqueeze(1)
    rolen = rolen.unsqueeze(1).unsqueeze(1)
    return 1.0 - torch.exp(-(grid_y.float() / rolen - grid_x.float() / rilen) ** 2 / (2 * sigma ** 2))


class ProsodyAligner(nn.Module):

    def __init__(self, num_layers, guided_sigma=0.3, guided_layers=None, norm=None):
        super(ProsodyAligner, self).__init__()
        self.layers = nn.ModuleList([CrossAttenLayer(d_model=hparams['hidden_size'], nhead=2) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.guided_sigma = guided_sigma
        self.guided_layers = guided_layers if guided_layers is not None else num_layers

    def forward(self, src, local_emotion, src_key_padding_mask=None, emotion_key_padding_mask=None, forcing=False):
        output = src
        guided_loss = 0
        attn_emo_list = []
        for i, mod in enumerate(self.layers):
            output, attn_emo = mod(output, local_emotion, emotion_key_padding_mask=emotion_key_padding_mask, forcing=forcing)
            attn_emo_list.append(attn_emo.unsqueeze(1))
            if i < self.guided_layers and src_key_padding_mask is not None:
                s_length = (~src_key_padding_mask).float().sum(-1)
                emo_length = (~emotion_key_padding_mask).float().sum(-1)
                attn_w_emo = _make_guided_attention_mask(src_key_padding_mask.size(-1), s_length, emotion_key_padding_mask.size(-1), emo_length, self.guided_sigma)
                g_loss_emo = attn_emo * attn_w_emo
                non_padding_mask = (~src_key_padding_mask).unsqueeze(-1) & (~emotion_key_padding_mask).unsqueeze(1)
                guided_loss = g_loss_emo[non_padding_mask].mean() + guided_loss
        if self.norm is not None:
            output = self.norm(output)
        return output, guided_loss, attn_emo_list


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

    def __init__(self, channels, out_dims, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True):
        super(ConvBlocks, self).__init__()
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels, kernel_size, d, n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple, dropout=dropout, ln_eps=ln_eps) for d in dilations])
        if norm_type == 'bn':
            norm = nn.BatchNorm1d(channels)
        elif norm_type == 'in':
            norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'gn':
            norm = nn.GroupNorm(8, channels)
        elif norm_type == 'ln':
            norm = LayerNorm(channels, dim=1, eps=ln_eps)
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(channels, out_dims, kernel_size=3, padding=1)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        x = x.transpose(1, 2)
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        return x.transpose(1, 2)


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


class LocalStyleAdaptor(nn.Module):

    def __init__(self, hidden_size, num_vq_codes=64, padding_idx=0):
        super(LocalStyleAdaptor, self).__init__()
        self.encoder = ConvBlocks(80, hidden_size, [1] * 5, 5, dropout=hparams['vae_dropout'])
        self.n_embed = num_vq_codes
        self.vqvae = VQEmbeddingEMA(self.n_embed, hidden_size, commitment_cost=hparams['lambda_commit'])
        self.wavenet = WN(hidden_channels=80, gin_channels=80, kernel_size=3, dilation_rate=1, n_layers=4)
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size

    def forward(self, ref_mels, mel2ph=None, no_vq=False):
        """

        :param ref_mels: [B, T, 80]
        :return: [B, 1, H]
        """
        padding_mask = ref_mels[:, :, 0].eq(self.padding_idx).data
        ref_mels = self.wavenet(ref_mels.transpose(1, 2), x_mask=(~padding_mask).unsqueeze(1).repeat([1, 80, 1])).transpose(1, 2)
        if mel2ph is not None:
            ref_ph, _ = group_hidden_by_segs(ref_mels, mel2ph, torch.max(mel2ph))
        else:
            ref_ph = ref_mels
        prosody = self.encoder(ref_ph)
        if no_vq:
            return prosody
        z, vq_loss, vq_tokens, ppl = self.vqvae(prosody)
        vq_loss = vq_loss.mean()
        return z, vq_loss, ppl


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


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


class Pad(nn.ZeroPad2d):

    def __init__(self, kernel_size, dilation):
        pad_total = dilation * (kernel_size - 1)
        begin = pad_total // 2
        end = pad_total - begin
        super(Pad, self).__init__((begin, end, begin, end))


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""

    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = dilation * (kernel_size - 1)
        if causal:
            super(ZeroTemporalPad, self).__init__((total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((begin, end))


class TextConvEncoder(ConvBlocks):

    def __init__(self, embed_tokens, channels, out_dims, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True):
        super().__init__(channels, out_dims, dilations, kernel_size, norm_type, layers_in_block, c_multiple, dropout, ln_eps, init_weights)
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(channels)

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

    def __init__(self, channels, g_channels, out_dims, dilations, kernel_size, norm_type='ln', layers_in_block=2, c_multiple=2, dropout=0.0, ln_eps=1e-05, init_weights=True, is_BTC=True):
        super().__init__(channels, out_dims, dilations, kernel_size, norm_type, layers_in_block, c_multiple, dropout, ln_eps, init_weights)
        self.g_prenet = nn.Conv1d(g_channels, channels, 3, padding=1)
        self.is_BTC = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, g, x_mask):
        if self.is_BTC:
            x = x.transpose(1, 2)
            g = g.transpose(1, 2)
            x_mask = x_mask.transpose(1, 2)
        x = x + self.g_prenet(g)
        x = x * x_mask
        if not self.is_BTC:
            x = x.transpose(1, 2)
        x = super(ConditionalConvBlocks, self).forward(x)
        if not self.is_BTC:
            x = x.transpose(1, 2)
        return x


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


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


class ConvTBC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)


class DecSALayer(nn.Module):

    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, act='gelu', norm='ln'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        if norm == 'ln':
            self.layer_norm1 = LayerNorm(c)
        elif norm == 'gn':
            self.layer_norm1 = GroupNorm1DTBC(8, c)
        self.self_attn = MultiheadAttention(c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        if norm == 'ln':
            self.layer_norm2 = LayerNorm(c)
        elif norm == 'gn':
            self.layer_norm2 = GroupNorm1DTBC(8, c)
        self.encoder_attn = MultiheadAttention(c, num_heads, encoder_decoder_attention=True, dropout=attention_dropout, bias=False)
        if norm == 'ln':
            self.layer_norm3 = LayerNorm(c)
        elif norm == 'gn':
            self.layer_norm3 = GroupNorm1DTBC(8, c)
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


class ConvBlock(nn.Module):

    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == 'ln':
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == 'wn':
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == 'none':
                pass
            elif self.norm == 'ln':
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):

    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0, strides=None, res=True):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.res = res
        self.in_proj = Linear(idim, n_chans)
        if strides is None:
            strides = [1] * n_layers
        else:
            assert len(strides) == n_layers
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=strides[idx], norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)
        hiddens = []
        for f in self.conv:
            x_ = f(x)
            x = x + x_ if self.res else x_
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)
            return x, hiddens
        return x


class ConvGlobalStacks(nn.Module):

    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0, strides=[2, 2, 2, 2, 2]):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.pooling = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=strides[idx], norm=norm, dropout=dropout))
            self.pooling.append(nn.MaxPool1d(strides[idx]))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)
        for f, p in zip(self.conv, self.pooling):
            x = f(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x.mean(1))
        return x


class ConvLSTMStacks(nn.Module):

    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=3, norm='gn', dropout=0):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=1, norm=norm, dropout=dropout))
        self.lstm = nn.LSTM(n_chans, n_chans, 1, batch_first=True, bidirectional=True)
        self.out_proj = Linear(n_chans * 2, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)
        for f in self.conv:
            x = x + f(x)
        x = x.transpose(1, -1)
        x, _ = self.lstm(x)
        x = self.out_proj(x)
        return x


class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualLayer, self).__init__()
        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=out_channels, affine=True))
        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=out_channels, affine=True))
        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=in_channels, affine=True))

    def forward(self, input):
        """

        :param input: [B, H, T]
        :return: input: [B, H, T]
        """
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)
        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class ConvGLUStacks(nn.Module):

    def __init__(self, idim=80, n_layers=3, n_chans=256, odim=32, kernel_size=5, dropout=0):
        super().__init__()
        self.convs = []
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.convs.append(nn.Sequential(ResidualLayer(n_chans, n_chans, kernel_size, kernel_size // 2), nn.Dropout(dropout)))
        self.convs = nn.Sequential(*self.convs)
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)
        x = self.convs(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)
        return x


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
        return self.dropout(x) + self.dropout(pos_emb)


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


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
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
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding == 'SAME' else (kernel_size - 1, 0), 0), torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        if hparams['dur_loss'] in ['mse', 'huber']:
            odims = 1
        elif hparams['dur_loss'] == 'mog':
            odims = 15
        elif hparams['dur_loss'] == 'crf':
            odims = 32
            self.crf = CRF(odims, batch_first=True)
        self.linear = torch.nn.Linear(n_chans, odims)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))
        xs = xs * (1 - x_masks.float())[:, :, None]
        if is_inference:
            return self.out2dur(xs), xs
        elif hparams['dur_loss'] in ['mse']:
            xs = xs.squeeze(-1)
        return xs

    def out2dur(self, xs):
        if hparams['dur_loss'] in ['mse']:
            xs = xs.squeeze(-1)
            dur = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        elif hparams['dur_loss'] == 'mog':
            return NotImplementedError
        elif hparams['dur_loss'] == 'crf':
            dur = torch.LongTensor(self.crf.decode(xs))
        return dur

    def forward(self, xs, x_masks=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


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
            self.conv += [torch.nn.Sequential(torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2) if padding == 'SAME' else (kernel_size - 1, 0), 0), torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0), torch.nn.ReLU(), LayerNorm(n_chans, dim=1), torch.nn.Dropout(dropout_rate))]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
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
        return xs


class EnergyPredictor(PitchPredictor):
    pass


class FastspeechDecoder(FFTBlocks):

    def __init__(self, hidden_size=None, num_layers=None, kernel_size=None, num_heads=None):
        num_heads = hparams['num_heads'] if num_heads is None else num_heads
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['dec_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads)


FS_DECODERS = {'fft': lambda hp: FastspeechDecoder(hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads'])}


class FastspeechEncoder(FFTBlocks):

    def __init__(self, embed_tokens, hidden_size=None, num_layers=None, kernel_size=None, num_heads=2):
        hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        kernel_size = hparams['enc_ffn_kernel_size'] if kernel_size is None else kernel_size
        num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        super().__init__(hidden_size, num_layers, kernel_size, num_heads=num_heads, use_pos_embed=False)
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        if hparams.get('rel_pos') is not None and hparams['rel_pos']:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS)

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens)
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

    def forward_embedding(self, txt_tokens):
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        if hparams['use_pos_embed']:
            positions = self.embed_positions(txt_tokens)
            x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


FS_ENCODERS = {'fft': lambda hp, embed_tokens, d: FastspeechEncoder(embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'], num_heads=hp['num_heads'])}


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
        """
        assert alpha > 0
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None]
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph


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


def denorm_f0(f0, uv, hparams, pitch_padding=None, min=None, max=None):
    if hparams['pitch_norm'] == 'standard':
        f0 = f0 * hparams['f0_std'] + hparams['f0_mean']
    if hparams['pitch_norm'] == 'log':
        f0 = 2 ** f0
    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and hparams['use_uv']:
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


def norm_f0(f0, uv, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = (f0 - hparams['f0_mean']) / hparams['f0_std']
    if hparams['pitch_norm'] == 'log':
        f0 = torch.log2(f0) if is_torch else np.log2(f0)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    return f0


class FastSpeech2(nn.Module):

    def __init__(self, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = out_dims
        if out_dims is None:
            self.out_dims = hparams['audio_num_mel_bins']
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_id']:
            self.spk_embed_proj = Embedding(hparams['num_spk'] + 1, self.hidden_size)
            if hparams['use_split_spk_id']:
                self.spk_embed_f0 = Embedding(hparams['num_spk'] + 1, self.hidden_size)
                self.spk_embed_dur = Embedding(hparams['num_spk'] + 1, self.hidden_size)
        elif hparams['use_spk_embed']:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['dur_predictor_layers'], dropout_rate=hparams['predictor_dropout'], padding=hparams['ffn_padding'], kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, self.padding_idx)
            if hparams['pitch_type'] == 'cwt':
                h = hparams['cwt_hidden_size']
                cwt_out_dims = 10
                if hparams['use_uv']:
                    cwt_out_dims = cwt_out_dims + 1
                self.cwt_predictor = nn.Sequential(nn.Linear(self.hidden_size, h), PitchPredictor(h, n_chans=predictor_hidden, n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=cwt_out_dims, padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel']))
                self.cwt_stats_layers = nn.Sequential(nn.Linear(self.hidden_size, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
            else:
                self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=2 if hparams['pitch_type'] == 'frame' else 1, padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])
        if hparams['use_energy_embed']:
            self.energy_embed = Embedding(256, self.hidden_size, self.padding_idx)
            self.energy_predictor = EnergyPredictor(self.hidden_size, n_chans=predictor_hidden, n_layers=hparams['predictor_layers'], dropout_rate=hparams['predictor_dropout'], odim=1, padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None, ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False, spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        ret = {}
        encoder_out = self.encoder(txt_tokens)
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        var_embed = 0
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding
        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        if hparams['use_pitch_embed']:
            pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        if skip_decoder:
            return ret
        ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def add_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        dur_input = dur_input.detach() + hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if mel2ph is None:
            dur, xs = self.dur_predictor.inference(dur_input, src_padding)
            ret['dur'] = xs
            ret['dur_choice'] = dur
            mel2ph = self.length_regulator(dur, src_padding).detach()
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_padding)
        ret['mel2ph'] = mel2ph
        return mel2ph

    def add_energy(self, decoder_inp, energy, ret):
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        if energy is None:
            energy = energy_pred
        energy = torch.clamp(energy * 256 // 4, max=255).long()
        energy_embed = self.energy_embed(energy)
        return energy_embed

    def add_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if hparams['pitch_type'] == 'ph':
            pitch_pred_inp = encoder_out.detach() + hparams['predictor_grad'] * (encoder_out - encoder_out.detach())
            pitch_padding = encoder_out.sum().abs() == 0
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            ret['f0_denorm'] = f0_denorm = denorm_f0(f0, None, hparams, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)
            pitch = F.pad(pitch, [1, 0])
            pitch = torch.gather(pitch, 1, mel2ph)
            pitch_embed = self.pitch_embed(pitch)
            return pitch_embed
        decoder_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
        pitch_padding = mel2ph == 0
        if hparams['pitch_type'] == 'cwt':
            pitch_padding = None
            ret['cwt'] = cwt_out = self.cwt_predictor(decoder_inp)
            stats_out = self.cwt_stats_layers(encoder_out[:, 0, :])
            mean = ret['f0_mean'] = stats_out[:, 0]
            std = ret['f0_std'] = stats_out[:, 1]
            cwt_spec = cwt_out[:, :, :10]
            if f0 is None:
                std = std * hparams['cwt_std_scale']
                f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
                if hparams['use_uv']:
                    assert cwt_out.shape[-1] == 11
                    uv = cwt_out[:, :, -1] > 0
        elif hparams['pitch_ar']:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp, f0 if self.training else None)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(decoder_inp)
            if f0 is None:
                f0 = pitch_pred[:, :, 0]
            if hparams['use_uv'] and uv is None:
                uv = pitch_pred[:, :, 1] > 0
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        if pitch_padding is not None:
            f0[pitch_padding] = 0
        pitch = f0_to_coarse(f0_denorm)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def run_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        f0 = cwt2f0(cwt_spec, mean, std, hparams['cwt_scales'])
        f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None, hparams)
        return f0_norm

    def out2mel(self, out):
        return out

    @staticmethod
    def mel_norm(x):
        return (x + 5.5) / (6.3 / 2) - 1

    @staticmethod
    def mel_denorm(x):
        return (x + 1) * (6.3 / 2) - 5.5

    def expand_states(self, h, mel2ph):
        h = F.pad(h, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2ph_)
        return h


class Prenet(nn.Module):

    def __init__(self, in_dim=80, out_dim=256, kernel=5, n_layers=3, strides=None):
        super(Prenet, self).__init__()
        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(nn.Sequential(nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=padding, stride=self.strides[l]), nn.ReLU(), nn.BatchNorm1d(out_dim)))
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """

        :param x: [B, T, 80]
        :return: [L, B, T, H], [B, T, H]
        """
        padding_mask = x.abs().sum(-1).eq(0).data
        nonpadding_mask_TB = 1 - padding_mask.float()[:, None, :]
        x = x.transpose(1, 2)
        hiddens = []
        for i, l in enumerate(self.layers):
            nonpadding_mask_TB = nonpadding_mask_TB[:, :, ::self.strides[i]]
            x = l(x) * nonpadding_mask_TB
        hiddens.append(x)
        hiddens = torch.stack(hiddens, 0)
        hiddens = hiddens.transpose(2, 3)
        x = self.out_proj(x.transpose(1, 2))
        x = x * nonpadding_mask_TB.transpose(1, 2)
        return hiddens, x


class PitchExtractor(nn.Module):

    def __init__(self, n_mel_bins=80, conv_layers=2):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.conv_layers = conv_layers
        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=[1, 1, 1])
        if self.conv_layers > 0:
            self.mel_encoder = ConvStacks(idim=self.hidden_size, n_chans=self.hidden_size, odim=self.hidden_size, n_layers=self.conv_layers)
        self.pitch_predictor = PitchPredictor(self.hidden_size, n_chans=self.predictor_hidden, n_layers=5, dropout_rate=0.1, odim=2, padding=hparams['ffn_padding'], kernel_size=hparams['predictor_kernel'])

    def forward(self, mel_input=None):
        ret = {}
        mel_hidden = self.mel_prenet(mel_input)[1]
        if self.conv_layers > 0:
            mel_hidden = self.mel_encoder(mel_hidden)
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(mel_hidden)
        pitch_padding = mel_input.abs().sum(-1) == 0
        use_uv = hparams['pitch_type'] == 'frame' and hparams['use_uv']
        ret['f0_denorm_pred'] = denorm_f0(pitch_pred[:, :, 0], pitch_pred[:, :, 1] > 0 if use_uv else None, hparams, pitch_padding=pitch_padding)
        return ret


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


class HifiGanGenerator(torch.nn.Module):

    def __init__(self, h, c_out=1):
        super(HifiGanGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h['resblock_kernel_sizes'])
        self.num_upsamples = len(h['upsample_rates'])
        if h['use_pitch_embed']:
            self.harmonic_num = 8
            self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(h['upsample_rates']))
            self.m_source = SourceModuleHnNSF(sampling_rate=h['audio_sample_rate'], harmonic_num=self.harmonic_num)
            self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(80, h['upsample_initial_channel'], 7, 1, padding=3))
        resblock = ResBlock1 if h['resblock'] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes'])):
            c_cur = h['upsample_initial_channel'] // 2 ** (i + 1)
            self.ups.append(weight_norm(ConvTranspose1d(c_cur * 2, c_cur, k, u, padding=(k - u) // 2)))
            if h['use_pitch_embed']:
                if i + 1 < len(h['upsample_rates']):
                    stride_f0 = np.prod(h['upsample_rates'][i + 1:])
                    self.noise_convs.append(Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                else:
                    self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h['upsample_initial_channel'] // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes'])):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, c_out, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0=None):
        if f0 is not None:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if f0 is not None:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
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


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self, in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True, upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_final_nonlinear_activation=True, use_weight_norm=True, use_causal_conv=False):
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
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [torch.nn.ConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2, bias=bias)]
            else:
                layers += [CausalConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // 2 ** (i + 1), dilation=stack_kernel_size ** j, bias=bias, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad, pad_params=pad_params, use_causal_conv=use_causal_conv)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias)]
        else:
            layers += [CausalConv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias, pad=pad, pad_params=pad_params)]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

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

    def __init__(self, in_channels=1, out_channels=1, scales=3, downsample_pooling='AvgPool1d', downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 1, 'count_include_pad': False}, kernel_sizes=[5, 3], channels=16, max_downsample_channels=1024, bias=True, downsample_scales=[4, 4, 4, 4], nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_weight_norm=True):
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

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, bias=True, use_weight_norm=True, use_causal_conv=False, upsample_conditional_features=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}, use_pitch_embed=False):
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
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c=None, pitch=None, **kwargs):
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

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T)

        """
        for f in self.conv_layers:
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


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def forward(self, *inputs, **kwargs):
        self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)
        if torch.is_grad_enabled():
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output


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
        logging.info(f'load module from checkpoint: {last_ckpt_path}')
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
        task = task_cls()
        self.ddp_init(gpu_idx, task)
        self.run_single_process(task)

    def run_single_process(self, task):
        """Sanity check a few things before starting actual training.

        :param task:
        """
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
            task_ref.build_tensorboard(save_dir=self.work_dir, name='lightning_logs', version='lastest')
        else:
            os.makedirs('tmp', exist_ok=True)
            task_ref.build_tensorboard(save_dir='tmp', name='tb_tmp', version='lastest')
        self.logger = task_ref.logger
        try:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()
        except KeyboardInterrupt as e:
            task_ref.on_keyboard_interrupt()

    def run_evaluation(self, test=False):
        eval_results = self.evaluate(self.task, test, tqdm_desc='Valid' if not test else 'test')
        if eval_results is not None and 'tb_log' in eval_results:
            tb_log_output = eval_results['tb_log']
            self.log_metrics_to_tb(tb_log_output)
        if self.proc_rank == 0 and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        task.zero_grad()
        task.eval()
        torch.set_grad_enabled(False)
        task_ref = self.get_task_ref()
        if test:
            ret = task_ref.test_start()
            if ret == 'EXIT':
                return
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
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                train_pbar.set_postfix(**pbar_metrics)
                should_check_val = self.global_step % self.val_check_interval == 0 and not self.fisrt_epoch
                if should_check_val:
                    self.run_evaluation()
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
        if len([k for k in checkpoint['state_dict'].keys() if '.' in k]) > 0:
            task_ref.load_state_dict(checkpoint['state_dict'])
        else:
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
            subprocess.check_call(f'rm -rf "{old_ckpt}"', shell=True)
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

    def ddp_init(self, gpu_idx, task):
        self.proc_rank = gpu_idx
        task.trainer = self
        self.init_ddp_connection(self.proc_rank, self.num_gpus)
        torch.cuda.set_device(gpu_idx)
        self.root_gpu = gpu_idx
        self.task = task

    def configure_ddp(self, task):
        task = DDP(task, device_ids=[self.root_gpu], find_unused_parameters=True)
        if dist.get_rank() != 0 and not self.debug:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
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
        metrics['epoch'] = self.current_epoch
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
                if value is not None and not isinstance(value, list) and fn.__name__ in ['test_dataloader', 'val_dataloader']:
                    value = [value]
            except AttributeError as e:
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)
        return value
    return _get_data_loader


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

    def build_tensorboard(self, save_dir, name, version, **kwargs):
        root_dir = os.path.join(save_dir, name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, 'version_' + str(version))
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    def on_train_start(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

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
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
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

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        None

    def on_train_end(self):
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
        all_losses_meter = {'total_loss': utils.AvgrageMeter()}
        for output in outputs:
            if len(output) == 0 or output is None:
                continue
            if isinstance(output, dict):
                assert 'losses' in output, 'Key "losses" should exist in validation output.'
                n = output.pop('nsamples', 1)
                losses = utils.tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                assert len(output) == 2, 'Validation output should only consist of two elements: (total_loss, losses)'
                n = 1
                total_loss, losses = output
                losses = utils.tensors_to_scalars(losses)
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()
            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
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

    def load_ckpt(self, ckpt_base_dir, current_model_name=None, model_name='model', force=True, strict=True):
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(self.__getattr__(current_model_name), ckpt_base_dir, current_model_name, force, strict)

    @classmethod
    def start(cls):
        os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])
        work_dir = hparams['work_dir']
        trainer = Trainer(work_dir=work_dir, val_check_interval=hparams['val_check_interval'], tb_log_interval=hparams['tb_log_interval'], max_updates=hparams['max_updates'], num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000, accumulate_grad_batches=hparams['accumulate_grad_batches'], print_nan_grads=hparams['print_nan_grads'], resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0), amp=hparams['amp'], monitor_key=hparams['valid_monitor_key'], monitor_mode=hparams['valid_monitor_mode'], num_ckpt_keep=hparams['num_ckpt_keep'], save_best=hparams['save_best'], seed=hparams['seed'], debug=hparams['debug'])
        if not hparams['infer']:
            if len(hparams['save_codes']) > 0:
                t = datetime.now().strftime('%Y%m%d%H%M%S')
                code_dir = f'{work_dir}/codes/{t}'
                subprocess.check_call(f'mkdir -p "{code_dir}"', shell=True)
                for c in hparams['save_codes']:
                    if os.path.exists(c):
                        subprocess.check_call(f'rsync -av --exclude=__pycache__  "{c}" "{code_dir}/"', shell=True)
                None
            trainer.fit(cls)
        else:
            trainer.test(cls)

    def on_keyboard_interrupt(self):
        pass


class BaseVocoder:

    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """
        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError


class RSQRTSchedule(object):

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.warmup_updates = hparams['warmup_updates']
        self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-07)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return self.get_lr()


EOS = '<EOS>'


PAD = '<pad>'


UNK = '<UNK>'


RESERVED_TOKENS = [PAD, EOS, UNK]


NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)


SEG = '|'


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end ids."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids


class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
        self._num_reserved_ids = num_reserved_ids

    @property
    def num_reserved_ids(self):
        return self._num_reserved_ids

    def encode(self, s):
        """Transform a human-readable string into a sequence of int ids.

        The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
        num_reserved_ids) are reserved.

        EOS is not appended.

        Args:
        s: human-readable string to be converted.

        Returns:
        ids: list of integers
        """
        return [(int(w) + self._num_reserved_ids) for w in s.split()]

    def decode(self, ids, strip_extraneous=False):
        """Transform a sequence of int ids into a human-readable string.

        EOS is not expected in ids.

        Args:
        ids: list of integers to be converted.
        strip_extraneous: bool, whether to strip off extraneous tokens
            (EOS and PAD).

        Returns:
        s: human-readable string.
        """
        if strip_extraneous:
            ids = strip_ids(ids, list(range(self._num_reserved_ids or 0)))
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        """Transform a sequence of int ids into a their string versions.

        This method supports transforming individual input/output ids to their
        string versions so that sequence to/from text conversions can be visualized
        in a human readable format.

        Args:
        ids: list of integers to be converted.

        Returns:
        strs: list of human-readable string.
        """
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return [str(d) for d in decoded_ids]

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self, vocab_filename, reverse=False, vocab_list=None, replace_oov=None, num_reserved_ids=NUM_RESERVED_TOKENS):
        """Initialize from a file or list, one token per line.

        Handling of reserved tokens works as follows:
        - When initializing from a list, we add reserved tokens to the vocab.
        - When initializing from a file, we do not add reserved tokens to the vocab.
        - When saving vocab files, we save reserved tokens to the file.

        Args:
            vocab_filename: If not None, the full filename to read vocab from. If this
                is not None, then vocab_list should be None.
            reverse: Boolean indicating if tokens should be reversed during encoding
                and decoding.
            vocab_list: If not None, a list of elements of the vocabulary. If this is
                not None, then vocab_filename should be None.
            replace_oov: If not None, every out-of-vocabulary token seen when
                encoding will be replaced by this string (which must be in vocab).
            num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
        """
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        if vocab_filename:
            self._init_vocab_from_file(vocab_filename)
        else:
            assert vocab_list is not None
            self._init_vocab_from_list(vocab_list)
        self.pad_index = self._token_to_id[PAD]
        self.eos_index = self._token_to_id[EOS]
        self.unk_index = self._token_to_id[UNK]
        self.seg_index = self._token_to_id[SEG] if SEG in self._token_to_id else self.eos_index

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        if self._replace_oov is not None:
            tokens = [(t if t in self._token_to_id else self._replace_oov) for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_eos=False, strip_padding=False):
        if strip_padding and self.pad() in list(ids):
            pad_pos = list(ids).index(self.pad())
            ids = ids[:pad_pos]
        if strip_eos and self.eos() in list(ids):
            eos_pos = list(ids).index(self.eos())
            ids = ids[:eos_pos]
        return ' '.join(self.decode_list(ids))

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def __len__(self):
        return self.vocab_size

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, 'ID_%d' % idx)

    def _init_vocab_from_file(self, filename):
        """Load vocab from a file.

        Args:
        filename: The file to load vocabulary from.
        """
        with open(filename) as f:
            tokens = [token.strip() for token in f.readlines()]

        def token_gen():
            for token in tokens:
                yield token
        self._init_vocab(token_gen(), add_reserved_tokens=False)

    def _init_vocab_from_list(self, vocab_list):
        """Initialize tokens from a list of tokens.

        It is ok if reserved tokens appear in the vocab list. They will be
        removed. The set of tokens in vocab_list should be unique.

        Args:
        vocab_list: A list of tokens.
        """

        def token_gen():
            for token in vocab_list:
                if token not in RESERVED_TOKENS:
                    yield token
        self._init_vocab(token_gen())

    def _init_vocab(self, token_generator, add_reserved_tokens=True):
        """Initialize vocabulary with tokens from token_generator."""
        self._id_to_token = {}
        non_reserved_start_index = 0
        if add_reserved_tokens:
            self._id_to_token.update(enumerate(RESERVED_TOKENS))
            non_reserved_start_index = len(RESERVED_TOKENS)
        self._id_to_token.update(enumerate(token_generator, start=non_reserved_start_index))
        self._token_to_id = dict((v, k) for k, v in six.iteritems(self._id_to_token))

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def seg(self):
        return self.seg_index

    def store_to_file(self, filename):
        """Write vocab file to disk.

        Vocab files have one token per line. The file ends in a newline. Reserved
        tokens are written to the vocab file as well.

        Args:
        filename: Full path of the file to store the vocab to.
        """
        with open(filename, 'w') as f:
            for i in range(len(self._id_to_token)):
                f.write(self._id_to_token[i] + '\n')

    def sil_phonemes(self):
        return [p for p in self._id_to_token.values() if not p[0].isalpha()]


VOCODERS = {}


def get_vocoder_cls(hparams):
    if hparams['vocoder'] in VOCODERS:
        return VOCODERS[hparams['vocoder']]
    else:
        vocoder_cls = hparams['vocoder']
        pkg = '.'.join(vocoder_cls.split('.')[:-1])
        cls_name = vocoder_cls.split('.')[-1]
        vocoder_cls = getattr(importlib.import_module(pkg), cls_name)
        return vocoder_cls


class TtsTask(BaseTask):

    def __init__(self, *args, **kwargs):
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        super().__init__(*args, **kwargs)

    def build_scheduler(self, optimizer):
        return RSQRTSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])
        return optimizer

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None, required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches
        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences, required_batch_size_multiple=required_batch_size_multiple)
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])
        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batches, num_workers=num_workers, pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])
        return optimizer

    def test_start(self):
        self.saving_result_pool = Pool(8)
        self.saving_results_futures = []
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor()
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()

    def test_end(self, outputs):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    def weights_nonzero_speech(self, target):
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)


class BaseConcatDataset(ConcatDataset):

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def _sizes(self):
        if not hasattr(self, 'sizes'):
            self.sizes = list(chain.from_iterable([d._sizes for d in self.datasets]))
        return self.sizes

    def size(self, index):
        return min(self._sizes[index], hparams['max_frames'])

    def num_tokens(self, index):
        return self.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.datasets[0].shuffle:
            indices = np.random.permutation(len(self))
            if self.datasets[0].sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return self.datasets[0].num_workers


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, shuffle):
        super().__init__()
        self.hparams = hparams
        self.shuffle = shuffle
        self.sort_by_len = hparams['sort_by_len']
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self._sizes[index], hparams['max_frames'])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', hparams['ds_workers']))


class IndexedDataset:

    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f'{path}.idx', allow_pickle=True).item()['offsets']
        self.data_file = open(f'{path}.data', 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


class BaseTTSDataset(BaseDataset):

    def __init__(self, prefix, shuffle=False, test_items=None, test_sizes=None, data_dir=None):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.ext_mel2ph = None

        def load_size():
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        if prefix == 'test':
            if test_items is not None:
                self.indexed_ds, self.sizes = test_items, test_sizes
            else:
                load_size()
            if hparams['num_test_samples'] > 0:
                self.avail_idxs = [x for x in range(hparams['num_test_samples']) if x < len(self.sizes)]
                if len(hparams['test_ids']) > 0:
                    self.avail_idxs = hparams['test_ids'] + self.avail_idxs
            else:
                self.avail_idxs = list(range(len(self.sizes)))
        else:
            load_size()
            self.avail_idxs = list(range(len(self.sizes)))
        if hparams['min_frames'] > 0:
            self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
        self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        sample = {'id': index, 'item_name': item['item_name'], 'text': item['txt'], 'txt_token': phone, 'mel': spec, 'mel_nonpadding': spec.abs().sum(-1) > 0}
        if hparams['use_spk_embed']:
            sample['spk_embed'] = torch.Tensor(item['spk_embed'])
        if hparams['use_spk_id']:
            sample['spk_id'] = item['spk_id']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        batch = {'id': id, 'item_name': item_names, 'nsamples': len(samples), 'text': text, 'txt_tokens': txt_tokens, 'txt_lengths': txt_lengths, 'mels': mels, 'mel_lengths': mel_lengths}
        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch


class NoneSchedule(object):

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.step(0)

    def step(self, num_updates):
        self.lr = self.constant_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return self.get_lr()


def get_pitch(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
    f0_min = 80
    f0_max = 750
    if hparams['hop_size'] == 128:
        pad_size = 4
    elif hparams['hop_size'] == 256:
        pad_size = 2
    else:
        assert False
    f0 = parselmouth.Sound(wav_data, hparams['audio_sample_rate']).to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    lpad = pad_size * 2
    rpad = len(mel) - len(f0) - lpad
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


class TTSBaseTask(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = BaseTTSDataset
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_valid_tokens = hparams['max_valid_tokens']
        if self.max_valid_tokens == -1:
            hparams['max_valid_tokens'] = self.max_valid_tokens = self.max_tokens
        self.max_valid_sentences = hparams['max_valid_sentences']
        if self.max_valid_sentences == -1:
            hparams['max_valid_sentences'] = self.max_valid_sentences = self.max_sentences
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}

    @data_loader
    def train_dataloader(self):
        if hparams['train_sets'] != '':
            train_sets = hparams['train_sets'].split('|')
            binary_data_dir = hparams['binary_data_dir']
            file_to_cmp = ['phone_set.json']
            if os.path.exists(f'{binary_data_dir}/word_set.json'):
                file_to_cmp.append('word_set.json')
            if hparams['use_spk_id']:
                file_to_cmp.append('spk_map.json')
            for f in file_to_cmp:
                for ds_name in train_sets:
                    base_file = os.path.join(binary_data_dir, f)
                    ds_file = os.path.join(ds_name, f)
                    assert filecmp.cmp(base_file, ds_file), f'{f} in {ds_name} is not same with that in {binary_data_dir}.'
            train_dataset = BaseConcatDataset([self.dataset_cls(prefix='train', shuffle=True, data_dir=ds_name) for ds_name in train_sets])
        else:
            train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences, endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        self.test_dl = self.build_dataloader(test_dataset, False, self.max_valid_tokens, self.max_valid_sentences, batch_by_size=False)
        return self.test_dl

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None, required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches
        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences, required_batch_size_multiple=required_batch_size_multiple)
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])
        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batches, num_workers=num_workers, pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

    def build_scheduler(self, optimizer):
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer)
        else:
            return NoneSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']), weight_decay=hparams['weight_decay'])
        return optimizer

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def test_start(self):
        self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
        self.saving_results_futures = []
        self.results_id = 0
        self.gen_dir = os.path.join(hparams['work_dir'], f"generated_{self.trainer.global_step}_{hparams['gen_dir_name']}")
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def after_infer(self, predictions, sil_start_frame=0):
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')
        mel_gt = prediction['mels']
        mel2ph_gt = prediction.get('mel2ph')
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel_pred = prediction['outputs']
        mel2ph_pred = prediction.get('mel2ph_pred')
        f0_gt = prediction.get('f0')
        f0_pred = prediction.get('f0_pred')
        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)
        if 'encdec_attn' in prediction:
            encdec_attn = prediction['encdec_attn']
            encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
            txt_lengths = prediction.get('txt_lengths')
            encdec_attn = encdec_attn.T[:txt_lengths, :len(mel_gt)]
        else:
            encdec_attn = None
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'[{self.results_id:06d}][{item_name}][%s]'
        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            if 'encdec_attn' in prediction:
                os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            self.saving_results_futures.append(self.saving_result_pool.apply_async(self.save_result, args=[wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred, encdec_attn]))
            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(self.saving_result_pool.apply_async(self.save_result, args=[wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph_gt]))
                if hparams['save_f0']:
                    import matplotlib.pyplot as plt
                    f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label='$\\hat{f_0}$')
                    plt.plot(f0_gt_, label='$f_0$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)
            None
        self.results_id += 1
        return {'item_name': item_name, 'text': text, 'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()), 'wav_fn_pred': base_fn % 'P', 'wav_fn_gt': base_fn % 'G'}

    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, alignment=None):
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'], norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch(wav_out, mel, hparams)
        f0 = f0 / 10 * (f0 > 0)
        plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(' ')
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = i % 20 + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black', alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)
        if hparams.get('save_mel_npy', False):
            np.save(f'{gen_dir}/npy/{base_fn}', mel)
        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
            decoded_txt = str_phs.split(' ')
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=6)
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close(fig)

    def test_end(self, outputs):
        pd.DataFrame(outputs).to_csv(f'{self.gen_dir}/meta.csv')
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    def weights_nonzero_speech(self, target):
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r


class EndlessDistributedSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [i for _ in range(1000) for i in torch.randperm(len(self.dataset), generator=g).tolist()]
        else:
            indices = [i for _ in range(1000) for i in list(range(len(self.dataset)))]
        indices = indices[:len(indices) // self.num_replicas * self.num_replicas]
        indices = indices[self.rank::self.num_replicas]
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class VocoderDataset(BaseDataset):

    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.hparams = hparams
        self.prefix = prefix
        self.data_dir = hparams['binary_data_dir']
        self.is_infer = prefix == 'test'
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.aux_context_window = hparams['aux_context_window']
        self.hop_size = hparams['hop_size']
        if self.is_infer and hparams['test_input_dir'] != '':
            self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            self.avail_idxs = [i for i, _ in enumerate(self.sizes)]
        elif self.is_infer and hparams['test_mel_dir'] != '':
            self.indexed_ds, self.sizes = self.load_mel_inputs(hparams['test_mel_dir'])
            self.avail_idxs = [i for i, _ in enumerate(self.sizes)]
        else:
            self.indexed_ds = None
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            self.avail_idxs = [idx for idx, s in enumerate(self.sizes) if s - 2 * self.aux_context_window > self.batch_max_frames]
            None
            self.sizes = [s for idx, s in enumerate(self.sizes) if s - 2 * self.aux_context_window > self.batch_max_frames]

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        item = self.indexed_ds[index]
        return item

    def __getitem__(self, index):
        index = self.avail_idxs[index]
        item = self._get_item(index)
        sample = {'id': index, 'item_name': item['item_name'], 'mel': torch.FloatTensor(item['mel']), 'wav': torch.FloatTensor(item['wav'].astype(np.float32))}
        if 'pitch' in item:
            sample['pitch'] = torch.LongTensor(item['pitch'])
            sample['f0'] = torch.FloatTensor(item['f0'])
        if hparams.get('use_spk_embed', False):
            sample['spk_embed'] = torch.Tensor(item['spk_embed'])
        if hparams.get('use_emo_embed', False):
            sample['emo_embed'] = torch.Tensor(item['emo_embed'])
        return sample

    def collater(self, batch):
        if len(batch) == 0:
            return {}
        y_batch, c_batch, p_batch, f0_batch = [], [], [], []
        item_name = []
        have_pitch = 'pitch' in batch[0]
        for idx in range(len(batch)):
            item_name.append(batch[idx]['item_name'])
            x, c = batch[idx]['wav'] if self.hparams['use_wav'] else None, batch[idx]['mel'].squeeze(0)
            if have_pitch:
                p = batch[idx]['pitch']
                f0 = batch[idx]['f0']
            if self.hparams['use_wav']:
                self._assert_ready_for_upsampling(x, c, self.hop_size, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(c) - 2 * self.aux_context_window - 1
                batch_max_steps = batch_max_frames * self.hop_size
                interval_start = self.aux_context_window
                interval_end = len(c) - batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                if self.hparams['use_wav']:
                    y = x[start_step:start_step + batch_max_steps]
                c = c[start_frame - self.aux_context_window:start_frame + self.aux_context_window + batch_max_frames]
                if have_pitch:
                    p = p[start_frame - self.aux_context_window:start_frame + self.aux_context_window + batch_max_frames]
                    f0 = f0[start_frame - self.aux_context_window:start_frame + self.aux_context_window + batch_max_frames]
                if self.hparams['use_wav']:
                    self._assert_ready_for_upsampling(y, c, self.hop_size, self.aux_context_window)
            else:
                None
                continue
            if self.hparams['use_wav']:
                y_batch += [y.reshape(-1, 1)]
            c_batch += [c]
            if have_pitch:
                p_batch += [p]
                f0_batch += [f0]
        if self.hparams['use_wav']:
            y_batch = utils.collate_2d(y_batch, 0).transpose(2, 1)
        c_batch = utils.collate_2d(c_batch, 0).transpose(2, 1)
        if have_pitch:
            p_batch = utils.collate_1d(p_batch, 0)
            f0_batch = utils.collate_1d(f0_batch, 0)
        else:
            p_batch, f0_batch = None, None
        if self.hparams['use_wav']:
            z_batch = torch.randn(y_batch.size())
        else:
            z_batch = []
        return {'z': z_batch, 'mels': c_batch, 'wavs': y_batch, 'pitches': p_batch, 'f0': f0_batch, 'item_name': item_name}

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = sorted(glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/**/*.mp3'))
        sizes = []
        items = []
        binarizer_cls = hparams.get('binarizer_cls', 'data_gen.tts.base_binarizer.BaseBinarizer')
        pkg = '.'.join(binarizer_cls.split('.')[:-1])
        cls_name = binarizer_cls.split('.')[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']
        for wav_fn in inp_wav_paths:
            item_name = wav_fn[len(test_input_dir) + 1:].replace('/', '_')
            item = binarizer_cls.process_item(item_name, wav_fn, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

    def load_mel_inputs(self, test_input_dir, spk_id=0):
        inp_mel_paths = sorted(glob.glob(f'{test_input_dir}/*.npy'))
        sizes = []
        items = []
        binarizer_cls = hparams.get('binarizer_cls', 'data_gen.tts.base_binarizer.BaseBinarizer')
        pkg = '.'.join(binarizer_cls.split('.')[:-1])
        cls_name = binarizer_cls.split('.')[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']
        for mel in inp_mel_paths:
            mel_input = np.load(mel)
            mel_input = torch.FloatTensor(mel_input)
            item_name = mel[len(test_input_dir) + 1:].replace('/', '_')
            item = binarizer_cls.process_mel_item(item_name, mel_input, None, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes


class VocoderBaseTask(BaseTask):

    def __init__(self):
        super(VocoderBaseTask, self).__init__()
        self.max_sentences = hparams['max_sentences']
        self.max_valid_sentences = hparams['max_valid_sentences']
        if self.max_valid_sentences == -1:
            hparams['max_valid_sentences'] = self.max_valid_sentences = self.max_sentences
        self.dataset_cls = VocoderDataset

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls('train', shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_sentences, hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls('valid', shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_valid_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls('test', shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_valid_sentences)

    def build_dataloader(self, dataset, shuffle, max_sentences, endless=False):
        world_size = 1
        rank = 0
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        sampler_cls = DistributedSampler if not endless else EndlessDistributedSampler
        train_sampler = sampler_cls(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return torch.utils.data.DataLoader(dataset=dataset, shuffle=False, collate_fn=dataset.collater, batch_size=max_sentences, num_workers=dataset.num_workers, sampler=train_sampler, pin_memory=True)

    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f"generated_{self.trainer.global_step}_{hparams['gen_dir_name']}")
        os.makedirs(self.gen_dir, exist_ok=True)

    def test_end(self, outputs):
        return {}


class DP(DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        for t in itertools.chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError('module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}'.format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            if self.module.training:
                return self.module.training_step(*inputs[0], **kwargs[0])
            elif self.module.testing:
                return self.module.test_step(*inputs[0], **kwargs[0])
            else:
                return self.module.validation_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

