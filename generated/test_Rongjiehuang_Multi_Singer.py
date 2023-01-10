import sys
_module = sys.modules[__name__]
del sys
datasets = _module
audio_mel_dataset = _module
collater = _module
launch = _module
audio = _module
config = _module
data_objects = _module
random_cycler = _module
speaker = _module
speaker_batch = _module
speaker_verification_dataset = _module
utterance = _module
inference = _module
model = _module
params_data = _module
params_model = _module
plot_umap = _module
preprocess = _module
train = _module
visualizations = _module
audio_preprocess = _module
audio_world_process = _module
inference = _module
layers = _module
causal_conv = _module
pqmf = _module
residual_block = _module
residual_stack = _module
tf_layers = _module
upsample = _module
losses = _module
stft_loss = _module
Discriminator = _module
Generator = _module
models = _module
optimizers = _module
radam = _module
preprocess = _module
train = _module
utils = _module
display = _module
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


import logging


import numpy as np


from torch.utils.data import Dataset


import torch


from scipy.ndimage.morphology import binary_dilation


from typing import Optional


from typing import Union


import torchaudio


from torch.utils.data import DataLoader


from matplotlib import cm


import matplotlib.pyplot as plt


from scipy.interpolate import interp1d


from sklearn.metrics import roc_curve


from torch.nn.utils import clip_grad_norm_


from scipy.optimize import brentq


from torch import nn


import time


import torch.nn.functional as F


from scipy.signal import kaiser


import math


from torch.optim import *


from torch.optim.optimizer import Optimizer


import random


from collections import defaultdict


from sklearn.metrics.pairwise import cosine_similarity


import matplotlib


mel_n_channels = 80


model_embedding_size = 256


model_hidden_size = 256


model_num_layers = 3


class SpeakerEncoder(nn.Module):

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

    def forward_perceptual(self, utterances, hidden_init=None):
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
        return embeds_raw

    def forward_perceptual2(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        return hidden

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)
        centroids_excl = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroids_excl /= utterances_per_speaker - 1
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker, speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long()
        loss = self.loss_fn(sim_matrix, target)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return loss, eer


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


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
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

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.

        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

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
            h_analysis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - taps / 2) + (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - taps / 2) - (-1) ** k * np.pi / 4)
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


class ResidualEmbeddingBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self, kernel_size=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, embed_channels=256, dropout=0.0, dilation=1, bias=True, use_causal_conv=False):
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
        super(ResidualEmbeddingBlock, self).__init__()
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
        if aux_channels > 0:
            self.conv1x1_embed = Conv1d1x1(embed_channels, gate_channels, bias=False)
        else:
            self.conv1x1_embed = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c, embed):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).
            embed (Tensor): Local conditioning auxiliary tensor (B, embed_channels, T).

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
        if c is not None and embed is not None:
            assert self.conv1x1_aux is not None
            assert self.conv1x1_embed is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            embed = self.conv1x1_embed(embed)
            ea, eb = embed.split(embed.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca + ea, xb + cb + eb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, s


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
            Tensor: Output tensor (B, channels, T).

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

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
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
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
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


class Unconditional_Discriminator(torch.nn.Module):
    """Unconditional Discriminator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=10, conv_channels=64, dilation_factor=1, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, bias=True, use_weight_norm=True):
        """Initialize Unconditional Discriminator module.

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
        super(Unconditional_Discriminator, self).__init__()
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


class SingerConditional_Discriminator(torch.nn.Module):
    """SingerConditional Discriminator module."""

    def __init__(self, in_channels=1, out_channels=256, kernel_sizes=[5, 3], channels=16, max_downsample_channels=1024, bias=True, downsample_scales=[4, 4, 4, 4], nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, model_hidden_size=256, model_num_layers=3):
        """Initialize SingerConditional Discriminator module.
        """
        super(SingerConditional_Discriminator, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=model_hidden_size, hidden_size=model_hidden_size, num_layers=model_num_layers)
        self.linear = torch.nn.Linear(model_hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.layers += [torch.nn.Sequential(getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params), torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_size=downsample_scale * 10 + 1, stride=downsample_scale, padding=downsample_scale * 5, groups=in_chs // 4, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.layers += [torch.nn.Conv1d(out_chs, out_channels, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2, bias=bias)]

    def forward(self, x, embed):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            embed (Tensor): Local conditioning auxiliary features (B, C ,1).

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        for f in self.layers:
            x = f(x)
        frames_batch = x.permute(2, 0, 1)
        output, (hn, cn) = self.lstm(frames_batch)
        p = output[-1] + embed.squeeze(2)
        p = self.relu(self.linear(p))
        return p


class Generator1(torch.nn.Module):
    """Generator1 module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, bias=True, use_weight_norm=True, use_causal_conv=False, upsample_conditional_features=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}):
        """Initialize Generator module.

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
        super(Generator1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
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
            self.upsample_factor = np.prod(upsample_params['upsample_scales'])
        else:
            self.upsample_net = None
            self.upsample_factor = 1
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=bias, use_causal_conv=use_causal_conv)
            self.conv_layers += [conv]
        self.last_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, skip_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, in_channels, bias=True)])
        self.pqmf_conv1 = Conv1d(1, 128, kernel_size, 1, padding=3)
        self.pqmf_conv2 = Conv1d(128, 1, kernel_size, 1, padding=3)
        if use_weight_norm:
            self.apply_weight_norm()
        self.pqmf = PQMF()

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) * 4 == x.size(-1)
        x = self.pqmf.analysis(x)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = self.pqmf.synthesis(x)
        x = self.pqmf_conv1(x)
        x = self.pqmf_conv2(x)
        return x

    def inference(self, c=None, x=None):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor * 4)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)

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


class Generator2(torch.nn.Module):
    """Generator2 module."""

    def __init__(self, in_channels=1, out_channels=4, kernel_sizes=[7, 5], layers=[16, 15], stacks=[8, 5], residual_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, bias=True, use_weight_norm=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}):
        super(Generator2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.stacks = stacks
        self.pqmf = PQMF(out_channels)
        self.aux_context_window = aux_context_window
        self.low_first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)
        self.up_first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)
        upsample_params.update({'aux_channels': aux_channels, 'aux_context_window': aux_context_window})
        self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        self.low_conv_layers = torch.nn.ModuleList()
        for layer in range(layers[0]):
            dilation = 2 ** (layer % stacks[0])
            conv = ResidualBlock(kernel_size=kernel_sizes[0], residual_channels=residual_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=bias)
            self.low_conv_layers += [conv]
        self.up_conv_layers = torch.nn.ModuleList()
        for layer in range(layers[1]):
            dilation = 2 ** (layer % stacks[1])
            conv = ResidualBlock(kernel_size=kernel_sizes[1], residual_channels=residual_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=bias)
            self.up_conv_layers += [conv]
        self.last_low_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(residual_channels, residual_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(residual_channels, out_channels // 2, bias=True)])
        self.last_up_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(residual_channels, residual_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(residual_channels, out_channels // 2, bias=True)])
        self.upsample_factor = np.prod(upsample_params['upsample_scales'])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)
        x1 = self.low_first_conv(x)
        x2 = self.up_first_conv(x)
        skips = 0
        for f in self.low_conv_layers:
            x1, h = f(x1, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.low_conv_layers))
        x1 = skips
        for f in self.last_low_conv_layers:
            x1 = f(x1)
        skips = 0
        for f in self.up_conv_layers:
            x2, h = f(x2, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.up_conv_layers))
        x2 = skips
        for f in self.last_up_conv_layers:
            x2 = f(x2)
        w = torch.cat((x1, x2), dim=1)
        res = self.pqmf.synthesis(w)
        return res

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

