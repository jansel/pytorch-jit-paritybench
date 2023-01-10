import sys
_module = sys.modules[__name__]
del sys
speech_enhance = _module
audio_zen = _module
acoustics = _module
beamforming = _module
feature = _module
mask = _module
utils = _module
constant = _module
dataset = _module
base_dataset = _module
inferencer = _module
base_inferencer = _module
loss = _module
metrics = _module
model = _module
base_model = _module
module = _module
attention_model = _module
causal_conv = _module
feature_norm = _module
sequence_model = _module
trainer = _module
base_trainer = _module
utils = _module
fullsubnet = _module
dataset_inference = _module
dataset_train = _module
dataset_validation = _module
inferencer = _module
fullsubnet = _module
trainer = _module
fullsubnet_plus = _module
inferencer = _module
fullsubnet_plus = _module
trainer = _module
tools = _module
analyse = _module
calculate_metrics = _module
collect_lst = _module
dns_mos = _module
gen_lst = _module
inference = _module
noisyspeech_synthesizer = _module
resample_dir = _module
train = _module
logger = _module
plot = _module
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


import numpy as np


import torch


import torch.nn as nn


from torch.utils import data


import time


from functools import partial


from torch.nn import functional


from torch.utils.data import DataLoader


import torch.nn.init as init


import matplotlib.pyplot as plt


from torch.cuda.amp import GradScaler


from torch.nn.parallel import DistributedDataParallel


from torch.utils.tensorboard import SummaryWriter


from copy import deepcopy


from functools import reduce


from torch.cuda.amp import autocast


import random


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.utils.data import DistributedSampler


import torch.nn.functional as F


def init_stft_kernel(frame_len, frame_hop, num_fft=None, window='sqrt_hann'):
    if window != 'sqrt_hann':
        raise RuntimeError('Now only support sqrt hanning window in order to make signal perfectly reconstructed')
    if not num_fft:
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    kernel = torch.transpose(kernel, 0, 2) * window
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class CustomSTFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self, frame_len, frame_hop, window='sqrt_hann', num_fft=None):
        super(CustomSTFTBase, self).__init__()
        K = init_stft_kernel(frame_len, frame_hop, num_fft=num_fft, window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError('detect nan in STFT kernels: {:d}'.format(num_nan))

    def extra_repr(self):
        return 'window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}'.format(self.window, self.stride, self.K.requires_grad, self.K.shape)


class CustomSTFT(CustomSTFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(CustomSTFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError('Expect 2D/3D tensor, but got {:d}D'.format(x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        c = torch.nn.functional.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class CustomISTFT(CustomSTFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(CustomISTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError('Expect 2D/3D tensor, but got {:d}D'.format(p.dim()))
        self.check_nan()
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        c = torch.cat([r, i], dim=1)
        s = torch.nn.functional.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError('{} accept 3D tensor as input'.format(self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DirectionalFeatureComputer(nn.Module):

    def __init__(self, n_fft, win_length, hop_length, input_features, mic_pairs, lps_channel, use_cos_IPD=True, use_sin_IPD=False, eps=1e-08):
        super().__init__()
        self.eps = eps
        self.input_features = input_features
        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD
        self.lps_channel = lps_channel
        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += self.num_freqs
            self.lps_layer_norm = ChannelWiseLayerNorm(self.num_freqs)
        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_freqs * self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_freqs * self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  of shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape
        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)
        directional_feature = []
        if 'LPS' in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)
            lps = self.lps_layer_norm(lps)
            directional_feature.append(lps)
        if 'IPD' in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)
            cos_ipd = cos_ipd.view(batch_size, -1, num_frames)
            sin_ipd = sin_ipd.view(batch_size, -1, num_frames)
            directional_feature.append(cos_ipd)
            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)
        directional_feature = torch.cat(directional_feature, dim=1)
        return directional_feature, magnitude, phase, real, imag


class ChannelDirectionalFeatureComputer(nn.Module):

    def __init__(self, n_fft, win_length, hop_length, input_features, mic_pairs, lps_channel, use_cos_IPD=True, use_sin_IPD=False, eps=1e-08):
        super().__init__()
        self.eps = eps
        self.input_features = input_features
        self.stft = CustomSTFT(frame_len=win_length, frame_hop=hop_length, num_fft=n_fft)
        self.num_freqs = n_fft // 2 + 1
        self.mic_pairs = np.array(mic_pairs)
        self.num_mic_pairs = self.mic_pairs.shape[0]
        self.ipd_left = [t[0] for t in mic_pairs]
        self.ipd_right = [t[1] for t in mic_pairs]
        self.use_cos_IPD = use_cos_IPD
        self.use_sin_IPD = use_sin_IPD
        self.lps_channel = lps_channel
        self.directional_feature_dim = 0
        if 'LPS' in self.input_features:
            self.directional_feature_dim += 1
        if 'IPD' in self.input_features:
            self.directional_feature_dim += self.num_mic_pairs
            if self.use_sin_IPD:
                self.directional_feature_dim += self.num_mic_pairs

    def compute_ipd(self, phase):
        """
        Args
            phase: phase of shape [B, M, F, K]
        Returns
            IPD  pf shape [B, I, F, K]
        """
        cos_ipd = torch.cos(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        sin_ipd = torch.sin(phase[:, self.ipd_left] - phase[:, self.ipd_right])
        return cos_ipd, sin_ipd

    def forward(self, y):
        """
        Args:
            y: input mixture waveform with shape [B, M, T]

        Notes:
            B - batch_size
            M - num_channels
            C - num_speakers
            F - num_freqs
            T - seq_len or num_samples
            K - num_frames
            I - IPD feature_size

        Returns:
            Spatial features and directional features of shape [B, ?, K]
        """
        batch_size, num_channels, num_samples = y.shape
        y = y.view(-1, num_samples)
        magnitude, phase, real, imag = self.stft(y)
        _, num_freqs, num_frames = phase.shape
        magnitude = magnitude.view(batch_size, num_channels, num_freqs, num_frames)
        phase = phase.view(batch_size, num_channels, num_freqs, num_frames)
        real = real.view(batch_size, num_channels, num_freqs, num_frames)
        imag = imag.view(batch_size, num_channels, num_freqs, num_frames)
        directional_feature = []
        if 'LPS' in self.input_features:
            lps = torch.log(magnitude[:, self.lps_channel, ...] ** 2 + self.eps)
            lps = lps[:, None, ...]
            directional_feature.append(lps)
        if 'IPD' in self.input_features:
            cos_ipd, sin_ipd = self.compute_ipd(phase)
            directional_feature.append(cos_ipd)
            if self.use_sin_IPD:
                directional_feature.append(sin_ipd)
        directional_feature = torch.cat(directional_feature, dim=1)
        return directional_feature, magnitude, phase, real, imag


EPSILON = np.finfo(np.float32).eps


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f'The dim of input is {input.dim()}. It should be four dim.'
        batch_size, num_channels, num_freqs, num_frames = input.size()
        if num_neighbor < 1:
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)
        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1
        output = functional.pad(output, [0, 0, num_neighbor, num_neighbor], mode='reflect')
        output = functional.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f'n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}'
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()
        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []
        for idx in range(3):
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)
            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))
        return torch.cat(final_selected, dim=0)

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        与 forgetting norm相同，但使用拼接后模型的中间频带来计算均值
        效果不好
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)
            else:
                mu = alpha * mu + (1 - alpha) * input[:, n_freqs // 2 - 1, idx].reshape(batch_size, 1)
            mu_list.append(mu)
        mu = torch.stack(mu_list, dim=-1)
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        输入为三维，通过不断估计邻近的均值来作为当前 norm 时的均值

        Args:
            input: [B, F, T]
            sample_length_in_training: 训练时的长度，用于计算平滑因子

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)
                mu = alpha * mu + (1 - alpha) * current_frame_mu
            mu_list.append(mu)
        mu = torch.stack(mu_list, dim=-1)
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)
        step_sum = torch.sum(input, dim=1)
        cumulative_sum = torch.cumsum(step_sum, dim=-1)
        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)
        entry_count = entry_count.expand_as(cumulative_sum)
        cum_mean = cumulative_sum / entry_count
        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)
        cum_mean[:, :, :sample_length_in_training] = initial_mu
        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        normed = input / (mu + 1e-05)
        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)
        step_sum = torch.sum(input, dim=1)
        cumulative_sum = torch.cumsum(step_sum, dim=-1)
        entry_count = torch.arange(num_freqs, num_freqs * num_frames + 1, num_freqs, dtype=input.dtype, device=input.device)
        entry_count = entry_count.reshape(1, num_frames)
        entry_count = entry_count.expand_as(cumulative_sum)
        cumulative_mean = cumulative_sum / entry_count
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        normed = input / (cumulative_mean + EPSILON)
        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)
        normed = (input - mu) / (std + 1e-05)
        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)
        step_sum = torch.sum(input, dim=1)
        step_pow_sum = torch.sum(torch.square(input), dim=1)
        cumulative_sum = torch.cumsum(step_sum, dim=-1)
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)
        entry_count = torch.arange(num_freqs, num_freqs * num_frames + 1, num_freqs, dtype=input.dtype, device=input.device)
        entry_count = entry_count.reshape(1, num_frames)
        entry_count = entry_count.expand_as(cumulative_sum)
        cumulative_mean = cumulative_sum / entry_count
        cumulative_var = (cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(2)
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)
        normed = (input - cumulative_mean) / cumulative_std
        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == 'offline_laplace_norm':
            norm = self.offline_laplace_norm
        elif norm_type == 'cumulative_laplace_norm':
            norm = self.cumulative_laplace_norm
        elif norm_type == 'offline_gaussian_norm':
            norm = self.offline_gaussian_norm
        elif norm_type == 'cumulative_layer_norm':
            norm = self.cumulative_layer_norm
        else:
            raise NotImplementedError('You must set up a type of Norm. e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.')
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        squeeze_tensor = input_tensor.mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels // subband_num), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.middleConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels // subband_num), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.largeConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels // subband_num), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelTimeSenseSEWeightLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseSEWeightLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.middleConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.largeConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels), nn.AdaptiveAvgPool1d(1), nn.ReLU(inplace=True))
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor, fc_out_2.view(a, b, 1)


class ChannelDeepTimeSenseSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelDeepTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels), nn.ReLU(inplace=True), nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.middleConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels), nn.ReLU(inplace=True), nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.largeConv1d = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels), nn.ReLU(inplace=True), nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1))
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class SelfAttentionlayer(nn.Module):
    """
    Easy self attention.
    """

    def __init__(self, amp_dim=257, att_dim=257):
        super(SelfAttentionlayer, self).__init__()
        self.d_k = amp_dim
        self.q_linear = nn.Linear(amp_dim, att_dim)
        self.k_linear = nn.Linear(amp_dim, att_dim)
        self.v_linear = nn.Linear(amp_dim, att_dim)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(att_dim, amp_dim)

    def forward(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        output = self.attention(q, k, v)
        output = self.out(output)
        return output

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.sigmoid(scores)
        output = torch.matmul(scores, v)
        return output


class Conv_Attention_Block(nn.Module):

    def __init__(self, num_channels, kersize=[3, 5, 10]):
        """
        Args:
            num_channels: No of input channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.conv1d = nn.Conv1d(num_channels, num_channels, kernel_size=kersize, groups=num_channels)
        self.attention = SelfAttentionlayer(amp_dim=num_channels, att_dim=num_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.active_funtion = nn.ReLU(inplace=True)

    def forward(self, input):
        input = self.conv1d(input).permute(0, 2, 1)
        input = self.attention(input, input, input)
        output = self.active_funtion(self.avgpool(input.permute(0, 2, 1)))
        return output


class ChannelTimeSenseAttentionSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10]):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelTimeSenseAttentionSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.smallConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[0])
        self.middleConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[1])
        self.largeConv1d = Conv_Attention_Block(num_channels=num_channels, kersize=kersize[2])
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelCBAMLayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelCBAMLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, T)
        :return: output tensor
        """
        mean_squeeze_tensor = input_tensor.mean(dim=2)
        max_squeeze_tensor, _ = torch.max(input_tensor, dim=2)
        mean_fc_out_1 = self.relu(self.fc1(mean_squeeze_tensor))
        max_fc_out_1 = self.relu(self.fc1(max_squeeze_tensor))
        fc_out_1 = mean_fc_out_1 + max_fc_out_1
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = mean_squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor


class ChannelECAlayer(nn.Module):
    """
     a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ChannelECAlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CausalConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, encoder_activate_function, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 2), stride=(2, 1), padding=(0, 1), **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = getattr(nn, encoder_activate_function)()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 2), stride=(2, 1), output_padding=output_padding)
        self.norm = nn.BatchNorm2d(out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.

        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]
        x = self.norm(x)
        x = self.activation(x)
        return x


class TCNBlock(nn.Module):

    def __init__(self, in_channels=257, hidden_channel=512, out_channels=257, kernel_size=3, dilation=1, use_skip_connection=True, causal=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        padding = dilation * (kernel_size - 1) // 2 if not causal else dilation * (kernel_size - 1)
        self.depthwise_conv = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1, groups=hidden_channel, padding=padding, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


class STCNBlock(nn.Module):

    def __init__(self, in_channels=257, hidden_channel=512, out_channels=257, kernel_size=3, dilation=1, use_skip_connection=True, causal=False):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channel, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        padding = dilation * (kernel_size - 1) // 2 if not causal else dilation * (kernel_size - 1)
        self.depthwise_conv = nn.Conv1d(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=1, groups=hidden_channel, padding=padding, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.sconv = nn.Conv1d(hidden_channel, out_channels, 1)
        self.causal = causal
        self.padding = padding
        self.use_skip_connection = use_skip_connection

    def forward(self, x):
        """
            x: [channels, T]
        """
        if self.use_skip_connection:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return x + output
        else:
            y = self.conv1x1(x)
            y = self.norm1(self.prelu1(y))
            y = self.depthwise_conv(y)
            if self.causal:
                y = y[:, :, :-self.padding]
            y = self.norm2(self.prelu2(y))
            output = self.sconv(y)
            return output


class CumulativeMagSpectralNorm(nn.Module):

    def __init__(self, cumulative=False, use_mid_freq_mu=False):
        """

        Args:
            cumulative: 是否采用累积的方式计算 mu
            use_mid_freq_mu: 仅采用中心频率的 mu 来代替全局 mu

        Notes:
            先算均值再累加 等同于 先累加再算均值

        """
        super().__init__()
        self.eps = 1e-06
        self.cumulative = cumulative
        self.use_mid_freq_mu = use_mid_freq_mu

    def forward(self, input):
        assert input.ndim == 4, f'{self.__name__} only support 4D input.'
        batch_size, n_channels, n_freqs, n_frames = input.size()
        device = input.device
        data_type = input.dtype
        input = input.reshape(batch_size * n_channels, n_freqs, n_frames)
        if self.use_mid_freq_mu:
            step_sum = input[:, int(n_freqs // 2 - 1), :]
        else:
            step_sum = torch.mean(input, dim=1)
        if self.cumulative:
            cumulative_sum = torch.cumsum(step_sum, dim=-1)
            entry_count = torch.arange(1, n_frames + 1, dtype=data_type, device=device)
            entry_count = entry_count.reshape(1, n_frames)
            entry_count = entry_count.expand_as(cumulative_sum)
            mu = cumulative_sum / entry_count
            mu = mu.reshape(batch_size * n_channels, 1, n_frames)
        else:
            mu = torch.mean(step_sum, dim=-1)
            mu = mu.reshape(batch_size * n_channels, 1, 1)
        input_normed = input / (mu + self.eps)
        input_normed = input_normed.reshape(batch_size, n_channels, n_freqs, n_frames)
        return input_normed


class SequenceModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, sequence_model='GRU', output_activate_function='Tanh'):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        self.sequence_model_type = sequence_model
        if self.sequence_model_type == 'LSTM':
            self.sequence_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.sequence_model_type == 'GRU':
            self.sequence_model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.sequence_model_type == 'TCN':
            self.sequence_model = nn.Sequential(TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=1), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=2), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=5), TCNBlock(in_channels=input_size, out_channels=input_size, dilation=9), nn.ReLU())
        elif self.sequence_model_type == 'TCN-subband':
            self.sequence_model = nn.Sequential(TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=9), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=1), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=2), TCNBlock(in_channels=input_size, hidden_channel=hidden_size, out_channels=input_size, dilation=5), TCNBlock(in_channels=input_size, hidden_channel=384, out_channels=input_size, dilation=9), nn.ReLU())
        else:
            raise NotImplementedError(f'Not implemented {sequence_model}')
        if self.sequence_model_type == 'LSTM' or self.sequence_model_type == 'GRU':
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)
        elif self.sequence_model_type == 'TCN':
            self.fc_output_layer = nn.Linear(input_size, output_size)
        else:
            self.fc_output_layer = nn.Linear(input_size, output_size)
        if output_activate_function:
            if output_activate_function == 'Tanh':
                self.activate_function = nn.Tanh()
            elif output_activate_function == 'ReLU':
                self.activate_function = nn.ReLU()
            elif output_activate_function == 'ReLU6':
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f'Not implemented activation function {self.activate_function}')
        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        if self.sequence_model_type == 'TCN' or self.sequence_model_type == 'TCN-subband':
            x = self.sequence_model(x)
            o = self.fc_output_layer(x.permute(0, 2, 1))
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1)
            return o
        else:
            self.sequence_model.flatten_parameters()
            x = x.permute(0, 2, 1).contiguous()
            o, _ = self.sequence_model(x)
            o = self.fc_output_layer(o)
            if self.output_activate_function:
                o = self.activate_function(o)
            o = o.permute(0, 2, 1).contiguous()
        return o


class Complex_SequenceModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, sequence_model='GRU', output_activate_function='Tanh'):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        if sequence_model == 'LSTM':
            self.real_sequence_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            self.imag_sequence_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif sequence_model == 'GRU':
            self.real_sequence_model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            self.imag_sequence_model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise NotImplementedError(f'Not implemented {sequence_model}')
        if bidirectional:
            self.real_fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            self.imag_fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.real_fc_output_layer = nn.Linear(hidden_size, output_size)
            self.imag_fc_output_layer = nn.Linear(hidden_size, output_size)
        if output_activate_function:
            if output_activate_function == 'Tanh':
                self.activate_function = nn.Tanh()
            elif output_activate_function == 'ReLU':
                self.activate_function = nn.ReLU()
            elif output_activate_function == 'ReLU6':
                self.activate_function = nn.ReLU6()
            elif output_activate_function == 'PReLU':
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(f'Not implemented activation function {self.activate_function}')
        self.output_activate_function = output_activate_function

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        self.real_sequence_model.flatten_parameters()
        self.imag_sequence_model.flatten_parameters()
        real, imag = torch.chunk(x, 2, 1)
        real = real.permute(0, 2, 1).contiguous()
        imag = imag.permute(0, 2, 1).contiguous()
        r2r = self.real_sequence_model(real)[0]
        r2i = self.imag_sequence_model(real)[0]
        i2r = self.real_sequence_model(imag)[0]
        i2i = self.imag_sequence_model(imag)[0]
        real_out = r2r - i2i
        imag_out = i2r + r2i
        real_out = self.real_fc_output_layer(real_out)
        imag_out = self.imag_fc_output_layer(imag_out)
        if self.output_activate_function:
            real_out = self.activate_function(real_out)
            imag_out = self.activate_function(imag_out)
        real_out = real_out.permute(0, 2, 1).contiguous()
        imag_out = imag_out.permute(0, 2, 1).contiguous()
        o = torch.cat([real_out, imag_out], 1)
        return o


def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.

    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f'Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups.'
    if num_groups <= 1:
        return input
    if num_freqs % num_groups != 0:
        input = input[..., :num_freqs - num_freqs % num_groups, :]
        num_freqs = input.shape[2]
    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)
        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)
        output.append(selected)
    return torch.cat(output, dim=0)


class Model(BaseModel):

    def __init__(self, num_freqs, look_ahead, sequence_model, fb_num_neighbors, sb_num_neighbors, fb_output_activate_function, sb_output_activate_function, fb_model_hidden_size, sb_model_hidden_size, norm_type='offline_laplace_norm', num_groups_in_drop_band=2, weight_init=True):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ('GRU', 'LSTM'), f'{self.__class__.__name__} only support GRU and LSTM.'
        self.fb_model = SequenceModel(input_size=num_freqs, output_size=num_freqs, hidden_size=fb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model=sequence_model, output_activate_function=fb_output_activate_function)
        self.sb_model = SequenceModel(input_size=sb_num_neighbors * 2 + 1 + (fb_num_neighbors * 2 + 1), output_size=2, hidden_size=sb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model=sequence_model, output_activate_function=sb_output_activate_function)
        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f'{self.__class__.__name__} takes the mag feature as inputs.'
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)
        sb_input = sb_input.reshape(batch_size * num_freqs, self.sb_num_neighbors * 2 + 1 + (self.fb_num_neighbors * 2 + 1), num_frames)
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()
        output = sb_mask[:, :, :, self.look_ahead:]
        return output


class FullSubNet_Plus(BaseModel):

    def __init__(self, num_freqs, look_ahead, sequence_model, fb_num_neighbors, sb_num_neighbors, fb_output_activate_function, sb_output_activate_function, fb_model_hidden_size, sb_model_hidden_size, channel_attention_model='SE', norm_type='offline_laplace_norm', num_groups_in_drop_band=2, output_size=2, subband_num=1, kersize=[3, 5, 10], weight_init=True):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            sb_num_neighbors: Number of the neighbor frequencies in each side
            look_ahead: Number of use of the future frames
            sequence_model: Chose one sequence model as the basic model (GRU, LSTM)
        """
        super().__init__()
        assert sequence_model in ('GRU', 'LSTM', 'TCN'), f'{self.__class__.__name__} only support GRU, LSTM and TCN.'
        if subband_num == 1:
            self.num_channels = num_freqs
        else:
            self.num_channels = num_freqs // subband_num + 1
        if channel_attention_model:
            if channel_attention_model == 'SE':
                self.channel_attention = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelSELayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelSELayer(num_channels=self.num_channels)
            elif channel_attention_model == 'ECA':
                self.channel_attention = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_real = ChannelECAlayer(channel=self.num_channels)
                self.channel_attention_imag = ChannelECAlayer(channel=self.num_channels)
            elif channel_attention_model == 'CBAM':
                self.channel_attention = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_real = ChannelCBAMLayer(num_channels=self.num_channels)
                self.channel_attention_imag = ChannelCBAMLayer(num_channels=self.num_channels)
            elif channel_attention_model == 'TSSE':
                self.channel_attention = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_real = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
                self.channel_attention_imag = ChannelTimeSenseSELayer(num_channels=self.num_channels, kersize=kersize)
            else:
                raise NotImplementedError(f'Not implemented channel attention model {self.channel_attention}')
        self.fb_model = SequenceModel(input_size=num_freqs, output_size=num_freqs, hidden_size=fb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model='TCN', output_activate_function=fb_output_activate_function)
        self.fb_model_real = SequenceModel(input_size=num_freqs, output_size=num_freqs, hidden_size=fb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model='TCN', output_activate_function=fb_output_activate_function)
        self.fb_model_imag = SequenceModel(input_size=num_freqs, output_size=num_freqs, hidden_size=fb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model='TCN', output_activate_function=fb_output_activate_function)
        self.sb_model = SequenceModel(input_size=sb_num_neighbors * 2 + 1 + 3 * (fb_num_neighbors * 2 + 1), output_size=output_size, hidden_size=sb_model_hidden_size, num_layers=2, bidirectional=False, sequence_model=sequence_model, output_activate_function=sb_output_activate_function)
        self.subband_num = subband_num
        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        self.output_size = output_size
        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            noisy_real: [B, 1, F, T]
            noisy_imag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])
        noisy_real = functional.pad(noisy_real, [0, self.look_ahead])
        noisy_imag = functional.pad(noisy_imag, [0, self.look_ahead])
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f'{self.__class__.__name__} takes the mag feature as inputs.'
        if self.subband_num == 1:
            fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
            fb_input = self.channel_attention(fb_input)
        else:
            pad_num = self.subband_num - num_freqs % self.subband_num
            fb_input = functional.pad(self.norm(noisy_mag), [0, 0, 0, pad_num], mode='reflect')
            fb_input = fb_input.reshape(batch_size, (num_freqs + pad_num) // self.subband_num, num_frames * self.subband_num)
            fb_input = self.channel_attention(fb_input)
            fb_input = fb_input.reshape(batch_size, num_channels * (num_freqs + pad_num), num_frames)[:, :num_freqs, :]
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)
        fbr_input = self.norm(noisy_real).reshape(batch_size, num_channels * num_freqs, num_frames)
        fbr_input = self.channel_attention_real(fbr_input)
        fbr_output = self.fb_model_real(fbr_input).reshape(batch_size, 1, num_freqs, num_frames)
        fbi_input = self.norm(noisy_imag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fbi_input = self.channel_attention_imag(fbi_input)
        fbi_output = self.fb_model_imag(fbi_input).reshape(batch_size, 1, num_freqs, num_frames)
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)
        fbr_output_unfolded = self.unfold(fbr_output, num_neighbor=self.fb_num_neighbors)
        fbr_output_unfolded = fbr_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)
        fbi_output_unfolded = self.unfold(fbi_output, num_neighbor=self.fb_num_neighbors)
        fbi_output_unfolded = fbi_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)
        noisy_mag_unfolded = self.unfold(fb_input.reshape(batch_size, 1, num_freqs, num_frames), num_neighbor=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded, fbr_output_unfolded, fbi_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)
        sb_input = sb_input.reshape(batch_size * num_freqs, self.sb_num_neighbors * 2 + 1 + 3 * (self.fb_num_neighbors * 2 + 1), num_frames)
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, self.output_size, num_frames).permute(0, 2, 1, 3).contiguous()
        output = sb_mask[:, :, :, self.look_ahead:]
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausalTransConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelCBAMLayer,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ChannelECAlayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ChannelSELayer,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ChannelWiseLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CumulativeMagSpectralNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_hit_thusz_RookieCJ_FullSubNet_plus(_paritybench_base):
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

