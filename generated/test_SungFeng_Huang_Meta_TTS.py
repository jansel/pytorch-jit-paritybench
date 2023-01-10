import sys
_module = sys.modules[__name__]
del sys
audio = _module
audio_processing = _module
stft = _module
tools = _module
config = _module
comet = _module
dataset = _module
centroid_similarity = _module
compute_mos = _module
main = _module
merge_image = _module
pair_similarity = _module
similarity_plot = _module
speaker_verification = _module
visualize = _module
wavs_to_dvector = _module
callbacks = _module
progressbar = _module
saver = _module
utils = _module
collate = _module
datamodules = _module
base_datamodule = _module
baseline_datamodule = _module
define = _module
meta_datamodule = _module
utils = _module
model = _module
fastspeech2 = _module
loss = _module
modules = _module
optimizer = _module
phoneme_embedding = _module
speaker_encoder = _module
optimizer = _module
sampler = _module
scheduler = _module
systems = _module
base_adaptor = _module
baseline = _module
imaml = _module
meta = _module
system = _module
utils = _module
utils = _module
main = _module
prepare_align = _module
preprocess = _module
libritts = _module
preprocessor = _module
vctk = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
pinyin = _module
symbols = _module
Constants = _module
Layers = _module
Models = _module
Modules = _module
SubLayers = _module
transformer = _module
model = _module
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


from scipy.io.wavfile import write


import math


from torch.utils.data import Dataset


import torchaudio


import torch.nn as nn


import random


import matplotlib.pyplot as plt


import matplotlib


import pandas as pd


import scipy


from torch.utils.data.dataset import Dataset


from sklearn.metrics import det_curve


from sklearn.metrics import roc_curve


from sklearn.metrics import auc


import scipy as sp


from sklearn.manifold import TSNE


from functools import partial


from collections import defaultdict


from torch.utils.data.dataset import ConcatDataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


import copy


from collections import OrderedDict


from torch import nn


from torch.nn import functional as F


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import BatchSampler


from torch.utils.data import DistributedSampler


from typing import List


from typing import Callable


from torch import Tensor


from torch.nn import Module


from matplotlib import pyplot as plt


from torch.autograd import grad as torch_grad


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.tensorboard import SummaryWriter


from scipy.io import wavfile


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


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config['preprocessing']['pitch']['feature']
        self.energy_feature_level = preprocess_config['preprocessing']['energy']['feature']
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        mel_targets, _, _, pitch_targets, energy_targets, duration_targets = inputs[6:]
        mel_predictions, postnet_mel_predictions, pitch_predictions, energy_predictions, log_duration_predictions, _, src_masks, mel_masks, _, _ = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, :mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        if self.pitch_feature_level == 'phoneme_level':
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == 'frame_level':
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)
        if self.energy_feature_level == 'phoneme_level':
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == 'frame_level':
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        return total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        self.input_size = model_config['transformer']['encoder_hidden']
        self.filter_size = model_config['variance_predictor']['filter_size']
        self.kernel = model_config['variance_predictor']['kernel_size']
        self.conv_output_size = model_config['variance_predictor']['filter_size']
        self.dropout = model_config['variance_predictor']['dropout']
        self.conv_layer = nn.Sequential(OrderedDict([('conv1d_1', Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)), ('relu_1', nn.ReLU()), ('layer_norm_1', nn.LayerNorm(self.filter_size)), ('dropout_1', nn.Dropout(self.dropout)), ('conv1d_2', Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)), ('relu_2', nn.ReLU()), ('layer_norm_2', nn.LayerNorm(self.filter_size)), ('dropout_2', nn.Dropout(self.dropout))]))
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.pitch_feature_level = preprocess_config['preprocessing']['pitch']['feature']
        self.energy_feature_level = preprocess_config['preprocessing']['energy']['feature']
        assert self.pitch_feature_level in ['phoneme_level', 'frame_level']
        assert self.energy_feature_level in ['phoneme_level', 'frame_level']
        pitch_quantization = model_config['variance_embedding']['pitch_quantization']
        energy_quantization = model_config['variance_embedding']['energy_quantization']
        n_bins = model_config['variance_embedding']['n_bins']
        assert pitch_quantization in ['linear', 'log']
        assert energy_quantization in ['linear', 'log']
        with open(os.path.join(preprocess_config['path']['preprocessed_path'], 'stats.json')) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats['pitch'][:2]
            energy_min, energy_max = stats['energy'][:2]
        if pitch_quantization == 'log':
            self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)), requires_grad=False)
        else:
            self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False)
        if energy_quantization == 'log':
            self.energy_bins = nn.Parameter(torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)), requires_grad=False)
        else:
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False)
        self.pitch_embedding = nn.Embedding(n_bins, model_config['transformer']['encoder_hidden'])
        self.energy_embedding = nn.Embedding(n_bins, model_config['transformer']['encoder_hidden'])

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask=None, max_len=None, pitch_target=None, energy_target=None, duration_target=None, p_control=1.0, e_control=1.0, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == 'phoneme_level':
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == 'phoneme_level':
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            x = x + energy_embedding
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction) - 1) * d_control, min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        if self.pitch_feature_level == 'frame_level':
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == 'frame_level':
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            x = x + energy_embedding
        return x, pitch_prediction, energy_prediction, log_duration_prediction, duration_rounded, mel_len, mel_mask


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, kernel_size, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_n_convolutions=5):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Sequential(ConvNorm(n_mel_channels, postnet_embedding_dim, kernel_size=postnet_kernel_size, stride=1, padding=int((postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(postnet_embedding_dim)))
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(nn.Sequential(ConvNorm(postnet_embedding_dim, postnet_embedding_dim, kernel_size=postnet_kernel_size, stride=1, padding=int((postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(postnet_embedding_dim)))
        self.convolutions.append(nn.Sequential(ConvNorm(postnet_embedding_dim, n_mel_channels, kernel_size=postnet_kernel_size, stride=1, padding=int((postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='linear'), nn.BatchNorm1d(n_mel_channels)))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.contiguous().transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)


_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


_pad = '_'


_punctuation = "!'(),.:;? "


_silences = ['@sp', '@spn', '@sil']


_special = '-'


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()
        n_position = config['max_seq_len'] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config['transformer']['encoder_hidden']
        n_layers = config['transformer']['encoder_layer']
        n_head = config['transformer']['encoder_head']
        d_k = d_v = config['transformer']['encoder_hidden'] // config['transformer']['encoder_head']
        d_model = config['transformer']['encoder_hidden']
        d_inner = config['transformer']['conv_filter_size']
        kernel_size = config['transformer']['conv_kernel_size']
        dropout = config['transformer']['encoder_dropout']
        self.max_seq_len = config['max_seq_len']
        self.d_model = d_model
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)
        self.layer_stack = nn.ModuleList([FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()
        n_position = config['max_seq_len'] + 1
        d_word_vec = config['transformer']['decoder_hidden']
        n_layers = config['transformer']['decoder_layer']
        n_head = config['transformer']['decoder_head']
        d_k = d_v = config['transformer']['decoder_hidden'] // config['transformer']['decoder_head']
        d_model = config['transformer']['decoder_hidden']
        d_inner = config['transformer']['conv_filter_size']
        kernel_size = config['transformer']['conv_kernel_size']
        dropout = config['transformer']['decoder_dropout']
        self.max_seq_len = config['max_seq_len']
        self.d_model = d_model
        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)
        self.layer_stack = nn.ModuleList([FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)])

    def forward(self, enc_seq, mask, return_attns=False):
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            max_len = min(max_len, self.max_seq_len)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        return dec_output, mask


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PostNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 80, 80])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_SungFeng_Huang_Meta_TTS(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

