import sys
_module = sys.modules[__name__]
del sys
audio = _module
audio_processing = _module
stft = _module
tools = _module
kss = _module
dataset = _module
evaluate = _module
fastspeech2 = _module
hparams = _module
loss = _module
modules = _module
optimizer = _module
prepare_align = _module
preprocess = _module
synthesize = _module
text = _module
cleaners = _module
korean = _module
num = _module
symbols = _module
train = _module
Constants = _module
Layers = _module
Models = _module
Modules = _module
SubLayers = _module
transformer = _module
utils = _module
vocgan_generator = _module

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


from torch.autograd import Variable


from scipy.io.wavfile import read


from scipy.io.wavfile import write


from sklearn.preprocessing import StandardScaler


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import math


import time


import torch.nn as nn


import re


from collections import OrderedDict


import copy


from string import punctuation


from torch.utils.tensorboard import SummaryWriter


from torch.nn import functional as F


import matplotlib


from matplotlib import pyplot as plt


from scipy.io import wavfile


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
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0).cpu()
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, Variable(self.inverse_basis, requires_grad=False), stride=self.hop_length, padding=0)
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

    def __init__(self, filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin=0.0, mel_fmax=8000.0):
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

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=hp.fft_conv1d_kernel_size[0], padding=(hp.fft_conv1d_kernel_size[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=hp.fft_conv1d_kernel_size[1], padding=(hp.fft_conv1d_kernel_size[1] - 1) // 2)
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

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        return enc_output, enc_slf_attn


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


EOS = '~'


PAD = '_'


JAMO_LEADS = ''.join([chr(_) for _ in range(4352, 4371)])


JAMO_TAILS = ''.join([chr(_) for _ in range(4520, 4547)])


JAMO_VOWELS = ''.join([chr(_) for _ in range(4449, 4470)])


PUNC = "!'(),-.:;?"


SPACE = ' '


VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE


_SILENCES = ['sp', 'spn', 'sil']


ALL_SYMBOLS = list(PAD + EOS + VALID_CHARS) + _SILENCES


KOR_SYMBOLS = ALL_SYMBOLS


kor_symbols = KOR_SYMBOLS


symbols = kor_symbols


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)
        return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(VariancePredictor, self).__init__()
        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout
        self.conv_layer = nn.Sequential(OrderedDict([('conv1d_1', Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2)), ('relu_1', nn.ReLU()), ('layer_norm_1', nn.LayerNorm(self.filter_size)), ('dropout_1', nn.Dropout(self.dropout)), ('conv1d_2', Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)), ('relu_2', nn.ReLU()), ('layer_norm_2', nn.LayerNorm(self.filter_size)), ('dropout_2', nn.Dropout(self.dropout))]))
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()
        self.energy_embedding_producer = Conv(1, hp.encoder_hidden, kernel_size=9, bias=False, padding=4)
        self.pitch_embedding_producer = Conv(1, hp.encoder_hidden, kernel_size=9, bias=False, padding=4)

    def forward(self, x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None, dur_pitch_energy_aug=None, f0_stat=None, energy_stat=None):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding_producer(pitch_target.unsqueeze(2))
        else:
            pitch_prediction = utils.de_norm(pitch_prediction, mean=f0_stat[0], std=f0_stat[1]) * dur_pitch_energy_aug[1]
            pitch_prediction = utils.standard_norm_torch(pitch_prediction, mean=f0_stat[0], std=f0_stat[1])
            pitch_embedding = self.pitch_embedding_producer(pitch_prediction.unsqueeze(2))
        energy_prediction = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding_producer(energy_target.unsqueeze(2))
        else:
            energy_prediction = utils.de_norm(energy_prediction, mean=energy_stat[0], std=energy_stat[1]) * dur_pitch_energy_aug[2]
            energy_prediction = utils.standard_norm_torch(energy_prediction, mean=energy_stat[0], std=energy_stat[1])
            energy_embedding = self.energy_embedding_producer(energy_prediction.unsqueeze(2))
        x = x + pitch_embedding + energy_embedding
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction) - hp.log_offset) * dur_pitch_energy_aug[0], min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)
        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, dur_pitch_energy_aug=None, f0_stat=None, energy_stat=None):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        encoder_output = self.encoder(src_seq, src_mask)
        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, dur_pitch_energy_aug, f0_stat, energy_stat)
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output
        return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, log_d_predicted, log_d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target, src_mask, mel_mask):
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False
        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(src_mask)
        p_target = p_target.masked_select(src_mask)
        e_predicted = e_predicted.masked_select(src_mask)
        e_target = e_target.masked_select(src_mask)
        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))
        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)
        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)
        return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss


class ResStack(nn.Module):

    def __init__(self, channel, dilation=1):
        super(ResStack, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(dilation), nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=dilation)), nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)))
        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.block[2])
        nn.utils.remove_weight_norm(self.block[4])
        nn.utils.remove_weight_norm(self.shortcut)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, mel_channel, n_residual_layers, ratios=[4, 4, 2, 2, 2, 2], mult=256, out_band=1):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.start = nn.Sequential(nn.ReflectionPad1d(3), nn.utils.weight_norm(nn.Conv1d(mel_channel, mult * 2, kernel_size=7, stride=1)))
        r = ratios[0]
        self.upsample_1 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.res_stack_1 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        r = ratios[1]
        mult = mult // 2
        self.upsample_2 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.res_stack_2 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        r = ratios[2]
        mult = mult // 2
        self.upsample_3 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.skip_upsample_1 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult, kernel_size=64, stride=32, padding=16, output_padding=0))
        self.res_stack_3 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        r = ratios[3]
        mult = mult // 2
        self.upsample_4 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.skip_upsample_2 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult, kernel_size=128, stride=64, padding=32, output_padding=0))
        self.res_stack_4 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        r = ratios[4]
        mult = mult // 2
        self.upsample_5 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.skip_upsample_3 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult, kernel_size=256, stride=128, padding=64, output_padding=0))
        self.res_stack_5 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        r = ratios[5]
        mult = mult // 2
        self.upsample_6 = nn.Sequential(nn.LeakyReLU(0.2), nn.utils.weight_norm(nn.ConvTranspose1d(mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2)))
        self.skip_upsample_4 = nn.utils.weight_norm(nn.ConvTranspose1d(mel_channel, mult, kernel_size=512, stride=256, padding=128, output_padding=0))
        self.res_stack_6 = nn.Sequential(*[ResStack(mult, dilation=3 ** j) for j in range(n_residual_layers)])
        self.out = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(3), nn.utils.weight_norm(nn.Conv1d(mult, out_band, kernel_size=7, stride=1)), nn.Tanh())
        self.apply(weights_init)

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0
        x = self.start(mel)
        x = self.upsample_1(x)
        x = self.res_stack_1(x)
        x = self.upsample_2(x)
        x = self.res_stack_2(x)
        x = self.upsample_3(x)
        x = x + self.skip_upsample_1(mel)
        x = self.res_stack_3(x)
        x = self.upsample_4(x)
        x = x + self.skip_upsample_2(mel)
        x = self.res_stack_4(x)
        x = self.upsample_5(x)
        x = x + self.skip_upsample_3(mel)
        x = self.res_stack_5(x)
        x = self.upsample_6(x)
        x = x + self.skip_upsample_4(mel)
        x = self.res_stack_6(x)
        out = self.out(x)
        return out

    def eval(self, inference=False):
        super(Generator, self).eval()
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def infer(self, mel):
        hop_length = 256
        zero = torch.full((1, self.mel_channel, 10), -11.5129)
        mel = torch.cat((mel, zero), dim=2)
        audio = self.forward(mel)
        return audio


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
    (Generator,
     lambda: ([], {'mel_channel': 4, 'n_residual_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PostNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 80, 80])], {}),
     False),
    (ResStack,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_HGU_DLLAB_Korean_FastSpeech2_Pytorch(_paritybench_base):
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

