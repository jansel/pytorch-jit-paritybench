import sys
_module = sys.modules[__name__]
del sys
audio = _module
audio_processing = _module
hparams = _module
stft = _module
tools = _module
ljspeech = _module
dataset = _module
fastspeech = _module
glow = _module
loss = _module
modules = _module
optimizer = _module
preprocess = _module
synthesis = _module
tacotron2 = _module
layers = _module
model = _module
utils = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
train = _module
Beam = _module
Constants = _module
Layers = _module
Models = _module
Modules = _module
SubLayers = _module
transformer = _module
utils = _module
waveglow = _module
convert_model = _module
glow = _module
inference = _module
mel2samp = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.nn import functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import math


import torch.nn as nn


import copy


from collections import OrderedDict


import time


from math import sqrt


from torch import nn


import random


import torch.utils.data


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
        forward_basis = torch.FloatTensor(fourier_basis[:, (None), :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, (None), :])
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
            inverse_transform[:, :, (approx_nonzero_indices)] /= window_sum[approx_nonzero_indices]
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


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
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
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
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
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=hp.fft_conv1d_kernel, padding=hp.fft_conv1d_padding)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=hp.fft_conv1d_kernel, padding=hp.fft_conv1d_padding)
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

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        return enc_output, enc_slf_attn


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


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


_special = '-'


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
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()
        self.input_size = hp.d_model
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = hp.dropout
        self.conv_layer = nn.Sequential(OrderedDict([('conv1d_1', Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=1)), ('layer_norm_1', nn.LayerNorm(self.filter_size)), ('relu_1', nn.ReLU()), ('dropout_1', nn.Dropout(self.dropout)), ('conv1d_2', Conv(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1)), ('layer_norm_2', nn.LayerNorm(self.filter_size)), ('relu_2', nn.ReLU()), ('dropout_2', nn.Dropout(self.dropout))]))
        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, duration_predictor_output, alpha=1.0, mel_max_length=None):
        output = list()
        for batch, expand_target in zip(x, duration_predictor_output):
            output.append(self.expand(batch, expand_target, alpha))
        if mel_max_length:
            output = utils.pad(output, mel_max_length)
        else:
            output = utils.pad(output)
        return output

    def expand(self, batch, predicted, alpha):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size * alpha), -1))
        out = torch.cat(out, 0)
        return out

    def rounding(self, num):
        if num - int(num) >= 0.5:
            return int(num) + 1
        else:
            return int(num)

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if self.training:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            for idx, ele in enumerate(duration_predictor_output[0]):
                duration_predictor_output[0][idx] = self.rounding(ele)
            output = self.LR(x, duration_predictor_output, alpha)
            mel_pos = torch.stack([torch.Tensor([(i + 1) for i in range(output.size(1))])]).long()
            return output, mel_pos


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


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()
        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()
        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        self.postnet = PostNet()

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_output, _ = self.encoder(src_seq, src_pos)
        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output, target=length_target, alpha=alpha, mel_max_length=mel_max_length)
            decoder_output = self.decoder(length_regulator_output, mel_pos)
            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
            return mel_output, mel_output_postnet, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output, alpha=alpha)
            decoder_output = self.decoder(length_regulator_output, decoder_pos)
            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
            return mel_output, mel_output_postnet


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
            W[:, (0)] = -1 * W[:, (0)]
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
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input
        spect = self.upsample(spect)
        assert spect.size(2) >= audio.size(1)
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]
            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)
            audio = torch.cat([audio_0, audio_1], 1)
        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

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
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]
            output = self.WN[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)
            audio = self.convinv[k](audio, reverse=True)
            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma * z, audio), 1)
        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


class FastSpeechLoss(nn.Module):
    """ FastSPeech Loss """

    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)
        duration_predictor_target.requires_grad = False
        duration_predictor_loss = self.l1_loss(duration_predicted, duration_predictor_target.float())
        return mel_loss, mel_postnet_loss, duration_predictor_loss


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=3, padding=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        x = input_
        x = self.w_2(torch.relu(self.w_1(x)))
        x = x + input_
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        if query_mask is not None:
            attn = attn * query_mask
        attn = self.attn_dropout(attn)
        result = torch.bmm(attn, value)
        return result, attn


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LocationLayer(nn.Module):

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class Prenet(nn.Module):

    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([LinearNorm(in_size, out_size, bias=False) for in_size, out_size in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Sequential(ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim, kernel_size=hparams.postnet_kernel_size, stride=1, padding=int((hparams.postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(hparams.postnet_embedding_dim)))
        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(nn.Sequential(ConvNorm(hparams.postnet_embedding_dim, hparams.postnet_embedding_dim, kernel_size=hparams.postnet_kernel_size, stride=1, padding=int((hparams.postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(hparams.postnet_embedding_dim)))
        self.convolutions.append(nn.Sequential(ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels, kernel_size=hparams.postnet_kernel_size, stride=1, padding=int((hparams.postnet_kernel_size - 1) / 2), dilation=1, w_init_gain='linear'), nn.BatchNorm1d(hparams.n_mel_channels)))

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x
    return torch.autograd.Variable(x)


class Tacotron2(nn.Module):

    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, (0), :], 1000.0)
        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths), encoder_outputs

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs, encoder_outputs


class PreNet(nn.Module):
    """
    Pre Net before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([('fc1', Linear(self.input_size, self.hidden_size)), ('relu1', nn.ReLU()), ('dropout1', nn.Dropout(p)), ('fc2', Linear(self.hidden_size, self.output_size)), ('relu2', nn.ReLU()), ('dropout2', nn.Dropout(p))]))

    def forward(self, input_):
        out = self.layer(input_)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (FFN,
     lambda: ([], {'num_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (FastSpeechLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Invertible1x1Conv,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearNorm,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LocationLayer,
     lambda: ([], {'attention_n_filters': 4, 'attention_kernel_size': 4, 'attention_dim': 4}),
     lambda: ([torch.rand([4, 2, 64])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'num_hidden_k': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PostNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 80, 80])], {}),
     False),
    (PreNet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (WaveGlowLoss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
]

class Test_xcmyz_FastSpeech(_paritybench_base):
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

