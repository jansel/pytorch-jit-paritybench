import sys
_module = sys.modules[__name__]
del sys
audio_processing = _module
hparams = _module
layers = _module
conv = _module
model = _module
module = _module
stft = _module
text = _module
cleaners = _module
cmudict = _module
numbers = _module
symbols = _module
train = _module
data_utils = _module
plot_image = _module
text2seq = _module
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


import torch


import numpy as np


from scipy.signal import get_window


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.utils.spectral_norm as sn


import torch.distributions as D


from torch.autograd import Variable


import random


import torch.utils.data


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import copy


import time


import torch.distributed as dist


from scipy.io.wavfile import write


from scipy.io.wavfile import read


from torch.utils.data.distributed import DistributedSampler


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


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
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0), mode='reflect')
        input_data = input_data.squeeze(1)
        forward_transform = F.conv1d(input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)
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


class Linear(nn.Linear):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))


class Conv1d(nn.Conv1d):

    def __init__(self, *args, activation=None, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        self.padding = self.dilation[0] * (self.kernel_size[0] - 1) // 2
        self.act = None
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('linear'))
        if not activation is None:
            self.act = activation
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs, mask=None):
        if self.act is None:
            outputs = super(Conv1d, self).forward(inputs)
        else:
            outputs = self.act(super(Conv1d, self).forward(inputs))
        if mask is None:
            return outputs
        else:
            outputs = outputs.masked_fill(mask.unsqueeze(1), 0)
            return outputs


class Softplus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = torch.log(1 + torch.exp(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * torch.sigmoid(ctx.saved_variables[0])


class CustomSoftplus(nn.Module):

    def forward(self, input_tensor):
        return Softplus.apply(input_tensor)


class BVAE_layer(nn.Module):

    def __init__(self, hdim, kernel_size, dilation=1, adj_dim=False):
        super(BVAE_layer, self).__init__()
        self.softplus = CustomSoftplus()
        if adj_dim == True:
            self.pre_conv = Conv1d(2 * hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)
        else:
            self.pre_conv = Conv1d(hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)
        self.up_conv_a = nn.ModuleList([sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu)), sn(Conv1d(hdim, 3 * hdim, kernel_size, bias=False))])
        self.up_conv_b = sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu))
        self.down_conv_a = nn.ModuleList([sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu)), sn(Conv1d(hdim, 5 * hdim, kernel_size, bias=False))])
        self.down_conv_b = nn.ModuleList([sn(Conv1d(2 * hdim, hdim, kernel_size, bias=False)), sn(Conv1d(hdim, hdim, kernel_size, activation=F.elu))])
        if adj_dim == True:
            self.post_conv = Conv1d(hdim, 2 * hdim, kernel_size, activation=F.elu, dilation=dilation)
        else:
            self.post_conv = Conv1d(hdim, hdim, kernel_size, activation=F.elu, dilation=dilation)

    def up(self, inputs, mask=None):
        inputs = self.pre_conv(inputs, mask)
        x = self.up_conv_a[0](inputs, mask)
        self.qz_mean, self.qz_std, h = self.up_conv_a[1](x, mask).chunk(3, 1)
        self.qz_std = self.softplus(self.qz_std)
        h = self.up_conv_b(h, mask)
        return (inputs + h) / 2 ** 0.5

    def down(self, inputs, mask=None, sample=False, temp=1):
        x = self.down_conv_a[0](inputs, mask)
        pz_mean, pz_std, rz_mean, rz_std, h = self.down_conv_a[1](x, mask).chunk(5, 1)
        pz_std, rz_std = self.softplus(pz_std), self.softplus(rz_std)
        if sample == True:
            prior = D.Normal(pz_mean, pz_std * temp)
            z = prior.rsample()
            kl = torch.zeros(inputs.size(0)).mean()
        else:
            prior = D.Normal(pz_mean, pz_std)
            posterior = D.Normal(pz_mean + self.qz_mean + rz_mean, pz_std * self.qz_std * rz_std)
            z = posterior.rsample().masked_fill(mask.unsqueeze(1), 0)
            kl = D.kl.kl_divergence(posterior, prior).mean()
        h = torch.cat((z, h), 1)
        h = self.down_conv_b[0](h, mask)
        h = self.down_conv_b[1](h, mask)
        outputs = self.post_conv((inputs + h) / 2 ** 0.5, mask)
        return outputs, kl


class BVAE_block(nn.Module):

    def __init__(self, hdim, kernel_size, n_layers, down_upsample):
        super(BVAE_block, self).__init__()
        self.down_upsample = down_upsample
        self.BVAE_layers = nn.ModuleList()
        for i in range(n_layers):
            self.BVAE_layers.append(BVAE_layer(hdim, kernel_size, dilation=2 ** i, adj_dim=down_upsample == 'F' and i == 0))

    def up(self, inputs, mask=None):
        if self.down_upsample == 'T':
            inputs = self.blur_pool(inputs, mask)
        x = inputs
        for layer in self.BVAE_layers:
            x = layer.up(x, mask)
        return x

    def down(self, inputs, mask=None, sample=False, temperature=1.0):
        x = inputs
        kl = 0
        for layer in reversed(self.BVAE_layers):
            x, curr_kl = layer.down(x, mask, sample, temperature)
            kl += curr_kl
        if self.down_upsample == 'T':
            x = x.repeat_interleave(2, -1)
        return x, kl

    def blur_pool(self, x, mask):
        blur_kernel = torch.tensor([[[0.25, 0.5, 0.25]]]).repeat(x.size(1), 1, 1)
        outputs = F.conv1d(x, blur_kernel, padding=1, stride=2, groups=x.size(1))
        outputs = outputs.masked_fill(mask.unsqueeze(1), 0)
        return outputs


class DurationPredictor(nn.Module):

    def __init__(self, hp):
        super(DurationPredictor, self).__init__()
        self.conv1 = Conv1d(hp.hidden_dim, hp.hidden_dim, 3, bias=False, activation=F.elu)
        self.conv2 = Conv1d(hp.hidden_dim, hp.hidden_dim, 3, bias=False, activation=F.elu)
        self.ln1 = nn.LayerNorm(hp.hidden_dim)
        self.ln2 = nn.LayerNorm(hp.hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear = Linear(hp.hidden_dim, 1)

    def forward(self, h, mask=None):
        x = self.conv1(h, mask)
        x = self.dropout(self.ln1(x.transpose(1, 2)))
        x = self.conv2(x.transpose(1, 2), mask)
        x = self.dropout(self.ln2(x.transpose(1, 2)))
        out = self.linear(x).exp() + 1
        return out.squeeze(-1)


def PositionalEncoding(d_model, lengths, w_s=None):
    L = int(lengths.max().item())
    if w_s is None:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0)
    else:
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(0) * w_s.unsqueeze(-1)
    div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
    pe = torch.zeros(len(lengths), L, d_model)
    pe[:, :, 0::2] = torch.sin(position.unsqueeze(-1) / div_term.unsqueeze(0))
    pe[:, :, 1::2] = torch.cos(position.unsqueeze(-1) / div_term.unsqueeze(0))
    return pe


class Prenet(nn.Module):

    def __init__(self, hp):
        super(Prenet, self).__init__()
        self.layers = nn.ModuleList([Conv1d(hp.n_mel_channels, hp.hidden_dim, 1, bias=True, activation=F.elu), Conv1d(hp.hidden_dim, hp.hidden_dim, 1, bias=True, activation=F.elu)])

    def forward(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            x = F.dropout(layer(x, mask), 0.5, training=True)
        return x


class Projection(nn.Module):

    def __init__(self, hdim, kernel_size, outdim):
        super(Projection, self).__init__()
        self.layers = nn.ModuleList([Conv1d(hdim, hdim, kernel_size, activation=F.elu), Conv1d(hdim, hdim, kernel_size, activation=F.elu), Conv1d(hdim, outdim, kernel_size)])

    def forward(self, x, mask=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.dropout(layer(x, mask), 0.5, training=self.training)
            else:
                x = layer(x, mask)
        return torch.sigmoid(x)


class TextEnc(nn.Module):

    def __init__(self, hp):
        super(TextEnc, self).__init__()
        self.Embedding = nn.Embedding(hp.n_symbols, hp.hidden_dim)
        self.conv_layers = nn.ModuleList([Conv1d(hp.hidden_dim, 2 * hp.hidden_dim, hp.kernel_size) for _ in range(7)])

    def forward(self, text, mask=None):
        embedded = F.dropout(self.Embedding(text), 0.1, training=self.training)
        x = embedded.transpose(1, 2)
        for conv in self.conv_layers:
            x1, x2 = torch.chunk(conv(x, mask), 2, dim=1)
            x = (x1 * torch.sigmoid(x2) + x) / 2 ** 0.5
            x = F.dropout(x, 0.1, training=self.training)
        key = x.transpose(1, 2)
        value = (key + embedded) / 2 ** 0.5
        return key, value


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len, device=lengths.device))
    mask = lengths.unsqueeze(1) <= ids
    return mask.detach()


class Model(nn.Module):

    def __init__(self, hp):
        super(Model, self).__init__()
        self.hp = hp
        self.ratio = hp.downsample_ratio
        self.text_mask = None
        self.mel_mask = None
        self.diag_mask = None
        self.Prenet = Prenet(hp)
        self.TextEnc = TextEnc(hp)
        self.BVAE_blocks = nn.ModuleList()
        for i in range(hp.n_blocks):
            ForT = 'F' if i % 2 == 0 else 'T'
            self.BVAE_blocks.append(BVAE_block(hp.hidden_dim // 2 ** (i // 2 + 1), hp.kernel_size, hp.n_layers, down_upsample=ForT))
        self.Query = Conv1d(hp.hidden_dim // self.ratio, hp.hidden_dim, hp.kernel_size, bias=False)
        self.Compress = Linear(hp.hidden_dim, hp.hidden_dim // self.ratio, bias=False)
        self.Projection = Projection(hp.hidden_dim, hp.kernel_size, hp.n_mel_channels)
        self.Duration = DurationPredictor(hp)

    def forward(self, text, melspec, text_lengths, mel_lengths):
        self.text_mask, self.mel_mask, self.diag_mask = self.prepare_mask(text_lengths, mel_lengths)
        key, value = self.TextEnc(text, self.text_mask)
        query = self.bottom_up(melspec, self.mel_mask)
        h, align = self.get_align(query, key, value, text_lengths, mel_lengths, self.text_mask, self.mel_mask)
        mel_pred, kl_loss = self.top_down(h, self.mel_mask)
        duration_out = self.get_duration(value, self.text_mask)
        recon_loss, duration_loss, align_loss = self.compute_loss(mel_pred, melspec, duration_out, align, mel_lengths, self.text_mask, self.mel_mask, self.diag_mask)
        return recon_loss, kl_loss, duration_loss, align_loss

    def prepare_mask(self, text_lengths, mel_lengths):
        B, L, T = text_lengths.size(0), text_lengths.max().item(), mel_lengths.max().item()
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        x = (torch.arange(L).float().unsqueeze(0) / text_lengths.unsqueeze(1)).unsqueeze(1) - (torch.arange(T // self.ratio).float().unsqueeze(0) / (mel_lengths // self.ratio).unsqueeze(1)).unsqueeze(2)
        diag_mask = (-12.5 * torch.pow(x, 2)).exp()
        diag_mask = diag_mask.masked_fill(text_mask.unsqueeze(1), 0)
        diag_mask = diag_mask.masked_fill(mel_mask[:, ::self.ratio].unsqueeze(-1), 0)
        return text_mask, mel_mask, diag_mask

    def bottom_up(self, melspec, mel_mask):
        x = self.Prenet(melspec, mel_mask)
        for i, block in enumerate(self.BVAE_blocks):
            x = block.up(x, mel_mask[:, ::2 ** ((i + 1) // 2)])
        query = self.Query(x, mel_mask[:, ::self.ratio]).transpose(1, 2)
        return query

    def top_down(self, h, mel_mask):
        kl = 0
        for i, block in enumerate(reversed(self.BVAE_blocks)):
            h, curr_kl = block.down(h, mel_mask[:, ::2 ** (len(self.BVAE_blocks) // 2 - (i + 1) // 2)])
            kl += curr_kl
        mel_pred = self.Projection(h, mel_mask)
        return mel_pred, kl

    def get_align(self, q, k, v, text_lengths, mel_lengths, text_mask, mel_mask):
        q = q + PositionalEncoding(self.hp.hidden_dim, mel_lengths // self.ratio)
        k = k + PositionalEncoding(self.hp.hidden_dim, text_lengths, 1.0 * mel_lengths / self.ratio / text_lengths)
        q = q * self.hp.hidden_dim ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores.masked_fill(text_mask.unsqueeze(1), -float('inf'))
        align = scores.softmax(-1)
        align = align.masked_fill(mel_mask[:, ::self.ratio].unsqueeze(-1), 0)
        if self.training:
            align_oh = self.jitter(F.one_hot(align.max(-1)[1], align.size(-1)), mel_lengths)
        else:
            align_oh = F.one_hot(align.max(-1)[1], align.size(-1))
        align_oh = align_oh.masked_fill(mel_mask[:, ::self.ratio].unsqueeze(-1), 0)
        attn_output = torch.bmm(align + (align_oh - align).detach(), v)
        attn_output = self.Compress(attn_output).transpose(1, 2)
        return attn_output, align

    def compute_loss(self, mel_pred, mel_target, duration_out, align, mel_lengths, text_mask, mel_mask, diag_mask):
        recon_loss = nn.L1Loss()(mel_pred.masked_select(~mel_mask.unsqueeze(1)), mel_target.masked_select(~mel_mask.unsqueeze(1)))
        duration_target = self.align2duration(align, mel_lengths)
        duration_target_flat = duration_target.masked_select(~text_mask)
        duration_target_flat[duration_target_flat <= 0] = 1
        duration_out_flat = duration_out.masked_select(~text_mask)
        duration_loss = nn.MSELoss()(torch.log(duration_out_flat + 1e-05), torch.log(duration_target_flat + 1e-05))
        align_losses = align * (1 - diag_mask)
        align_loss = torch.mean(align_losses.masked_select(diag_mask.bool()))
        return recon_loss, duration_loss, align_loss

    def inference(self, text, alpha=1.0, temperature=1.0):
        assert len(text) == 1, 'You must encode only one sentence at once'
        text_lengths = torch.tensor([text.size(1)])
        key, value = self.TextEnc(text)
        durations = self.get_duration(value)
        h, durations = self.LengthRegulator(value, durations, alpha)
        h = self.Compress(h).transpose(1, 2)
        if isinstance(temperature, float):
            temperature = [temperature] * len(self.BVAE_blocks)
        for i, block in enumerate(reversed(self.BVAE_blocks)):
            h, _ = block.down(h, sample=True, temperature=temperature[i])
        mel_out = self.Projection(h)
        return mel_out, durations

    def get_duration(self, value, mask=None):
        durations = self.Duration(value.transpose(1, 2).detach(), mask)
        return durations

    def align2duration(self, alignments, mel_lengths):
        max_ids = torch.max(alignments, dim=2)[1]
        max_ids_oh = F.one_hot(max_ids, alignments.size(2))
        mask = get_mask_from_lengths(mel_lengths // self.ratio).unsqueeze(-1)
        max_ids_oh.masked_fill_(mask, 0)
        durations = max_ids_oh.sum(dim=1)
        return durations

    def LengthRegulator(self, hidden_states, durations, alpha=1.0):
        durations = torch.round(durations * alpha)
        durations[durations <= 0] = 1
        return hidden_states.repeat_interleave(durations[0], dim=1), durations

    def jitter(self, alignments, mel_lengths):
        B, T, _ = alignments.size()
        batch_indices = torch.arange(B).unsqueeze(1)
        jitter_indices = torch.arange(T).unsqueeze(0).repeat(B, 1)
        jitter_indices = torch.round(jitter_indices + (2 * torch.rand(jitter_indices.size()) - 1).to(alignments.device))
        jitter_indices = torch.where(jitter_indices < (mel_lengths // self.ratio).unsqueeze(1), jitter_indices, (mel_lengths // self.ratio - 1).unsqueeze(-1).repeat(1, T))
        jitter_indices[jitter_indices <= 0] = 0
        alignments = alignments[batch_indices, jitter_indices]
        alignments.masked_fill_(self.mel_mask[:, ::self.ratio].unsqueeze(-1), 0)
        return alignments


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
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ConvNorm,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (CustomSoftplus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
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
    (Projection,
     lambda: ([], {'hdim': 4, 'kernel_size': 4, 'outdim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (WaveGlowLoss,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
]

class Test_LEEYOONHYUNG_BVAE_TTS(_paritybench_base):
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

