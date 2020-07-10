import sys
_module = sys.modules[__name__]
del sys
compute_sil_duration_gap_stats = _module
src = _module
clarinet = _module
data = _module
loss = _module
modules = _module
preprocessing = _module
synthesize = _module
synthesize_student = _module
train = _module
train_student = _module
wavenet = _module
wavenet_iaf = _module
dataset = _module
audio_loader = _module
audio_parser = _module
ljspeech = _module
noise_injector = _module
preprocess = _module
spectrogram_dataset = _module
spectrogram_parser = _module
vctk = _module
vctk_dataset = _module
vctk_features_dataset = _module
vctk_features_stream = _module
vctk_speech_stream = _module
error_handling = _module
color_print = _module
console_logger = _module
exception_decorators = _module
logger_factory = _module
evaluation = _module
alignment_stats = _module
embedding_space_stats = _module
gradient_stats = _module
losses_plotter = _module
utils = _module
experiments = _module
base_trainer = _module
checkpoint_utils = _module
convolutional_trainer = _module
device_configuration = _module
evaluator = _module
experiment = _module
experiments = _module
pipeline_factory = _module
flow_wavenet = _module
data = _module
model = _module
modules = _module
synthesize = _module
train = _module
main = _module
models = _module
convolutional_encoder = _module
convolutional_vq_vae = _module
deconvolutional_decoder = _module
vector_quantizer = _module
vector_quantizer_ema = _module
wavenet_decoder = _module
wavenet_vq_vae = _module
conv1d_builder = _module
conv_transpose1d_builder = _module
jitter = _module
residual = _module
residual_stack = _module
speech_utils = _module
global_conditioning = _module
mu_law = _module
speech_features = _module
wavenet_vocoder = _module
builder = _module
conv = _module
mixture = _module
modules = _module
util = _module
wavenet = _module
global_conditioning_test = _module

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


import numpy as np


import torch


import random


from torch.utils.data import Dataset


import math


from torch.distributions.normal import Normal


import torch.nn as nn


import torch.nn.functional as F


import time


from torch.utils.data import DataLoader


from torch import optim


from torch import nn


import scipy


import torch.optim as optim


from math import log


from math import pi


from itertools import combinations


from itertools import product


from torch.nn import functional as F


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, causal=True):
        super(Conv, self).__init__()
        self.causal = causal
        if self.causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation, cin_channels=None, local_conditioning=True, causal=False):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True if skip_channels is not None else False
        self.filter_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        if self.skip:
            self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
            nn.init.kaiming_normal_(self.skip_conv.weight)
        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)
        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)
        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


def gaussian_loss(y_hat, y, log_std_min=-7.0):
    assert y_hat.dim() == 3
    assert y_hat.size(1) == 2
    y_hat = y_hat.transpose(1, 2)
    mean = y_hat[:, :, :1]
    log_std = torch.clamp(y_hat[:, :, 1:], min=log_std_min)
    log_probs = -0.5 * (-math.log(2.0 * math.pi) - 2.0 * log_std - torch.pow(y - mean, 2) * torch.exp(-2.0 * log_std))
    return log_probs.squeeze()


class GaussianLoss(nn.Module):

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, input, target, size_average=True):
        losses = gaussian_loss(input, target)
        if size_average:
            return losses.mean()
        else:
            return losses.mean(1).sum(0)


def KL_gaussians(mu_q, logs_q, mu_p, logs_p, log_std_min=-7.0, regularization=True):
    logs_q = torch.clamp(logs_q, min=log_std_min)
    logs_p = torch.clamp(logs_p, min=log_std_min)
    KL_loss = logs_p - logs_q + 0.5 * ((torch.exp(2.0 * logs_q) + torch.pow(mu_p - mu_q, 2)) * torch.exp(-2.0 * logs_p) - 1.0)
    if regularization:
        reg_loss = torch.pow(logs_q - logs_p, 2)
    else:
        reg_loss = None
    return KL_loss, reg_loss


class KL_Loss(nn.Module):

    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, mu_q, logs_q, mu_p, logs_p, regularization=True, size_average=True):
        KL_loss, reg_loss = KL_gaussians(mu_q, logs_q, mu_p, logs_p, regularization=regularization)
        loss_tot = KL_loss + reg_loss * 4.0
        if size_average:
            return loss_tot.mean(), KL_loss.mean(), reg_loss.mean()
        else:
            return loss_tot.sum(), KL_loss.sum(), reg_loss.sum()


class STFT(torch.nn.Module):

    def __init__(self, filter_length=1024, hop_length=256):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.tensor(fourier_basis[:, (None), :])
        inverse_basis = torch.tensor(np.linalg.pinv(scale * fourier_basis).T[:, (None), :])
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def forward(self, input_data):
        num_batches, _, num_samples = input_data.size()
        self.num_samples = num_samples
        forward_transform = F.conv1d(input_data, self.forward_basis, stride=self.hop_length, padding=self.filter_length)
        cutoff = int(self.filter_length / 2 + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase, self.inverse_basis, stride=self.hop_length, padding=0)
        inverse_transform = inverse_transform[:, :, self.filter_length:]
        inverse_transform = inverse_transform[:, :, :self.num_samples]
        return inverse_transform


class ZeroConv1d(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class Wavenet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, num_blocks=1, num_layers=6, residual_channels=256, gate_channels=256, skip_channels=256, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet, self).__init__()
        self.skip = True if skip_channels is not None else False
        self.front_conv = nn.Sequential(Conv(in_channels, residual_channels, 3, causal=causal), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for n in range(num_layers):
                self.res_blocks.append(ResBlock(residual_channels, gate_channels, skip_channels, kernel_size, dilation=2 ** n, cin_channels=cin_channels, local_conditioning=True, causal=causal))
        last_channels = skip_channels if self.skip else residual_channels
        self.final_conv = nn.Sequential(nn.ReLU(), Conv(last_channels, last_channels, 1, causal=causal), nn.ReLU(), ZeroConv1d(last_channels, out_channels))

    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            if self.skip:
                h, s = f(h, c)
                skip += s
            else:
                h, _ = f(h, c)
        if self.skip:
            out = self.final_conv(skip)
        else:
            out = self.final_conv(h)
        return out


class Wavenet_Flow(nn.Module):

    def __init__(self, out_channels=1, num_blocks=4, num_layers=6, front_channels=32, residual_channels=64, gate_channels=32, skip_channels=None, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Flow, self).__init__()
        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.front_channels = front_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size
        self.front_conv = nn.Sequential(Conv(1, self.residual_channels, self.front_channels, causal=self.causal), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        self.res_blocks_fast = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels, self.gate_channels, self.skip_channels, self.kernel_size, dilation=self.kernel_size ** n, cin_channels=self.cin_channels, local_conditioning=True, causal=self.causal, mode='SAME'))
        self.final_conv = nn.Sequential(nn.ReLU(), Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal), nn.ReLU(), Conv(self.skip_channels, self.out_channels, 1, causal=self.causal))

    def forward(self, x, c):
        return self.wavenet(x, c)

    def wavenet(self, tensor, c=None):
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [(2 ** (i % self.num_layers)) for i in range(self.num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + 1 + (self.front_channels - 1)


class Wavenet_Student(nn.Module):

    def __init__(self, num_blocks_student=[1, 1, 1, 4], num_layers=6, front_channels=32, residual_channels=128, gate_channels=256, skip_channels=128, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Student, self).__init__()
        self.num_blocks = num_blocks_student
        self.num_flow = len(self.num_blocks)
        self.num_layers = num_layers
        self.iafs = nn.ModuleList()
        for i in range(self.num_flow):
            self.iafs.append(Wavenet_Flow(out_channels=2, num_blocks=self.num_blocks[i], num_layers=self.num_layers, front_channels=front_channels, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, kernel_size=kernel_size, cin_channels=cin_channels, causal=causal))

    def forward(self, z, c):
        return self.iaf(z, c)

    def iaf(self, z, c_up):
        mu_tot, logs_tot = 0.0, 0.0
        for i, iaf in enumerate(self.iafs):
            mu_logs = iaf(z, c_up)
            mu = mu_logs[:, 0:1, :-1]
            logs = mu_logs[:, 1:, :-1]
            mu_tot = mu_tot * torch.exp(logs) + mu
            logs_tot = logs_tot + logs
            z = z[:, :, 1:] * torch.exp(logs) + mu
            z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        return z, mu_tot, logs_tot

    def receptive_field(self):
        receptive_field = 1
        for iaf in self.iafs:
            receptive_field += iaf.receptive_field_size() - 1
        return receptive_field

    def generate(self, z, c_up):
        x, _, _ = self.iaf(z, c_up)
        return x


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):

    def __init__(self, in_channel, logdet=True, pretrained=False):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))
        self.initialized = pretrained
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-06))

    def forward(self, x):
        B, _, T = x.size()
        if not self.initialized:
            self.initialize(x)
            self.initialized = True
        log_abs = logabs(self.scale)
        logdet = torch.sum(log_abs) * B * T
        if self.logdet:
            return self.scale * (x + self.loc), logdet
        else:
            return self.scale * (x + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class AffineCoupling(nn.Module):

    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, affine=True):
        super().__init__()
        self.affine = affine
        self.net = Wavenet(in_channels=in_channel // 2, out_channels=in_channel if self.affine else in_channel // 2, num_blocks=1, num_layers=num_layer, residual_channels=filter_size, gate_channels=filter_size, skip_channels=filter_size, kernel_size=3, cin_channels=cin_channel // 2, causal=False)

    def forward(self, x, c=None):
        in_a, in_b = x.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a, c_a).chunk(2, 1)
            out_b = (in_b - t) * torch.exp(-log_s)
            logdet = torch.sum(-log_s)
        else:
            net_out = self.net(in_a, c_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, c=None):
        out_a, out_b = output.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a, c_a).chunk(2, 1)
            in_b = out_b * torch.exp(log_s) + t
        else:
            net_out = self.net(out_a, c_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


def change_order(x, c=None):
    x_a, x_b = x.chunk(2, 1)
    c_a, c_b = c.chunk(2, 1)
    return torch.cat([x_b, x_a], 1), torch.cat([c_b, c_a], 1)


class Flow(nn.Module):

    def __init__(self, in_channel, cin_channel, filter_size, num_layer, affine=True, pretrained=False):
        super().__init__()
        self.actnorm = ActNorm(in_channel, pretrained=pretrained)
        self.coupling = AffineCoupling(in_channel, cin_channel, filter_size=filter_size, num_layer=num_layer, affine=affine)

    def forward(self, x, c=None):
        out, logdet = self.actnorm(x)
        out, det = self.coupling(out, c)
        out, c = change_order(out, c)
        if det is not None:
            logdet = logdet + det
        return out, c, logdet

    def reverse(self, output, c=None):
        output, c = change_order(output, c)
        x = self.coupling.reverse(output, c)
        x = self.actnorm.reverse(x)
        return x, c


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):

    def __init__(self, in_channel, cin_channel, n_flow, n_layer, affine=True, pretrained=False, split=False):
        super().__init__()
        self.split = split
        squeeze_dim = in_channel * 2
        squeeze_dim_c = cin_channel * 2
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, squeeze_dim_c, filter_size=256, num_layer=n_layer, affine=affine, pretrained=pretrained))
        if self.split:
            self.prior = Wavenet(in_channels=squeeze_dim // 2, out_channels=squeeze_dim, num_blocks=1, num_layers=2, residual_channels=256, gate_channels=256, skip_channels=256, kernel_size=3, cin_channels=squeeze_dim_c, causal=False)

    def forward(self, x, c):
        b_size, n_channel, T = x.size()
        squeezed_x = x.view(b_size, n_channel, T // 2, 2).permute(0, 1, 3, 2)
        out = squeezed_x.contiguous().view(b_size, n_channel * 2, T // 2)
        squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
        c = squeezed_c.contiguous().view(b_size, -1, T // 2)
        logdet, log_p = 0, 0
        for flow in self.flows:
            out, c, det = flow(out, c)
            logdet = logdet + det
        if self.split:
            out, z = out.chunk(2, 1)
            mean, log_sd = self.prior(out, c).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd).sum()
        return out, c, logdet, log_p

    def reverse(self, output, c, eps=None):
        if self.split:
            mean, log_sd = self.prior(output, c).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)
            x = torch.cat([output, z_new], 1)
        else:
            x = output
        for flow in self.flows[::-1]:
            x, c = flow.reverse(x, c)
        b_size, n_channel, T = x.size()
        unsqueezed_x = x.view(b_size, n_channel // 2, 2, T).permute(0, 1, 3, 2)
        unsqueezed_x = unsqueezed_x.contiguous().view(b_size, n_channel // 2, T * 2)
        unsqueezed_c = c.view(b_size, -1, 2, T).permute(0, 1, 3, 2)
        unsqueezed_c = unsqueezed_c.contiguous().view(b_size, -1, T * 2)
        return unsqueezed_x, unsqueezed_c


class Flowavenet(nn.Module):

    def __init__(self, in_channel, cin_channel, n_block, n_flow, n_layer, affine=True, pretrained=False, block_per_split=8):
        super().__init__()
        self.block_per_split = block_per_split
        self.blocks = nn.ModuleList()
        self.n_block = n_block
        for i in range(self.n_block):
            split = False if (i + 1) % self.block_per_split or i == self.n_block - 1 else True
            self.blocks.append(Block(in_channel, cin_channel, n_flow, n_layer, affine=affine, pretrained=pretrained, split=split))
            cin_channel *= 2
            if not split:
                in_channel *= 2
        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0
        out = x
        c = self.upsample(c)
        for block in self.blocks:
            out, c, logdet_new, logp_new = block(out, c)
            logdet = logdet + logdet_new
            log_p_sum = log_p_sum + logp_new
        log_p_sum += 0.5 * (-log(2.0 * pi) - out.pow(2)).sum()
        logdet = logdet / (B * T)
        log_p = log_p_sum / (B * T)
        return log_p, logdet

    def reverse(self, z, c):
        _, _, T = z.size()
        _, _, t_c = c.size()
        if T != t_c:
            c = self.upsample(c)
        z_list = []
        x = z
        for i in range(self.n_block):
            b_size, _, T = x.size()
            squeezed_x = x.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            x = squeezed_x.contiguous().view(b_size, -1, T // 2)
            squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            c = squeezed_c.contiguous().view(b_size, -1, T // 2)
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                x, z = x.chunk(2, 1)
                z_list.append(z)
        for i, block in enumerate(self.blocks[::-1]):
            index = self.n_block - i
            if not (index % self.block_per_split or index == self.n_block):
                x, c = block.reverse(x, c, z_list[index // self.block_per_split - 1])
            else:
                x, c = block.reverse(x, c)
        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c


class ColorPrint(object):
    """ Colored printing functions for strings that use universal ANSI escape sequences.

    fail: bold red, pass: bold green, warn: bold yellow, 
    info: bold blue, bold: bold white

    :source: https://stackoverflow.com/a/47622205
    """

    @staticmethod
    def print_fail(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_pass(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_info(message, end='\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_major_fail(message, end='\n'):
        sys.stdout.write('\x1b[1;35m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)


class ConsoleLogger(object):

    @staticmethod
    def status(message):
        if os.name == 'nt':
            None
        else:
            ColorPrint.print_info('[~] {message}'.format(message=message))

    @staticmethod
    def success(message):
        if os.name == 'nt':
            None
        else:
            ColorPrint.print_pass('[+] {message}'.format(message=message))

    @staticmethod
    def error(message):
        if sys.exc_info()[2]:
            line = traceback.extract_tb(sys.exc_info()[2])[-1].lineno
            error_message = '[-] {message} with cause: {cause} (line {line})'.format(message=message, cause=str(sys.exc_info()[1]), line=line)
        else:
            error_message = '[-] {message}'.format(message=message)
        if os.name == 'nt':
            None
        else:
            ColorPrint.print_fail(error_message)

    @staticmethod
    def warn(message):
        if os.name == 'nt':
            None
        else:
            ColorPrint.print_warn('[-] {message}'.format(message=message))

    @staticmethod
    def critical(message):
        if sys.exc_info()[2]:
            line = traceback.extract_tb(sys.exc_info()[2])[-1].lineno
            error_message = '[!] {message} with cause: {cause} (line {line})'.format(message=message, cause=str(sys.exc_info()[1]), line=line)
        else:
            error_message = '[!] {message}'.format(message=message)
        if os.name == 'nt':
            None
        else:
            ColorPrint.print_major_fail(error_message)


class Conv1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class Residual(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False)
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)
        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)
        self._block = nn.Sequential(relu_1, conv_1, relu_2, conv_2)

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal)] * self._num_residual_layers)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ConvolutionalEncoder(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal, input_features_type, features_filters, sampling_rate, device, verbose=False):
        super(ConvolutionalEncoder, self).__init__()
        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """
        self._conv_1 = Conv1DBuilder.build(in_channels=features_filters, out_channels=num_hiddens, kernel_size=3, use_kaiming_normal=use_kaiming_normal, padding=1)
        self._conv_2 = Conv1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, use_kaiming_normal=use_kaiming_normal, padding=1)
        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=4, stride=2, use_kaiming_normal=use_kaiming_normal, padding=2)
        """
        2 convolutional layers with length 3 and
        residual connections.
        """
        self._conv_4 = Conv1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, use_kaiming_normal=use_kaiming_normal, padding=1)
        self._conv_5 = Conv1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, use_kaiming_normal=use_kaiming_normal, padding=1)
        """
        4 feedforward ReLu layers with residual connections.
        """
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens, use_kaiming_normal=use_kaiming_normal)
        self._input_features_type = input_features_type
        self._features_filters = features_filters
        self._sampling_rate = sampling_rate
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            ConsoleLogger.status('inputs size: {}'.format(inputs.size()))
        x_conv_1 = F.relu(self._conv_1(inputs))
        if self._verbose:
            ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))
        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            ConsoleLogger.status('_conv_2 output size: {}'.format(x.size()))
        x_conv_3 = F.relu(self._conv_3(x))
        if self._verbose:
            ConsoleLogger.status('_conv_3 output size: {}'.format(x_conv_3.size()))
        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self._verbose:
            ConsoleLogger.status('_conv_4 output size: {}'.format(x_conv_4.size()))
        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            ConsoleLogger.status('x_conv_5 output size: {}'.format(x_conv_5.size()))
        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._verbose:
            ConsoleLogger.status('_residual_stack output size: {}'.format(x.size()))
        return x


class ConvTranspose1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class GlobalConditioning(object):

    @staticmethod
    def compute(speaker_dic, speaker_ids, x_one_hot, device, gin_channels=128, expand=True):
        speakers_embedding = GlobalConditioning._Embedding(len(speaker_dic), gin_channels, padding_idx=None, std=0.1)
        B, _, T = x_one_hot.size()
        global_conditioning = speakers_embedding(speaker_ids.view(B, -1).long())
        global_conditioning = global_conditioning.transpose(1, 2)
        assert global_conditioning.dim() == 3
        """
        Return the global conditioning if the expand
        option is set to False
        """
        if not expand:
            return global_conditioning
        expanded_global_conditioning = GlobalConditioning._expand_global_features(B, T, global_conditioning, bct=True)
        return expanded_global_conditioning

    @staticmethod
    def _Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        m.weight.data.normal_(0, std)
        return m

    @staticmethod
    def _expand_global_features(B, T, g, bct=True):
        """
        Expand global conditioning features to all time steps

        Args:
            B (int): Batch size.
            T (int): Time length.
            g (Tensor): Global features, (B x C) or (B x C x 1).
            bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

        Returns:
            Tensor: B x C x T or B x T x C or None
        """
        if g is None:
            return None
        g = g.unsqueeze(-1) if g.dim() == 2 else g
        if bct:
            g_bct = g.expand(B, -1, T)
            return g_bct.contiguous()
        else:
            g_btc = g.expand(B, -1, T).transpose(1, 2)
            return g_btc.contiguous()


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()
        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, (i)] = original_quantized[:, :, (neighbor_index)]
        return quantized


class DeconvolutionalDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal, use_jitter, jitter_probability, use_speaker_conditioning, device, verbose=False):
        super(DeconvolutionalDecoder, self).__init__()
        self._use_jitter = use_jitter
        self._use_speaker_conditioning = use_speaker_conditioning
        self._device = device
        self._verbose = verbose
        if self._use_jitter:
            self._jitter = Jitter(jitter_probability)
        in_channels = in_channels + 40 if self._use_speaker_conditioning else in_channels
        self._conv_1 = Conv1DBuilder.build(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, padding=1, use_kaiming_normal=use_kaiming_normal)
        self._upsample = nn.Upsample(scale_factor=2)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens, use_kaiming_normal=use_kaiming_normal)
        self._conv_trans_1 = ConvTranspose1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, padding=1, use_kaiming_normal=use_kaiming_normal)
        self._conv_trans_2 = ConvTranspose1DBuilder.build(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, padding=0, use_kaiming_normal=use_kaiming_normal)
        self._conv_trans_3 = ConvTranspose1DBuilder.build(in_channels=num_hiddens, out_channels=out_channels, kernel_size=2, padding=0, use_kaiming_normal=use_kaiming_normal)

    def forward(self, inputs, speaker_dic, speaker_id):
        x = inputs
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] input size: {}'.format(x.size()))
        if self._use_jitter and self.training:
            x = self._jitter(x)
        if self._use_speaker_conditioning:
            speaker_embedding = GlobalConditioning.compute(speaker_dic, speaker_id, x, device=self._device, gin_channels=40, expand=True)
            x = torch.cat([x, speaker_embedding], dim=1)
        x = self._conv_1(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_1 output size: {}'.format(x.size()))
        x = self._upsample(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _upsample output size: {}'.format(x.size()))
        x = self._residual_stack(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _residual_stack output size: {}'.format(x.size()))
        x = F.relu(self._conv_trans_1(x))
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_1 output size: {}'.format(x.size()))
        x = F.relu(self._conv_trans_2(x))
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_2 output size: {}'.format(x.size()))
        x = self._conv_trans_3(x)
        if self._verbose:
            ConsoleLogger.status('[FEATURES_DEC] _conv_trans_3 output size: {}'.format(x.size()))
        return x


class VectorQuantizer(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.

    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, device):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._device = device

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """
        inputs = inputs.permute(1, 2, 0).contiguous()
        input_shape = inputs.shape
        _, time, batch_size = input_shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float)
        encodings.scatter_(1, encoding_indices, 1)
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2) for items in combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).view(batch_size, -1)
        else:
            encoding_distances = None
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [torch.dist(items[0], items[1], 2) for items in combinations(self._embedding.weight, r=2)]
            embedding_distances = torch.tensor(_embedding_distances)
        else:
            embedding_distances = None
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2) for items in product(flat_input, self._embedding.weight.detach())]
            frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).view(batch_size, time, -1)
        else:
            frames_vs_embedding_distances = None
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + commitment_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return vq_loss, quantized.permute(2, 0, 1).contiguous(), perplexity, encodings.view(batch_size, time, -1), distances.view(batch_size, time, -1), encoding_indices, {'e_latent_loss': e_latent_loss.item(), 'q_latent_loss': q_latent_loss.item(), 'commitment_loss': commitment_loss.item(), 'vq_loss': vq_loss.item()}, encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized

    @property
    def embedding(self):
        return self._embedding


class VectorQuantizerEMA(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.

    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937
    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.
    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms (see
            equation 4 in the paper).
        decay: float, decay for the moving averages.
        epsilon: small float constant to avoid numerical instability.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, device, epsilon=1e-05):
        super(VectorQuantizerEMA, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._device = device
        self._epsilon = epsilon

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.
        
        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """
        inputs = inputs.permute(1, 2, 0).contiguous()
        input_shape = inputs.shape
        _, time, batch_size = input_shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float)
        encodings.scatter_(1, encoding_indices, 1)
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2) for items in combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).view(batch_size, -1)
        else:
            encoding_distances = None
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [torch.dist(items[0], items[1], 2) for items in combinations(self._embedding.weight, r=2)]
            embedding_distances = torch.tensor(_embedding_distances)
        else:
            embedding_distances = None
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2) for items in product(flat_input, self._embedding.weight.detach())]
            frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).view(batch_size, time, -1)
        else:
            frames_vs_embedding_distances = None
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = commitment_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return vq_loss, quantized.permute(2, 0, 1).contiguous(), perplexity, encodings.view(batch_size, time, -1), distances.view(batch_size, time, -1), encoding_indices, {'vq_loss': vq_loss.item()}, encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized

    @property
    def embedding(self):
        return self._embedding


class ConvolutionalVQVAE(nn.Module):

    def __init__(self, configuration, device):
        super(ConvolutionalVQVAE, self).__init__()
        self._output_features_filters = configuration['output_features_filters'] * 3 if configuration['augment_output_features'] else configuration['output_features_filters']
        self._output_features_dim = configuration['output_features_dim']
        self._verbose = configuration['verbose']
        self._encoder = ConvolutionalEncoder(in_channels=configuration['input_features_dim'], num_hiddens=configuration['num_hiddens'], num_residual_layers=configuration['num_residual_layers'], num_residual_hiddens=configuration['num_hiddens'], use_kaiming_normal=configuration['use_kaiming_normal'], input_features_type=configuration['input_features_type'], features_filters=configuration['input_features_filters'] * 3 if configuration['augment_input_features'] else configuration['input_features_filters'], sampling_rate=configuration['sampling_rate'], device=device, verbose=self._verbose)
        self._pre_vq_conv = nn.Conv1d(in_channels=configuration['num_hiddens'], out_channels=configuration['embedding_dim'], kernel_size=3, padding=1)
        if configuration['decay'] > 0.0:
            self._vq = VectorQuantizerEMA(num_embeddings=configuration['num_embeddings'], embedding_dim=configuration['embedding_dim'], commitment_cost=configuration['commitment_cost'], decay=configuration['decay'], device=device)
        else:
            self._vq = VectorQuantizer(num_embeddings=configuration['num_embeddings'], embedding_dim=configuration['embedding_dim'], commitment_cost=configuration['commitment_cost'], device=device)
        self._decoder = DeconvolutionalDecoder(in_channels=configuration['embedding_dim'], out_channels=self._output_features_filters, num_hiddens=configuration['num_hiddens'], num_residual_layers=configuration['num_residual_layers'], num_residual_hiddens=configuration['residual_channels'], use_kaiming_normal=configuration['use_kaiming_normal'], use_jitter=configuration['use_jitter'], jitter_probability=configuration['jitter_probability'], use_speaker_conditioning=configuration['use_speaker_conditioning'], device=device, verbose=self._verbose)
        self._device = device
        self._record_codebook_stats = configuration['record_codebook_stats']

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x, speaker_dic, speaker_id):
        x = x.permute(0, 2, 1).contiguous().float()
        z = self._encoder(x)
        if self._verbose:
            ConsoleLogger.status('[ConvVQVAE] _encoder output size: {}'.format(z.size()))
        z = self._pre_vq_conv(z)
        if self._verbose:
            ConsoleLogger.status('[ConvVQVAE] _pre_vq_conv output size: {}'.format(z.size()))
        vq_loss, quantized, perplexity, _, _, encoding_indices, losses, _, _, _, concatenated_quantized = self._vq(z, record_codebook_stats=self._record_codebook_stats)
        reconstructed_x = self._decoder(quantized, speaker_dic, speaker_id)
        input_features_size = x.size(2)
        output_features_size = reconstructed_x.size(2)
        reconstructed_x = reconstructed_x.view(-1, self._output_features_filters, output_features_size)
        reconstructed_x = reconstructed_x[:, :, :-(output_features_size - input_features_size)]
        return reconstructed_x, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    m = conv.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(std_mul * (1.0 - dropout) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True, weight_normalization=True):
    """1-by-1 convolution layer
    """
    if weight_normalization:
        assert bias
        return Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias, std_mul=1.0)
    else:
        return conv.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


def ConvTranspose2d(in_channels, out_channels, kernel_size, weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    if weight_normalization:
        return nn.utils.weight_norm(m)
    else:
        return m


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def _conv1x1_forward(conv, x, is_incremental):
    """Conv1x1 forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualConv1dGLU(nn.Module):
    """Residual dilated conv1d + Gated linear unit

    Args:
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels. If None, set to same
          as ``residual_channels``.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        dropout (float): Dropout probability.
        padding (int): Padding for convolution layers. If None, proper padding
          is computed depends on dilation and kernel_size.
        dilation (int): Dilation factor.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
    """

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels=None, cin_channels=-1, gin_channels=-1, dropout=1 - 0.95, padding=None, dilation=1, causal=True, bias=True, weight_normalization=True, *args, **kwargs):
        super(ResidualConv1dGLU, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        if weight_normalization:
            assert bias
            self.conv = Conv1d(residual_channels, gate_channels, kernel_size, *args, padding=padding, dilation=dilation, bias=bias, std_mul=1.0, **kwargs)
        else:
            self.conv = conv.Conv1d(residual_channels, gate_channels, kernel_size, *args, padding=padding, dilation=dilation, bias=bias, **kwargs)
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=bias, weight_normalization=weight_normalization)
        else:
            self.conv1x1c = None
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=bias, weight_normalization=weight_normalization)
        else:
            self.conv1x1g = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias, weight_normalization=weight_normalization)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias, weight_normalization=weight_normalization)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """Forward

        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
            is_incremental (Bool) : Whether incremental mode or not

        Returns:
            Tensor: output
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            x = x[:, :, :residual.size(-1)] if self.causal else x
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb
        x = torch.tanh(a) * torch.sigmoid(b)
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)
        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip, self.conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Tensor): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Tensor: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda x: 2 ** x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


def to_one_hot(tensor, n, fill_with=1.0):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0):
    """
    Sample from discretized mixture of logistic distributions

    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value

    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = to_one_hot(argmax, nr_mix)
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
        legacy (bool) Use legacy code or not. Default is True for backward
          compatibility.
    """

    def __init__(self, out_channels=256, layers=20, stacks=2, residual_channels=512, gate_channels=512, skip_out_channels=512, kernel_size=3, dropout=1 - 0.95, cin_channels=-1, gin_channels=-1, n_speakers=None, weight_normalization=True, upsample_conditional_features=False, upsample_scales=None, freq_axis_kernel_size=3, scalar_input=False, use_speaker_embedding=True, legacy=True):
        super(WaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.legacy = legacy
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)
        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualConv1dGLU(residual_channels, gate_channels, kernel_size=kernel_size, skip_out_channels=skip_out_channels, bias=True, dilation=dilation, dropout=dropout, cin_channels=cin_channels, gin_channels=gin_channels, weight_normalization=weight_normalization)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([nn.ReLU(inplace=True), Conv1d1x1(skip_out_channels, skip_out_channels, weight_normalization=weight_normalization), nn.ReLU(inplace=True), Conv1d1x1(skip_out_channels, out_channels, weight_normalization=weight_normalization)])
        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None
        if upsample_conditional_features:
            self.upsample_conv = nn.ModuleList()
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s), padding=(freq_axis_padding, 0), dilation=1, stride=(1, s), weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                self.upsample_conv.append(nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None
        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1).long())
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_bct = _expand_global_features(B, T, g, bct=True)
        if c is not None and self.upsample_conv is not None:
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            c = c.squeeze(1)
            assert c.size(-1) == x.size(-1)
        x = self.first_conv(x)
        skips = None
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            if skips is None:
                skips = h
            else:
                skips += h
                if self.legacy:
                    skips *= math.sqrt(0.5)
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        x = F.softmax(x, dim=1) if softmax else x
        return x

    def incremental_forward(self, initial_input=None, c=None, g=None, T=100, test_inputs=None, tqdm=lambda x: x, softmax=True, quantize=True, log_scale_min=-7.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Tensor): Initial decoder input, (B x C x 1)
            c (Tensor): Local conditioning features, shape (B x C' x T)
            g (Tensor): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Tensor): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Tensor: Generated one-hot encoded samples. B x C x T　
              or scaler vector B x 1 x T
        """
        self.clear_buffer()
        B = 1
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            elif test_inputs.size(1) == self.out_channels:
                test_inputs = test_inputs.transpose(1, 2).contiguous()
            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(1)
            else:
                T = max(T, test_inputs.size(1))
        T = int(T)
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g, bct=False)
        if c is not None and self.upsample_conv is not None:
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            c = c.squeeze(1)
            assert c.size(-1) == T
        if c is not None and c.size(-1) == T:
            c = c.transpose(1, 2).contiguous()
        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = torch.zeros(B, 1, 1)
            else:
                initial_input = torch.zeros(B, 1, self.out_channels)
                initial_input[:, :, (127)] = 1
            if next(self.parameters()).is_cuda:
                initial_input = initial_input
        elif initial_input.size(1) == self.out_channels:
            initial_input = initial_input.transpose(1, 2).contiguous()
        current_input = initial_input
        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, (t), :].unsqueeze(1)
            elif t > 0:
                current_input = outputs[-1]
            ct = None if c is None else c[:, (t), :].unsqueeze(1)
            gt = None if g is None else g_btc[:, (t), :].unsqueeze(1)
            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = None
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                if self.legacy:
                    skips = h if skips is None else (skips + h) * math.sqrt(0.5)
                else:
                    skips = h if skips is None else skips + h
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)
            if self.scalar_input:
                x = sample_from_discretized_mix_logistic(x.view(B, -1, 1), log_scale_min=log_scale_min)
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
                if quantize:
                    sample = np.random.choice(np.arange(self.out_channels), p=x.view(-1).data.cpu().numpy())
                    x.zero_()
                    x[:, (sample)] = 1.0
            outputs += [x.data]
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()
        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


class WaveNetDecoder(nn.Module):

    def __init__(self, configuration, speaker_dic, device):
        super(WaveNetDecoder, self).__init__()
        self._use_jitter = configuration['use_jitter']
        if self._use_jitter:
            self._jitter = Jitter(configuration['jitter_probability'])
        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        """
        self._conv_1 = Conv1DBuilder.build(in_channels=64, out_channels=768, kernel_size=2, use_kaiming_normal=configuration['use_kaiming_normal'])
        self._wavenet = WaveNet(configuration['quantize'], configuration['n_layers'], configuration['n_loop'], configuration['residual_channels'], configuration['gate_channels'], configuration['skip_out_channels'], configuration['filter_size'], cin_channels=configuration['local_condition_dim'], gin_channels=configuration['global_condition_dim'], n_speakers=len(speaker_dic), upsample_conditional_features=True, upsample_scales=[2, 2, 2, 2, 2, 12])
        self._device = device

    def forward(self, y, local_condition, global_condition):
        if self._use_jitter and self.training:
            local_condition = self._jitter(local_condition)
        local_condition = self._conv_1(local_condition)
        x = self._wavenet(y, local_condition, global_condition)
        return x


class WaveNetVQVAE(nn.Module):

    def __init__(self, configuration, speaker_dic, device):
        super(WaveNetVQVAE, self).__init__()
        self._encoder = ConvolutionalEncoder(in_channels=configuration['input_features_dim'], num_hiddens=configuration['num_hiddens'], num_residual_layers=configuration['num_residual_layers'], num_residual_hiddens=configuration['residual_channels'], use_kaiming_normal=configuration['use_kaiming_normal'], input_features_type=configuration['input_features_type'], features_filters=configuration['input_features_filters'] * 3 if configuration['augment_input_features'] else configuration['input_features_filters'], sampling_rate=configuration['sampling_rate'], device=device)
        self._pre_vq_conv = nn.Conv1d(in_channels=configuration['num_hiddens'], out_channels=configuration['embedding_dim'], kernel_size=1, stride=1, padding=1)
        if configuration['decay'] > 0.0:
            self._vq = VectorQuantizerEMA(num_embeddings=configuration['num_embeddings'], embedding_dim=configuration['embedding_dim'], commitment_cost=configuration['commitment_cost'], decay=configuration['decay'], device=device)
        else:
            self._vq = VectorQuantizer(num_embeddings=configuration['num_embeddings'], embedding_dim=configuration['embedding_dim'], commitment_cost=configuration['commitment_cost'], device=device)
        self._decoder = WaveNetDecoder(configuration, speaker_dic, device)
        self._device = device
        self._record_codebook_stats = configuration['record_codebook_stats']

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x_enc, x_dec, global_condition):
        z = self._encoder(x_enc)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, _, _, encoding_indices, losses, _, _, _, concatenated_quantized = self._vq(z, record_codebook_stats=self._record_codebook_stats)
        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        x_dec = x_dec.squeeze(-1)
        reconstructed_x = self._decoder(x_dec, local_condition, global_condition)
        reconstructed_x = reconstructed_x.unsqueeze(-1)
        x_dec = x_dec.unsqueeze(-1)
        return reconstructed_x, x_dec, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, configuration, speaker_dic, device):
        model = WaveNetVQVAE(configuration, speaker_dic, device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActNorm,
     lambda: ([], {'in_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Block,
     lambda: ([], {'in_channel': 4, 'cin_channel': 4, 'n_flow': 4, 'n_layer': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     False),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ConvolutionalEncoder,
     lambda: ([], {'in_channels': 4, 'num_hiddens': 4, 'num_residual_layers': 1, 'num_residual_hiddens': 4, 'use_kaiming_normal': 4, 'input_features_type': 4, 'features_filters': 4, 'sampling_rate': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (Jitter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (KL_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'in_channels': 4, 'num_hiddens': 4, 'num_residual_hiddens': 4, 'use_kaiming_normal': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ResidualStack,
     lambda: ([], {'in_channels': 4, 'num_hiddens': 4, 'num_residual_layers': 1, 'num_residual_hiddens': 4, 'use_kaiming_normal': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (STFT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {}),
     False),
    (VectorQuantizer,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'commitment_cost': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (VectorQuantizerEMA,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'commitment_cost': 4, 'decay': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ZeroConv1d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
]

class Test_swasun_VQ_VAE_Speech(_paritybench_base):
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

