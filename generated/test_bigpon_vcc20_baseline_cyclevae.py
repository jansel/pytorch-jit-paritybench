import sys
_module = sys.modules[__name__]
del sys
runpwg = _module
calc_stats = _module
feature_extract = _module
gru_vae = _module
parallel_wavegan = _module
bin = _module
decode = _module
train = _module
datasets = _module
audio_world_dataset = _module
distributed = _module
launch = _module
layers = _module
residual_block = _module
residual_stack = _module
upsample = _module
losses = _module
stft_loss = _module
models = _module
melgan = _module
parallel_wavegan = _module
optimizers = _module
radam = _module
utils = _module
setup = _module
dataset = _module

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


import math


import numpy as np


import torch


import torch.multiprocessing as mp


from torch import nn


import itertools


import time


from torch.autograd import Variable


from torchvision import transforms


from torch.utils.data import DataLoader


import torch.nn.functional as F


from collections import defaultdict


import matplotlib


from torch.utils.data import Dataset


from sklearn.preprocessing import StandardScaler


from torch.optim.optimizer import Optimizer


class TwoSidedDilConv1d(nn.Module):
    """1D TWO-SIDED DILATED CONVOLUTION"""

    def __init__(self, in_dim=55, kernel_size=3, layers=2):
        super(TwoSidedDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.rec_field = self.kernel_size ** self.layers
        self.padding = int((self.rec_field - 1) / 2)
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim * self.kernel_size ** i, self.in_dim * self.kernel_size ** (i + 1), self.kernel_size, dilation=self.kernel_size ** i)]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim * self.kernel_size ** (i + 1), self.kernel_size, padding=self.padding)]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        x = self.conv[0](x)
        for i in range(1, self.layers):
            x = self.conv[i](x)
        return x


class CausalDilConv1d(nn.Module):
    """1D Causal DILATED CONVOLUTION"""

    def __init__(self, in_dim=11, kernel_size=2, layers=2):
        super(CausalDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.padding_list = [(self.kernel_size ** (i + 1) - self.kernel_size ** i) for i in range(self.layers)]
        logging.info(self.padding_list)
        self.padding = sum(self.padding_list)
        self.rec_field = self.padding + 1
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim * (sum(self.padding_list[:i]) + 1), self.in_dim * (sum(self.padding_list[:i + 1]) + 1), self.kernel_size, dilation=self.kernel_size ** i)]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim * (sum(self.padding_list[:i + 1]) + 1), self.kernel_size, padding=self.padding)]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        x = self.conv[0](x)
        for i in range(1, self.layers):
            x = self.conv[i](x)
        return x[:, :, :-self.padding]


class GRU_RNN(nn.Module):
    """GRU-RNN for FEATURE MAPPING

    Args:
        in_dim (int): input dimension
        out_dim (int): RNN output dimension
        hidden_units (int): GRU hidden units amount
        hidden_layers (int): GRU hidden layers amount
        kernel_size (int): kernel size for input convolutional layers
        dilation_size (int): dilation size for input convolutional layers
        do_prob (float): drop-out probability
        scale_in_flag (bool): flag to use input normalization layer
        scale_out_flag (bool): flag to use output de-normalization layer
        scale_in_out_flag (bool): flag to use output normalization layer for after performing input norm.
                                                                        (e.g., for Gaussian noise injection)
        [Weights & biases of norm/de-norm layers should be set with training data stats]
    """

    def __init__(self, in_dim=39, out_dim=35, hidden_units=1024, hidden_layers=1, kernel_size=3, dilation_size=2, do_prob=0, scale_in_flag=True, scale_out_flag=True, causal_conv=False, spk_dim=None):
        super(GRU_RNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.scale_in_flag = scale_in_flag
        self.scale_out_flag = scale_out_flag
        self.causal_conv = causal_conv
        self.spk_dim = spk_dim
        if self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)
        if not self.causal_conv:
            self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, layers=self.dilation_size)
        else:
            self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, layers=self.dilation_size)
        self.receptive_field = self.conv.rec_field
        self.tot_in_dim = self.in_dim * self.receptive_field + self.out_dim
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)
        self.out_1 = nn.Conv1d(self.hidden_units, self.out_dim, 1)
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.out_dim, self.out_dim, 1)

    def forward(self, x, y_in, h_in=None, do=False):
        """Forward calculation

        Args:
            x (Variable): float tensor variable with the shape  (T x C_in)

        Return:
            (Variable): float tensor variable with the shape (T x C_out)
        """
        if len(x.shape) > 2:
            batch_flag = True
            T = x.shape[1]
            if self.scale_in_flag:
                x_in = self.scale_in(x.transpose(1, 2))
            else:
                x_in = x.transpose(1, 2)
        else:
            batch_flag = False
            T = x.shape[0]
            if self.scale_in_flag:
                x_in = self.scale_in(torch.unsqueeze(x.transpose(0, 1), 0))
            else:
                x_in = torch.unsqueeze(x.transpose(0, 1), 0)
        if self.do_prob > 0 and do:
            x_conv = self.conv_drop(self.conv(x_in).transpose(1, 2))
        else:
            x_conv = self.conv(x_in).transpose(1, 2)
        if h_in is not None:
            out, h = self.gru(torch.cat((x_conv[:, :1], y_in), 2), h_in)
        else:
            out, h = self.gru(torch.cat((x_conv[:, :1], y_in), 2))
        if self.do_prob > 0 and do:
            y_in = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
        else:
            y_in = self.out_1(out.transpose(1, 2)).transpose(1, 2)
        if self.spk_dim is not None:
            y_in = torch.cat((F.selu(y_in[:, :, :self.spk_dim]), y_in[:, :, self.spk_dim:]), 2)
        trj = y_in
        if self.spk_dim is None:
            if self.do_prob > 0 and do:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    y_in = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                    trj = torch.cat((trj, y_in), 1)
            else:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    y_in = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                    trj = torch.cat((trj, y_in), 1)
        elif self.do_prob > 0 and do:
            for i in range(1, T):
                out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                y_in = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                y_in = torch.cat((F.selu(y_in[:, :, :self.spk_dim]), y_in[:, :, self.spk_dim:]), 2)
                trj = torch.cat((trj, y_in), 1)
        else:
            for i in range(1, T):
                out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                y_in = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                y_in = torch.cat((F.selu(y_in[:, :, :self.spk_dim]), y_in[:, :, self.spk_dim:]), 2)
                trj = torch.cat((trj, y_in), 1)
        if self.scale_out_flag:
            if batch_flag:
                trj_out = self.scale_out(trj.transpose(1, 2)).transpose(1, 2)
            else:
                trj_out = torch.squeeze(self.scale_out(trj.transpose(1, 2)).transpose(1, 2), 0)
            return trj_out, y_in, h
        else:
            if not batch_flag:
                trj = trj.view(-1, self.out_dim)
            return trj, y_in, h


class MCDloss(nn.Module):
    """ spectral loss based on mel-cepstrum distortion (MCD) """

    def __init__(self):
        super(MCDloss, self).__init__()
        self.frac10ln2 = 10.0 / 2.302585092994046
        self.sqrt2 = 1.4142135623730951

    def forward(self, x, y, twf=None, L2=False):
        """
            twf is time-warping function, none means exact same time-alignment
            L2 means using squared loss (L2-based loss), false means using abs./L1-based loss; default false
        """
        if twf is None:
            if not L2:
                mcd = self.frac10ln2 * self.sqrt2 * torch.sum(torch.abs(x - y), 1)
            else:
                mcd = self.frac10ln2 * torch.sqrt(2.0 * torch.sum((x - y).pow(2), 1))
        elif not L2:
            mcd = self.frac10ln2 * self.sqrt2 * torch.sum(torch.abs(torch.index_select(x, 0, twf) - y), 1)
        else:
            mcd = self.frac10ln2 * torch.sqrt(2.0 * torch.sum((torch.index_select(x, 0, twf) - y).pow(2), 1))
        mcd_sum = torch.sum(mcd)
        mcd_mean = torch.mean(mcd)
        mcd_std = torch.std(mcd)
        return mcd_sum, mcd_mean, mcd_std


def sampling_vae_laplace_batch(param, lat_dim=None):
    if lat_dim is None:
        lat_dim = int(param.shape[1] / 2)
    mu = param[:, :, :lat_dim]
    sigma = param[:, :, lat_dim:]
    eps = torch.empty(param.shape[0], param.shape[1], lat_dim).uniform_(-0.4999, 0.5)
    return mu - torch.exp(sigma) * eps.sign() * torch.log1p(-2 * eps.abs())


class GRU_RNN_STOCHASTIC(nn.Module):
    """STOCHASTIC GRU-RNN for FEATURE MAPPING

    Args:
        in_dim (int): input dimension
        out_dim (int): RNN output dimension
        hidden_units (int): GRU hidden units amount
        hidden_layers (int): GRU hidden layers amount
        kernel_size (int): kernel size for input convolutional layers
        dilation_size (int): dilation size for input convolutional layers
        do_prob (float): drop-out probability
        scale_in_flag (bool): flag to use input normalization layer
        scale_out_flag (bool): flag to use output de-normalization layer
        [Weights & biases of norm/de-norm layers should be set with training data stats]
    """

    def __init__(self, in_dim=55, out_dim=50, hidden_units=1024, hidden_layers=1, kernel_size=3, dilation_size=2, do_prob=0, spk_dim=None, scale_in_flag=True, scale_out_flag=True, causal_conv=False, arparam=True):
        super(GRU_RNN_STOCHASTIC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.scale_in_flag = scale_in_flag
        self.scale_out_flag = scale_out_flag
        self.spk_dim = spk_dim
        self.causal_conv = causal_conv
        if self.spk_dim is not None:
            self.mu_dim = self.spk_dim + self.out_dim
        else:
            self.mu_dim = self.out_dim
        self.arparam = arparam
        if self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)
        if not self.causal_conv:
            self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, layers=self.dilation_size)
        else:
            self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, layers=self.dilation_size)
        self.receptive_field = self.conv.rec_field
        if self.arparam:
            if self.spk_dim is not None:
                self.tot_in_dim = self.in_dim * self.receptive_field + self.out_dim * 2 + self.spk_dim
            else:
                self.tot_in_dim = self.in_dim * self.receptive_field + self.out_dim * 2
        elif self.spk_dim is not None:
            self.tot_in_dim = self.in_dim * self.receptive_field + self.out_dim + self.spk_dim
        else:
            self.tot_in_dim = self.in_dim * self.receptive_field + self.out_dim
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, batch_first=True)
        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)
        if self.spk_dim is not None:
            self.out_1 = nn.Conv1d(self.hidden_units, self.spk_dim + self.out_dim * 2, 1)
        else:
            self.out_1 = nn.Conv1d(self.hidden_units, self.out_dim * 2, 1)
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.out_dim, self.out_dim, 1)

    def forward(self, x, y_in, h_in=None, noise=0, do=False, sampling=True):
        """Forward calculation

        Args:
            x (Variable): float tensor variable with the shape  (T x C_in) or (B x T x C_in)

        Return:
            (Variable): float tensor variable with the shape (T x C_out) or (B x T x C_out)
        """
        if len(x.shape) > 2:
            batch_flag = True
            T = x.shape[1]
            if self.scale_in_flag:
                x_in = self.scale_in(x.transpose(1, 2))
            else:
                x_in = x.transpose(1, 2)
        else:
            batch_flag = False
            T = x.shape[0]
            if self.scale_in_flag:
                x_in = self.scale_in(torch.unsqueeze(x.transpose(0, 1), 0))
            else:
                x_in = torch.unsqueeze(x.transpose(0, 1), 0)
        if noise > 0:
            x_noise = torch.normal(mean=0, std=noise * torch.ones(x_in.shape[0], x_in.shape[1], x_in.shape[2]))
            x_in = x_in + x_noise
        if self.do_prob > 0 and do:
            x_conv = self.conv_drop(self.conv(x_in).transpose(1, 2))
        else:
            x_conv = self.conv(x_in).transpose(1, 2)
        if h_in is not None:
            out, h = self.gru(torch.cat((x_conv[:, :1], y_in), 2), h_in)
        else:
            out, h = self.gru(torch.cat((x_conv[:, :1], y_in), 2))
        if self.do_prob > 0 and do:
            out = self.gru_drop(out)
        out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
        if self.spk_dim is not None:
            out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
            if not self.arparam:
                if sampling:
                    out = sampling_vae_laplace_batch(out_param[:, :, self.spk_dim:], lat_dim=self.out_dim)
                else:
                    out = out[:, :, self.spk_dim:self.mu_dim]
        else:
            out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
            if not self.arparam:
                if sampling:
                    out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                else:
                    out = out[:, :, :self.mu_dim]
        trj_out_param = out_param
        if not self.arparam:
            trj_out = out
        if self.arparam:
            y_in = out_param
        elif self.spk_dim is not None:
            y_in = torch.cat((out_param[:, :, :self.spk_dim], out), 2)
        else:
            y_in = out
        if self.do_prob > 0 and do:
            if self.arparam:
                if self.spk_dim is not None:
                    for i in range(1, T):
                        out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                        out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                        trj_out_param = torch.cat((trj_out_param, out_param), 1)
                        y_in = out_param
                else:
                    for i in range(1, T):
                        out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                        out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                        trj_out_param = torch.cat((trj_out_param, out_param), 1)
                        y_in = out_param
            elif sampling:
                if self.spk_dim is not None:
                    for i in range(1, T):
                        out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                        out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                        out = sampling_vae_laplace_batch(out_param[:, :, self.spk_dim:], lat_dim=self.out_dim)
                        trj_out_param = torch.cat((trj_out_param, out_param), 1)
                        trj_out = torch.cat((trj_out, out), 1)
                        y_in = torch.cat((out_param[:, :, :self.spk_dim], out), 2)
                else:
                    for i in range(1, T):
                        out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                        out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                        out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                        trj_out_param = torch.cat((trj_out_param, out_param), 1)
                        trj_out = torch.cat((trj_out, out), 1)
                        y_in = out
            elif self.spk_dim is not None:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    out = out[:, :, self.spk_dim:self.mu_dim]
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    trj_out = torch.cat((trj_out, out), 1)
                    y_in = torch.cat((out_param[:, :, :self.spk_dim], out), 2)
            else:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(self.gru_drop(out).transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    out = out[:, :, :self.mu_dim]
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    trj_out = torch.cat((trj_out, out), 1)
                    y_in = out
        elif self.arparam:
            if self.spk_dim is not None:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    y_in = out_param
            else:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    y_in = out_param
        elif sampling:
            if self.spk_dim is not None:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    out = sampling_vae_laplace_batch(out_param[:, :, self.spk_dim:], lat_dim=self.out_dim)
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    trj_out = torch.cat((trj_out, out), 1)
                    y_in = torch.cat((out_param[:, :, :self.spk_dim], out), 2)
            else:
                for i in range(1, T):
                    out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                    out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                    out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                    out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                    trj_out_param = torch.cat((trj_out_param, out_param), 1)
                    trj_out = torch.cat((trj_out, out), 1)
                    y_in = out
        elif self.spk_dim is not None:
            for i in range(1, T):
                out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                out_param = torch.cat((F.selu(out[:, :, :self.spk_dim]), out[:, :, self.spk_dim:self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                out = out[:, :, self.spk_dim:self.mu_dim]
                trj_out_param = torch.cat((trj_out_param, out_param), 1)
                trj_out = torch.cat((trj_out, out), 1)
                y_in = torch.cat((out_param[:, :, :self.spk_dim], out), 2)
        else:
            for i in range(1, T):
                out, h = self.gru(torch.cat((x_conv[:, i:i + 1], y_in), 2), h)
                out = self.out_1(out.transpose(1, 2)).transpose(1, 2)
                out_param = torch.cat((out[:, :, :self.mu_dim], F.logsigmoid(out[:, :, self.mu_dim:])), 2)
                out = out[:, :, :self.mu_dim]
                trj_out_param = torch.cat((trj_out_param, out_param), 1)
                trj_out = torch.cat((trj_out, out), 1)
                y_in = out
        if self.spk_dim is not None:
            trj_map = trj_out_param[:, :, self.spk_dim:self.mu_dim]
        else:
            trj_map = trj_out_param[:, :, :self.mu_dim]
        if self.arparam:
            if self.spk_dim is not None:
                trj_out = trj_out_param[:, :, self.spk_dim:]
            else:
                trj_out = trj_out_param
            if sampling:
                trj_out = sampling_vae_laplace_batch(trj_out, lat_dim=self.out_dim)
            else:
                trj_out = trj_out[:, :, :self.out_dim]
        if self.scale_out_flag:
            trj_out = self.scale_out(trj_out.transpose(1, 2)).transpose(1, 2)
            trj_map = self.scale_out(trj_map.transpose(1, 2)).transpose(1, 2)
        if not batch_flag:
            trj_out = torch.squeeze(trj_out, 0)
            trj_out_param = torch.squeeze(trj_out_param, 0)
            trj_map = torch.squeeze(trj_map, 0)
        return trj_out, trj_out_param, y_in, h, trj_map


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

    def __init__(self, kernel_size=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, dropout=0.0, padding=None, dilation=1, bias=True, use_causal_conv=False):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            padding (int): Padding for convolution layers. If None, proper padding is
                computed depends on dilation and kernel_size.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.

        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        if padding is None:
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
        assert not use_causal_conv, 'Not supported yet.'
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        padding = (kernel_size - 1) // 2 * dilation
        self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), getattr(torch.nn, pad)(padding, **pad_params), torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
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

    def __init__(self, upsample_scales, upsample_activation='none', upsample_activation_params={}, mode='nearest', freq_axis_kernel_size=1, use_causal_conv=False):
        """Initialize upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1, mode)
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
            if upsample_activation != 'none':
                nonlinear = getattr(torch.nn, upsample_activation)(**upsample_activation_params)
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

    def __init__(self, upsample_scales, upsample_activation='none', upsample_activation_params={}, mode='nearest', freq_axis_kernel_size=1, aux_channels=80, aux_context_window=0, use_causal_conv=False):
        """Initialize convolution + upsampling network module.

        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
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
        self.upsample = UpsampleNetwork(upsample_scales=upsample_scales, upsample_activation=upsample_activation, upsample_activation_params=upsample_activation_params, mode=mode, freq_axis_kernel_size=freq_axis_kernel_size, use_causal_conv=use_causal_conv)

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

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

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
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
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


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self, in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True, upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_final_nolinear_activation=True, use_weight_norm=True, use_causal_conv=False):
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
            use_final_nolinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.

        """
        super(MelGANGenerator, self).__init__()
        assert not use_causal_conv, 'Not supported yet.'
        assert channels >= np.prod(upsample_scales)
        assert channels % 2 ** len(upsample_scales) == 0
        layers = []
        layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        for i, upsample_scale in enumerate(upsample_scales):
            if upsample_scale == 1:
                layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(in_channels=channels // 2 ** i, out_channels=channels // 2 ** (i + 1), kernel_size=3, stride=1, padding=1)]
            else:
                layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.ConvTranspose1d(in_channels=channels // 2 ** i, out_channels=channels // 2 ** (i + 1), kernel_size=upsample_scale * 2, stride=upsample_scale, padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // 2 ** (i + 1), dilation=stack_kernel_size ** j, bias=bias, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad, pad_params=pad_params, use_causal_conv=use_causal_conv)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias)]
        if use_final_nolinear_activation:
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
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

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
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, use_weight_norm=True, use_causal_conv=False, upsample_conditional_features=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}):
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
            if upsample_net == 'ConvInUpsampleNetwork':
                upsample_params.update({'aux_channels': aux_channels, 'aux_context_window': aux_context_window})
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=True, use_causal_conv=use_causal_conv)
            self.conv_layers += [conv]
        self.last_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, skip_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, out_channels, bias=True)])
        if use_weight_norm:
            self.apply_weight_norm()

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
            assert c.size(-1) == x.size(-1)
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

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=10, conv_channels=64, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, bias=True, use_weight_norm=True):
        """Initialize Parallel WaveGAN Discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (int): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i
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

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, dropout=0.0, use_weight_norm=True, use_causal_conv=False, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}):
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
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=-1, dilation=dilation, dropout=dropout, bias=True, use_causal_conv=use_causal_conv)
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

