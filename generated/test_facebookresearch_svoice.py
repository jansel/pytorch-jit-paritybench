import sys
_module = sys.modules[__name__]
del sys
make_dataset = _module
svoice = _module
data = _module
audio = _module
data = _module
preprocess = _module
distrib = _module
evaluate = _module
evaluate_auto_select = _module
executor = _module
models = _module
sisnr_loss = _module
swave = _module
separate = _module
solver = _module
utils = _module
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


import math


import torchaudio


import torch as th


from torch.nn import functional as F


import logging


import re


import numpy as np


import torch


import torch.utils.data as data


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torch.nn.parallel.distributed import DistributedDataParallel


from itertools import permutations


import torch.nn.functional as F


import torch.nn as nn


from torch.autograd import Variable


import time


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


import functools


import inspect


class MulCatBlock(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=False):
        super(MulCatBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = nn.LSTM(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.gate_rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.gate_rnn_proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input):
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.rnn_proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape).contiguous()
        gate_rnn_output, _ = self.gate_rnn(output)
        gate_rnn_output = self.gate_rnn_proj(gate_rnn_output.contiguous().view(-1, gate_rnn_output.shape[2])).view(output.shape).contiguous()
        gated_output = torch.mul(rnn_output, gate_rnn_output)
        gated_output = torch.cat([gated_output, output], 2)
        gated_output = self.block_projection(gated_output.contiguous().view(-1, gated_output.shape[2])).view(output.shape)
        return gated_output


class ByPass(nn.Module):

    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


class DPMulCat(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_spk, dropout=0, num_layers=1, bidirectional=True, input_normalize=False):
        super(DPMulCat, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers
        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])
        for i in range(num_layers):
            self.rows_grnn.append(MulCatBlock(input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.cols_grnn.append(MulCatBlock(input_size, hidden_size, dropout, bidirectional=bidirectional))
            if self.in_norm:
                self.rows_normalization.append(nn.GroupNorm(1, input_size, eps=1e-08))
                self.cols_normalization.append(nn.GroupNorm(1, input_size, eps=1e-08))
            else:
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1))

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1).contiguous()
            row_output = self.rows_normalization[i](row_output)
            if self.training:
                output = output + row_output
            else:
                output += row_output
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2).contiguous()
            col_output = self.cols_normalization[i](col_output).contiguous()
            if self.training:
                output = output + col_output
            else:
                output += col_output
            output_i = self.output(output)
            if self.training or i == self.num_layers - 1:
                output_all.append(output_i)
        return output_all


class Separator(nn.Module):

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_spk=2, layer=4, segment_size=100, input_normalize=False, bidirectional=True):
        super(Separator, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.input_normalize = input_normalize
        self.rnn_model = DPMulCat(self.feature_dim, self.hidden_dim, self.feature_dim, self.num_spk, num_layers=layer, bidirectional=bidirectional, input_normalize=input_normalize)

    def pad_segment(self, input, segment_size):
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def create_chuncks(self, input, segment_size):
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
        return segments.contiguous(), rest

    def merge_chuncks(self, input, rest):
        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]
        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()

    def forward(self, input):
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        output_all = self.rnn_model(enc_segments)
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length
    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.clone().detach().long()
    frame = frame.contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Decoder(nn.Module):

    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = nn.AvgPool2d((1, self.L))(est_source)
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source


class Encoder(nn.Module):

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv(mixture))
        return mixture_w


def capture_init(init):
    """
    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = args, kwargs
        init(self, *args, **kwargs)
    return __init__


class SWave(nn.Module):

    @capture_init
    def __init__(self, N, L, H, R, C, sr, segment, input_normalize):
        super(SWave, self).__init__()
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = N, L, H, R, C, sr, segment
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        self.segment_size = int(np.sqrt(2 * self.sr * self.segment / (self.L / 2)))
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(self.filter_dim + self.N, self.N, self.H, self.filter_dim, self.num_spk, self.layer, self.segment_size, self.input_normalize)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)
        T_mix = mixture.size(-1)
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(mixture.shape[0], self.C, self.N, mixture_w.shape[2])
            output_ii = self.decoder(output_ii)
            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            outputs.append(output_ii)
        return torch.stack(outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ByPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DPMulCat,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4, 'num_spk': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'L': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {'L': 4, 'N': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MulCatBlock,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SWave,
     lambda: ([], {'N': 4, 'L': 4, 'H': 4, 'R': 4, 'C': 4, 'sr': 4, 'segment': 4, 'input_normalize': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Separator,
     lambda: ([], {'input_dim': 4, 'feature_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_facebookresearch_svoice(_paritybench_base):
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

