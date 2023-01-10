import sys
_module = sys.modules[__name__]
del sys
amazing = _module
data = _module
models = _module
pit_criterion = _module
preprocess = _module
separate = _module
solver = _module
train = _module
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


import torch


import math


import numpy as np


import torch.utils.data as data


import torch.nn as nn


import torch.nn.functional as F


import numpy


from numpy.lib import stride_tricks


from torch.autograd import Variable


from itertools import permutations


import time


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size, dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True))
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-08))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-08))
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)
            row_output = self.row_rnn[i](row_input)
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            row_output = self.row_norm[i](row_output)
            output = output + row_output
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)
            col_output = self.col_rnn[i](col_input)
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + col_output
        output = self.output(output)
        return output


class DPRNN_base(nn.Module):

    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=4, segment_size=100, bidirectional=True, rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.eps = 1e-08
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)
        self.DPRNN = DPRNN(rnn_type, self.feature_dim, self.hidden_dim, self.feature_dim * self.num_spk, num_layers=layer, bidirectional=bidirectional)

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

    def split_feature(self, input, segment_size):
        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)
        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
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
        pass


class BF_module(DPRNN_base):

    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, E, seq_length = input.shape
        enc_feature = self.BN(input)
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)
        output = self.DPRNN(enc_segments).view(batch_size * self.num_spk, self.feature_dim, self.segment_size, -1)
        output = self.merge_feature(output, enc_rest)
        bf_filter = self.output(output) * self.output_gate(output)
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1, self.feature_dim)
        return bf_filter


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
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)
    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()
    frame = frame.contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Decoder(nn.Module):

    def __init__(self, E, W):
        super(Decoder, self).__init__()
        self.E, self.W = E, W
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [B, E, L]
            est_mask: [B, C, E, L]
        Returns:
            est_source: [B, C, T]
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(source_w, 2, 3)
        est_source = self.basis_signals(source_w)
        est_source = overlap_and_add(est_source, self.W // 2)
        return est_source


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        self.W, self.N = W, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


class FaSNet_base(nn.Module):

    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=250, nspk=2, win_len=2):
        super(FaSNet_base, self).__init__()
        self.window = win_len
        self.stride = self.window // 2
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-08
        self.encoder = Encoder(win_len, enc_dim)
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-08)
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim, self.num_spk, self.layer, self.segment_size)
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)
        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        B, _ = input.size()
        mixture_w = self.encoder(input)
        score_ = self.enc_LN(mixture_w)
        score_ = self.separator(score_)
        score_ = score_.view(B * self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()
        score = self.mask_conv1x1(score_)
        score = score.view(B, self.num_spk, self.enc_dim, -1)
        est_mask = F.relu(score)
        est_source = self.decoder(mixture_w, est_mask)
        return est_source


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BF_module,
     lambda: ([], {'input_dim': 4, 'feature_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DPRNN_base,
     lambda: ([], {'input_dim': 4, 'feature_dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'E': 4, 'W': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FaSNet_base,
     lambda: ([], {'enc_dim': 4, 'feature_dim': 4, 'hidden_dim': 4, 'layer': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_ShiZiqiang_dual_path_RNNs_DPRNNs_based_speech_separation(_paritybench_base):
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

