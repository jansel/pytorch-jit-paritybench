import sys
_module = sys.modules[__name__]
del sys
conv_tasnet = _module
data = _module
evaluate = _module
pit_criterion = _module
preprocess = _module
separate = _module
solver = _module
train = _module
utils = _module
learn_conv1d = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from itertools import permutations


import time


class ConvTasNet(nn.Module):

    def __init__(self, N, L, B, H, P, X, R, C, norm_type='gLN', causal=
        False, mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ConvTasNet, self).__init__()
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = (N,
            L, B, H, P, X, R, C)
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type,
            causal, mask_nonlinear)
        self.decoder = Decoder(N, L)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['L'], package['B'], package['H'],
            package['P'], package['X'], package['R'], package['C'],
            norm_type=package['norm_type'], causal=package['causal'],
            mask_nonlinear=package['mask_nonlinear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C,
            'norm_type': model.norm_type, 'causal': model.causal,
            'mask_nonlinear': model.mask_nonlinear, 'state_dict': model.
            state_dict(), 'optim_dict': optimizer.state_dict(), 'epoch': epoch}
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=
            False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


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
    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame,
        subframe_step)
    frame = signal.new_tensor(frame).long()
    frame = frame.contiguous().view(-1)
    result = signal.new_zeros(*outer_dimensions, output_subframes,
        subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class Decoder(nn.Module):

    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask
        source_w = torch.transpose(source_w, 2, 3)
        est_source = self.basis_signals(source_w)
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source


class TemporalConvNet(nn.Module):

    def __init__(self, N, B, H, P, X, R, C, norm_type='gLN', causal=False,
        mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        layer_norm = ChannelwiseLayerNorm(N)
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation if causal else (P - 1
                    ) * dilation // 2
                blocks += [TemporalBlock(B, H, P, stride=1, padding=padding,
                    dilation=dilation, norm_type=norm_type, causal=causal)]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1,
            temporal_conv_net, mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)
        score = score.view(M, self.C, N, K)
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        else:
            raise ValueError('Unsupported mask non-linear function')
        return est_mask


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == 'gLN':
        return GlobalLayerNorm(channel_size)
    elif norm_type == 'cLN':
        return ChannelwiseLayerNorm(channel_size)
    else:
        return nn.BatchNorm1d(channel_size)


class TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, norm_type='gLN', causal=False):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels,
            kernel_size, stride, padding, dilation, norm_type, causal)
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        return out + residual


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, norm_type='gLN', causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            in_channels, bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
                pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm,
                pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


EPS = 1e-08


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = torch.pow(y - mean, 2).mean(dim=1, keepdim=True).mean(dim=2,
            keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kaituoxu_Conv_TasNet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ChannelwiseLayerNorm(*[], **{'channel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Chomp1d(*[], **{'chomp_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Decoder(*[], **{'N': 4, 'L': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DepthwiseSeparableConv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}), [torch.rand([4, 4, 64])], {})

    def test_004(self):
        self._check(Encoder(*[], **{'L': 4, 'N': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(GlobalLayerNorm(*[], **{'channel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(TemporalConvNet(*[], **{'N': 4, 'B': 4, 'H': 4, 'P': 4, 'X': 4, 'R': 4, 'C': 4}), [torch.rand([4, 4, 2])], {})

