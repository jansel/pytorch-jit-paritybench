import sys
_module = sys.modules[__name__]
del sys
datasets = _module
feature_utils = _module
melgan_vocoder = _module
model = _module
tfan_module = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils import weight_norm


import re


import torch.nn.utils.spectral_norm as spectral_norm


class Audio2Mel(nn.Module):

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None):
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window', window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), 'reflect').squeeze(1)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-05))
        return log_mel_spec


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(nn.LeakyReLU(0.2), nn.ReflectionPad1d(dilation), WNConv1d(dim, dim, kernel_size=3, dilation=dilation), nn.LeakyReLU(0.2), WNConv1d(dim, dim, kernel_size=1))
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()
        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=out_channels, affine=True))
        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=out_channels, affine=True))
        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding), nn.InstanceNorm1d(num_features=in_channels, affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)
        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


class TFAN_1D(nn.Module):
    """
    as paper said, it has best performance when N=3, kernal_size in h is 5
    """

    def __init__(self, norm_nc, ks=5, label_nc=128, N=3):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm1d(norm_nc, affine=False)
        self.repeat_N = N
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv1d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        temp = segmap
        for i in range(self.repeat_N):
            temp = self.mlp_shared(temp)
        actv = temp
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class TFAN_2D(nn.Module):
    """
    as paper said, it has best performance when N=3, kernal_size in h is 5
    """

    def __init__(self, norm_nc, ks=5, label_nc=128, N=3):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.repeat_N = N
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        temp = segmap
        for i in range(self.repeat_N):
            temp = self.mlp_shared(temp)
        actv = temp
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class downSample_Generator(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))

    def forward(self, input):
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))
        self.conv1_gates = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=1, padding=(2, 7))
        self.downSample1 = downSample_Generator(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.downSample2 = downSample_Generator(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.conv2dto1dLayer = nn.Conv1d(in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv2dto1dLayer_tfan = TFAN_1D(256)
        self.residualLayer1 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv1dto2dLayer = nn.Conv1d(in_channels=256, out_channels=2304, kernel_size=1, stride=1, padding=0)
        self.conv1dto2dLayer_tfan = TFAN_1D(2304)
        self.upSample1 = self.upSample(in_channels=256, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.upSample1_tfan = TFAN_2D(1024 // 4)
        self.glu = GLU()
        self.upSample2 = self.upSample(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.upSample2_tfan = TFAN_2D(512 // 4)
        self.lastConvLayer = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm1d(num_features=out_channels, affine=True), GLU())
        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.PixelShuffle(upscale_factor=2))
        return self.convLayer

    def forward(self, input):
        None
        input = input.unsqueeze(1)
        None
        seg_1d = input
        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))
        None
        downsample1 = self.downSample1(conv1)
        None
        downsample2 = self.downSample2(downsample1)
        None
        reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        conv2dto1d_layer = self.conv2dto1dLayer_tfan(conv2dto1d_layer, seg_1d)
        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        conv1dto2d_layer = self.conv1dto2dLayer_tfan(conv1dto2d_layer, seg_1d)
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 9, -1)
        seg_2d = reshape1dto2d
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_1 = self.upSample1_tfan(upsample_layer_1, seg_2d)
        upsample_layer_1 = self.glu(upsample_layer_1)
        upsample_layer_2 = self.upSample2(upsample_layer_1)
        upsample_layer_2 = self.upSample2_tfan(upsample_layer_2, seg_2d)
        upsample_layer_2 = self.glu(upsample_layer_2)
        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


class PixelShuffle(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), GLU())
        self.downSample1 = self.downSample(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.downSample2 = self.downSample(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=[2, 2], padding=1)
        self.downSample3 = self.downSample(in_channels=512, out_channels=1024, kernel_size=[3, 3], stride=[2, 2], padding=1)
        self.downSample4 = self.downSample(in_channels=1024, out_channels=1024, kernel_size=[1, 10], stride=(1, 1), padding=(0, 2))
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(1, 3), stride=[1, 1], padding=[0, 1]))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True), GLU())
        return convLayer

    def forward(self, input):
        input = input.unsqueeze(1)
        conv_layer_1 = self.convLayer1(input)
        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)
        output = torch.sigmoid(self.outputConvLayer(downsample3))
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelShuffle,
     lambda: ([], {'upscale_factor': 1.0}),
     lambda: ([torch.rand([4, 2, 8])], {}),
     True),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (downSample_Generator,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jackaduma_CycleGAN_VC3(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

