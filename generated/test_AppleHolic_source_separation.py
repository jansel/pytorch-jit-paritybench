import sys
_module = sys.modules[__name__]
del sys
setup = _module
source_separation = _module
dataset = _module
hyperopt_run = _module
models = _module
modules = _module
settings = _module
synthesize = _module
train = _module
train_jointly = _module
trainer = _module

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


import numpy as np


from torch.utils.data import ConcatDataset


from typing import List


from typing import Tuple


from typing import Any


from typing import Callable


import torch


import torch.nn as nn


from typing import Dict


from torch.optim.lr_scheduler import MultiStepLR


import torch.nn.functional as F


from torch.nn.init import calculate_gain


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class _ComplexConvNd(nn.Module):
    """
    Implement Complex Convolution
    A: real weight
    B: img weight
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.transposed = transposed
        self.A = self.make_weight(in_channels, out_channels, kernel_size)
        self.B = self.make_weight(in_channels, out_channels, kernel_size)
        self.reset_parameters()

    def make_weight(self, in_ch, out_ch, kernel_size):
        if self.transposed:
            tensor = nn.Parameter(torch.Tensor(in_ch, out_ch // 2, *kernel_size))
        else:
            tensor = nn.Parameter(torch.Tensor(out_ch, in_ch // 2, *kernel_size))
        return tensor

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.A)
        gain = calculate_gain('leaky_relu', 0)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std
        with torch.no_grad():
            self.A.uniform_(-bound * (1 / np.pi ** 2), bound * (1 / np.pi ** 2))
            self.B.uniform_(-1 / np.pi, 1 / np.pi)


class ComplexConv1d(_ComplexConvNd):
    """
    Complex Convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = padding
        dilation = single(dilation)
        super(ComplexConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, single(0))

    def forward(self, x):
        """
        Implemented complex convolution using combining 'grouped convolution' and 'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex conved result
        """
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), 'reflect')
        real_part = F.conv1d(x, self.A, None, stride=self.stride, padding=0, dilation=self.dilation, groups=2)
        spl = self.in_channels // 2
        weight_B = torch.cat([self.B[:spl].data * -1, self.B[spl:].data])
        idea_part = F.conv1d(x, weight_B, None, stride=self.stride, padding=0, dilation=self.dilation, groups=2)
        return real_part + idea_part


class ComplexConvBlock(nn.Module):
    """
    Convolution block
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int=0, layers: int=4, bn_func=nn.BatchNorm1d, act_func=nn.LeakyReLU, skip_res: bool=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.skip_res = skip_res
        for idx in range(layers):
            in_ = in_channels if idx == 0 else out_channels
            self.blocks.append(nn.Sequential(*[bn_func(in_), act_func(), ComplexConv1d(in_, out_channels, kernel_size, padding=padding)]))

    def forward(self, x: torch.tensor) ->torch.tensor:
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)
        if temp.size() != x.size() or self.skip_res:
            return x
        else:
            return x + temp


class ComplexActLayer(nn.Module):
    """
    Activation differently 'real' part and 'img' part
    In implemented DCUnet on this repository, Real part is activated to log space.
    And Phase(img) part, it is distributed in [-pi, pi]...
    """

    def forward(self, x):
        real, img = x.chunk(2, 1)
        return torch.cat([F.leaky_relu_(real), torch.tanh(img) * np.pi], dim=1)


class ComplexTransposedConv1d(_ComplexConvNd):
    """
    Complex Transposed Convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = padding
        dilation = single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding)

    def forward(self, x, output_size=None):
        """
        Implemented complex transposed convolution using combining 'grouped convolution' and 'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex transposed convolution result
        """
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), 'reflect')
        real_part = F.conv_transpose1d(x, self.A, None, stride=self.stride, padding=0, dilation=self.dilation, groups=2)
        spl = self.out_channels // 2
        weight_B = torch.cat([self.B[:spl] * -1, self.B[spl:]])
        idea_part = F.conv_transpose1d(x, weight_B, None, stride=self.stride, padding=0, dilation=self.dilation, groups=2)
        if self.output_padding:
            real_part = F.pad(real_part, (self.output_padding, self.output_padding), 'reflect')
            idea_part = F.pad(idea_part, (self.output_padding, self.output_padding), 'reflect')
        return real_part + idea_part


class SpectrogramUnet(nn.Module):

    def __init__(self, spec_dim: int, hidden_dim: int, filter_len: int, hop_len: int, layers: int=3, block_layers: int=3, kernel_size: int=5, is_mask: bool=False, norm: str='bn', act: str='tanh'):
        super().__init__()
        self.layers = layers
        self.is_mask = is_mask
        self.stft = STFT(filter_len, hop_len)
        if norm == 'bn':
            self.bn_func = nn.BatchNorm1d
        elif norm == 'ins':
            self.bn_func = lambda x: nn.InstanceNorm1d(x, affine=True)
        else:
            raise NotImplementedError('{} is not implemented !'.format(norm))
        if act == 'tanh':
            self.act_func = nn.Tanh
            self.act_out = nn.Tanh
        elif act == 'comp':
            self.act_func = ComplexActLayer
            self.act_out = lambda : ComplexActLayer(is_out=True)
        else:
            raise NotImplementedError('{} is not implemented !'.format(act))
        self.prev_conv = ComplexConv1d(spec_dim * 2, hidden_dim, 1)
        self.down = nn.ModuleList()
        self.down_pool = nn.MaxPool1d(3, stride=2, padding=1)
        for idx in range(self.layers):
            block = ComplexConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func, layers=block_layers)
            self.down.append(block)
        self.up = nn.ModuleList()
        for idx in range(self.layers):
            in_c = hidden_dim if idx == 0 else hidden_dim * 2
            self.up.append(nn.Sequential(ComplexConvBlock(in_c, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func, layers=block_layers), self.bn_func(hidden_dim), self.act_func(), ComplexTransposedConv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)))
        self.out_conv = nn.Sequential(ComplexConvBlock(hidden_dim * 2, spec_dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func), self.bn_func(spec_dim * 2), self.act_func())
        self.refine_conv = nn.Sequential(ComplexConvBlock(spec_dim * 4, spec_dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func), self.bn_func(spec_dim * 2), self.act_func())

    def log_stft(self, wav):
        mag, phase = self.stft.transform(wav)
        return torch.log(mag + 1), phase

    def exp_istft(self, log_mag, phase):
        mag = np.e ** log_mag - 1
        wav = self.stft.inverse(mag, phase)
        return wav

    def adjust_diff(self, x, target):
        size_diff = target.size()[-1] - x.size()[-1]
        assert size_diff >= 0
        if size_diff > 0:
            x = F.pad(x.unsqueeze(1), (size_diff // 2, size_diff // 2), 'reflect').squeeze(1)
        return x

    def masking(self, mag, phase, origin_mag, origin_phase):
        abs_mag = torch.abs(mag)
        mag_mask = torch.tanh(abs_mag)
        phase_mask = mag / abs_mag
        mag = mag_mask * origin_mag
        phase = phase_mask * (origin_phase + phase)
        return mag, phase

    def forward(self, wav):
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)
        x = self.prev_conv(origin_x)
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(down_cache[self.layers - (idx + 1)], size=[x.size()[2]], mode='linear', align_corners=False)
            x = concat_complex(x, res, dim=1)
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(x, size=[origin_mag.size(2)], mode='linear', align_corners=False)
        x = self.refine_conv(concat_complex(x, origin_x))

        def to_wav(stft):
            mag, phase = stft.chunk(2, 1)
            if self.is_mask:
                mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            out = self.exp_istft(mag, phase)
            out = self.adjust_diff(out, wav)
            return out
        refine_wav = to_wav(x)
        return refine_wav


class RefineSpectrogramUnet(SpectrogramUnet):

    def __init__(self, spec_dim: int, hidden_dim: int, filter_len: int, hop_len: int, layers: int=4, block_layers: int=4, kernel_size: int=3, is_mask: bool=True, norm: str='ins', act: str='comp', refine_layers: int=1, add_spec_results: bool=False):
        super().__init__(spec_dim, hidden_dim, filter_len, hop_len, layers, block_layers, kernel_size, is_mask, norm, act)
        self.add_spec_results = add_spec_results
        self.refine_conv = nn.ModuleList([nn.Sequential(ComplexConvBlock(spec_dim * 2, spec_dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func), self.bn_func(spec_dim * 2), self.act_func())] * refine_layers)

    def forward(self, wav):
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)
        x = self.prev_conv(origin_x)
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(down_cache[self.layers - (idx + 1)], size=[x.size()[2]], mode='linear', align_corners=False)
            x = concat_complex(x, res, dim=1)
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(x, size=[origin_mag.size(2)], mode='linear', align_corners=False)
        for idx, refine_module in enumerate(self.refine_conv):
            x = refine_module(x)
            mag, phase = x.chunk(2, 1)
            mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            if idx < len(self.refine_conv) - 1:
                x = torch.cat([mag, phase], dim=1)
        phase = phase.clamp(-np.pi, np.pi)
        out = self.exp_istft(mag, phase)
        out = self.adjust_diff(out, wav)
        if self.add_spec_results:
            out = out, mag, phase
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ComplexActLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_AppleHolic_source_separation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

