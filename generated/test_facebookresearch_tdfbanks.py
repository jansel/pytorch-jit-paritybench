import sys
_module = sys.modules[__name__]
del sys
main = _module
melfilters = _module
model = _module
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


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


class TDFbanks(nn.Module):

    def __init__(self, mode, nfilters, samplerate=16000, wlen=25, wstride=10, compression='log', preemp=False, mvn=False):
        super(TDFbanks, self).__init__()
        window_size = samplerate * wlen // 1000 + 1
        window_stride = samplerate * wstride // 1000
        padding_size = (window_size - 1) // 2
        self.preemp = None
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, 1, padding=1, groups=1, bias=False)
        self.complex_conv = nn.Conv1d(1, 2 * nfilters, window_size, 1, padding=padding_size, groups=1, bias=False)
        self.modulus = nn.LPPool1d(2, 2, stride=2)
        self.lowpass = nn.Conv1d(nfilters, nfilters, window_size, window_stride, padding=0, groups=nfilters, bias=False)
        if mode == 'Fixed':
            for param in self.parameters():
                param.requires_grad = False
        elif mode == 'learnfbanks':
            if preemp:
                self.preemp.weight.requires_grad = False
            self.lowpass.weight.requires_grad = False
        if mvn:
            self.instancenorm = nn.InstanceNorm1d(nfilters, momentum=1)
        self.nfilters = nfilters
        self.fs = samplerate
        self.wlen = wlen
        self.wstride = wstride
        self.compression = compression
        self.mvn = mvn

    def initialize(self, min_freq=0, max_freq=8000, nfft=512, window_type='hanning', normalize_energy=False, alpha=0.97):
        if self.preemp:
            self.preemp.weight.data[0][0][0] = -alpha
            self.preemp.weight.data[0][0][1] = 1
        self.complex_init = melfilters.Gabor(self.nfilters, min_freq, max_freq, self.fs, self.wlen, self.wstride, nfft, normalize_energy)
        for idx, gabor in enumerate(self.complex_init.gaborfilters):
            self.complex_conv.weight.data[2 * idx][0].copy_(torch.from_numpy(np.real(gabor)))
            self.complex_conv.weight.data[2 * idx + 1][0].copy_(torch.from_numpy(np.imag(gabor)))
        self.lowpass_init = utils.window(window_type, self.fs * self.wlen // 1000 + 1)
        for idx in range(self.nfilters):
            self.lowpass.weight.data[idx][0].copy_(torch.from_numpy(self.lowpass_init))

    def forward(self, x):
        x = x.view(1, 1, -1)
        if self.preemp:
            x = self.preemp(x)
        x = self.complex_conv(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x.pow(2), 2, 2, 0, False).mul(2)
        x = x.transpose(1, 2)
        x = self.lowpass(x)
        x = x.abs()
        x = x + 1
        if self.compression == 'log':
            x = x.log()
        if self.mvn:
            x = self.instancenorm(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TDFbanks,
     lambda: ([], {'mode': 4, 'nfilters': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_facebookresearch_tdfbanks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

