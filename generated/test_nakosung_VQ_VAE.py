import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
logger = _module
main = _module
model = _module
setup = _module
solver = _module

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


from torch.utils import data


from torchvision import transforms


import numpy as np


import torch


from torch.backends import cudnn


import torch.nn as nn


import math


from torch.autograd import Variable


from torchvision.utils import save_image


class Generator(nn.Module):
    """Generator. Vector Quantised Variational Auto-Encoder."""

    def __init__(self, image_size=64, z_dim=256, conv_dim=64, code_dim=16, k_dim=256):
        super(Generator, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.code_dim = code_dim
        self.dict = nn.Embedding(k_dim, z_dim)
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU())
        repeat_num = int(math.log2(image_size / code_dim))
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, conv_dim * (i + 2), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i + 2)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i + 2)
        layers.append(nn.Conv2d(curr_dim, z_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)
        layers = []
        layers.append(nn.ConvTranspose2d(z_dim, curr_dim, kernel_size=1))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU())
        for i in reversed(range(repeat_num)):
            layers.append(nn.ConvTranspose2d(curr_dim, conv_dim * (i + 1), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i + 1)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i + 1)
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        initrange = 1.0 / self.k_dim
        self.dict.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        h = self.encoder(x)
        sz = h.size()
        org_h = h
        h = h.permute(0, 2, 3, 1)
        h = h.contiguous()
        Z = h.view(-1, self.z_dim)
        W = self.dict.weight

        def L2_dist(a, b):
            return (a - b) ** 2
        j = L2_dist(Z[:, (None)], W[(None), :]).sum(2).min(1)[1]
        W_j = W[j]
        Z_sg = Z.detach()
        W_j_sg = W_j.detach()
        h = W_j.view(sz[0], sz[2], sz[3], sz[1])
        h = h.permute(0, 3, 1, 2)

        def hook(grad):
            nonlocal org_h
            self.saved_grad = grad
            self.saved_h = org_h
            return grad
        h.register_hook(hook)
        return self.decoder(h), L2_dist(Z, W_j_sg).sum(1).mean(), L2_dist(Z_sg, W_j).sum(1).mean()

    def bwd(self):
        self.saved_h.backward(self.saved_grad)

    def decode(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.decoder(z)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_nakosung_VQ_VAE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

