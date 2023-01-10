import sys
_module = sys.modules[__name__]
del sys
examples = _module
example_gan = _module
models = _module
acgan = _module
began = _module
dcgan = _module
gan = _module
lsgan = _module
wgan = _module
wgan_gp = _module

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


from collections import OrderedDict


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from torchvision.datasets import MNIST


class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'img_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_nocotan_pytorch_lightning_gans(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

