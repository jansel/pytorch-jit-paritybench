import sys
_module = sys.modules[__name__]
del sys
gan_cifar10 = _module
gan_language = _module
gan_mnist = _module
gan_toy = _module
language_helpers = _module
tflib = _module
cifar10 = _module
inception_score = _module
mnist = _module
ops = _module
batchnorm = _module
conv1d = _module
conv2d = _module
deconv2d = _module
layernorm = _module
linear = _module
plot = _module
save_images = _module
small_imagenet = _module

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


import time


import numpy as np


import torch


import torchvision


from torch import nn


from torch import autograd


from torch import optim


import torch.autograd as autograd


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from sklearn.preprocessing import OneHotEncoder


import sklearn.datasets


import random


DIM = 512


FIXED_GENERATOR = False


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, 2))
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True), nn.Linear(DIM, 1))
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(nn.ReLU(True), nn.Conv1d(DIM, DIM, 5, padding=2), nn.ReLU(True), nn.Conv1d(DIM, DIM, 5, padding=2))

    def forward(self, input):
        output = self.res_block(input)
        return input + 0.3 * output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([2, 2])], {}),
     True),
    (Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([2, 2]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64])], {}),
     True),
]

class Test_caogang_wgan_gp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

