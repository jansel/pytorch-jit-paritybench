import sys
_module = sys.modules[__name__]
del sys
data = _module
base_data_loader = _module
custom_dataset_data_loader = _module
data_loader = _module
dataset = _module
image_folder = _module
unpack_mnist = _module
models = _module
base_model = _module
networks = _module
test_model = _module
wgan_gp_model = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
get_data = _module
html = _module
png = _module
util = _module
visualizer = _module

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


import torch.utils.data


import random


import torchvision.transforms as transforms


import torch


import torch.utils.data as data


import torch.nn as nn


from torch.nn import init


import functools


from torch.autograd import Variable


from torch.optim import lr_scheduler


import numpy as np


from collections import OrderedDict


from torch.autograd import grad


import inspect


import re


import collections


class Generator(nn.Module):

    def __init__(self, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        preprocess = nn.Sequential(nn.Linear(128, 4 * 4 * 4 * ngf), nn.ReLU(True))
        block1 = nn.Sequential(nn.ConvTranspose2d(4 * ngf, 2 * ngf, 5), nn.ReLU(True))
        block2 = nn.Sequential(nn.ConvTranspose2d(2 * ngf, ngf, 5), nn.ReLU(True))
        deconv_out = nn.ConvTranspose2d(ngf, 1, 8, stride=2)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.ngf, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, self.OUTPUT_DIM)


class Discriminator(nn.Module):

    def __init__(self, ndf, gpu_ids=[]):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.gpu_ids = gpu_ids
        main = nn.Sequential(nn.Conv2d(1, ndf, 5, stride=2, padding=2), nn.ReLU(True), nn.Conv2d(ndf, 2 * ndf, 5, stride=2, padding=2), nn.ReLU(True), nn.Conv2d(2 * ndf, 4 * ndf, 5, stride=2, padding=2), nn.ReLU(True))
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * ndf, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.ndf)
        out = self.output(out)
        return out.view(-1)


class NoiseZ(nn.Module):

    def __init__(self, batchSize):
        super(NoiseZ, self).__init__()
        self.Z = nn.Parameter(torch.randn(batchSize, 128), requires_grad=True)

    def forward(self, input):
        out = self.Z * input
        return out


class Generator_Z(nn.Module):

    def __init__(self, batchSize, OUTPUT_DIM, ngf, gpu_ids=[]):
        super(Generator_Z, self).__init__()
        self.OUTPUT_DIM = OUTPUT_DIM
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        preprocess = nn.Sequential(nn.Linear(128, 4 * 4 * 4 * ngf), nn.ReLU(True))
        block1 = nn.Sequential(nn.ConvTranspose2d(4 * ngf, 2 * ngf, 5), nn.ReLU(True))
        block2 = nn.Sequential(nn.ConvTranspose2d(2 * ngf, ngf, 5), nn.ReLU(True))
        deconv_out = nn.ConvTranspose2d(ngf, 1, 8, stride=2)
        self.noise = NoiseZ(batchSize)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input_noise = self.noise(input)
        output = self.preprocess(input_noise)
        output = output.view(-1, 4 * self.ngf, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, self.OUTPUT_DIM)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'ndf': 4}),
     lambda: ([torch.rand([4, 1, 28, 28])], {}),
     True),
    (Generator_Z,
     lambda: ([], {'batchSize': 4, 'OUTPUT_DIM': 4, 'ngf': 4}),
     lambda: ([torch.rand([4, 4, 4, 128])], {}),
     True),
    (NoiseZ,
     lambda: ([], {'batchSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 128])], {}),
     True),
]

class Test_oyxhust_wgan_gp_anomaly(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

