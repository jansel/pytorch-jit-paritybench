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


import random


DIM = 512


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(nn.Linear(128, 4 * 4 * 4 * DIM), nn.
            BatchNorm2d(4 * 4 * 4 * DIM), nn.ReLU(True))
        block1 = nn.Sequential(nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2,
            stride=2), nn.BatchNorm2d(2 * DIM), nn.ReLU(True))
        block2 = nn.Sequential(nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2
            ), nn.BatchNorm2d(DIM), nn.ReLU(True))
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)
        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(nn.Conv2d(3, DIM, 3, 2, padding=1), nn.
            LeakyReLU(), nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1), nn.
            LeakyReLU(), nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1), nn.
            LeakyReLU())
        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        output = self.linear(output)
        return output


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(nn.ReLU(True), nn.Conv1d(DIM, DIM, 5,
            padding=2), nn.ReLU(True), nn.Conv1d(DIM, DIM, 5, padding=2))

    def forward(self, input):
        output = self.res_block(input)
        return input + 0.3 * output


BATCH_SIZE = 256


SEQ_LEN = 32


DATA_DIR = './data_language'


MAX_N_EXAMPLES = 10000000


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, DIM * SEQ_LEN)
        self.block = nn.Sequential(ResBlock(), ResBlock(), ResBlock(),
            ResBlock(), ResBlock())
        self.conv1 = nn.Conv1d(DIM, len(charmap), 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(BATCH_SIZE * SEQ_LEN, -1)
        output = self.softmax(output)
        return output.view(shape)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(ResBlock(), ResBlock(), ResBlock(),
            ResBlock(), ResBlock())
        self.conv1d = nn.Conv1d(len(charmap), DIM, 1)
        self.linear = nn.Linear(SEQ_LEN * DIM, 1)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, SEQ_LEN * DIM)
        output = self.linear(output)
        return output


OUTPUT_DIM = 784


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(nn.Linear(128, 4 * 4 * 4 * DIM), nn.ReLU
            (True))
        block1 = nn.Sequential(nn.ConvTranspose2d(4 * DIM, 2 * DIM, 5), nn.
            ReLU(True))
        block2 = nn.Sequential(nn.ConvTranspose2d(2 * DIM, DIM, 5), nn.ReLU
            (True))
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, OUTPUT_DIM)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(nn.Conv2d(1, DIM, 5, stride=2, padding=2), nn.
            ReLU(True), nn.Conv2d(DIM, 2 * DIM, 5, stride=2, padding=2), nn
            .ReLU(True), nn.Conv2d(2 * DIM, 4 * DIM, 5, stride=2, padding=2
            ), nn.ReLU(True))
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * DIM)
        out = self.output(out)
        return out.view(-1)


FIXED_GENERATOR = False


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(
            DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True),
            nn.Linear(DIM, 2))
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
        main = nn.Sequential(nn.Linear(2, DIM), nn.ReLU(True), nn.Linear(
            DIM, DIM), nn.ReLU(True), nn.Linear(DIM, DIM), nn.ReLU(True),
            nn.Linear(DIM, 1))
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_caogang_wgan_gp(_paritybench_base):
    pass
    def test_000(self):
        self._check(Discriminator(*[], **{}), [torch.rand([2, 2])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Generator(*[], **{}), [torch.rand([2, 2]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ResBlock(*[], **{}), [torch.rand([4, 512, 64])], {})

