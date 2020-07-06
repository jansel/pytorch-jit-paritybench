import sys
_module = sys.modules[__name__]
del sys
CartoonGAN = _module
edge_promoting = _module
networks = _module
test = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.optim as optim


from torchvision import transforms


import torch.nn.functional as F


import itertools


import random


import numpy as np


from torchvision import datasets


from torch.autograd import Variable


class resnet_block(nn.Module):

    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)
        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))
        return input + x


class generator(nn.Module):

    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(nn.Conv2d(in_nc, nf, 7, 1, 3), nn.InstanceNorm2d(nf), nn.ReLU(True), nn.Conv2d(nf, nf * 2, 3, 2, 1), nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), nn.InstanceNorm2d(nf * 2), nn.ReLU(True), nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), nn.InstanceNorm2d(nf * 4), nn.ReLU(True))
        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, 3, 1, 1))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.up_convs = nn.Sequential(nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), nn.InstanceNorm2d(nf * 2), nn.ReLU(True), nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), nn.Conv2d(nf, nf, 3, 1, 1), nn.InstanceNorm2d(nf), nn.ReLU(True), nn.Conv2d(nf, out_nc, 7, 1, 3), nn.Tanh())
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)
        return output


class discriminator(nn.Module):

    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(nf, nf * 2, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.InstanceNorm2d(nf * 4), nn.LeakyReLU(0.2, True), nn.Conv2d(nf * 4, nf * 4, 3, 2, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(nf * 4, nf * 8, 3, 1, 1), nn.InstanceNorm2d(nf * 8), nn.LeakyReLU(0.2, True), nn.Conv2d(nf * 8, nf * 8, 3, 1, 1), nn.InstanceNorm2d(nf * 8), nn.LeakyReLU(0.2, True), nn.Conv2d(nf * 8, out_nc, 3, 1, 1), nn.Sigmoid())
        utils.initialize_weights(self)

    def forward(self, input):
        output = self.convs(input)
        return output


class VGG19(nn.Module):

    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([25088, 25088])], {}),
     False),
]

class Test_znxlwm_pytorch_CartoonGAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

