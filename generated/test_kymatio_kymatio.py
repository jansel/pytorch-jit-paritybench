import sys
_module = sys.modules[__name__]
del sys
benchmarks = _module
common = _module
scattering1d = _module
scattering2d = _module
scattering3d = _module
conf = _module
classif_keras = _module
plot_classif_torch = _module
plot_filters = _module
plot_real_signal = _module
plot_synthetic = _module
reconstruct_torch = _module
cifar_resnet_torch = _module
cifar_small_sample = _module
cifar_torch = _module
long_mnist_classify_torch = _module
mnist_keras = _module
plot_invert_scattering_torch = _module
plot_scattering_disk = _module
plot_sklearn = _module
regularized_inverse_scattering_MNIST_torch = _module
scattering3d_qm7_torch = _module
kymatio = _module
backend = _module
base_backend = _module
numpy_backend = _module
tensorflow_backend = _module
torch_backend = _module
torch_skcuda_backend = _module
caching = _module
datasets = _module
frontend = _module
base_frontend = _module
entry = _module
keras_frontend = _module
numpy_frontend = _module
sklearn_frontend = _module
tensorflow_frontend = _module
torch_frontend = _module
keras = _module
numpy = _module
torch_backend = _module
core = _module
filter_bank = _module
utils = _module
torch_backend = _module
sklearn = _module
tensorflow = _module
torch = _module
version = _module
setup = _module
test_torch_backend = _module
test_filters_scattering1d = _module
test_numpy_scattering1d = _module
test_tensorflow_scattering1d = _module
test_torch_scattering1d = _module
test_utils_scattering1d = _module
test_frontend_scattering2d = _module
test_keras_scattering2d = _module
test_numpy_scattering2d = _module
test_sklearn_2d = _module
test_tensorflow_scattering2d = _module
test_torch_scattering2d = _module
test_numpy_scattering3d = _module
test_tensorflow_scattering3d = _module
test_torch_scattering3d = _module
test_utils_scattering3d = _module

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


from torch.nn import Linear


from torch.nn import NLLLoss


from torch.nn import LogSoftmax


from torch.nn import Sequential


from torch.optim import Adam


from scipy.io import wavfile


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.optim


from torchvision import datasets


from torchvision import transforms


from numpy.random import RandomState


import math


from torch import optim


from scipy.misc import face


import torch.optim as optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


from collections import namedtuple


from torch.nn import ReflectionPad2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Scattering2dResNet(nn.Module):

    def __init__(self, in_channels, k=2, n=4, num_classes=10):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(nn.BatchNorm2d(in_channels, eps=
            1e-05, affine=False), nn.Conv2d(in_channels, self.ichannels,
            kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d
            (self.ichannels), nn.ReLU(True))
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.K, 8, 8)
        x = self.init_conv(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Scattering2dResNet(nn.Module):

    def __init__(self, in_channels, k=2, n=4, num_classes=10, standard=False):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        if standard:
            self.init_conv = nn.Sequential(nn.Conv2d(3, self.ichannels,
                kernel_size=3, stride=1, padding=1, bias=False), nn.
                BatchNorm2d(self.ichannels), nn.ReLU(True))
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(nn.BatchNorm2d(in_channels, eps=
                1e-05, affine=False), nn.Conv2d(in_channels, self.ichannels,
                kernel_size=3, stride=1, padding=1, bias=False), nn.
                BatchNorm2d(self.ichannels), nn.ReLU(True))
            self.standard = False
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            x = x.view(x.size(0), self.K, 8, 8)
        x = self.init_conv(x)
        if self.standard:
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Scattering2dCNN(nn.Module):
    """
        Simple CNN with 3x3 convs based on VGG
    """

    def __init__(self, in_channels, classifier_type='cnn'):
        super(Scattering2dCNN, self).__init__()
        self.in_channels = in_channels
        self.classifier_type = classifier_type
        self.build()

    def build(self):
        cfg = [256, 256, 256, 'M', 512, 512, 512, 1024, 1024]
        layers = []
        self.K = self.in_channels
        self.bn = nn.BatchNorm2d(self.K)
        if self.classifier_type == 'cnn':
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(self.in_channels, v, kernel_size=3,
                        padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                        ]
                    self.in_channels = v
            layers += [nn.AdaptiveAvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(1024 * 4, 10)
        elif self.classifier_type == 'mlp':
            self.classifier = nn.Sequential(nn.Linear(self.K * 8 * 8, 1024),
                nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024,
                10))
            self.features = None
        elif self.classifier_type == 'linear':
            self.classifier = nn.Linear(self.K * 8 * 8, 10)
            self.features = None

    def forward(self, x):
        x = self.bn(x.view(-1, self.K, 8, 8))
        if self.features:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class View(nn.Module):

    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


class Generator(nn.Module):

    def __init__(self, num_input_channels, num_hidden_channels,
        num_output_channels=1, filter_size=3):
        super(Generator, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.filter_size = filter_size
        self.build()

    def build(self):
        padding = (self.filter_size - 1) // 2
        self.main = nn.Sequential(nn.ReflectionPad2d(padding), nn.Conv2d(
            self.num_input_channels, self.num_hidden_channels, self.
            filter_size, bias=False), nn.BatchNorm2d(self.
            num_hidden_channels, eps=0.001, momentum=0.9), nn.ReLU(inplace=
            True), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False), nn.ReflectionPad2d(padding), nn.Conv2d(
            self.num_hidden_channels, self.num_hidden_channels, self.
            filter_size, bias=False), nn.BatchNorm2d(self.
            num_hidden_channels, eps=0.001, momentum=0.9), nn.ReLU(inplace=
            True), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False), nn.ReflectionPad2d(padding), nn.Conv2d(
            self.num_hidden_channels, self.num_output_channels, self.
            filter_size, bias=False), nn.BatchNorm2d(self.
            num_output_channels, eps=0.001, momentum=0.9), nn.Tanh())

    def forward(self, input_tensor):
        return self.main(input_tensor)


def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')


class ScatteringTorch(nn.Module):

    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""
        input_checks(x)
        return self.scattering(x)
    _doc_array = 'torch.Tensor'
    _doc_array_n = ''
    _doc_alias_name = 'forward'
    _doc_alias_call = '.forward'
    _doc_frontend_paragraph = """
        This class inherits from `torch.nn.Module`. As a result, it has all
        the same capabilities, including transferring the object to the GPU
        using the `cuda` or `to` methods. This object would then take GPU
        tensors as input and output the scattering coefficients of those
        tensors.
        """
    _doc_sample = 'torch.randn({shape})'
    _doc_has_shape = True
    _doc_has_out_type = True


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kymatio_kymatio(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Generator(*[], **{'num_input_channels': 4, 'num_hidden_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Scattering2dCNN(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Scattering2dResNet(*[], **{'in_channels': 4}), [torch.rand([4, 4, 8, 8])], {})

    def test_005(self):
        self._check(View(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

