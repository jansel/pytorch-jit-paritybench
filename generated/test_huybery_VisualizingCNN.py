import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
vgg16_conv = _module
vgg16_deconv = _module
utils = _module
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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


from torch.autograd import Variable


from torchvision.transforms import transforms


import numpy as np


from functools import partial


import torchvision.models as models


import torchvision


from collections import OrderedDict


class Vgg16Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=1000):
        """
        Input
            number of class, default is 1k.
        """
        super(Vgg16Conv, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2, return_indices=True), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2, return_indices=True), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2, return_indices=True), nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2, return_indices=True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, stride=2, return_indices=True))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, num_cls), nn.Softmax(dim=1))
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
        for idx, layer in enumerate(vgg16_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data

    def check(self):
        model = models.vgg16(pretrained=True)
        return model

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output


class Vgg16Deconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """

    def __init__(self):
        super(Vgg16Deconv, self).__init__()
        self.features = nn.Sequential(nn.MaxUnpool2d(2, stride=2), nn.ReLU(), nn.ConvTranspose2d(512, 512, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(512, 512, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(512, 512, 3, padding=1), nn.MaxUnpool2d(2, stride=2), nn.ReLU(), nn.ConvTranspose2d(512, 512, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(512, 512, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(512, 256, 3, padding=1), nn.MaxUnpool2d(2, stride=2), nn.ReLU(), nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(256, 256, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(256, 128, 3, padding=1), nn.MaxUnpool2d(2, stride=2), nn.ReLU(), nn.ConvTranspose2d(128, 128, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 3, padding=1), nn.MaxUnpool2d(2, stride=2), nn.ReLU(), nn.ConvTranspose2d(64, 64, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(64, 3, 3, padding=1))
        self.conv2deconv_indices = {(0): 30, (2): 28, (5): 25, (7): 23, (10): 20, (12): 18, (14): 16, (17): 13, (19): 11, (21): 9, (24): 6, (26): 4, (28): 2}
        self.unpool2pool_indices = {(26): 4, (21): 9, (14): 16, (7): 23, (0): 30}
        self.init_weight()

    def init_weight(self):
        vgg16_pretrained = models.vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data

    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')
        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x

