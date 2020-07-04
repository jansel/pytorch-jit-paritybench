import sys
_module = sys.modules[__name__]
del sys
capsnet = _module
capsule = _module
config = _module
loss = _module
main = _module
utils = _module

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


import torch.nn.functional as F


from torch import nn


from torch.autograd import Variable


_global_config['NUM_CLASSES'] = 4


class CapsuleNet(nn.Module):

    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size
            =9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8,
            num_route_nodes=-1, in_channels=256, out_channels=32,
            kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=config.NUM_CLASSES,
            num_route_nodes=32 * 6 * 6, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(nn.Linear(16 * config.NUM_CLASSES, 512
            ), nn.ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace
            =True), nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            if torch.is_available():
                y = Variable(torch.eye(config.NUM_CLASSES)).index_select(dim
                    =0, index=max_length_indices)
            else:
                y = Variable(torch.eye(config.NUM_CLASSES)).index_select(dim
                    =0, index=max_length_indices)
        reconstructions = self.decoder((x * y[:, :, (None)]).view(x.size(0),
            -1))
        return classes, reconstructions


_global_config['NUM_ROUTING_ITERATIONS'] = 4


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels,
        out_channels, kernel_size=None, stride=None, num_iterations=config.
        NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules,
                num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Conv2d(in_channels,
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=0) for _ in range(num_capsules)])

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[(None), :, :, (None), :] @ self.route_weights[:, (
                None), :, :, :]
            logits = Variable(torch.zeros(*priors.size()))
            if torch.is_available():
                logits = logits
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)
                    )
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in
                self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        margin_loss = margin_loss.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_leftthomas_CapsNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(CapsuleLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

