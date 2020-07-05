import sys
_module = sys.modules[__name__]
del sys
envs = _module
main = _module
model = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.multiprocessing as mp


from torch.autograd import Variable


def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * F.elu(x, alpha)


class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space, small_net=False):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.small_net = small_net
        if self.small_net:
            self.linear1 = nn.Linear(num_inputs, 64)
            self.linear2 = nn.Linear(64, 64)
            self.actor_linear = nn.Linear(64, num_outputs)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
            self.actor_linear = nn.Linear(256, num_outputs)
        self.train()

    def forward(self, inputs):
        if self.small_net:
            x = selu(self.linear1(inputs))
            x = selu(self.linear2(x))
            return self.actor_linear(x)
        else:
            inputs, (hx, cx) = inputs
            x = selu(self.conv1(inputs))
            x = selu(self.conv2(x))
            x = selu(self.conv3(x))
            x = selu(self.conv4(x))
            x = x.view(-1, 32 * 3 * 3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
            return self.actor_linear(x), (hx, cx)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]

