import sys
_module = sys.modules[__name__]
del sys
const = _module
human = _module
lib = _module
dataset = _module
evaluate = _module
game = _module
go = _module
gtp = _module
play = _module
process = _module
train = _module
utils = _module
main = _module
models = _module
agent = _module
feature = _module
mcts = _module
policy = _module
value = _module
purge = _module
viewer = _module

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


import torch


import torch.nn as nn


import numpy as np


import time


import torch.nn.functional as F


from copy import deepcopy


from torch.autograd import Variable


from torch.utils.data import DataLoader


class AlphaLoss(torch.nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas
    
    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, winner, self_play_winner, probas, self_play_probas):
        value_error = (self_play_winner - winner) ** 2
        policy_error = torch.sum(-self_play_probas * (1e-06 + probas).log(), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
        return total_error


class BasicBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


BLOCKS = 10


class Extractor(nn.Module):
    """
    This network is used as a feature extractor, takes as input the 'state' defined in
    the AlphaGo Zero paper
    - The state of the past n turns of the board (7 in the paper) for each player.
      This means that the first n matrices of the input state will be 1 and 0, where 1
      is a stone. 
      This is done to take into consideration Go rules (repetitions are forbidden) and
      give a sense of time

    - The color of the stone that is next to play. This could have been a single bit, but
      for implementation purposes, it is actually expended to the whole matrix size.
      If it is black turn, then the last matrix of the input state will be a NxN matrix
      full of 1, where N is the size of the board, 19 in the case of AlphaGo.
      This is done to take into consideration the komi.

    The ouput is a series of feature maps that retains the meaningful informations
    contained in the input state in order to make a good prediction on both which is more
    likely to win the game from the current state, and also which move is the best one to
    make. 
    """

    def __init__(self, inplanes, outplanes):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        for block in range(BLOCKS):
            setattr(self, 'res{}'.format(block), BasicBlock(outplanes, outplanes))

    def forward(self, x):
        """
        x : tensor representing the state
        feature_maps : result of the residual layers forward pass
        """
        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS - 1):
            x = getattr(self, 'res{}'.format(block))(x)
        feature_maps = getattr(self, 'res{}'.format(BLOCKS - 1))(x)
        return feature_maps


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    """

    def __init__(self, inplanes, outplanes):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(outplanes - 1, outplanes)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = self.fc(x)
        probas = self.logsoftmax(x).exp()
        return probas


class ValueNet(nn.Module):
    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1. 
    """

    def __init__(self, inplanes, outplanes):
        super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(outplanes - 1, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = F.relu(self.fc1(x))
        winning = F.tanh(self.fc2(x))
        return winning


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlphaLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([16, 4]), torch.rand([16, 4]), torch.rand([64, 4]), torch.rand([64, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Extractor,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_dylandjian_SuperGo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

