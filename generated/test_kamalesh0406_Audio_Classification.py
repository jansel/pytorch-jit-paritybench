import sys
_module = sys.modules[__name__]
del sys
dataloaders = _module
datasetaug = _module
datasetnormal = _module
models = _module
densenet = _module
inception = _module
resnet = _module
preprocessing = _module
preprocessingESC = _module
preprocessingGTZAN = _module
preprocessingUSC = _module
train = _module
utils = _module
validate = _module

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


from torch.utils.data import *


import torchvision


import pandas as pd


import numpy as np


import torch


import torchaudio


import random


import torch.nn as nn


import torchvision.models as models


import time


class DenseNet(nn.Module):

    def __init__(self, dataset, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 50 if dataset == 'ESC' else 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class Inception(nn.Module):

    def __init__(self, dataset, pretrained=True):
        super(Inception, self).__init__()
        num_classes = 50 if dataset == 'ESC' else 10
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class ResNet(nn.Module):

    def __init__(self, dataset, pretrained=True):
        super(ResNet, self).__init__()
        num_classes = 50 if dataset == 'ESC' else 10
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseNet,
     lambda: ([], {'dataset': torch.rand([4, 4, 4, 4])}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNet,
     lambda: ([], {'dataset': torch.rand([4, 4, 4, 4])}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_kamalesh0406_Audio_Classification(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

