import sys
_module = sys.modules[__name__]
del sys
dataset = _module
device = _module
imageaug = _module
loss = _module
main = _module
metrics = _module
models = _module
base = _module
resnet = _module
tests = _module
center_loss_test = _module
trainer = _module
utils = _module

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


from torch import nn


device = torch.device('cuda')


class FaceModel(nn.Module):

    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        if num_classes:
            self.register_buffer('centers', (torch.rand(num_classes, feature_dim) - 0.5) * 2)
            self.classifier = nn.Linear(self.feature_dim, num_classes)

