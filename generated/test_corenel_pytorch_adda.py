import sys
_module = sys.modules[__name__]
del sys
core = _module
adapt = _module
pretrain = _module
test = _module
datasets = _module
mnist = _module
usps = _module
main = _module
models = _module
discriminator = _module
lenet = _module
params = _module
utils = _module

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


import torch


import torch.optim as optim


from torch import nn


import torch.nn as nn


from torchvision import datasets


from torchvision import transforms


import numpy as np


import torch.utils.data as data


import torch.nn.functional as F


import random


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.restored = False
        self.layer = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.ReLU(), nn.Linear(hidden_dims, hidden_dims), nn.ReLU(), nn.Linear(hidden_dims, output_dims), nn.LogSoftmax())

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()
        self.restored = False
        self.encoder = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5), nn.MaxPool2d(kernel_size=2), nn.ReLU(), nn.Conv2d(20, 50, kernel_size=5), nn.Dropout2d(), nn.MaxPool2d(kernel_size=2), nn.ReLU())
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'input_dims': 4, 'hidden_dims': 4, 'output_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_corenel_pytorch_adda(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

