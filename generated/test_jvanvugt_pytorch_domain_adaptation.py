import sys
_module = sys.modules[__name__]
del sys
adda = _module
config = _module
data = _module
models = _module
revgrad = _module
test_model = _module
train_source = _module
utils = _module
wdgrl = _module

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


from torch import nn


from torch.utils.data import DataLoader


from torchvision.datasets import MNIST


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


import numpy as np


from torch.utils.data import Dataset


from torchvision import datasets


from torchvision import transforms


import torch.nn.functional as F


from torch.utils.data.sampler import SubsetRandomSampler


from torch.autograd import Function


from torch.autograd import grad


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(3, 10, kernel_size=5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(10, 20, kernel_size=5), nn.MaxPool2d(2), nn.Dropout2d())
        self.classifier = nn.Sequential(nn.Linear(320, 50), nn.ReLU(), nn.Dropout(), nn.Linear(50, 10))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):

    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GradientReversal,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_jvanvugt_pytorch_domain_adaptation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

