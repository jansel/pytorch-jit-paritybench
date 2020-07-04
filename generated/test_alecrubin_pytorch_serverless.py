import sys
_module = sys.modules[__name__]
del sys
api = _module
predict = _module
fastai = _module
conv_builder = _module
core = _module
imports = _module
initializers = _module
layers = _module
model = _module
models = _module
resnext_101_32x4d = _module
resnext_101_64x4d = _module
resnext_50_32x4d = _module
torch_imports = _module
transforms = _module
lib = _module
utils = _module
setup = _module

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


from torch import nn


import warnings


import torchvision


from torch.autograd import Variable


from torch.nn.init import kaiming_normal


from torchvision.transforms import Compose


from torchvision.models import resnet18


from torchvision.models import resnet34


from torchvision.models import resnet50


from torchvision.models import resnet101


from torchvision.models import resnet152


from torchvision.models import vgg16_bn


from torchvision.models import vgg19_bn


from torchvision.models import densenet121


from torchvision.models import densenet161


from torchvision.models import densenet169


from torchvision.models import densenet201


class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_alecrubin_pytorch_serverless(_paritybench_base):
    pass
    def test_000(self):
        self._check(AdaptiveConcatPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Lambda(*[], **{'f': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

