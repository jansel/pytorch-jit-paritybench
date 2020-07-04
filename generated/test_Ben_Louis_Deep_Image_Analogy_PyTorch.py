import sys
_module = sys.modules[__name__]
del sys
DeepAnalogy = _module
PatchMatchOrig = _module
VGG19 = _module
lbfgs = _module
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


import torchvision.models as models


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import copy


class FeatureExtractor(nn.Sequential):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x):
        list = []
        for module in self._modules:
            x = self._modules[module](x)
            list.append(x)
        return list


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Ben_Louis_Deep_Image_Analogy_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(FeatureExtractor(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

