import sys
_module = sys.modules[__name__]
del sys
VBMF = _module
dataset = _module
decompositions = _module
main = _module

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


import numpy as np


import torch


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data as data


import torchvision.datasets as datasets


import torchvision.models as models


import torchvision.transforms as transforms


from torch.autograd import Variable


from torchvision import models


import torchvision


import torch.nn.functional as F


import time


from itertools import chain


class ModifiedVGG16Model(torch.nn.Module):

    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ModifiedVGG16Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 243, 243])], {}),
     True),
]

class Test_jacobgil_pytorch_tensor_decompositions(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

