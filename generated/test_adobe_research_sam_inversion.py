import sys
_module = sys.modules[__name__]
del sys
dataloader_utils = _module
img_utils = _module
latent_utils = _module
loss_utils = _module
mask_utils = _module
misc = _module
model_utils = _module
sam_inv_optimization = _module
segmenter_utils = _module
single_latent_inv = _module
train_invertibility = _module

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


import random


import torch


import torchvision


import numpy as np


from torch.nn import functional as F


import torchvision.transforms as transforms


import matplotlib.pyplot as plt


class layer_head(torch.nn.Module):

    def __init__(self):
        super(layer_head, self).__init__()
        self.m = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1), torch.nn.BatchNorm2d(4), torch.nn.LeakyReLU(0.2), torch.nn.ConvTranspose2d(4, 1, kernel_size=1, stride=1, padding=0), torch.nn.ReLU())

    def forward(self, x):
        return self.m(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (layer_head,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 4, 4])], {}),
     True),
]

class Test_adobe_research_sam_inversion(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

