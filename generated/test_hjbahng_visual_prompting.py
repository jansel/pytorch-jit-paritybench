import sys
_module = sys.modules[__name__]
del sys
main_clip = _module
main_vision = _module
prompters = _module
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


import time


import random


import torch


import torch.backends.cudnn as cudnn


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.utils.data import DataLoader


from torchvision.datasets import CIFAR100


import numpy as np


import torchvision.models as models


import torchvision.transforms as transforms


import torch.nn as nn


class PadPrompter(nn.Module):

    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


class FixedPatchPrompter(nn.Module):

    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize])
        prompt[:, :, :self.psize, :self.psize] = self.patch
        return x + prompt


class RandomPatchPrompter(nn.Module):

    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)
        prompt = torch.zeros([1, 3, self.isize, self.isize])
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        return x + prompt


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FixedPatchPrompter,
     lambda: ([], {'args': _mock_config(image_size=4, prompt_size=4)}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (PadPrompter,
     lambda: ([], {'args': _mock_config(prompt_size=4, image_size=8)}),
     lambda: ([torch.rand([4, 3, 8, 8])], {}),
     True),
]

class Test_hjbahng_visual_prompting(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

