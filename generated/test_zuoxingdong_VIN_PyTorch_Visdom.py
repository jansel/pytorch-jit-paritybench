import sys
_module = sys.modules[__name__]
del sys
VIN = _module
dataset = _module
run = _module
vis = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


import time


import torchvision.transforms as transforms


def attention(tensor, params):
    """Attention model for grid world
    """
    S1, S2, args = params
    num_data = tensor.size()[0]
    slice_s1 = S1.expand(args.imsize, 1, args.ch_q, num_data)
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_s1).squeeze(2)
    slice_s2 = S2.expand(1, args.ch_q, num_data)
    slice_s2 = slice_s2.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_s2).squeeze(2)
    return q_out


class VIN(nn.Module):
    """Value Iteration Network architecture"""

    def __init__(self, args):
        super(VIN, self).__init__()
        self.conv_h = nn.Conv2d(in_channels=args.ch_i, out_channels=args.ch_h, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True)
        self.conv_r = nn.Conv2d(in_channels=args.ch_h, out_channels=1, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.conv_q = nn.Conv2d(in_channels=2, out_channels=args.ch_q, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.fc1 = nn.Linear(in_features=args.ch_q, out_features=8, bias=False)
        self.grid_image = None
        self.reward_image = None
        self.value_images = []

    def forward(self, X, S1, S2, args, record_images=False):
        h = self.conv_h(X)
        r = self.conv_r(h)
        if record_images:
            self.grid_image = X.data[0].cpu().numpy()
            self.reward_image = r.data[0].cpu().numpy()
        v = torch.zeros(r.size())
        v = v if X.is_cuda else v
        v = Variable(v)
        for _ in range(args.k):
            rv = torch.cat([r, v], 1)
            q = self.conv_q(rv)
            v, _ = torch.max(q, 1)
            if record_images:
                self.value_images.append(v.data[0].cpu().numpy())
        rv = torch.cat([r, v], 1)
        q = self.conv_q(rv)
        q_out = attention(q, [S1.long(), S2.long(), args])
        logits = self.fc1(q_out)
        return logits

