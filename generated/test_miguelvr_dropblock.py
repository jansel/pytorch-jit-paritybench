import sys
_module = sys.modules[__name__]
del sys
dropblock = _module
dropblock = _module
scheduler = _module
setup = _module
test_dropblock2d = _module
test_dropblock3d = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


from torch import nn


import numpy as np


import time


import torch.nn as nn


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim(
            ) == 4, 'Expected input with 4 dimensions (bsize, channels, height, width)'
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            gamma = self._compute_gamma(x)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[:, (None), :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, (None), :, :], kernel_size=
            (self.block_size, self.block_size), stride=(1, 1), padding=self
            .block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / self.block_size ** 2


class LinearScheduler(nn.Module):

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value,
            num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_miguelvr_dropblock(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DropBlock2D(*[], **{'drop_prob': 4, 'block_size': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(LinearScheduler(*[], **{'dropblock': ReLU(), 'start_value': 4, 'stop_value': 4, 'nr_steps': 4}), [torch.rand([4, 4, 4, 4])], {})

