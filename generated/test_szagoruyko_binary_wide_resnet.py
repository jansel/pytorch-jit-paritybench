import sys
_module = sys.modules[__name__]
del sys
evaluate_packed = _module
main = _module
wrn_mcdonnell = _module

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


import numpy as np


import torch


from torch.optim import SGD


from torch.optim.lr_scheduler import CosineAnnealingLR


import torch.utils.data


from torch.utils.data import DataLoader


from torch.nn.functional import cross_entropy


from torch.nn import DataParallel


from torch.backends import cudnn


from collections import OrderedDict


import math


from torch import nn


import torch.nn.functional as F


class ForwardSign(torch.autograd.Function):
    """Fake sign op for 1-bit weights.

    See eq. (1) in https://arxiv.org/abs/1802.08530

    Does He-init like forward, and nothing on backward.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])
            ) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g


class ModuleBinarizable(nn.Module):

    def __init__(self, binarize=False):
        super().__init__()
        self.binarize = binarize

    def _get_weight(self, name):
        w = getattr(self, name)
        return ForwardSign.apply(w) if self.binarize else w

    def forward(self):
        pass


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_szagoruyko_binary_wide_resnet(_paritybench_base):
    pass
    def test_000(self):
        self._check(ModuleBinarizable(*[], **{}), [], {})

