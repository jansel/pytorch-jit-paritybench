import sys
_module = sys.modules[__name__]
del sys
mnist = _module
jacobian = _module
jacobian = _module
setup = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import torch.autograd as autograd


class MLP(nn.Module):
    """
    Simple MLP to demonstrate Jacobian regularization.
    """

    def __init__(self, in_channel=1, im_size=28, num_classes=10, fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
        compression = in_channel * im_size * im_size
        self.compression = compression
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, num_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class JacobianReg(nn.Module):
    """
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    """

    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        """
        B, C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C * torch.norm(Jv) ** 2 / (num_proj * B)
        R = 1 / 2 * J2
        return R

    def _random_vector(self, C, B):
        """
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        """
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        """
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        """
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v, retain_graph=True, create_graph=create_graph)
        return grad_x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {}),
     True),
]

class Test_facebookresearch_jacobian_regularizer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

