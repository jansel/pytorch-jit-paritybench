import sys
_module = sys.modules[__name__]
del sys
protonets = _module
data = _module
base = _module
omniglot = _module
engine = _module
models = _module
factory = _module
few_shot = _module
utils = _module
log = _module
model = _module
eval = _module
run_eval = _module
run_train = _module
run_trainval = _module
train = _module
trainval = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from functools import partial


import numpy as np


from torchvision.transforms import ToTensor


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import torchvision


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


class Protonet(nn.Module):

    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs'])
        xq = Variable(sample['xq'])
        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if xq.is_cuda:
            target_inds = target_inds
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), xq.view(n_class * n_query, *xq.size()[2:])], 0)
        z = self.encoder.forward(x)
        z_dim = z.size(-1)
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]
        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return loss_val, {'loss': loss_val.item(), 'acc': acc_val.item()}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jakesnell_prototypical_networks(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

