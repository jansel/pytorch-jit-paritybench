import sys
_module = sys.modules[__name__]
del sys
logistic_regression = _module
main = _module
modules = _module
byol = _module
transformations = _module
simclr = _module
process_features = _module

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


from collections import defaultdict


import numpy as np


import torch


import torch.nn as nn


from torchvision import datasets


from torchvision import models


from torchvision import transforms


from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


import copy


import random


from functools import wraps


from torch import nn


import torch.nn.functional as F


class RandomApply(nn.Module):

    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class MLP(nn.Module):

    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):

    def inner_fn(fn):

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class NetWrapper(nn.Module):

    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()
        if self.layer == -1:
            return self.net(x)
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection


class EMA:

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(nn.Module):

    def __init__(self, net, image_size, hidden_layer=-2, projection_size=256, projection_hidden_size=4096, augment_fn=None, moving_average_decay=0.99):
        super().__init__()
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two
        return loss.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'dim': 4, 'projection_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (RandomApply,
     lambda: ([], {'fn': _mock_layer(), 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Spijkervet_BYOL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

