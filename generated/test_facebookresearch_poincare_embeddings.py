import sys
_module = sys.modules[__name__]
del sys
embed = _module
hype = _module
checkpoint = _module
common = _module
energy_function = _module
graph = _module
hypernymy_eval = _module
manifolds = _module
euclidean = _module
lorentz = _module
manifold = _module
poincare = _module
rsgd = _module
sn = _module
train = _module
reconstruction = _module
setup = _module
transitive_closure = _module

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


import torch as th


import numpy as np


import logging


import torch.multiprocessing as mp


import time


import torch


import warnings


from torch.autograd import Function


import torch.nn.functional as F


from collections import defaultdict as ddict


from numpy.random import choice


from torch.utils.data import Dataset as DS


from sklearn.metrics import average_precision_score


from functools import partial


from torch.nn import Embedding


from abc import abstractmethod


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from numpy.random import randint


from torch.utils import data as torch_data


class EnergyFunction(torch.nn.Module):

    def __init__(self, manifold, dim, size, sparse=False, **kwargs):
        super().__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.nobjects = size
        self.manifold.init_weights(self.lt)

    def forward(self, inputs):
        e = self.lt(inputs)
        with torch.no_grad():
            e = self.manifold.normalize(e)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        return self.energy(s, o).squeeze(-1)

    def optim_params(self):
        return [{'params': self.lt.parameters(), 'rgrad': self.manifold.rgrad, 'expm': self.manifold.expm, 'logm': self.manifold.logm, 'ptransp': self.manifold.ptransp}]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


class DistanceEnergyFunction(EnergyFunction):

    def energy(self, s, o):
        return self.manifold.distance(s, o)

    def loss(self, inp, target, **kwargs):
        return F.cross_entropy(inp.neg(), target)


class EntailmentConeEnergyFunction(EnergyFunction):

    def __init__(self, *args, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.manifold.K is not None, 'K cannot be none for EntailmentConeEnergyFunction'
        assert hasattr(self.manifold, 'angle_at_u'), 'Missing `angle_at_u` method'
        self.margin = margin

    def energy(self, s, o):
        energy = self.manifold.angle_at_u(o, s) - self.manifold.half_aperture(o)
        return energy.clamp(min=0)

    def loss(self, inp, target, **kwargs):
        loss = inp[:, (0)].clamp_(min=0).sum()
        loss += (self.margin - inp[:, 1:]).clamp_(min=0).sum()
        return loss / inp.numel()

