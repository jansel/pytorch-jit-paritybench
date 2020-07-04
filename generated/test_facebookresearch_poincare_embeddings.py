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
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


import torch as th


from torch.autograd import Function


import numpy as np


from torch.nn import Embedding


from abc import abstractmethod


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
        return [{'params': self.lt.parameters(), 'rgrad': self.manifold.
            rgrad, 'expm': self.manifold.expm, 'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp}]

    def loss_function(self, inp, target, **kwargs):
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_poincare_embeddings(_paritybench_base):
    pass
