import sys
_module = sys.modules[__name__]
del sys
conf = _module
discrete_hmm = _module
eeg_slds = _module
kalman_filter = _module
minipyro = _module
mixed_hmm = _module
experiment = _module
model = _module
seal_data = _module
pcfg = _module
sensor = _module
slds = _module
vae = _module
funsor = _module
adjoint = _module
affine = _module
cnf = _module
compat = _module
ops = _module
delta = _module
distribution = _module
domains = _module
einsum = _module
numpy_log = _module
numpy_map = _module
util = _module
gaussian = _module
integrate = _module
interpreter = _module
jax = _module
distributions = _module
joint = _module
memoize = _module
montecarlo = _module
optimizer = _module
pyro = _module
convert = _module
hmm = _module
registry = _module
sum_product = _module
tensor = _module
terms = _module
testing = _module
torch = _module
util = _module
update_headers = _module
setup = _module
test = _module
conftest = _module
test_bart = _module
test_sensor_fusion = _module
test_convert = _module
test_distribution = _module
test_hmm = _module
test_pyroapi = _module
test_adjoint = _module
test_affine = _module
test_alpha_conversion = _module
test_cnf = _module
test_delta = _module
test_einsum = _module
test_gaussian = _module
test_import = _module
test_integrate = _module
test_joint = _module
test_memoize = _module
test_minipyro = _module
test_optimizer = _module
test_samplers = _module
test_sum_product = _module
test_tensor = _module
test_terms = _module

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


from collections import OrderedDict


import numpy as np


import torch


import torch.nn as nn


import itertools


import math


from torch.optim import Adam


import torch.utils.data


from torch import nn


from torch import optim


from torch.nn import functional as F


import functools


import inspect


import re


def _log_det_tri(x):
    return ops.log(ops.diagonal(x, -1, -2)).sum(-1)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, image):
        image = image.reshape(image.shape[:-2] + (-1,))
        h1 = F.relu(self.fc1(image))
        loc = self.fc21(h1)
        scale = self.fc22(h1).exp()
        return loc, scale


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out.reshape(out.shape[:-1] + (28, 28))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pyro_ppl_funsor(_paritybench_base):
    pass

    def test_000(self):
        self._check(Decoder(*[], **{}), [torch.rand([20, 20])], {})
