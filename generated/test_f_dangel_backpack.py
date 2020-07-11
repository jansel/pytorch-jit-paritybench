import sys
_module = sys.modules[__name__]
del sys
backpack = _module
context = _module
core = _module
derivatives = _module
avgpool2d = _module
basederivatives = _module
batchnorm1d = _module
conv2d = _module
crossentropyloss = _module
dropout = _module
elementwise = _module
flatten = _module
linear = _module
maxpool2d = _module
mseloss = _module
relu = _module
shape_check = _module
sigmoid = _module
tanh = _module
zeropad2d = _module
extensions = _module
backprop_extension = _module
curvature = _module
curvmatprod = _module
activations = _module
cmpbase = _module
losses = _module
padding = _module
pooling = _module
firstorder = _module
base = _module
batch_grad = _module
batch_grad_base = _module
batch_l2_grad = _module
conv2d = _module
linear = _module
gradient = _module
sum_grad_squared = _module
conv2d = _module
linear = _module
variance = _module
variance_base = _module
mat_to_mat_jac_base = _module
module_extension = _module
secondorder = _module
diag_ggn = _module
diag_ggn_base = _module
diag_hessian = _module
conv2d = _module
diag_h_base = _module
linear = _module
hbp = _module
conv2d = _module
hbp_options = _module
hbpbase = _module
linear = _module
hessianfree = _module
ggnvp = _module
hvp = _module
lop = _module
rop = _module
utils = _module
conv = _module
convert_parameters = _module
ein = _module
examples = _module
kroneckers = _module
linear = _module
unsqueeze = _module
example_all_in_one = _module
example_diag_ggn_optimizer = _module
example_differential_privacy = _module
example_first_order_resnet = _module
conf = _module
setup = _module
test = _module
automated_bn_test = _module
automated_kfac_test = _module
automated_test = _module
benchmark = _module
functionality = _module
jvp = _module
jvp_activations = _module
jvp_avgpool2d = _module
jvp_conv2d = _module
jvp_linear = _module
jvp_maxpool2d = _module
jvp_zeropad2d = _module
bugfixes_test = _module
conv2d_test = _module
implementation = _module
implementation_autograd = _module
implementation_bpext = _module
interface_test = _module
layers = _module
layers_test = _module
linear_test = _module
networks = _module
problems = _module
test_ea_jac_t_mat_jac_prod = _module
test_problem = _module
test_problems_activations = _module
test_problems_bn = _module
test_problems_convolutions = _module
test_problems_kfacs = _module
test_problems_linear = _module
test_problems_padding = _module
test_problems_pooling = _module
test_second_order_warnings = _module
test_simple_resnet = _module
test_sqrt_hessian = _module
test_sqrt_hessian_sampled = _module
test_sum_hessian = _module
utils_test = _module

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


import inspect


import torch


from torch.nn import AvgPool2d


from torch.nn import Conv2d


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import Dropout


from torch.nn import Linear


from torch.nn import MaxPool2d


from torch.nn import ReLU


from torch.nn import Sigmoid


from torch.nn import Tanh


from torch.nn import ZeroPad2d


import torch.nn


from torch.nn import ConvTranspose2d


import warnings


from warnings import warn


from torch import einsum


from torch.nn import BatchNorm1d


from torch.nn.functional import conv2d


from math import sqrt


from torch import diag


from torch import diag_embed


from torch import multinomial


from torch import ones_like


from torch import softmax


from torch import sqrt as torchsqrt


from torch.nn.functional import one_hot


from torch import eq


from torch.nn import Flatten


from torch import zeros


from torch.nn.functional import max_pool2d


from torch import eye


from torch import normal


from torch import gt


from torch.nn import functional


from torch.nn import Sequential


from numpy import prod


from torch import clamp


from torch.nn import Unfold


import torchvision


import matplotlib.pyplot as plt


from torch.optim import Optimizer


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from functools import partial


from torch import allclose


from torch import randn


import itertools


from random import choice


from random import randint


from torch import Tensor


from torch import nn


import random


import scipy.linalg


class MyFirstResNet(torch.nn.Module):

    def __init__(self, C_in=1, C_hid=5, input_dim=(28, 28), output_dim=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(C_in, C_hid, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(C_hid, C_hid, kernel_size=3, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(input_dim[0] * input_dim[1] * C_hid, output_dim)
        if C_in == C_hid:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv2(F.relu(self.conv1(x)))
        x += residual
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


class Identity(torch.nn.Module):
    """Identity operation."""

    def forward(self, input):
        return input


class Parallel(torch.nn.Sequential):
    """Feed input to multiple modules, sum the result.

              |-----|
        | ->  | f_1 |  -> |
        |     |-----|     |
        |                 |
        |     |-----|     |
    x ->| ->  | f_2 |  -> + -> f₁(x) + f₂(x) + ...
        |     |-----|     |
        |                 |
        |     |-----|     |
        | ->  | ... |  -> |
              |-----|

    """

    def forward(self, input):
        """Process input with all modules, sum the output."""
        for idx, module in enumerate(self.children()):
            if idx == 0:
                output = module(input)
            else:
                output = output + module(input)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_f_dangel_backpack(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

