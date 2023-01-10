import sys
_module = sys.modules[__name__]
del sys
data = _module
flow = _module
masks = _module
train_variational_autoencoder_jax = _module
train_variational_autoencoder_pytorch = _module
train_variational_autoencoder_tensorflow = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.nn import functional as F


import random


import time


import torch.utils


import torch.utils.data


from torch import nn


class MaskedLinear(nn.Module):
    """Linear layer with some input-output connections masked."""

    def __init__(self, in_features, out_features, mask, context_features=None, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        if context_features is not None:
            self.cond_linear = nn.Linear(context_features, out_features, bias=False)

    def forward(self, input, context=None):
        output = F.linear(input, self.mask * self.linear.weight, self.linear.bias)
        if context is None:
            return output
        else:
            return output + self.cond_linear(context)


class MADE(nn.Module):
    """Implements MADE: Masked Autoencoder for Distribution Estimation.

    Follows https://arxiv.org/abs/1502.03509

    This is used to build MAF: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057).
    """

    def __init__(self, num_input, num_outputs_per_input, num_hidden, num_context):
        super().__init__()
        self._m = []
        degrees = masks.create_degrees(input_size=num_input, hidden_units=[num_hidden] * 2, input_order='left-to-right', hidden_degrees='equal')
        self._masks = masks.create_masks(degrees)
        self._masks[-1] = np.hstack([self._masks[-1] for _ in range(num_outputs_per_input)])
        self._masks = [torch.from_numpy(m.T) for m in self._masks]
        modules = []
        self.input_context_net = MaskedLinear(num_input, num_hidden, self._masks[0], num_context)
        self.net = nn.Sequential(nn.ReLU(), MaskedLinear(num_hidden, num_hidden, self._masks[1], context_features=None), nn.ReLU(), MaskedLinear(num_hidden, num_outputs_per_input * num_input, self._masks[2], context_features=None))

    def forward(self, input, context=None):
        hidden = self.input_context_net(input, context)
        return self.net(hidden)


class InverseAutoregressiveFlow(nn.Module):
    """Inverse Autoregressive Flows with LSTM-type update. One block.

    Eq 11-14 of https://arxiv.org/abs/1606.04934
    """

    def __init__(self, num_input, num_hidden, num_context):
        super().__init__()
        self.made = MADE(num_input=num_input, num_outputs_per_input=2, num_hidden=num_hidden, num_context=num_context)
        self.sigmoid_arg_bias = nn.Parameter(torch.ones(num_input) * 2)
        self.sigmoid = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, input, context=None):
        m, s = torch.chunk(self.made(input, context), chunks=2, dim=-1)
        s = s + self.sigmoid_arg_bias
        sigmoid = self.sigmoid(s)
        z = sigmoid * input + (1 - sigmoid) * m
        return z, -self.log_sigmoid(s)


class FlowSequential(nn.Sequential):
    """Forward pass."""

    def forward(self, input, context=None):
        total_log_prob = torch.zeros_like(input, device=input.device)
        for block in self._modules.values():
            input, log_prob = block(input, context)
            total_log_prob += log_prob
        return input, total_log_prob


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).

    From https://github.com/ikostrikov/pytorch-flows/blob/master/main.py
    """

    def __init__(self, num_input):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_input)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, context=None, mode='forward'):
        if mode == 'forward':
            return inputs[:, :, self.perm], torch.zeros_like(inputs, device=inputs.device)
        elif mode == 'inverse':
            return inputs[:, :, self.inv_perm], torch.zeros_like(inputs, device=inputs.device)
        else:
            raise ValueError('Mode must be one of {forward, inverse}.')


class BernoulliLogProb(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        return -self.bce_with_logits(logits, target)


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        return self.net(input)


class NormalLogProb(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)


class Model(nn.Module):
    """Variational autoencoder, parameterized by a generative network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer('p_z_loc', torch.zeros(latent_size))
        self.register_buffer('p_z_scale', torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2)

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        logits = self.generative_network(z)
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x


class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=data_size, output_size=latent_size * 2, hidden_size=latent_size * 2)
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z = loc + scale * eps
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = NeuralNetwork(input_size=data_size, output_size=latent_size * 3, hidden_size=hidden_size)
        modules = []
        for _ in range(flow_depth):
            modules.append(flow.InverseAutoregressiveFlow(num_input=latent_size, num_hidden=hidden_size, num_context=latent_size))
            modules.append(flow.Reverse(latent_size))
        self.q_z_flow = flow.FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(self.inference_network(x).unsqueeze(1), chunks=3, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BernoulliLogProb,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FlowSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Model,
     lambda: ([], {'latent_size': 4, 'data_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeuralNetwork,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormalLogProb,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reverse,
     lambda: ([], {'num_input': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VariationalMeanField,
     lambda: ([], {'latent_size': 4, 'data_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_altosaar_variational_autoencoder(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

