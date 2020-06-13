import sys
_module = sys.modules[__name__]
del sys
data = _module
flow = _module
train_variational_autoencoder_pytorch = _module
train_variational_autoencoder_tensorflow = _module

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


import torch.nn as nn


from torch.nn import functional as F


import torch.utils


import torch.utils.data


from torch import nn


import logging


import random


class InverseAutoregressiveFlow(nn.Module):
    """Inverse Autoregressive Flows with LSTM-type update. One block.
  
  Eq 11-14 of https://arxiv.org/abs/1606.04934
  """

    def __init__(self, num_input, num_hidden, num_context):
        super().__init__()
        self.made = MADE(num_input=num_input, num_output=num_input * 2,
            num_hidden=num_hidden, num_context=num_context)
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


class MaskedLinear(nn.Module):
    """Linear layer with some input-output connections masked."""

    def __init__(self, in_features, out_features, mask, context_features=
        None, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.register_buffer('mask', mask)
        if context_features is not None:
            self.cond_linear = nn.Linear(context_features, out_features,
                bias=False)

    def forward(self, input, context=None):
        output = F.linear(input, self.mask * self.linear.weight, self.
            linear.bias)
        if context is None:
            return output
        else:
            return output + self.cond_linear(context)


class MADE(nn.Module):
    """Implements MADE: Masked Autoencoder for Distribution Estimation.

  Follows https://arxiv.org/abs/1502.03509

  This is used to build MAF: Masked Autoregressive Flow (https://arxiv.org/abs/1705.07057).
  """

    def __init__(self, num_input, num_output, num_hidden, num_context):
        super().__init__()
        self._m = []
        self._masks = []
        self._build_masks(num_input, num_output, num_hidden, num_layers=3)
        self._check_masks()
        modules = []
        self.input_context_net = MaskedLinear(num_input, num_hidden, self.
            _masks[0], num_context)
        modules.append(nn.ReLU())
        modules.append(MaskedLinear(num_hidden, num_hidden, self._masks[1],
            context_features=None))
        modules.append(nn.ReLU())
        modules.append(MaskedLinear(num_hidden, num_output, self._masks[2],
            context_features=None))
        self.net = nn.Sequential(*modules)

    def _build_masks(self, num_input, num_output, num_hidden, num_layers):
        """Build the masks according to Eq 12 and 13 in the MADE paper."""
        rng = np.random.RandomState(0)
        self._m.append(np.arange(1, num_input + 1))
        for i in range(1, num_layers + 1):
            if i == num_layers:
                m = np.arange(1, num_input + 1)
                assert num_output % num_input == 0, 'num_output must be multiple of num_input'
                self._m.append(np.hstack([m for _ in range(num_output //
                    num_input)]))
            else:
                self._m.append(rng.randint(1, num_input, size=num_hidden))
            if i == num_layers:
                mask = self._m[i][(None), :] > self._m[i - 1][:, (None)]
            else:
                mask = self._m[i][(None), :] >= self._m[i - 1][:, (None)]
            self._masks.append(torch.from_numpy(mask.astype(np.float32).T))

    def _check_masks(self):
        """Check that the connectivity matrix between layers is lower triangular."""
        prev = self._masks[0].t()
        for i in range(1, len(self._masks)):
            prev = prev @ self._masks[i].t()
        final = prev.numpy()
        num_input = self._masks[0].shape[1]
        num_output = self._masks[-1].shape[0]
        assert final.shape == (num_input, num_output)
        if num_output == num_input:
            assert np.triu(final).all() == 0
        else:
            for submat in np.split(final, indices_or_sections=num_output //
                num_input, axis=1):
                assert np.triu(submat).all() == 0

    def forward(self, input, context=None):
        hidden = self.input_context_net(input, context)
        return self.net(hidden)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
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
            return inputs[:, :, (self.perm)], torch.zeros_like(inputs,
                device=inputs.device)
        elif mode == 'inverse':
            return inputs[:, :, (self.inv_perm)], torch.zeros_like(inputs,
                device=inputs.device)
        else:
            raise ValueError('Mode must be one of {forward, inverse}.')


class Model(nn.Module):
    """Bernoulli model parameterized by a generative network with Gaussian latents for MNIST."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer('p_z_loc', torch.zeros(latent_size))
        self.register_buffer('p_z_scale', torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(input_size=latent_size,
            output_size=data_size, hidden_size=latent_size * 2)

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1,
            keepdim=True)
        logits = self.generative_network(z)
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x


class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.inference_network = NeuralNetwork(input_size=data_size,
            output_size=latent_size * 2, hidden_size=latent_size * 2)
        self.log_q_z = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg = torch.chunk(self.inference_network(x).unsqueeze(1),
            chunks=2, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=
            loc.device)
        z = loc + scale * eps
        log_q_z = self.log_q_z(loc, scale, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, latent_size, data_size, flow_depth):
        super().__init__()
        hidden_size = latent_size * 2
        self.inference_network = NeuralNetwork(input_size=data_size,
            output_size=latent_size * 3, hidden_size=hidden_size)
        modules = []
        for _ in range(flow_depth):
            modules.append(flow.InverseAutoregressiveFlow(num_input=
                latent_size, num_hidden=hidden_size, num_context=latent_size))
            modules.append(flow.Reverse(latent_size))
        self.q_z_flow = flow.FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(self.inference_network(x).unsqueeze
            (1), chunks=3, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=
            loc.device)
        z_0 = loc + scale * eps
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        modules = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear
            (hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size,
            output_size)]
        self.net = nn.Sequential(*modules)

    def forward(self, input):
        return self.net(input)


class NormalLogProb(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (
            2 * var)


class BernoulliLogProb(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target):
        return -self.bce_with_logits(logits, target)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_altosaar_variational_autoencoder(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(InverseAutoregressiveFlow(*[], **{'num_input': 4, 'num_hidden': 4, 'num_context': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(FlowSequential(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MADE(*[], **{'num_input': 4, 'num_output': 4, 'num_hidden': 4, 'num_context': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Reverse(*[], **{'num_input': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Model(*[], **{'latent_size': 4, 'data_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(VariationalMeanField(*[], **{'latent_size': 4, 'data_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(VariationalFlow(*[], **{'latent_size': 4, 'data_size': 4, 'flow_depth': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(NeuralNetwork(*[], **{'input_size': 4, 'output_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(NormalLogProb(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(BernoulliLogProb(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

