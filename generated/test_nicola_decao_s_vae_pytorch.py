import sys
_module = sys.modules[__name__]
del sys
mnist = _module
hyperspherical_vae = _module
distributions = _module
hyperspherical_uniform = _module
von_mises_fisher = _module
ops = _module
ive = _module
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data


from torchvision import datasets


from torchvision import transforms


from collections import defaultdict


import math


from torch.distributions.kl import register_kl


import scipy.special


from numbers import Number


class HypersphericalUniform(torch.distributions.Distribution):
    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device='cpu'):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        output = torch.distributions.Normal(0, 1).sample((shape if isinstance(shape, torch.Size) else torch.Size([shape])) + torch.Size([self._dim + 1]))
        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        if torch.__version__ >= '1.0.0':
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]))
        else:
            lgamma = torch.lgamma(torch.Tensor([(self._dim + 1) / 2], device=self.device))
        return math.log(2) + (self._dim + 1) / 2 * math.log(math.pi) - lgamma


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        return torch.Tensor(output)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z)


ive = IveFunction.apply


class VonMisesFisher(torch.distributions.Distribution):
    arg_constraints = {'loc': torch.distributions.constraints.real, 'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))
        self.k = k
        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)
        v = torch.distributions.Normal(0, 1).sample(shape + torch.Size(self.loc.shape)).transpose(0, -1)[1:].transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)
        w_ = torch.sqrt(torch.clamp(1 - w ** 2, 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)
        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape)
        self.__w = 1 + torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt(4 * self.scale ** 2 + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(torch.max(torch.tensor([0.0], dtype=self.dtype, device=self.device), self.scale - 10), torch.tensor([1.0], dtype=self.dtype, device=self.device))
        b = b_app * s + b_true * (1 - s)
        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = 4 * a * b / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(mask.any(dim=dim), mask.float().argmax(dim=1).squeeze(), torch.tensor(invalid_val, device=x.device))
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b), torch.zeros_like(b), torch.ones_like(b) == 1
        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)
        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = torch.distributions.Beta(con1, con2).sample(sample_shape).type(self.dtype)
            u = torch.distributions.Uniform(0 + eps, 1 - eps).sample(sample_shape).type(self.dtype)
            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = 2 * a * b / (1 - (1 - b) * e_)
            accept = (self.__m - 1.0) * t.log() - t + d > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))
            reject = accept_idx < 0
            accept = ~reject if torch.__version__ >= '1.2.0' else 1 - reject
            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-05)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        output = -self.scale * ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        return output.view(*output.shape[:-1]) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*output.shape[:-1])

    def _log_normalization(self):
        output = -((self.__m / 2 - 1) * torch.log(self.scale) - self.__m / 2 * math.log(2 * math.pi) - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))
        return output.view(*output.shape[:-1])


class ModelVAE(torch.nn.Module):

    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)
        if self.distribution == 'normal':
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        if self.distribution == 'normal':
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented
        return z_mean, z_var

    def decode(self, z):
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        return x

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, x_


class Ive(torch.nn.Module):

    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Ive,
     lambda: ([], {'v': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_nicola_decao_s_vae_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

