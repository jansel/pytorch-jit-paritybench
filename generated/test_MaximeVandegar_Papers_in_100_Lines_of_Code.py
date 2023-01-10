import sys
_module = sys.modules[__name__]
del sys
adversarial_feature_learning = _module
ali = _module
VAEs = _module
cgan = _module
inverse_rendering = _module
GANs = _module
GWOT = _module
siren = _module
semi_supervised_learning = _module
lsgan = _module
maxout_networks = _module
maml = _module
mfn = _module
NICE = _module
nerf = _module
reptile = _module
gradient_based_HO = _module
snl = _module
conv_gan = _module
Flows = _module

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


import torch


import torch.nn as nn


import numpy as np


from matplotlib import pyplot as plt


import torch.optim as optim


import scipy.io


from torch.nn import functional as F


import matplotlib.pyplot as plt


from scipy.stats import norm


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.distributions.multivariate_normal import MultivariateNormal


from torch.distributions.uniform import Uniform


import torchvision


from torch.distributions.transformed_distribution import TransformedDistribution


from torch.distributions.transforms import SigmoidTransform


from torch.distributions.transforms import AffineTransform


import copy


from typing import Callable


from typing import Tuple


import torchvision.transforms as transforms


import math


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.network = nn.Sequential(nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=False), nn.ReLU(), nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class E(nn.Module):

    def __init__(self):
        super(E, self).__init__()
        self.network = nn.Sequential(nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1024), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=False), nn.LeakyReLU(0.2), nn.Linear(1024, 50), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.network = nn.Sequential(nn.Linear(784 + 50, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1024), nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=False), nn.LeakyReLU(0.2), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x, z):
        return self.network(torch.cat((z, x), dim=1))


class GeneratorZ(nn.Module):

    def __init__(self):
        super(GeneratorZ, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False), nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False), nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False), nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True), nn.BatchNorm2d(512, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))

    def forward(self, x):
        z = self.network(x)
        mu, sigma = z[:, :256, :, :], z[:, 256:, :, :]
        return mu, sigma

    def sample(self, x):
        mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        return torch.randn(sigma.shape, device=x.device) * sigma + mu


class GeneratorX(nn.Module):

    def __init__(self):
        super(GeneratorX, self).__init__()
        self.network = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(256, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(256, 128, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(128, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(128, 64, 4, stride=1, padding=0, bias=False), nn.BatchNorm2d(64, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(32, momentum=0.05), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):

    def __init__(self, data_dim=2, context_dim=2, hidden_dim=200, constrain_mean=False):
        super(Model, self).__init__()
        """
        Model p(y|x) as N(mu, sigma) where mu and sigma are Neural Networks
        """
        self.h = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.log_var = nn.Sequential(nn.Linear(hidden_dim, data_dim))
        if constrain_mean:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim), nn.Sigmoid())
        else:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim))

    def get_mean_and_log_var(self, x):
        h = self.h(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def forward(self, epsilon, x):
        """
        Sample y ~ p(y|x) using the reparametrization trick
        """
        mu, log_var = self.get_mean_and_log_var(x)
        sigma = torch.sqrt(torch.exp(log_var))
        return epsilon * sigma + mu

    def compute_log_density(self, y, x):
        """
        Compute log p(y|x)
        """
        mu, log_var = self.get_mean_and_log_var(x)
        log_density = -0.5 * (torch.log(2 * torch.tensor(np.pi)) + log_var + (y - mu) ** 2 / (torch.exp(log_var) + 1e-10)).sum(dim=1)
        return log_density

    def compute_KL(self, x):
        """
        Assume that p(x) is a normal gaussian distribution; N(0, 1)
        """
        mu, log_var = self.get_mean_and_log_var(x)
        return -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.network = nn.Sequential(nn.ConvTranspose2d(100, 1024, kernel_size=(4, 4), stride=(1, 1), bias=False), nn.ReLU(), nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(3), nn.Tanh())

    def forward(self, noise):
        return self.network(noise)


class NerfModel(nn.Module):

    def __init__(self, embedding_dim_pos=20, embedding_dim_direction=8, hidden_dim=128):
        super(NerfModel, self).__init__()
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 3 + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim + 1))
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 3 + hidden_dim, hidden_dim // 2), nn.ReLU())
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = torch.empty(x.shape[0], x.shape[1] * 2 * L, device=x.device)
        for i in range(x.shape[1]):
            for j in range(L):
                out[:, i * (2 * L) + 2 * j] = torch.sin(2 ** j * x[:, i])
                out[:, i * (2 * L) + 2 * j + 1] = torch.cos(2 ** j * x[:, i])
        return out

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos // 2)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction // 2)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


class SineLayer(nn.Module):

    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):

    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1):
        super(Siren, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, hidden_dim), SineLayer(w0), nn.Linear(hidden_dim, out_dim))
        with torch.no_grad():
            self.net[0].weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            self.net[2].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[4].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[6].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)
            self.net[8].weight.uniform_(-np.sqrt(6.0 / hidden_dim) / w0, np.sqrt(6.0 / hidden_dim) / w0)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):

    def __init__(self, input_dim=784, output_dim=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ELU(), nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.layers(x)


class GaussianNoiseLayer(nn.Module):

    def __init__(self, sigma):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.train:
            noise = torch.randn(x.shape, device=x.device) * self.sigma
            return x + noise
        else:
            return x


class Maxout(nn.Module):

    def __init__(self, din, dout, k):
        super(Maxout, self).__init__()
        self.net = nn.Linear(din, k * dout)
        self.k = k
        self.dout = dout

    def forward(self, x):
        return torch.max(self.net(x).reshape(-1, self.k * self.dout).reshape(-1, self.dout, self.k), dim=-1).values


class GaborFilter(nn.Module):

    def __init__(self, in_dim, out_dim, alpha, beta=1.0):
        super(GaborFilter, self).__init__()
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,)))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.linear.weight.data *= 128.0 * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        norm = (x ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * x @ self.mu.T
        return torch.exp(-self.gamma.unsqueeze(0) / 2.0 * norm) * torch.sin(self.linear(x))


class GaborNet(nn.Module):

    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4):
        super(GaborNet, self).__init__()
        self.k = k
        self.gabon_filters = nn.ModuleList([GaborFilter(in_dim, hidden_dim, alpha=6.0 / k) for _ in range(k)])
        self.linear = nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])
        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):
        zi = self.gabon_filters[0](x)
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)
        return self.linear[self.k - 1](zi)


class NICE(nn.Module):

    def __init__(self, data_dim=28 * 28, hidden_dim=1000):
        super().__init__()
        self.m = torch.nn.ModuleList([nn.Sequential(nn.Linear(data_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, data_dim // 2)) for i in range(4)])
        self.s = torch.nn.Parameter(torch.randn(data_dim))

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.m)):
            x_i1 = x[:, ::2] if i % 2 == 0 else x[:, 1::2]
            x_i2 = x[:, 1::2] if i % 2 == 0 else x[:, ::2]
            h_i1 = x_i1
            h_i2 = x_i2 + self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = h_i1
            x[:, 1::2] = h_i2
        z = torch.exp(self.s) * x
        log_jacobian = torch.sum(self.s)
        return z, log_jacobian

    def invert(self, z):
        x = z.clone() / torch.exp(self.s)
        for i in range(len(self.m) - 1, -1, -1):
            h_i1 = x[:, ::2]
            h_i2 = x[:, 1::2]
            x_i1 = h_i1
            x_i2 = h_i2 - self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = x_i1 if i % 2 == 0 else x_i2
            x[:, 1::2] = x_i2 if i % 2 == 0 else x_i1
        return x


class WeightDecay(nn.Module):

    def __init__(self, model, device):
        super(WeightDecay, self).__init__()
        self.positive_constraint = torch.nn.Softplus()
        idx = 0
        self.parameter_dict = {}
        for m in model.parameters():
            self.parameter_dict[str(idx)] = torch.nn.Parameter(torch.rand(m.shape, device=device))
            idx += 1
        self.params = torch.nn.ParameterDict(self.parameter_dict)

    def forward(self, model):
        regularization = 0.0
        for coefficients, weights in zip(self.parameters(), model.parameters()):
            regularization += (self.positive_constraint(coefficients) * weights ** 2).sum()
        return regularization


class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.u = nn.Parameter(torch.rand(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: 1 - self.h(z) ** 2

    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return x, log_det


class NormalizingFlow(nn.Module):

    def __init__(self, flow_length, data_dim):
        super().__init__()
        self.layers = nn.Sequential(*(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (GaborFilter,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'alpha': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianNoiseLayer,
     lambda: ([], {'sigma': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 100, 4, 4])], {}),
     True),
    (GeneratorX,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 4])], {}),
     True),
    (GeneratorZ,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Maxout,
     lambda: ([], {'din': 4, 'dout': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormalizingFlow,
     lambda: ([], {'flow_length': 4, 'data_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PlanarFlow,
     lambda: ([], {'data_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SineLayer,
     lambda: ([], {'w0': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MaximeVandegar_Papers_in_100_Lines_of_Code(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

