import sys
_module = sys.modules[__name__]
del sys
pvae = _module
datasets = _module
datasets = _module
distributions = _module
ars = _module
hyperbolic_radius = _module
hyperspherical_uniform = _module
riemannian_normal = _module
wrapped_normal = _module
main = _module
manifolds = _module
euclidean = _module
poincareball = _module
models = _module
architectures = _module
mnist = _module
tabular = _module
vae = _module
objectives = _module
ops = _module
manifold_layers = _module
utils = _module
vis = _module
setup = _module
tests = _module
test_hyperbolic_radius = _module
test_hyperspherical_uniform = _module

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


import torch.utils.data


import numpy as np


import math


from torch.autograd import Function


from torch.autograd import grad


import torch.distributions as dist


from numbers import Number


from torch.distributions.utils import _standard_normal


from torch.distributions import constraints


from torch.nn import functional as F


from torch.distributions import Normal


from torch.distributions import Independent


from torch.distributions.utils import broadcast_all


from collections import defaultdict


from torch import optim


import torch.nn as nn


import torch.nn.functional as F


from numpy import prod


from torch.utils.data import DataLoader


from torchvision.utils import save_image


from torchvision import datasets


from torchvision import transforms


from sklearn.model_selection._split import _validate_shuffle_split


from torch import nn


from torch.nn.parameter import Parameter


from torch.nn import init


import time


from torch.autograd import Variable


import matplotlib.pyplot as plt


import matplotlib


from scipy.optimize import minimize_scalar


class Constants(object):
    eta = 1e-05
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88
    logfloorc = -104
    invsqrt2pi = 1.0 / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi / 2)


def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)


class EncLinear(nn.Module):
    """ Usual encoder """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncLinear, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)
        return mu, F.softplus(self.fc22(e)) + Constants.eta, self.manifold


class DecLinear(nn.Module):
    """ Usual decoder """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinear, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)
        return mu, torch.ones_like(mu)


class EncWrapped(nn.Module):
    """ Usual encoder followed by an exponential map """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta, self.manifold


class DecWrapped(nn.Module):
    """ Usual encoder preceded by a logarithm map """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)
        return mu, torch.ones_like(mu)


class RiemannianLayer(nn.Module):

    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight)

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias)

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad():
                self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):

    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight, signed=True, norm=self.weight_norm)
        return res


class DecGeo(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGeo, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)
        return mu, torch.ones_like(mu)


class MobiusLayer(RiemannianLayer):

    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncMob, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta, self.manifold


class LogZero(nn.Module):

    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)


class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """

    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecMob, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)
        return mu, torch.ones_like(mu)


class DecBernouilliWrapper(nn.Module):
    """ Wrapper for Bernoulli likelihood """

    def __init__(self, dec):
        super(DecBernouilliWrapper, self).__init__()
        self.dec = dec

    def forward(self, z):
        mu, _ = self.dec.forward(z)
        return torch.tensor(1.0), mu


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    else:
        return params[0]


class VAE(nn.Module):

    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std
        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: -F.binary_cross_entropy_with_logits(self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value), value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value, reduction='none')

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))
        return mean, means.view(-1, *means.size()[2:]), samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))
        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset):
        pass


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, **kwargs):
        super(Linear, self).__init__(in_features, out_features)


class ExpZero(nn.Module):

    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_emilemathieu_pvae(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

