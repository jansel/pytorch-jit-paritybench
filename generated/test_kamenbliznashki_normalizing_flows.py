import sys
_module = sys.modules[__name__]
del sys
bnaf = _module
data = _module
datasets = _module
bsds300 = _module
celeba = _module
download_celeba = _module
gas = _module
hepmass = _module
miniboone = _module
mnist = _module
moons = _module
power = _module
toy = _module
util = _module
glow = _module
maf = _module
planar_flow = _module
test = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.distributions as D


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import math


import time


from functools import partial


import numpy as np


import torchvision.transforms as T


from torch.utils.data import Dataset


from sklearn.datasets import make_moons


from torchvision.utils import save_image


from torchvision.utils import make_grid


from torch.utils.checkpoint import checkpoint


from torchvision.datasets import MNIST


import copy


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)
        self.register_buffer('mask', mask)
        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class Tanh(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, sum_logdets):
        logdet = -2 * (x - math.log(2) + F.softplus(-2 * x))
        sum_logdets = sum_logdets + logdet.view_as(sum_logdets)
        return x.tanh(), sum_logdets


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class BNAF(nn.Module):

    def __init__(self, data_dim, n_hidden, hidden_dim):
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(data_dim))
        self.register_buffer('base_dist_var', torch.ones(data_dim))
        modules = []
        modules += [MaskedLinear(data_dim, hidden_dim, data_dim), Tanh()]
        for _ in range(n_hidden):
            modules += [MaskedLinear(hidden_dim, hidden_dim, data_dim), Tanh()]
        modules += [MaskedLinear(hidden_dim, data_dim, data_dim)]
        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        return self.net(x)


class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """

    def __init__(self, param_dim=(1, 3, 1, 1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            self.bias.squeeze().data.copy_(x.transpose(0, 1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0, 1).flatten(1).std(1, False) + 1e-06).view_as(self.bias)
            self.initialized += 1
        z = (x - self.bias) / self.scale
        logdet = -self.scale.abs().log().sum() * x.shape[2] * x.shape[3]
        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """

    def __init__(self, n_channels=3, lu_factorize=False):
        super().__init__()
        self.lu_factorize = lu_factorize
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]
        if lu_factorize:
            p, l, u = torch.btriunpack(*w.unsqueeze(0).btrifact())
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            logdet = self.log_s.sum() * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W
        return F.conv2d(x, self.w.view(C, C, 1, 1)), logdet

    def inverse(self, z):
        B, C, H, W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = -self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = -torch.slogdet(self.w)[-1] * H * W
        return F.conv2d(z, w_inv.view(C, C, 1, 1)), logdet


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """

    def __init__(self, n_channels, width):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels // 2, width, kernel_size=3, padding=1, bias=False)
        self.actnorm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels, 1, 1))
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)
        h = F.relu(self.actnorm1(self.conv1(x_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:, 0::2, :, :]
        s = h[:, 1::2, :, :]
        s = torch.sigmoid(s + 2.0)
        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)
        logdet = s.log().sum([1, 2, 3])
        return z, logdet

    def inverse(self, z):
        z_a, z_b = z.chunk(2, 1)
        h = F.relu(self.actnorm1(self.conv1(z_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:, 0::2, :, :]
        s = h[:, 1::2, :, :]
        s = torch.sigmoid(s + 2.0)
        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)
        logdet = -s.log().sum([1, 2, 3])
        return x, logdet


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of 
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, 4 * C, H // 2, W // 2)
        return x

    def inverse(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, 2 * H, 2 * W)
        return x


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, padding=1)
        self.log_scale_factor = nn.Parameter(torch.zeros(2 * n_channels, 1, 1))
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        z2 = (x2 - m) * torch.exp(-logs)
        logdet = -logs.sum([1, 2, 3])
        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1, 2, 3])
        return x2, logdet


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """

    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        z2, logdet = self.gaussianize(x1, x2)
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)
        return x, logdet


class Preprocess(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        logdet = -math.log(256) * x[0].numel()
        return x - 0.5, logdet

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """

    def __init__(self, n_channels, width, lu_factorize=False):
        super().__init__(Actnorm(param_dim=(1, n_channels, 1, 1)), Invertible1x1Conv(n_channels, lu_factorize), AffineCoupling(n_channels, width))


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """

    def __init__(self, n_channels, width, depth, checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4 * n_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.split = Split(4 * n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        x1, z2, logdet_split = self.split(x)
        logdet = logdet_flowsteps + logdet_split
        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)
        logdet = logdet_flowsteps + logdet_split
        return x, logdet


class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""

    def __init__(self, width, depth, n_levels, input_dims=(3, 32, 32), checkpoint_grads=False, lu_factorize=False):
        super().__init__()
        in_channels, H, W = input_dims
        out_channels = int(in_channels * 4 ** (n_levels + 1) / 2 ** n_levels)
        out_HW = int(H / 2 ** (n_levels + 1))
        self.output_dims = out_channels, out_HW, out_HW
        self.preprocess = Preprocess()
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * 2 ** i, width, depth, checkpoint_grads, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, width, lu_factorize) for _ in range(depth)], checkpoint_grads=checkpoint_grads)
        self.gaussianize = Gaussianize(out_channels)
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x):
        x, sum_logdets = self.preprocess(x)
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x)
            sum_logdets = sum_logdets + logdet
            zs.append(z)
        x = self.squeeze(x)
        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)
        return zs, sum_logdets

    def inverse(self, zs=None, batch_size=None, z_std=1.0):
        if zs is None:
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
            zs = [z_std * self.base_dist.sample((batch_size, *self.output_dims)).squeeze()]
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])
        x, logdet = self.flowstep.inverse(z)
        sum_logdets = sum_logdets + logdet
        x = self.squeeze.inverse(x)
        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (self.base_dist.sample(x.shape).squeeze() if len(zs) == 1 else zs[-i - 2])
            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, bits_per_pixel=False):
        zs, logdet = self.forward(x)
        log_prob = sum(self.base_dist.log_prob(z).sum([1, 2, 3]) for z in zs) + logdet
        if bits_per_pixel:
            log_prob /= math.log(2) * x[0].numel()
        return log_prob


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()
        self.register_buffer('mask', mask)
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)
        self.t_net = copy.deepcopy(self.s_net)
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        mx = x * self.mask
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)
        log_abs_det_jacobian = -(1 - self.mask) * s
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        mu = u * self.mask
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)
        log_abs_det_jacobian = (1 - self.mask) * s
        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-05):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det_jacobian.expand_as(x)


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    degrees = []
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]
    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]
    return masks, degrees[0]


class MADE(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        D = u.shape[1]
        x = torch.zeros_like(u)
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, (i)] = u[:, (i)] * torch.exp(loga[:, (i)]) + m[:, (i)]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MADEMOG(nn.Module):
    """ Mixture of Gaussians MADE """

    def __init__(self, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, n_components * 3 * input_size, masks[-1].repeat(n_components * 3, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        N, L = x.shape
        C = self.n_components
        m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)
        x = x.repeat(1, C).view(N, C, L)
        u = (x - m) * torch.exp(-loga)
        log_abs_det_jacobian = -loga
        self.logr = logr - logr.logsumexp(1, keepdim=True)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        N, C, L = u.shape
        x = torch.zeros(N, L)
        for i in self.input_degrees:
            m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)
            logr = logr - logr.logsumexp(1, keepdim=True)
            z = D.Categorical(logits=logr[:, :, (i)]).sample().unsqueeze(-1)
            u_z = torch.gather(u[:, :, (i)], 1, z).squeeze()
            m_z = torch.gather(m[:, :, (i)], 1, z).squeeze()
            loga_z = torch.gather(loga[:, :, (i)], 1, z).squeeze()
            x[:, (i)] = u_z * torch.exp(loga_z) + m_z
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        log_probs = torch.logsumexp(self.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)
        return log_probs.sum(1)


class MAF(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class MAFMOG(nn.Module):
    """ MAF on mixture of gaussian MADE """

    def __init__(self, n_blocks, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', batch_norm=True):
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.maf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, batch_norm)
        input_degrees = self.maf.input_degrees
        self.mademog = MADEMOG(n_components, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, input_degrees)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return u, sum_log_abs_det_jacobians

    def inverse(self, u, y=None):
        x, made_log_abs_dets = self.mademog.inverse(u, y)
        x, maf_log_abs_dets = self.maf.inverse(x, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        log_probs = torch.logsumexp(self.mademog.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)
        return log_probs.sum(1)


class RealNVP(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class PlanarTransform(nn.Module):

    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = -1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        psi = (1 - torch.tanh(z @ self.w.t() + self.b) ** 2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-06).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return f_z, sum_log_abs_det_jacobians


class AffineTransform(nn.Module):

    def __init__(self, learnable=False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(2)).requires_grad_(learnable)

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x
        sum_log_abs_det_jacobians = self.logsigma.sum()
        return z, sum_log_abs_det_jacobians


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineCoupling,
     lambda: ([], {'n_channels': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AffineTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 2])], {}),
     True),
    (BatchNorm,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FlowSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Gaussianize,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MADE,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MADEMOG,
     lambda: ([], {'n_components': 4, 'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MAF,
     lambda: ([], {'n_blocks': 4, 'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MAFMOG,
     lambda: ([], {'n_blocks': 4, 'n_components': 4, 'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PlanarTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([2, 2])], {}),
     False),
    (Preprocess,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RealNVP,
     lambda: ([], {'n_blocks': 4, 'input_size': 4, 'hidden_size': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Split,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Tanh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kamenbliznashki_normalizing_flows(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

