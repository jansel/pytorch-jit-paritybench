import sys
_module = sys.modules[__name__]
del sys
args = _module
datasets = _module
demo = _module
metrics = _module
evaluation_metrics = _module
pytorch_structural_losses = _module
match_cost = _module
nn_distance = _module
setup = _module
models = _module
cnf = _module
diffeq_layers = _module
flow = _module
networks = _module
normalization = _module
odefunc = _module
test = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch import optim


from torch import nn


from torch.nn import Parameter


import copy


from collections import defaultdict


import torch.distributed as dist


import warnings


import torch.distributed


import random


import torch.multiprocessing as mp


import time


import scipy.misc


from torch.backends import cudnn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx, integration_times, reverse)
            return x, logpx


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class CNF(nn.Module):

    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-05, rtol=1e-05, use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter('sqrt_end_time', nn.Parameter(torch.sqrt(torch.tensor(T))))
        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError('Regularization not supported')
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.conditional = conditional

    def forward(self, x, context=None, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1)
        else:
            _logpx = logpx
        if self.conditional:
            assert context is not None
            states = x, _logpx, context
            atol = [self.atol] * 3
            rtol = [self.rtol] * 3
        else:
            states = x, _logpx
            atol = [self.atol] * 2
            rtol = [self.rtol] * 2
        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack([torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time])
            else:
                integration_times = torch.tensor([0.0, self.T], requires_grad=False)
        if reverse:
            integration_times = _flip(integration_times, 0)
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(self.odefunc, states, integration_times, atol=atol, rtol=rtol, method=self.solver, options=self.solver_options)
        else:
            state_t = odeint(self.odefunc, states, integration_times, atol=self.test_atol, rtol=self.test_rtol, method=self.test_solver)
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
        z_t, logpz_t = state_t[:2]
        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


class IgnoreLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, context, x):
        return self._layer(x)


class ConcatLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1 + dim_c, dim_out)

    def forward(self, context, x, c):
        if x.dim() == 3:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x_context = torch.cat((x, context), dim=2)
        return self._layer(x_context)


class ConcatLinear_v2(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)

    def forward(self, context, x):
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            bias = bias.unsqueeze(1)
        return self._layer(x) + bias


class SquashLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper(context))
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ScaleLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(ScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ConcatSquashLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class ConcatScaleLinear(nn.Module):

    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper_gate(context)
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class Encoder(nn.Module):

    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)
        return m, v


def build_model(args, input_dim, hidden_dims, context_dim, num_blocks, conditional):

    def build_cnf():
        diffeq = ODEnet(hidden_dims=hidden_dims, input_shape=(input_dim,), context_dim=context_dim, layer_type=args.layer_type, nonlinearity=args.nonlinearity)
        odefunc = ODEfunc(diffeq=diffeq)
        cnf = CNF(odefunc=odefunc, T=args.time_length, train_T=args.train_T, conditional=conditional, solver=args.solver, use_adjoint=args.use_adjoint, atol=args.atol, rtol=args.rtol)
        return cnf
    chain = [build_cnf() for _ in range(num_blocks)]
    if args.batch_norm:
        bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn) for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag, sync=args.sync_bn)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_latent_cnf(args):
    dims = tuple(map(int, args.latent_dims.split('-')))
    model = build_model(args, args.zdim, dims, 0, args.latent_num_blocks, False).cuda()
    print('Number of trainable parameters of Latent CNF: {}'.format(count_parameters(model)))
    return model


def get_point_cnf(args):
    dims = tuple(map(int, args.dims.split('-')))
    model = build_model(args, args.input_dim, dims, args.zdim, args.num_blocks, True).cuda()
    print('Number of trainable parameters of Point CNF: {}'.format(count_parameters(model)))
    return model


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()
    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class PointFlow(nn.Module):

    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        self.encoder = Encoder(zdim=args.zdim, input_dim=args.input_dim, use_deterministic_encoder=args.use_deterministic_encoder)
        self.point_cnf = get_point_cnf(args)
        self.latent_cnf = get_latent_cnf(args) if args.use_latent_flow else nn.Sequential()

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size())
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1.0 + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_cnf = f(self.point_cnf)
        self.latent_cnf = f(self.latent_cnf)

    def make_optimizer(self, args):

        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_cnf.parameters()) + list(list(self.latent_cnf.parameters())))
        return opt

    def forward(self, x, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size)
        else:
            entropy = self.gaussian_entropy(z_sigma)
        if self.use_latent_flow:
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = torch.zeros(batch_size, 1)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.0).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()
        if self.distributed:
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_pz.mean()
        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)
        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)
        return {'entropy': entropy_log.cpu().detach().item() if not isinstance(entropy_log, float) else entropy_log, 'prior_nats': prior_nats, 'recon_nats': recon_nats}

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow, 'Sampling requires `self.use_latent_flow` to be True.'
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std)
        return x


class MovingBatchNormNd(nn.Module):

    def __init__(self, num_features, eps=0.0001, decay=0.1, bn_lag=0.0, affine=True, sync=False):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.sync = sync
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, c=None, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        num_channels = x.size(-1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()
        if self.training:
            x_t = x.transpose(0, 1).reshape(num_channels, -1)
            batch_mean = torch.mean(x_t, dim=1)
            if self.sync:
                batch_ex2 = torch.mean(x_t ** 2, dim=1)
                batch_mean = reduce_tensor(batch_mean)
                batch_ex2 = reduce_tensor(batch_ex2)
                batch_var = batch_ex2 - batch_mean ** 2
            else:
                batch_var = torch.var(x_t, dim=1)
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= 1.0 - self.bn_lag ** (self.step[0] + 1)
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= 1.0 - self.bn_lag ** (self.step[0] + 1)
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)
        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))
        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var
        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)
        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag}, affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'softplus': nn.Softplus(), 'elu': nn.ELU(), 'swish': Swish(), 'square': Lambda(lambda x: x ** 2), 'identity': Lambda(lambda x: x)}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type='concat', nonlinearity='softplus'):
        super(ODEnet, self).__init__()
        base_layer = {'ignore': diffeq_layers.IgnoreLinear, 'squash': diffeq_layers.SquashLinear, 'scale': diffeq_layers.ScaleLinear, 'concat': diffeq_layers.ConcatLinear, 'concat_v2': diffeq_layers.ConcatLinear_v2, 'concatsquash': diffeq_layers.ConcatSquashLinear, 'concatscale': diffeq_layers.ConcatScaleLinear}[layer_type]
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        for dim_out in (hidden_dims + (input_shape[0],)):
            layer_kwargs = {}
            layer = base_layer(hidden_shape[0], dim_out, context_dim, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, context, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(context, dx)
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)
    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1
    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, '(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s' % (f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx


class ODEfunc(nn.Module):

    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer('_num_evals', torch.tensor(0.0))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True)
        with torch.set_grad_enabled(True):
            if len(states) == 3:
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)
                dy = self.diffeq(tc, y)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)
                return dy, -divergence, torch.zeros_like(c).requires_grad_(True)
            elif len(states) == 2:
                dy = self.diffeq(t, y)
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -divergence
            else:
                assert 0, '`len(states)` should be 2 or 3'


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConcatLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 9]), torch.rand([4, 4, 4, 9]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatLinear_v2,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 5]), torch.rand([4, 4])], {}),
     True),
    (ConcatScaleLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 5]), torch.rand([4, 4])], {}),
     True),
    (ConcatSquashLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 5]), torch.rand([4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'zdim': 4}),
     lambda: ([torch.rand([4, 3, 3])], {}),
     False),
    (IgnoreLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 5]), torch.rand([4, 4])], {}),
     True),
    (SquashLinear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dim_c': 4}),
     lambda: ([torch.rand([4, 5]), torch.rand([4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_stevenygd_PointFlow(_paritybench_base):
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

