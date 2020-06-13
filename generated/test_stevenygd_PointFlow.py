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


import scipy.misc


from torch.backends import cudnn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None,
        integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, integration_times, reverse
                    )
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx,
                    integration_times, reverse)
            return x, logpx


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long,
        device=x.device)
    return x[tuple(indices)]


class CNF(nn.Module):

    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False,
        regularization_fns=None, solver='dopri5', atol=1e-05, rtol=1e-05,
        use_adjoint=True):
        super(CNF, self).__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter('sqrt_end_time', nn.Parameter(torch.
                sqrt(torch.tensor(T))))
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

    def forward(self, x, context=None, logpx=None, integration_times=None,
        reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
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
                integration_times = torch.stack([torch.tensor(0.0).to(x), 
                    self.sqrt_end_time * self.sqrt_end_time]).to(x)
            else:
                integration_times = torch.tensor([0.0, self.T],
                    requires_grad=False).to(x)
        if reverse:
            integration_times = _flip(integration_times, 0)
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(self.odefunc, states, integration_times.to(x),
                atol=atol, rtol=rtol, method=self.solver, options=self.
                solver_options)
        else:
            state_t = odeint(self.odefunc, states, integration_times.to(x),
                atol=self.test_atol, rtol=self.test_rtol, method=self.
                test_solver)
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()
    rt /= world_size
    return rt


class MovingBatchNormNd(nn.Module):

    def __init__(self, num_features, eps=0.0001, decay=0.1, bn_lag=0.0,
        affine=True, sync=False):
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
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean -
                    used_mean.detach())
                used_mean /= 1.0 - self.bn_lag ** (self.step[0] + 1)
                used_var = batch_var - (1 - self.bn_lag) * (batch_var -
                    used_var.detach())
                used_var /= 1.0 - self.bn_lag ** (self.step[0] + 1)
            self.running_mean -= self.decay * (self.running_mean -
                batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data
                )
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
            return y, logpx - self._logdetgrad(x, used_var).sum(-1, keepdim
                =True)

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
            return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim
                =True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag}, affine={affine})'
            .format(name=self.__class__.__name__, **self.__dict__))


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


NONLINEARITIES = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'softplus': nn.
    Softplus(), 'elu': nn.ELU(), 'swish': Swish(), 'square': Lambda(lambda
    x: x ** 2), 'identity': Lambda(lambda x: x)}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, context_dim, layer_type=
        'concat', nonlinearity='softplus'):
        super(ODEnet, self).__init__()
        base_layer = {'ignore': diffeq_layers.IgnoreLinear, 'squash':
            diffeq_layers.SquashLinear, 'scale': diffeq_layers.ScaleLinear,
            'concat': diffeq_layers.ConcatLinear, 'concat_v2':
            diffeq_layers.ConcatLinear_v2, 'concatsquash': diffeq_layers.
            ConcatSquashLinear, 'concatscale': diffeq_layers.ConcatScaleLinear
            }[layer_type]
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        for dim_out in (hidden_dims + (input_shape[0],)):
            layer_kwargs = {}
            layer = base_layer(hidden_shape[0], dim_out, context_dim, **
                layer_kwargs)
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
    assert approx_tr_dzdx.requires_grad, '(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s' % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e
        .requires_grad, e_dzdx_e.requires_grad, cnt)
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
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(
            True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)
        with torch.set_grad_enabled(True):
            if len(states) == 3:
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)
                dy = self.diffeq(tc, y)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)
                return dy, -divergence, torch.zeros_like(c).requires_grad_(True
                    )
            elif len(states) == 2:
                dy = self.diffeq(t, y)
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -divergence
            else:
                assert 0, '`len(states)` should be 2 or 3'


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_stevenygd_PointFlow(_paritybench_base):
    pass
    def test_000(self):
        self._check(IgnoreLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConcatLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([4, 4, 4, 9]), torch.rand([4, 4, 4, 9]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ConcatLinear_v2(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([5, 5]), torch.rand([4, 4, 5, 4])], {})

    def test_003(self):
        self._check(SquashLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([5, 5]), torch.rand([4, 4, 5, 4])], {})

    def test_004(self):
        self._check(ScaleLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([5, 5]), torch.rand([4, 4, 5, 4])], {})

    def test_005(self):
        self._check(ConcatSquashLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([5, 5]), torch.rand([4, 4, 5, 4])], {})

    def test_006(self):
        self._check(ConcatScaleLinear(*[], **{'dim_in': 4, 'dim_out': 4, 'dim_c': 4}), [torch.rand([5, 5]), torch.rand([4, 4, 5, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(Encoder(*[], **{'zdim': 4}), [torch.rand([4, 3, 3])], {})

    def test_008(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

