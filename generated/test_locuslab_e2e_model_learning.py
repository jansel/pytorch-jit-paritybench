import sys
_module = sys.modules[__name__]
del sys
calc_stats = _module
main = _module
model_classes = _module
nets = _module
batch = _module
main = _module
mle = _module
mle_net = _module
plot = _module
policy_net = _module
task_net = _module
main = _module
model_classes = _module
nets = _module
plot = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import scipy.io as sio


import numpy as np


import torch


import torch.nn as nn


from torch.autograd import Variable


from torch.autograd import Function


import torch.optim as optim


import torch.cuda


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from functools import reduce


import scipy.stats as st


from torch.nn.parameter import Parameter


class Net(nn.Module):

    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()
        X_ = np.hstack([X, np.ones((X.shape[0], 1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W, b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1, :].T)
        b.data = torch.Tensor(Theta[(-1), :])
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, [[nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1]))

    def forward(self, x):
        return self.lin(x) + self.net(x), self.sig.expand(x.size(0), self.sig.size(1))

    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred - Y) ** 2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)


class ScheduleBattery(nn.Module):
    """ Get battery schedule that maximizes objective """

    def __init__(self, params):
        super(ScheduleBattery, self).__init__()
        self.T = params['T']
        T = params['T']
        eps = params['epsilon']
        IT = torch.eye(T)
        eff = params['eff']
        in_max = params['in_max']
        out_max = params['out_max']
        self.B = params['B']
        self.lam = params['lambda']
        D1 = torch.cat([torch.eye(T - 1), torch.zeros(1, T - 1)], 0)
        D2 = torch.cat([torch.zeros(1, T - 1), torch.eye(T - 1)], 0)
        self.Q = Variable(block([[eps * torch.eye(T), 0, 0], [0, eps * torch.eye(T), 0], [0, 0, self.lam * torch.eye(T)]]))
        Ae_list = [[torch.zeros(1, T), torch.zeros(1, T), torch.ones(1, 1), torch.zeros(1, T - 1)], [D1.t() * eff, -D1.t(), D1.t() - D2.t()]]
        self.Ae = Variable(torch.cat(map(lambda x: torch.cat(x, 1), Ae_list), 0))
        self.be = Variable(torch.cat([self.B / 2 * torch.ones(1), torch.zeros(T - 1)]))
        self.A = Variable(block([[torch.eye(T), 0, 0], [-torch.eye(T), 0, 0], [0, torch.eye(T), 0], [0, -torch.eye(T), 0], [0, 0, torch.eye(T)], [0, 0, -torch.eye(T)]]))
        self.b = Variable(torch.Tensor([in_max] * T + [0] * T + [out_max] * T + [0] * T + [self.B] * T + [0] * T))

    def forward(self, log_prices):
        prices = torch.exp(log_prices)
        nBatch = prices.size(0)
        T = self.T
        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        c = torch.cat([prices, -prices, -Variable(self.lam * self.B * torch.ones(T)).unsqueeze(0).expand(nBatch, T)], 1)
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))
        Ae = self.Ae.unsqueeze(0).expand(nBatch, self.Ae.size(0), self.Ae.size(1))
        be = self.be.unsqueeze(0).expand(nBatch, self.be.size(0))
        out = QPFunction(verbose=True)(Q.double(), c.double(), A.double(), b.double(), Ae.double(), be.double())
        return out


class SolveNewsvendor(nn.Module):
    """ Solve newsvendor scheduling problem """

    def __init__(self, params, eps=0.01):
        super(SolveNewsvendor, self).__init__()
        k = len(params['d'])
        self.Q = Variable(torch.diag(torch.Tensor([params['c_quad']] + [params['b_quad']] * k + [params['h_quad']] * k)))
        self.p = Variable(torch.Tensor([params['c_lin']] + [params['b_lin']] * k + [params['h_lin']] * k))
        self.G = Variable(torch.cat([torch.cat([-torch.ones(k, 1), -torch.eye(k), torch.zeros(k, k)], 1), torch.cat([torch.ones(k, 1), torch.zeros(k, k), -torch.eye(k)], 1), -torch.eye(1 + 2 * k)], 0))
        self.h = Variable(torch.Tensor(np.concatenate([-params['d'], params['d'], np.zeros(1 + 2 * k)])))
        self.one = Variable(torch.Tensor([1]))
        self.eps_eye = eps * Variable(torch.eye(1 + 2 * k)).unsqueeze(0)

    def forward(self, y):
        nBatch, k = y.size()
        Q_scale = torch.cat([torch.diag(torch.cat([self.one, y[i], y[i]])).unsqueeze(0) for i in range(nBatch)], 0)
        Q = self.Q.unsqueeze(0).expand_as(Q_scale).mul(Q_scale)
        p_scale = torch.cat([Variable(torch.ones(nBatch, 1)), y, y], 1)
        p = self.p.unsqueeze(0).expand_as(p_scale).mul(p_scale)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        e = Variable(torch.Tensor()).double()
        out = QPFunction(verbose=False)(Q.double(), p.double(), G.double(), h.double(), e, e).float()
        return out[:, :1]


class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""

    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params['c_ramp']
        self.n = params['n']
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = Variable(torch.DoubleTensor(np.vstack([D, -D])))
        self.h = Variable((self.c_ramp * torch.ones((self.n - 1) * 2)).double())
        self.e = Variable(torch.Tensor().double())

    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) for i in range(nBatch)], 0).double()
        p = (dg - d2g * z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out


class GLinearApprox(Function):
    """ Linear (gradient) approximation of G function at z"""

    def __init__(self, gamma_under, gamma_over):
        self.gamma_under = gamma_under
        self.gamma_over = gamma_over

    def forward(self, z, mu, sig):
        self.save_for_backward(z, mu, sig)
        p = st.norm(mu.cpu().numpy(), sig.cpu().numpy())
        return torch.DoubleTensor((self.gamma_under + self.gamma_over) * p.cdf(z.cpu().numpy()) - self.gamma_under)

    def backward(self, grad_output):
        z, mu, sig = self.saved_tensors
        p = st.norm(mu.cpu().numpy(), sig.cpu().numpy())
        pz = torch.DoubleTensor(p.pdf(z.cpu().numpy()))
        dz = (self.gamma_under + self.gamma_over) * pz
        dmu = -dz
        dsig = -(self.gamma_under + self.gamma_over) * (z - mu) / sig * pz
        return grad_output * dz, grad_output * dmu, grad_output * dsig


class GQuadraticApprox(Function):
    """ Quadratic (gradient) approximation of G function at z"""

    def __init__(self, gamma_under, gamma_over):
        self.gamma_under = gamma_under
        self.gamma_over = gamma_over

    def forward(self, z, mu, sig):
        self.save_for_backward(z, mu, sig)
        p = st.norm(mu.cpu().numpy(), sig.cpu().numpy())
        return torch.DoubleTensor((self.gamma_under + self.gamma_over) * p.pdf(z.cpu().numpy()))

    def backward(self, grad_output):
        z, mu, sig = self.saved_tensors
        p = st.norm(mu.cpu().numpy(), sig.cpu().numpy())
        pz = torch.DoubleTensor(p.pdf(z.cpu().numpy()))
        dz = -(self.gamma_under + self.gamma_over) * (z - mu) / sig ** 2 * pz
        dmu = -dz
        dsig = (self.gamma_under + self.gamma_over) * ((z - mu) ** 2 - sig ** 2) / sig ** 3 * pz
        return grad_output * dz, grad_output * dmu, grad_output * dsig


class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """

    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params['c_ramp']
        self.n = params['n']
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = Variable(torch.DoubleTensor(np.vstack([D, -D])))
        self.h = Variable((self.c_ramp * torch.ones((self.n - 1) * 2)).double())
        self.e = Variable(torch.Tensor().double())

    def forward(self, mu, sig):
        nBatch, n = mu.size()
        z0 = Variable(1.0 * mu.data, requires_grad=False)
        mu0 = Variable(1.0 * mu.data, requires_grad=False)
        sig0 = Variable(1.0 * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params['gamma_under'], self.params['gamma_over'])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params['gamma_under'], self.params['gamma_over'])(z0, mu0, sig0)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0 - z0_new).norm().data[0]
            None
            z0 = z0_new
            if solution_diff < 1e-10:
                break
        dg = GLinearApprox(self.params['gamma_under'], self.params['gamma_over'])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params['gamma_under'], self.params['gamma_over'])(z0, mu, sig)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)

