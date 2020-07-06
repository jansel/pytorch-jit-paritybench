import sys
_module = sys.modules[__name__]
del sys
gym_pendulum = _module
gym_pendulum_approximate = _module
mpc = _module
dynamics = _module
env_dx = _module
cartpole = _module
control = _module
pendulum = _module
lqr_step = _module
mpc = _module
pnqp = _module
torch_numdiff = _module
util = _module
setup = _module
test_dynamics = _module
test_mpc = _module

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


import logging


import math


import time


import numpy as np


import torch


import torch.autograd


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.functional as F


from torch import nn


from torch.nn.parameter import Parameter


from torch.nn import Module


import numpy.random as npr


from collections import namedtuple


from enum import Enum


import torch.nn as nn


import itertools


from torch.autograd import grad


import numpy.testing as npt


from numpy.testing import dec


ACTS = {'sigmoid': torch.sigmoid, 'relu': F.relu, 'elu': F.elu}


class NNDynamics(nn.Module):

    def __init__(self, n_state, n_ctrl, hidden_sizes=[100], activation='sigmoid', passthrough=True):
        super().__init__()
        self.passthrough = passthrough
        self.fcs = []
        in_sz = n_state + n_ctrl
        for out_sz in (hidden_sizes + [n_state]):
            fc = nn.Linear(in_sz, out_sz)
            self.fcs.append(fc)
            in_sz = out_sz
        self.fcs = nn.ModuleList(self.fcs)
        assert activation in ACTS.keys()
        act_f = ACTS[activation]
        self.activation = activation
        self.acts = [act_f] * (len(self.fcs) - 1) + [lambda x: x]
        self.Ws = [y.weight for y in self.fcs]
        self.zs = []

    def __getstate__(self):
        return self.fcs, self.activation, self.passthrough

    def __setstate__(self, state):
        super().__init__()
        if len(state) == 2:
            self.fcs, self.activation = state
            self.passthrough = True
        else:
            self.fcs, self.activation, self.passthrough = state
        act_f = ACTS[self.activation]
        self.acts = [act_f] * (len(self.fcs) - 1) + [lambda x: x]
        self.Ws = [y.weight for y in self.fcs]

    def forward(self, x, u):
        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)
        self.zs = []
        z = torch.cat((x, u), 1)
        for act, fc in zip(self.acts, self.fcs):
            z = act(fc(z))
            self.zs.append(z)
        self.zs = self.zs[:-1]
        if self.passthrough:
            z += x
        if x_dim == 1:
            z = z.squeeze(0)
        return z

    def grad_input(self, x, u):
        assert isinstance(x, Variable) == isinstance(u, Variable)
        diff = isinstance(x, Variable)
        x_dim, u_dim = x.ndimension(), u.ndimension()
        n_batch, n_state = x.size()
        _, n_ctrl = u.size()
        if not diff:
            Ws = [W.data for W in self.Ws]
            zs = [z.data for z in self.zs]
        else:
            Ws = self.Ws
            zs = self.zs
        assert len(zs) == len(Ws) - 1
        grad = Ws[-1].repeat(n_batch, 1, 1)
        for i in range(len(zs) - 1, 0 - 1, -1):
            n_out, n_in = Ws[i].size()
            if self.activation == 'relu':
                I = util.get_data_maybe(zs[i] <= 0.0).unsqueeze(2).repeat(1, 1, n_in)
                Wi_grad = Ws[i].repeat(n_batch, 1, 1)
                Wi_grad[I] = 0.0
            elif self.activation == 'sigmoid':
                d = zs[i] * (1.0 - zs[i])
                d = d.unsqueeze(2).expand(n_batch, n_out, n_in)
                Wi_grad = Ws[i].repeat(n_batch, 1, 1) * d
            else:
                assert False
            grad = grad.bmm(Wi_grad)
        R = grad[:, :, :n_state]
        S = grad[:, :, n_state:]
        if self.passthrough:
            I = torch.eye(n_state).type_as(util.get_data_maybe(R)).unsqueeze(0).repeat(n_batch, 1, 1)
            if diff:
                I = Variable(I)
            R = R + I
        if x_dim == 1:
            R = R.squeeze(0)
            S = S.squeeze(0)
        return R, S


class CtrlPassthroughDynamics(nn.Module):

    def __init__(self, dynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, tilde_x, u):
        tilde_x_dim, u_dim = tilde_x.ndimension(), u.ndimension()
        if tilde_x_dim == 1:
            tilde_x = tilde_x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)
        n_ctrl = u.size(1)
        x = tilde_x[:, n_ctrl:]
        xtp1 = self.dynamics(x, u)
        tilde_xtp1 = torch.cat((u, xtp1), dim=1)
        if tilde_x_dim == 1:
            tilde_xtp1 = tilde_xtp1.squeeze()
        return tilde_xtp1

    def grad_input(self, x, u):
        assert False, 'Unimplemented'


class AffineDynamics(nn.Module):

    def __init__(self, A, B, c=None):
        super(AffineDynamics, self).__init__()
        assert A.ndimension() == 2
        assert B.ndimension() == 2
        if c is not None:
            assert c.ndimension() == 1
        self.A = A
        self.B = B
        self.c = c

    def forward(self, x, u):
        if not isinstance(x, Variable) and isinstance(self.A, Variable):
            A = self.A.data
            B = self.B.data
            c = self.c.data if self.c is not None else 0.0
        else:
            A = self.A
            B = self.B
            c = self.c if self.c is not None else 0.0
        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)
        z = x.mm(A.t()) + u.mm(B.t()) + c
        if x_dim == 1:
            z = z.squeeze(0)
        return z

    def grad_input(self, x, u):
        n_batch = x.size(0)
        A, B = self.A, self.B
        A = A.unsqueeze(0).repeat(n_batch, 1, 1)
        B = B.unsqueeze(0).repeat(n_batch, 1, 1)
        if not isinstance(x, Variable) and isinstance(A, Variable):
            A, B = A.data, B.data
        return A, B


class CartpoleDx(nn.Module):

    def __init__(self, params=None):
        super().__init__()
        self.n_state = 5
        self.n_ctrl = 1
        if params is None:
            self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        assert len(self.params) == 4
        self.force_mag = 100.0
        self.theta_threshold_radians = np.pi
        self.x_threshold = 2.4
        self.max_velocity = 10
        self.dt = 0.05
        self.lower = -self.force_mag
        self.upper = self.force_mag
        self.goal_state = torch.Tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        self.goal_weights = torch.Tensor([0.1, 0.1, 1.0, 1.0, 0.1])
        self.ctrl_penalty = 0.001
        self.mpc_eps = 0.0001
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length
        u = torch.clamp(u[:, (0)], -self.force_mag, self.force_mag)
        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        th = torch.atan2(sin_th, cos_th)
        cart_in = (u + polemass_length * dth ** 2 * sin_th) / total_mass
        th_acc = (gravity * sin_th - cos_th * cart_in) / (length * (4.0 / 3.0 - masspole * cos_th ** 2 / total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass
        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc
        state = torch.stack((x, dx, torch.cos(th), torch.sin(th), dth), 1)
        return state

    def get_frame(self, state):
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 5
        x, dx, cos_th, sin_th, dth = torch.unbind(state)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th * length * 2
        th_y = cos_th * length * 2
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot((x, x + th_x), (0, th_y), color='k')
        ax.set_xlim((-5.0, 5.0))
        ax.set_ylim((-2.0, 2.0))
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights) * self.goal_state
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


class PendulumDx(nn.Module):

    def __init__(self, params=None, simple=True):
        super().__init__()
        self.simple = simple
        self.max_torque = 2.0
        self.dt = 0.05
        self.n_state = 3
        self.n_ctrl = 1
        if params is None:
            if simple:
                self.params = Variable(torch.Tensor((10.0, 1.0, 1.0)))
            else:
                self.params = Variable(torch.Tensor((10.0, 1.0, 1.0, 0.0, 0.0)))
        else:
            self.params = params
        assert len(self.params) == 3 if simple else 5
        self.goal_state = torch.Tensor([1.0, 0.0, 0.0])
        self.goal_weights = torch.Tensor([1.0, 1.0, 0.1])
        self.ctrl_penalty = 0.001
        self.lower, self.upper = -2.0, 2.0
        self.mpc_eps = 0.001
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        squeeze = x.ndimension() == 1
        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 3
        assert u.shape[1] == 1
        assert u.ndimension() == 2
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params
        if not hasattr(self, 'simple') or self.simple:
            g, m, l = torch.unbind(self.params)
        else:
            g, m, l, d, b = torch.unbind(self.params)
        u = torch.clamp(u, -self.max_torque, self.max_torque)[:, (0)]
        cos_th, sin_th, dth = torch.unbind(x, dim=1)
        th = torch.atan2(sin_th, cos_th)
        if not hasattr(self, 'simple') or self.simple:
            newdth = dth + self.dt * (-3.0 * g / (2.0 * l) * -sin_th + 3.0 * u / (m * l ** 2))
        else:
            sin_th_bias = torch.sin(th + b)
            newdth = dth + self.dt * (-3.0 * g / (2.0 * l) * -sin_th_bias + 3.0 * u / (m * l ** 2) - d * th)
        newth = th + newdth * self.dt
        state = torch.stack((torch.cos(newth), torch.sin(newth), newdth), dim=1)
        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 3
        l = self.params[2].item()
        cos_th, sin_th, dth = torch.unbind(x)
        th = np.arctan2(sin_th, cos_th)
        x = sin_th * l
        y = cos_th * l
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()
        ax.plot((0, x), (0, y), color='k')
        ax.set_xlim((-l * 1.2, l * 1.2))
        ax.set_ylim((-l * 1.2, l * 1.2))
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights) * self.goal_state
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


class SlewRateCost(Module):
    """Hacky way of adding the slew rate penalty to costs."""

    def __init__(self, cost, slew_C, n_state, n_ctrl):
        super().__init__()
        self.cost = cost
        self.slew_C = slew_C
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def forward(self, tau):
        true_tau = tau[:, self.n_ctrl:]
        true_cost = self.cost(true_tau)
        slew_cost = 0.5 * util.bquad(tau, self.slew_C[0])
        return true_cost + slew_cost

    def grad_input(self, x, u):
        raise NotImplementedError('Implement grad_input')


class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')


LqrForOut = namedtuple('lqrForOut', 'objs full_du_norm alpha_du_norm mean_alphas costs')


def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11 * torch.eye(n).type_as(H).expand_as(H)

    def obj(x):
        return 0.5 * util.bquad(x, H) + util.bdot(q, x)
    if x_init is None:
        if n == 1:
            x_init = -(1.0 / H.squeeze(2)) * q
        else:
            H_lu = H.lu()
            x_init = -q.unsqueeze(2).lu_solve(*H_lu).squeeze(2)
    else:
        x_init = x_init.clone()
    x = util.eclamp(x_init, lower, upper)
    J = torch.ones(n_batch).type_as(x).byte()
    for i in range(n_iter):
        g = util.bmv(H, x) + q
        Ic = ((x == lower) & (g > 0) | (x == upper) & (g < 0)).float()
        If = 1 - Ic
        if If.is_cuda:
            Hff_I = util.bger(If.float(), If.float()).type_as(If)
            not_Hff_I = 1 - Hff_I
            Hfc_I = util.bger(If.float(), Ic.float()).type_as(If)
        else:
            Hff_I = util.bger(If, If)
            not_Hff_I = 1 - Hff_I
            Hfc_I = util.bger(If, Ic)
        g_ = g.clone()
        g_[Ic.bool()] = 0.0
        H_ = H.clone()
        H_[not_Hff_I.bool()] = 0.0
        H_ += pnqp_I
        if n == 1:
            dx = -(1.0 / H_.squeeze(2)) * g_
        else:
            H_lu_ = H_.lu()
            dx = -g_.unsqueeze(2).lu_solve(*H_lu_).squeeze(2)
        J = torch.norm(dx, 2, 1) >= 0.0001
        m = J.sum().item()
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i
        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0
        while max_armijo <= GAMMA and count < 10:
            maybe_x = util.eclamp(x + torch.diag(alpha).mm(dx), lower, upper)
            armijos = (GAMMA + 1e-06) * torch.ones(n_batch).type_as(x)
            armijos[J] = (obj(x) - obj(maybe_x))[J] / util.bdot(g, x - maybe_x)[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1
        x = maybe_x
    None
    return x, H_ if n == 1 else H_lu_, If, i


class LQRStep(Function):
    """A single step of the box-constrained iLQR solver.

    Required Args:
        n_state, n_ctrl, T
        x_init: The initial state [n_batch, n_state]

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
            TODO: Better support automatic expansion of these.
        TODO
    """

    def __init__(self, n_state, n_ctrl, T, u_lower=None, u_upper=None, u_zero_I=None, delta_u=None, linesearch_decay=0.2, max_linesearch_iter=10, true_cost=None, true_dynamics=None, delta_space=True, current_x=None, current_u=None, verbose=0, back_eps=0.001, no_op_forward=False):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = util.get_data_maybe(u_lower)
        self.u_upper = util.get_data_maybe(u_upper)
        if isinstance(self.u_lower, int):
            self.u_lower = float(self.u_lower)
        if isinstance(self.u_upper, int):
            self.u_upper = float(self.u_upper)
        if isinstance(self.u_lower, np.float32):
            self.u_lower = u_lower.item()
        if isinstance(self.u_upper, np.float32):
            self.u_upper = u_upper.item()
        self.u_zero_I = u_zero_I
        self.delta_u = delta_u
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.true_cost = true_cost
        self.true_dynamics = true_dynamics
        self.delta_space = delta_space
        self.current_x = util.get_data_maybe(current_x)
        self.current_u = util.get_data_maybe(current_u)
        self.verbose = verbose
        self.back_eps = back_eps
        self.no_op_forward = no_op_forward

    def forward(self, x_init, C, c, F, f=None):
        if self.no_op_forward:
            self.save_for_backward(x_init, C, c, F, f, self.current_x, self.current_u)
            return self.current_x, self.current_u
        if self.delta_space:
            assert self.current_x is not None
            assert self.current_u is not None
            c_back = []
            for t in range(self.T):
                xt = self.current_x[t]
                ut = self.current_u[t]
                xut = torch.cat((xt, ut), 1)
                c_back.append(util.bmv(C[t], xut) + c[t])
            c_back = torch.stack(c_back)
            f_back = None
        else:
            assert False
        Ks, ks, self.back_out = self.lqr_backward(C, c_back, F, f_back)
        new_x, new_u, self.for_out = self.lqr_forward(x_init, C, c, F, f, Ks, ks)
        self.save_for_backward(x_init, C, c, F, f, new_x, new_u)
        return new_x, new_u

    def backward(self, dl_dx, dl_du):
        start = time.time()
        x_init, C, c, F, f, new_x, new_u = self.saved_tensors
        r = []
        for t in range(self.T):
            rt = torch.cat((dl_dx[t], dl_du[t]), 1)
            r.append(rt)
        r = torch.stack(r)
        if self.u_lower is None:
            I = None
        else:
            I = (torch.abs(new_u - self.u_lower) <= 1e-08) | (torch.abs(new_u - self.u_upper) <= 1e-08)
        dx_init = Variable(torch.zeros_like(x_init))
        _mpc = mpc.MPC(self.n_state, self.n_ctrl, self.T, u_zero_I=I, u_init=None, lqr_iter=1, verbose=-1, n_batch=C.size(1), delta_u=None, exit_unconverged=False, eps=self.back_eps)
        dx, du, _ = _mpc(dx_init, mpc.QuadCost(C, -r), mpc.LinDx(F, None))
        dx, du = dx.data, du.data
        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)
        dC = torch.zeros_like(C)
        for t in range(self.T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
            dC[t] = dCt
        dc = -dxu
        lams = []
        prev_lam = None
        for t in range(self.T - 1, -1, -1):
            Ct_xx = C[(t), :, :self.n_state, :self.n_state]
            Ct_xu = C[(t), :, :self.n_state, self.n_state:]
            ct_x = c[(t), :, :self.n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[(t), :, :, :self.n_state].transpose(1, 2)
                lamt += util.bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))
        dlams = []
        prev_dlam = None
        for t in range(self.T - 1, -1, -1):
            dCt_xx = C[(t), :, :self.n_state, :self.n_state]
            dCt_xu = C[(t), :, :self.n_state, self.n_state:]
            drt_x = -r[(t), :, :self.n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[(t), :, :, :self.n_state].transpose(1, 2)
                dlamt += util.bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))
        dF = torch.zeros_like(F)
        for t in range(self.T - 1):
            xut = xu[t]
            lamt = lams[t + 1]
            dxut = dxu[t]
            dlamt = dlams[t + 1]
            dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))
        if f.nelement() > 0:
            _dlams = dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            df = torch.Tensor()
        dx_init = -dlams[0]
        self.backward_time = time.time() - start
        return dx_init, dC, dc, dF, df

    def lqr_backward(self, C, c, F, f):
        n_batch = C.size(1)
        u = self.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(self.T - 1, -1, -1):
            if t == self.T - 1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1, 2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
            n_state = self.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]
            if self.u_lower is None:
                if self.n_ctrl == 1 and self.u_zero_I is None:
                    Kt = -(1.0 / Qt_uu) * Qt_ux
                    kt = -(1.0 / Qt_uu.squeeze(2)) * qt_u
                elif self.u_zero_I is None:
                    Qt_uu_inv = [torch.pinverse(Qt_uu[i]) for i in range(Qt_uu.shape[0])]
                    Qt_uu_inv = torch.stack(Qt_uu_inv)
                    Kt = -Qt_uu_inv.bmm(Qt_ux)
                    kt = util.bmv(-Qt_uu_inv, qt_u)
                else:
                    I = self.u_zero_I[t].float()
                    notI = 1 - I
                    qt_u_ = qt_u.clone()
                    qt_u_[I.bool()] = 0
                    Qt_uu_ = Qt_uu.clone()
                    if I.is_cuda:
                        notI_ = notI.float()
                        Qt_uu_I = (1 - util.bger(notI_, notI_)).type_as(I)
                    else:
                        Qt_uu_I = 1 - util.bger(notI, notI)
                    Qt_uu_[Qt_uu_I.bool()] = 0.0
                    Qt_uu_[util.bdiag(I).bool()] += 1e-08
                    Qt_ux_ = Qt_ux.clone()
                    Qt_ux_[I.unsqueeze(2).repeat(1, 1, Qt_ux.size(2)).bool()] = 0.0
                    if self.n_ctrl == 1:
                        Kt = -(1.0 / Qt_uu_) * Qt_ux_
                        kt = -(1.0 / Qt_uu.squeeze(2)) * qt_u_
                    else:
                        Qt_uu_LU_ = Qt_uu_.lu()
                        Kt = -Qt_ux_.lu_solve(*Qt_uu_LU_)
                        kt = -qt_u_.unsqueeze(2).lu_solve(*Qt_uu_LU_).squeeze(2)
            else:
                assert self.delta_space
                lb = self.get_bound('lower', t) - u[t]
                ub = self.get_bound('upper', t) - u[t]
                if self.delta_u is not None:
                    lb[lb < -self.delta_u] = -self.delta_u
                    ub[ub > self.delta_u] = self.delta_u
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(Qt_uu, qt_u, lb, ub, x_init=prev_kt, n_iter=20)
                if self.verbose > 1:
                    None
                n_total_qp_iter += 1 + n_qp_iter
                prev_kt = kt
                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[(1 - If).unsqueeze(2).repeat(1, 1, Qt_ux.size(2)).bool()] = 0
                if self.n_ctrl == 1:
                    Kt = -(1.0 / Qt_uu_free_LU * Qt_ux_)
                else:
                    Kt = -Qt_ux_.lu_solve(*Qt_uu_free_LU)
            Kt_T = Kt.transpose(1, 2)
            Ks.append(Kt)
            ks.append(kt)
            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)
        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)

    def lqr_forward(self, x_init, C, c, F, f, Ks, ks):
        x = self.current_x
        u = self.current_u
        n_batch = C.size(1)
        old_cost = util.get_cost(self.T, u, self.true_cost, self.true_dynamics, x=x)
        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None
        i = 0
        while (current_cost is None or old_cost is not None and torch.any(current_cost > old_cost).cpu().item() == 1) and i < self.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(self.T):
                t_rev = self.T - 1 - t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)
                assert not (self.delta_u is not None and self.u_lower is None)
                if self.u_zero_I is not None:
                    new_ut[self.u_zero_I[t]] = 0.0
                if self.u_lower is not None:
                    lb = self.get_bound('lower', t)
                    ub = self.get_bound('upper', t)
                    if self.delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - self.delta_u
                        ub = u[t] + self.delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    new_ut = util.eclamp(new_ut, lb, ub)
                new_u.append(new_ut)
                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < self.T - 1:
                    if isinstance(self.true_dynamics, mpc.LinDx):
                        F, f = self.true_dynamics.F, self.true_dynamics.f
                        new_xtp1 = util.bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                    else:
                        new_xtp1 = self.true_dynamics(Variable(new_xt), Variable(new_ut)).data
                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t + 1])
                if isinstance(self.true_cost, mpc.QuadCost):
                    C, c = self.true_cost.C, self.true_cost.c
                    obj = 0.5 * util.bquad(new_xut, C[t]) + util.bdot(new_xut, c[t])
                else:
                    obj = self.true_cost(new_xut)
                objs.append(obj)
            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)
            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u - new_u).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)
            alphas[current_cost > old_cost] *= self.linesearch_decay
            i += 1
        alphas[current_cost > old_cost] /= self.linesearch_decay
        alpha_du_norm = (u - new_u).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)
        return new_x, new_u, LqrForOut(objs, full_du_norm, alpha_du_norm, torch.mean(alphas), current_cost)

    def get_bound(self, side, t):
        v = getattr(self, 'u_' + side)
        if isinstance(v, float):
            return v
        else:
            return v[t]


LinDx = namedtuple('LinDx', 'F f')


QuadCost = namedtuple('QuadCost', 'C c')


class MPC(Module):
    """A differentiable box-constrained iLQR solver.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        lqr_iter: The number of LQR iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(self, n_state, n_ctrl, T, u_lower=None, u_upper=None, u_zero_I=None, u_init=None, lqr_iter=10, grad_method=GradMethods.ANALYTIC, delta_u=None, verbose=0, eps=1e-07, back_eps=1e-07, n_batch=None, linesearch_decay=0.2, max_linesearch_iter=10, exit_unconverged=True, detach_unconverged=True, backprop=True, slew_rate_penalty=None, prev_ctrl=None, not_improved_lim=5, best_cost_eps=0.0001):
        super().__init__()
        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper
        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)
        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)
        self.u_zero_I = util.detach_maybe(u_zero_I)
        self.u_init = util.detach_maybe(u_init)
        self.lqr_iter = lqr_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps
        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl

    def forward(self, x_init, cost, dx):
        assert isinstance(cost, QuadCost) or isinstance(cost, Module) or isinstance(cost, Function)
        assert isinstance(dx, LinDx) or isinstance(dx, Module) or isinstance(dx, Function)
        if self.n_batch is not None:
            n_batch = self.n_batch
        elif isinstance(cost, QuadCost) and cost.C.ndimension() == 4:
            n_batch = cost.C.size(1)
        else:
            None
            sys.exit(-1)
        if isinstance(cost, QuadCost):
            C, c = cost
            if C.ndimension() == 2:
                C = C.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, self.n_state + self.n_ctrl, -1)
            elif C.ndimension() == 3:
                C = C.unsqueeze(1).expand(self.T, n_batch, self.n_state + self.n_ctrl, -1)
            if c.ndimension() == 1:
                c = c.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, -1)
            elif c.ndimension() == 2:
                c = c.unsqueeze(1).expand(self.T, n_batch, -1)
            if C.ndimension() != 4 or c.ndimension() != 3:
                None
                sys.exit(-1)
            cost = QuadCost(C, c)
        assert x_init.ndimension() == 2 and x_init.size(0) == n_batch
        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x_init.data)
        if self.verbose > 0:
            None
        best = None
        n_not_improved = 0
        for i in range(self.lqr_iter):
            u = Variable(util.detach_maybe(u), requires_grad=True)
            x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)
            if isinstance(dx, LinDx):
                F, f = dx.F, dx.f
            else:
                F, f = self.linearize_dynamics(x, util.detach_maybe(u), dx, diff=False)
            if isinstance(cost, QuadCost):
                C, c = cost.C, cost.c
            else:
                C, c, _ = self.approximate_cost(x, util.detach_maybe(u), cost, diff=False)
            x, u, _lqr = self.solve_lqr_subproblem(x_init, C, c, F, f, cost, dx, x, u)
            back_out, for_out = _lqr.back_out, _lqr.for_out
            n_not_improved += 1
            assert x.ndimension() == 3
            assert u.ndimension() == 3
            if best is None:
                best = {'x': list(torch.split(x, split_size_or_sections=1, dim=1)), 'u': list(torch.split(u, split_size_or_sections=1, dim=1)), 'costs': for_out.costs, 'full_du_norm': for_out.full_du_norm}
            else:
                for j in range(n_batch):
                    if for_out.costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:, (j)].unsqueeze(1)
                        best['u'][j] = u[:, (j)].unsqueeze(1)
                        best['costs'][j] = for_out.costs[j]
                        best['full_du_norm'][j] = for_out.full_du_norm[j]
            if self.verbose > 0:
                util.table_log('lqr', (('iter', i), ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'), ('||full_du||_max', max(for_out.full_du_norm).item(), '{:.2e}'), ('mean(alphas)', for_out.mean_alphas.item(), '{:.2e}'), ('total_qp_iters', back_out.n_total_qp_iter)))
            if max(for_out.full_du_norm) < self.eps or n_not_improved > self.not_improved_lim:
                break
        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1)
        full_du_norm = best['full_du_norm']
        if isinstance(dx, LinDx):
            F, f = dx.F, dx.f
        else:
            F, f = self.linearize_dynamics(x, u, dx, diff=True)
        if isinstance(cost, QuadCost):
            C, c = cost.C, cost.c
        else:
            C, c, _ = self.approximate_cost(x, u, cost, diff=True)
        x, u, _ = self.solve_lqr_subproblem(x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)
        if self.detach_unconverged:
            if max(best['full_du_norm']) > self.eps:
                if self.exit_unconverged:
                    assert False
                if self.verbose >= 0:
                    None
                    None
                I = for_out.full_du_norm < self.eps
                Ix = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(x)).type_as(x.data)
                Iu = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(u)).type_as(u.data)
                x = x * Ix + x.clone().detach() * (1.0 - Ix)
                u = u * Iu + u.clone().detach() * (1.0 - Iu)
        costs = best['costs']
        return x, u, costs

    def solve_lqr_subproblem(self, x_init, C, c, F, f, cost, dynamics, x, u, no_op_forward=False):
        if self.slew_rate_penalty is None or isinstance(cost, Module):
            _lqr = LQRStep(n_state=self.n_state, n_ctrl=self.n_ctrl, T=self.T, u_lower=self.u_lower, u_upper=self.u_upper, u_zero_I=self.u_zero_I, true_cost=cost, true_dynamics=dynamics, delta_u=self.delta_u, linesearch_decay=self.linesearch_decay, max_linesearch_iter=self.max_linesearch_iter, delta_space=True, current_x=x, current_u=u, back_eps=self.back_eps, no_op_forward=no_op_forward)
            e = Variable(torch.Tensor())
            x, u = _lqr(x_init, C, c, F, f if f is not None else e)
            return x, u, _lqr
        else:
            nsc = self.n_state + self.n_ctrl
            _n_state = nsc
            _nsc = _n_state + self.n_ctrl
            n_batch = C.size(1)
            _C = torch.zeros(self.T, n_batch, _nsc, _nsc).type_as(C)
            half_gamI = self.slew_rate_penalty * torch.eye(self.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(self.T, n_batch, 1, 1)
            _C[:, :, :self.n_ctrl, :self.n_ctrl] = half_gamI
            _C[:, :, -self.n_ctrl:, :self.n_ctrl] = -half_gamI
            _C[:, :, :self.n_ctrl, -self.n_ctrl:] = -half_gamI
            _C[:, :, -self.n_ctrl:, -self.n_ctrl:] = half_gamI
            slew_C = _C.clone()
            _C = _C + torch.nn.ZeroPad2d((self.n_ctrl, 0, self.n_ctrl, 0))(C)
            _c = torch.cat((torch.zeros(self.T, n_batch, self.n_ctrl).type_as(c), c), 2)
            _F0 = torch.cat((torch.zeros(self.n_ctrl, self.n_state + self.n_ctrl), torch.eye(self.n_ctrl)), 1).type_as(F).unsqueeze(0).unsqueeze(0).repeat(self.T - 1, n_batch, 1, 1)
            _F1 = torch.cat((torch.zeros(self.T - 1, n_batch, self.n_state, self.n_ctrl).type_as(F), F), 3)
            _F = torch.cat((_F0, _F1), 2)
            if f is not None:
                _f = torch.cat((torch.zeros(self.T - 1, n_batch, self.n_ctrl).type_as(f), f), 2)
            else:
                _f = Variable(torch.Tensor())
            u_data = util.detach_maybe(u)
            if self.prev_ctrl is not None:
                prev_u = self.prev_ctrl
                if prev_u.ndimension() == 1:
                    prev_u = prev_u.unsqueeze(0)
                if prev_u.ndimension() == 2:
                    prev_u = prev_u.unsqueeze(0)
                prev_u = prev_u.data
            else:
                prev_u = torch.zeros(1, n_batch, self.n_ctrl).type_as(u)
            utm1s = torch.cat((prev_u, u_data[:-1])).clone()
            _x = torch.cat((utm1s, x), 2)
            _x_init = torch.cat((Variable(prev_u[0]), x_init), 1)
            if not isinstance(dynamics, LinDx):
                _dynamics = CtrlPassthroughDynamics(dynamics)
            else:
                _dynamics = None
            if isinstance(cost, QuadCost):
                _true_cost = QuadCost(_C, _c)
            else:
                _true_cost = SlewRateCost(cost, slew_C, self.n_state, self.n_ctrl)
            _lqr = LQRStep(n_state=_n_state, n_ctrl=self.n_ctrl, T=self.T, u_lower=self.u_lower, u_upper=self.u_upper, u_zero_I=self.u_zero_I, true_cost=_true_cost, true_dynamics=_dynamics, delta_u=self.delta_u, linesearch_decay=self.linesearch_decay, max_linesearch_iter=self.max_linesearch_iter, delta_space=True, current_x=_x, current_u=u, back_eps=self.back_eps, no_op_forward=no_op_forward)
            x, u = _lqr(_x_init, _C, _c, _F, _f)
            x = x[:, :, self.n_ctrl:]
            return x, u, _lqr

    def approximate_cost(self, x, u, Cf, diff=True):
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2).data
            tau = Variable(tau, requires_grad=True)
            if self.slew_rate_penalty is not None:
                None
                sys.exit(-1)
                differences = tau[1:, :, -self.n_ctrl:] - tau[:-1, :, -self.n_ctrl:]
                slew_penalty = (self.slew_rate_penalty * differences.pow(2)).sum(-1)
            costs = list()
            hessians = list()
            grads = list()
            for t in range(self.T):
                tau_t = tau[t]
                if self.slew_rate_penalty is not None:
                    cost = Cf(tau_t) + (slew_penalty[t - 1] if t > 0 else 0)
                else:
                    cost = Cf(tau_t)
                grad = torch.autograd.grad(cost.sum(), tau_t, retain_graph=True)[0]
                hessian = list()
                for v_i in range(tau.shape[2]):
                    hessian.append(torch.autograd.grad(grad[:, (v_i)].sum(), tau_t, retain_graph=True)[0])
                hessian = torch.stack(hessian, dim=-1)
                costs.append(cost)
                grads.append(grad - util.bmv(hessian, tau_t))
                hessians.append(hessian)
            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)
            if not diff:
                return hessians.data, grads.data, costs.data
            return hessians, grads, costs

    def linearize_dynamics(self, x, u, dynamics, diff):
        n_batch = x[0].size(0)
        if self.grad_method == GradMethods.ANALYTIC:
            _u = Variable(u[:-1].view(-1, self.n_ctrl), requires_grad=True)
            _x = Variable(x[:-1].contiguous().view(-1, self.n_state), requires_grad=True)
            _new_x = dynamics(_x, _u)
            if not diff:
                _new_x = _new_x.data
                _x = _x.data
                _u = _u.data
            R, S = dynamics.grad_input(_x, _u)
            f = _new_x - util.bmv(R, _x) - util.bmv(S, _u)
            f = f.view(self.T - 1, n_batch, self.n_state)
            R = R.contiguous().view(self.T - 1, n_batch, self.n_state, self.n_state)
            S = S.contiguous().view(self.T - 1, n_batch, self.n_state, self.n_ctrl)
            F = torch.cat((R, S), 3)
            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f
        else:
            x_init = x[0]
            x = [x_init]
            F, f = [], []
            for t in range(self.T):
                if t < self.T - 1:
                    xt = Variable(x[t], requires_grad=True)
                    ut = Variable(u[t], requires_grad=True)
                    xut = torch.cat((xt, ut), 1)
                    new_x = dynamics(xt, ut)
                    if self.grad_method in [GradMethods.AUTO_DIFF, GradMethods.ANALYTIC_CHECK]:
                        Rt, St = [], []
                        for j in range(self.n_state):
                            Rj, Sj = torch.autograd.grad(new_x[:, (j)].sum(), [xt, ut], retain_graph=True)
                            if not diff:
                                Rj, Sj = Rj.data, Sj.data
                            Rt.append(Rj)
                            St.append(Sj)
                        Rt = torch.stack(Rt, dim=1)
                        St = torch.stack(St, dim=1)
                        if self.grad_method == GradMethods.ANALYTIC_CHECK:
                            assert False
                            Rt_autograd, St_autograd = Rt, St
                            Rt, St = dynamics.grad_input(xt, ut)
                            eps = 1e-08
                            if torch.max(torch.abs(Rt - Rt_autograd)).data[0] > eps or torch.max(torch.abs(St - St_autograd)).data[0] > eps:
                                None
                            else:
                                None
                            sys.exit(0)
                    elif self.grad_method == GradMethods.FINITE_DIFF:
                        Rt, St = [], []
                        for i in range(n_batch):
                            Ri = util.jacobian(lambda s: dynamics(s, ut[i]), xt[i], 0.0001)
                            Si = util.jacobian(lambda a: dynamics(xt[i], a), ut[i], 0.0001)
                            if not diff:
                                Ri, Si = Ri.data, Si.data
                            Rt.append(Ri)
                            St.append(Si)
                        Rt = torch.stack(Rt)
                        St = torch.stack(St)
                    else:
                        assert False
                    Ft = torch.cat((Rt, St), 2)
                    F.append(Ft)
                    if not diff:
                        xt, ut, new_x = xt.data, ut.data, new_x.data
                    ft = new_x - util.bmv(Rt, xt) - util.bmv(St, ut)
                    f.append(ft)
                if t < self.T - 1:
                    x.append(util.detach_maybe(new_x))
            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NNDynamics,
     lambda: ([], {'n_state': 4, 'n_ctrl': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_locuslab_mpc_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

