import sys
_module = sys.modules[__name__]
del sys
conv_lstm = _module
core = _module
convert2torch = _module
gqn_dataset = _module
model = _module
representation = _module
scheduler = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


from torch.nn import functional as F


from torch.distributions import Normal


from torch.distributions.kl import kl_divergence


class Conv2dLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(Conv2dLSTMCell, self).__init__()
        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        in_channels += out_channels
        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        cell, hidden = states
        input = torch.cat((hidden, input), dim=1)
        forget_gate = torch.sigmoid(self.forget(input))
        input_gate = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate = torch.tanh(self.state(input))
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)
        return cell, hidden


class InferenceCore(nn.Module):

    def __init__(self):
        super(InferenceCore, self).__init__()
        self.downsample_x = nn.Conv2d(3, 3, kernel_size=4, stride=4,
            padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=
            16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16,
            stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(128, 128, kernel_size=4, stride=4,
            padding=0, bias=False)
        self.core = Conv2dLSTMCell(3 + 7 + 256 + 2 * 128, 128, kernel_size=
            5, stride=1, padding=2)

    def forward(self, x, v, r, c_e, h_e, h_g, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2) != h_e.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))
        return c_e, h_e


class GenerationCore(nn.Module):

    def __init__(self):
        super(GenerationCore, self).__init__()
        self.upsample_v = nn.ConvTranspose2d(7, 7, kernel_size=16, stride=
            16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(256, 256, kernel_size=16,
            stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(7 + 256 + 3, 128, kernel_size=5, stride=
            1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(128, 128, kernel_size=4,
            stride=4, padding=0, bias=False)

    def forward(self, v, r, c_g, h_g, u, z):
        v = self.upsample_v(v.view(-1, 7, 1, 1))
        if r.size(2) != h_g.size(2):
            r = self.upsample_r(r)
        c_g, h_g = self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u
        return c_g, h_g, u


class GQN(nn.Module):

    def __init__(self, representation='pool', L=12, shared_core=False):
        super(GQN, self).__init__()
        self.L = L
        self.representation = representation
        if representation == 'pyramid':
            self.phi = Pyramid()
        elif representation == 'tower':
            self.phi = Tower()
        elif representation == 'pool':
            self.phi = Pool()
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore()
            self.generation_core = GenerationCore()
        else:
            self.inference_core = nn.ModuleList([InferenceCore() for _ in
                range(L)])
            self.generation_core = nn.ModuleList([GenerationCore() for _ in
                range(L)])
        self.eta_pi = nn.Conv2d(128, 2 * 3, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(128, 2 * 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x, v, v_q, x_q, sigma):
        B, M, *_ = x.size()
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, (k)], v[:, (k)])
            r += r_k
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        elbo = 0
        for l in range(self.L):
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5 * logvar_pi)
            pi = Normal(mu_pi, std_pi)
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u
                    )
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5 * logvar_q)
            q = Normal(mu_q, std_q)
            z = q.rsample()
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[
            1, 2, 3])
        return elbo

    def generate(self, x, v, v_q):
        B, M, *_ = x.size()
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, (k)], v[:, (k)])
            r += r_k
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        for l in range(self.L):
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5 * logvar_pi)
            pi = Normal(mu_pi, std_pi)
            z = pi.sample()
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
        mu = self.eta_g(u)
        return torch.clamp(mu, 0, 1)

    def kl_divergence(self, x, v, v_q, x_q):
        B, M, *_ = x.size()
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, (k)], v[:, (k)])
            r += r_k
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        kl = 0
        for l in range(self.L):
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5 * logvar_pi)
            pi = Normal(mu_pi, std_pi)
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u
                    )
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5 * logvar_q)
            q = Normal(mu_q, std_q)
            z = q.rsample()
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
            kl += torch.sum(kl_divergence(q, pi), dim=[1, 2, 3])
        return kl

    def reconstruct(self, x, v, v_q, x_q):
        B, M, *_ = x.size()
        if self.representation == 'tower':
            r = x.new_zeros((B, 256, 16, 16))
        else:
            r = x.new_zeros((B, 256, 1, 1))
        for k in range(M):
            r_k = self.phi(x[:, (k)], v[:, (k)])
            r += r_k
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
        for l in range(self.L):
            if self.shared_core:
                c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x_q, v_q, r, c_e, h_e, h_g, u
                    )
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5 * logvar_q)
            q = Normal(mu_q, std_q)
            z = q.rsample()
            if self.shared_core:
                c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](v_q, r, c_g, h_g, u, z)
        mu = self.eta_g(u)
        return torch.clamp(mu, 0, 1)


class Pyramid(nn.Module):

    def __init__(self):
        super(Pyramid, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(7 + 3, 32, kernel_size=2, stride
            =2), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.
            ReLU(), nn.Conv2d(64, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8), nn.ReLU())

    def forward(self, x, v):
        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.net(torch.cat((v, x), dim=1))
        return r


class Tower(nn.Module):

    def __init__(self):
        super(Tower, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256 + 7, 256, kernel_size=3, stride=1, padding=1
            )
        self.conv6 = nn.Conv2d(256 + 7, 128, kernel_size=3, stride=1, padding=1
            )
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

    def forward(self, x, v):
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))
        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5(skip_in))
        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        return r


class Pool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256 + 7, 256, kernel_size=3, stride=1, padding=1
            )
        self.conv6 = nn.Conv2d(256 + 7, 128, kernel_size=3, stride=1, padding=1
            )
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x, v):
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))
        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5(skip_in))
        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        r = self.pool(r)
        return r


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_iShohei220_torch_gqn(_paritybench_base):
    pass
    def test_000(self):
        self._check(Pool(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 7, 1, 1])], {})

    def test_001(self):
        self._check(Pyramid(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 7, 1, 1])], {})

    def test_002(self):
        self._check(Tower(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 7, 1, 1])], {})

