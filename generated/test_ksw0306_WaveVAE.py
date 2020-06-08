import sys
_module = sys.modules[__name__]
del sys
data = _module
model = _module
modules = _module
preprocessing = _module
synthesize = _module
train = _module
wavenet = _module
wavenet_iaf = _module

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


from torch import nn


from torch.nn import functional as F


from torch.distributions.normal import Normal


from math import log


from math import pi


import torch.nn as nn


import torch.nn.functional as F


import math


from torch import optim


from torch.utils.data import DataLoader


import numpy as np


class WaveVAE(nn.Module):

    def __init__(self):
        super(WaveVAE, self).__init__()
        self.encoder = Wavenet(out_channels=2, num_blocks=2, num_layers=10,
            residual_channels=128, gate_channels=256, skip_channels=128,
            kernel_size=2, cin_channels=80, upsample_scales=[16, 16])
        self.decoder = Wavenet_Student(num_blocks_student=[1, 1, 1, 1, 1, 1
            ], num_layers=10)
        self.log_eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, c):
        mu_logs = self.encoder(x, c)
        mu = mu_logs[:, 0:1, :-1]
        logs = mu_logs[:, 1:, :-1]
        q_0 = Normal(mu.new_zeros(mu.size()), mu.new_ones(mu.size()))
        mean_q = (x[:, :, 1:] - mu) * torch.exp(-logs)
        z = q_0.sample() * torch.exp(self.log_eps) + mean_q
        z_prior = q_0.sample()
        z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        z_prior = F.pad(z_prior, pad=(1, 0), mode='constant', value=0)
        c_up = self.encoder.upsample(c)
        x_rec, mu_p, log_p = self.decoder(z, c_up)
        x_prior = self.decoder.generate(z_prior, c_up)
        loss_recon = -0.5 * (-log(2.0 * pi) - 2.0 * log_p - torch.pow(x[:,
            :, 1:] - mu_p, 2) * torch.exp(-2.0 * log_p))
        loss_kl = 0.5 * (mean_q ** 2 + torch.exp(self.log_eps) ** 2 - 1
            ) - self.log_eps
        return x_rec, x_prior, loss_recon.mean(), loss_kl.mean()

    def generate(self, z, c):
        c_up = self.encoder.upsample(c)
        x_sample = self.decoder.generate(z, c_up)
        return x_sample


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
        causal=False, mode='SAME'):
        super(Conv, self).__init__()
        self.causal = causal
        self.mode = mode
        if self.causal and self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1)
        elif self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1) // 2
        else:
            self.padding = 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels,
        kernel_size, dilation, cin_channels=None, local_conditioning=True,
        causal=False, mode='SAME'):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.mode = mode
        self.filter_conv = Conv(in_channels, out_channels, kernel_size,
            dilation, causal, mode)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size,
            dilation, causal, mode)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        nn.init.kaiming_normal_(self.skip_conv.weight)
        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels,
                kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels,
                kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)
        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)
        out = F.tanh(h_filter) * F.sigmoid(h_gate)
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        if self.mode == 'SAME':
            return (tensor + res) * math.sqrt(0.5), skip
        else:
            return (tensor[:, :, 1:] + res) * math.sqrt(0.5), skip

    def remove_weight_norm(self):
        self.filter_conv.remove_weight_norm()
        self.gate_conv.remove_weight_norm()
        nn.utils.remove_weight_norm(self.res_conv)
        nn.utils.remove_weight_norm(self.skip_conv)
        nn.utils.remove_weight_norm(self.filter_conv_c)
        nn.utils.remove_weight_norm(self.gate_conv_c)


class Wavenet(nn.Module):

    def __init__(self, out_channels=1, num_blocks=3, num_layers=10,
        residual_channels=512, gate_channels=512, skip_channels=512,
        kernel_size=2, cin_channels=128, upsample_scales=None, causal=True):
        super(Wavenet, self).__init__()
        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size
        self.front_channels = 32
        self.front_conv = nn.Sequential(Conv(1, self.residual_channels,
            self.front_channels, causal=self.causal), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels,
                    self.gate_channels, self.skip_channels, self.
                    kernel_size, dilation=self.kernel_size ** n,
                    cin_channels=self.cin_channels, local_conditioning=True,
                    causal=self.causal, mode='SAME'))
        self.final_conv = nn.Sequential(nn.ReLU(), Conv(self.skip_channels,
            self.skip_channels, 1, causal=self.causal), nn.ReLU(), Conv(
            self.skip_channels, self.out_channels, 1, causal=self.causal))
        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2
                ), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        c = self.upsample(c)
        out = self.wavenet(x, c)
        return out

    def upsample(self, c):
        if self.upsample_conv is not None:
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            c = c.squeeze(1)
        return c

    def wavenet(self, tensor, c=None):
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [(2 ** (i % self.num_layers)) for i in range(self.
            num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations
            ) + self.front_channels


class Wavenet_Student(nn.Module):

    def __init__(self, num_blocks_student=[1, 1, 1, 1, 1, 1], num_layers=10,
        front_channels=32, residual_channels=64, gate_channels=128,
        skip_channels=64, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Student, self).__init__()
        self.num_blocks = num_blocks_student
        self.num_flow = len(self.num_blocks)
        self.num_layers = num_layers
        self.iafs = nn.ModuleList()
        for i in range(self.num_flow):
            self.iafs.append(Wavenet_Flow(out_channels=2, num_blocks=self.
                num_blocks[i], num_layers=self.num_layers, front_channels=
                front_channels, residual_channels=residual_channels,
                gate_channels=gate_channels, skip_channels=skip_channels,
                kernel_size=kernel_size, cin_channels=cin_channels, causal=
                causal))

    def forward(self, z, c):
        return self.iaf(z, c)

    def iaf(self, z, c_up):
        mu_tot, logs_tot = 0.0, 0.0
        for i, iaf in enumerate(self.iafs):
            mu_logs = iaf(z, c_up)
            mu = mu_logs[:, 0:1, :-1]
            logs = mu_logs[:, 1:, :-1]
            mu_tot = mu_tot * torch.exp(logs) + mu
            logs_tot = logs_tot + logs
            z = z[:, :, 1:] * torch.exp(logs) + mu
            z = F.pad(z, pad=(1, 0), mode='constant', value=0)
        return z, mu_tot, logs_tot

    def receptive_field(self):
        receptive_field = 1
        for iaf in self.iafs:
            receptive_field += iaf.receptive_field_size() - 1
        return receptive_field

    def generate(self, z, c_up):
        x, _, _ = self.iaf(z, c_up)
        return x

    def remove_weight_norm(self):
        for iaf in self.iafs:
            iaf.remove_weight_norm()


class Wavenet_Flow(nn.Module):

    def __init__(self, out_channels=1, num_blocks=1, num_layers=10,
        front_channels=32, residual_channels=64, gate_channels=32,
        skip_channels=None, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet_Flow, self).__init__()
        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.front_channels = front_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size
        self.front_conv = nn.Sequential(Conv(1, self.residual_channels,
            self.front_channels, causal=self.causal), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        self.res_blocks_fast = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels,
                    self.gate_channels, self.skip_channels, self.
                    kernel_size, dilation=2 ** n, cin_channels=self.
                    cin_channels, local_conditioning=True, causal=self.
                    causal, mode='SAME'))
        self.final_conv = nn.Sequential(nn.ReLU(), Conv(self.skip_channels,
            self.skip_channels, 1, causal=self.causal), nn.ReLU(), Conv(
            self.skip_channels, self.out_channels, 1, causal=self.causal))

    def forward(self, x, c):
        return self.wavenet(x, c)

    def wavenet(self, tensor, c=None):
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [(2 ** (i % self.num_layers)) for i in range(self.
            num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + 1 + (self
            .front_channels - 1)

    def remove_weight_norm(self):
        for f in self.res_blocks:
            f.remove_weight_norm()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ksw0306_WaveVAE(_paritybench_base):
    pass

    def test_000(self):
        self._check(Conv(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})
    @_fails_compile()

    def test_001(self):
        self._check(Wavenet_Student(*[], **{}), [torch.rand([4, 1, 64]), torch.rand([4, 80, 64])], {})
