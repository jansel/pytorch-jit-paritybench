import sys
_module = sys.modules[__name__]
del sys
data = _module
model = _module
modules = _module
preprocessing = _module
synthesize = _module
train = _module
train_apex = _module

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


from math import log


from math import pi


import math


import torch.nn as nn


from torch import optim


from torch.utils.data import DataLoader


from torch.distributions.normal import Normal


import numpy as np


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):

    def __init__(self, in_channel, logdet=True, pretrained=False):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))
        self.initialized = pretrained
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-06))

    def forward(self, x):
        B, _, T = x.size()
        if not self.initialized:
            self.initialize(x)
            self.initialized = True
        log_abs = logabs(self.scale)
        logdet = torch.sum(log_abs) * B * T
        if self.logdet:
            return self.scale * (x + self.loc), logdet
        else:
            return self.scale * (x + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class AffineCoupling(nn.Module):

    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=
        6, affine=True):
        super().__init__()
        self.affine = affine
        self.net = Wavenet(in_channels=in_channel // 2, out_channels=
            in_channel if self.affine else in_channel // 2, num_blocks=1,
            num_layers=num_layer, residual_channels=filter_size,
            gate_channels=filter_size, skip_channels=filter_size,
            kernel_size=3, cin_channels=cin_channel // 2, causal=False)

    def forward(self, x, c=None):
        in_a, in_b = x.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a, c_a).chunk(2, 1)
            out_b = (in_b - t) * torch.exp(-log_s)
            logdet = torch.sum(-log_s)
        else:
            net_out = self.net(in_a, c_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, c=None):
        out_a, out_b = output.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a, c_a).chunk(2, 1)
            in_b = out_b * torch.exp(log_s) + t
        else:
            net_out = self.net(out_a, c_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


def change_order(x, c=None):
    x_a, x_b = x.chunk(2, 1)
    c_a, c_b = c.chunk(2, 1)
    return torch.cat([x_b, x_a], 1), torch.cat([c_b, c_a], 1)


class Flow(nn.Module):

    def __init__(self, in_channel, cin_channel, filter_size, num_layer,
        affine=True, pretrained=False):
        super().__init__()
        self.actnorm = ActNorm(in_channel, pretrained=pretrained)
        self.coupling = AffineCoupling(in_channel, cin_channel, filter_size
            =filter_size, num_layer=num_layer, affine=affine)

    def forward(self, x, c=None):
        out, logdet = self.actnorm(x)
        out, det = self.coupling(out, c)
        out, c = change_order(out, c)
        if det is not None:
            logdet = logdet + det
        return out, c, logdet

    def reverse(self, output, c=None):
        output, c = change_order(output, c)
        x = self.coupling.reverse(output, c)
        x = self.actnorm.reverse(x)
        return x, c


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(
        2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):

    def __init__(self, in_channel, cin_channel, n_flow, n_layer, affine=
        True, pretrained=False, split=False):
        super().__init__()
        self.split = split
        squeeze_dim = in_channel * 2
        squeeze_dim_c = cin_channel * 2
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, squeeze_dim_c, filter_size=
                256, num_layer=n_layer, affine=affine, pretrained=pretrained))
        if self.split:
            self.prior = Wavenet(in_channels=squeeze_dim // 2, out_channels
                =squeeze_dim, num_blocks=1, num_layers=2, residual_channels
                =256, gate_channels=256, skip_channels=256, kernel_size=3,
                cin_channels=squeeze_dim_c, causal=False)

    def forward(self, x, c):
        b_size, n_channel, T = x.size()
        squeezed_x = x.view(b_size, n_channel, T // 2, 2).permute(0, 1, 3, 2)
        out = squeezed_x.contiguous().view(b_size, n_channel * 2, T // 2)
        squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
        c = squeezed_c.contiguous().view(b_size, -1, T // 2)
        logdet, log_p = 0, 0
        for flow in self.flows:
            out, c, det = flow(out, c)
            logdet = logdet + det
        if self.split:
            out, z = out.chunk(2, 1)
            mean, log_sd = self.prior(out, c).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd).sum()
        return out, c, logdet, log_p

    def reverse(self, output, c, eps=None):
        if self.split:
            mean, log_sd = self.prior(output, c).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)
            x = torch.cat([output, z_new], 1)
        else:
            x = output
        for flow in self.flows[::-1]:
            x, c = flow.reverse(x, c)
        b_size, n_channel, T = x.size()
        unsqueezed_x = x.view(b_size, n_channel // 2, 2, T).permute(0, 1, 3, 2)
        unsqueezed_x = unsqueezed_x.contiguous().view(b_size, n_channel // 
            2, T * 2)
        unsqueezed_c = c.view(b_size, -1, 2, T).permute(0, 1, 3, 2)
        unsqueezed_c = unsqueezed_c.contiguous().view(b_size, -1, T * 2)
        return unsqueezed_x, unsqueezed_c


class Flowavenet(nn.Module):

    def __init__(self, in_channel, cin_channel, n_block, n_flow, n_layer,
        affine=True, pretrained=False, block_per_split=8):
        super().__init__()
        self.block_per_split = block_per_split
        self.blocks = nn.ModuleList()
        self.n_block = n_block
        for i in range(self.n_block):
            split = False if (i + 1
                ) % self.block_per_split or i == self.n_block - 1 else True
            self.blocks.append(Block(in_channel, cin_channel, n_flow,
                n_layer, affine=affine, pretrained=pretrained, split=split))
            cin_channel *= 2
            if not split:
                in_channel *= 2
        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2
                ), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0
        out = x
        c = self.upsample(c)
        for block in self.blocks:
            out, c, logdet_new, logp_new = block(out, c)
            logdet = logdet + logdet_new
            log_p_sum = log_p_sum + logp_new
        log_p_sum += 0.5 * (-log(2.0 * pi) - out.pow(2)).sum()
        logdet = logdet / (B * T)
        log_p = log_p_sum / (B * T)
        return log_p, logdet

    def reverse(self, z, c):
        _, _, T = z.size()
        _, _, t_c = c.size()
        if T != t_c:
            c = self.upsample(c)
        z_list = []
        x = z
        for i in range(self.n_block):
            b_size, _, T = x.size()
            squeezed_x = x.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            x = squeezed_x.contiguous().view(b_size, -1, T // 2)
            squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            c = squeezed_c.contiguous().view(b_size, -1, T // 2)
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                x, z = x.chunk(2, 1)
                z_list.append(z)
        for i, block in enumerate(self.blocks[::-1]):
            index = self.n_block - i
            if not (index % self.block_per_split or index == self.n_block):
                x, c = block.reverse(x, c, z_list[index // self.
                    block_per_split - 1])
            else:
                x, c = block.reverse(x, c)
        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
        causal=True):
        super(Conv, self).__init__()
        self.causal = causal
        if self.causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out


class ZeroConv1d(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, skip_channels,
        kernel_size, dilation, cin_channels=None, local_conditioning=True,
        causal=False):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True if skip_channels is not None else False
        self.filter_conv = Conv(in_channels, out_channels, kernel_size,
            dilation, causal)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size,
            dilation, causal)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        if self.skip:
            self.skip_conv = nn.Conv1d(out_channels, skip_channels,
                kernel_size=1)
            self.skip_conv = nn.utils.weight_norm(self.skip_conv)
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
        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


class Wavenet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, num_blocks=1,
        num_layers=6, residual_channels=256, gate_channels=256,
        skip_channels=256, kernel_size=3, cin_channels=80, causal=True):
        super(Wavenet, self).__init__()
        self.skip = True if skip_channels is not None else False
        self.front_conv = nn.Sequential(Conv(in_channels, residual_channels,
            3, causal=causal), nn.ReLU())
        self.res_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for n in range(num_layers):
                self.res_blocks.append(ResBlock(residual_channels,
                    gate_channels, skip_channels, kernel_size, dilation=2 **
                    n, cin_channels=cin_channels, local_conditioning=True,
                    causal=causal))
        last_channels = skip_channels if self.skip else residual_channels
        self.final_conv = nn.Sequential(nn.ReLU(), Conv(last_channels,
            last_channels, 1, causal=causal), nn.ReLU(), ZeroConv1d(
            last_channels, out_channels))

    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            if self.skip:
                h, s = f(h, c)
                skip += s
            else:
                h, _ = f(h, c)
        if self.skip:
            out = self.final_conv(skip)
        else:
            out = self.final_conv(h)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ksw0306_FloWaveNet(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(ActNorm(*[], **{'in_channel': 4}), [torch.rand([4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(Block(*[], **{'in_channel': 4, 'cin_channel': 4, 'n_flow': 4, 'n_layer': 1}), [torch.rand([4, 4, 4]), torch.rand([4, 1, 4, 4])], {})

    def test_002(self):
        self._check(Conv(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 64])], {})

    def test_003(self):
        self._check(ZeroConv1d(*[], **{'in_channel': 4, 'out_channel': 4}), [torch.rand([4, 4, 64])], {})
