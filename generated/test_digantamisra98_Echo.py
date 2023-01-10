import sys
_module = sys.modules[__name__]
del sys
Activation = _module
m_ops = _module
t_ops = _module
tf_ops = _module
cv = _module
t_attn = _module
echoAI = _module
utils = _module
torch_utils = _module
setup = _module
test = _module
test_t_ops = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Function


from torch.nn.parameter import Parameter


from torch.distributions.exponential import Exponential


import math


class Aria2(nn.Module):

    def __init__(self, beta=0.5, alpha=1.0):
        """
        Init method.
        """
        super(Aria2, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return torch.pow(1 + torch.exp(-self.beta * input), -self.alpha)


def swish_function(input, swish, eswish, beta, param):
    if swish is True and eswish is False:
        return input * torch.sigmoid(param * input)
    if eswish is True and swish is False:
        return beta * input * torch.sigmoid(input)


class Swish(nn.Module):

    def __init__(self, eswish=False, swish=False, beta=1.735, flatten=False, pfts=False):
        """
        Init method.
        """
        super(Swish, self).__init__()
        self.swish = swish
        self.eswish = eswish
        self.flatten = flatten
        self.beta = None
        self.param = None
        if eswish is not False:
            self.beta = beta
        if swish is not False:
            self.param = nn.Parameter(torch.randn(1))
            self.param.requires_grad = True
        if flatten is not False:
            if pfts is not False:
                self.const = nn.Parameter(torch.tensor(-0.2))
                self.const.requires_grad = True
            else:
                self.const = -0.2
        if eswish is not False and swish is not False and flatten is not False:
            raise RuntimeError('Advisable to run either Swish or E-Swish or Flatten T-Swish')

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.swish is not False:
            return swish_function(input, self.swish, self.eswish, self.beta, self.param)
        if self.eswish is not False:
            return swish_function(input, self.swish, self.eswish, self.beta, self.param)
        if self.flatten is not False:
            return (input >= 0).float() * (input * swish_function(input, self.swish, self.eswish, self.beta, self.param) + self.const) + (input < 0).float() * self.const


class Elish(nn.Module):

    def __init__(self, hard=False):
        """
        Init method.
        """
        super(Elish, self).__init__()
        self.hard = hard
        if hard is not False:
            self.a = torch.tensor(0.0)
            self.b = torch.tensor(1.0)

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.hard is False:
            return (input >= 0).float() * swish_function(input, False, False, None, None) + (input < 0).float() * (torch.exp(input) - 1) * torch.sigmoid(input)
        else:
            return (input >= 0).float() * input * torch.max(self.a, torch.min(self.b, (input + 1.0) / 2.0)) + (input < 0).float() * (torch.exp(input - 1) * torch.max(self.a, torch.min(self.b, (input + 1.0) / 2.0)))


def isru(input, alpha):
    return input / torch.sqrt(1 + alpha * torch.pow(input, 2))


class ISRU(nn.Module):

    def __init__(self, alpha=1.0, isrlu=False):
        """
        Init method.
        """
        super().__init__()
        self.alpha = alpha
        self.isrlu = isrlu

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.isrlu is not False:
            return (input < 0).float() * isru(input, self.apha) + (input >= 0).float() * input
        else:
            return isru(input, self.apha)


class NLReLU(nn.Module):

    def __init__(self, beta=1.0, inplace=False):
        """
        Init method.
        """
        super().__init__()
        self.beta = beta
        self.inplace = inplace

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if inplace:
            return torch.log(F.relu_(x).mul_(self.beta).add_(1), out=x)
        else:
            return torch.log(1 + self.beta * F.relu(x))


class SoftClipping(nn.Module):

    def __init__(self, alpha=0.5):
        """
        Init method.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return 1 / self.alpha * torch.log((1 + torch.exp(self.alpha * input)) / (1 + torch.exp(self.alpha * (input - 1))))


class SoftExponential(nn.Module):

    def __init__(self, alpha=None):
        """
        Init method.
        """
        super(SoftExponential, self).__init__()
        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        """
        Forward pass of the function
        """
        if self.alpha == 0.0:
            return x
        if self.alpha < 0.0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha
        if self.alpha > 0.0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


class SQNL(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(SQNL, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return (input > 2).float() + (input - torch.pow(input, 2) / 4) * (input >= 0).float() * (input <= 2).float() + (input + torch.pow(input, 2) / 4) * (input < 0).float() * (input >= -2).float() - (input < -2).float()


class SReLU(nn.Module):

    def __init__(self, in_features, parameters=None):
        """
        Init method.
        """
        super(SReLU, self).__init__()
        self.in_features = in_features
        if parameters is None:
            self.tr = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad=True))
            self.tl = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad=True))
            self.ar = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad=True))
            self.al = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad=True))
        else:
            self.tr, self.tl, self.ar, self.al = parameters

    def forward(self, input):
        """
        Forward pass of the function
        """
        return (input >= self.tr).float() * (self.tr + self.ar * (input + self.tr)) + (input < self.tr).float() * (input > self.tl).float() * input + (input <= self.tl).float() * (self.tl + self.al * (input + self.tl))


class brelu_function(Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        input_shape = input.shape[0]
        even_indices = [i for i in range(0, input_shape, 2)]
        odd_indices = [i for i in range(1, input_shape, 2)]
        output = input.clone()
        output[even_indices] = output[even_indices].clamp(min=0)
        output[odd_indices] = 0 - output[odd_indices]
        output[odd_indices] = -output[odd_indices].clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None
        input, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            input_shape = input.shape[0]
            even_indices = [i for i in range(0, input_shape, 2)]
            odd_indices = [i for i in range(1, input_shape, 2)]
            grad_input[even_indices] = (input[even_indices] >= 0).float() * grad_input[even_indices]
            grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[odd_indices]
        return grad_input


class BReLU(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(BReLU, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function
        """
        return brelu_function.apply(input)


class APL(nn.Module):

    def __init__(self, s=1):
        """
        Init method.
        """
        super(APL, self).__init__()
        self.a = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(s)])
        self.b = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(s)])
        self.s = s

    def forward(self, input):
        """
        Forward pass of the function.
        """
        part_1 = torch.clamp_min(input, min=0.0)
        part_2 = 0
        for i in range(self.s):
            part_2 += self.a[i] * torch.clamp_min(-input + self.b[i], min=0)
        return part_1 + part_2


class Maxout(nn.Module):

    def __init__(self, pool_size=1):
        """
        Init method.
        """
        super(Maxout, self).__init__()
        self._pool_size = pool_size

    def forward(self, input):
        """
        Forward pass of the function.
        """
        assert input.shape[1] % self._pool_size == 0, 'Wrong input last dim size ({}) for Maxout({})'.format(input.shape[1], self._pool_size)
        m, i = input.view(*input.shape[:1], input.shape[1] // self._pool_size, self._pool_size, *input.shape[2:]).max(2)
        return m


class Funnel(nn.Module):

    def __init__(self, in_channels):
        """
        Init method.
        """
        super(Funnel, self).__init__()
        self.conv_funnel = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_funnel = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        """
        Forward pass of the function
        """
        tau = self.conv_funnel(input)
        tau = self.bn_funnel(tau)
        output = torch.max(input, tau)
        return output


class SLAF(nn.Module):

    def __init__(self, k=2):
        """
        Init method.
        """
        super(SLAF, self).__init__()
        self.k = k
        self.coeff = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for i in range(k)])

    def forward(self, input):
        """
        Forward pass of the function
        """
        out = sum([(self.coeff[k] * torch.pow(input, k)) for k in range(self.k)])
        return out


class AReLU(nn.Module):

    def __init__(self, alpha=0.9, beta=2.0):
        super(AReLU, self).__init__()
        """
        Init method.
        """
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        """
        Forward pass of the function
        """
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)
        return F.relu(input) * beta - F.relu(-input) * alpha


class FReLU(nn.Module):

    def __init__(self, in_channels):
        """
        Init method.
        """
        super(FReLU, self).__init__()
        self.bias = nn.Parameter(torch.randn(1))
        self.bias.requires_grad = True

    def forward(self, input):
        """
        Forward pass of the function
        """
        return F.relu(input) + self.bias


class DICE(nn.Module):

    def __init__(self, emb_size, dim=2, epsilon=1e-08):
        super(DICE, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.dim == 2:
            self.alpha = torch.zeros((emb_size,))
        else:
            self.alpha = torch.zeros((emb_size, 1))

    def forward(self, input):
        assert input.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(input))
            out = self.alpha * (1 - x_p) * input + x_p * input
        else:
            input = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(input))
            out = self.alpha * (1 - x_p) * input + x_p * input
            out = torch.transpose(out, 1, 2)
        return out


class MPeLU(nn.Module):

    def __init__(self, alpha=0.25, beta=1.0):
        super(MPeLU, self).__init__()
        """
        Init method.
        """
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))
        self.elu = nn.ELU(self.alpha)

    def forward(self, input):
        """
        Forward pass of the function
        """
        return (input > 0).float() * input + (input <= 0).float() * self.elu(self.beta * input)


class TanhSoft(nn.Module):

    def __init__(self, alpha=0.0, beta=0.6, gamma=1.0, delta=0.0):
        super(TanhSoft, self).__init__()
        """
        Init method.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        if self.alpha > 1:
            raise RuntimeError('Alpha should be less than equal to 1')
        if self.beta < 0:
            raise RuntimeError('Beta should be greater than equal to 0')
        if self.gamma <= 0:
            raise RuntimeError('Gamma should be greater than 0')
        if self.delta > 1 or self.delta < 0:
            raise RuntimeError('Delta should be in range of [0,1]')

    def forward(self, input):
        """
        Forward pass of the function
        """
        return torch.tanh(self.alpha * input + self.beta * torch.exp(self.gamma * input)) * torch.log(self.delta + torch.exp(input))


class ProbAct(nn.Module):

    def __init__(self, num_parameters=1, init=0):
        """
        Init method.
        """
        super(ProbAct, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        """
        Forward pass of the function
        """
        mu = input
        if mu.is_cuda:
            eps = torch.FloatTensor(mu.size()).normal_(mean=0, std=1)
        else:
            eps = torch.FloatTensor(mu.size()).normal_(mean=0, std=1)
        return F.relu(mu) + self.weight * eps

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)


class XUnit(nn.Module):

    def __init__(self, out_channels, kernel_size=9):
        """
        Init method.
        """
        super(XUnit, self).__init__()
        self.module = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=out_channels), nn.BatchNorm2d(out_channels))

    def forward(self, input):
        """
        Forward pass of the function
        """
        x1 = self.module(input)
        out = torch.exp(-torch.mul(x1, x1))
        return torch.mul(input, out)


class EIS(nn.Module):

    def __init__(self, alpha, beta, gamma, delta, theta, version=0):
        super(EIS, self).__init__()
        """
        Init method.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.version = version
        if self.alpha > 1 or self.alpha < 0:
            raise RuntimeError('Alpha should be in range [0,1]')
        if self.beta < 0:
            raise RuntimeError('Beta should be greater than equal to 0')
        if self.gamma < 0:
            raise RuntimeError('Gamma should be greater than equal to 0')
        if self.delta < 0:
            raise RuntimeError('Delta should be greater than equal to 0')
        if self.theta < 0:
            raise RuntimeError('Theta should be greater than equal to 0')
        if self.version not in [0, 1, 2, 3]:
            raise RuntimeError('EIS Version is not supported')
        if self.version == 1:
            self.alpha = 1.0
            self.beta = 0.0
            self.gamma = 1.0
            self.delta = 1.16
            self.theta = 1.0
        if self.version == 2:
            self.alpha = 1.0
            self.delta = 0.0
            self.theta = 0.0
            self.beta = 1.0
            self.gamma = 1.0
        if self.version == 3:
            self.alpha = 0.0
            self.beta = 1.0
            self.gamma = 0.0
            self.delta = 0.68
            self.theta = 1.7

    def forward(self, input):
        """
        Forward pass of the function
        """
        num = input * torch.pow(F.softplus(input), self.alpha)
        den = torch.sqrt(self.beta + self.gamma * torch.pow(input, 2)) + self.delta * torch.exp(-self.theta * input)
        return num / den


class Seagull(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(Seagull, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function
        """
        return torch.log(1 + torch.pow(input, 2))


class Snake(nn.Module):

    def __init__(self, in_features, alpha=None, alpha_trainable=True):
        """
        Init method.
        """
        super(Snake, self).__init__()
        self.in_features = in_features
        if alpha is not None:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
        else:
            m = Exponential(torch.tensor([0.1]))
            self.alpha = Parameter(m.sample(in_features))
        self.alpha.requiresGrad = alpha_trainable

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return x + 1.0 / self.alpha * torch.pow(torch.sin(x * self.alpha), 2)


class Sine(nn.Module):

    def __init__(self, w0=1.0):
        """
        Init method.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return torch.sin(self.w0 * x)


def exists(val):
    return val is not None


class Siren(nn.Module):

    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, use_bias=True, activation=None):
        """
        Init method.
        """
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        w_std = 1 / dim if self.is_first else math.sqrt(c / dim) / w0
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        """
        Forward pass of the function.
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class TripletAttention(nn.Module):

    def __init__(self, no_spatial=False, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = torch_utils.AttentionGate(kernel_size)
        self.hc = torch_utils.AttentionGate(kernel_size)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = torch_utils.AttentionGate(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class SpatialGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class CBAM(nn.Module):

    def __init__(self, gate_channels, kernel_size=3, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, bam=False, num_layers=1, bn=False, dilation_conv_num=2, dilation_val=4):
        super(CBAM, self).__init__()
        self.bam = bam
        self.no_spatial = no_spatial
        if self.bam:
            self.dilatedGate = SpatialGate(gate_channels, reduction_ratio, dilation_conv_num, dilation_val)
            self.ChannelGate = torch_utils.ChannelGate(gate_channels, reduction_ratio, pool_types, bam=self.bam, num_layers=num_layers, bn=bn)
        else:
            self.ChannelGate = torch_utils.ChannelGate(gate_channels, reduction_ratio, pool_types)
            if not no_spatial:
                self.SpatialGate = torch_utils.AttentionGate(kernel_size)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.bam:
            if not self.no_spatial:
                x_out = self.SpatialGate(x_out)
            return x_out
        else:
            att = 1 + F.sigmoid(self.ChannelGate(x) * self.dilatedGate(x))
            return att * x


class SE(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16):
        super(SE, self).__init__()
        self.ChannelGate = torch_utils.ChannelGate(gate_channels, reduction_ratio, ['avg'])

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out


class ECA(nn.Module):

    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs(math.log2(self.channels) / self.gamma + self.b / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class AttentionGate(nn.Module):

    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.conv_bn = nn.Sequential(nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm2d(1, eps=1e-05, momentum=0.01, affine=True))

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.conv_bn(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, pool_types=['avg', 'max'], bam=False, num_layers=1, bn=False):
        super(ChannelGate, self).__init__()
        self.bam = bam
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            if bn is True:
                self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.gate_c(lp_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        if self.bam:
            return channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
            return x * scale


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (APL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Aria2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttentionGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DICE,
     lambda: ([], {'emb_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ECA,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FReLU,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Funnel,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Maxout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProbAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SE,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SLAF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SQNL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SReLU,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Seagull,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Siren,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Snake,
     lambda: ([], {'in_features': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftClipping,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftExponential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TanhSoft,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TripletAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (XUnit,
     lambda: ([], {'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_digantamisra98_Echo(_paritybench_base):
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

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

