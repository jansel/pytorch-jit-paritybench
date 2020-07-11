import sys
_module = sys.modules[__name__]
del sys
CrossReplicaBN = _module
data_loader = _module
debug = _module
demo = _module
main = _module
model_resnet = _module
parameter = _module
spectral = _module
trainer = _module
utils = _module

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


from torch import nn


from torch.nn.parameter import Parameter


from torch.nn import functional as F


import torchvision.datasets as dsets


from torchvision import transforms


from torch.backends import cudnn


from torch.nn import init


import functools


from torch.autograd import Variable


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import torch.nn.functional as F


from torch import Tensor


from torch.nn import Parameter


import time


import torch.nn as nn


from torchvision.utils import save_image


class _BatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
        super(_BatchNorm, self)._load_from_state_dict(state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ScaledCrossReplicaBatchNorm2d(_BatchNorm):
    """Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size).

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momemtum} \\times x_t`,
        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        init_conv(self.query_conv)
        init_conv(self.key_conv)
        init_conv(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class ConditionalNorm(nn.Module):

    def __init__(self, in_channel, n_condition=148):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.embed = nn.Linear(n_condition, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, class_id):
        out = self.bn(input)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=[3, 3], padding=1, stride=1, n_class=None, bn=True, activation=F.relu, upsample=True, downsample=False):
        super().__init__()
        gain = 2 ** 0.5
        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True))
        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True
        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input
        if self.bn:
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip


class Generator(nn.Module):

    def __init__(self, code_dim=100, n_class=1000, chn=96, debug=False):
        super().__init__()
        self.linear = SpectralNorm(nn.Linear(n_class, 128, bias=False))
        if debug:
            chn = 8
        self.first_view = 16 * chn
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn))
        self.conv = nn.ModuleList([GBlock(16 * chn, 16 * chn, n_class=n_class), GBlock(16 * chn, 8 * chn, n_class=n_class), GBlock(8 * chn, 4 * chn, n_class=n_class), GBlock(4 * chn, 2 * chn, n_class=n_class), SelfAttention(2 * chn), GBlock(2 * chn, 1 * chn, n_class=n_class)])
        self.ScaledCrossReplicaBN = ScaledCrossReplicaBatchNorm2d(1 * chn)
        self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1))

    def forward(self, input, class_id):
        codes = torch.split(input, 20, 1)
        class_emb = self.linear(class_id)
        out = self.G_linear(codes[0])
        out = out.view(-1, self.first_view, 4, 4)
        ids = 1
        for i, conv in enumerate(self.conv):
            if isinstance(conv, GBlock):
                conv_code = codes[ids]
                ids = ids + 1
                condition = torch.cat([conv_code, class_emb], 1)
                out = conv(out, condition)
            else:
                out = conv(out)
        out = self.ScaledCrossReplicaBN(out)
        out = F.relu(out)
        out = self.colorize(out)
        return F.tanh(out)


class Spectral_Norm:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = Spectral_Norm(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    Spectral_Norm.apply(module, name)
    return module


class Discriminator(nn.Module):

    def __init__(self, n_class=1000, chn=96, debug=False):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GBlock(in_channel, out_channel, bn=False, upsample=False, downsample=downsample)
        gain = 2 ** 0.5
        if debug:
            chn = 8
        self.debug = debug
        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 1 * chn, 3, padding=1)), nn.ReLU(), SpectralNorm(nn.Conv2d(1 * chn, 1 * chn, 3, padding=1)), nn.AvgPool2d(2))
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 1 * chn, 1))
        self.conv = nn.Sequential(conv(1 * chn, 1 * chn, downsample=True), SelfAttention(1 * chn), conv(1 * chn, 2 * chn, downsample=True), conv(2 * chn, 4 * chn, downsample=True), conv(4 * chn, 8 * chn, downsample=True), conv(8 * chn, 16 * chn, downsample=True), conv(16 * chn, 16 * chn, downsample=False))
        self.linear = SpectralNorm(nn.Linear(16 * chn, 1))
        self.embed = nn.Embedding(n_class, 16 * chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, input, class_id):
        out = self.pre_conv(input)
        out = out + self.pre_skip(F.avg_pool2d(input, 2))
        out = self.conv(out)
        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze(1)
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)
        return out_linear + prod


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ScaledCrossReplicaBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'in_dim': 18}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
]

class Test_sxhxliang_BigGAN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

