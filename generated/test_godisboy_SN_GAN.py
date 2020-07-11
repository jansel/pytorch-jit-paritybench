import sys
_module = sys.modules[__name__]
del sys
models = _module
models = _module
snres_discriminator = _module
snres_generator = _module
src = _module
functions = _module
max_sv = _module
snlayers = _module
snconv2d = _module
snlinear = _module
test = _module
train = _module

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


import torch.nn as nn


import torch


from torch.nn.modules import conv


from torch.nn.modules import Linear


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import torchvision.utils as vutils


from torch.autograd import Variable


import torch.utils.data


from torch.nn.modules.utils import _triple


import torch.backends.cudnn as cudnn


import random


import numpy as np


import matplotlib.pyplot as plt


class _netG(nn.Module):

    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.convT1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(10, ngf * 4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.main = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input, input_c):
        out1 = self.convT1(input)
        out2 = self.convT2(input_c)
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        return output


def _l2normalize(v, eps=1e-12):
    return v / ((v ** 2).sum() ** 0.5 + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)), torch.transpose(_u, 0, 1))
    return sigma, _v


class SNConv2d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _ = max_singular_value(w_mat)
        self.weight.data = self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _netD(nn.Module):

    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.conv1_1 = SNConv2d(nc, ndf / 2, 3, 1, 1, bias=False)
        self.conv1_2 = SNConv2d(10, ndf / 2, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.main = nn.Sequential(SNConv2d(ndf, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, input, input_c):
        out1 = self.lrelu(self.conv1_1(input))
        out2 = self.lrelu(self.conv1_2(input_c))
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        return output.view(-1, 1).squeeze(1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
        super(ResBlock, self).__init__()
        hidden_channels = in_channels
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

    def forward_residual_connect(self, input):
        out = self.conv_sc(input)
        if self.upsample:
            out = self.upsampling(out)
        return out

    def forward(self, input):
        out = self.relu(self.bn1(input))
        out = self.conv1(out)
        if self.upsample:
            out = self.upsampling(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out_res = self.forward_residual_connect(input)
        return out + out_res


class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)

    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)


class SNLinear(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`

           W(Tensor): Spectrally normalized weight

           u (Tensor): the right largest singular value of W.
       """

    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)


class SNResDiscriminator(nn.Module):

    def __init__(self, ndf=64, ndlayers=4):
        super(SNResDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf * 16, 1), nn.Sigmoid())

    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(3, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf * 2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)

    def forward(self, input):
        out = self.res_d(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)


class SNResGenerator(nn.Module):

    def __init__(self, ngf, z=128, nlayers=4):
        super(SNResGenerator, self).__init__()
        self.input_layer = nn.Linear(z, 4 ** 2 * ngf * 16)
        self.generator = self.make_model(ngf, nlayers)

    def make_model(self, ngf, nlayers):
        model = []
        tngf = ngf * 16
        for i in range(nlayers):
            model += [ResBlock(tngf, tngf / 2, upsample=True)]
            tngf /= 2
        model += [nn.BatchNorm2d(ngf)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def forward(self, z):
        out = self.input_layer(z)
        out = out.view(z.size(0), -1, 4, 4)
        out = self.generator(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SNLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_netG,
     lambda: ([], {'nz': 4, 'nc': 4, 'ngf': 4}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 10, 64, 64])], {}),
     True),
]

class Test_godisboy_SN_GAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

