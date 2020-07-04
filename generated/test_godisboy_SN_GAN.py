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


class _netG(nn.Module):

    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0,
            bias=True), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.
            ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True), nn.
            BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4,
            ngf * 2, 4, 2, 1, bias=True), nn.BatchNorm2d(ngf * 2), nn.ReLU(
            True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True), nn
            .BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 3,
            1, 1, bias=True), nn.Tanh())

    def forward(self, input):
        output = self.main(input)
        return output


class _netD(nn.Module):

    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.main = nn.Sequential(SNConv2d(nc, ndf, 3, 1, 1, bias=True), nn
            .LeakyReLU(0.1, inplace=True), SNConv2d(ndf, ndf, 4, 2, 1, bias
            =True), nn.LeakyReLU(0.1, inplace=True), SNConv2d(ndf, ndf * 2,
            3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True), SNConv2d(
            ndf * 2, ndf * 2, 4, 2, 1, bias=True), nn.LeakyReLU(0.1,
            inplace=True), SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True), SNConv2d(ndf * 4, ndf * 4, 4, 
            2, 1, bias=True), nn.LeakyReLU(0.1, inplace=True), SNConv2d(ndf *
            4, ndf * 8, 3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True
            ), SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
        use_BN=False, downsample=False):
        super(ResBlock, self).__init__()
        hidden_channels = in_channels
        self.downsample = downsample
        self.resblock = self.make_res_block(in_channels, out_channels,
            hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels,
            out_channels)

    def make_res_block(self, in_channels, out_channels, hidden_channels,
        use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]
        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3,
            padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3,
            padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)
            ]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)


class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels,
            out_channels)

    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3,
            padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)
            ]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)


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


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
        upsample=False):
        super(ResBlock, self).__init__()
        hidden_channels = in_channels
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3,
            padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3,
            padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)
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


def _l2normalize(v, eps=1e-12):
    return v / ((v ** 2).sum() ** 0.5 + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)),
            eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)),
        torch.transpose(_u, 0, 1))
    return sigma, _v


class SNConv2d(conv._ConvNd):
    """Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \\begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \\sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \\star input(N_i, k)
        \\end{array}

    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    | :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    | :attr:`padding` controls the amount of implicit zero-paddings on both
    |  sides for :attr:`padding` number of points for each dimension.
    | :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_channels` and `out_channels` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_channels`, each input channel is convolved with its
                 own set of filters (of size `out_channels // in_channels`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels = K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(in\\_channels=C_{in}, out\\_channels=C_{in} * K, ..., groups=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\\_size[1] - 1) - 1) / stride[1] + 1)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

        W(Tensor): Spectrally normalized weight

        u (Tensor): the right largest singular value of W.

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, False, _pair(0), groups,
            bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


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


class SNConv2d(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, False, _pair(0), groups,
            bias)

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _ = max_singular_value(w_mat)
        self.weight.data = self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


class _netG(nn.Module):

    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(nz + 200, ngf * 8, 4, 
            1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.
            ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.
            BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4,
            ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU
            (True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc,
            4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        output = self.main(input)
        return output


class _netG(nn.Module):

    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.convT1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0,
            bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(10, ngf * 4, 4, 1, 0,
            bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.main = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2,
            1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.
            ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.
            BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2,
            ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input, input_c):
        out1 = self.convT1(input)
        out2 = self.convT2(input_c)
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        return output


class _netD(nn.Module):

    def __init__(self, nc, ndf):
        super(_netD, self).__init__()
        self.conv1_1 = SNConv2d(nc, ndf / 2, 3, 1, 1, bias=False)
        self.conv1_2 = SNConv2d(10, ndf / 2, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.main = nn.Sequential(SNConv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf, ndf * 2, 3, 1, 1,
            bias=False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 2,
            ndf * 2, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False), nn.LeakyReLU(
            0.2, inplace=True), SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), SNConv2d(ndf * 4, ndf *
            8, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), nn.LeakyReLU(
            0.2, inplace=True), SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, input, input_c):
        out1 = self.lrelu(self.conv1_1(input))
        out2 = self.lrelu(self.conv1_2(input_c))
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        return output.view(-1, 1).squeeze(1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_godisboy_SN_GAN(_paritybench_base):
    pass
    def test_000(self):
        self._check(ResBlock(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(SNLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(_netG(*[], **{'nz': 4, 'nc': 4, 'ngf': 4}), [torch.rand([4, 4, 64, 64]), torch.rand([4, 10, 64, 64])], {})

