import sys
_module = sys.modules[__name__]
del sys
correlation_test = _module
dataset_test = _module
googledrive_test = _module
image_test = _module
imfilter_test = _module
initializer_test = _module
loader_test = _module
model_test = _module
motion_test = _module
space_to_depth_test = _module
training_test = _module
utility_test = _module
vgg_test = _module
virtualfile_test = _module
CelebA = _module
DND = _module
FFmpegHelper = _module
FastMetrics = _module
Image2Raw = _module
MakeHDF = _module
Misc = _module
NtireHelper = _module
Raw2Image = _module
SeqVisual = _module
Vimeo = _module
YoukuPackage = _module
check_dataset = _module
eval = _module
train = _module
Environment = _module
Trainer = _module
Framework = _module
Model = _module
Srcnn = _module
Models = _module
Keras = _module
Dense = _module
Discriminator = _module
Residual = _module
Arch = _module
GAN = _module
LayersHelper = _module
Motion = _module
Noise = _module
SuperResolution = _module
Carn = _module
Crdn = _module
Dbpn = _module
Dcscn = _module
DnCnn = _module
Drcn = _module
Drrn = _module
Drsr = _module
Drsr_v2 = _module
Duf = _module
Edsr = _module
Espcn = _module
FFDNet = _module
Gan = _module
Idn = _module
LapSrn = _module
MemNet = _module
Msrn = _module
Nlrn = _module
Rcan = _module
Rdn = _module
SRDenseNet = _module
SRFeat = _module
SrGan = _module
Vdsr = _module
Vespcn = _module
Util = _module
TF = _module
Environment = _module
Summary = _module
Trainer = _module
Bicubic = _module
Carn = _module
Classic = _module
Contrib = _module
ntire19 = _module
denoise = _module
edrn = _module
frn = _module
ran2 = _module
ntire20 = _module
xiaozhong = _module
ops = _module
discriminator = _module
loss = _module
network = _module
Crdn = _module
Dbpn = _module
Drn = _module
Edsr = _module
Esrgan = _module
Ffdnet = _module
Frvsr = _module
Mldn = _module
Model = _module
Msrn = _module
NTIRE19 = _module
NTIRE20 = _module
Blocks = _module
Discriminator = _module
Distortion = _module
Initializer = _module
Loss = _module
Motion = _module
Scale = _module
Ops = _module
SISR = _module
Qprn = _module
Rbpn = _module
Rcan = _module
SRFeat = _module
Sofvsr = _module
Spmc = _module
Srmd = _module
TecoGAN = _module
Vespcn = _module
Metrics = _module
Utility = _module
Torch = _module
Backend = _module
Crop = _module
Dataset = _module
FloDecoder = _module
Loader = _module
NVDecoder = _module
Transform = _module
VirtualFile = _module
YVDecoder = _module
DataLoader = _module
Config = _module
Ensemble = _module
GoogleDriveDownloader = _module
Hook = _module
ImageProcess = _module
LearningRateScheduler = _module
Math = _module
PcaPrecompute = _module
VisualizeOpticalFlow = _module
VSR = _module
prepare_data = _module
setup = _module

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


import tensorflow as tf


import torch


import logging


import torch.nn as nn


import torchvision as tv


import torch.nn.functional as F


import copy


import torchvision


import random


from torch import nn


from collections import OrderedDict


import math


from torch.autograd import Variable


from torch.nn import Parameter


from torch.nn import functional as F


from torch.nn.modules.utils import _pair


class Cubic(nn.Module):

    def __init__(self, scale):
        super(Cubic, self).__init__()
        self.to_pil = tv.transforms.ToPILImage()
        self.to_tensor = tv.transforms.ToTensor()
        self.scale = scale

    def forward(self, x):
        if self.scale == 1:
            return x
        ret = []
        for img in [i[0] for i in x.split(1, dim=0)]:
            img = self.to_pil(img.cpu())
            w = img.width
            h = img.height
            img = img.resize([w * self.scale, h * self.scale], 3)
            img = self.to_tensor(img)
            ret.append(img)
        return torch.stack(ret)


class EResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, group):
        super(EResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        out = self.body(x)
        return out + x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class Activation(nn.Module):

    def __init__(self, act, **kwargs):
        super(Activation, self).__init__()
        if act is None:
            self.f = lambda t: t
        if isinstance(act, str):
            self.name = act.lower()
            in_place = kwargs.get('in_place', True)
            if self.name == 'relu':
                self.f = nn.ReLU(in_place)
            elif self.name == 'prelu':
                self.f = nn.PReLU(num_parameters=kwargs.get('num_parameters', 1), init=kwargs.get('init', 0.25))
            elif self.name in ('lrelu', 'leaky', 'leakyrelu'):
                self.f = nn.LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01), inplace=in_place)
            elif self.name == 'tanh':
                self.f = nn.Tanh()
            elif self.name == 'sigmoid':
                self.f = nn.Sigmoid()
        elif callable(act):
            self.f = act

    def forward(self, x):
        return self.f(x)


class EasyConv2d(nn.Module):
    """ Convolution maker, to construct commonly used conv block with default
  configurations.

  Support to build Conv2D, ConvTransposed2D, along with selectable normalization
  and activations.
  Support normalization:
  - Batchnorm2D
  - Spectralnorm2D
  Support activation:
  - Relu
  - PRelu
  - LeakyRelu
  - Tanh
  - Sigmoid
  - Customized callable functions

  Args:
      in_channels (int): Number of channels in the input image
      out_channels (int): Number of channels produced by the convolution
      kernel_size (int or tuple): Size of the convolving kernel
      stride (int or tuple, optional): Stride of the convolution. Default: 1
      padding (str, optional): 'same' means $out_size=in_size // stride$ or
                                $out_size=in_size * stride$ (ConvTransposed);
                                'valid' means padding zero.
      dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
      groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
      use_bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
      use_bn (bool, optional): If ``True``, adds Batchnorm2D module to the output.
      use_sn (bool, optional): If ``True``, adds Spectralnorm2D module to the output.
      transposed (bool, optional): If ``True``, use ConvTransposed instead of Conv2D.
  """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, activation=None, use_bias=True, use_bn=False, use_sn=False, transposed=False, **kwargs):
        super(EasyConv2d, self).__init__()
        padding = padding.lower()
        assert padding in ('same', 'valid')
        if transposed:
            assert padding == 'same'
            q = kernel_size % 2
            p = (kernel_size + q - stride) // 2
            net = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, p, q, groups, use_bias, dilation)]
        else:
            if padding == 'same':
                padding_ = (dilation * (kernel_size - 1) - stride + 2) // 2
            else:
                padding_ = 0
            net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_, dilation, groups, use_bias)]
        if use_sn:
            net[0] = nn.utils.spectral_norm(net[0])
        if use_bn:
            net += [nn.BatchNorm2d(out_channels, eps=kwargs.get('eps', 1e-05), momentum=kwargs.get('momentum', 0.1), affine=kwargs.get('affine', True), track_running_stats=kwargs.get('track_running_stats', True))]
        if activation:
            net += [Activation(activation, in_place=True, **kwargs)]
        self.body = nn.Sequential(*net)

    def forward(self, x):
        return self.body(x)

    def initialize_(self, kernel, bias=None):
        """initialize the convolutional weights from external sources

    Args:
        kernel: kernel weight. Shape=[OUT, IN, K, K]
        bias: bias weight. Shape=[OUT]
    """
        dtype = self.body[0].weight.dtype
        device = self.body[0].weight.device
        kernel = torch.tensor(kernel, dtype=dtype, device=device, requires_grad=True)
        assert kernel.shape == self.body[0].weight.shape, 'Wrong kernel shape!'
        if bias is not None:
            bias = torch.tensor(bias, dtype=dtype, device=device, requires_grad=True)
            assert bias.shape == self.body[0].bias.shape, 'Wrong bias shape!'
        self.body[0].weight.data.copy_(kernel)
        self.body[0].bias.data.copy_(bias)


class RB(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=3, activation=None, use_bias=True, use_bn=False, use_sn=False, act_first=None):
        super(RB, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=use_bias)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=use_bias)
        if use_sn:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
        net = [conv1, Activation(activation, in_place=True), conv2]
        if use_bn:
            net.insert(1, nn.BatchNorm2d(out_channels))
            if act_first:
                net = [nn.BatchNorm2d(in_channels), Activation(activation, in_place=True)] + net
            else:
                net.append(nn.BatchNorm2d(out_channels))
        self.body = nn.Sequential(*net)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.body(x)
        if hasattr(self, 'shortcut'):
            sc = self.shortcut(x)
            return out + sc
        return out + x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, group=1):
        """ CARN cascading residual block
    """
        super(Block, self).__init__()
        if group == 1:
            self.b1 = RB(in_channels, out_channels, activation='relu')
            self.b2 = RB(out_channels, out_channels, activation='relu')
            self.b3 = RB(out_channels, out_channels, activation='relu')
        elif group > 1:
            self.b1 = EResidualBlock(64, 64, group=group)
            self.b2 = self.b3 = self.b1
        self.c1 = EasyConv2d(in_channels + out_channels, out_channels, 1, activation='relu')
        self.c2 = EasyConv2d(in_channels + out_channels * 2, out_channels, 1, activation='relu')
        self.c3 = EasyConv2d(in_channels + out_channels * 3, out_channels, 1, activation='relu')

    def forward(self, x):
        c0 = o0 = x
        b1 = F.relu(self.b1(o0))
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = F.relu(self.b2(o1))
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        b3 = F.relu(self.b3(o2))
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3


class Net(nn.Module):
    """
  SRMD CNN network. 12 conv layers
  """

    def __init__(self, scale=4, channels=3, layers=12, filters=128, pca_length=15):
        super(Net, self).__init__()
        self.pca_length = pca_length
        net = [EasyConv2d(channels + pca_length + 1, filters, 3, activation='relu')]
        net += [EasyConv2d(filters, filters, 3, activation='relu') for _ in range(layers - 2)]
        net += [EasyConv2d(filters, channels * scale ** 2, 3), nn.PixelShuffle(scale)]
        self.body = nn.Sequential(*net)

    def forward(self, x, kernel=None, noise=None):
        if kernel is None and noise is None:
            kernel = torch.zeros(x.shape[0], 15, 1, device=x.device, dtype=x.dtype)
            noise = torch.zeros(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        degpar = torch.cat([kernel, noise.reshape([-1, 1, 1])], dim=1)
        degpar = degpar.reshape([-1, 1 + self.pca_length, 1, 1])
        degpar = torch.ones_like(x)[:, 0:1] * degpar
        _x = torch.cat([x, degpar], dim=1)
        return self.body(_x)


class Espcn(nn.Module):

    def __init__(self, channel, scale):
        super(Espcn, self).__init__()
        conv1 = nn.Conv2d(channel, 64, 5, 1, 2)
        conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        conv3 = nn.Conv2d(32, channel * scale * scale, 3, 1, 1)
        ps = nn.PixelShuffle(scale)
        self.body = nn.Sequential(conv1, nn.Tanh(), conv2, nn.Tanh(), conv3, nn.Tanh(), ps)

    def forward(self, x):
        return self.body(x)


class Srcnn(nn.Module):

    def __init__(self, channel, filters=(9, 5, 5)):
        super(Srcnn, self).__init__()
        self.net = nn.Sequential(EasyConv2d(channel, 64, filters[0], activation='relu'), EasyConv2d(64, 32, filters[1], activation='relu'), EasyConv2d(32, channel, filters[2], activation=None))

    def forward(self, x):
        return self.net(x)


class Vdsr(nn.Module):

    def __init__(self, channel, layers=20):
        super(Vdsr, self).__init__()
        net = [EasyConv2d(channel, 64, 3, activation='relu')]
        for i in range(1, layers - 1):
            net.append(EasyConv2d(64, 64, 3, activation='relu'))
        net.append(EasyConv2d(64, channel, 3))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x) + x


class DnCnn(nn.Module):

    def __init__(self, channel, layers, bn):
        super(DnCnn, self).__init__()
        net = [EasyConv2d(channel, 64, 3, activation='relu', use_bn=bn)]
        for i in range(1, layers - 1):
            net.append(EasyConv2d(64, 64, 3, activation='relu', use_bn=bn))
        net.append(EasyConv2d(64, channel, 3))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x) + x


def _pop_shape(x, shape):
    if shape == 2:
        return x[0, ..., 0]
    elif shape == 3:
        return x[0]
    elif shape == 4:
        return x
    else:
        raise ValueError('Unsupported shape! Must be 2/3/4')


def _push_shape_4d(x):
    dim = x.dim()
    if dim == 2:
        return x.unsqueeze(0).unsqueeze(1), 2
    elif dim == 3:
        return x.unsqueeze(0), 3
    elif dim == 4:
        return x, 4
    else:
        raise ValueError('Unsupported tensor! Must be 2D/3D/4D')


def bicubic_filter(x, a=-0.5):
    if x < 0:
        x = -x
    if x < 1:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0


def list_rshift(l, s):
    for _ in range(s):
        l.insert(0, l.pop(-1))
    return l


def weights_upsample(scale_factor):
    if scale_factor < 1:
        ss = int(1 / scale_factor + 0.5)
    else:
        ss = int(scale_factor + 0.5)
    support = 2
    ksize = support * 2 + 1
    weights = [[] for _ in range(ss)]
    for i in range(ss):
        for lambd in range(ksize):
            dist = int((1 + ss + 2 * i) / 2 / ss) + lambd - 1.5 - (2 * i + 1) / 2 / ss
            weights[i].append(bicubic_filter(dist))
    w = [(np.array([i]) / np.sum(i)) for i in weights]
    w = list_rshift(w, ss - ss // 2)
    kernels = []
    for i in range(len(w)):
        for j in range(len(w)):
            kernels.append(np.matmul(w[i].transpose(), w[j]))
    return kernels, ss


def upsample(img, scale, border='reflect'):
    """Bicubical upsample via **CONV2D**. Using PIL's kernel.

  Args:
    img: a tf tensor of 2/3/4-D.
    scale: must be integer >= 2.
    border: padding mode. Recommend to 'REFLECT'.
  """
    device = img.device
    kernels, s = weights_upsample(scale)
    if s == 1:
        return img
    kernels = [k.astype('float32') for k in kernels]
    kernels = [torch.from_numpy(k) for k in kernels]
    p1 = 1 + s // 2
    p2 = 3
    img, shape = _push_shape_4d(img)
    img_ex = F.pad(img, [p1, p2, p1, p2], mode=border)
    c = img_ex.shape[1]
    assert c is not None, 'img must define channel number'
    c = int(c)
    filters = [(torch.reshape(torch.eye(c, c), [c, c, 1, 1]) * k) for k in kernels]
    weights = torch.stack(filters, dim=0).transpose(0, 1).reshape([-1, c, 5, 5])
    img_s = F.conv2d(img_ex, weights)
    img_s = F.pixel_shuffle(img_s, s)
    more = s // 2 * s
    crop = slice(more - s // 2, -(s // 2))
    img_s = _pop_shape(img_s[..., crop, crop], shape)
    return img_s


class Drcn(nn.Module):

    def __init__(self, scale, channel, n_recur, filters):
        from torch.nn import Parameter
        super(Drcn, self).__init__()
        self.entry = nn.Sequential(EasyConv2d(channel, filters, 3, activation='relu'), EasyConv2d(filters, filters, 3, activation='relu'))
        self.exit = nn.Sequential(EasyConv2d(filters, filters, 3, activation='relu'), EasyConv2d(filters, channel, 3))
        self.conv = EasyConv2d(filters, filters, 3, activation='relu')
        self.output_weights = Parameter(torch.empty(n_recur + 1))
        torch.nn.init.uniform_(self.output_weights, 0, 1)
        self.n_recur = n_recur
        self.scale = scale

    def forward(self, x):
        bic = upsample(x, self.scale)
        y = [self.entry(bic)]
        for i in range(self.n_recur):
            y.append(self.conv(y[-1]))
        sr = [self.exit(i) for i in y[1:]]
        final = bic * self.output_weights[0]
        for i in range(len(sr)):
            final = final + self.output_weights[i + 1] * sr[i]
        return final


class Drrn(nn.Module):

    def __init__(self, channel, n_ru, n_rb, filters):
        super(Drrn, self).__init__()
        self.entry0 = EasyConv2d(channel, filters, 3, activation='relu')
        for i in range(1, n_rb):
            setattr(self, f'entry{i}', EasyConv2d(filters, filters, 3, activation='relu'))
        self.n_rb = n_rb
        self.rb = RB(filters, kernel_size=3, activation='relu')
        self.n_ru = n_ru
        self.exit = EasyConv2d(filters, channel, 3)

    def forward(self, x):
        for i in range(self.n_rb):
            entry = getattr(self, f'entry{i}')
            y = entry(x)
            for j in range(self.n_ru):
                y = self.rb(y)
            x = y
        return self.exit(x)


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()
        self.ch = channel

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        y = self.sigmoid(y)
        return x * y


class MeanShift(nn.Conv2d):

    def __init__(self, mean_rgb, sub, rgb_range=1.0):
        super(MeanShift, self).__init__(3, 3, 1)
        sign = -1 if sub else 1
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(mean_rgb) * sign * rgb_range
        for params in self.parameters():
            params.requires_grad = False


class Residual_Block(nn.Module):

    def __init__(self, inChannels, growRate0, wn, kSize=3, stride=1):
        super(Residual_Block, self).__init__()
        Cin = inChannels
        G0 = growRate0
        self.conv = nn.Sequential(*[wn(nn.Conv2d(Cin, G0, kSize, padding=(kSize - 1) // 2, stride=stride)), nn.ReLU(inplace=True), wn(nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=stride)), CALayer(Cin, wn, 16)])

    def forward(self, x):
        out = self.conv(x)
        out += x
        return out


class RG(nn.Module):

    def __init__(self, growRate0, nConvLayers, wn, kSize=3):
        super(RG, self).__init__()
        G0 = growRate0
        C = nConvLayers
        convs_residual = []
        for c in range(C):
            convs_residual.append(Residual_Block(G0, G0, wn))
        self.convs_residual = nn.Sequential(*convs_residual)
        self.last_conv = wn(nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1))

    def forward(self, x):
        x = self.last_conv(self.convs_residual(x)) + x
        return x


_logger = logging.getLogger('VSR.VESPCN')


class EDRN(nn.Module):

    def __init__(self, args):
        _logger.info('LICENSE: EDRN is implemented by IVIP-Lab. @yyknight https://github.com/yyknight/NTIRE2019_EDRN')
        super(EDRN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.EDRNkSize
        rgb_mean = 0.4313, 0.4162, 0.3861
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.D, C, G = {'B': (4, 10, 16)}[args.EDRNconfig]
        self.SFENet1 = wn(nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1))
        self.encoder1 = nn.Sequential(*[wn(nn.Conv2d(G0, 2 * G0, kSize, padding=(kSize - 1) // 2, stride=2)), nn.BatchNorm2d(2 * G0), nn.ReLU(inplace=True)])
        self.encoder2 = nn.Sequential(*[wn(nn.Conv2d(2 * G0, 4 * G0, kSize, padding=(kSize - 1) // 2, stride=2)), nn.BatchNorm2d(4 * G0), nn.ReLU(inplace=True)])
        self.decoder1 = nn.Sequential(*[wn(nn.ConvTranspose2d(4 * G0, 2 * G0, 3, padding=1, output_padding=1, stride=2)), nn.BatchNorm2d(2 * G0), nn.ReLU(inplace=True)])
        self.decoder2 = nn.Sequential(*[wn(nn.ConvTranspose2d(2 * G0, G0, 3, padding=1, output_padding=1, stride=2)), nn.BatchNorm2d(G0), nn.ReLU()])
        RGs0 = [RG(growRate0=4 * G0, nConvLayers=C, wn=wn) for _ in range(self.D)]
        RGs0.append(wn(nn.Conv2d(4 * G0, 4 * G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        RGs1 = [RG(growRate0=2 * G0, nConvLayers=C, wn=wn) for _ in range(self.D // 2)]
        RGs1.append(wn(nn.Conv2d(2 * G0, 2 * G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        RGs2 = [RG(growRate0=G0, nConvLayers=C, wn=wn) for _ in range(self.D // 4)]
        RGs2.append(wn(nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)))
        self.RGs0 = nn.Sequential(*RGs0)
        self.RGs1 = nn.Sequential(*RGs1)
        self.RGs2 = nn.Sequential(*RGs2)
        self.restoration = wn(nn.Conv2d(G0, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1))
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        f__2 = self.encoder1(f__1)
        f__3 = self.encoder2(f__2)
        x = f__3
        x = self.decoder1(self.RGs0(x) + f__3)
        x = self.decoder2(self.RGs1(x) + f__2)
        x = self.RGs2(x) + f__1
        x = self.restoration(x)
        x = self.add_mean(x)
        return x


class RCAB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.LeakyReLU(0.2, True), res_scale=1, px=1):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        if px != 1:
            modules_body.append(common.invPixelShuffle(px))
        modules_body.append(CALayer(n_feat * px ** 2, reduction))
        if px != 1:
            modules_body.append(nn.PixelShuffle(px))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class _UpsampleLinear(nn.Module):

    def __init__(self, scale):
        super(_UpsampleLinear, self).__init__()
        self._mode = 'linear', 'bilinear', 'trilinear'
        self.scale = scale

    def forward(self, x, scale=None):
        scale = scale or self.scale
        mode = self._mode[x.dim() - 3]
        return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)


class _UpsampleNearest(nn.Module):

    def __init__(self, scale):
        super(_UpsampleNearest, self).__init__()
        self.scale = scale

    def forward(self, x, scale=None):
        scale = scale or self.scale
        return F.interpolate(x, scale_factor=scale)


class Upsample(nn.Module):

    def __init__(self, channel, scale, method='ps', name='Upsample', **kwargs):
        super(Upsample, self).__init__()
        self.name = name
        self.channel = channel
        self.scale = scale
        self.method = method.lower()
        self.group = kwargs.get('group', 1)
        self.kernel_size = kwargs.get('kernel_size', 3)
        _allowed_methods = 'ps', 'nearest', 'deconv', 'linear'
        assert self.method in _allowed_methods
        act = kwargs.get('activation')
        samplers = []
        while scale > 1:
            if scale % 2 == 1 or scale == 2:
                samplers.append(self.upsampler(self.method, scale, act))
                break
            else:
                samplers.append(self.upsampler(self.method, 2, act))
                scale //= 2
        self.body = nn.Sequential(*samplers)

    def upsampler(self, method, scale, activation=None):
        body = []
        k = self.kernel_size
        if method == 'ps':
            p = k // 2
            s = 1
            body = [nn.Conv2d(self.channel, self.channel * scale * scale, k, s, p, groups=self.group), nn.PixelShuffle(scale)]
            if activation:
                body.insert(1, Activation(activation))
        if method == 'deconv':
            q = k % 2
            p = (k + q) // 2 - 1
            s = scale
            body = [nn.ConvTranspose2d(self.channel, self.channel, k, s, p, q, groups=self.group)]
            if activation:
                body.insert(1, Activation(activation))
        if method == 'nearest':
            body = [_UpsampleNearest(scale), nn.Conv2d(self.channel, self.channel, k, 1, k // 2, groups=self.group)]
            if activation:
                body.append(Activation(activation))
        if method == 'linear':
            body = [_UpsampleLinear(scale), nn.Conv2d(self.channel, self.channel, k, 1, k // 2, groups=self.group)]
            if activation:
                body.append(Activation(activation))
        return nn.Sequential(*body)

    def forward(self, x, **kwargs):
        return self.body(x)

    def extra_repr(self):
        return f'{self.name}: scale={self.scale}'


class Generator(torch.nn.Module):
    """ Generator for SRFeat:
  Single Image Super-Resolution with Feature Discrimination (ECCV 2018)
  """

    def __init__(self, channel, scale, filters, num_rb):
        super(Generator, self).__init__()
        self.head = EasyConv2d(channel, filters, 9)
        for i in range(num_rb):
            setattr(self, f'rb_{i:02d}', RB(filters, 3, 'lrelu', use_bn=True))
            setattr(self, f'merge_{i:02d}', EasyConv2d(filters, filters, 1))
        self.tail = torch.nn.Sequential(Upsample(filters, scale), EasyConv2d(filters, channel, 3))
        self.num_rb = num_rb

    def forward(self, inputs):
        x = self.head(inputs)
        feat = []
        for i in range(self.num_rb):
            x = getattr(self, f'rb_{i:02d}')(x)
            feat.append(getattr(self, f'merge_{i:02d}')(x))
        x = self.tail(x + torch.stack(feat, dim=0).sum(0).squeeze(0))
        return x


def to_list(x, repeat=1):
    """convert x to list object

    Args:
       x: any object to convert
       repeat: if x is to make as [x], repeat `repeat` elements in the list
  """
    if isinstance(x, (Generator, tuple, set)):
        return list(x)
    elif isinstance(x, list):
        return x
    elif isinstance(x, dict):
        return list(x.values())
    elif x is not None:
        return [x] * repeat
    else:
        return []


class Rcab(nn.Module):

    def __init__(self, channels, ratio=16, name='RCAB', **kwargs):
        super(Rcab, self).__init__()
        self.name = name
        self.ratio = ratio
        in_c, out_c = to_list(channels, 2)
        ks = kwargs.get('kernel_size', 3)
        padding = kwargs.get('padding', ks // 2)
        group = kwargs.get('group', 1)
        bias = kwargs.get('bias', True)
        self.c1 = nn.Sequential(nn.Conv2d(in_c, out_c, ks, 1, padding, 1, group, bias), nn.ReLU(True))
        self.c2 = nn.Conv2d(out_c, out_c, ks, 1, padding, 1, group, bias)
        self.c3 = nn.Sequential(nn.Conv2d(out_c, out_c // ratio, 1, groups=group, bias=bias), nn.ReLU(True))
        self.c4 = nn.Sequential(nn.Conv2d(out_c // ratio, in_c, 1, groups=group, bias=bias), nn.Sigmoid())
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        x = self.c1(inputs)
        y = self.c2(x)
        x = self.pooling(y)
        x = self.c3(x)
        x = self.c4(x)
        y = x * y
        return inputs + y

    def extra_repr(self):
        return f'{self.name}: ratio={self.ratio}'


class ResidualGroup(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [Rcab(n_feat, reduction, kernel_size=kernel_size) for _ in range(n_resblocks)]
        modules_body.append(EasyConv2d(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class sub_pixel(nn.Module):

    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class Dilated_block(nn.Module):

    def __init__(self, inChannel):
        super(Dilated_block, self).__init__()
        self.conv11 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=2, dilation=2)
        self.conv21 = nn.Conv2d(inChannel, inChannel, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=3, dilation=3)
        self.conv31 = nn.Conv2d(inChannel, inChannel, kernel_size=7, padding=3)
        self.conv32 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=3, dilation=3)
        self.Fuse = nn.Conv2d(inChannel * 3, inChannel, kernel_size=1, padding=0)

    def forward(self, x):
        i = x
        x1 = self.conv12(F.leaky_relu(self.conv11(x), 0.1))
        x2 = self.conv22(F.leaky_relu(self.conv21(x), 0.1))
        x3 = self.conv32(F.leaky_relu(self.conv31(x), 0.1))
        x = self.Fuse(torch.cat([x1, x2, x3], dim=1))
        return x + i


class Global_att(nn.Module):

    def __init__(self, inChannel, num_unit, prev):
        super(Global_att, self).__init__()
        self.inChannel = inChannel
        self.num_unit = num_unit
        self.Fuse = nn.Conv2d(inChannel * num_unit, inChannel, 1)
        self.Trans = nn.Conv2d(inChannel, inChannel, 3, padding=1)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.cnt = 0
        nn.init.constant_(self.alpha, 0)

    def forward(self, x_ori, x, res_list, temp_list):
        x_ = torch.cat(res_list, dim=1)
        """
    x_ = x_.view(-1, self.num_unit, self.inChannel, *x.size()[2:])
    x_ = torch.transpose(x_,1,2).contiguous()
    x_ = x_.view(-1, self.num_unit*self.inChannel, *x.size()[2:])
    """
        mask = self.Trans(F.relu(self.Fuse(x_)))
        x = mask + x_ori
        return x


class Channel_att_score3(nn.Module):

    def __init__(self, inChannel, num_prev, r=8):
        super(Channel_att_score3, self).__init__()
        self.num_prev = num_prev
        self.fc1_1 = nn.Conv2d(inChannel, inChannel // r, kernel_size=1, bias=False)
        self.fc1_2 = nn.Conv2d(inChannel // r, inChannel, kernel_size=1, bias=False)
        if self.num_prev > 0:
            self.fc1_3 = nn.Conv2d(inChannel, inChannel // r, kernel_size=[num_prev + 1, 1], padding=0, bias=False)
            self.fc1_4 = nn.Conv2d(inChannel // r, inChannel, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.FloatTensor(1).zero_())
        self.beta = nn.Parameter(torch.FloatTensor(1).zero_())
        self.register_parameter('norm_alpha', self.alpha)
        self.register_parameter('norm_beta', self.beta)
        self.alpha1 = nn.Parameter(torch.FloatTensor(1).zero_())
        self.beta1 = nn.Parameter(torch.FloatTensor(1).zero_())
        self.register_parameter('norm_alpha1', self.alpha1)
        self.register_parameter('norm_beta1', self.beta1)
        nn.init.constant_(self.alpha, 1)
        nn.init.constant_(self.alpha1, 1)
        self.activ1 = nn.ReLU()
        self.activ2 = nn.Sigmoid()

    def forward(self, x, MP_list, GP_list):
        MP = F.max_pool2d(x, kernel_size=x.size()[2:])
        GP = F.avg_pool2d(x, kernel_size=x.size()[2:])
        MP_list_ = copy.copy(MP_list)
        GP_list_ = copy.copy(GP_list)
        GP_list_.append(GP)
        t2 = GP
        m = t2.mean(dim=1, keepdim=True)
        std = ((t2 - m) ** 2).mean(dim=1, keepdim=True)
        t2 = (t2 - m) / std.sqrt() * self.alpha + self.beta
        t2 = self.fc1_1(t2)
        t = self.fc1_2(self.activ1(t2))
        output = 1 + torch.tanh(t)
        if self.num_prev == 0:
            return output.squeeze(3).squeeze(2), output * x
        else:
            x2 = torch.cat(GP_list_, dim=2)
            m = x2.mean(dim=1, keepdim=True)
            std = ((x2 - m) ** 2).mean(dim=1, keepdim=True)
            x2 = (x2 - m) / std.sqrt() * self.alpha1 + self.beta1
            x2 = self.fc1_3(x2)
            x2 = self.fc1_4(self.activ1(x2))
            output = 1 + torch.tanh(x2)
            return output.squeeze(3).squeeze(2), output * x


class ST_Unit5(nn.Module):

    def __init__(self, nChannel, prev=0, bias=True, dilation=1, conv1=None, conv2=None):
        super(ST_Unit5, self).__init__()
        self.nChannel = nChannel
        self.prev = prev
        self.att_c = Channel_att_score3(nChannel, self.prev)
        """
    self.Conv = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1,bias=bias)
    self.Linear = nn.Conv2d(nChannel, nChannel*9, kernel_size=1, padding=0,bias=bias)
    self.Subpixel = nn.PixelShuffle(3)
    #self.att_s = Spatial_att_score2(prev)

    var = np.sqrt(6 / (64 * 3 * 3 + 64 * 3 * 3))
    self.filters = nn.Parameter(torch.FloatTensor(64,64,3,3).uniform_(-var,var))
    self.register_parameter('conv1x1',self.filters)
    """
        self.Conv_1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=dilation, dilation=dilation, bias=bias) if conv1 is None else conv1
        self.DWconv_3x3 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias) if conv2 is None else conv2
        """
    self.conv3x3_1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)
    self.conv3x3_2 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=2, dilation=2, bias=bias)
    self.conv1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=3, dilation=3, bias=bias)
    self.conv1x1_2 = nn.Conv2d(nChannel*4, nChannel, kernel_size=1, padding=0, bias=bias)
    """

    def forward(self, x, pre_res, MP_list, GP_list):
        ori = x[:, -self.nChannel:, :, :]
        scale = None
        """
    ka = F.avg_pool2d(self.Conv(x),kernel_size=x.size()[2:])
    ka = self.Linear(F.relu(ka))
    ka = self.Subpixel(ka)
    ka_ = ka.view(-1,self.nChannel,9,1)
    ka_ = F.softmax(ka_/8.,dim=2)*9
    ka = ka_.view(-1,self.nChannel,3,3)
    ka = torch.mean(ka,dim=0,keepdim=False).unsqueeze(0)

    x = F.conv2d(x,self.filters*ka,padding=1)
    """
        x1 = self.DWconv_3x3(F.relu(self.Conv_1x1(x)))
        i = x1
        scale, x1 = self.att_c(x1, MP_list, GP_list)
        return x1, x1 + ori, scale


class ST_Block5(nn.Module):

    def __init__(self, num_unit, nChannel, prev=0, bias=True):
        super(ST_Block5, self).__init__()
        self.nChannel = nChannel
        self.num_unit = num_unit
        self.Conv_1x1 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, dilation=1, bias=bias)
        self.DWconv_3x3 = nn.Conv2d(nChannel, nChannel, kernel_size=3, padding=1, bias=bias)
        self.prev = prev
        self.Spatial_att = Global_att(nChannel, num_unit, prev)
        self.unit = torch.nn.ModuleList([ST_Unit5(64, i, dilation=1, conv1=None, conv2=None) for i in range(num_unit)])
        self.Fuse = nn.Conv2d(nChannel * 2, nChannel, kernel_size=1, padding=0, bias=bias)
        self.Infuse = nn.Conv2d(nChannel * (prev + 1), nChannel, kernel_size=1, padding=0, bias=bias)
        if prev > 0:
            self.Conv = nn.Conv2d(nChannel * 2, nChannel, kernel_size=1, padding=0, bias=bias)

    def forward(self, x, res_list, temp_list):
        i = x[:, -self.nChannel:, :, :]
        x = i
        if self.prev == 0:
            x = self.Infuse(x)
        else:
            r = torch.cat(res_list, dim=1)
            x = self.Infuse(r)
        ori = x
        out_list = []
        x_list = []
        MP_list = []
        GP_list = []
        for i, model in enumerate(self.unit):
            res, x, scale = model(x, out_list, MP_list, GP_list)
            if i != self.num_unit - 1:
                GP_list.append(F.avg_pool2d(res, kernel_size=x.size()[2:]))
            out_list.append(res)
            x_list.append(x)
        x_ = self.Spatial_att(ori, x, out_list, temp_list)
        x = self.Fuse(torch.cat([x_, x], dim=1))
        """
    if self.prev==0:x = self.lastFuse(x)
    else:
        r = torch.cat(temp_list,dim=1)
        x = torch.cat([x,r],dim=1)
        x = self.lastFuse(x)
    """
        return x, x + i, x_list


class Spatial_att_score2(nn.Module):

    def __init__(self, inChannel, num_block):
        super(Spatial_att_score2, self).__init__()
        self.conv1 = nn.Conv2d(inChannel * num_block, inChannel, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(inChannel * 2, inChannel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(inChannel, 1, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.FloatTensor(1).zero_())
        self.beta = nn.Parameter(torch.FloatTensor(1).zero_())
        nn.init.constant_(self.alpha, 1)

    def forward(self, x, res_list):
        res = torch.cat(res_list[1:], dim=1)
        score = self.conv1(res)
        score = self.conv2(torch.cat([score, res_list[0]], dim=1))
        i = score
        score = self.conv3(torch.tanh(score))
        score = self.conv4(torch.tanh(score))
        score = self.conv5(torch.tanh(score + i))
        """
    t = score
    m = t.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    std = ((t - m) ** 2).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    t = (t - m) / (std.sqrt()) * self.alpha + self.beta
    score = t
    """
        mask = F.sigmoid(score)
        return mask


class RAN(nn.Module):

    def __init__(self, args):
        super(RAN, self).__init__()
        nChannel = args.n_colors
        nFeat = args.n_feats
        scale = args.scale[0]
        self.args = args
        self.plot_cnt = 0
        self.args = args
        self.id = [None] * 200
        self.idx = 0
        rgb_mean = 0.45738, 0.43637, 0.40293
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std, -1)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.Mod = torch.nn.ModuleList([ST_Block5(args.n_resblocks, nFeat, i) for i in range(args.n_resgroups)])
        self.spatial_att = Spatial_att_score2(nFeat, args.n_resgroups)
        self.G_F1 = nn.Conv2d(nFeat * args.n_resgroups, nFeat, kernel_size=1, padding=0, bias=True)
        self.G_F33 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.G_F3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.G_F2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.Dblock = Dilated_block(nFeat)
        if scale == 4:
            self.up = nn.Sequential(nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1, bias=True), sub_pixel(2), nn.Conv2d(nFeat, nFeat * scale, kernel_size=3, padding=1, bias=True), sub_pixel(2))
        else:
            self.up = nn.Sequential(nn.Conv2d(nFeat, nFeat * scale * scale, kernel_size=3, padding=1, bias=True), sub_pixel(2))
        self.HR = nn.Sequential(nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(0.1), nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        self.plot_cnt += 1
        i = x
        x_list = []
        out_list = []
        x = self.conv1(x)
        x = self.conv2(x)
        x_list.append(x)
        out_list.append(x)
        io = x
        for ii, submodel in enumerate(self.Mod):
            if ii == 3:
                inp = x.detach()
            res, x, scale = submodel(x, x_list, out_list)
            if False:
                x += io
                io = x
            out_list.append(res)
            x_list.append(x)
            if ii == 3:
                s = scale
        s = None
        if self.plot_cnt == 200 and s is not None:
            self.plot_cnt = 0
            for idx, sle in enumerate(s):
                self.id[idx] = self.vis2.images(sle[0, :, :, :].unsqueeze(dim=1), nrow=6, win=self.id[idx], opts={'title': 'BLOCK-15'})
        x = self.G_F1(torch.cat(out_list[1:], dim=1))
        x = self.G_F33(x)
        if not self.args.nvis:
            self.id[self.idx] = self.vis2.heatmap(mask[0, 0, :, :], win=self.id[self.idx], opts={'title': 'SA'})
        self.idx += 1
        if self.idx == 20:
            self.idx = 0
        x = self.HR(x) + i
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
        super(NLayerDiscriminator, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class Discriminator_VGG_128(nn.Module):

    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_256(nn.Module):

    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_256, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_512(nn.Module):

    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_512, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv6_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn6_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv6_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn6_1 = nn.BatchNorm2d(nf * 8, affine=True)
        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))
        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))
        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))
        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))
        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))
        fea = self.lrelu(self.bn6_0(self.conv6_0(fea)))
        fea = self.lrelu(self.bn6_1(self.conv6_1(fea)))
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:feature_layer + 1])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            dev = x.device
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class GANLoss(nn.Module):

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf=nf, gc=gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class Rdb(nn.Module):

    def __init__(self, channels, filters, depth=3, scaling=1.0, name='Rdb', **kwargs):
        super(Rdb, self).__init__()
        self.name = name
        self.depth = depth
        self.scaling = scaling
        for i in range(depth):
            conv = EasyConv2d(channels + filters * i, filters, **kwargs)
            setattr(self, f'conv_{i}', conv)
        try:
            kwargs.pop('activation')
        except KeyError:
            pass
        conv = EasyConv2d(channels + filters * (depth - 1), channels, **kwargs)
        setattr(self, f'conv_{depth - 1}', conv)

    def forward(self, inputs):
        fl = [inputs]
        for i in range(self.depth):
            conv = getattr(self, f'conv_{i}')
            fl.append(conv(torch.cat(fl, dim=1)))
        return fl[-1] * self.scaling + inputs

    def extra_repr(self):
        return f'{self.name}: depth={self.depth}, scaling={self.scaling}'


class CascadeRdn(nn.Module):

    def __init__(self, channels, filters, depth=3, use_ca=False, name='CascadeRdn', **kwargs):
        super(CascadeRdn, self).__init__()
        self.name = name
        self.depth = to_list(depth, 2)
        self.ca = use_ca
        for i in range(self.depth[0]):
            setattr(self, f'conv11_{i}', nn.Conv2d(channels + filters * (i + 1), filters, 1))
            setattr(self, f'rdn_{i}', Rdb(channels, filters, self.depth[1], **kwargs))
            if use_ca:
                setattr(self, f'rcab_{i}', Rcab(channels))

    def forward(self, inputs):
        fl = [inputs]
        x = inputs
        for i in range(self.depth[0]):
            rdn = getattr(self, f'rdn_{i}')
            x = rdn(x)
            if self.ca:
                rcab = getattr(self, f'rcab_{i}')
                x = rcab(x)
            fl.append(x)
            c11 = getattr(self, f'conv11_{i}')
            x = c11(torch.cat(fl, dim=1))
        return x

    def extra_repr(self):
        return f'{self.name}: depth={self.depth}, ca={self.ca}'


class Crdn(nn.Module):

    def __init__(self, blocks=(4, 4), **kwargs):
        super(Crdn, self).__init__()
        self.blocks = to_list(blocks, 2)
        self.entry = nn.Sequential(nn.Conv2d(3, 32, 7, 1, 3), nn.Conv2d(32, 32, 5, 1, 2))
        self.exit = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.Conv2d(32, 3, 3, 1, 1))
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.up1 = Upsample([128, 64])
        self.up2 = Upsample([64, 32])
        self.cb1 = CascadeRdn(32, 32, 3, True)
        self.cb2 = CascadeRdn(64, 64, 3, True)
        self.cb3 = CascadeRdn(128, 128, 3, True)
        self.cb4 = CascadeRdn(128, 128, 3, True)
        self.cb5 = CascadeRdn(64, 64, 3, True)
        self.cb6 = CascadeRdn(32, 32, 3, True)

    def forward(self, inputs):
        entry = self.entry(inputs)
        x1 = self.cb1(entry)
        x = self.down1(x1)
        x2 = self.cb2(x)
        x = self.down2(x2)
        x = self.cb3(x)
        x = self.cb4(x)
        x = self.up1(x, x2)
        x = self.cb5(x)
        x = self.up2(x, x1)
        x = self.cb6(x)
        x += entry
        out = self.exit(x)
        return out


class UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, activation='prelu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)
        self.up_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)
        self.up_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, activation='prelu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)
        self.down_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)
        self.down_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, num_stages=1, activation='prelu'):
        super(D_UpBlock, self).__init__()
        self.conv = EasyConv2d(num_filter * num_stages, num_filter, 1, activation=activation)
        self.up_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)
        self.up_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)
        self.up_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, num_stages=1, activation='prelu'):
        super(D_DownBlock, self).__init__()
        self.conv = EasyConv2d(num_filter * num_stages, num_filter, 1, activation=activation)
        self.down_conv1 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)
        self.down_conv2 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation, transposed=True)
        self.down_conv3 = EasyConv2d(num_filter, num_filter, kernel_size, stride, activation=activation)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class Dbpn(nn.Module):

    def __init__(self, channels, scale, base_filter=64, feat=256, num_stages=7):
        super(Dbpn, self).__init__()
        kernel, stride = self.get_kernel_stride(scale)
        self.feat0 = EasyConv2d(channels, feat, 3, activation='prelu')
        self.feat1 = EasyConv2d(feat, base_filter, 1, activation='prelu')
        self.up1 = UpBlock(base_filter, kernel, stride)
        self.down1 = DownBlock(base_filter, kernel, stride)
        self.up2 = UpBlock(base_filter, kernel, stride)
        for i in range(2, num_stages):
            self.__setattr__(f'down{i}', D_DownBlock(base_filter, kernel, stride, i))
            self.__setattr__(f'up{i + 1}', D_UpBlock(base_filter, kernel, stride, i))
        self.num_stages = num_stages
        self.output_conv = EasyConv2d(num_stages * base_filter, channels, 3)

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        h = h2
        concat_h = h1
        concat_l = l1
        for i in range(2, self.num_stages):
            concat_h = torch.cat((h, concat_h), 1)
            l = self.__getattr__(f'down{i}')(concat_h)
            concat_l = torch.cat((l, concat_l), 1)
            h = self.__getattr__(f'up{i + 1}')(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        return x

    @staticmethod
    def get_kernel_stride(scale):
        if scale == 2:
            return 6, 2
        elif scale == 4:
            return 8, 4
        elif scale == 8:
            return 12, 8


class NoiseExtractor(nn.Module):

    def __init__(self, channel=32, layers=7, bn=False, **kwargs):
        super(NoiseExtractor, self).__init__()
        convs = [nn.Conv2d(3, channel, 3, 1, 1), nn.ReLU(True)]
        if bn:
            convs.insert(-1, nn.BatchNorm2d(channel))
        for i in range(1, layers - 1):
            convs += [nn.Conv2d(channel, channel, 3, 1, 1), nn.ReLU(True)]
            if bn:
                convs.insert(-1, nn.BatchNorm2d(channel))
        convs += [nn.Conv2d(channel, 3, 3, 1, 1)]
        self.body = nn.Sequential(*convs)

    def forward(self, x):
        return self.body(x)


class NoiseShifter(nn.Module):

    def __init__(self, channel=3, layers=8, bn=False, **kwargs):
        super(NoiseShifter, self).__init__()
        f = kwargs.get('filters', 32)
        ks = kwargs.get('kernel_size', 3)
        convs = [EasyConv2d(channel, f, ks, use_bn=bn, activation='lrelu')]
        for i in range(1, layers - 1):
            convs += [EasyConv2d(f, f, ks, use_bn=bn, activation='lrelu')]
        convs += [EasyConv2d(f, channel, ks, activation='sigmoid')]
        self.body = nn.Sequential(*convs)

    def forward(self, x):
        return self.body(x)


class NCL(nn.Module):

    def __init__(self, channels, filters=32, layers=3, **kwargs):
        super(NCL, self).__init__()
        ks = kwargs.get('kernel_size', 3)
        c = channels
        f = filters
        conv = []
        for i in range(1, layers):
            if i == 1:
                conv.append(EasyConv2d(3, f, ks, activation='lrelu'))
            else:
                conv.append(EasyConv2d(f, f, ks, activation='lrelu'))
        self.gamma = nn.Sequential(*conv, EasyConv2d(f, c, ks, activation='sigmoid'))
        self.beta = nn.Sequential(*conv.copy(), EasyConv2d(f, c, ks))

    def forward(self, x, noise=None):
        if noise is None:
            return x
        return x * self.gamma(noise) + self.beta(noise)


class CRDB(nn.Module):

    def __init__(self, channels, depth=3, scaling=1.0, name='Rdb', **kwargs):
        super(CRDB, self).__init__()
        self.name = name
        self.depth = depth
        self.scaling = scaling
        ks = kwargs.get('kernel_size', 3)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', ks // 2)
        dilation = kwargs.get('dilation', 1)
        group = kwargs.get('group', 1)
        bias = kwargs.get('bias', True)
        c = channels
        for i in range(depth):
            conv = nn.Conv2d(c + c * i, c, ks, stride, padding, dilation, group, bias)
            if i < depth - 1:
                conv = nn.Sequential(conv, nn.ReLU(True))
            setattr(self, f'conv_{i}', conv)
        self.ncl = NCL(c)

    def forward(self, inputs, noise):
        fl = [inputs]
        for i in range(self.depth):
            conv = getattr(self, f'conv_{i}')
            fl.append(conv(torch.cat(fl, dim=1)))
        y = fl[-1] * self.scaling + inputs
        return self.ncl(y, noise)


class Drn(nn.Module):

    def __init__(self, channel, scale, n_cb, **kwargs):
        super(Drn, self).__init__()
        f = kwargs.get('filters', 64)
        self.entry = nn.Sequential(nn.Conv2d(channel, f, 3, 1, 1), nn.Conv2d(f, f, 3, 1, 1))
        for i in range(n_cb):
            setattr(self, f'cb{i}', CascadeRdn(f))
        self.n_cb = n_cb
        self.tail = nn.Sequential(Upsample(f, scale), nn.Conv2d(f, channel, 3, 1, 1))

    def forward(self, x, noise=None):
        x0 = self.entry(x)
        x = x0
        for i in range(self.n_cb):
            cb = getattr(self, f'cb{i}')
            x = cb(x, noise)
        x += x0
        return self.tail(x)


class Edsr(nn.Module):

    def __init__(self, scale, channel, n_resblocks, n_feats, rgb_range):
        super(Edsr, self).__init__()
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.404), True, rgb_range)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.404), False, rgb_range)
        m_head = [EasyConv2d(channel, n_feats, 3)]
        m_body = [RB(n_feats, n_feats, 3, activation='relu') for _ in range(n_resblocks)]
        m_body.append(EasyConv2d(n_feats, n_feats, 3))
        m_tail = [Upsample(n_feats, scale), EasyConv2d(n_feats, channel, 3)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, **kwargs):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x) + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


class MultiscaleUpsample(nn.Module):

    def __init__(self, channel, scales=(2, 3, 4), **kwargs):
        super(MultiscaleUpsample, self).__init__()
        for i in scales:
            self.__setattr__(f'up{i}', Upsample(channel, i, **kwargs))

    def forward(self, x, scale):
        return self.__getattr__(f'up{scale}')(x)


class Mdsr(nn.Module):

    def __init__(self, scales, channel, n_resblocks, n_feats, rgb_range):
        super(Mdsr, self).__init__()
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.404), True, rgb_range)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.404), False, rgb_range)
        m_head = [EasyConv2d(channel, n_feats, 3)]
        self.pre_process = nn.ModuleList([nn.Sequential(RB(n_feats, kernel_size=5, activation='relu'), RB(n_feats, kernel_size=5, activation='relu')) for _ in scales])
        m_body = [RB(n_feats, kernel_size=3, activation='relu') for _ in range(n_resblocks)]
        m_body.append(EasyConv2d(n_feats, n_feats, 3))
        self.upsample = MultiscaleUpsample(n_feats, scales)
        m_tail = [EasyConv2d(n_feats, channel, 3)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale](x)
        res = self.body(x) + x
        x = self.upsample(res, scale)
        x = self.tail(x)
        x = self.add_mean(x)
        return x


class Rrdb(nn.Module):
    """
  Residual in Residual Dense Block
  """

    def __init__(self, nc, gc=32, depth=5, scaling=1.0, **kwargs):
        super(Rrdb, self).__init__()
        self.RDB1 = Rdb(nc, gc, depth, scaling, **kwargs)
        self.RDB2 = Rdb(nc, gc, depth, scaling, **kwargs)
        self.RDB3 = Rdb(nc, gc, depth, scaling, **kwargs)
        self.scaling = scaling

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.scaling) + x


class RRDB_Net(nn.Module):

    def __init__(self, channel, scale, nf, nb, gc=32):
        super(RRDB_Net, self).__init__()
        self.head = EasyConv2d(channel, nf, kernel_size=3)
        rb_blocks = [Rrdb(nf, gc, 5, 0.2, kernel_size=3, activation=Activation('lrelu', negative_slope=0.2)) for _ in range(nb)]
        LR_conv = EasyConv2d(nf, nf, kernel_size=3)
        upsampler = [Upsample(nf, scale, 'nearest', activation=Activation('lrelu', negative_slope=0.2))]
        HR_conv0 = EasyConv2d(nf, nf, kernel_size=3, activation='lrelu', negative_slope=0.2)
        HR_conv1 = EasyConv2d(nf, channel, kernel_size=3)
        self.body = nn.Sequential(*rb_blocks, LR_conv)
        self.tail = nn.Sequential(*upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x


class ReluRB(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(ReluRB, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, outchannels, 3, 1, 1)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs


class SRNet(nn.Module):

    def __init__(self, scale, channel, depth):
        super(SRNet, self).__init__()
        self.entry = EasyConv2d(channel * depth, 64, 3)
        self.exit = EasyConv2d(64, channel, 3)
        self.body = nn.Sequential(ReluRB(64, 64), ReluRB(64, 64), ReluRB(64, 64), nn.ReLU(True))
        self.conv = EasyConv2d(64, 64 * scale ** 2, 3)
        self.up = nn.PixelShuffle(scale)

    def forward(self, inputs):
        x = self.entry(inputs)
        y = self.body(x) + x
        y = self.conv(y)
        y = self.up(y)
        y = self.exit(y)
        return y


class Flownet(nn.Module):

    def __init__(self, channel):
        """Flow estimation network

    Originally from paper "FlowNet: Learning Optical Flow with Convolutional
    Networks" and adapted according to paper "Frame-Recurrent Video
    Super-Resolution".
    See Frvsr.py

    Args:
      channel: input channels of each sequential frame
    """
        super(Flownet, self).__init__()
        f = 32
        layers = []
        in_c = channel * 2
        for i in range(3):
            layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.MaxPool2d(2)]
            in_c = f
            f *= 2
        for i in range(3):
            layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.Conv2d(f, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            in_c = f
            f //= 2
        layers += [nn.Conv2d(in_c, f, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(f, 2, 3, 1, 1), nn.Tanh()]
        self.body = nn.Sequential(*layers)

    def forward(self, target, ref, gain=1):
        """Estimate densely optical flow from `ref` to `target`

    Args:
      target: frame A
      ref: frame B
      gain: a scalar multiplied to final flow map
    """
        x = torch.cat((target, ref), 1)
        x = self.body(x) * gain
        return x


def nd_meshgrid(*size, permute=None):
    _error_msg = 'Permute index must match mesh dimensions, should have {} indexes but got {}'
    size = np.array(size)
    ranges = []
    for x in size:
        ranges.append(np.linspace(-1, 1, x))
    mesh = np.stack(np.meshgrid(*ranges, indexing='ij'))
    if permute is not None:
        if len(permute) != len(size):
            raise ValueError(_error_msg.format(len(size), len(permute)))
        mesh = mesh[permute]
    return mesh.transpose(*range(1, mesh.ndim), 0)


class STN(nn.Module):
    """Spatial transformer network.
    For optical flow based frame warping.

  Args:
    mode: sampling interpolation mode of `grid_sample`
    padding_mode: can be `zeros` | `borders`
    normalized: flow value is normalized to [-1, 1] or absolute value
  """

    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize

    def forward(self, inputs, u, v=None, gain=1):
        batch = inputs.size(0)
        device = inputs.device
        mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
        mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
        if v is None:
            assert u.shape[1] == 2, 'optical flow must have 2 channels'
            _u, _v = u[:, 0], u[:, 1]
        else:
            _u, _v = u, v
        if not self.norm:
            h, w = inputs.shape[-2:]
            _u = _u / w * 2
            _v = _v / h * 2
        flow = torch.stack([_u, _v], dim=-1) * gain
        assert flow.shape == mesh.shape, f'Shape mis-match: {flow.shape} != {mesh.shape}'
        mesh = mesh + flow
        return F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode)


class SpaceToDim(nn.Module):

    def __init__(self, scale_factor, dims=(-2, -1), dim=0):
        super(SpaceToDim, self).__init__()
        self.scale_factor = scale_factor
        self.dims = dims
        self.dim = dim

    def forward(self, x):
        _shape = list(x.shape)
        shape = _shape.copy()
        dims = [x.dim() + self.dims[0] if self.dims[0] < 0 else self.dims[0], x.dim() + self.dims[1] if self.dims[1] < 0 else self.dims[1]]
        dims = [max(abs(dims[0]), abs(dims[1])), min(abs(dims[0]), abs(dims[1]))]
        if self.dim in dims:
            raise RuntimeError("Integrate dimension can't be space dimension!")
        shape[dims[0]] //= self.scale_factor
        shape[dims[1]] //= self.scale_factor
        shape.insert(dims[0] + 1, self.scale_factor)
        shape.insert(dims[1] + 1, self.scale_factor)
        dim = self.dim if self.dim < dims[1] else self.dim + 1
        dim = dim if dim <= dims[0] else dim + 1
        x = x.reshape(*shape)
        perm = [dim, dims[1] + 1, dims[0] + 2]
        perm = [i for i in range(min(perm))] + perm
        perm.extend(i for i in range(x.dim()) if i not in perm)
        x = x.permute(*perm)
        shape = _shape
        shape[self.dim] *= self.scale_factor ** 2
        shape[self.dims[0]] //= self.scale_factor
        shape[self.dims[1]] //= self.scale_factor
        return x.reshape(*shape)

    def extra_repr(self):
        return f'scale_factor={self.scale_factor}'


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.body = SpaceToDim(block_size, dim=1)

    def forward(self, x):
        return self.body(x)


class FRNet(nn.Module):

    def __init__(self, channel, scale, n_rb):
        super(FRNet, self).__init__()
        self.fnet = Flownet(channel)
        self.warp = STN(padding_mode='border')
        self.snet = SRNet(channel, scale, n_rb)
        self.space_to_depth = SpaceToDepth(scale)
        self.scale = scale

    def forward(self, lr, last_lr, last_sr):
        flow = self.fnet(lr, last_lr, gain=32)
        flow2 = self.scale * upsample(flow, self.scale)
        hw = self.warp(last_sr, flow2[:, 0], flow2[:, 1])
        lw = self.warp(last_lr, flow[:, 0], flow[:, 1])
        hws = self.space_to_depth(hw)
        y = self.snet(hws, lr)
        return y, hw, lw, flow2


class NoiseRemover(nn.Module):

    def __init__(self, in_channel, up, **kwargs):
        super(NoiseRemover, self).__init__()
        entry = nn.Conv2d(in_channel, 64, 3, 1, 1)
        rdn1 = CascadeRdn(64, 3, True)
        rdn2 = CascadeRdn(64, 3, True)
        exits = nn.Conv2d(64, 3, 3, 1, 1)
        if up:
            up = Upsample(64, 2)
            self.body = nn.Sequential(entry, rdn1, rdn2, up, exits)
        else:
            self.body = nn.Sequential(entry, rdn1, rdn2, exits)

    def forward(self, x, noise=None):
        if noise is not None:
            x = torch.cat([x, noise], dim=1)
        x = self.body(x)
        return x


class Mldn(nn.Module):

    def __init__(self):
        super(Mldn, self).__init__()
        self.ne = NoiseExtractor(bn=True)
        self.sub_x8 = NoiseRemover(6, True)
        self.sub_x4 = NoiseRemover(6, True)
        self.sub_x2 = NoiseRemover(6, True)
        self.main = NoiseRemover(6, False)

    def forward(self, x, x2, x4, x8):
        noise = self.ne(x8)
        up4 = self.sub_x8(x8, noise)
        up2 = self.sub_x4(x4, up4)
        up1 = self.sub_x2(x2, up2)
        clean = self.main(x, up1)
        return clean, up1, up2, up4, noise


class MSRB(nn.Module):

    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()
        self.conv_3_1 = EasyConv2d(n_feats, n_feats, 3)
        self.conv_3_2 = EasyConv2d(n_feats * 2, n_feats * 2, 3)
        self.conv_5_1 = EasyConv2d(n_feats, n_feats, 5)
        self.conv_5_2 = EasyConv2d(n_feats * 2, n_feats * 2, 5)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)

    def forward(self, x):
        input_1 = x
        output_3_1 = F.relu(self.conv_3_1(input_1))
        output_5_1 = F.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = F.relu(self.conv_3_2(input_2))
        output_5_2 = F.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class Msrn(nn.Module):

    def __init__(self, channel, scale, n_feats, n_blocks, rgb_range):
        super(Msrn, self).__init__()
        self.n_blocks = n_blocks
        rgb_mean = 0.4488, 0.4371, 0.404
        self.sub_mean = MeanShift(rgb_mean, True, rgb_range)
        modules_head = [EasyConv2d(channel, n_feats, 3)]
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(MSRB(n_feats=n_feats))
        modules_tail = [EasyConv2d(n_feats * (self.n_blocks + 1), n_feats, 1), EasyConv2d(n_feats, n_feats, 3), Upsample(n_feats, scale), EasyConv2d(n_feats, channel, 3)]
        self.add_mean = MeanShift(rgb_mean, False, rgb_range)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)
        res = torch.cat(MSRB_out, 1)
        x = self.tail(res)
        x = self.add_mean(x)
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module (ECCV 18)
  - CA: channel attention module
  - SA: spatial attention module

  Args:
    channels: input channel of tensors
    channel_reduction: reduction ratio in `CA`
    spatial_first: put SA ahead of CA (default: CA->SA)
  """


    class CA(nn.Module):

        def __init__(self, channels, ratio=16):
            super(CBAM.CA, self).__init__()
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.mlp = nn.Sequential(nn.Conv2d(channels, channels // ratio, 1), nn.ReLU(), nn.Conv2d(channels // ratio, channels, 1))

        def forward(self, x):
            maxpool = self.max_pool(x)
            avgpool = self.avg_pool(x)
            att = F.sigmoid(self.mlp(maxpool) + self.mlp(avgpool))
            return att * x


    class SA(nn.Module):

        def __init__(self, kernel_size=7):
            super(CBAM.SA, self).__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, 1, kernel_size // 2)

        def forward(self, x):
            max_c_pool = x.max(dim=1, keepdim=True)
            avg_c_pool = x.mean(dim=1, keepdim=True)
            y = torch.cat([max_c_pool, avg_c_pool], dim=1)
            att = F.sigmoid(self.conv(y))
            return att * x

    def __init__(self, channels, channel_reduction=16, spatial_first=None):
        super(CBAM, self).__init__()
        self.channel_attention = CBAM.CA(channels, ratio=channel_reduction)
        self.spatial_attention = CBAM.SA(7)
        self.spatial_first = spatial_first

    def forward(self, inputs):
        if self.spatial_first:
            x = self.spatial_attention(inputs)
            return self.channel_attention(x)
        else:
            x = self.channel_attention(inputs)
            return self.spatial_attention(x)


def zip(url):
    url = Path(url)
    os.chdir(url)
    cmd = 'zip youku_results.zip *.y4m'
    subprocess.call(cmd, shell=True)
    subprocess.call('rm *.y4m', shell=True)


class Conv2dLSTMCell(nn.Module):
    """ConvLSTM cell.
  Copied from https://gist.github.com/Kaixhin/57901e91e5c5a8bac3eb0cbbdd3aba81
  Special thanks to @Kaixhin
  """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = Parameter(torch.Tensor(4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = Parameter(torch.Tensor(4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = Parameter(torch.Tensor(3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * out_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * out_channels))
            self.bias_ch = Parameter(torch.Tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch', None)
        self.register_buffer('wc_blank', torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h_0, c_0 = hx
        wx = F.conv2d(input, self.weight_ih, self.bias_ih, self.stride, self.padding, self.dilation, self.groups)
        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride, self.padding_h, self.dilation, self.groups)
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride, self.padding_h, self.dilation, self.groups)
        v = Variable(self.wc_blank).reshape((1, -1, 1, 1))
        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], v.expand(wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)
        i = torch.sigmoid(wxhc[:, :self.out_channels])
        f = torch.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = torch.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = torch.sigmoid(wxhc[:, 3 * self.out_channels:])
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, (h_1, c_1)


def _pull_conv_args(**kwargs):

    def _get_and_pop(d: dict, key, default=None):
        if key in d:
            return d.pop(key)
        return d.get(key, default)
    f = _get_and_pop(kwargs, 'filters', 64)
    ks = _get_and_pop(kwargs, 'kernel_size', 3)
    activation = _get_and_pop(kwargs, 'activation', 'leaky')
    bias = _get_and_pop(kwargs, 'bias', True)
    norm = _get_and_pop(kwargs, 'norm', '')
    bn = norm.lower() in ('bn', 'batch')
    sn = norm.lower() in ('sn', 'spectral')
    return f, ks, activation, bias, bn, sn, kwargs


class DCGAN(nn.Module):
    """DCGAN-like discriminator:
    stack of conv2d layers, stride down to 4x4

  Args:
    channel: input tensor channel
    num_layers: number of total cnn layers
    norm: could be "None", "SN/Spectral" or "BN/Batch"
    leaky: leaky slope
    favor: some pre-defined topology:
      'A': s1 s2 s1 s2 ...
      'B': s1 s2 s2 s2 ...
    kwargs: additional options to common CNN

  Note: Since the input before FC layer is B*N*4*4, the input shape can be
    derived as 4 * (2 ** n_strided), where $n_{strided}=num_layers / 2$ in
    favor 'A' and $n_{strided}=num_layers - 1$ in favor 'B'.
  """

    def __init__(self, channel, num_layers, scale=4, norm=None, favor='A', **kwargs):
        super(DCGAN, self).__init__()
        f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
        net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn, use_bias=bias, negative_slope=0.2)]
        self.n_strided = 0
        counter = 1
        assert favor in ('A', 'B', 'C'), 'favor must be A | B | C'
        while True:
            f *= 2
            net.append(EasyConv2d(f // 2, f, ks + 1, 2, activation=act, use_bias=bias, use_bn=bn, use_sn=sn, **unparsed))
            self.n_strided += 1
            counter += 1
            if counter >= num_layers:
                break
            if favor in ('A', 'C'):
                net.append(EasyConv2d(f, f, ks, 1, activation=act, use_bias=bias, use_bn=bn, use_sn=sn, **unparsed))
                counter += 1
                if counter >= num_layers:
                    break
        if favor == 'C':
            self.body = nn.Sequential(*net, nn.AdaptiveAvgPool2d(1))
            linear = [nn.Linear(f, 100, bias), Activation(act, in_place=True, **unparsed), nn.Linear(100, 1, bias)]
        else:
            self.body = nn.Sequential(*net)
            linear = [nn.Linear(f * scale * scale, 100, bias), Activation(act, in_place=True, **unparsed), nn.Linear(100, 1, bias)]
        if sn:
            linear[0] = nn.utils.spectral_norm(linear[0])
            linear[2] = nn.utils.spectral_norm(linear[2])
        self.linear = nn.Sequential(*linear)

    def forward(self, x):
        y = self.body(x).flatten(1)
        return self.linear(y)


class Residual(nn.Module):
    """Resnet-like discriminator.
    Stack of residual block, avg_pooling down to 4x4.

  Args:
    channel: input tensor channel
    num_residual: number of total cnn layers
    norm: could be "None", "SN/Spectral" or "BN/Batch"
    leaky: leaky slope
    favor: some pre-defined topology:
      'A': norm before 1st conv in residual
      'B': norm after 2nd conv in residual
    kwargs: additional options to common CNN

  Note: there is always activation and norm after 1st conv; if channel mis-
    matches, a 1x1 conv is used for shortcut
  """

    def __init__(self, channel, num_residual, norm=None, favor='A', **kwargs):
        super(Residual, self).__init__()
        f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
        net = [EasyConv2d(channel, f, ks, activation=act, use_bn=bn, use_sn=sn, use_bias=bias, **unparsed)]
        for i in range(num_residual):
            net.append(RB(f, ks, act, bias, bn, sn, favor == 'A'))
            net.append(nn.AvgPool2d(2))
        net.append(Activation(act, in_place=True, **unparsed))
        self.body = nn.Sequential(*net)
        linear = [nn.Linear(f * 4 * 4, 100, bias), Activation(act, in_place=True, **unparsed), nn.Linear(100, 1, bias)]
        if sn:
            linear[0] = nn.utils.spectral_norm(linear[0])
            linear[2] = nn.utils.spectral_norm(linear[2])
        self.linear = nn.Sequential(*linear)
        self.n_strided = num_residual

    def forward(self, x):
        assert x.size(2) == x.size(3) == 4 * 2 ** self.n_strided
        y = self.body(x).flatten(1)
        return self.linear(y)


class PatchGAN(nn.Module):
    """Defines a PatchGAN discriminator
  Args:
      channel: the number of channels in input images
      num_layers: number of total cnn layers
      norm: could be "None", "SN/Spectral" or "BN/Batch"
  """

    def __init__(self, channel, num_layers=3, norm=None, **kwargs):
        super(PatchGAN, self).__init__()
        f, ks, act, bias, bn, sn, unparsed = _pull_conv_args(norm=norm, **kwargs)
        sequence = [EasyConv2d(channel, f, ks + 1, 2, activation=act, use_bn=bn, use_sn=sn, use_bias=bias, **unparsed)]
        in_c = f
        out_c = f * 2
        for n in range(1, num_layers):
            sequence.append(EasyConv2d(in_c, out_c, ks + 1, 2, activation=act, use_bn=bn, use_sn=sn, use_bias=bias, **unparsed))
            in_c = out_c
            out_c *= 2
        sequence += [EasyConv2d(in_c, out_c, ks, activation=act, use_bn=bn, use_sn=sn, use_bias=bias, **unparsed), EasyConv2d(out_c, 1, 1)]
        self.body = nn.Sequential(*sequence)

    def forward(self, x):
        return self.body(x)


def gaussian_kernel(kernel_size: (int, tuple, list), width: float):
    """generate a gaussian kernel

  Args:
      kernel_size: the size of generated gaussian kernel. If is a scalar, the
                   kernel is a square matrix, or it's a kernel of HxW.
      width: the standard deviation of gaussian kernel. If width is 0, the
             kernel is identity, if width reaches to +inf, the kernel becomes
             averaging kernel.
  """
    kernel_size = np.asarray(to_list(kernel_size, 2), np.float)
    half_ksize = (kernel_size - 1) / 2.0
    x, y = np.mgrid[-half_ksize[0]:half_ksize[0] + 1, -half_ksize[1]:half_ksize[1] + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * width ** 2))
    return kernel / (kernel.sum() + 1e-08)


def gaussian_noise(inputs: torch.Tensor, stddev=None, sigma_max=0.06, channel_wise=1):
    """Add channel wise gaussian noise."""
    if stddev is None:
        stddev = torch.rand(channel_wise) * sigma_max
    stddev = torch.tensor(stddev, device=inputs.device)
    if DATA_FORMAT == 'channels_first':
        stddev = stddev.reshape([1, -1] + [1] * (inputs.ndim - 2))
    else:
        stddev = stddev.reshape([1] * (inputs.ndim - 1) + [-1])
    noise_map = torch.randn_like(inputs) * stddev
    return noise_map


def imfilter(image: torch.Tensor, kernel: torch.Tensor, padding=None):
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)
        assert image.dim() == 4, f'Dim of image must be 4, but is {image.dim()}'
        if kernel.dtype != image.dtype:
            kernel = kernel
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0)
            kernel = torch.cat([kernel] * image.shape[0])
        assert kernel.dim() == 3, f'Dim of kernel must be 3, but is {kernel.dim()}'
        ret = []
        for i, k in zip(image.split(1), kernel.split(1)):
            _c = i.shape[1]
            _k = k.unsqueeze(0)
            _p = torch.zeros_like(_k)
            _m = []
            for j in range(_c):
                t = [_p] * _c
                t[j] = _k
                _m.append(torch.cat(t, dim=1))
            _k = torch.cat(_m, dim=0)
            if padding is None:
                ret.append(F.conv2d(i, _k, padding=[(x // 2) for x in kernel.shape[1:]]))
            elif callable(padding):
                ret.append(F.conv2d(padding(i), _k))
            else:
                raise ValueError('Wrong padding value!')
        return torch.cat(ret)


def poisson_noise(inputs: torch.Tensor, stddev=None, sigma_max=0.16, channel_wise=1):
    """Add poisson noise to inputs."""
    if stddev is None:
        stddev = torch.rand(channel_wise) * sigma_max
    stddev = torch.tensor(stddev, device=inputs.device)
    if DATA_FORMAT == 'channels_first':
        stddev = stddev.reshape([1, -1] + [1] * (inputs.ndim - 2))
    else:
        stddev = stddev.reshape([1] * (inputs.ndim - 1) + [-1])
    sigma_map = (1 - inputs) * stddev
    return torch.randn_like(inputs) * sigma_map


class Distorter(nn.Module):
    """Randomly add the noise and blur of an image.

  Args:
      gaussian_noise_std (float or tuple of float (min, max)): How much to
          additive gaussian white noise. gaussian_noise_std is chosen uniformly
          from [0, std] or the given [min, max]. Should be non negative numbers.
      poisson_noise_std (float or tuple of float (min, max)): How much to
          poisson noise. poisson_noise_std is chosen uniformly from [0, std] or
          the given [min, max]. Should be non negative numbers.
      gaussian_blur_std (float or tuple of float (min, max)): How much to
          blur kernel. gaussian_blur_std is chosen uniformly from [0, std] or
          the given [min, max]. Should be non negative numbers.
  """

    def __init__(self, gaussian_noise_std=0, poisson_noise_std=0, gaussian_blur_std=0):
        super(Distorter, self).__init__()
        self.awgn = self._check_input(gaussian_noise_std, 'awgn', center=0, bound=(0, 75 / 255), clip_first_on_zero=True)
        self.poisson = self._check_input(poisson_noise_std, 'poisson', center=0, bound=(0, 50 / 255), clip_first_on_zero=True)
        self.blur = self._check_input(gaussian_blur_std, 'blur', center=0)
        self.blur_padding = nn.ReflectionPad2d(7)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(name, bound))
        else:
            if value < 0:
                raise ValueError('If {} is a single number, it must be non negative.'.format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        if value[0] == value[1] == center:
            value = None
        return value

    def forward(self, img):
        factors = []
        blur_factor = 0
        if self.blur is not None:
            blur_factor = random.uniform(*self.blur)
            img = imfilter(img, torch.tensor(gaussian_kernel(15, blur_factor), device=img.device), self.blur_padding)
        awgn_factor = 0, 0, 0
        if self.awgn is not None:
            _r = random.uniform(*self.awgn)
            _g = random.uniform(*self.awgn)
            _b = random.uniform(*self.awgn)
            img += gaussian_noise(img, stddev=(_r, _g, _b))
            awgn_factor = _r, _g, _b
        poisson_factor = _r, _g, _b
        if self.poisson is not None:
            _r = random.uniform(*self.poisson)
            _g = random.uniform(*self.poisson)
            _b = random.uniform(*self.poisson)
            img += poisson_noise(img, stddev=(_r, _g, _b))
            poisson_factor = _r, _g, _b
        fac = [blur_factor, *awgn_factor, *poisson_factor]
        factors.append(torch.tensor(fac))
        img = img.clamp(0, 1)
        return img, torch.stack(factors)


def gan_bce_loss(x, as_real: bool):
    """vanilla GAN binary cross entropy loss"""
    if as_real:
        return F.binary_cross_entropy_with_logits(x, torch.ones_like(x))
    else:
        return F.binary_cross_entropy_with_logits(x, torch.zeros_like(x))


def ragan_bce_loss(x, y, x_real_than_y: bool=True):
    """relativistic average GAN loss"""
    if x_real_than_y:
        return F.binary_cross_entropy_with_logits(x - y.mean(), torch.ones_like(x)) + F.binary_cross_entropy_with_logits(y - x.mean(), torch.zeros_like(y))
    else:
        return F.binary_cross_entropy_with_logits(y - x.mean(), torch.ones_like(x)) + F.binary_cross_entropy_with_logits(x - y.mean(), torch.zeros_like(y))


def rgan_bce_loss(x, y, x_real_than_y: bool=True):
    """relativistic GAN loss"""
    if x_real_than_y:
        return F.binary_cross_entropy_with_logits(x - y, torch.ones_like(x))
    else:
        return F.binary_cross_entropy_with_logits(y - x, torch.ones_like(x))


class GeneratorLoss(nn.Module):

    def __init__(self, name='GAN'):
        self.type = name
        super(GeneratorLoss, self).__init__()

    def forward(self, x, y=None):
        if self.type == 'RGAN':
            return rgan_bce_loss(x, y, True)
        elif self.type == 'RAGAN':
            return ragan_bce_loss(x, y, True)
        else:
            return gan_bce_loss(x, True)


class DiscriminatorLoss(nn.Module):

    def __init__(self, name='GAN'):
        self.type = name
        super(DiscriminatorLoss, self).__init__()

    def forward(self, x, y=None):
        if self.type == 'RGAN':
            return rgan_bce_loss(x, y, False)
        elif self.type == 'RAGAN':
            return ragan_bce_loss(x, y, False)
        else:
            return gan_bce_loss(x, False) + gan_bce_loss(y, True)


class VggFeatureLoss(nn.Module):
    _LAYER_NAME = {'block1_conv1': 1, 'block1_conv2': 3, 'block2_conv1': 6, 'block2_conv2': 8, 'block3_conv1': 11, 'block3_conv2': 13, 'block3_conv3': 15, 'block3_conv4': 17, 'block4_conv1': 20, 'block4_conv2': 22, 'block4_conv3': 24, 'block4_conv4': 26, 'block5_conv1': 29, 'block5_conv2': 31, 'block5_conv3': 33, 'block5_conv4': 35}
    """VGG19 based perceptual loss from ECCV 2016.
  
  Args:
    layer_names: a list of `_LAYER_NAME` strings, specify features to forward.
    before_relu: forward features before ReLU activation.
    external_weights: a path to an external vgg weights file, default download
      from model zoo.
  """

    def __init__(self, layer_names, before_relu=False, external_weights=None):
        super(VggFeatureLoss, self).__init__()
        if not external_weights:
            net = torchvision.models.vgg19(pretrained=True)
        else:
            net = torchvision.models.vgg19()
            net.load_state_dict(torch.load(external_weights))
        for p in net.parameters():
            p.requires_grad = False
        self.childs = nn.Sequential(*net.features.children())
        self.eval()
        self.exit_id = [(self._LAYER_NAME[n] - int(before_relu)) for n in layer_names]

    def normalize(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        assert x.size(1) == 3, 'wrong channel! must be 3!!'
        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        mean = mean
        std = std
        return (x - mean) / std

    def forward(self, x):
        exits = []
        x = self.normalize(x)
        for i, fn in enumerate(self.childs.children()):
            x = fn(x)
            if i in self.exit_id:
                exits.append(x)
            if i >= max(self.exit_id):
                break
        return exits


def transpose(x: torch.Tensor, dims):
    """transpose like numpy and tensorflow"""
    _dims = list(dims)
    for i in range(len(_dims)):
        if _dims[i] != i:
            x = x.transpose(i, _dims[i])
            j = _dims.index(i)
            _dims[i], _dims[j] = i, _dims[i]
    return x


def irtranspose(x: torch.Tensor, dims):
    """back transpose.
    `x = irtranspose(transpose(x, d), d)`
  """
    _dims = list(dims)
    _ir_dims = [_dims.index(i) for i in range(len(_dims))]
    return transpose(x, _ir_dims)


class STTN(nn.Module):
    """Spatio-temporal transformer network. (ECCV 2018)

  Args:
    transpose_ncthw: how input tensor be transposed to format NCTHW
    mode: sampling interpolation mode of `grid_sample`
    padding_mode: can be `zeros` | `borders`
    normalize: flow value is normalized to [-1, 1] or absolute value
  """

    def __init__(self, transpose_ncthw=(0, 1, 2, 3, 4), normalize=False, mode='bilinear', padding_mode='zeros'):
        super(STTN, self).__init__()
        self.normalized = normalize
        self.mode = mode
        self.padding_mode = padding_mode
        self.t = transpose_ncthw

    def forward(self, inputs, d, u, v):
        _error_msg = 'STTN only works for 5D tensor but got {}D input!'
        if inputs.dim() != 5:
            raise ValueError(_error_msg.format(inputs.dim()))
        device = inputs.device
        batch, channel, t, h, w = (inputs.shape[i] for i in self.t)
        mesh = nd_meshgrid(t, h, w, permute=[2, 1, 0])
        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
        mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
        _d, _u, _v = d, u, v
        if not self.normalized:
            _d = d / t * 2
            _u = u / w * 2
            _v = v / h * 2
        st_flow = torch.stack([_u, _v, _d], dim=-1)
        st_flow = st_flow.unsqueeze(1).repeat_interleave(t, dim=1)
        assert st_flow.shape == mesh.shape, f'Shape mis-match: {st_flow.shape} != {mesh.shape}'
        mesh = mesh + st_flow
        inputs = transpose(inputs, self.t)
        warp = F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode)
        warp = warp[:, :, 0:1]
        return irtranspose(warp, self.t)


class CoarseFineFlownet(nn.Module):

    def __init__(self, channel):
        """Coarse to fine flow estimation network

    Originally from paper "Real-Time Video Super-Resolution with Spatio-Temporal
    Networks and Motion Compensation".
    See Vespcn.py
    """
        super(CoarseFineFlownet, self).__init__()
        in_c = channel * 2
        conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
        conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv3 = nn.Sequential(nn.Conv2d(24, 24, 5, 2, 2), nn.ReLU(True))
        conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv5 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.Tanh())
        up1 = nn.PixelShuffle(4)
        self.coarse_flow = nn.Sequential(conv1, conv2, conv3, conv4, conv5, up1)
        in_c = channel * 3 + 2
        conv1 = nn.Sequential(nn.Conv2d(in_c, 24, 5, 2, 2), nn.ReLU(True))
        conv2 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv3 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv4 = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1), nn.ReLU(True))
        conv5 = nn.Sequential(nn.Conv2d(24, 8, 3, 1, 1), nn.Tanh())
        up2 = nn.PixelShuffle(2)
        self.fine_flow = nn.Sequential(conv1, conv2, conv3, conv4, conv5, up2)
        self.warp_c = STN(padding_mode='border')

    def forward(self, target, ref, gain=1):
        """Estimate optical flow from `ref` frame to `target` frame"""
        flow_c = self.coarse_flow(torch.cat((ref, target), 1))
        wc = self.warp_c(ref, flow_c[:, 0], flow_c[:, 1])
        flow_f = self.fine_flow(torch.cat((ref, target, flow_c, wc), 1)) + flow_c
        flow_f *= gain
        return flow_f


class SpaceToBatch(nn.Module):

    def __init__(self, block_size):
        super(SpaceToBatch, self).__init__()
        self.body = SpaceToDim(block_size, dim=0)

    def forward(self, x):
        return self.body(x)


class Fnet(nn.Module):

    def __init__(self, channel, L=2, gain=64):
        super(Fnet, self).__init__()
        self.lq_entry = nn.Sequential(nn.Conv2d(channel * (L + 1), 16, 3, 1, 1), SpaceToDepth(4), nn.Conv2d(256, 64, 1, 1, 0), Rdb(64), Rdb(64))
        self.hq_entry = nn.Sequential(nn.Conv2d(channel * L, 16, 3, 1, 1), SpaceToDepth(4), nn.Conv2d(256, 64, 1, 1, 0), Rdb(64), Rdb(64))
        self.flownet = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), Rdb(64), Rdb(64), Upsample(64, 4), nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())
        gain = torch.as_tensor([L, gain, gain], dtype=torch.float32)
        self.gain = gain.reshape(1, 3, 1, 1)

    def forward(self, lq, hq):
        x = torch.cat(lq, dim=1)
        y = torch.cat(hq, dim=1)
        x = self.lq_entry(x)
        y = self.hq_entry(y)
        z = torch.cat([x, y], dim=1)
        flow = self.flownet(z)
        gain = self.gain
        return flow * gain


class Unet(nn.Module):

    def __init__(self, channel, N=2):
        super(Unet, self).__init__()
        self.entry = nn.Sequential(nn.Conv2d(channel * N, 32, 3, 1, 1), SpaceToDepth(2), nn.Conv2d(128, 32, 1, 1, 0))
        self.exit = nn.Sequential(Upsample(32, 2), nn.Conv2d(32, channel, 3, 1, 1))
        self.down1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.up1 = RsrUp([64, 32])
        self.cb = CascadeRdn(64, 3, True)

    def forward(self, *inputs):
        inp = torch.cat(inputs, dim=1)
        c0 = self.entry(inp)
        c1 = self.down1(c0)
        x = self.cb(c1)
        c2 = self.up1(x, c0)
        out = self.exit(c2)
        return out


class TecoGenerator(nn.Module):
    """Generator in TecoGAN.

  Note: the flow estimation net `Fnet` shares with FRVSR.

  Args:
    filters: basic filter numbers [default: 64]
    num_rb: number of residual blocks [default: 16]
  """

    def __init__(self, channel, scale, filters, num_rb):
        super(TecoGenerator, self).__init__()
        rbs = []
        for i in range(num_rb):
            rbs.append(RB(filters, filters, 3, 'relu'))
        self.body = nn.Sequential(EasyConv2d(channel * (1 + scale ** 2), filters, 3, activation='relu'), *rbs, Upsample(filters, scale, 'nearest', activation='relu'), EasyConv2d(filters, channel, 3))

    def forward(self, x, prev, residual=None):
        """`residual` is the bicubically upsampled HR images"""
        sr = self.body(torch.cat((x, prev), dim=1))
        if residual is not None:
            sr += residual
        return sr


class Composer(nn.Module):

    def __init__(self, scale, channel, gain=24, filters=64, n_rb=16):
        super(Composer, self).__init__()
        self.fnet = Flownet(channel)
        self.gnet = TecoGenerator(channel, scale, filters, n_rb)
        self.warpper = STN(padding_mode='border')
        self.spd = SpaceToDepth(scale)
        self.scale = scale
        self.gain = gain

    def forward(self, lr, lr_pre, sr_pre, detach_fnet=None):
        """
    Args:
       lr: t_1 lr frame
       lr_pre: t_0 lr frame
       sr_pre: t_0 sr frame
       detach_fnet: detach BP to fnet
    """
        flow = self.fnet(lr, lr_pre, gain=self.gain)
        flow_up = self.scale * upsample(flow, self.scale)
        u, v = [x.squeeze(1) for x in flow_up.split(1, dim=1)]
        sr_warp = self.warpper(sr_pre, u, v)
        bi = upsample(lr, self.scale)
        if detach_fnet:
            sr = self.gnet(lr, self.spd(sr_warp.detach()), bi)
        else:
            sr = self.gnet(lr, self.spd(sr_warp), bi)
        return sr, sr_warp, flow, flow_up


class DbpnS(nn.Module):

    def __init__(self, scale, base_filter, feat, num_stages):
        super(DbpnS, self).__init__()
        kernel, stride = Dbpn.get_kernel_stride(scale)
        self.feat1 = EasyConv2d(base_filter, feat, 1, activation='prelu')
        for i in range(num_stages):
            self.__setattr__(f'up{i}', UpBlock(feat, kernel, stride))
            if i < num_stages - 1:
                self.__setattr__(f'down{i}', DownBlock(feat, kernel, stride))
        self.num_stages = num_stages
        self.output_conv = EasyConv2d(feat * num_stages, feat, 1)

    def forward(self, x):
        x = self.feat1(x)
        h1 = [self.__getattr__('up0')(x)]
        d1 = []
        for i in range(self.num_stages):
            d1.append(self.__getattr__(f'down{i}')(h1[-1]))
            h1.append(self.__getattr__(f'up{i + 1}')(d1[-1]))
        x = self.output_conv(torch.cat(h1, 1))
        return x


class Rbpn(nn.Module):

    def __init__(self, channel, scale, base_filter, feat, n_resblock, nFrames):
        super(Rbpn, self).__init__()
        self.nFrames = nFrames
        kernel, stride = Dbpn.get_kernel_stride(scale)
        self.feat0 = EasyConv2d(channel, base_filter, 3, activation='prelu')
        self.feat1 = EasyConv2d(8, base_filter, 3, activation='prelu')
        self.DBPN = DbpnS(scale, base_filter, feat, 3)
        modules_body1 = [RB(base_filter, kernel_size=3, activation='prelu') for _ in range(n_resblock)]
        modules_body1.append(EasyConv2d(base_filter, feat, kernel, stride, activation='prelu', transposed=True))
        self.res_feat1 = nn.Sequential(*modules_body1)
        modules_body2 = [RB(feat, kernel_size=3, activation='prelu') for _ in range(n_resblock)]
        modules_body2.append(EasyConv2d(feat, feat, 3, activation='prelu'))
        self.res_feat2 = nn.Sequential(*modules_body2)
        modules_body3 = [RB(feat, kernel_size=3, activation='prelu') for _ in range(n_resblock)]
        modules_body3.append(EasyConv2d(feat, base_filter, kernel, stride, activation='prelu'))
        self.res_feat3 = nn.Sequential(*modules_body3)
        self.output = EasyConv2d((nFrames - 1) * feat, channel, 3)

    def forward(self, x, neigbor, flow):
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])
            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)
        out = torch.cat(Ht, 1)
        output = self.output(out)
        return output


class Rcan(nn.Module):

    def __init__(self, channel, scale, n_resgroups, n_resblocks, n_feats, reduction, rgb_range):
        super(Rcan, self).__init__()
        rgb_mean = 0.4488, 0.4371, 0.404
        self.sub_mean = MeanShift(rgb_mean, True, rgb_range)
        modules_head = [EasyConv2d(channel, n_feats, 3)]
        modules_body = [ResidualGroup(n_feats, 3, reduction, n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(EasyConv2d(n_feats, n_feats, 3))
        modules_tail = [Upsample(n_feats, scale), EasyConv2d(n_feats, channel, 3)]
        self.add_mean = MeanShift(rgb_mean, False, rgb_range)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x) + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


class make_dense(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(make_dense, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        out = self.leaky_relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):

    def __init__(self, nDenselayer, channels, growth):
        super(RDB, self).__init__()
        modules = []
        channels_buffer = channels
        for i in range(nDenselayer):
            modules.append(make_dense(channels_buffer, growth))
            channels_buffer += growth
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(channels_buffer, channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class OFRnet(nn.Module):

    def __init__(self, upscale_factor):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.upscale_factor = upscale_factor
        self.conv_L1_1 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)
        self.RDB1_1 = RDB(4, 32, 32)
        self.RDB1_2 = RDB(4, 32, 32)
        self.bottleneck_L1 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L1_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        self.conv_L2_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
        self.RDB2_1 = RDB(4, 32, 32)
        self.RDB2_2 = RDB(4, 32, 32)
        self.bottleneck_L2 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L2_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        self.conv_L3_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
        self.RDB3_1 = RDB(4, 32, 32)
        self.RDB3_2 = RDB(4, 32, 32)
        self.bottleneck_L3 = nn.Conv2d(64, 2 * upscale_factor ** 2, 3, 1, 1, bias=False)
        self.conv_L3_2 = nn.Conv2d(2 * upscale_factor ** 2, 2 * upscale_factor ** 2, 3, 1, 1, bias=True)
        self.warper = STN()

    def forward(self, x):
        x_L1 = self.pool(x)
        _, _, h, w = x_L1.size()
        input_L1 = self.conv_L1_1(x_L1)
        buffer_1 = self.RDB1_1(input_L1)
        buffer_2 = self.RDB1_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L1 = self.bottleneck_L1(buffer)
        optical_flow_L1 = self.conv_L1_2(optical_flow_L1)
        optical_flow_L1_upscaled = self.upsample(optical_flow_L1)
        x_L2 = self.warper(x[:, 0, :, :].unsqueeze(1), optical_flow_L1_upscaled, gain=16)
        x_L2_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L2
        x_L2 = torch.cat((x, x_L2, x_L2_res, optical_flow_L1_upscaled), 1)
        input_L2 = self.conv_L2_1(x_L2)
        buffer_1 = self.RDB2_1(input_L2)
        buffer_2 = self.RDB2_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L2 = self.bottleneck_L2(buffer)
        optical_flow_L2 = self.conv_L2_2(optical_flow_L2)
        optical_flow_L2 = optical_flow_L2 + optical_flow_L1_upscaled
        x_L3 = self.warper(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_L2, gain=16)
        x_L3_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L3
        x_L3 = torch.cat((x, x_L3, x_L3_res, optical_flow_L2), 1)
        input_L3 = self.conv_L3_1(x_L3)
        buffer_1 = self.RDB3_1(input_L3)
        buffer_2 = self.RDB3_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L3 = self.bottleneck_L3(buffer)
        optical_flow_L3 = self.conv_L3_2(optical_flow_L3)
        optical_flow_L3 = self.shuffle(optical_flow_L3) + self.final_upsample(optical_flow_L2)
        return optical_flow_L3, optical_flow_L2, optical_flow_L1


class SRnet(nn.Module):

    def __init__(self, s, c, d):
        """
    Args:
      s: scale factor
      c: channel numbers
      d: video sequence number
    """
        super(SRnet, self).__init__()
        self.conv = nn.Conv2d(c * (2 * s ** 2 + d), 64, 3, 1, 1, bias=False)
        self.RDB_1 = RDB(5, 64, 32)
        self.RDB_2 = RDB(5, 64, 32)
        self.RDB_3 = RDB(5, 64, 32)
        self.RDB_4 = RDB(5, 64, 32)
        self.RDB_5 = RDB(5, 64, 32)
        self.bottleneck = nn.Conv2d(384, c * s ** 2, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(c * s ** 2, c * s ** 2, 3, 1, 1, bias=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=s)

    def forward(self, x):
        input = self.conv(x)
        buffer_1 = self.RDB_1(input)
        buffer_2 = self.RDB_2(buffer_1)
        buffer_3 = self.RDB_3(buffer_2)
        buffer_4 = self.RDB_4(buffer_3)
        buffer_5 = self.RDB_5(buffer_4)
        output = torch.cat((buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, input), 1)
        output = self.bottleneck(output)
        output = self.conv_2(output)
        output = self.shuffle(output)
        return output


class Sofvsr(nn.Module):

    def __init__(self, scale, channel, depth):
        super(Sofvsr, self).__init__()
        self.upscale_factor = scale
        self.c = channel
        self.OFRnet = OFRnet(upscale_factor=scale)
        self.SRnet = SRnet(scale, channel, depth)
        self.warper = STN()

    def forward(self, x):
        input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)
        input_21 = torch.cat((torch.unsqueeze(x[:, 2, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)
        flow_01_L3, flow_01_L2, flow_01_L1 = self.OFRnet(input_01)
        flow_21_L3, flow_21_L2, flow_21_L1 = self.OFRnet(input_21)
        draft_cube = x
        for i in range(self.upscale_factor):
            for j in range(self.upscale_factor):
                draft_01 = self.warper(x[:, :self.c, :, :], flow_01_L3[:, :, i::self.upscale_factor, j::self.upscale_factor] / self.upscale_factor, gain=16)
                draft_21 = self.warper(x[:, self.c * 2:, :, :], flow_21_L3[:, :, i::self.upscale_factor, j::self.upscale_factor] / self.upscale_factor, gain=16)
                draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)
        output = self.SRnet(draft_cube)
        return output, (flow_01_L3, flow_01_L2, flow_01_L1), (flow_21_L3, flow_21_L2, flow_21_L1)


class ZeroUpsample(nn.Module):

    def __init__(self, scale_factor):
        super(ZeroUpsample, self).__init__()
        self.ps = nn.PixelShuffle(scale_factor)
        self.scale = scale_factor

    def forward(self, x):
        z = torch.zeros_like(x).repeat_interleave(self.scale ** 2 - 1, dim=1)
        x = torch.cat((x, z), dim=1)
        return self.ps(x)


class SubPixelMotionCompensation(nn.Module):

    def __init__(self, scale):
        super(SubPixelMotionCompensation, self).__init__()
        self.zero_up = ZeroUpsample(scale)
        self.warper = STN()
        self.scale = scale

    def forward(self, x, u=0, v=0, flow=None):
        if flow is not None:
            u = flow[:, 0]
            v = flow[:, 1]
        x2 = self.zero_up(x)
        u2 = self.zero_up(u.unsqueeze(1)) * self.scale
        v2 = self.zero_up(v.unsqueeze(1)) * self.scale
        return self.warper(x2, u2.squeeze(1), v2.squeeze(1))


class MotionEstimation(nn.Module):

    def __init__(self, channel, gain=32):
        super(MotionEstimation, self).__init__()
        self.gain = gain
        self.flownet = CoarseFineFlownet(channel)

    def forward(self, target, ref, to_tuple=None):
        flow = self.flownet(target, ref, self.gain)
        if to_tuple:
            return flow[:, 0], flow[:, 1]
        return flow


class DetailFusion(nn.Module):

    def __init__(self, channel, base_filter):
        super(DetailFusion, self).__init__()
        f = base_filter
        self.enc1 = EasyConv2d(channel, f, 5, activation='relu')
        self.enc2 = nn.Sequential(EasyConv2d(f, f * 2, 3, 2, activation='relu'), EasyConv2d(f * 2, f * 2, 3, activation='relu'))
        self.enc3 = EasyConv2d(f * 2, f * 4, 3, 2, activation='relu')
        self.lstm = Conv2dLSTMCell(f * 4, f * 4, 3, 1, 1)
        self.dec1 = nn.Sequential(EasyConv2d(f * 4, f * 4, 3, activation='relu'), nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1), nn.ReLU(True))
        self.dec2 = nn.Sequential(EasyConv2d(f * 2, f * 2, 3, activation='relu'), nn.ConvTranspose2d(f * 2, f, 4, 2, 1), nn.ReLU(True))
        self.dec3 = nn.Sequential(EasyConv2d(f, f, 3, activation='relu'), EasyConv2d(f, channel, 5))

    def forward(self, x, hx):
        add1 = self.enc1(x)
        add2 = self.enc2(add1)
        h0 = self.enc3(add2)
        x, hx = self.lstm(h0, hx)
        x = self.dec1(x)
        x = self.dec2(x + add2)
        x = self.dec3(x + add1)
        return x, hx


class DetailRevealer(nn.Module):

    def __init__(self, scale, channel, **kwargs):
        super(DetailRevealer, self).__init__()
        self.base_filter = kwargs.get('base_filter', 32)
        self.me = MotionEstimation(channel, gain=kwargs.get('gain', 32))
        self.spmc = SubPixelMotionCompensation(scale)
        self.vsr = DetailFusion(channel, self.base_filter)
        self.scale = scale
        self.hidden_state = None

    def reset(self):
        self.hidden_state = None

    def forward(self, target, ref):
        flow = self.me(target, ref)
        hr_ref = self.spmc(ref, flow=flow)
        hr_target = upsample(target, self.scale)
        if self.hidden_state is None:
            batch, _, height, width = hr_ref.shape
            hidden_shape = batch, self.base_filter * 4, height // 4, width // 4
            hx = torch.zeros(hidden_shape, device=ref.device), torch.zeros(hidden_shape, device=ref.device)
        else:
            hx = self.hidden_state
        res, hx = self.vsr(hr_ref, hx)
        sr = hr_target + res
        self.hidden_state = hx
        return sr, flow


class TecoDiscriminator(nn.Module):

    def __init__(self, channel, filters, patch_size):
        super(TecoDiscriminator, self).__init__()
        f = filters
        self.conv0 = EasyConv2d(channel * 6, f, 3, activation='leaky')
        self.conv1 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
        self.conv2 = EasyConv2d(f, f, 4, 2, activation='leaky', use_bn=True)
        self.conv3 = EasyConv2d(f, f * 2, 4, 2, activation='leaky', use_bn=True)
        self.conv4 = EasyConv2d(f * 2, f * 4, 4, 2, activation='leaky', use_bn=True)
        self.linear = nn.Linear(f * 4 * (patch_size // 16) ** 2, 1)

    def forward(self, x):
        """The inputs `x` is the concat of 8 tensors.
      Note that we remove the duplicated gt/yt in paper (9 - 1 = 8).
    """
        l0 = self.conv0(x)
        l1 = self.conv1(l0)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        y = self.linear(l4.flatten(1))
        return y, (l1, l2, l3, l4)


class MotionCompensation(nn.Module):

    def __init__(self, channel, gain=32):
        super(MotionCompensation, self).__init__()
        self.gain = gain
        self.flownet = CoarseFineFlownet(channel)
        self.warp_f = STN(padding_mode='border')

    def forward(self, target, ref):
        flow = self.flownet(target, ref, self.gain)
        warping = self.warp_f(ref, flow[:, 0], flow[:, 1])
        return warping, flow


class Vespcn(nn.Module):

    def __init__(self, scale, channel, depth):
        super(Vespcn, self).__init__()
        self.sr = SRNet(scale, channel, depth)
        self.mc = MotionCompensation(channel)
        self.depth = depth

    def forward(self, *inputs):
        center = self.depth // 2
        target = inputs[center]
        refs = inputs[:center] + inputs[center + 1:]
        warps = []
        flows = []
        for r in refs:
            warp, flow = self.mc(target, r)
            warps.append(warp)
            flows.append(flow)
        warps.append(target)
        x = torch.cat(warps, 1)
        sr = self.sr(x)
        return sr, warps[:-1], flows


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CRDB,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoarseFineFlownet,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Cubic,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (D_DownBlock,
     lambda: ([], {'num_filter': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (D_UpBlock,
     lambda: ([], {'num_filter': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dilated_block,
     lambda: ([], {'inChannel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DnCnn,
     lambda: ([], {'channel': 4, 'layers': 1, 'bn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock,
     lambda: ([], {'num_filter': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Drcn,
     lambda: ([], {'scale': 1.0, 'channel': 4, 'n_recur': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Drrn,
     lambda: ([], {'channel': 4, 'n_ru': 4, 'n_rb': 4, 'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'group': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EasyConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneratorLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSRB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (MotionCompensation,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MotionEstimation,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NCL,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (NoiseExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NoiseShifter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RCAB,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RDB,
     lambda: ([], {'nDenselayer': 1, 'channels': 4, 'growth': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDB,
     lambda: ([], {'nf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDBNet,
     lambda: ([], {'in_nc': 4, 'out_nc': 4, 'nf': 4, 'nb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RRDB_Net,
     lambda: ([], {'channel': 4, 'scale': 1.0, 'nf': 4, 'nb': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReluRB,
     lambda: ([], {'inchannels': 4, 'outchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SRnet,
     lambda: ([], {'s': 4, 'c': 4, 'd': 4}),
     lambda: ([torch.rand([4, 144, 64, 64])], {}),
     True),
    (SpaceToBatch,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpaceToDepth,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Srcnn,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpBlock,
     lambda: ([], {'num_filter': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'channel': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vdsr,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_UpsampleLinear,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_UpsampleNearest,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (make_dense,
     lambda: ([], {'channels_in': 4, 'channels_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_LoSealL_VideoSuperResolution(_paritybench_base):
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

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

