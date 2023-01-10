import sys
_module = sys.modules[__name__]
del sys
develop = _module
compare = _module
convert_rosinality_ckpt = _module
core = _module
dataset = _module
distiller = _module
loss = _module
diffaug = _module
distiller_loss = _module
non_saturating_gan_loss = _module
perceptual_loss = _module
model_zoo = _module
models = _module
discriminator = _module
inception_v3 = _module
mapping_network = _module
mobile_synthesis_network = _module
modules = _module
constant_input = _module
functional = _module
idwt = _module
idwt_upsample = _module
legacy = _module
mobile_synthesis_block = _module
modulated_conv2d = _module
multichannel_image = _module
noise_injection = _module
ops = _module
fused_act = _module
fused_act_cuda = _module
upfirdn2d = _module
upfirdn2d_cuda = _module
styled_conv2d = _module
synthesis_network = _module
utils = _module
utils = _module
demo = _module
evaluate_fid = _module
generate = _module
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


import torch


import numpy as np


import random


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import grad


import torchvision.models as models


from torchvision import models


import math


from torch import nn


from torch.nn import functional as F


from torch.autograd import Function


from torch.utils.cpp_extension import load


import torchvision.transforms as TF


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


def _SFB2D(low, highs, g0_row, g1_row, g0_col, g1_col, mode):
    mode = int_to_mode(mode)
    lh, hl, hh = torch.unbind(highs, dim=2)
    lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
    return y


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """

    def __init__(self, wave='db1', mode='zero', trace_model=False):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 2:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = g0_col, g1_col
        elif len(wave) == 4:
            g0_col, g1_col = wave[0], wave[1]
            g0_row, g1_row = wave[2], wave[3]
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode
        self.trace_model = trace_model

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = mode_to_int(self.mode)
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            if not self.trace_model:
                ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
            else:
                ll = _SFB2D(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_op.upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_op.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_op.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    elif torch.cuda.is_available():
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    else:
        raise NotImplemented
    return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, trace_model=False):
    if input.device.type == 'cpu':
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        if trace_model:
            return F.leaky_relu(input + bias.view(1, input.size(1)), negative_slope=0.2) * scale
        else:
            return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2) * scale
    elif torch.cuda.is_available():
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    else:
        raise NotImplemented


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, trace_model=False):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
        self.trace_model = trace_model

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale, self.trace_model)


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ConvLayer(nn.Sequential):

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate))
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, trace_model=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul
        self.trace_model = trace_model

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, trace_model=self.trace_model)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class Discriminator(nn.Module):

    def __init__(self, size, channels_in=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], activate=True):
        super().__init__()
        channels = {(4): 512, (8): 512, (16): 512, (32): 512, (64): 256 * channel_multiplier, (128): 128 * channel_multiplier, (256): 64 * channel_multiplier, (512): 32 * channel_multiplier, (1024): 16 * channel_multiplier}
        convs = [ConvLayer(channels_in, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'), EqualLinear(channels[4], 1))
        self.activate = activate

    def forward(self, x):
        out = self.convs(x)
        out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        if self.activate:
            out = out.sigmoid()
        return {'out': out}

    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out


class R1Regularization(nn.Module):

    def __init__(self, r1_gamma=10):
        super().__init__()
        self.r1_gamma = r1_gamma

    def forward(self, x, x_pred):
        grad_x = grad(outputs=x_pred.sum(), inputs=x, create_graph=True)[0]
        grad_penalty = (grad_x.view(grad_x.size(0), -1).norm(2, dim=1) ** 2).mean()
        r1_loss = 0.5 * self.r1_gamma * grad_penalty
        return r1_loss


def get_default_transforms():
    return nn.Sequential(kornia.augmentation.RandomHorizontalFlip(), kornia.augmentation.RandomAffine(translate=(0.1, 0.3), scale=(0.7, 1.2), degrees=(-20, 20)), kornia.augmentation.RandomErasing())


class NonSaturatingGANLoss(nn.Module):

    def __init__(self, image_size, channels_in=3, r1_gamma=10):
        super().__init__()
        self.m = Discriminator(size=image_size, channels_in=channels_in, activate=False)
        self.transforms = get_default_transforms()
        self.r1_reg = R1Regularization(r1_gamma)

    def forward(self, x, diffaug_mode=True):
        if self.transforms is not None and diffaug_mode:
            x = self.transforms(x)
        return self.m(x)

    def loss_g(self, fake, *args, **kwargs):
        fake_loss = F.softplus(-self(fake, True)['out']).mean()
        return fake_loss

    def loss_d(self, fake, real):
        fake, real = fake.detach(), real.detach()
        fake_pred = self(fake, True)['out']
        fake_loss = F.softplus(fake_pred).mean()
        real_pred = self(real, True)['out']
        real_loss = F.softplus(-real_pred).mean()
        total_loss = fake_loss + real_loss
        return total_loss

    def reg_d(self, real):
        real.requires_grad = True
        real_pred = self(real, False)['out']
        r1_loss = self.r1_reg(real, real_pred)
        return r1_loss


class PerceptualNetwork(nn.Module):

    def __init__(self, arch='vgg16', layers={'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}):
        super().__init__()
        assert hasattr(models, arch)
        self.net = getattr(models, arch)(pretrained=True).features
        self.layers = layers

    def forward(self, x):
        out = {}
        for name, m in self.net._modules.items():
            x = m(x)
            if name in self.layers:
                out[name] = x
        return out


class PerceptualLoss(nn.Module):

    def __init__(self, size=None, arch='vgg16', layers={'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}):
        super().__init__()
        self.size = size
        self.net = PerceptualNetwork(arch, layers)

    def forward(self, pred, gt):
        if self.size is not None:
            pred = self._resize(pred)
            gt = self._resize(gt).detach()
        pred_out = self.net(pred)
        with torch.no_grad():
            pred_gt = self.net(gt)
        loss = 0
        for k, v in pred_out.items():
            loss += F.l1_loss(v, pred_gt[k])
        return loss

    def _resize(self, img):
        return F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)


class DistillerLoss(nn.Module):

    def __init__(self, discriminator_size, perceptual_size=256, loss_weights={'l1': 1.0, 'l2': 1.0, 'loss_p': 1.0, 'loss_g': 0.5}):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(perceptual_size)
        self.gan_loss = NonSaturatingGANLoss(image_size=int(discriminator_size))
        self.loss_weights = loss_weights
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode='zero', wave='db1')

    def loss_g(self, pred, gt):
        loss = {'l1': 0, 'l2': 0}
        for _pred in pred['freq']:
            _pred_rgb = self.dwt_to_img(_pred)
            _gt_rgb = F.interpolate(gt['img'], size=_pred_rgb.size(-1), mode='bilinear', align_corners=True)
            _gt_freq = self.img_to_dwt(_gt_rgb)
            loss['l1'] += self.l1_loss(_pred_rgb, _gt_rgb)
            loss['l2'] += self.l2_loss(_pred_rgb, _gt_rgb)
            loss['l1'] += self.l1_loss(_pred, _gt_freq)
            loss['l2'] += self.l2_loss(_pred, _gt_freq)
        loss['loss_p'] = self.perceptual_loss(pred['img'], gt['img'])
        loss['loss_g'] = self.gan_loss.loss_g(pred['img'], gt['img'])
        loss['loss'] = 0
        for k, w in self.loss_weights.items():
            if loss[k] is not None:
                loss['loss'] += w * loss[k]
            else:
                del loss[k]
        return loss

    def loss_d(self, pred, gt):
        loss = {}
        loss['loss'] = loss['loss_d'] = self.gan_loss.loss_d(pred['img'].detach(), gt['img'])
        return loss

    def reg_d(self, real):
        out = {}
        out['loss'] = out['d_reg'] = self.gan_loss.reg_d(real['img'])
        return out

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


class MappingNetwork(nn.Module):

    def __init__(self, style_dim, n_layers, lr_mlp=0.01):
        super().__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class IDWTUpsaplme(nn.Module):

    def __init__(self, channels_in, style_dim):
        super().__init__()
        self.channels = channels_in // 4
        assert self.channels * 4 == channels_in
        self.idwt = DWTInverse(mode='zero', wave='db1')
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)

    def forward(self, x, style):
        b, _, h, w = x.size()
        x = self.modulation(style).view(b, -1, 1, 1) * x
        low = x[:, :self.channels]
        high = x[:, self.channels:]
        high = high.view(b, self.channels, 3, h, w)
        x = self.idwt((low, [high]))
        return x


class ModulatedConv2d(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels_out, channels_in, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer('style_inv', torch.randn(1, 1, channels_in, 1, 1))
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-08)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class MultichannelIamge(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size=1):
        super().__init__()
        self.conv = ModulatedConv2d(channels_in, channels_out, style_dim, kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))

    def forward(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.trace_model = False

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if not hasattr(self, 'noise') and self.trace_model:
            self.register_buffer('noise', noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise


class StyledConv2d(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size, demodulate=True, conv_module=ModulatedConv2d):
        super().__init__()
        self.conv = conv_module(channels_in, channels_out, style_dim, kernel_size, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.act(out + self.bias)
        return out


class MobileSynthesisBlock(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size=3, conv_module=ModulatedConv2d):
        super().__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(channels_in // 4, channels_out, style_dim, kernel_size, conv_module=conv_module)
        self.conv2 = StyledConv2d(channels_out, channels_out, style_dim, kernel_size, conv_module=conv_module)
        self.to_img = MultichannelIamge(channels_in=channels_out, channels_out=12, style_dim=style_dim, kernel_size=1)

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])
        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3


class ModulatedDWConv2d(nn.Module):

    def __init__(self, channels_in, channels_out, style_dim, kernel_size, demodulate=True):
        super().__init__()
        self.weight_dw = nn.Parameter(torch.randn(channels_in, 1, kernel_size, kernel_size))
        self.weight_permute = nn.Parameter(torch.randn(channels_out, channels_in, 1, 1))
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer('style_inv', torch.randn(1, 1, channels_in, 1, 1))
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-08)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)


class NoiseManager:

    def __init__(self, noise, device, trace_model=False):
        self.device = device
        self.noise_lut = {}
        if noise is not None:
            for i in range(len(noise)):
                if not None in noise:
                    self.noise_lut[noise[i].size(-1)] = noise[i]
        self.trace_model = trace_model

    def __call__(self, size, b=1):
        if self.trace_model:
            return None if b == 1 else [None] * b
        if size in self.noise_lut:
            return self.noise_lut[size]
        else:
            return torch.randn(b, 1, size, size)


class MobileSynthesisNetwork(nn.Module):

    def __init__(self, style_dim, channels=[512, 512, 512, 512, 512, 256, 128, 64]):
        super().__init__()
        self.style_dim = style_dim
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv2d(channels[0], channels[0], style_dim, kernel_size=3)
        self.to_img1 = MultichannelIamge(channels_in=channels[0], channels_out=12, style_dim=style_dim, kernel_size=1)
        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(MobileSynthesisBlock(channels_in, channels_out, style_dim, 3, conv_module=ModulatedDWConv2d))
            channels_in = channels_out
        self.idwt = DWTInverse(mode='zero', wave='db1')
        self.register_buffer('device_info', torch.zeros(1))
        self.trace_model = False

    def forward(self, style, noise=None):
        out = {'noise': [], 'freq': [], 'img': None}
        noise = NoiseManager(noise, self.device_info.device, self.trace_model)
        hidden = self.input(style)
        out['noise'].append(noise(hidden.size(-1)))
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=out['noise'][-1])
        img = self.to_img1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out['freq'].append(img)
        for i, m in enumerate(self.layers):
            out['noise'].append(noise(2 ** (i + 3), 2))
            _style = style if style.ndim == 2 else style[:, m.wsize() * i + 1:m.wsize() * i + m.wsize() + 1, :]
            hidden, freq = m(hidden, _style, noise=out['noise'][-1])
            out['freq'].append(freq)
        out['img'] = self.dwt_to_img(out['freq'][-1])
        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2


class Upsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * factor ** 2
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class StyledConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class SynthesisBlock(nn.Module):

    def __init__(self, in_channel, out_channel, style_dim, kernel_size=3, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.conv1 = StyledConv(in_channel, out_channel, kernel_size, style_dim, upsample=True, blur_kernel=blur_kernel)
        self.conv2 = StyledConv(out_channel, out_channel, kernel_size, style_dim, blur_kernel=blur_kernel)
        self.to_rgb = ToRGB(out_channel, style_dim)

    def forward(self, hidden, style, noise=[None, None]):
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])
        rgb = self.to_rgb(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, rgb


class SynthesisNetwork(nn.Module):

    def __init__(self, size, style_dim, blur_kernel=[1, 3, 3, 1], channels=[512, 512, 512, 512, 512, 256, 128, 64, 32]):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.input = ConstantInput(channels[0])
        self.conv1 = StyledConv(channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(channels[0], style_dim, upsample=False)
        self.layers = nn.ModuleList()
        in_channel = channels[0]
        for out_channel in channels[1:]:
            self.layers.append(SynthesisBlock(in_channel, out_channel, style_dim, 3, blur_kernel=blur_kernel))
            in_channel = out_channel
        self.upsample = Upsample(blur_kernel)

    def forward(self, style, noise=None):
        out = {'noise': [], 'rgb': [], 'img': None}
        hidden = self.input(style)
        if noise is None:
            _noise = torch.randn(1, 1, hidden.size(-1), hidden.size(-1))
        else:
            _noise = noise[0]
        out['noise'].append(_noise)
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=_noise)
        img = self.to_rgb1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out['rgb'].append(img)
        for i, m in enumerate(self.layers):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            if noise is None:
                _noise = torch.randn(*shape)
            else:
                _noise = noise[i + 1]
            out['noise'].append(_noise)
            _style = style if style.ndim == 2 else style[:, 3 * i + 1:3 * i + 4, :]
            hidden, rgb = m(hidden, style, _noise)
            out['rgb'].append(rgb)
            img = self.upsample(img) + rgb
        out['img'] = img
        return out

    def wsize(self):
        return len(self.layers) * 3 + 2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConstantInput,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EqualConv2d,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FusedLeakyReLU,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MappingNetwork,
     lambda: ([], {'style_dim': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModulatedConv2d,
     lambda: ([], {'channels_in': 4, 'channels_out': 4, 'style_dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (ModulatedDWConv2d,
     lambda: ([], {'channels_in': 4, 'channels_out': 4, 'style_dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (MultichannelIamge,
     lambda: ([], {'channels_in': 4, 'channels_out': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (NoiseInjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PerceptualLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (PerceptualNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyledConv2d,
     lambda: ([], {'channels_in': 4, 'channels_out': 4, 'style_dim': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_bes_dev_MobileStyleGAN_pytorch(_paritybench_base):
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

