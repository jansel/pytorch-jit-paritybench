import sys
_module = sys.modules[__name__]
del sys
loss = _module
networks_gan = _module
networks_stylegan = _module
opts = _module
torchvision_sunner = _module
constant = _module
data = _module
base_dataset = _module
image_dataset = _module
loader = _module
video_dataset = _module
read = _module
transforms = _module
base = _module
categorical = _module
complex = _module
function = _module
simple = _module
utils = _module
train = _module
train_stylegan = _module

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


from torch.autograd import Variable


from torch.autograd import grad


import torch.autograd as autograd


import torch.nn as nn


import torch


import numpy as np


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn.init import kaiming_normal_


from torch.utils.data import DataLoader


import torch.utils.data as Data


import random


import math


from collections import Iterator


import torch.utils.data as data


import torchvision.transforms as transforms


from torchvision.utils import save_image


from matplotlib import pyplot as plt


import torch.optim as optim


from torch import nn


class Generator(nn.Module):

    def __init__(self, z_dims=512, d=64):
        super().__init__()
        self.deconv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(z_dims, d * 8, 4, 1, 0))
        self.deconv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1))
        self.deconv3 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1))
        self.deconv4 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1))
        self.deconv5 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 2, d, 4, 2, 1))
        self.deconv6 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv5(x))
        x = F.tanh(self.deconv6(x))
        return x


class Discriminator(nn.Module):

    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        self.layer6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        out = F.leaky_relu(self.layer1(input), 0.2, inplace=True)
        out = F.leaky_relu(self.layer2(out), 0.2, inplace=True)
        out = F.leaky_relu(self.layer3(out), 0.2, inplace=True)
        out = F.leaky_relu(self.layer4(out), 0.2, inplace=True)
        out = F.leaky_relu(self.layer5(out), 0.2, inplace=True)
        out = F.leaky_relu(self.layer6(out), 0.2, inplace=True)
        return out.view(-1, 1)


class ApplyNoise(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise


class FC(nn.Module):

    def __init__(self, in_channels, out_channels, gain=2 ** 0.5, use_wscale=False, lrmul=1.0, bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** -0.5
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """

    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, (0)] + 1.0) + style[:, (1)]
        return x


class Blur2d(nn.Module):

    def __init__(self, f=[1, 2, 1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, 'kernel f must be an instance of python built_in type list!'
        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, (None)] * f[(None), :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            kernel = self.f.expand(x.size(1), -1, -1, -1)
            x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.f.size(2) - 1) / 2), groups=x.size(1))
            return x
        else:
            return x


class Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** -0.5
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class PixelNorm(nn.Module):

    def __init__(self, epsilon=1e-08):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp = torch.mul(x, x)
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)
        return x * tmp1


class InstanceNorm(nn.Module):

    def __init__(self, epsilon=1e-08):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):

    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles):
        super(LayerEpilogue, self).__init__()
        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None
        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None
        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        return x


class GBlock(nn.Module):

    def __init__(self, res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, noise_input, dlatent_size=512, use_style=True, f=None, factor=2, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        self.res = res
        self.blur = Blur2d(f)
        self.noise_input = noise_input
        if res < 7:
            self.up_sample = Upscale2d(factor)
        else:
            self.up_sample = nn.ConvTranspose2d(self.nf(res - 3), self.nf(res - 2), 4, stride=2, padding=1)
        self.adaIn1 = LayerEpilogue(self.nf(res - 2), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_style)
        self.conv1 = Conv2d(input_channels=self.nf(res - 2), output_channels=self.nf(res - 2), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res - 2), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res * 2 - 4], dlatent[:, (self.res * 2 - 4)])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res * 2 - 3], dlatent[:, (self.res * 2 - 3)])
        return x


class G_mapping(nn.Module):

    def __init__(self, mapping_fmaps=512, dlatent_size=512, resolution=1024, normalize_latents=True, use_wscale=True, lrmul=0.01, gain=2 ** 0.5):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale), FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale))
        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):

    def __init__(self, dlatent_size, resolution=1024, fmap_base=8192, num_channels=3, structure='fixed', fmap_max=512, fmap_decay=1.0, f=None, use_pixel_norm=False, use_instance_norm=True, use_wscale=True, use_noise=True, use_style=True):
        """
            2019.3.31
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 x 1024.
        :param fmap_base:
        :param num_channels:
        :param structure: only support 'fixed' mode.
        :param fmap_max:
        """
        super(G_synthesis, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape))
        self.blur = Blur2d(f)
        self.channel_shrinkage = Conv2d(input_channels=self.nf(self.resolution_log2 - 2), output_channels=self.nf(self.resolution_log2), kernel_size=3, use_wscale=use_wscale)
        self.torgb = Conv2d(self.nf(self.resolution_log2), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_style)
        self.conv1 = Conv2d(input_channels=self.nf(1), output_channels=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_style)
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 8
        self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 9
        self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)
        res = 10
        self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm, self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        if self.structure == 'fixed':
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, (0)])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, (1)])
            x = self.GBlock1(x, dlatent)
            x = self.GBlock2(x, dlatent)
            x = self.GBlock3(x, dlatent)
            x = self.GBlock4(x, dlatent)
            x = self.GBlock5(x, dlatent)
            x = self.GBlock6(x, dlatent)
            x = self.GBlock7(x, dlatent)
            x = self.GBlock8(x, dlatent)
            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(nn.Module):

    def __init__(self, mapping_fmaps=512, style_mixing_prob=0.9, truncation_psi=0.7, truncation_cutoff=8, **kwargs):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, (i), :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """
            dlatents1 = dlatents1 * torch.Tensor(coefs)
        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):

    def __init__(self, resolution=1024, fmap_base=8192, num_channels=3, structure='fixed', fmap_max=512, fmap_decay=1.0, f=None):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / 2.0 ** (stage * fmap_decay)), fmap_max)
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.resolution_log2 - 1), kernel_size=1)
        self.structure = structure
        self.blur2d = Blur2d(f)
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2 - 5), self.nf(self.resolution_log2 - 5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2 - 6), self.nf(self.resolution_log2 - 6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2 - 7), self.nf(self.resolution_log2 - 7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2 - 8), self.nf(self.resolution_log2 - 8), kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(self.nf(self.resolution_log2 - 1), self.nf(self.resolution_log2 - 1), kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf(self.resolution_log2 - 1), self.nf(self.resolution_log2 - 2), kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(self.nf(self.resolution_log2 - 2), self.nf(self.resolution_log2 - 3), kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(self.nf(self.resolution_log2 - 3), self.nf(self.resolution_log2 - 4), kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(self.nf(self.resolution_log2 - 4), self.nf(self.resolution_log2 - 5), kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(self.nf(self.resolution_log2 - 5), self.nf(self.resolution_log2 - 6), kernel_size=3, padding=(1, 1))
        self.conv7 = nn.Conv2d(self.nf(self.resolution_log2 - 6), self.nf(self.resolution_log2 - 7), kernel_size=3, padding=(1, 1))
        self.conv8 = nn.Conv2d(self.nf(self.resolution_log2 - 7), self.nf(self.resolution_log2 - 8), kernel_size=3, padding=(1, 1))
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2 - 8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)
            res -= 1
            x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ApplyNoise,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ApplyStyle,
     lambda: ([], {'latent_size': 4, 'channels': 4, 'use_wscale': 1.0}),
     lambda: ([torch.rand([64, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Blur2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (FC,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 1, 1])], {}),
     True),
    (InstanceNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upscale2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tomguluson92_StyleGAN_PyTorch(_paritybench_base):
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

