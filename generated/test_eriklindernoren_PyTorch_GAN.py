import sys
_module = sys.modules[__name__]
del sys
aae = _module
acgan = _module
began = _module
bgan = _module
bicyclegan = _module
datasets = _module
models = _module
ccgan = _module
datasets = _module
models = _module
cgan = _module
clustergan = _module
cogan = _module
mnistm = _module
context_encoder = _module
datasets = _module
models = _module
cyclegan = _module
datasets = _module
models = _module
utils = _module
dcgan = _module
datasets = _module
discogan = _module
models = _module
dragan = _module
datasets = _module
dualgan = _module
models = _module
ebgan = _module
datasets = _module
esrgan = _module
models = _module
test_on_image = _module
gan = _module
infogan = _module
lsgan = _module
datasets = _module
models = _module
munit = _module
datasets = _module
models = _module
pix2pix = _module
mnistm = _module
pixelda = _module
relativistic_gan = _module
sgan = _module
softmax_gan = _module
datasets = _module
models = _module
srgan = _module
datasets = _module
models = _module
stargan = _module
datasets = _module
models = _module
unit = _module
wgan = _module
wgan_div = _module
wgan_gp = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import math


import itertools


import torchvision.transforms as transforms


from torchvision.utils import save_image


from torch.utils.data import DataLoader


from torchvision import datasets


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import torch


import time


import random


from torch.utils.data import Dataset


from torchvision.models import resnet18


import scipy


import torch.utils.data as data


from torchvision.utils import make_grid


import torch.autograd as autograd


from torchvision.models import vgg19


from torchvision import transforms


class ResidualBlock(nn.Module):

    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), nn.InstanceNorm2d(features), nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), nn.InstanceNorm2d(features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Encoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(dim * 2), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(3):
            layers += [ResidualBlock(dim)]
        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        Tensor = torch.FloatTensor if mu.is_cuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, mu.shape)))
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x)
        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ='relu'):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Decoder(nn.Module):

    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()
        layers = []
        dim = dim * 2 ** n_upsample
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm='adain')]
        for _ in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2), nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2), LayerNorm(dim // 2), nn.ReLU(inplace=True)]
            dim = dim // 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, content_code, style_code):
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(opt.latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class UNetDown(nn.Module):

    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class MultiDiscriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module('disc_%d' % i, nn.Sequential(*discriminator_block(in_channels, 64, normalize=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512), nn.Conv2d(512, 1, 3, padding=1)))
        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return 'shape={}'.format(self.shape)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """

    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()
        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = 128, 7, 7
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        self.model = nn.Sequential(torch.nn.Linear(self.latent_dim + self.n_c, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, self.iels), nn.BatchNorm1d(self.iels), nn.LeakyReLU(0.2, inplace=True), Reshape(self.ishape), nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True), nn.Sigmoid())
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


def softmax(x):
    return F.softmax(x, dim=1)


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = 128, 5, 5
        self.iels = int(np.prod(self.cshape))
        self.lshape = self.iels,
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), Reshape(self.lshape), torch.nn.Linear(self.iels, 1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, latent_dim + n_c))
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """

    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = 128, 5, 5
        self.iels = int(np.prod(self.cshape))
        self.lshape = self.iels,
        self.wass = wass_metric
        self.verbose = verbose
        self.model = nn.Sequential(nn.Conv2d(self.channels, 64, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, stride=2, bias=True), nn.LeakyReLU(0.2, inplace=True), Reshape(self.lshape), torch.nn.Linear(self.iels, 1024), nn.LeakyReLU(0.2, inplace=True), torch.nn.Linear(1024, 1))
        if not self.wass:
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
        initialize_weights(self)
        if self.verbose:
            None
            None

    def forward(self, img):
        validity = self.model(img)
        return validity


class CoupledGenerators(nn.Module):

    def __init__(self):
        super(CoupledGenerators, self).__init__()
        self.init_size = opt.img_size // 4
        self.fc = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        self.shared_conv = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2))
        self.G1 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())
        self.G2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):

    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block
        self.shared_conv = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        ds_size = opt.img_size // 2 ** 4
        self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)
        return validity1, validity2


class GeneratorResNet(nn.Module):

    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape
        model = [nn.Conv2d(channels + c_dim, 64, 7, stride=1, padding=3, bias=False), nn.InstanceNorm2d(64, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        curr_dim = 64
        for _ in range(2):
            model += [nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
            curr_dim *= 2
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]
        for _ in range(2):
            model += [nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
            curr_dim = curr_dim // 2
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, c), 1)
        return self.model(x)


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, out_channels, 4, padding=1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):

    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters))

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):

    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), nn.PixelShuffle(upscale_factor=2)]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(), nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class ContentEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(dim * 2), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm='in')]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True)]
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2
        for _ in range(n_downsample - 2):
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, 'Please assign weight and bias before calling AdaIN!'
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, h, w)
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers
        self.model = nn.Sequential(*block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512))
        input_size = opt.img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DenseResidualBlock,
     lambda: ([], {'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (GeneratorRRDB,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GeneratorResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64]), torch.rand([4, 4, 1, 1])], {}),
     True),
    (GeneratorUNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MultiDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Reshape,
     lambda: ([], {}),
     lambda: ([torch.rand([4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualInResidualDenseBlock,
     lambda: ([], {'filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StyleEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (UNetDown,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNetUp,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {}),
     True),
]

class Test_eriklindernoren_PyTorch_GAN(_paritybench_base):
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

