import sys
_module = sys.modules[__name__]
del sys
GAN = _module
base = _module
util = _module
fid_score = _module
inception = _module
inception_score = _module
interactive = _module
cifar10 = _module
mnist = _module
main = _module
mugan = _module
base = _module
model_resnet = _module
model_resnet_bnd = _module
model_resnet_nosigm = _module
model_resnet_old = _module
model_resnet_old_specnorm = _module
model_resnet_ppresnet = _module
model_resnet_preproc = _module
model_resnet_specnorm = _module
model_resnet_specnorm_nosigm = _module
model_resnet_specnorm_preproc = _module
models_64x64 = _module
spectral_normalization = _module
image = _module
spectral_norm = _module
task_launcher = _module
fid = _module
utils = _module
affine_transforms = _module
analytical_helper_script = _module
helpers = _module
load_data = _module
logger = _module
lr_scheduler = _module
main = _module
models = _module
caffe_cifar = _module
densenet = _module
imagenet_resnet = _module
preact_resnet_temp = _module
preresnet = _module
res_utils = _module
resnet = _module
resnet_old = _module
resnext = _module
wide_resnet = _module
plots = _module

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


import torch


import time


import numpy as np


import torch.nn as nn


import torch.optim as optim


from torch.autograd import Variable


from torch.autograd import grad


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from collections import OrderedDict


from torch import optim


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torch.nn.functional as F


from torchvision import models


from torch import nn


from torch.nn import functional as F


import torch.utils.data


from torchvision.models.inception import inception_v3


from scipy.stats import entropy


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


from torch import Tensor


from torch.nn import Parameter


from torch.nn.functional import normalize


from torch.nn.parameter import Parameter


import scipy.misc


import random


import torch.backends.cudnn as cudnn


import torchvision.datasets as dset


import torchvision.transforms as transforms


from collections import Counter


from torch.nn import init


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.init as init


class generator(nn.Module):

    def __init__(self, dataset='mnist'):
        super(generator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 62
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 62
            self.output_dim = 3
        self.fc = nn.Sequential(nn.Linear(self.input_dim, 1024), nn.
            BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 128 * (self.
            input_height // 4) * (self.input_width // 4)), nn.BatchNorm1d(
            128 * (self.input_height // 4) * (self.input_width // 4)), nn.
            ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.ConvTranspose2d(64, self.
            output_dim, 4, 2, 1), nn.Sigmoid())
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, self.input_height // 4, self.input_width // 4)
        x = self.deconv(x)
        return x


class discriminator(nn.Module):

    def __init__(self, dataset='mnist'):
        super(discriminator, self).__init__()
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1
        elif dataset == 'celebA':
            self.input_height = 64
            self.input_width = 64
            self.input_dim = 3
            self.output_dim = 1
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(
            128), nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(nn.Linear(128 * (self.input_height // 4) *
            (self.input_width // 4), 1024), nn.BatchNorm1d(1024), nn.
            LeakyReLU(0.2), nn.Linear(1024, self.output_dim), nn.Sigmoid())
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4)
            )
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=
        True, normalize_input=True, requires_grad=False):
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
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.
                MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.
                Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception
                .Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.
                Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
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
            x = F.upsample(x, size=(299, 299), mode='bilinear')
        if self.normalize_input:
            x = x.clone()
            x[:, (0)] = x[:, (0)] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, (1)] = x[:, (1)] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, (2)] = x[:, (2)] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


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


class discriminator(nn.Module):

    def __init__(self, input_width, input_height, input_dim, output_dim,
        out_nonlinearity=None):
        super(discriminator, self).__init__()
        assert out_nonlinearity in [None, 'sigmoid']
        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(
            128), nn.LeakyReLU(0.2))
        self.fc = [nn.Linear(128 * (self.input_height // 4) * (self.
            input_width // 4), 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(
            0.2), nn.Linear(1024, self.output_dim)]
        if out_nonlinearity == 'sigmoid':
            self.fc += [nn.Sigmoid()]
        self.fc = nn.Sequential(*self.fc)
        initialize_weights(self)

    def forward(self, input):
        """Returns a list of outputs where the last one is D(x)
        and others are hidden states"""
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4)
            )
        preconv = x
        x = self.fc(x)
        return preconv, x

    def partial_forward(self, preconv):
        return self.fc(preconv)


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Upsample(scale_factor=2), self.conv1, nn.BatchNorm2d(
            out_channels), nn.ReLU(), self.conv2)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class SpectralNorm(object):

    def __init__(self, name='weight', n_power_iterations=1, eps=1e-12):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_org')
        u = getattr(module, self.name + '_u')
        height = weight.size(0)
        weight_mat = weight.view(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=
                    self.eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
            sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = module._parameters[self.name + '_org']
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_org')
        module.register_parameter(self.name, weight)

    def __call__(self, module, inputs):
        weight, u = self.compute_weight(module)
        setattr(module, self.name, weight)
        with torch.no_grad():
            getattr(module, self.name).copy_(weight)

    @staticmethod
    def apply(module, name, n_power_iterations, eps):
        fn = SpectralNorm(name, n_power_iterations, eps)
        weight = module._parameters[name]
        height = weight.size(0)
        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps
            )
        delattr(module, fn.name)
        module.register_parameter(fn.name + '_org', weight)
        module.register_buffer(fn.name, weight)
        module.register_buffer(fn.name + '_u', u)
        module.register_forward_pre_hook(fn)
        return fn


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        if stride == 1:
            self.model = nn.Sequential(nn.ReLU(), self.spec_norm(self.conv1
                ), nn.ReLU(), self.spec_norm(self.conv2))
        else:
            self.model = nn.Sequential(nn.ReLU(), self.spec_norm(self.conv1
                ), nn.ReLU(), self.spec_norm(self.conv2), nn.AvgPool2d(2,
                stride=stride, padding=0))
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1,
                padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(self.spec_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0))
        else:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0
            )
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.model = nn.Sequential(self.spec_norm(self.conv1), nn.ReLU(),
            self.spec_norm(self.conv2), nn.AvgPool2d(2))
        self.bypass = nn.Sequential(nn.AvgPool2d(2), self.spec_norm(self.
            bypass_conv))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


GEN_SIZE = 128


channels = 3


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)
        self.model = nn.Sequential(ResBlockGenerator(GEN_SIZE, GEN_SIZE,
            stride=2), ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), nn.BatchNorm2d
            (GEN_SIZE), nn.ReLU(), self.final, nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))


DISC_SIZE = 128


class Discriminator(nn.Module):

    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self._init = FirstResBlockDiscriminator(channels, DISC_SIZE, stride
            =2, spec_norm=spec_norm)
        self._init2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2,
            spec_norm=spec_norm)
        self.model = nn.Sequential(ResBlockDiscriminator(DISC_SIZE,
            DISC_SIZE, spec_norm=spec_norm), ResBlockDiscriminator(
            DISC_SIZE, DISC_SIZE, spec_norm=spec_norm), nn.ReLU(), nn.
            AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        """
        Return a tuple of intermediate states, and also
          the final output.
        """
        init = self._init(x)
        init2 = self._init2(init)
        return (init, init2), self.partial_forward(init2, 1)

    def partial_forward(self, hs, idx):
        """
        Compute the output of the discriminator, given
          either the result of the first or second layer.
        """
        assert idx in [0, 1]
        if idx == 0:
            result = self.fc(self.model(self._init2(hs)).view(-1, DISC_SIZE))
        else:
            result = self.fc(self.model(hs).view(-1, DISC_SIZE))
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Upsample(scale_factor=2), self.conv1, nn.BatchNorm2d(
            out_channels), nn.ReLU(), self.conv2)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        if stride == 1:
            self.model = nn.Sequential(nn.ReLU(), self.conv1, nn.
                BatchNorm2d(out_channels), nn.ReLU(), self.conv2, nn.
                BatchNorm2d(out_channels))
        else:
            self.model = nn.Sequential(nn.ReLU(), self.conv1, nn.
                BatchNorm2d(out_channels), nn.ReLU(), self.conv2, nn.
                BatchNorm2d(out_channels), nn.AvgPool2d(2, stride=stride,
                padding=0))
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1,
                padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(self.bypass_conv, nn.AvgPool2d(2,
                stride=stride, padding=0))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0
            )
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        self.model = nn.Sequential(self.conv1, nn.BatchNorm2d(out_channels),
            nn.ReLU(), self.conv2, nn.BatchNorm2d(out_channels), nn.
            AvgPool2d(2))
        self.bypass = nn.Sequential(nn.AvgPool2d(2), self.bypass_conv)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)
        self.model = nn.Sequential(ResBlockGenerator(GEN_SIZE, GEN_SIZE,
            stride=2), ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), nn.BatchNorm2d
            (GEN_SIZE), nn.ReLU(), self.final, nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))


class Discriminator(nn.Module):

    def __init__(self, sigmoid=False):
        super(Discriminator, self).__init__()
        self._init = FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2)
        self._init2 = ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2)
        self.model = nn.Sequential(ResBlockDiscriminator(DISC_SIZE,
            DISC_SIZE), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE), nn.
            ReLU(), nn.AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        """
        Return a tuple of intermediate states, and also
          the final output.
        """
        init = self._init(x)
        init2 = self._init2(init)
        return (init, init2), self.partial_forward(init2, 1)

    def partial_forward(self, hs, idx):
        """
        Compute the output of the discriminator, given
          either the result of the first or second layer.
        """
        assert idx in [0, 1]
        if idx == 0:
            result = self.fc(self.model(self._init2(hs)).view(-1, DISC_SIZE))
        else:
            result = self.fc(self.model(hs).view(-1, DISC_SIZE))
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Upsample(scale_factor=2), self.conv1, nn.BatchNorm2d(
            out_channels), nn.ReLU(), self.conv2)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        if stride == 1:
            self.model = nn.Sequential(nn.ReLU(), self.spec_norm(self.conv1
                ), nn.ReLU(), self.spec_norm(self.conv2))
        else:
            self.model = nn.Sequential(nn.ReLU(), self.spec_norm(self.conv1
                ), nn.ReLU(), self.spec_norm(self.conv2), nn.AvgPool2d(2,
                stride=stride, padding=0))
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1,
                padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
            self.bypass = nn.Sequential(self.spec_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0
            )
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.model = nn.Sequential(self.spec_norm(self.conv1), nn.ReLU(),
            self.spec_norm(self.conv2), nn.AvgPool2d(2))
        self.bypass = nn.Sequential(nn.AvgPool2d(2), self.spec_norm(self.
            bypass_conv))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)
        self.model = nn.Sequential(ResBlockGenerator(GEN_SIZE, GEN_SIZE,
            stride=2), ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2), nn.BatchNorm2d
            (GEN_SIZE), nn.ReLU(), self.final, nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))


class Discriminator(nn.Module):

    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.model = nn.Sequential(FirstResBlockDiscriminator(channels,
            DISC_SIZE, stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2, spec_norm
            =spec_norm), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE,
            spec_norm=spec_norm), ResBlockDiscriminator(DISC_SIZE,
            DISC_SIZE, spec_norm=spec_norm), nn.ReLU(), nn.AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        pre_fc = self.model(x).view(-1, DISC_SIZE)
        result = self.fc(pre_fc)
        if self.sigmoid:
            return pre_fc, self.sigm(result)
        else:
            return pre_fc, result

    def partial_forward(self, x):
        if self.sigmoid:
            return self.sigm(self.fc(x))
        else:
            return self.fc(x)


class Discriminator(nn.Module):
    """
    This discriminator differs from the one in model_resnet
      in that we have a preprocessor conv right before the
      main model.
    """

    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.preproc = ResBlockDiscriminator(3, 16, stride=1, spec_norm=
            spec_norm)
        self.model = nn.Sequential(FirstResBlockDiscriminator(16, DISC_SIZE,
            stride=2, spec_norm=spec_norm), ResBlockDiscriminator(DISC_SIZE,
            DISC_SIZE, stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, spec_norm=spec_norm
            ), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, spec_norm=
            spec_norm), nn.ReLU(), nn.AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        preproc = self.preproc(x)
        return preproc, self.partial_forward(preproc)

    def partial_forward(self, preproc, idx=-1):
        pre_fc = self.model(preproc).view(-1, DISC_SIZE)
        result = self.fc(pre_fc)
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result


class Discriminator(nn.Module):
    """
    This discriminator differs from the one in model_resnet
      in that we have a preprocessor conv right before the
      main model.
    """

    def __init__(self, spec_norm=False, sigmoid=False):
        super(Discriminator, self).__init__()
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.preproc = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.model = nn.Sequential(FirstResBlockDiscriminator(16, DISC_SIZE,
            stride=2, spec_norm=spec_norm), ResBlockDiscriminator(DISC_SIZE,
            DISC_SIZE, stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, spec_norm=spec_norm
            ), ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, spec_norm=
            spec_norm), nn.ReLU(), nn.AvgPool2d(8))
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = self.spec_norm(self.fc)
        if sigmoid:
            self.sigm = nn.Sigmoid()
        self.sigmoid = sigmoid

    def forward(self, x):
        preproc = self.preproc(x)
        return preproc, self.partial_forward(preproc)

    def partial_forward(self, preproc, idx=-1):
        pre_fc = self.model(preproc).view(-1, DISC_SIZE)
        result = self.fc(pre_fc)
        if self.sigmoid:
            return self.sigm(result)
        else:
            return result


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
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Generator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                padding=2, output_padding=1, bias=False), nn.BatchNorm2d(
                out_dim), nn.ReLU())
        self.l1 = nn.Sequential(nn.Linear(in_dim, dim * 8 * 4 * 4, bias=
            False), nn.BatchNorm1d(dim * 8 * 4 * 4), nn.ReLU())
        self.l2_5 = nn.Sequential(dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2), nn.ConvTranspose2d(dim * 2, 3,
            5, 2, padding=2, output_padding=1), nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, 5, 2, 2), nn.
                BatchNorm2d(out_dim), nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.
            LeakyReLU(0.2), conv_bn_lrelu(dim, dim * 2), conv_bn_lrelu(dim *
            2, dim * 4), conv_bn_lrelu(dim * 4, dim * 8), nn.Conv2d(dim * 8,
            1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class DiscriminatorWGANGP(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, 5, 2, 2), nn.
                InstanceNorm2d(out_dim, affine=True), nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.
            LeakyReLU(0.2), conv_ln_lrelu(dim, dim * 2), conv_ln_lrelu(dim *
            2, dim * 4), nn.Conv2d(dim * 4, 1, 4), nn.Sigmoid())

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1, 1)
        return None, y


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
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data),
                u.data))
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


class CifarCaffeNet(nn.Module):

    def __init__(self, num_classes):
        super(CifarCaffeNet, self).__init__()
        self.num_classes = num_classes
        self.block_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride
            =1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(
            ), nn.BatchNorm2d(32))
        self.block_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3,
            stride=1, padding=1), nn.Conv2d(32, 64, kernel_size=3, stride=1,
            padding=1), nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64))
        self.block_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,
            stride=1, padding=1), nn.Conv2d(64, 128, kernel_size=3, stride=
            1, padding=1), nn.ReLU(), nn.AvgPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128))
        self.classifier = nn.Linear(128 * 9, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block_1.forward(x)
        x = self.block_2.forward(x)
        x = self.block_3.forward(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias
            =False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        if bottleneck:
            nDenseBlocks = int((depth - 4) / 6)
        else:
            nDenseBlocks = int((depth - 4) / 3)
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias
            =False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def mixup_data(x, y, alpha):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[(index), :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def per_image_standardization(x):
    y = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    mean = y.mean(dim=1, keepdim=True).expand_as(y)
    std = y.std(dim=1, keepdim=True).expand_as(y)
    adjusted_std = torch.max(std, 1.0 / torch.sqrt(torch.cuda.FloatTensor([
        x.shape[1] * x.shape[2] * x.shape[3]])))
    y = (y - mean) / adjusted_std
    standarized_input = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return standarized_input


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, initial_channels, num_classes,
        per_img_std=False):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[
            0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels * 2,
            num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels * 4,
            num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels * 8,
            num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels * 8 * block.expansion,
            num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=0.1,
        layer_mix=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        if mixup_hidden == True:
            if layer_mix == None:
                layer_mix = random.randint(0, 2)
            out = x
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.conv1(x)
            out = self.layer1(out)
            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer2(out)
            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer3(out)
            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer4(out)
            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            lam = torch.tensor(lam)
            lam = lam.repeat(y_a.size())
            return out, y_a, y_b, lam
        else:
            out = x
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        """
        if layer_mix == 'rand':
            if self.mixup_hidden:
                layer_mix = random.randint(0,2)
            else:
                layer_mix = 0

        out = x

        if lam is not None:
            lam = torch.max(lam, 1-lam)
            if target_reweighted is None:
                target_reweighted = to_one_hot(target,self.num_classes)
            else:
                assert target is None
            if layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = self.layer1(out)

        if lam is not None and layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer2(out)

        if lam is not None and layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if lam is None:
            return out
        else:
            return out, target_reweighted
	"""


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def get_lambda(alpha=1.0):
    """Return lambda"""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    return lam


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (
        1 - lam)
    return out, target_reweighted


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    return y_onehot


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, initial_channels, num_classes,
        per_img_std=False, stride=1):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[
            0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels * 2,
            num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels * 4,
            num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels * 8,
            num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels * 8 * block.expansion,
            num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target=None, mixup=False, mixup_hidden=False,
        mixup_alpha=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None
        out = x
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32'))
            lam = Variable(lam)
        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.conv1(out)
        out = self.layer1(out)
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.layer2(out)
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.layer3(out)
        if layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if target is not None:
            return out, target_reweighted
        else:
            return out


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride,
            padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride,
            padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, per_img_std=False):
        super(ResNet, self).__init__()
        self.per_img_std = per_img_std
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=0.1,
        layer_mix=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        if mixup_hidden == True:
            if layer_mix == None:
                layer_mix = random.randint(0, 2)
            out = x
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer2(out)
            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer3(out)
            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.layer4(out)
            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            lam = torch.tensor(lam)
            lam = lam.repeat(y_a.size())
            return out, y_a, y_b, lam
        else:
            out = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, block, depth, num_classes, dropout):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()
        assert (depth - 2
            ) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        None
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding
            =1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.
                expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1,
        downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride
            =1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)
        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=
            stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D * C)
        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """

    def __init__(self, block, depth, cardinality, base_width, num_classes,
        dropout, per_img_std=False):
        super(CifarResNeXt, self).__init__()
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        assert (depth - 2
            ) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.
            base_width, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality,
                self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup=False, mixup_hidden=False,
        mixup_alpha=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None
        out = x
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32'))
            lam = Variable(lam)
        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.conv_1_3x3(out)
        out = F.relu(self.bn_1(out), inplace=True)
        out = self.stage_1(out)
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.stage_2(out)
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.stage_3(out)
        if layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        if target is not None:
            return out, target_reweighted
        else:
            return out


act = torch.nn.ReLU()


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
            bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.conv1(act(self.bn1(x)))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, per_img_std=False,
        stride=1):
        super(Wide_ResNet, self).__init__()
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        None
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0], stride=stride)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    """
    ## Modified WRN architecture###
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        #self.mixup_hidden = mixup_hidden

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    """

    def forward(self, x, target=None, mixup=False, mixup_hidden=False,
        mixup_alpha=None):
        if self.per_img_std:
            x = per_image_standardization(x)
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None
        out = x
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32'))
            lam = Variable(lam)
        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.conv1(out)
        out = self.layer1(out)
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.layer2(out)
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = self.layer3(out)
        if layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted,
                lam=lam)
        out = act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if target is not None:
            return out, target_reweighted
        else:
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_vikasverma1077_manifold_mixup(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Discriminator(*[], **{'in_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_003(self):
        self._check(DiscriminatorWGANGP(*[], **{'in_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_004(self):
        self._check(DownsampleA(*[], **{'nIn': 4, 'nOut': 4, 'stride': 2}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(DownsampleC(*[], **{'nIn': 1, 'nOut': 4, 'stride': 1}), [torch.rand([4, 1, 64, 64])], {})

    def test_006(self):
        self._check(DownsampleD(*[], **{'nIn': 4, 'nOut': 4, 'stride': 2}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(FirstResBlockDiscriminator(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(Generator(*[], **{'in_dim': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(LayerNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(PreActBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(PreActBottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(ResBlockDiscriminator(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(ResNetBasicblock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(SingleLayer(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(Transition(*[], **{'nChannels': 4, 'nOutChannels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(discriminator(*[], **{'input_width': 4, 'input_height': 4, 'input_dim': 4, 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(generator(*[], **{}), [torch.rand([62, 62])], {})

    def test_018(self):
        self._check(wide_basic(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

