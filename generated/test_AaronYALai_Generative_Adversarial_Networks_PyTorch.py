import sys
_module = sys.modules[__name__]
del sys
DCGAN = _module
run_DCGAN = _module
GAN = _module
run_GAN = _module
ImprovedGAN = _module
run_ImprovedGAN = _module
InfoGAN = _module
run_InfoGAN = _module
LAPGAN = _module
run_LAPGAN = _module
tests = _module
test_GANs = _module

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


import torch.nn.functional as F


import torch


import torch.optim as optim


import torchvision


import torchvision.transforms as transforms


import numpy as np


from torch.autograd import Variable


class DCGAN_Discriminator(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=1):
        super(DCGAN_Discriminator, self).__init__()
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, featmap_dim / 4, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(featmap_dim / 4, featmap_dim / 2, 5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv3 = nn.Conv2d(featmap_dim / 2, featmap_dim, 5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)
        self.fc = nn.Linear(featmap_dim * 4 * 4, 1)

    def forward(self, x):
        """
        Strided convulation layers,
        Batch Normalization after convulation but not at input layer,
        LeakyReLU activation function with slope 0.2.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = F.sigmoid(self.fc(x))
        return x


class DCGAN_Generator(nn.Module):

    def __init__(self, featmap_dim=1024, n_channel=1, noise_dim=100):
        super(DCGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, featmap_dim / 2, 5, stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim / 2, featmap_dim / 4, 6, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim / 4, n_channel, 6, stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 256)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.fc1(x)))
        x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1 * 28 * 28)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x


class ImprovedGAN_Discriminator(nn.Module):

    def __init__(self, featmap_dim=512, n_channel=1, use_gpu=False, n_B=128, n_C=16):
        """
        Minibatch discrimination: learn a tensor to encode side information
        from other examples in the same minibatch.
        """
        super(ImprovedGAN_Discriminator, self).__init__()
        self.use_gpu = use_gpu
        self.n_B = n_B
        self.n_C = n_C
        self.featmap_dim = featmap_dim
        self.conv1 = nn.Conv2d(n_channel, featmap_dim / 4, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(featmap_dim / 4, featmap_dim / 2, 5, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv3 = nn.Conv2d(featmap_dim / 2, featmap_dim, 5, stride=2, padding=2)
        self.BN3 = nn.BatchNorm2d(featmap_dim)
        T_ten_init = torch.randn(featmap_dim * 4 * 4, n_B * n_C) * 0.1
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)
        self.fc = nn.Linear(featmap_dim * 4 * 4 + n_B, 1)

    def forward(self, x):
        """
        Architecture is similar to DCGANs
        Add minibatch discrimination => Improved GAN.
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        T_tensor = self.T_tensor
        if self.use_gpu:
            T_tensor = T_tensor
        Ms = x.mm(T_tensor)
        Ms = Ms.view(-1, self.n_B, self.n_C)
        out_tensor = []
        for i in range(Ms.size()[0]):
            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i
            out_tensor.append(out_i)
        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.n_B)
        x = torch.cat((x, out_T), 1)
        x = F.sigmoid(self.fc(x))
        return x


class ImprovedGAN_Generator(nn.Module):

    def __init__(self, featmap_dim=1024, n_channel=1, noise_dim=100):
        super(ImprovedGAN_Generator, self).__init__()
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * featmap_dim)
        self.conv1 = nn.ConvTranspose2d(featmap_dim, featmap_dim / 2, 5, stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(featmap_dim / 2)
        self.conv2 = nn.ConvTranspose2d(featmap_dim / 2, featmap_dim / 4, 6, stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(featmap_dim / 4)
        self.conv3 = nn.ConvTranspose2d(featmap_dim / 4, n_channel, 6, stride=2, padding=2)

    def forward(self, x):
        """
        Project noise to featureMap * width * height,
        Batch Normalization after convulation but not at output layer,
        ReLU activation function.
        """
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.tanh(self.conv3(x))
        return x


class InfoGAN_Discriminator(nn.Module):

    def __init__(self, n_layer=3, n_conti=2, n_discrete=1, num_category=10, use_gpu=False, featmap_dim=256, n_channel=1):
        """
        InfoGAN Discriminator, have additional outputs for latent codes.
        Architecture brought from DCGAN.
        """
        super(InfoGAN_Discriminator, self).__init__()
        self.n_layer = n_layer
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category
        self.featmap_dim = featmap_dim
        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == self.n_layer - 1:
                n_conv_in = n_channel
            else:
                n_conv_in = int(featmap_dim / 2 ** (layer + 1))
            n_conv_out = int(featmap_dim / 2 ** layer)
            _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5, stride=2, padding=2)
            if use_gpu:
                _conv = _conv
            convs.append(_conv)
            if layer != self.n_layer - 1:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN
                BNs.append(_BN)
        n_hidden = featmap_dim * 4 * 4
        n_output = 1 + n_conti + n_discrete * num_category
        self.fc = nn.Linear(n_hidden, n_output)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)

    def forward(self, x):
        """
        Output the probability of being in real dataset
        plus the conditional distributions of latent codes.
        """
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == 0:
                x = F.leaky_relu(conv_layer(x), negative_slope=0.2)
            else:
                BN_layer = self.BNs[self.n_layer - layer - 1]
                x = F.leaky_relu(BN_layer(conv_layer(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        x = self.fc(x)
        x[:, 0] = F.sigmoid(x[:, 0].clone())
        for j in range(self.n_discrete):
            start = 1 + self.n_conti + j * self.num_category
            end = start + self.num_category
            x[:, start:end] = F.softmax(x[:, start:end].clone())
        return x


class InfoGAN_Generator(nn.Module):

    def __init__(self, noise_dim=10, n_layer=3, n_conti=2, n_discrete=1, num_category=10, use_gpu=False, featmap_dim=256, n_channel=1):
        """
        InfoGAN Generator, have an additional input branch for latent codes.
        Architecture brought from DCGAN.
        """
        super(InfoGAN_Generator, self).__init__()
        self.n_layer = n_layer
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category
        n_input = noise_dim + n_conti + n_discrete * num_category
        self.featmap_dim = featmap_dim
        self.fc_in = nn.Linear(n_input, featmap_dim * 4 * 4)
        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == 0:
                n_conv_out = n_channel
            else:
                n_conv_out = featmap_dim / 2 ** (self.n_layer - layer)
            n_conv_in = featmap_dim / 2 ** (self.n_layer - layer - 1)
            n_width = 5 if layer == self.n_layer - 1 else 6
            _conv = nn.ConvTranspose2d(n_conv_in, n_conv_out, n_width, stride=2, padding=2)
            if use_gpu:
                _conv = _conv
            convs.append(_conv)
            if layer != 0:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN
                BNs.append(_BN)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)

    def forward(self, x):
        """
        Input the random noise plus latent codes to generate fake images.
        """
        x = self.fc_in(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == self.n_layer - 1:
                x = F.tanh(conv_layer(x))
            else:
                BN_layer = self.BNs[self.n_layer - layer - 2]
                x = F.relu(BN_layer(conv_layer(x)))
        return x


class CondiGAN_Discriminator(nn.Module):

    def __init__(self, n_layer=3, condition=True, n_condition=100, use_gpu=False, featmap_dim=256, n_channel=1, condi_featmap_dim=256):
        """
        Conditional Discriminator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Discriminator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition
        self.featmap_dim = featmap_dim
        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == self.n_layer - 1:
                n_conv_in = n_channel
            else:
                n_conv_in = int(featmap_dim / 2 ** (layer + 1))
            n_conv_out = int(featmap_dim / 2 ** layer)
            _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5, stride=2, padding=2)
            if use_gpu:
                _conv = _conv
            convs.append(_conv)
            if layer != self.n_layer - 1:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN
                BNs.append(_BN)
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim
            convs_condi = []
            BNs_condi = []
            for layer in range(self.n_layer):
                if layer == self.n_layer - 1:
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / 2 ** (layer + 1))
                n_conv_out = int(condi_featmap_dim / 2 ** layer)
                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5, stride=2, padding=2)
                if use_gpu:
                    _conv = _conv
                convs_condi.append(_conv)
                if layer != self.n_layer - 1:
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN
                    BNs_condi.append(_BN)
            self.fc_c = nn.Linear(condi_featmap_dim * 4 * 4, n_condition)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        if self.condition:
            self.convs_condi = nn.ModuleList(convs_condi)
            self.BNs_condi = nn.ModuleList(BNs_condi)
        n_hidden = featmap_dim * 4 * 4
        if self.condition:
            n_hidden += n_condition
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the last layer
        """
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == 0:
                x = F.leaky_relu(conv_layer(x), negative_slope=0.2)
            else:
                BN_layer = self.BNs[self.n_layer - layer - 1]
                x = F.leaky_relu(BN_layer(conv_layer(x)), negative_slope=0.2)
        x = x.view(-1, self.featmap_dim * 4 * 4)
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)), negative_slope=0.2)
            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            condi_x = self.fc_c(condi_x)
            x = torch.cat((x, condi_x), 1)
        x = F.sigmoid(self.fc(x))
        return x


class CondiGAN_Generator(nn.Module):

    def __init__(self, noise_dim=10, n_layer=3, condition=True, n_condition=100, use_gpu=False, featmap_dim=256, n_channel=1, condi_featmap_dim=256):
        """
        Conditional Generator.
        Architecture brought from DCGAN.
        """
        super(CondiGAN_Generator, self).__init__()
        self.n_layer = n_layer
        self.condition = condition
        if self.condition:
            self.condi_featmap_dim = condi_featmap_dim
            convs_condi = []
            BNs_condi = []
            for layer in range(self.n_layer):
                if layer == self.n_layer - 1:
                    n_conv_in = n_channel
                else:
                    n_conv_in = int(condi_featmap_dim / 2 ** (layer + 1))
                n_conv_out = int(condi_featmap_dim / 2 ** layer)
                _conv = nn.Conv2d(n_conv_in, n_conv_out, kernel_size=5, stride=2, padding=2)
                if use_gpu:
                    _conv = _conv
                convs_condi.append(_conv)
                if layer != self.n_layer - 1:
                    _BN = nn.BatchNorm2d(n_conv_out)
                    if use_gpu:
                        _BN = _BN
                    BNs_condi.append(_BN)
            self.fc_c = nn.Linear(condi_featmap_dim * 4 * 4, n_condition)
        n_input = noise_dim
        if self.condition:
            n_input += n_condition
        self.featmap_dim = featmap_dim
        self.fc1 = nn.Linear(n_input, int(featmap_dim * 4 * 4))
        convs = []
        BNs = []
        for layer in range(self.n_layer):
            if layer == 0:
                n_conv_out = n_channel
            else:
                n_conv_out = featmap_dim / 2 ** (self.n_layer - layer)
            n_conv_in = featmap_dim / 2 ** (self.n_layer - layer - 1)
            n_width = 5 if layer == self.n_layer - 1 else 6
            _conv = nn.ConvTranspose2d(n_conv_in, n_conv_out, n_width, stride=2, padding=2)
            if use_gpu:
                _conv = _conv
            convs.append(_conv)
            if layer != 0:
                _BN = nn.BatchNorm2d(n_conv_out)
                if use_gpu:
                    _BN = _BN
                BNs.append(_BN)
        self.convs = nn.ModuleList(convs)
        self.BNs = nn.ModuleList(BNs)
        if self.condition:
            self.convs_condi = nn.ModuleList(convs_condi)
            self.BNs_condi = nn.ModuleList(BNs_condi)

    def forward(self, x, condi_x=None):
        """
        Concatenate CNN-processed extra information vector at the first layer
        """
        if self.condition:
            for layer in range(self.n_layer):
                _conv = self.convs_condi[self.n_layer - layer - 1]
                if layer == 0:
                    condi_x = F.leaky_relu(_conv(condi_x), negative_slope=0.2)
                else:
                    BN_layer = self.BNs_condi[self.n_layer - layer - 1]
                    condi_x = F.leaky_relu(BN_layer(_conv(condi_x)), negative_slope=0.2)
            condi_x = condi_x.view(-1, self.condi_featmap_dim * 4 * 4)
            condi_x = self.fc_c(condi_x)
            x = torch.cat((x, condi_x), 1)
        x = self.fc1(x)
        x = x.view(-1, self.featmap_dim, 4, 4)
        for layer in range(self.n_layer):
            conv_layer = self.convs[self.n_layer - layer - 1]
            if layer == self.n_layer - 1:
                x = F.tanh(conv_layer(x))
            else:
                BN_layer = self.BNs[self.n_layer - layer - 2]
                x = F.relu(BN_layer(conv_layer(x)))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (InfoGAN_Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
]

class Test_AaronYALai_Generative_Adversarial_Networks_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

