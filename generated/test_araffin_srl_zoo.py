import sys
_module = sys.modules[__name__]
del sys
conf = _module
evaluation = _module
enjoy_latent = _module
gather_results = _module
knn_images = _module
predict_dataset = _module
predict_reward = _module
losses = _module
losses = _module
utils = _module
models = _module
autoencoders = _module
custom_layers = _module
forward_inverse = _module
learner = _module
models = _module
modules = _module
priors = _module
supervised = _module
triplet = _module
vae = _module
pipeline = _module
plotting = _module
interactive_plot = _module
losses_plot = _module
representation_plot = _module
preprocessing = _module
data_loader = _module
preprocess = _module
server = _module
srl_baselines = _module
pca = _module
supervised = _module
tests = _module
common = _module
test_modules = _module
test_pipeline = _module
train = _module
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


from collections import OrderedDict


import numpy as np


import torch as th


from sklearn.neighbors import KNeighborsClassifier


import time


import torch.nn as nn


from sklearn.model_selection import train_test_split


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch


from collections import defaultdict


import torchvision.models as models


from torch.autograd import Function


import random


from torch.multiprocessing import Queue


from torch.multiprocessing import Process


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param device: (pytorch device)
    """

    def __init__(self, batch_size, input_dim, device, std, mean=0):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device
        self.noise = th.zeros(batch_size, input_dim, device=self.device)

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class GaussianNoiseVariant(nn.Module):
    """
    Variant of the Gaussian Noise layer that does not require fixed batch_size
    It recreates a tensor at each call
    :param device: (pytorch device)
    :param std: (float) standard deviation
    :param mean: (float)
    """

    def __init__(self, device, std, mean=0):
        super(GaussianNoiseVariant, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device

    def forward(self, x):
        if self.training:
            noise = th.zeros(x.size(), device=self.device)
            noise.data.normal_(self.mean, std=self.std)
            return x + noise
        return x


class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelSRL, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


N_CHANNELS = 3


def getNChannels():
    return N_CHANNELS


class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelAutoEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), conv3x3(in_planes=64, out_planes=64, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), conv3x3(in_planes=64, out_planes=64, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, getNChannels(), kernel_size=4, stride=2))

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.encode(x)
        decoded = self.decode(encoded).view(input_shape)
        return encoded, decoded


class BaseModelVAE(BaseModelAutoEncoder):
    """
    Base Class for a SRL network (VAE family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelVAE, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)[0]

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (th.Tensor)
        :param logvar: (th.Tensor)
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).view(input_shape)
        return decoded, mu, logvar


class CustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=2):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), conv3x3(in_planes=64, out_planes=64, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), conv3x3(in_planes=64, out_planes=64, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Linear(6 * 6 * 64, state_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SRLConvolutionalNetwork(BaseModelSRL):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, cuda=False, noise_std=1e-06):
        super(SRLConvolutionalNetwork, self).__init__()
        self.device = th.device('cuda' if th.cuda.is_available() and cuda else 'cpu')
        self.resnet = models.resnet18(pretrained=False)
        n_units = self.resnet.fc.in_features
        None
        self.resnet.fc = nn.Sequential(nn.Linear(n_units, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, state_dim))
        self.resnet = self.resnet
        self.noise = GaussianNoiseVariant(self.device, noise_std)

    def forward(self, x):
        x = self.resnet(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLCustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network for State Representation Learning
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, cuda=False, noise_std=1e-06):
        super(SRLCustomCNN, self).__init__()
        self.cnn = CustomCNN(state_dim)
        self.device = th.device('cuda' if th.cuda.is_available() and cuda else 'cpu')
        self.cnn = self.cnn
        self.noise = GaussianNoiseVariant(self.device, noise_std)

    def forward(self, x):
        x = self.cnn(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLDenseNetwork(BaseModelSRL):
    """
    Dense Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param state_dim: (int)
    :param noise_std: (float)  To avoid NaN (states must be different)
    :param cuda: (bool)
    :param n_hidden: (int)
    """

    def __init__(self, input_dim, state_dim=2, cuda=False, n_hidden=64, noise_std=1e-06):
        super(SRLDenseNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, state_dim))
        self.device = th.device('cuda' if th.cuda.is_available() and cuda else 'cpu')
        self.fc = self.fc
        self.noise = GaussianNoiseVariant(self.device, noise_std)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLLinear(BaseModelSRL):
    """
    Dense Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param state_dim: (int)
    :param cuda: (bool)
    """

    def __init__(self, input_dim, state_dim=2, cuda=False):
        super(SRLLinear, self).__init__()
        self.fc = nn.Linear(input_dim, state_dim)
        self.device = th.device('cuda' if th.cuda.is_available() and cuda else 'cpu')
        self.fc = self.fc

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish states from two different episodes
    :input_dim: (int) input_dim = 2 * state_dim
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianNoiseVariant,
     lambda: ([], {'device': 0, 'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SRLConvolutionalNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SRLDenseNetwork,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SRLLinear,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_araffin_srl_zoo(_paritybench_base):
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

