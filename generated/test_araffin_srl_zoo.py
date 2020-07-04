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


import time


import numpy as np


import torch as th


import torch.nn as nn


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torchvision.models as models


from torch.autograd import Function


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


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish states from two different episodes
    :input_dim: (int) input_dim = 2 * state_dim
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=
            True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 
            1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_araffin_srl_zoo(_paritybench_base):
    pass
    def test_000(self):
        self._check(Discriminator(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(GaussianNoiseVariant(*[], **{'device': 0, 'std': 4}), [torch.rand([4, 4, 4, 4])], {})

