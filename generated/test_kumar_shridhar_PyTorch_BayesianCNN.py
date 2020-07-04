import sys
_module = sys.modules[__name__]
del sys
config_mixtures = _module
gmm = _module
main = _module
mixture_experiment = _module
temp_gmm = _module
train_splitted = _module
utils_mixture = _module
master = _module
config_bayesian = _module
config_frequentist = _module
data = _module
BBBConv = _module
BBBLinear = _module
BBB = _module
BBBConv = _module
BBBLinear = _module
BBB_LRT = _module
layers = _module
misc = _module
main_bayesian = _module
main_frequentist = _module
metrics = _module
Bayesian3Conv3FC = _module
BayesianAlexNet = _module
BayesianLeNet = _module
AlexNet = _module
LeNet = _module
ThreeConvThreeFC = _module
test_models = _module
uncertainty_estimation = _module
utils = _module

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


import numpy as np


from math import pi


import torch.nn as nn


from torch.nn import functional as F


import math


import torch.nn.functional as F


from torch.nn import Parameter


from torch import nn


from torch.optim import Adam


from torch.optim import lr_scheduler


import torchvision


import torchvision.transforms as transforms


class GaussianMixture(torch.nn.Module):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data. Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, k: number of components, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self, n_components, n_features, mu_init=None, var_init=
        None, eps=1e-06):
        """
        Initializes the model and brings all tensors into their required shape. The class expects data to be fed as a flat tensor in (n, d). The class owns:
            x:              torch.Tensor (n, k, d)
            mu:             torch.Tensor (1, k, d)
            var:            torch.Tensor (1, k, d)
            pi:             torch.Tensor (1, k, 1)
            eps:            float
            n_components:   int
            n_features:     int
            score:          float
        args:
            n_components:   int
            n_features:     int
            mu_init:        torch.Tensor (1, k, d)
            var_init:       torch.Tensor (1, k, d)
            eps:            float
        """
        super(GaussianMixture, self).__init__()
        self.eps = eps
        self.n_components = n_components
        self.n_features = n_features
        self.log_likelihood = -np.inf
        self.mu_init = mu_init
        self.var_init = var_init
        self._init_params()

    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.
                n_features
                ), 'Input mu_init does not have required tensor dimensions (1, %i, %i)' % (
                self.n_components, self.n_features)
            self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
        else:
            self.mu = torch.nn.Parameter(torch.randn(1, self.n_components,
                self.n_features), requires_grad=False)
        if self.var_init is not None:
            assert self.var_init.size() == (1, self.n_components, self.
                n_features
                ), 'Input var_init does not have required tensor dimensions (1, %i, %i)' % (
                self.n_components, self.n_features)
            self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
        else:
            self.var = torch.nn.Parameter(torch.ones(1, self.n_components,
                self.n_features), requires_grad=False)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1),
            requires_grad=False).fill_(1.0 / self.n_components)
        self.params_fitted = False

    def bic(self, x):
        """
        Bayesian information criterion for samples x.
        args:
            x:      torch.Tensor (n, d) or (n, k, d)
        returns:
            bic:    float
        """
        n = x.shape[0]
        if len(x.size()) == 2:
            x = x.unsqueeze(1).expand(n, self.n_components, x.size(1))
        bic = -2.0 * self.__score(self.pi, self.__p_k(x, self.mu, self.var),
            sum_data=True) * n + self.n_components * np.log(n)
        return bic

    def fit(self, x, warm_start=False, delta=1e-08, n_iter=1000):
        """
        Public method that fits data to the model.
        args:
            n_iter:     int
            delta:      float
        """
        if not warm_start and self.params_fitted:
            self._init_params()
        if len(x.size()) == 2:
            x = x.unsqueeze(1).expand(x.size(0), self.n_components, x.size(1))
        i = 0
        j = np.inf
        while i <= n_iter and j >= delta:
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var
            self.__em(x)
            self.log_likelihood = self.__score(self.pi, self.__p_k(x, self.
                mu, self.var))
            if self.log_likelihood.abs() == float('Inf'
                ) or self.log_likelihood == float('nan'):
                self.__init__(self.n_components, self.n_features)
            i += 1
            j = self.log_likelihood - log_likelihood_old
            if j <= delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)
        self.params_fitted = True

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each. If probs=True returns normalized probabilities of class membership instead.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
            probs:      bool
        returns:
            y:          torch.LongTensor (n)
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(1).expand(x.size(0), self.n_components, x.size(1))
        p_k = self.__p_k(x, self.mu, self.var)
        if probs:
            return p_k / (p_k.sum(1, keepdim=True) + self.eps)
        else:
            _, predictions = torch.max(p_k, 1)
            return torch.squeeze(predictions).type(torch.LongTensor)

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        returns:
            y:          torch.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def score_samples(self, x):
        """
        Computes log-likelihood of data (x) under the current model.
        args:
            x:          torch.Tensor (n, d) or (n, k, d)
        returns:
            score:      torch.LongTensor (n)
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(1).expand(x.size(0), self.n_components, x.size(1))
        score = self.__score(self.pi, self.__p_k(x, self.mu, self.var),
            sum_data=False)
        return score

    def __p_k(self, x, mu, var):
        """
        Returns a tensor with dimensions (n, k, 1) indicating the likelihood of data belonging to the k-th Gaussian.
        args:
            x:      torch.Tensor (n, k, d)
            mu:     torch.Tensor (1, k, d)
            var:    torch.Tensor (1, k, d)
        returns:
            p_k:    torch.Tensor (n, k, 1)
        """
        mu = mu.expand(x.size(0), self.n_components, self.n_features)
        var = var.expand(x.size(0), self.n_components, self.n_features)
        exponent = torch.exp(-0.5 * torch.sum((x - mu) * (x - mu) / var, 2,
            keepdim=True))
        prefactor = torch.rsqrt((2.0 * pi) ** self.n_features * torch.prod(
            var, dim=2, keepdim=True) + self.eps)
        return prefactor * exponent

    def __e_step(self, pi, p_k):
        """
        Computes weights that indicate the probabilistic belief that a data point was generated by one of the k mixture components. This is the so-called expectation step of the EM-algorithm.
        args:
            pi:         torch.Tensor (1, k, 1)
            p_k:        torch.Tensor (n, k, 1)
        returns:
            weights:    torch.Tensor (n, k, 1)
        """
        weights = pi * p_k
        return torch.div(weights, torch.sum(weights, 1, keepdim=True) +
            self.eps)

    def __m_step(self, x, weights):
        """
        Updates the model's parameters. This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, k, d)
            weights:    torch.Tensor (n, k, 1)
        returns:
            pi_new:     torch.Tensor (1, k, 1)
            mu_new:     torch.Tensor (1, k, d)
            var_new:    torch.Tensor (1, k, d)
        """
        n_k = torch.sum(weights, 0, keepdim=True)
        pi_new = torch.div(n_k, torch.sum(n_k, 1, keepdim=True) + self.eps)
        mu_new = torch.div(torch.sum(weights * x, 0, keepdim=True), n_k +
            self.eps)
        var_new = torch.div(torch.sum(weights * (x - mu_new) * (x - mu_new),
            0, keepdim=True), n_k + self.eps)
        return pi_new, mu_new, var_new

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, k, d)
        """
        weights = self.__e_step(self.pi, self.__p_k(x, self.mu, self.var))
        pi_new, mu_new, var_new = self.__m_step(x, weights)
        self.__update_pi(pi_new)
        self.__update_mu(mu_new)
        self.__update_var(var_new)

    def __score(self, pi, p_k, sum_data=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            pi:         torch.Tensor (1, k, 1)
            p_k:        torch.Tensor (n, k, 1)
        """
        weights = pi * p_k
        if sum_data:
            return torch.sum(torch.log(torch.sum(weights, 1) + self.eps))
        else:
            return torch.log(torch.sum(weights, 1) + self.eps)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.n_components, self.n_features), (1, self
            .n_components, self.n_features)
            ], 'Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)' % (
            self.n_components, self.n_features, self.n_components, self.
            n_features)
        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        assert var.size() in [(self.n_components, self.n_features), (1,
            self.n_components, self.n_features)
            ], 'Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)' % (
            self.n_components, self.n_features, self.n_components, self.
            n_features)
        if var.size() == (self.n_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features):
            self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [(1, self.n_components, 1)
            ], 'Input pi does not have required tensor dimensions (%i, %i, %i)' % (
            1, self.n_components, 1)
        self.pi.data = pi


class Pass(nn.Module):

    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        return x, kl


class ELBO(nn.Module):

    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        return F.nll_loss(input, target, reduction='mean'
            ) * self.train_size + beta * kl


class AlexNet(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(inputs, 64, kernel_size=11,
            stride=4, padding=5), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 192,
            kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=2, stride=2), nn.Conv2d(192, 384, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Conv2d
            (384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn
            .Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=
            True), nn.Dropout(p=0.5), nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):

    def __init__(self, num_classes, inputs=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class ThreeConvThreeFC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """

    def __init__(self, outputs, inputs):
        super(ThreeConvThreeFC, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(inputs, 32, 5, stride=1,
            padding=2), nn.Softplus(), nn.MaxPool2d(kernel_size=3, stride=2
            ), nn.Conv2d(32, 64, 5, stride=1, padding=2), nn.Softplus(), nn
            .MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 128, 5,
            stride=1, padding=1), nn.Softplus(), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.classifier = nn.Sequential(FlattenLayer(2 * 2 * 128), nn.
            Linear(2 * 2 * 128, 1000), nn.Softplus(), nn.Linear(1000, 1000),
            nn.Softplus(), nn.Linear(1000, outputs))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kumar_shridhar_PyTorch_BayesianCNN(_paritybench_base):
    pass
    def test_000(self):
        self._check(FlattenLayer(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ModuleWrapper(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Pass(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ThreeConvThreeFC(*[], **{'outputs': 4, 'inputs': 4}), [torch.rand([4, 4, 64, 64])], {})

