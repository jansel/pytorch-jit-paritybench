import sys
_module = sys.modules[__name__]
del sys
blitz = _module
bayesian_LeNet_mnist = _module
bayesian_regression_boston = _module
cifar10_bvgg = _module
losses = _module
kl_divergence = _module
tests = _module
kl_divergence_test = _module
models = _module
b_vgg = _module
modules = _module
base_bayesian_module = _module
conv_bayesian_layer = _module
embedding_bayesian_layer = _module
gru_bayesian_layer = _module
linear_bayesian_layer = _module
lstm_bayesian_layer = _module
base_bayesian_module_test = _module
conv_bayesian_layer_test = _module
embadding_bayesian_test = _module
gru_bayesian_layer_test = _module
linear_bayesian_layer_test = _module
lstm_bayesian_layer_test = _module
weight_sampler_test = _module
weight_sampler = _module
utils = _module
minibatch_weighting = _module
variational_estimator_test = _module
variational_estimator = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torchvision.datasets as dsets


import torchvision.transforms as transforms


import numpy as np


from sklearn.datasets import load_boston


from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.datasets as datasets


import torch.nn


import math


import torch.nn.init as init


from torch import nn


from torch.nn import functional as F


import torch.functional as F


class BayesianModule(nn.Module):
    """
    creates base class for BNN, in order to enable specific behavior
    """

    def init(self):
        super().__init__()


class GaussianVariational(nn.Module):

    def __init__(self, mu, rho):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean

        sets those weights as the current ones

        returns:
            torch.tensor with same shape as self.mu and self.rho
        """
        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self):
        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        assert self.w is not None, "You can only have a log posterior for W if you've already sampled it"
        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))
        log_posteriors = -log_sqrt2pi - torch.log(self.sigma) - (self.w - self.mu) ** 2 / (2 * self.sigma ** 2)
        return log_posteriors.mean()


class ScaleMixturePrior(nn.Module):

    def __init__(self, pi=1, sigma1=0.1, sigma2=0.001, dist=None):
        super().__init__()
        if dist is None:
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)
        if dist is not None:
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))
        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        if self.dist2 is None:
            prob_n2 = 0
        prior_pdf = self.pi * prob_n1 + (1 - self.pi) * prob_n2
        return torch.log(prior_pdf).mean()


class BayesianConv2d(BayesianModule):
    """
    Bayesian Linear layer, implements a Convolution 2D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, stride=1, padding=0, dilation=1, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        if self.freeze:
            return self.forward_frozen(x)
        w = self.weight_sampler.sample()
        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = torch.zeros(self.out_channels)
            b_log_posterior = 0
            b_log_prior = 0
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior
        return F.conv2d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward_frozen(self, x):
        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, 'The bias inputed should be this layer parameter, not a clone.'
        else:
            bias = torch.zeros(self.out_channels)
        return F.conv2d(input=x, weight=self.weight_mu, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_features, out_features, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.4, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        if self.freeze:
            return self.forward_frozen(x)
        w = self.weight_sampler.sample()
        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = torch.zeros(self.out_features)
            b_log_posterior = 0
            b_log_prior = 0
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior
        return F.linear(x, w, b)

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))


def kl_divergence_from_nn(model):
    """
    Gathers the KL Divergence from a nn.Module object
    Works by gathering each Bayesian layer kl divergence and summing it, doing nothing with the non Bayesian ones
    """
    kl_divergence = 0
    for module in model.modules():
        if isinstance(module, BayesianModule):
            kl_divergence += module.log_variational_posterior - module.log_prior
    return kl_divergence


def variational_estimator(nn_class):
    """
    This decorator adds some util methods to a nn.Module, in order to facilitate the handling of Bayesian Deep Learning features

    Parameters:
        nn_class: torch.nn.Module -> Torch neural network module

    Returns a nn.Module with methods for:
        (1) Gathering the KL Divergence along its BayesianModules;
        (2) Sample the Elbo Loss along its variational inferences (helps training)
        (3) Freeze the model, in order to predict using only their weight distribution means
        (4) Specifying the variational parameters by using some prior weights after training the NN as a deterministic model
    """

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights

            Parameters:
                N/a

            Returns torch.tensor with 0 dim.      
        
        """
        return kl_divergence_from_nn(self)
    setattr(nn_class, 'nn_kl_divergence', nn_kl_divergence)

    def sample_elbo(self, inputs, labels, criterion, sample_nbr, complexity_cost_weight=1):
        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels

                The ELBO Loss consists of the sum of the KL Divergence of the model 
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss). 

                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.

            Parameters:
                inputs: torch.tensor -> the input data to the model
                labels: torch.tensor -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                            the performance cost for the model
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to 
                            gather the loss to be .backwarded in the optimization of the model.        
        
        """
        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr
    setattr(nn_class, 'sample_elbo', sample_elbo)

    def freeze_model(self):
        """
        Freezes the model by making it predict using only the expected value to their BayesianModules' weights distributions
        """
        for module in self.modules():
            if isinstance(module, BayesianModule):
                module.freeze = True
    setattr(nn_class, 'freeze_', freeze_model)

    def unfreeze_model(self):
        """
        Unfreezes the model by letting it draw its weights with uncertanity from their correspondent distributions
        """
        for module in self.modules():
            if isinstance(module, BayesianModule):
                module.freeze = False
    setattr(nn_class, 'unfreeze_', unfreeze_model)

    def moped(self, delta=0.1):
        """
        Sets the sigma for the posterior distribution to delta * mu as proposed in

        @misc{krishnan2019specifying,
            title={Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes},
            author={Ranganath Krishnan and Mahesh Subedar and Omesh Tickoo},
            year={2019},
            eprint={1906.05323},
            archivePrefix={arXiv},
            primaryClass={cs.NE}
        }   


        """
        for module in self.modules():
            if isinstance(module, BayesianModule):
                for attr in module.modules():
                    if isinstance(attr, GaussianVariational):
                        attr.rho.data = torch.log(torch.expm1(delta * torch.abs(attr.mu.data)) + 1e-10)
        self.unfreeze_()
    setattr(nn_class, 'MOPED_', moped)

    def mfvi_forward(self, inputs, sample_nbr=10):
        """
        Performs mean-field variational inference for the variational estimator model:
            Performs sample_nbr forward passes with uncertainty on the weights, returning its mean and standard deviation

        Parameters:
            inputs: torch.tensor -> the input data to the model
            sample_nbr: int -> number of forward passes to be done on the data
        Returns:
            mean_: torch.tensor -> mean of the perdictions along each of the features of each datapoint on the batch axis
            std_: torch.tensor -> std of the predictions along each of the features of each datapoint on the batch axis


        """
        result = torch.stack([self(inputs) for _ in range(sample_nbr)])
        return result.mean(dim=0), result.std(dim=0)
    setattr(nn_class, 'mfvi_forward', mfvi_forward)
    return nn_class


@variational_estimator
class BayesianCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5, 5))
        self.conv2 = BayesianConv2d(6, 16, (5, 5))
        self.fc1 = BayesianLinear(256, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc3 = BayesianLinear(84, 10)

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


@variational_estimator
class BayesianRegressor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 512)
        self.blinear2 = BayesianLinear(512, output_dim)

    def forward(self, x):
        x_ = self.blinear1(x)
        return self.blinear2(x_)


@variational_estimator
class VGG(nn.Module):
    """
    VGG model 
    """

    def __init__(self, features, out_nodes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(BayesianLinear(512, 512), nn.ReLU(True), BayesianLinear(512, 512), nn.ReLU(True), BayesianLinear(512, out_nodes))
        for m in self.modules():
            if isinstance(m, BayesianConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mu.data.normal_(0, math.sqrt(2.0 / n))
                m.bias_mu.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BayesianConv1d(BayesianModule):
    """
    Bayesian Linear layer, implements a Convolution 1D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, stride=1, padding=0, dilation=1, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        if self.freeze:
            return self.forward_frozen(x)
        w = self.weight_sampler.sample()
        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = torch.zeros(self.out_channels)
            b_log_posterior = 0
            b_log_prior = 0
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior
        return F.conv1d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward_frozen(self, x):
        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, 'The bias inputed should be this layer parameter, not a clone.'
        else:
            bias = torch.zeros(self.out_channels)
        return F.conv1d(input=x, weight=self.weight_mu, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class BayesianConv3d(BayesianModule):
    """
    Bayesian Linear layer, implements a Convolution 3D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups=1, stride=1, padding=0, dilation=1, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        if self.freeze:
            return self.forward_frozen(x)
        w = self.weight_sampler.sample()
        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = torch.zeros(self.out_channels)
            b_log_posterior = 0
            b_log_prior = 0
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior
        return F.conv3d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward_frozen(self, x):
        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, 'The bias inputed should be this layer parameter, not a clone.'
        else:
            bias = torch.zeros(self.out_channels)
        return F.conv3d(input=x, weight=self.weight_mu, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class BayesianEmbedding(BayesianModule):
    """
    Bayesian Embedding layer, implements the embedding layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        num_embedding int -> Size of the vocabulary
        embedding_dim int -> Dimension of the embedding
        prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
        prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
        prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
        freeze -> wheter the model is instaced as frozen (will use deterministic weights on the feedforward op)
        padding_idx int -> If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index
        max_norm float -> If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
        norm_type float -> The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq -> If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
        sparse bool -> If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init

    
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.freeze = freeze
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight_mu = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        if self.freeze:
            return self.forward_frozen(x)
        w = self.weight_sampler.sample()
        self.log_variational_posterior = self.weight_sampler.log_posterior()
        self.log_prior = self.weight_prior_dist.log_prior(w)
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def forward_frozen(self, x):
        return F.embedding(x, self.weight_mu, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class BayesianGRU(BayesianModule):
    """
    Bayesian GRU layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_features, out_features, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = GaussianVariational(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = GaussianVariational(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.bias = None
        self.weight_ih_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.weight_hh_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_weights(self):
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = 0
            b_log_posterior = 0
            b_log_prior = 0
        bias = b
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_log_posterior + self.weight_ih_sampler.log_posterior()
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh)
        return weight_ih, weight_hh, bias

    def get_frozen_weights(self):
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias = self.bias_mu
        else:
            bias = 0
        return weight_ih, weight_hh, bias

    def forward_(self, x, hidden_states):
        weight_ih, weight_hh, bias = self.sample_weights()
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if hidden_states is None:
            h_t = torch.zeros(bs, self.out_features)
        else:
            h_t = hidden_states
        HS = self.out_features
        hidden_seq = []
        for t in range(seq_sz):
            x_t = x[:, (t), :]
            A_t = x_t @ weight_ih[:, :HS * 2] + h_t @ weight_hh[:, :HS * 2] + bias[:HS * 2]
            r_t, z_t = torch.sigmoid(A_t[:, :HS]), torch.sigmoid(A_t[:, HS:HS * 2])
            n_t = torch.tanh(x_t @ weight_ih[:, HS * 2:HS * 3] + bias[HS * 2:HS * 3] + r_t * (h_t @ weight_hh[:, HS * 3:HS * 4] + bias[HS * 3:HS * 4]))
            h_t = (1 - z_t) * n_t + z_t * h_t
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, h_t

    def forward_frozen(self, x, hidden_states):
        weight_ih, weight_hh, bias = self.get_frozen_weights()
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if hidden_states is None:
            h_t = torch.zeros(bs, self.out_features)
        else:
            h_t = hidden_states
        HS = self.out_features
        hidden_seq = []
        for t in range(seq_sz):
            x_t = x[:, (t), :]
            A_t = x_t @ weight_ih[:, :HS * 2] + h_t @ weight_hh[:, :HS * 2] + bias[:HS * 2]
            r_t, z_t = torch.sigmoid(A_t[:, :HS]), torch.sigmoid(A_t[:, HS:HS * 2])
            n_t = torch.tanh(x_t @ weight_ih[:, HS * 2:HS * 3] + bias[HS * 2:HS * 3] + r_t * (h_t @ weight_hh[:, HS * 3:HS * 4] + bias[HS * 3:HS * 4]))
            h_t = (1 - z_t) * n_t + z_t * h_t
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, h_t

    def forward(self, x, hidden_states=None):
        if self.freeze:
            return self.forward_frozen(x, hidden_states)
        return self.forward_(x, hidden_states)


class BayesianLSTM(BayesianModule):
    """
    Bayesian LSTM layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """

    def __init__(self, in_features, out_features, bias=True, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1, posterior_mu_init=0, posterior_rho_init=-6.0, freeze=False, prior_dist=None, peephole=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze
        self.peephole = peephole
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.weight_ih_mu = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_ih_rho = nn.Parameter(torch.Tensor(in_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_ih_sampler = GaussianVariational(self.weight_ih_mu, self.weight_ih_rho)
        self.weight_ih = None
        self.weight_hh_mu = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_mu_init, 0.1))
        self.weight_hh_rho = nn.Parameter(torch.Tensor(out_features, out_features * 4).normal_(posterior_rho_init, 0.1))
        self.weight_hh_sampler = GaussianVariational(self.weight_hh_mu, self.weight_hh_rho)
        self.weight_hh = None
        self.bias_mu = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)
        self.bias = None
        self.weight_ih_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.weight_hh_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_weights(self):
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = 0
            b_log_posterior = 0
            b_log_prior = 0
        bias = b
        self.log_variational_posterior = self.weight_hh_sampler.log_posterior() + b_log_posterior + self.weight_ih_sampler.log_posterior()
        self.log_prior = self.weight_ih_prior_dist.log_prior(weight_ih) + b_log_prior + self.weight_hh_prior_dist.log_prior(weight_hh)
        return weight_ih, weight_hh, bias

    def get_frozen_weights(self):
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias = self.bias_mu
        else:
            bias = 0
        return weight_ih, weight_hh, bias

    def forward_(self, x, hidden_states):
        weight_ih, weight_hh, bias = self.sample_weights()
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if hidden_states is None:
            h_t, c_t = torch.zeros(bs, self.out_features), torch.zeros(bs, self.out_features)
        else:
            h_t, c_t = hidden_states
        HS = self.out_features
        hidden_seq = []
        for t in range(seq_sz):
            x_t = x[:, (t), :]
            if self.peephole:
                gates = x_t @ weight_ih + c_t @ weight_hh + bias
            else:
                gates = x_t @ weight_ih + h_t @ weight_hh + bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            i_t, f_t, o_t = torch.sigmoid(gates[:, :HS]), torch.sigmoid(gates[:, HS:HS * 2]), torch.sigmoid(gates[:, HS * 3:])
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ weight_ih + bias)[:, HS * 2:HS * 3]
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def forward_frozen(self, x, hidden_states):
        weight_ih, weight_hh, bias = self.get_frozen_weights()
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if hidden_states is None:
            h_t, c_t = torch.zeros(bs, self.out_features), torch.zeros(bs, self.out_features)
        else:
            h_t, c_t = hidden_states
        HS = self.out_features
        hidden_seq = []
        for t in range(seq_sz):
            x_t = x[:, (t), :]
            if self.peephole:
                gates = x_t @ weight_ih + c_t @ weight_hh + bias
            else:
                gates = x_t @ weight_ih + h_t @ weight_hh + bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            i_t, f_t, o_t = torch.sigmoid(gates[:, :HS]), torch.sigmoid(gates[:, HS:HS * 2]), torch.sigmoid(gates[:, HS * 3:])
            if self.peephole:
                c_t = f_t * c_t + i_t * torch.sigmoid(x_t @ weight_ih + bias)[:, HS * 2:HS * 3]
                h_t = torch.sigmoid(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def forward(self, x, hidden_states=None):
        if self.freeze:
            return self.forward_frozen(x, hidden_states)
        return self.forward_(x, hidden_states)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BayesianConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (BayesianEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (BayesianGRU,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BayesianLSTM,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BayesianLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BayesianRegressor,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG,
     lambda: ([], {'features': _mock_layer()}),
     lambda: ([torch.rand([512, 512])], {}),
     False),
]

class Test_piEsposito_blitz_bayesian_deep_learning(_paritybench_base):
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

