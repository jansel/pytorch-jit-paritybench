import sys
_module = sys.modules[__name__]
del sys
conf = _module
pyprob = _module
address_dictionary = _module
concurrency = _module
diagnostics = _module
distributions = _module
bernoulli = _module
beta = _module
binomial = _module
categorical = _module
distribution = _module
empirical = _module
exponential = _module
gamma = _module
log_normal = _module
mixture = _module
normal = _module
poisson = _module
truncated_normal = _module
uniform = _module
weibull = _module
graph = _module
model = _module
nn = _module
dataset = _module
embedding_cnn_2d_5c = _module
embedding_cnn_3d_5c = _module
embedding_feedforward = _module
inference_network = _module
inference_network_feedforward = _module
inference_network_lstm = _module
optimizer_larc = _module
proposal_categorical_categorical = _module
proposal_normal_normal = _module
proposal_normal_normal_mixture = _module
proposal_poisson_truncated_normal_mixture = _module
proposal_uniform_beta = _module
proposal_uniform_beta_mixture = _module
proposal_uniform_truncated_normal_mixture = _module
Bernoulli = _module
Beta = _module
Binomial = _module
Categorical = _module
Distribution = _module
Exponential = _module
Gamma = _module
Handshake = _module
HandshakeResult = _module
LogNormal = _module
Message = _module
MessageBody = _module
Normal = _module
Observe = _module
ObserveResult = _module
Poisson = _module
Reset = _module
Run = _module
RunResult = _module
Sample = _module
SampleResult = _module
Tag = _module
TagResult = _module
Tensor = _module
Uniform = _module
Weibull = _module
ppx = _module
remote = _module
state = _module
trace = _module
util = _module
setup = _module
conftest = _module
gum_marsaglia = _module
gum_marsaglia = _module
rejection_sampling = _module
proposal_uniform_truncated_normal_mixture = _module
test_dataset = _module
test_diagnostics = _module
test_distributions = _module
test_distributions_remote = _module
test_inference = _module
test_inference_remote = _module
test_model = _module
test_model_remote = _module
test_nn = _module
test_state = _module
test_trace = _module
test_train = _module
test_util = _module

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


from collections import OrderedDict


from collections import defaultdict


import numpy as np


import matplotlib as mpl


import matplotlib.pyplot as plt


import time


import re


from torch.distributions.kl import kl_divergence


import copy


import collections


import random


import math


import enum


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


from torch.utils.data import Sampler


import torch.distributed as dist


import uuid


from collections import Counter


import torch.nn as nn


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.utils.data import DataLoader


from torch.optim import Optimizer


import inspect


from functools import reduce


import torch.multiprocessing


import torch.nn.functional as F


import functools


class EmbeddingCNN2D5C(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv2d(input_channels, 64, 3)
        self._conv2 = nn.Conv2d(64, 64, 3)
        self._conv3 = nn.Conv2d(64, 128, 3)
        self._conv4 = nn.Conv2d(128, 128, 3)
        self._conv5 = nn.Conv2d(128, 128, 3)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool2d(2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        x = torch.relu(self._lin2(x))
        return x.view(torch.Size([-1]) + self._output_shape)


class EmbeddingCNN3D5C(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv3d(input_channels, 64, 3)
        self._conv2 = nn.Conv3d(64, 64, 3)
        self._conv3 = nn.Conv3d(64, 128, 3)
        self._conv4 = nn.Conv3d(128, 128, 3)
        self._conv5 = nn.Conv3d(128, 128, 3)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        None
        None
        None

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool3d(2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool3d(2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        return x.view(torch.Size([-1]) + self._output_shape)


class EmbeddingFeedForward(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=3, activation=torch.relu, activation_last=torch.relu, input_is_one_hot_index=False, input_one_hot_dim=None):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        self._input_dim = util.prod(self._input_shape)
        self._output_dim = util.prod(self._output_shape)
        self._input_is_one_hot_index = input_is_one_hot_index
        self._input_one_hot_dim = input_one_hot_dim
        if input_is_one_hot_index:
            if self._input_dim != 1:
                raise ValueError('If input_is_one_hot_index==True, input_dim should be 1 (the index of one-hot value in a vector of length input_one_hot_dim.)')
            self._input_dim = input_one_hot_dim
        if num_layers < 1:
            raise ValueError('Expecting num_layers >= 1')
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(self._input_dim, self._output_dim))
        else:
            hidden_dim = int((self._input_dim + self._output_dim) / 2)
            layers.append(nn.Linear(self._input_dim, hidden_dim))
            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, self._output_dim))
        self._activation = activation
        self._activation_last = activation_last
        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        if self._input_is_one_hot_index:
            x = torch.stack([util.one_hot(self._input_one_hot_dim, int(v)) for v in x])
        else:
            x = x.view(-1, self._input_dim).float()
        for i in range(len(self._layers)):
            layer = self._layers[i]
            x = layer(x)
            if i == len(self._layers) - 1:
                if self._activation_last is not None:
                    x = self._activation_last(x)
            else:
                x = self._activation(x)
        return x.view(torch.Size([-1]) + self._output_shape)


class InferenceNetwork(enum.Enum):
    FEEDFORWARD = 0
    LSTM = 1


class Distribution:

    def __init__(self, name, address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size(), torch_dist=None):
        self.name = name
        self._address_suffix = address_suffix
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._torch_dist = torch_dist

    @property
    def batch_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.batch_shape
        else:
            return self._batch_shape

    @property
    def event_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.event_shape
        else:
            return self._event_shape

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        return torch.exp(self.log_prob(util.to_tensor(value)))

    def plot(self, min_val=-10, max_val=10, step_size=0.1, figsize=(10, 5), xlabel=None, ylabel='Probability', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, fig=None, *args, **kwargs):
        if fig is None:
            if not show:
                mpl.rcParams['axes.unicode_minus'] = False
                plt.switch_backend('agg')
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
        xvals = np.arange(min_val, max_val, step_size)
        plt.plot(xvals, [torch.exp(self.log_prob(x)) for x in xvals], *args, **kwargs)
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if file_name is not None:
            plt.savefig(file_name)
        if show:
            plt.show()

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            return self._torch_dist.variance
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    @staticmethod
    def kl_divergence(distribution_1, distribution_2):
        if distribution_1._torch_dist is None or distribution_2._torch_dist is None:
            raise ValueError('KL divergence is not currently supported for this pair of distributions.')
        return torch.distributions.kl.kl_divergence(distribution_1._torch_dist, distribution_2._torch_dist)


class Categorical(Distribution):

    def __init__(self, probs=None, logits=None):
        if probs is not None:
            probs = util.to_tensor(probs)
            if probs.dim() == 0:
                raise ValueError('probs cannot be a scalar.')
        if logits is not None:
            logits = util.to_tensor(logits)
            if logits.dim() == 0:
                raise ValueError('logits cannot be a scalar.')
        torch_dist = torch.distributions.Categorical(probs=probs, logits=logits)
        self._probs = torch_dist.probs
        self._logits = torch_dist.logits
        self._num_categories = self._probs.size(-1)
        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(self._probs.size(-1)), torch_dist=torch_dist)

    def __repr__(self):
        return 'Categorical(num_categories: {}, probs:{})'.format(self.num_categories, self.probs)

    @property
    def num_categories(self):
        return self._num_categories

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return self._logits


class Normal(Distribution):

    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.Normal(loc, scale))

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)


class Poisson(Distribution):

    def __init__(self, rate):
        rate = util.to_tensor(rate)
        super().__init__(name='Poisson', address_suffix='Poisson', torch_dist=torch.distributions.Poisson(rate))

    def __repr__(self):
        return 'Poisson(rate: {})'.format(self.rate)

    @property
    def rate(self):
        return self._torch_dist.mean


class ProposalCategoricalCategorical(nn.Module):

    def __init__(self, input_shape, num_categories, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([num_categories]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon
        return Categorical(probs)


class Mixture(Distribution):

    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length = len(distributions)
        if probs is None:
            self._probs = util.to_tensor(torch.zeros(self.length)).fill_(1.0 / self.length)
        else:
            self._probs = util.to_tensor(probs)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        self._log_probs = torch.log(util.clamp_probs(self._probs))
        event_shape = torch.Size()
        if self._probs.dim() == 1:
            batch_shape = torch.Size()
            self._batch_length = 0
        elif self._probs.dim() == 2:
            batch_shape = torch.Size([self._probs.size(0)])
            self._batch_length = self._probs.size(0)
        else:
            raise ValueError('Expecting a 1d or 2d (batched) mixture probabilities.')
        self._mixing_dist = Categorical(self._probs)
        self._mean = None
        self._variance = None
        super().__init__(name='Mixture', address_suffix='Mixture({})'.format(', '.join([d._address_suffix for d in self._distributions])), batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value, sum=False):
        if self._batch_length == 0:
            value = util.to_tensor(value).squeeze()
            lp = torch.logsumexp(self._log_probs + util.to_tensor([d.log_prob(value) for d in self._distributions]), dim=0)
        else:
            value = util.to_tensor(value).view(self._batch_length)
            lp = torch.logsumexp(self._log_probs + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).view(-1, self._batch_length).t(), dim=1)
        return torch.sum(lp) if sum else lp

    def sample(self):
        if self._batch_length == 0:
            i = int(self._mixing_dist.sample())
            return self._distributions[i].sample()
        else:
            indices = self._mixing_dist.sample()
            dist_samples = []
            for d in self._distributions:
                sample = d.sample()
                if sample.dim() == 0:
                    sample = sample.unsqueeze(-1)
                dist_samples.append(sample)
            ret = []
            for b in range(self._batch_length):
                i = int(indices[b])
                ret.append(dist_samples[i][b])
            return util.to_tensor(ret)

    @property
    def mean(self):
        if self._mean is None:
            means = torch.stack([d.mean for d in self._distributions])
            if self._batch_length == 0:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = torch.stack([((d.mean - self.mean).pow(2) + d.variance) for d in self._distributions])
            if self._batch_length == 0:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
        return self._variance


class ProposalNormalNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(batch_size, -1)
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(batch_size, -1)
        prior_means = prior_means.expand_as(means)
        prior_stddevs = prior_stddevs.expand_as(stddevs)
        means = prior_means + means * prior_stddevs
        stddevs = stddevs * prior_stddevs
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        distributions = [Normal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size)) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class TruncatedNormal(Distribution):

    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high, clamp_mean_between_low_high=False):
        self._mean_non_truncated = util.to_tensor(mean_non_truncated)
        self._stddev_non_truncated = util.to_tensor(stddev_non_truncated)
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        if clamp_mean_between_low_high:
            self._mean_non_truncated = torch.max(torch.min(self._mean_non_truncated, self._high), self._low)
        if self._mean_non_truncated.dim() == 0:
            self._batch_length = 0
        elif self._mean_non_truncated.dim() == 1 or self._mean_non_truncated.dim() == 2:
            self._batch_length = self._mean_non_truncated.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._standard_normal_dist = Normal(util.to_tensor(torch.zeros_like(self._mean_non_truncated)), util.to_tensor(torch.ones_like(self._stddev_non_truncated)))
        self._alpha = (self._low - self._mean_non_truncated) / self._stddev_non_truncated
        self._beta = (self._high - self._mean_non_truncated) / self._stddev_non_truncated
        self._standard_normal_cdf_alpha = self._standard_normal_dist.cdf(self._alpha)
        self._standard_normal_cdf_beta = self._standard_normal_dist.cdf(self._beta)
        self._Z = self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha
        self._log_stddev_Z = torch.log(self._stddev_non_truncated * self._Z)
        self._mean = None
        self._variance = None
        batch_shape = self._mean_non_truncated.size()
        event_shape = torch.Size()
        super().__init__(name='TruncatedNormal', address_suffix='TruncatedNormal', batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value, sum=False):
        value = util.to_tensor(value)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        lp = torch.log(lb.mul(ub)) + self._standard_normal_dist.log_prob((value - self._mean_non_truncated) / self._stddev_non_truncated) - self._log_stddev_Z
        if self._batch_length == 1:
            lp = lp.squeeze(0)
        if util.has_nan_or_inf(lp):
            None
            None
            None
            None
        return torch.sum(lp) if sum else lp

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def mean_non_truncated(self):
        return self._mean_non_truncated

    @property
    def stddev_non_truncated(self):
        return self._stddev_non_truncated

    @property
    def variance_non_truncated(self):
        return self._stddev_non_truncated.pow(2)

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._mean_non_truncated + self._stddev_non_truncated * (self._standard_normal_dist.prob(self._alpha) - self._standard_normal_dist.prob(self._beta)) / self._Z
            if self._batch_length == 1:
                self._mean = self._mean.squeeze(0)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            standard_normal_prob_alpha = self._standard_normal_dist.prob(self._alpha)
            standard_normal_prob_beta = self._standard_normal_dist.prob(self._beta)
            self._variance = self._stddev_non_truncated.pow(2) * (1 + (self._alpha * standard_normal_prob_alpha - self._beta * standard_normal_prob_beta) / self._Z - ((standard_normal_prob_alpha - standard_normal_prob_beta) / self._Z).pow(2))
            if self._batch_length == 1:
                self._variance = self._variance.squeeze(0)
        return self._variance

    def sample(self):
        shape = self._low.size()
        attempt_count = 0
        ret = util.to_tensor(torch.zeros(shape).fill_(float('NaN')))
        outside_domain = True
        while util.has_nan_or_inf(ret) or outside_domain:
            attempt_count += 1
            if attempt_count == 10000:
                None
            rand = util.to_tensor(torch.zeros(shape).uniform_())
            ret = self._standard_normal_dist.icdf(self._standard_normal_cdf_alpha + rand * (self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha)) * self._stddev_non_truncated + self._mean_non_truncated
            lb = ret.ge(self._low).type_as(self._low)
            ub = ret.lt(self._high).type_as(self._low)
            outside_domain = int(torch.sum(lb.mul(ub))) == 0
        if self._batch_length == 1:
            ret = ret.squeeze(0)
        return ret


class ProposalPoissonTruncatedNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, low=0, high=40, num_layers=2, mixture_components=10):
        super().__init__()
        self._low = low
        self._high = high
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        means = torch.sigmoid(means)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        prior_lows = torch.zeros(batch_size).fill_(self._low)
        prior_highs = torch.zeros(batch_size).fill_(self._high)
        means = prior_lows.view(batch_size, -1).expand_as(means) + means * (prior_highs - prior_lows).view(batch_size, -1).expand_as(means)
        distributions = [TruncatedNormal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class ProposalUniformTruncatedNormalMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        means = torch.sigmoid(means)
        stddevs = torch.sigmoid(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        prior_lows = torch.stack([util.to_tensor(v.distribution.low) for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([util.to_tensor(v.distribution.high) for v in prior_variables]).view(batch_size)
        prior_range = (prior_highs - prior_lows).view(batch_size, -1)
        means = prior_lows.view(batch_size, -1) + means * prior_range
        stddevs = prior_range / 1000 + stddevs * prior_range * 10
        distributions = [TruncatedNormal(means[:, i:i + 1].view(batch_size), stddevs[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class Uniform(Distribution):

    def __init__(self, low, high):
        low = util.to_tensor(low)
        high = util.to_tensor(high)
        super().__init__(name='Uniform', address_suffix='Uniform', torch_dist=torch.distributions.Uniform(low, high))

    def __repr__(self):
        return 'Uniform(low: {}, high: {})'.format(self.low, self.high)

    @property
    def low(self):
        return self._torch_dist.low

    @property
    def high(self):
        return self._torch_dist.high


class ProposalNormalNormal(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._output_dim].view(batch_size, -1)
        stddevs = torch.exp(x[:, self._output_dim:]).view(batch_size, -1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(means.size())
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(stddevs.size())
        means = prior_means + means * prior_stddevs
        stddevs = stddevs * prior_stddevs
        means = means.view(self._output_shape)
        stddevs = stddevs.view(self._output_shape)
        return Normal(means, stddevs)


class Beta(Distribution):

    def __init__(self, concentration1, concentration0, low=0, high=1):
        concentration1 = util.to_tensor(concentration1)
        concentration0 = util.to_tensor(concentration0)
        super().__init__(name='Beta', address_suffix='Beta', torch_dist=torch.distributions.Beta(concentration1, concentration0))
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        self._range = self._high - self._low

    def __repr__(self):
        return 'Beta(concentration1:{}, concentration0:{}, low:{}, high:{})'.format(self.concentration1, self.concentration0, self.low, self.high)

    @property
    def concentration1(self):
        return self._torch_dist.concentration1

    @property
    def concentration0(self):
        return self._torch_dist.concentration0

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def sample(self):
        return self._low + super().sample() * self._range

    def log_prob(self, value, sum=False):
        lp = super().log_prob((util.to_tensor(value) - self._low) / self._range, sum=False)
        return torch.sum(lp) if sum else lp

    @property
    def mean(self):
        return self._low + super().mean * self._range

    @property
    def variance(self):
        return super().variance * self._range * self._range


class ProposalUniformBeta(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=torch.relu)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        x = self._ff(x)
        concentration1s = 1.0 + x[:, :self._output_dim].view(self._output_shape)
        concentration0s = 1.0 + x[:, self._output_dim:].view(self._output_shape)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(concentration1s.size())
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(concentration1s.size())
        return Beta(concentration1s, concentration0s, low=prior_lows, high=prior_highs)


class ProposalUniformBetaMixture(nn.Module):

    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        concentration1s = x[:, :self._mixture_components].view(batch_size, -1)
        concentration0s = x[:, self._mixture_components:2 * self._mixture_components].view(batch_size, -1)
        concentration1s = 1.0 + torch.relu(concentration1s)
        concentration0s = 1.0 + torch.relu(concentration0s)
        coeffs = x[:, 2 * self._mixture_components:].view(batch_size, -1)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(batch_size)
        distributions = [Beta(concentration1s[:, i:i + 1].view(batch_size), concentration0s[:, i:i + 1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)

