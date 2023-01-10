import sys
_module = sys.modules[__name__]
del sys
ctabgan = _module
evaluation = _module
data_preparation = _module
rdp_accountant = _module
ctabgan_synthesizer = _module
transformer = _module
ctabgan_synthesizer = _module
transformer = _module
pipeline_ctabganp = _module
train_sample_ctabganp = _module
tune_ctabgan = _module
model = _module
ctabgan_synthesizer = _module
transformer = _module
pipeline_ctabgan = _module
train_sample_ctabgan = _module
ctgan = _module
data = _module
data_sampler = _module
data_transformer = _module
demo = _module
synthesizers = _module
base = _module
ctgan = _module
tvae = _module
setup = _module
tasks = _module
test_ctgan = _module
test_tvae = _module
test_data_transformer = _module
unit = _module
synthesizer = _module
test_base = _module
test_ctgan = _module
test_tvae = _module
pipeline_tvae = _module
train_sample_tvae = _module
tune_tvae = _module
lib = _module
data = _module
deep = _module
env = _module
metrics = _module
util = _module
scripts = _module
eval_catboost = _module
eval_mlp = _module
eval_seeds = _module
eval_seeds_simple = _module
eval_simple = _module
pipeline = _module
resample_privacy = _module
sample = _module
train = _module
tune_ddpm = _module
tune_evaluation_model = _module
utils_train = _module
pipeline_smote = _module
sample_smote = _module
tune_smote = _module
tab_ddpm = _module
gaussian_multinomial_diffsuion = _module
modules = _module
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


import numpy as np


import pandas as pd


import torch


import torch.utils.data


import torch.optim as optim


from torch.optim import Adam


from torch.nn import functional as F


from torch.nn import Dropout


from torch.nn import LeakyReLU


from torch.nn import Linear


from torch.nn import Module


from torch.nn import ReLU


from torch.nn import Sequential


from torch.nn import Conv2d


from torch.nn import ConvTranspose2d


from torch.nn import Sigmoid


from torch.nn import init


from torch.nn import BCELoss


from torch.nn import CrossEntropyLoss


from torch.nn import SmoothL1Loss


from torch.nn import LayerNorm


from sklearn.mixture import BayesianGaussianMixture


import time


from torch.nn import BatchNorm2d


import warnings


from torch import optim


from torch.nn import BatchNorm1d


from torch.nn import functional


from torch.nn import Parameter


from torch.nn.functional import cross_entropy


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from sklearn.exceptions import ConvergenceWarning


from collections import Counter


from copy import deepcopy


from typing import Any


from typing import Literal


from typing import Optional


from typing import Union


from typing import cast


from typing import Tuple


from typing import Dict


from typing import List


from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline


import sklearn.preprocessing


from sklearn.impute import SimpleImputer


from sklearn.preprocessing import StandardScaler


from scipy.spatial.distance import cdist


from typing import Callable


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


import enum


import uuid


from typing import Type


from typing import TypeVar


from typing import get_args


from typing import get_origin


from sklearn.metrics import classification_report


from sklearn.metrics import r2_score


from sklearn.metrics import f1_score


from sklearn.utils import shuffle


from torch.optim import AdamW


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler


from sklearn.preprocessing import OneHotEncoder


from sklearn.metrics import pairwise_distances


import math


import torch.optim


from torch.profiler import record_function


from inspect import isfunction


class Classifier(Module):

    def __init__(self, input_dim, dis_dims, st_ed):
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        if st_ed[1] - st_ed[0] == 1:
            seq += [Linear(dim, 1)]
        elif st_ed[1] - st_ed[0] == 2:
            seq += [Linear(dim, 1), Sigmoid()]
        else:
            seq += [Linear(dim, st_ed[1] - st_ed[0])]
        self.seq = Sequential(*seq)

    def forward(self, input):
        label = None
        if self.str_end[1] - self.str_end[0] == 1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        new_imp = torch.cat((input[:, :self.str_end[0]], input[:, self.str_end[1]:]), 1)
        if (self.str_end[1] - self.str_end[0] == 2) | (self.str_end[1] - self.str_end[0] == 1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


class Discriminator(Module):
    """Discriminator for the CTGANSynthesizer."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        disc_interpolates = self(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size(), device=device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = (gradients_view ** 2).mean() * lambda_
        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGANSynthesizer."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class Encoder(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def cos_sin(x: Tensor) ->Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


class FoundNANsError(BaseException):
    """Found NANs during sampling"""

    def __init__(self, message='Found NANs during sampling.'):
        super(FoundNANsError, self).__init__(message)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule_name}')


def index_to_log_onehot(x, num_classes):
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i], num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1)
    log_onehot = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_onehot


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, 'at least one argument must be a Tensor'
    logvar1, logvar2 = [(x if isinstance(x, torch.Tensor) else torch.tensor(x)) for x in (logvar1, logvar2)]
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def ohe_to_categories(ohe, K):
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = []
    for i in range(len(indices) - 1):
        res.append(ohe[:, indices[i]:indices[i + 1]].argmax(dim=1))
    return torch.stack(res, dim=1)


@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) ->torch.Tensor:
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


@torch.jit.script
def sliced_logsumexp(x, slices):
    lse = torch.logcumsumexp(torch.nn.functional.pad(x, [1, 0, 0, 0], value=-float('inf')), dim=-1)
    slice_starts = slices[:-1]
    slice_ends = slices[1:]
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    slice_lse_repeated = torch.repeat_interleave(slice_lse, slice_ends - slice_starts, dim=-1)
    return slice_lse_repeated


def sum_except_batch(x, num_dims=1):
    """
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class GaussianMultinomialDiffusion(torch.nn.Module):

    def __init__(self, num_classes: np.array, num_numerical_features: int, denoise_fn, num_timesteps=1000, gaussian_loss_type='mse', gaussian_parametrization='eps', multinomial_loss_type='vb_stochastic', parametrization='x0', scheduler='cosine', device=torch.device('cpu')):
        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')
        if multinomial_loss_type == 'vb_all':
            None
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
        self.num_classes_expanded = torch.from_numpy(np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))]))
        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets))
        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler
        alphas = 1.0 - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1.0 - alphas
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.from_numpy(np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))).float()
        self.posterior_mean_coef1 = (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float()
        self.posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod)).float()
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1e-05
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-05
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1e-05
        self.register_buffer('alphas', alphas.float())
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float())
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float())
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float())
        self.register_buffer('Lt_history', torch.zeros(num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(num_timesteps))

    def gaussian_q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0), (1.0 - self.alphas)[1:]], dim=0)
        model_log_variance = torch.log(model_variance)
        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)
        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape, f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'
        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}

    def _vb_terms_bpd(self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.gaussian_p_mean_variance(model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out['mean'], out['log_variance'])
        kl = mean_flat(kl) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out['mean'], log_scales=0.5 * out['log_variance'])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = torch.where(t == 0, decoder_nll, kl)
        return {'output': output, 'pred_xstart': out['pred_xstart'], 'out_mean': out['mean'], 'true_mean': true_mean}

    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms['loss'] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms['loss'] = self._vb_terms_bpd(model_output=model_out, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, model_kwargs=model_kwargs)['output']
        return terms['loss']

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(self, model_out, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        out = self.gaussian_p_mean_variance(model_out, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_probs = log_add_exp(log_x_t + log_alpha_t, log_1_min_alpha_t - torch.log(self.num_classes_expanded))
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(log_x_start + log_cumprod_alpha_t, log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded))
        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):
        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'
        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)
        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)
        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size=16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            kl = self.compute_Lt(log_x_start=log_x_start, log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array), t=t_array, out_dict=out_dict)
            loss += kl
        loss += self.kl_prior(log_x_start)
        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()
        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))
        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)
        if detach_mean:
            log_model_prob = log_model_prob.detach()
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl
        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):
        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(model_out, log_x_start, log_x_t, t, out_dict)
            kl_prior = self.kl_prior(log_x_start)
            vb_loss = kl / pt + kl_prior
            return vb_loss
        elif self.multinomial_loss_type == 'vb_all':
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)
        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)
            t, pt = self.sample_time(b, device, 'importance')
            kl = self.compute_Lt(log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)
            kl_prior = self.kl_prior(log_x_start)
            loss = kl / pt + kl_prior
            return -loss

    def mixed_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')
        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)
        model_out = self._denoise_fn(x_in, t, **out_dict)
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]
        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes)
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)
        return loss_multi.mean(), loss_gauss.mean()

    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device
        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat
            model_out = self._denoise_fn(torch.cat([x_num_t, log_x_cat_t], dim=1), t_array, **out_dict)
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(model_out=model_out_cat, log_x_start=log_x_cat, log_x_t=log_x_cat_t, t=t_array, out_dict=out_dict)
            out = self._vb_terms_bpd(model_out_num, x_start=x_num, x_t=x_num_t, t=t_array, clip_denoised=False)
            multinomial_loss.append(kl)
            gaussian_loss.append(out['output'])
            xstart_mse.append(mean_flat((out['pred_xstart'] - x_num) ** 2))
            out_mean.append(mean_flat(out['out_mean']))
            true_mean.append(mean_flat(out['true_mean']))
            eps = self._predict_eps_from_xstart(x_num_t, t_array, out['pred_xstart'])
            mse.append(mean_flat((eps - noise) ** 2))
        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)
        prior_gauss = self._prior_gaussian(x_num)
        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)
        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {'total_gaussian': total_gauss, 'total_multinomial': total_multin, 'losses_gaussian': gaussian_loss, 'losses_multinimial': multinomial_loss, 'xstart_mse': xstart_mse, 'mse': mse, 'out_mean': out_mean, 'true_mean': true_mean}

    @torch.no_grad()
    def gaussian_ddim_step(self, model_out_num, x, t, clip_denoised=False, denoised_fn=None, eta=0.0):
        out = self.gaussian_p_mean_variance(model_out_num, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=None)
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = torch.randn_like(x)
        mean_pred = out['pred_xstart'] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample

    @torch.no_grad()
    def gaussian_ddim_sample(self, noise, T, out_dict, eta=0.0):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            None
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(out_num, x, t_array)
        None
        return x

    @torch.no_grad()
    def gaussian_ddim_reverse_step(self, model_out_num, x, t, clip_denoised=False, eta=0.0):
        assert eta == 0.0, 'Eta must be zero.'
        out = self.gaussian_p_mean_variance(model_out_num, x, t, clip_denoised=clip_denoised, denoised_fn=None, model_kwargs=None)
        eps = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out['pred_xstart']) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out['pred_xstart'] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(self, x, T, out_dict):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            None
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(out_num, x, t_array, eta=0.0)
        None
        return x

    @torch.no_grad()
    def multinomial_ddim_step(self, model_out_cat, log_x_t, t, out_dict, eta=0.0):
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)
        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        log_ps = torch.stack([torch.log(coef1) + log_x_t, torch.log(coef2) + log_x0, torch.log(coef3) - torch.log(self.num_classes_expanded)], dim=2)
        log_prob = torch.logsumexp(log_ps, dim=2)
        out = self.log_sample_categorical(log_prob)
        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)
        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {'y': y.long()}
        for i in reversed(range(0, self.num_timesteps)):
            None
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)
        None
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def sample(self, num_samples, y_dist):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
            log_z = self.log_sample_categorical(uniform_logits)
        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {'y': y.long()}
        for i in reversed(range(0, self.num_timesteps)):
            None
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)
        None
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            None
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        b = batch_size
        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]
            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]
        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]
        return x_gen, y_gen


class SiLU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def reglu(x: Tensor) ->Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) ->Tensor:
        return reglu(x)


def geglu(x: Tensor) ->Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) ->Tensor:
        return geglu(x)


ModuleType = Union[str, Callable[..., nn.Module]]


def _make_nn_module(module_type: ModuleType, *args) ->nn.Module:
    return (ReGLU() if module_type == 'ReGLU' else GEGLU() if module_type == 'GEGLU' else getattr(nn, module_type)(*args)) if isinstance(module_type, str) else module_type(*args)


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """


    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(self, *, d_in: int, d_out: int, bias: bool, activation: ModuleType, dropout: float) ->None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) ->Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self, *, d_in: int, d_layers: List[int], dropouts: Union[float, List[float]], activation: Union[str, Callable[[], nn.Module]], d_out: int) ->None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']
        self.blocks = nn.ModuleList([MLP.Block(d_in=d_layers[i - 1] if i else d_in, d_out=d, bias=True, activation=activation, dropout=dropout) for i, (d, dropout) in enumerate(zip(d_layers, dropouts))])
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(cls: Type['MLP'], d_in: int, d_layers: List[int], dropout: float, d_out: int) ->'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, 'if d_layers contains more than two elements, then all elements except for the first and the last ones must be equal.'
        return MLP(d_in=d_in, d_layers=d_layers, dropouts=dropout, activation='ReLU', d_out=d_out)

    def forward(self, x: Tensor) ->Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """


    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(self, *, d_main: int, d_hidden: int, bias_first: bool, bias_second: bool, dropout_first: float, dropout_second: float, normalization: ModuleType, activation: ModuleType, skip_connection: bool) ->None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) ->Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x


    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(self, *, d_in: int, d_out: int, bias: bool, normalization: ModuleType, activation: ModuleType) ->None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) ->Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(self, *, d_in: int, n_blocks: int, d_main: int, d_hidden: int, dropout_first: float, dropout_second: float, normalization: ModuleType, activation: ModuleType, d_out: int) ->None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(*[ResNet.Block(d_main=d_main, d_hidden=d_hidden, bias_first=True, bias_second=True, dropout_first=dropout_first, dropout_second=dropout_second, normalization=normalization, activation=activation, skip_connection=True) for _ in range(n_blocks)])
        self.head = ResNet.Head(d_in=d_main, d_out=d_out, bias=True, normalization=normalization, activation=activation)

    @classmethod
    def make_baseline(cls: Type['ResNet'], *, d_in: int, n_blocks: int, d_main: int, d_hidden: int, dropout_first: float, dropout_second: float, d_out: int) ->'ResNet':
        """Create a "baseline" `ResNet`.
        This variation of ResNet was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`
        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(d_in=d_in, n_blocks=n_blocks, d_main=d_main, d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second, normalization='BatchNorm1d', activation='ReLU', d_out=d_out)

    def forward(self, x: Tensor) ->Tensor:
        x = x.float()
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPDiffusion(nn.Module):

    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond
        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in
        self.mlp = MLP.make_baseline(**rtdl_params)
        if self.num_classes > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)


class ResNetDiffusion(nn.Module):

    def __init__(self, d_in, num_classes, rtdl_params, dim_t=256):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        rtdl_params['d_in'] = d_in
        rtdl_params['d_out'] = d_in
        rtdl_params['emb_d'] = dim_t
        self.resnet = ResNet.make_baseline(**rtdl_params)
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if y is not None and self.num_classes > 0:
            emb += self.label_emb(y.squeeze())
        return self.resnet(x, emb)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier,
     lambda: ([], {'input_dim': 4, 'dis_dims': [4, 4], 'st_ed': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'embedding_dim': 4, 'decompress_dims': [4, 4], 'data_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'data_dim': 4, 'compress_dims': [4, 4], 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Generator,
     lambda: ([], {'embedding_dim': 4, 'generator_dim': [4, 4], 'data_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'d_in': 4, 'd_layers': [4, 4], 'dropouts': 0.5, 'activation': _mock_layer, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'d_in': 4, 'n_blocks': 4, 'd_main': 4, 'd_hidden': 4, 'dropout_first': 0.5, 'dropout_second': 0.5, 'normalization': _mock_layer, 'activation': _mock_layer, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'i': 4, 'o': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_rotot0_tab_ddpm(_paritybench_base):
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

