import sys
_module = sys.modules[__name__]
del sys
conf = _module
adult_census = _module
adult_census_attention_mlp = _module
adult_census_bayesian_tabmlp = _module
adult_census_cont_den_full_example = _module
adult_census_cont_den_run_all_models = _module
adult_census_enc_dec_full_example = _module
adult_census_enc_dec_run_all_models = _module
adult_census_tabnet = _module
adult_census_transformers = _module
airbnb_all_modes_multiclass = _module
airbnb_all_modes_regr = _module
airbnb_data_preprocessing = _module
bio_imbalanced_loader = _module
california_housing_fds_lds = _module
download_images = _module
pytorch_widedeep = _module
bayesian_models = _module
_base_bayesian_model = _module
_weight_sampler = _module
bayesian_nn = _module
modules = _module
bayesian_embedding = _module
bayesian_linear = _module
tabular = _module
bayesian_embeddings_layers = _module
bayesian_wide = _module
bayesian_mlp = _module
_layers = _module
bayesian_tab_mlp = _module
callbacks = _module
dataloaders = _module
datasets = _module
_base = _module
data = _module
initializers = _module
losses = _module
metrics = _module
models = _module
_get_activation_fn = _module
fds_layer = _module
image = _module
_layers = _module
vision = _module
_base_tabular_model = _module
embeddings_layers = _module
linear = _module
wide = _module
mlp = _module
_attention_layers = _module
_encoders = _module
_layers = _module
context_attention_mlp = _module
self_attention_mlp = _module
tab_mlp = _module
resnet = _module
_layers = _module
tab_resnet = _module
self_supervised = _module
_augmentations = _module
_denoise_mlps = _module
_random_obfuscator = _module
contrastive_denoising_model = _module
encoder_decoder_model = _module
tabnet = _module
_layers = _module
_utils = _module
sparsemax = _module
tab_net = _module
transformers = _module
_attention_layers = _module
_encoders = _module
ft_transformer = _module
saint = _module
tab_fastformer = _module
tab_perceiver = _module
tab_transformer = _module
text = _module
_encoders = _module
attentive_rnn = _module
basic_rnn = _module
stacked_attentive_rnn = _module
wide_deep = _module
preprocessing = _module
base_preprocessor = _module
image_preprocessor = _module
tab_preprocessor = _module
text_preprocessor = _module
wide_preprocessor = _module
self_supervised_training = _module
_base_contrastive_denoising_trainer = _module
_base_encoder_decoder_trainer = _module
contrastive_denoising_trainer = _module
encoder_decoder_trainer = _module
tab2vec = _module
training = _module
_base_bayesian_trainer = _module
_base_trainer = _module
_finetune = _module
_loss_and_obj_aliases = _module
_multiple_lr_scheduler = _module
_multiple_optimizer = _module
_multiple_transforms = _module
_trainer_utils = _module
_wd_dataset = _module
bayesian_trainer = _module
trainer = _module
utils = _module
deeptabular_utils = _module
fastai_transforms = _module
general_utils = _module
image_utils = _module
text_utils = _module
version = _module
wdtypes = _module
setup = _module
tests = _module
test_b_losses = _module
test_mc_bayes_tabmlp = _module
test_mc_bayes_wide = _module
test_b_callbacks = _module
test_b_data_inputs = _module
test_b_fit_methods = _module
test_b_miscellaneous = _module
test_b_t2v = _module
test_data_utils = _module
test_du_base_preprocessor = _module
test_du_image = _module
test_du_tabular = _module
test_du_text = _module
test_du_wide = _module
test_fastai_transforms = _module
test_datasets = _module
test_finetune = _module
test_finetuning_routines = _module
test_losses = _module
test_metrics = _module
test_torchmetrics = _module
test_model_components = _module
test_mc_attn_tab_mlp = _module
test_mc_image = _module
test_mc_tab_mlp = _module
test_mc_tab_resnet = _module
test_mc_tab_tabnet = _module
test_mc_text = _module
test_mc_transformers = _module
test_mc_wide = _module
test_wide_deep = _module
test_model_functioning = _module
test_callbacks = _module
test_data_inputs = _module
test_fit_methods = _module
test_initializers = _module
test_miscellaneous = _module
test_self_supervised = _module
test_ss_callbacks = _module
test_ss_miscellaneous = _module
test_ss_model_components = _module
test_ss_model_pretrain_method = _module
test_t2v = _module

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


import torch


import pandas as pd


from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split


from itertools import product


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


import time


import warnings


from torch.optim import SGD


from torch.optim import lr_scheduler


from sklearn.metrics import classification_report


from torch import nn


import math


import torch.nn.functional as F


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Tuple


from torch.utils.data import DataLoader


from torch.utils.data import WeightedRandomSampler


import re


from typing import Dict


import torch.nn as nn


import torchvision


from torch import einsum


from collections import OrderedDict


import inspect


from scipy.sparse import csc_matrix


from torch.autograd import Function


from abc import ABC


from abc import abstractmethod


from torch.utils.data import TensorDataset


from copy import deepcopy


from scipy.ndimage import convolve1d


from sklearn.utils import Bunch


from torch.utils.data import Dataset


from inspect import signature


from scipy.ndimage import gaussian_filter1d


from sklearn.exceptions import NotFittedError


from scipy.signal.windows import triang


from types import SimpleNamespace


from typing import Any


from typing import List


from typing import Match


from typing import Union


from typing import Callable


from typing import Iterable


from typing import Iterator


from typing import Optional


from typing import Generator


from typing import Collection


from torch import Tensor


from torch.nn import Module


from torch.optim.optimizer import Optimizer


from torchvision.transforms import Pad


from torchvision.transforms import Lambda


from torchvision.transforms import Resize


from torchvision.transforms import Compose


from torchvision.transforms import TenCrop


from torchvision.transforms import FiveCrop


from torchvision.transforms import Grayscale


from torchvision.transforms import CenterCrop


from torchvision.transforms import RandomCrop


from torchvision.transforms import ToPILImage


from torchvision.transforms import ColorJitter


from torchvision.transforms import PILToTensor


from torchvision.transforms import RandomApply


from torchvision.transforms import RandomOrder


from torchvision.transforms import GaussianBlur


from torchvision.transforms import RandomAffine


from torchvision.transforms import RandomChoice


from torchvision.transforms import RandomInvert


from torchvision.transforms import RandomErasing


from torchvision.transforms import RandomEqualize


from torchvision.transforms import RandomRotation


from torchvision.transforms import RandomSolarize


from torchvision.transforms import RandomGrayscale


from torchvision.transforms import RandomPosterize


from torchvision.transforms import ConvertImageDtype


from torchvision.transforms import InterpolationMode


from torchvision.transforms import RandomPerspective


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import RandomAutocontrast


from torchvision.transforms import RandomVerticalFlip


from torchvision.transforms import LinearTransformation


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomAdjustSharpness


from torchvision.models._api import WeightsEnum


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data.dataloader import DataLoader


from sklearn.metrics import mean_squared_error


import string


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import CyclicLR


from scipy import stats


from numpy.testing import assert_almost_equal


from sklearn.metrics import mean_squared_log_error


from sklearn.metrics import f1_score


from sklearn.metrics import r2_score


from sklearn.metrics import fbeta_score


from sklearn.metrics import recall_score


from sklearn.metrics import precision_score


from torchvision.models import MNASNet1_0_Weights


from torchvision.models import MobileNet_V2_Weights


from torchvision.models import SqueezeNet1_0_Weights


from torchvision.models import ResNeXt50_32X4D_Weights


from torchvision.models import Wide_ResNet50_2_Weights


from torchvision.models import ShuffleNet_V2_X0_5_Weights


from copy import copy


from itertools import chain


from copy import deepcopy as c


class BayesianModule(nn.Module):
    """Simply a 'hack' to facilitate the computation of the KL divergence for all
    Bayesian models
    """

    def init(self):
        super().__init__()


class BaseBayesianModel(nn.Module):
    """Base model containing the two methods common to all Bayesian models"""

    def init(self):
        super().__init__()

    def _kl_divergence(self):
        kld = 0
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kld += module.log_variational_posterior - module.log_prior
        return kld

    def sample_elbo(self, input: Tensor, target: Tensor, loss_fn: nn.Module, n_samples: int, n_batches: int) ->Tuple[Tensor, Tensor]:
        outputs_l = []
        kld = 0.0
        for _ in range(n_samples):
            outputs_l.append(self(input))
            kld += self._kl_divergence()
        outputs = torch.stack(outputs_l)
        complexity_cost = kld / n_batches
        likelihood_cost = loss_fn(outputs.mean(0), target)
        return outputs, complexity_cost + likelihood_cost


class GaussianPosterior(object):
    """Defines the Gaussian variational posterior as proposed in Weight
    Uncertainty in Neural Networks
    """

    def __init__(self, param_mu: Tensor, param_rho: Tensor):
        super().__init__()
        self.param_mu = param_mu
        self.param_rho = param_rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.param_rho))

    def sample(self) ->Tensor:
        epsilon = self.normal.sample(self.param_rho.size())
        return self.param_mu + self.sigma * epsilon

    def log_posterior(self, input: Tensor) ->Tensor:
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - (input - self.param_mu) ** 2 / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussianPrior(object):
    """Defines the Scale Mixture Prior as proposed in Weight Uncertainty in
    Neural Networks (Eq 7 in the original publication)
    """

    def __init__(self, pi: float, sigma1: float, sigma2: float):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, input: Tensor) ->Tensor:
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return torch.log(self.pi * prob1 + (1 - self.pi) * prob2).sum()


class BayesianEmbedding(BayesianModule):
    """A simple lookup table that looks up embeddings in a fixed dictionary and
    size.

    Parameters
    ----------
    n_embed: int
        number of embeddings. Typically referred as size of the vocabulary
    embed_dim: int
        Dimension of the embeddings
    padding_idx: int, optional, default = None
        If specified, the entries at ``padding_idx`` do not contribute to the
        gradient; therefore, the embedding vector at ``padding_idx`` is not
        updated during training, i.e. it remains as a fixed “pad”. For a
        newly constructed Embedding, the embedding vector at ``padding_idx``
        will default to all zeros, but can be updated to another value to be
        used as the padding vector
    max_norm: float, optional, default = None
        If given, each embedding vector with norm larger than ``max_norm`` is
        renormalized to have norm max_norm
    norm_type: float, optional, default = 2.
        The p of the p-norm to compute for the ``max_norm`` option.
    scale_grad_by_freq: bool, optional, default = False
        If given, this will scale gradients by the inverse of frequency of the
        words in the mini-batch.
    sparse: bool, optional, default = False
        If True, gradient w.r.t. weight matrix will be a sparse tensor. See
        Notes for more details regarding sparse gradients.
    prior_sigma_1: float, default = 1.0
        Prior of the sigma parameter for the first of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution
    posterior_mu_init: float = 0.0
        The posterior sample of the weights is defined as:

        .. math::
           \\begin{aligned}
           \\mathbf{w} &= \\mu + log(1 + exp(\\rho))
           \\end{aligned}

        where:

        .. math::
           \\begin{aligned}
           \\mathcal{N}(x\\vert \\mu, \\sigma) &= \\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\\\\
           \\log{\\mathcal{N}(x\\vert \\mu, \\sigma)} &= -\\log{\\sqrt{2\\pi}} -\\log{\\sigma} -\\frac{(x-\\mu)^2}{2\\sigma^2}\\\\
           \\end{aligned}

        :math:`\\mu` is initialised using a normal distributtion with mean
        ``posterior_rho_init`` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of :math:`\\mu`, :math:`\\rho` is initialised using a
        normal distributtion with mean ``posterior_rho_init`` and std equal to
        0.1.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
    >>> embedding = bnn.BayesianEmbedding(10, 3)
    >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    >>> out = embedding(input)
    """

    def __init__(self, n_embed: int, embed_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: Optional[float]=2.0, scale_grad_by_freq: Optional[bool]=False, sparse: Optional[bool]=False, prior_sigma_1: float=1.0, prior_sigma_2: float=0.002, prior_pi: float=0.8, posterior_mu_init: float=0.0, posterior_rho_init: float=-7.0):
        super(BayesianEmbedding, self).__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.weight_mu = nn.Parameter(torch.Tensor(n_embed, embed_dim).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(n_embed, embed_dim).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)
        self.weight_prior_dist = ScaleMixtureGaussianPrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) ->Tensor:
        if not self.training:
            return F.embedding(X, self.weight_mu, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        weight = self.weight_sampler.sample()
        self.log_variational_posterior = self.weight_sampler.log_posterior(weight)
        self.log_prior = self.weight_prior_dist.log_prior(weight)
        return F.embedding(X, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) ->str:
        s = '{n_embed}, {embed_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        if self.prior_sigma_1 != 1.0:
            s += ', prior_sigma_1={prior_sigma_1}'
        if self.prior_sigma_2 != 0.002:
            s += ', prior_sigma_2={prior_sigma_2}'
        if self.prior_pi != 0.8:
            s += ', prior_pi={prior_pi}'
        if self.posterior_mu_init != 0.0:
            s += ', posterior_mu_init={posterior_mu_init}'
        if self.posterior_rho_init != -7.0:
            s += ', posterior_rho_init={posterior_rho_init}'
        return s.format(**self.__dict__)


class BayesianLinear(BayesianModule):
    """Applies a linear transformation to the incoming data as proposed in Weight
    Uncertainity on Neural Networks

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
         size of each output sample
    use_bias: bool, default = True
        Boolean indicating if an additive bias will be learnt
    prior_sigma_1: float, default = 1.0
        Prior of the sigma parameter for the first of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution
    posterior_mu_init: float = 0.0
        The posterior sample of the weights is defined as:

        .. math::
           \\begin{aligned}
           \\mathbf{w} &= \\mu + log(1 + exp(\\rho))
           \\end{aligned}

        where:

        .. math::
           \\begin{aligned}
           \\mathcal{N}(x\\vert \\mu, \\sigma) &= \\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\\\\
           \\log{\\mathcal{N}(x\\vert \\mu, \\sigma)} &= -\\log{\\sqrt{2\\pi}} -\\log{\\sigma} -\\frac{(x-\\mu)^2}{2\\sigma^2}\\\\
           \\end{aligned}

        :math:`\\mu` is initialised using a normal distributtion with mean
        ``posterior_rho_init`` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of :math:`\\mu`, :math:`\\rho` is initialised using a
        normal distributtion with mean ``posterior_rho_init`` and std equal to
        0.1.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import bayesian_nn as bnn
    >>> linear = bnn.BayesianLinear(10, 6)
    >>> input = torch.rand(6, 10)
    >>> out = linear(input)
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool=True, prior_sigma_1: float=1.0, prior_sigma_2: float=0.002, prior_pi: float=0.8, posterior_mu_init: float=0.0, posterior_rho_init: float=-7.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = GaussianPosterior(self.bias_mu, self.bias_rho)
        else:
            self.bias_mu, self.bias_rho = None, None
        self.weight_prior_dist = ScaleMixtureGaussianPrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        if self.use_bias:
            self.bias_prior_dist = ScaleMixtureGaussianPrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2)
        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) ->Tensor:
        if not self.training:
            return F.linear(X, self.weight_mu, self.bias_mu)
        weight = self.weight_sampler.sample()
        if self.use_bias:
            bias = self.bias_sampler.sample()
            bias_log_posterior: Union[Tensor, float] = self.bias_sampler.log_posterior(bias)
            bias_log_prior: Union[Tensor, float] = self.bias_prior_dist.log_prior(bias)
        else:
            bias = None
            bias_log_posterior = 0.0
            bias_log_prior = 0.0
        self.log_variational_posterior = self.weight_sampler.log_posterior(weight) + bias_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(weight) + bias_log_prior
        return F.linear(X, weight, bias)

    def extra_repr(self) ->str:
        s = '{in_features}, {out_features}'
        if self.use_bias is not False:
            s += ', use_bias=True'
        if self.prior_sigma_1 != 1.0:
            s += ', prior_sigma_1={prior_sigma_1}'
        if self.prior_sigma_2 != 0.002:
            s += ', prior_sigma_2={prior_sigma_2}'
        if self.prior_pi != 0.8:
            s += ', prior_pi={prior_pi}'
        if self.posterior_mu_init != 0.0:
            s += ', posterior_mu_init={posterior_mu_init}'
        if self.posterior_rho_init != -7.0:
            s += ', posterior_rho_init={posterior_rho_init}'
        return s.format(**self.__dict__)


class BayesianContEmbeddings(BayesianModule):

    def __init__(self, n_cont_cols: int, embed_dim: int, prior_sigma_1: float, prior_sigma_2: float, prior_pi: float, posterior_mu_init: float, posterior_rho_init: float, use_bias: bool=False):
        super(BayesianContEmbeddings, self).__init__()
        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.weight_mu = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianPosterior(self.weight_mu, self.weight_rho)
        if use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = GaussianPosterior(self.bias_mu, self.bias_rho)
        else:
            self.bias_mu, self.bias_rho = None, None
        self.weight_prior_dist = ScaleMixtureGaussianPrior(prior_pi, prior_sigma_1, prior_sigma_2)
        if self.use_bias:
            self.bias_prior_dist = ScaleMixtureGaussianPrior(prior_pi, prior_sigma_1, prior_sigma_2)
        self.log_prior: Union[Tensor, float] = 0.0
        self.log_variational_posterior: Union[Tensor, float] = 0.0

    def forward(self, X: Tensor) ->Tensor:
        if not self.training:
            x = self.weight_mu.unsqueeze(0) * X.unsqueeze(2)
            if self.bias_mu is not None:
                x + self.bias_mu.unsqueeze(0)
            return x
        weight = self.weight_sampler.sample()
        if self.use_bias:
            bias = self.bias_sampler.sample()
            bias_log_posterior: Union[Tensor, float] = self.bias_sampler.log_posterior(bias)
            bias_log_prior: Union[Tensor, float] = self.bias_prior_dist.log_prior(bias)
        else:
            bias = 0.0
            bias_log_posterior = 0.0
            bias_log_prior = 0.0
        self.log_variational_posterior = self.weight_sampler.log_posterior(weight) + bias_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(weight) + bias_log_prior
        x = weight.unsqueeze(0) * X.unsqueeze(2) + bias
        return x

    def extra_repr(self) ->str:
        s = '{n_cont_cols}, {embed_dim}, use_bias={use_bias}'
        return s.format(**self.__dict__)


class BayesianDiffSizeCatEmbeddings(nn.Module):

    def __init__(self, column_idx: Dict[str, int], embed_input: List[Tuple[str, int, int]], prior_sigma_1: float, prior_sigma_2: float, prior_pi: float, posterior_mu_init: float, posterior_rho_init: float):
        super(BayesianDiffSizeCatEmbeddings, self).__init__()
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_layers = nn.ModuleDict({('emb_layer_' + col): bnn.BayesianEmbedding(val + 1, dim, padding_idx=0, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init) for col, val, dim in self.embed_input})
        self.emb_out_dim: int = int(np.sum([embed[2] for embed in self.embed_input]))

    def forward(self, X: Tensor) ->Tensor:
        embed = [self.embed_layers['emb_layer_' + col](X[:, self.column_idx[col]].long()) for col, _, _ in self.embed_input]
        x = torch.cat(embed, 1)
        return x


NormLayers = Union[nn.Identity, nn.LayerNorm, nn.BatchNorm1d]


class BayesianDiffSizeCatAndContEmbeddings(nn.Module):

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: List[Tuple[str, int, int]], continuous_cols: Optional[List[str]], embed_continuous: bool, cont_embed_dim: int, use_cont_bias: bool, cont_norm_layer: Optional[str], prior_sigma_1: float, prior_sigma_2: float, prior_pi: float, posterior_mu_init: float, posterior_rho_init: float):
        super(BayesianDiffSizeCatAndContEmbeddings, self).__init__()
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        if self.cat_embed_input is not None:
            self.cat_embed = BayesianDiffSizeCatEmbeddings(column_idx, cat_embed_input, prior_sigma_1, prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init)
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == 'layernorm':
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == 'batchnorm':
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = BayesianContEmbeddings(len(continuous_cols), cont_embed_dim, prior_sigma_1, prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init, use_cont_bias)
                self.cont_out_dim = len(continuous_cols) * cont_embed_dim
            else:
                self.cont_out_dim = len(continuous_cols)
        else:
            self.cont_out_dim = 0
        self.output_dim = self.cat_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) ->Tuple[Tensor, Any]:
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None
        if self.continuous_cols is not None:
            x_cont = self.cont_norm(X[:, self.cont_idx].float())
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, 'b s d -> b (s d)')
        else:
            x_cont = None
        return x_cat, x_cont


class BayesianWide(BaseBayesianModel):
    """Defines a `Wide` model. This is a linear model where the
    non-linearlities are captured via crossed-columns

    Parameters
    ----------
    input_dim: int
        size of the Embedding layer. `input_dim` is the summation of all the
        individual values for all the features that go through the wide
        component. For example, if the wide component receives 2 features with
        5 individual values each, `input_dim = 10`
    pred_dim: int
        size of the ouput tensor containing the predictions
    prior_sigma_1: float, default = 1.0
        The prior weight distribution is a scaled mixture of two Gaussian
        densities:

        $$
           \\begin{aligned}
           P(\\mathbf{w}) = \\prod_{i=j} \\pi N (\\mathbf{w}_j | 0, \\sigma_{1}^{2}) + (1 - \\pi) N (\\mathbf{w}_j | 0, \\sigma_{2}^{2})
           \\end{aligned}
        $$

        `prior_sigma_1` is the prior of the sigma parameter for the first of the two
        Gaussians that will be mixed to produce the prior weight
        distribution.
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution
    posterior_mu_init: float = 0.0
        The posterior sample of the weights is defined as:

        $$
           \\begin{aligned}
           \\mathbf{w} &= \\mu + log(1 + exp(\\rho))
           \\end{aligned}
        $$

        where:

        $$
           \\begin{aligned}
           \\mathcal{N}(x\\vert \\mu, \\sigma) &= \\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\\\\
           \\log{\\mathcal{N}(x\\vert \\mu, \\sigma)} &= -\\log{\\sqrt{2\\pi}} -\\log{\\sigma} -\\frac{(x-\\mu)^2}{2\\sigma^2}\\\\
           \\end{aligned}
        $$

        $\\mu$ is initialised using a normal distributtion with mean
        `posterior_mu_init` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of $\\mu$, $\\rho$ is initialised using a
        normal distributtion with mean `posterior_rho_init` and std equal to
        0.1.

    Attributes
    -----------
    bayesian_wide_linear: nn.Module
        the linear layer that comprises the wide branch of the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import BayesianWide
    >>> X = torch.empty(4, 4).random_(6)
    >>> wide = BayesianWide(input_dim=X.unique().size(0), pred_dim=1)
    >>> out = wide(X)
    """

    def __init__(self, input_dim: int, pred_dim: int=1, prior_sigma_1: float=1.0, prior_sigma_2: float=0.002, prior_pi: float=0.8, posterior_mu_init: float=0.0, posterior_rho_init: float=-7.0):
        super(BayesianWide, self).__init__()
        self.bayesian_wide_linear = bnn.BayesianEmbedding(n_embed=input_dim + 1, embed_dim=pred_dim, padding_idx=0, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, posterior_mu_init=posterior_mu_init, posterior_rho_init=posterior_rho_init)
        self.bias = nn.Parameter(torch.zeros(pred_dim))

    def forward(self, X: Tensor) ->Tensor:
        out = self.bayesian_wide_linear(X.long()).sum(dim=1) + self.bias
        return out


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class REGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


allowed_activations = ['relu', 'leaky_relu', 'tanh', 'gelu', 'geglu', 'reglu']


def get_activation_fn(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'geglu':
        return GEGLU()
    elif activation == 'reglu':
        return REGLU()
    elif activation == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError("Only the following activation functions are currently supported: {}. Note that 'geglu' and 'reglu' should only be used as transformer's activations".format(', '.join(allowed_activations)))


class BayesianMLP(nn.Module):

    def __init__(self, d_hidden: List[int], activation: str, use_bias: bool=True, prior_sigma_1: float=1.0, prior_sigma_2: float=0.002, prior_pi: float=0.8, posterior_mu_init: float=0.0, posterior_rho_init: float=-7.0):
        super(BayesianMLP, self).__init__()
        self.d_hidden = d_hidden
        self.activation = activation
        act_fn = get_activation_fn(activation)
        self.bayesian_mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            bayesian_dense_layer = nn.Sequential(*[bnn.BayesianLinear(d_hidden[i - 1], d_hidden[i], use_bias, prior_sigma_1, prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init), act_fn if i != len(d_hidden) - 1 else nn.Identity()])
            self.bayesian_mlp.add_module('bayesian_dense_layer_{}'.format(i - 1), bayesian_dense_layer)

    def forward(self, X: Tensor) ->Tensor:
        return self.bayesian_mlp(X)


class BayesianTabMlp(BaseBayesianModel):
    """Defines a `BayesianTabMlp` model.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of probabilistic dense layers (i.e. a MLP).

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabMlp` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    prior_sigma_1: float, default = 1.0
        The prior weight distribution is a scaled mixture of two Gaussian
        densities:

        $$
           \\begin{aligned}
           P(\\mathbf{w}) = \\prod_{i=j} \\pi N (\\mathbf{w}_j | 0, \\sigma_{1}^{2}) + (1 - \\pi) N (\\mathbf{w}_j | 0, \\sigma_{2}^{2})
           \\end{aligned}
        $$

        `prior_sigma_1` is the prior of the sigma parameter for the first of the two
        Gaussians that will be mixed to produce the prior weight
        distribution.
    prior_sigma_2: float, default = 0.002
        Prior of the sigma parameter for the second of the two Gaussian
        distributions that will be mixed to produce the prior weight
        distribution for each Bayesian linear and embedding layer
    prior_pi: float, default = 0.8
        Scaling factor that will be used to mix the Gaussians to produce the
        prior weight distribution ffor each Bayesian linear and embedding
        layer
    posterior_mu_init: float = 0.0
        The posterior sample of the weights is defined as:

        $$
           \\begin{aligned}
           \\mathbf{w} &= \\mu + log(1 + exp(\\rho))
           \\end{aligned}
        $$
        where:

        $$
           \\begin{aligned}
           \\mathcal{N}(x\\vert \\mu, \\sigma) &= \\frac{1}{\\sqrt{2\\pi}\\sigma}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\\\\
           \\log{\\mathcal{N}(x\\vert \\mu, \\sigma)} &= -\\log{\\sqrt{2\\pi}} -\\log{\\sigma} -\\frac{(x-\\mu)^2}{2\\sigma^2}\\\\
           \\end{aligned}
        $$

        $\\mu$ is initialised using a normal distributtion with mean
        `posterior_mu_init` and std equal to 0.1.
    posterior_rho_init: float = -7.0
        As in the case of $\\mu$, $\\rho$ is initialised using a
        normal distributtion with mean `posterior_rho_init` and std equal to
        0.1.

    Attributes
    ----------
    bayesian_cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    bayesian_tab_mlp: nn.Sequential
        mlp model that will receive the concatenation of the embeddings and
        the continuous columns

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.bayesian_models import BayesianTabMlp
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = BayesianTabMlp(mlp_hidden_dims=[8,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int, int]]]=None, cat_embed_dropout: float=0.1, cat_embed_activation: Optional[str]=None, continuous_cols: Optional[List[str]]=None, embed_continuous: bool=False, cont_embed_dim: int=32, cont_embed_dropout: float=0.1, cont_embed_activation: Optional[str]=None, use_cont_bias: bool=True, cont_norm_layer: str='batchnorm', mlp_hidden_dims: List[int]=[200, 100], mlp_activation: str='leaky_relu', prior_sigma_1: float=1, prior_sigma_2: float=0.002, prior_pi: float=0.8, posterior_mu_init: float=0.0, posterior_rho_init: float=-7.0, pred_dim=1):
        super(BayesianTabMlp, self).__init__()
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.pred_dim = pred_dim
        allowed_activations = ['relu', 'leaky_relu', 'tanh', 'gelu']
        if self.mlp_activation not in allowed_activations:
            raise ValueError("Currently, only the following activation functions are supported for the Bayesian MLP's dense layers: {}. Got '{}' instead".format(', '.join(allowed_activations), self.mlp_activation))
        self.cat_and_cont_embed = BayesianDiffSizeCatAndContEmbeddings(column_idx, cat_embed_input, continuous_cols, embed_continuous, cont_embed_dim, use_cont_bias, cont_norm_layer, prior_sigma_1, prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init)
        self.cat_embed_act_fn = get_activation_fn(cat_embed_activation) if cat_embed_activation is not None else None
        self.cont_embed_act_fn = get_activation_fn(cont_embed_activation) if cont_embed_activation is not None else None
        mlp_input_dim = self.cat_and_cont_embed.output_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims + [pred_dim]
        self.bayesian_tab_mlp = BayesianMLP(mlp_hidden_dims, mlp_activation, True, prior_sigma_1, prior_sigma_2, prior_pi, posterior_mu_init, posterior_rho_init)

    def forward(self, X: Tensor) ->Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = self.cat_embed_act_fn(x_cat) if self.cat_embed_act_fn is not None else x_cat
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return self.bayesian_tab_mlp(x)


class MSELoss(nn.Module):
    """Mean square error loss with the option of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = MSELoss()(input, target, lds_weight)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class MSLELoss(nn.Module):
    """Mean square log error loss with the option of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = MSLELoss()(input, target, lds_weight)
        """
        assert input.min() >= 0, """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, 'All target values must be >=0'
        loss = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class RMSELoss(nn.Module):
    """Root mean square error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = RMSELoss()(input, target, lds_weight)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class RMSLELoss(nn.Module):
    """Root mean square log error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = RMSLELoss()(input, target, lds_weight)
        """
        assert input.min() >= 0, """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, 'All target values must be >=0'
        loss = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))


class QuantileLoss(nn.Module):
    """Quantile loss defined as:

    $$
    Loss = max(q \\times (y-y_{pred}), (1-q) \\times (y_{pred}-y))
    $$

    All credits go to the implementation at
    [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss).

    Parameters
    ----------
    quantiles: List, default = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        List of quantiles
    """

    def __init__(self, quantiles: List[float]=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import QuantileLoss
        >>>
        >>> # REGRESSION
        >>> target = torch.tensor([[0.6, 1.5]]).view(-1, 1)
        >>> input = torch.tensor([[.1, .2,], [.4, .5]])
        >>> qloss = QuantileLoss([0.25, 0.75])
        >>> loss = qloss(input, target)
        """
        assert input.shape == torch.Size([target.shape[0], len(self.quantiles)]), f'The input and target have inconsistent shape. The dimension of the prediction of the model that is using QuantileLoss must be equal to number of quantiles, i.e. {len(self.quantiles)}.'
        target = target.view(-1, 1).float()
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - input[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        loss = torch.cat(losses, dim=2)
        return torch.mean(loss)


use_cuda = torch.cuda.is_available()


class FocalLoss(nn.Module):
    """Implementation of the [Focal loss](https://arxiv.org/pdf/1708.02002.pdf)
    for both binary and multiclass classification:

    $$
    FL(p_t) = \\alpha (1 - p_t)^{\\gamma} log(p_t)
    $$

    where, for a case of a binary classification problem

    $$
    \\begin{equation} p_t= \\begin{cases}p, & \\text{if $y=1$}.\\\\1-p, & \\text{otherwise}. \\end{cases} \\end{equation}
    $$

    Parameters
    ----------
    alpha: float
        Focal Loss `alpha` parameter
    gamma: float
        Focal Loss `gamma` parameter
    """

    def __init__(self, alpha: float=0.25, gamma: float=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _get_weight(self, p: Tensor, t: Tensor) ->Tensor:
        pt = p * t + (1 - p) * (1 - t)
        w = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (w * (1 - pt).pow(self.gamma)).detach()

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import FocalLoss
        >>>
        >>> # BINARY
        >>> target = torch.tensor([0, 1, 0, 1]).view(-1, 1)
        >>> input = torch.tensor([[0.6, 0.7, 0.3, 0.8]]).t()
        >>> loss = FocalLoss()(input, target)
        >>>
        >>> # MULTICLASS
        >>> target = torch.tensor([1, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
        >>> loss = FocalLoss()(input, target)
        """
        input_prob = torch.sigmoid(input)
        if input.size(1) == 1:
            input_prob = torch.cat([1 - input_prob, input_prob], axis=1)
            num_class = 2
        else:
            num_class = input_prob.size(1)
        binary_target = torch.eye(num_class)[target.squeeze().long()]
        if use_cuda:
            binary_target = binary_target
        binary_target = binary_target.contiguous()
        weight = self._get_weight(input_prob, binary_target)
        return F.binary_cross_entropy(input_prob, binary_target, weight, reduction='mean')


class BayesianRegressionLoss(nn.Module):
    """log Gaussian loss as specified in the original publication 'Weight
    Uncertainty in Neural Networks'
    Currently we do not use this loss as is proportional to the
    `BayesianSELoss` and the latter does not need a scale/noise_tolerance
    param
    """

    def __init__(self, noise_tolerance: float):
        super().__init__()
        self.noise_tolerance = noise_tolerance

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        return -torch.distributions.Normal(input, self.noise_tolerance).log_prob(target).sum()


class BayesianSELoss(nn.Module):
    """Squared Loss (log Gaussian) for the case of a regression as specified in
    the original publication
    [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import BayesianSELoss
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> loss = BayesianSELoss()(input, target)
        """
        return (0.5 * (input - target) ** 2).sum()


class TweedieLoss(nn.Module):
    """
    Tweedie loss for extremely unbalanced zero-inflated data

    All credits go to Wenbo Shi. See
    [this post](https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f)
    and the [original publication](https://arxiv.org/abs/1811.10192) for details.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None, p: float=1.5) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.
        p: float, default = 1.5
            the power to be used to compute the loss. See the original
            publication for details

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import TweedieLoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = TweedieLoss()(input, target, lds_weight)
        """
        assert input.min() > 0, """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, 'All target values must be >=0'
        loss = -target * torch.pow(input, 1 - p) / (1 - p) + torch.pow(input, 2 - p) / (2 - p)
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class ZILNLoss(nn.Module):
    """Adjusted implementation of the Zero Inflated LogNormal Loss

    See [A Deep Probabilistic Model for Customer Lifetime Value Prediction](https://arxiv.org/pdf/1912.07753.pdf)
    and the corresponding
    [code](https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions with spape (N,3), where N is the batch size
        target: Tensor
            Target tensor with the actual target values

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import ZILNLoss
        >>>
        >>> target = torch.tensor([[0., 1.5]]).view(-1, 1)
        >>> input = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        >>> loss = ZILNLoss()(input, target)
        """
        positive = target > 0
        positive = positive.float()
        assert input.shape == torch.Size([target.shape[0], 3]), "Wrong shape of the 'input' tensor. The pred_dim of the model that is using ZILNLoss must be equal to 3."
        positive_input = input[..., :1]
        classification_loss = F.binary_cross_entropy_with_logits(positive_input, positive, reduction='none').flatten()
        loc = input[..., 1:2]
        max_input = F.softplus(input[..., 2:])
        max_other = torch.sqrt(torch.Tensor([torch.finfo(torch.double).eps])).type(max_input.type())
        scale = torch.max(max_input, max_other)
        safe_labels = positive * target + (1 - positive) * torch.ones_like(target)
        regression_loss = -torch.mean(positive * torch.distributions.log_normal.LogNormal(loc=loc, scale=scale).log_prob(safe_labels), dim=-1)
        return torch.mean(classification_loss + regression_loss)


class L1Loss(nn.Module):
    """L1 loss adjusted for the possibility of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions
        target: Tensor
            Target tensor with the actual values
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import L1Loss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> loss = L1Loss()(input, target)
        """
        loss = F.l1_loss(input, target, reduction='none')
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class HuberLoss(nn.Module):
    """Hubbler Loss

    Based on [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self, beta: float=0.2):
        super().__init__()
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor]=None) ->Tensor:
        """
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            If we choose to use LDS this is the tensor of weights that will
            multiply the loss value.

        Examples
        --------
        >>> import torch
        >>>
        >>> from pytorch_widedeep.losses import HuberLoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> loss = HuberLoss()(input, target)
        """
        l1_loss = torch.abs(input - target)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / self.beta, l1_loss - 0.5 * self.beta)
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)


class InfoNCELoss(nn.Module):
    """InfoNCE Loss. Loss applied during the Contrastive Denoising Self
    Supervised Pre-training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    See [SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) and
    references therein

    Partially inspired by the code in this [repo](https://github.com/RElbers/info-nce-pytorch)

    Parameters:
    -----------
    temperature: float, default = 0.1
        The logits are divided by the temperature before computing the loss value
    reduction: str, default = "mean"
        Loss reduction method
    """

    def __init__(self, temperature: float=0.1, reduction: str='mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, g_projs: Tuple[Tensor, Tensor]) ->Tensor:
        """
        Parameters
        ----------
        g_projs: Tuple
            Tuple with the two tensors corresponding to the output of the two
            projection heads, as described 'SAINT: Improved Neural Networks
            for Tabular Data via Row Attention and Contrastive Pre-Training'.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import InfoNCELoss
        >>> g_projs = (torch.rand(5, 5), torch.rand(5, 5))
        >>> loss = InfoNCELoss()
        >>> res = loss(g_projs)
        """
        z, z_ = g_projs[0], g_projs[1]
        norm_z = F.normalize(z, dim=-1).flatten(1)
        norm_z_ = F.normalize(z_, dim=-1).flatten(1)
        logits = norm_z @ norm_z_.t() / self.temperature
        logits_ = norm_z_ @ norm_z.t() / self.temperature
        target = torch.arange(len(norm_z), device=norm_z.device)
        loss = F.cross_entropy(logits, target, reduction=self.reduction)
        loss_ = F.cross_entropy(logits_, target, reduction=self.reduction)
        return (loss + loss_) / 2.0


class DenoisingLoss(nn.Module):
    """Denoising Loss. Loss applied during the Contrastive Denoising Self
    Supervised Pre-training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    See [SAINT: Improved Neural Networks for Tabular Data via Row Attention
    and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) and
    references therein

    Parameters:
    -----------
    lambda_cat: float, default = 1.
        Multiplicative factor that will be applied to loss associated to the
        categorical features
    lambda_cont: float, default = 1.
        Multiplicative factor that will be applied to loss associated to the
        continuous features
    reduction: str, default = "mean"
        Loss reduction method
    """

    def __init__(self, lambda_cat: float=1.0, lambda_cont: float=1.0, reduction: str='mean'):
        super(DenoisingLoss, self).__init__()
        self.lambda_cat = lambda_cat
        self.lambda_cont = lambda_cont
        self.reduction = reduction

    def forward(self, x_cat_and_cat_: Optional[Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]], x_cont_and_cont_: Optional[Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]]) ->Tensor:
        """
        Parameters
        ----------
        x_cat_and_cat_: tuple of Tensors or lists of tuples
            Tuple of tensors containing the raw input features and their
            encodings, referred in the SAINT paper as $x$ and $x''$
            respectively. If one denoising MLP is used per categorical
            feature `x_cat_and_cat_` will be a list of tuples, one per
            categorical feature
        x_cont_and_cont_: tuple of Tensors or lists of tuples
            same as `x_cat_and_cat_` but for continuous columns

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import DenoisingLoss
        >>> x_cat_and_cat_ = (torch.empty(3).random_(3).long(), torch.randn(3, 3))
        >>> x_cont_and_cont_ = (torch.randn(3, 1), torch.randn(3, 1))
        >>> loss = DenoisingLoss()
        >>> res = loss(x_cat_and_cat_, x_cont_and_cont_)
        """
        loss_cat = self._compute_cat_loss(x_cat_and_cat_) if x_cat_and_cat_ is not None else torch.tensor(0.0)
        loss_cont = self._compute_cont_loss(x_cont_and_cont_) if x_cont_and_cont_ is not None else torch.tensor(0.0)
        return self.lambda_cat * loss_cat + self.lambda_cont * loss_cont

    def _compute_cat_loss(self, x_cat_and_cat_: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]) ->Tensor:
        loss_cat = torch.tensor(0.0, device=self._get_device(x_cat_and_cat_))
        if isinstance(x_cat_and_cat_, list):
            for x, x_ in x_cat_and_cat_:
                loss_cat += F.cross_entropy(x_, x, reduction=self.reduction)
        elif isinstance(x_cat_and_cat_, tuple):
            x, x_ = x_cat_and_cat_
            loss_cat += F.cross_entropy(x_, x, reduction=self.reduction)
        return loss_cat

    def _compute_cont_loss(self, x_cont_and_cont_) ->Tensor:
        loss_cont = torch.tensor(0.0, device=self._get_device(x_cont_and_cont_))
        if isinstance(x_cont_and_cont_, list):
            for x, x_ in x_cont_and_cont_:
                loss_cont += F.mse_loss(x_, x, reduction=self.reduction)
        elif isinstance(x_cont_and_cont_, tuple):
            x, x_ = x_cont_and_cont_
            loss_cont += F.mse_loss(x_, x, reduction=self.reduction)
        return loss_cont

    @staticmethod
    def _get_device(x_and_x_: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]):
        if isinstance(x_and_x_, tuple):
            device = x_and_x_[0].device
        elif isinstance(x_and_x_, list):
            device = x_and_x_[0][0].device
        return device


class EncoderDecoderLoss(nn.Module):
    """'_Standard_' Encoder Decoder Loss. Loss applied during the Endoder-Decoder
     Self-Supervised Pre-Training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    The implementation of this lost is based on that at the
    [tabnet repo](https://github.com/dreamquark-ai/tabnet), which is in itself an
    adaptation of that in the original paper [TabNet: Attentive
    Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442).

    Parameters:
    -----------
    eps: float
        Simply a small number to avoid dividing by zero
    """

    def __init__(self, eps: float=1e-09):
        super(EncoderDecoderLoss, self).__init__()
        self.eps = eps

    def forward(self, x_true: Tensor, x_pred: Tensor, mask: Tensor) ->Tensor:
        """
        Parameters
        ----------
        x_true: Tensor
            Embeddings of the input data
        x_pred: Tensor
            Reconstructed embeddings
        mask: Tensor
            Mask with 1s indicated that the reconstruction, and therefore the
            loss, is based on those features.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import EncoderDecoderLoss
        >>> x_true = torch.rand(3, 3)
        >>> x_pred = torch.rand(3, 3)
        >>> mask = torch.empty(3, 3).random_(2)
        >>> loss = EncoderDecoderLoss()
        >>> res = loss(x_true, x_pred, mask)
        """
        errors = x_pred - x_true
        reconstruction_errors = torch.mul(errors, mask) ** 2
        x_true_means = torch.mean(x_true, dim=0)
        x_true_means[x_true_means == 0] = 1
        x_true_stds = torch.std(x_true, dim=0) ** 2
        x_true_stds[x_true_stds == 0] = x_true_means[x_true_stds == 0]
        features_loss = torch.matmul(reconstruction_errors, 1 / x_true_stds)
        nb_reconstructed_variables = torch.sum(mask, dim=1)
        features_loss_norm = features_loss / (nb_reconstructed_variables + self.eps)
        loss = torch.mean(features_loss_norm)
        return loss


def find_bin(bin_edges: Union[np.ndarray, Tensor], values: Union[np.ndarray, Tensor], ret_value: bool=True) ->Union[np.ndarray, Tensor]:
    """Returns histograms left bin edge value or array indices from monotonically
    increasing array of bin edges for each value in values.
    If ret_value

    Parameters
    ----------
    bin_edges: Union[np.ndarray, Tensor]
        monotonically increasing array of bin edges
    values: Union[np.ndarray, Tensor]
        values for which we want corresponding bins
    ret_value: bool
        if True, return bin values else indices

    Returns
    -------
    left_bin_edges: Union[np.ndarray, Tensor]
        left bin edges
    """
    if type(bin_edges) == np.ndarray and type(values) == np.ndarray:
        indices: Union[np.ndarray, Tensor] = np.searchsorted(bin_edges, values, side='left')
        indices = np.where((indices == 0) | (indices == len(bin_edges)), indices, indices - 1)
        indices = np.where(indices != len(bin_edges), indices, indices - 2)
    elif type(bin_edges) == Tensor and type(values) == Tensor:
        bin_edges = bin_edges
        indices = torch.searchsorted(bin_edges, values, right=False)
        indices = torch.where((indices == 0) | (indices == len(bin_edges)), indices, indices - 1)
        indices = torch.where(indices != len(bin_edges), indices, indices - 2)
    else:
        raise TypeError('Both input arrays must be of teh same type, either np.ndarray of Tensor')
    return indices if not ret_value else bin_edges[indices]


def _laplace(x, sigma: Union[int, float]=2):
    return np.exp(-abs(x) / sigma) / (2.0 * sigma)


def set_default_attr(obj: Any, name: str, value: Any):
    """Set the `name` attribute of `obj` to `value` if the attribute does not
    already exist

    Parameters
    ----------
    obj: Object
        Object whose `name` attribute will be returned (after setting it to
        `value`, if necessary)
    name: String
        Name of the attribute to set to `value`, or to return
    value: Object
        Default value to give to `obj.name` if the attribute does not already
        exist

    Returns
    -------
    Object
        `obj.name` if it exists. Else, `value`

    Examples
    --------
    >>> foo = type("Foo", tuple(), {"my_attr": 32})
    >>> set_default_attr(foo, "my_attr", 99)
    32
    >>> set_default_attr(foo, "other_attr", 9000)
    9000
    >>> assert foo.my_attr == 32
    >>> assert foo.other_attr == 9000
    """
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value


def dense_layer(inp: int, out: int, activation: str, p: float, bn: bool, linear_first: bool):
    act_fn = get_activation_fn(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)


class MLP(nn.Module):

    def __init__(self, d_hidden: List[int], activation: str, dropout: Optional[Union[float, List[float]]], batchnorm: bool, batchnorm_last: bool, linear_first: bool):
        super(MLP, self).__init__()
        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)
        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module('dense_layer_{}'.format(i - 1), dense_layer(d_hidden[i - 1], d_hidden[i], activation, dropout[i - 1], batchnorm and (i != len(d_hidden) - 1 or batchnorm_last), linear_first))

    def forward(self, X: Tensor) ->Tensor:
        return self.mlp(X)


allowed_pretrained_models = ['resnet', 'shufflenet', 'resnext', 'wide_resnet', 'regnet', 'densenet', 'mobilenet', 'mnasnet', 'efficientnet', 'squeezenet']


def conv_layer(ni: int, nf: int, kernel_size: int=3, stride: int=1, maxpool: bool=True, adaptiveavgpool: bool=False):
    layer = nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, bias=False, padding=kernel_size // 2), nn.BatchNorm2d(nf, momentum=0.01), nn.LeakyReLU(negative_slope=0.1, inplace=True))
    if maxpool:
        layer.add_module('maxpool', nn.MaxPool2d(2, 2))
    if adaptiveavgpool:
        layer.add_module('adaptiveavgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layer


class ContEmbeddings(nn.Module):

    def __init__(self, n_cont_cols: int, embed_dim: int, embed_dropout: float, use_bias: bool):
        super(ContEmbeddings, self).__init__()
        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.use_bias = use_bias
        self.weight = nn.init.kaiming_uniform_(nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)), a=math.sqrt(5))
        self.bias = nn.init.kaiming_uniform_(nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)), a=math.sqrt(5)) if use_bias else None

    def forward(self, X: Tensor) ->Tensor:
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return F.dropout(x, self.embed_dropout, self.training)

    def extra_repr(self) ->str:
        s = '{n_cont_cols}, {embed_dim}, embed_dropout={embed_dropout}, use_bias={use_bias}'
        return s.format(**self.__dict__)


class DiffSizeCatEmbeddings(nn.Module):

    def __init__(self, column_idx: Dict[str, int], embed_input: List[Tuple[str, int, int]], embed_dropout: float, use_bias: bool):
        super(DiffSizeCatEmbeddings, self).__init__()
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.use_bias = use_bias
        self.embed_layers_names = None
        if self.embed_input is not None:
            self.embed_layers_names = {e[0]: e[0].replace('.', '_') for e in self.embed_input}
        self.embed_layers = nn.ModuleDict({('emb_layer_' + self.embed_layers_names[col]): nn.Embedding(val + 1, dim, padding_idx=0) for col, val, dim in self.embed_input})
        self.embedding_dropout = nn.Dropout(embed_dropout)
        if use_bias:
            self.biases = nn.ParameterDict()
            for col, _, dim in self.embed_input:
                bound = 1 / math.sqrt(dim)
                self.biases['bias_' + col] = nn.Parameter(nn.init.uniform_(torch.Tensor(dim), -bound, bound))
        self.emb_out_dim: int = int(np.sum([embed[2] for embed in self.embed_input]))

    def forward(self, X: Tensor) ->Tensor:
        embed = [(self.embed_layers['emb_layer_' + self.embed_layers_names[col]](X[:, self.column_idx[col]].long()) + (self.biases['bias_' + col].unsqueeze(0) if self.use_bias else torch.zeros(1, dim, device=X.device))) for col, _, dim in self.embed_input]
        x = torch.cat(embed, 1)
        x = self.embedding_dropout(x)
        return x


class DiffSizeCatAndContEmbeddings(nn.Module):

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: List[Tuple[str, int, int]], cat_embed_dropout: float, use_cat_bias: bool, continuous_cols: Optional[List[str]], cont_norm_layer: str, embed_continuous: bool, cont_embed_dim: int, cont_embed_dropout: float, use_cont_bias: bool):
        super(DiffSizeCatAndContEmbeddings, self).__init__()
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        if self.cat_embed_input is not None:
            self.cat_embed = DiffSizeCatEmbeddings(column_idx, cat_embed_input, cat_embed_dropout, use_cat_bias)
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == 'layernorm':
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == 'batchnorm':
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = ContEmbeddings(len(continuous_cols), cont_embed_dim, cont_embed_dropout, use_cont_bias)
                self.cont_out_dim = len(continuous_cols) * cont_embed_dim
            else:
                self.cont_out_dim = len(continuous_cols)
        else:
            self.cont_out_dim = 0
        self.output_dim = self.cat_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) ->Tuple[Tensor, Any]:
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None
        if self.continuous_cols is not None:
            x_cont = self.cont_norm(X[:, self.cont_idx].float())
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, 'b s d -> b (s d)')
        else:
            x_cont = None
        return x_cat, x_cont


class BaseTabularModelWithoutAttention(nn.Module):

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int, int]]], cat_embed_dropout: float, use_cat_bias: bool, cat_embed_activation: Optional[str], continuous_cols: Optional[List[str]], cont_norm_layer: str, embed_continuous: bool, cont_embed_dim: int, cont_embed_dropout: float, use_cont_bias: bool, cont_embed_activation: Optional[str]):
        super().__init__()
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation
        self.cat_and_cont_embed = DiffSizeCatAndContEmbeddings(column_idx, cat_embed_input, cat_embed_dropout, use_cat_bias, continuous_cols, cont_norm_layer, embed_continuous, cont_embed_dim, cont_embed_dropout, use_cont_bias)
        self.cat_embed_act_fn = get_activation_fn(cat_embed_activation) if cat_embed_activation is not None else None
        self.cont_embed_act_fn = get_activation_fn(cont_embed_activation) if cont_embed_activation is not None else None

    def _get_embeddings(self, X: Tensor) ->Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = self.cat_embed_act_fn(x_cat) if self.cat_embed_act_fn is not None else x_cat
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return x

    @property
    def output_dim(self) ->int:
        raise NotImplementedError


class FullEmbeddingDropout(nn.Module):

    def __init__(self, p: float):
        super(FullEmbeddingDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f'p probability has to be between 0 and 1, but got {p}')
        self.p = p

    def forward(self, X: Tensor) ->Tensor:
        if self.training:
            mask = X.new().resize_((X.size(1), 1)).bernoulli_(1 - self.p).expand_as(X) / (1 - self.p)
            return mask * X
        else:
            return X

    def extra_repr(self) ->str:
        return f'p={self.p}'


DropoutLayers = Union[nn.Dropout, FullEmbeddingDropout]


class SharedEmbeddings(nn.Module):

    def __init__(self, n_embed: int, embed_dim: int, embed_dropout: float, full_embed_dropout: bool=False, add_shared_embed: bool=False, frac_shared_embed=0.25):
        super(SharedEmbeddings, self).__init__()
        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(n_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))
        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def forward(self, X: Tensor) ->Tensor:
        out = self.dropout(self.embed(X))
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, :shared_embed.shape[1]] = shared_embed
        return out


class SameSizeCatEmbeddings(nn.Module):

    def __init__(self, embed_dim: int, column_idx: Dict[str, int], embed_input: Optional[List[Tuple[str, int]]], embed_dropout: float, use_bias: bool, full_embed_dropout: bool, shared_embed: bool, add_shared_embed: bool, frac_shared_embed: float):
        super(SameSizeCatEmbeddings, self).__init__()
        self.n_tokens = sum([ei[1] for ei in embed_input])
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.shared_embed = shared_embed
        self.with_cls_token = 'cls_token' in column_idx
        self.embed_layers_names = None
        if self.embed_input is not None:
            self.embed_layers_names = {e[0]: e[0].replace('.', '_') for e in self.embed_input}
        categorical_cols = [ei[0] for ei in embed_input]
        self.cat_idx = [self.column_idx[col] for col in categorical_cols]
        if use_bias:
            if shared_embed:
                warnings.warn("The current implementation of 'SharedEmbeddings' does not use bias", UserWarning)
            n_cat = len(categorical_cols) - 1 if self.with_cls_token else len(categorical_cols)
            self.bias = nn.init.kaiming_uniform_(nn.Parameter(torch.Tensor(n_cat, embed_dim)), a=math.sqrt(5))
        else:
            self.bias = None
        if self.shared_embed:
            self.embed: Union[nn.ModuleDict, nn.Embedding] = nn.ModuleDict({('emb_layer_' + self.embed_layers_names[col]): SharedEmbeddings(val if col == 'cls_token' else val + 1, embed_dim, embed_dropout, full_embed_dropout, add_shared_embed, frac_shared_embed) for col, val in self.embed_input})
        else:
            n_tokens = sum([ei[1] for ei in embed_input])
            self.embed = nn.Embedding(n_tokens + 1, embed_dim, padding_idx=0)
            if full_embed_dropout:
                self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
            else:
                self.dropout = nn.Dropout(embed_dropout)

    def forward(self, X: Tensor) ->Tensor:
        if self.shared_embed:
            cat_embed = [self.embed['emb_layer_' + self.embed_layers_names[col]](X[:, self.column_idx[col]].long()).unsqueeze(1) for col, _ in self.embed_input]
            x = torch.cat(cat_embed, 1)
        else:
            x = self.embed(X[:, self.cat_idx].long())
            if self.bias is not None:
                if self.with_cls_token:
                    bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
                else:
                    bias = self.bias
                x = x + bias.unsqueeze(0)
            x = self.dropout(x)
        return x


class SameSizeCatAndContEmbeddings(nn.Module):

    def __init__(self, embed_dim: int, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]], cat_embed_dropout: float, use_cat_bias: bool, full_embed_dropout: bool, shared_embed: bool, add_shared_embed: bool, frac_shared_embed: float, continuous_cols: Optional[List[str]], cont_norm_layer: str, embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool):
        super(SameSizeCatAndContEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        if cat_embed_input is not None:
            self.cat_embed = SameSizeCatEmbeddings(embed_dim, column_idx, cat_embed_input, cat_embed_dropout, use_cat_bias, full_embed_dropout, shared_embed, add_shared_embed, frac_shared_embed)
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == 'layernorm':
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == 'batchnorm':
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = ContEmbeddings(len(continuous_cols), embed_dim, cont_embed_dropout, use_cont_bias)

    def forward(self, X: Tensor) ->Tuple[Tensor, Any]:
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None
        if self.continuous_cols is not None:
            x_cont = self.cont_norm(X[:, self.cont_idx].float())
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
        else:
            x_cont = None
        return x_cat, x_cont


class BaseTabularModelWithAttention(nn.Module):

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]], cat_embed_dropout: float, use_cat_bias: bool, cat_embed_activation: Optional[str], full_embed_dropout: bool, shared_embed: bool, add_shared_embed: bool, frac_shared_embed: float, continuous_cols: Optional[List[str]], cont_norm_layer: str, embed_continuous: bool, cont_embed_dropout: float, use_cont_bias: bool, cont_embed_activation: Optional[str], input_dim: int):
        super().__init__()
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation
        self.input_dim = input_dim
        self.cat_and_cont_embed = SameSizeCatAndContEmbeddings(input_dim, column_idx, cat_embed_input, cat_embed_dropout, use_cat_bias, full_embed_dropout, shared_embed, add_shared_embed, frac_shared_embed, continuous_cols, cont_norm_layer, embed_continuous, cont_embed_dropout, use_cont_bias)
        self.cat_embed_act_fn = get_activation_fn(cat_embed_activation) if cat_embed_activation is not None else None
        self.cont_embed_act_fn = get_activation_fn(cont_embed_activation) if cont_embed_activation is not None else None

    def _get_embeddings(self, X: Tensor) ->Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = self.cat_embed_act_fn(x_cat) if self.cat_embed_act_fn is not None else x_cat
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return x

    @property
    def output_dim(self) ->int:
        raise NotImplementedError

    @property
    def attention_weights(self):
        raise NotImplementedError


class Wide(nn.Module):
    """Defines a `Wide` (linear) model where the non-linearities are
    captured via the so-called crossed-columns. This can be used as the
    `wide` component of a Wide & Deep model.

    Parameters
    -----------
    input_dim: int
        size of the Embedding layer. `input_dim` is the summation of all the
        individual values for all the features that go through the wide
        model. For example, if the wide model receives 2 features with
        5 individual values each, `input_dim = 10`
    pred_dim: int, default = 1
        size of the ouput tensor containing the predictions. Note that unlike
        all the other models, the wide model is connected directly to the
        output neuron(s) when used to build a Wide and Deep model. Therefore,
        it requires the `pred_dim` parameter.

    Attributes
    -----------
    wide_linear: nn.Module
        the linear layer that comprises the wide branch of the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Wide
    >>> X = torch.empty(4, 4).random_(6)
    >>> wide = Wide(input_dim=X.unique().size(0), pred_dim=1)
    >>> out = wide(X)
    """

    def __init__(self, input_dim: int, pred_dim: int=1):
        super(Wide, self).__init__()
        self.input_dim = input_dim
        self.pred_dim = pred_dim
        self.wide_linear = nn.Embedding(input_dim + 1, pred_dim, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(pred_dim))
        self._reset_parameters()

    def _reset_parameters(self) ->None:
        """initialize Embedding and bias like nn.Linear. See [original
        implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear).
        """
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) ->Tensor:
        """Forward pass. Simply connecting the Embedding layer with the ouput
        neuron(s)"""
        out = self.wide_linear(X.long()).sum(dim=1) + self.bias
        return out


class ContextAttention(nn.Module):
    """Attention mechanism inspired by `Hierarchical Attention Networks for
    Document Classification
    <https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf>`_
    """

    def __init__(self, input_dim: int, dropout: float, sum_along_seq: bool=False):
        super(ContextAttention, self).__init__()
        self.inp_proj = nn.Linear(input_dim, input_dim)
        self.context = nn.Linear(input_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sum_along_seq = sum_along_seq

    def forward(self, X: Tensor) ->Tensor:
        scores = torch.tanh_(self.inp_proj(X))
        attn_weights = self.context(scores).softmax(dim=1)
        self.attn_weights = attn_weights.squeeze(2)
        attn_weights = self.dropout(attn_weights)
        output = (attn_weights * X).sum(1) if self.sum_along_seq else attn_weights * X
        return output


class QueryKeySelfAttention(nn.Module):
    """Attention mechanism inspired by the well known multi-head attention. Here,
    rather than learning a value projection matrix that will be multiplied by
    the attention weights, we multiply such weights directly by the input
    tensor.

    The rationale behind this implementation comes, among other
    considerations, from the fact that Transformer based models tend to
    heavily overfit tabular. Therefore, by reducing the number of trainable
    parameters and multiply directly by the incoming tensor we help
    mitigating such overfitting
    """

    def __init__(self, input_dim: int, dropout: float, use_bias: bool, n_heads: int):
        super(QueryKeySelfAttention, self).__init__()
        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.qk_proj = nn.Linear(input_dim, input_dim * 2, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor) ->Tensor:
        q, k = self.qk_proj(X).chunk(2, dim=-1)
        q, k, x_rearr = map(lambda t: einops.rearrange(t, 'b m (h d) -> b h m d', h=self.n_heads), (q, k, X))
        scores = einsum('b h s d, b h l d -> b h s l', q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum('b h s l, b h l d -> b h s d', attn_weights, x_rearr)
        output = einops.rearrange(attn_output, 'b h s d -> b s (h d)', h=self.n_heads)
        return output


class AddNorm(nn.Module):
    """aka PosNorm"""

    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, sublayer: nn.Module) ->Tensor:
        return self.ln(X + self.dropout(sublayer(X)))


class ContextAttentionEncoder(nn.Module):

    def __init__(self, rnn: nn.Module, input_dim: int, attn_dropout: float, attn_concatenate: bool, with_addnorm: bool, sum_along_seq: bool):
        super(ContextAttentionEncoder, self).__init__()
        self.rnn = rnn
        self.bidirectional = self.rnn.bidirectional
        self.attn_concatenate = attn_concatenate
        self.with_addnorm = with_addnorm
        if with_addnorm:
            self.attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.attn = ContextAttention(input_dim, attn_dropout, sum_along_seq)

    def forward(self, X: Tensor, h: Tensor, c: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        if isinstance(self.rnn, nn.LSTM):
            o, (h, c) = self.rnn(X, (h, c))
        elif isinstance(self.rnn, nn.GRU):
            o, h = self.rnn(X, h)
        attn_inp = self._process_rnn_outputs(o, h)
        if self.with_addnorm:
            out = self.attn_addnorm(attn_inp, self.attn)
        else:
            out = self.attn(attn_inp)
        return out, c, h

    def _process_rnn_outputs(self, output: Tensor, hidden: Tensor) ->Tensor:
        if self.attn_concatenate:
            if self.bidirectional:
                bi_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                attn_inp = torch.cat([output, bi_hidden.unsqueeze(1).expand_as(output)], dim=2)
            else:
                attn_inp = torch.cat([output, hidden[-1].unsqueeze(1).expand_as(output)], dim=2)
        else:
            attn_inp = output
        return attn_inp


class SLP(nn.Module):

    def __init__(self, input_dim: int, dropout: float, activation: str, normalise: bool):
        super(SLP, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim * 2 if activation.endswith('glu') else input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        if normalise:
            self.norm: Union[nn.LayerNorm, nn.Identity] = nn.LayerNorm(input_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, X: Tensor) ->Tensor:
        return self.dropout(self.norm(self.activation(self.lin(X))))


class SelfAttentionEncoder(nn.Module):

    def __init__(self, input_dim: int, dropout: float, use_bias: bool, n_heads: int, with_addnorm: bool, activation: str):
        super(SelfAttentionEncoder, self).__init__()
        self.with_addnorm = with_addnorm
        self.attn = QueryKeySelfAttention(input_dim, dropout, use_bias, n_heads)
        if with_addnorm:
            self.attn_addnorm = AddNorm(input_dim, dropout)
            self.slp_addnorm = AddNorm(input_dim, dropout)
        self.slp = SLP(input_dim, dropout, activation, not with_addnorm)

    def forward(self, X: Tensor) ->Tensor:
        if self.with_addnorm:
            x = self.attn_addnorm(X, self.attn)
            out = self.slp_addnorm(x, self.slp)
        else:
            out = self.slp(self.attn(X))
        return out


class ContextAttentionMLP(BaseTabularModelWithAttention):
    """Defines a `ContextAttentionMLP` model that can be used as the
    `deeptabular` component of a Wide & Deep model or independently by
    itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features that are also embedded. These
    are then passed through a series of attention blocks. Each attention
    block is comprised by a `ContextAttentionEncoder`. Such encoder is in
    part inspired by the attention mechanism described in
    [Hierarchical Attention Networks for Document
    Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf).
    See `pytorch_widedeep.models.tabular.mlp._attention_layers` for details.


    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List
        List of Tuples with the column name and number of unique values per
        categorical columns. e.g. _[('education', 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.embeddings_layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind sharing part of the embeddings per column is to let
        the model learn which column is embedded at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings
        used to encode the categorical and/or continuous columns
    attn_dropout: float, default = 0.2
        Dropout for each attention block
    with_addnorm: bool = False,
        Boolean indicating if residual connections will be used in the
        attention blocks
    attn_activation: str, default = "leaky_relu"
        String indicating the activation function to be applied to the dense
        layer in each attention encoder. _'tanh'_, _'relu'_, _'leaky_relu'_
        and _'gelu'_ are supported.
    n_blocks: int, default = 3
        Number of attention blocks

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of attention encoders.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import ContextAttentionMLP
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = ContextAttentionMLP(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, attn_dropout: float=0.2, with_addnorm: bool=False, attn_activation: str='leaky_relu', n_blocks: int=3):
        super(ContextAttentionMLP, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.attn_dropout = attn_dropout
        self.with_addnorm = with_addnorm
        self.attn_activation = attn_activation
        self.n_blocks = n_blocks
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module('attention_block' + str(i), ContextAttentionEncoder(input_dim, attn_dropout, with_addnorm, attn_activation))

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            out = x[:, 0, :]
        else:
            out = x.flatten(1)
        return out

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.input_dim if self.with_cls_token else (self.n_cat + self.n_cont) * self.input_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights per block

        The shape of the attention weights is $(N, F)$, where $N$ is the batch
        size and $F$ is the number of features/columns in the dataset
        """
        return [blk.attn.attn_weights for blk in self.encoder]


class SelfAttentionMLP(BaseTabularModelWithAttention):
    """Defines a `SelfAttentionMLP` model that can be used as the
    deeptabular component of a Wide & Deep model or independently by
    itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features that are also embedded. These
    are then passed through a series of attention blocks. Each attention
    block is comprised by what we would refer as a simplified
    `SelfAttentionEncoder`. See
    `pytorch_widedeep.models.tabular.mlp._attention_layers` for details. The
    reason to use a simplified version of self attention is because we
    observed that the '_standard_' attention mechanism used in the
    TabTransformer has a notable tendency to overfit.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List
        List of Tuples with the column name and number of unique values per
        categorical column e.g. _[(education, 11), ...]_.
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.embeddings_layers.FullEmbeddingDropout`.
        If full_embed_dropout = True, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The of sharing part of the embeddings per column is to enable the
        model to distinguish the classes in one column from those in the
        other columns. In other words, the idea is to let the model learn
        which column is embedded at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        frac_shared_embed with the shared embeddings.
        See `pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if add_shared_embed
        = False) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    attn_dropout: float, default = 0.2
        Dropout for each attention block
    n_heads: int, default = 8
        Number of attention heads per attention block.
    use_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K projection
        layers.
    with_addnorm: bool = False,
        Boolean indicating if residual connections will be used in the attention blocks
    attn_activation: str, default = "leaky_relu"
        String indicating the activation function to be applied to the dense
        layer in each attention encoder. _'tanh'_, _'relu'_, _'leaky_relu'_
        and _'gelu'_ are supported.
    n_blocks: int, default = 3
        Number of attention blocks

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of attention encoders.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import SelfAttentionMLP
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = SelfAttentionMLP(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, attn_dropout: float=0.2, n_heads: int=8, use_bias: bool=False, with_addnorm: bool=False, attn_activation: str='leaky_relu', n_blocks: int=3):
        super(SelfAttentionMLP, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.use_bias = use_bias
        self.with_addnorm = with_addnorm
        self.attn_activation = attn_activation
        self.n_blocks = n_blocks
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module('attention_block' + str(i), SelfAttentionEncoder(input_dim, attn_dropout, use_bias, n_heads, with_addnorm, attn_activation))

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            out = x[:, 0, :]
        else:
            out = x.flatten(1)
        return out

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the WideDeep class
        """
        return self.input_dim if self.with_cls_token else (self.n_cat + self.n_cont) * self.input_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights per block

        The shape of the attention weights is $(N, H, F, F)$, where $N$ is the
        batch size, $H$ is the number of attention heads and $F$ is the
        number of features/columns in the dataset
        """
        return [blk.attn.attn_weights for blk in self.encoder]


class TabMlp(BaseTabularModelWithoutAttention):
    """Defines a `TabMlp` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of dense layers (i.e. a MLP).

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabMlp` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_.
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_dropout: float or List, default = 0.1
        float or List of floats with the dropout between the dense layers.
        e.g: _[0.5,0.5]_
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        mlp model that will receive the concatenation of the embeddings and
        the continuous columns

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabMlp
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabMlp(mlp_hidden_dims=[8,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str='batchnorm', embed_continuous: bool=False, cont_embed_dim: int=32, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, mlp_hidden_dims: List[int]=[200, 100], mlp_activation: str='relu', mlp_dropout: Union[float, List[float]]=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=False):
        super(TabMlp, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=embed_continuous, cont_embed_dim=cont_embed_dim, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation)
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        mlp_input_dim = self.cat_and_cont_embed.output_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
        self.encoder = MLP(mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        return self.encoder(x)

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class"""
        return self.mlp_hidden_dims[-1]


class TabMlpDecoder(nn.Module):
    """Companion decoder model for the `TabMlp` model (which can be considered
    an encoder itself).

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). The `TabMlpDecoder` will receive the output from the MLP
    and '_reconstruct_' the embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor that needs to be reconstructed.
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_dropout: float or List, default = 0.1
        float or List of floats with the dropout between the dense layers.
        e.g: _[0.5,0.5]_
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    decoder: nn.Module
        mlp model that will receive the output of the encoder

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabMlpDecoder
    >>> x_inp = torch.rand(3, 8)
    >>> decoder = TabMlpDecoder(embed_dim=32, mlp_hidden_dims=[8,16])
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(self, embed_dim: int, mlp_hidden_dims: List[int]=[100, 200], mlp_activation: str='relu', mlp_dropout: Union[float, List[float]]=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=False):
        super(TabMlpDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.decoder = MLP(mlp_hidden_dims + [self.embed_dim], mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)

    def forward(self, X: Tensor) ->Tensor:
        return self.decoder(X)


class BasicBlock(nn.Module):

    def __init__(self, inp: int, out: int, dropout: float=0.0, simplify: bool=False, resize: nn.Module=None):
        super(BasicBlock, self).__init__()
        self.simplify = simplify
        self.resize = resize
        self.lin1 = nn.Linear(inp, out, bias=False)
        self.bn1 = nn.BatchNorm1d(out)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if dropout > 0.0:
            self.dropout = True
            self.dp = nn.Dropout(dropout)
        else:
            self.dropout = False
        if not self.simplify:
            self.lin2 = nn.Linear(out, out, bias=False)
            self.bn2 = nn.BatchNorm1d(out)

    def forward(self, X: Tensor) ->Tensor:
        identity = X
        out = self.lin1(X)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp(out)
        if not self.simplify:
            out = self.lin2(out)
            out = self.bn2(out)
        if self.resize is not None:
            identity = self.resize(X)
        out += identity
        out = self.leaky_relu(out)
        return out


class DenseResnet(nn.Module):

    def __init__(self, input_dim: int, blocks_dims: List[int], dropout: float, simplify: bool):
        super(DenseResnet, self).__init__()
        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(OrderedDict([('lin_inp', nn.Linear(input_dim, blocks_dims[0], bias=False)), ('bn_inp', nn.BatchNorm1d(blocks_dims[0]))]))
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(nn.Linear(blocks_dims[i - 1], blocks_dims[i], bias=False), nn.BatchNorm1d(blocks_dims[i]))
            self.dense_resnet.add_module('block_{}'.format(i - 1), BasicBlock(blocks_dims[i - 1], blocks_dims[i], dropout, simplify, resize))

    def forward(self, X: Tensor) ->Tensor:
        return self.dense_resnet(X)


class TabResnet(BaseTabularModelWithoutAttention):
    """Defines a `TabResnet` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of Resnet blocks. See
    `pytorch_widedeep.models.tab_resnet._layers` for details on the
    structure of each block.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_.
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or `None`.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    blocks_dims: List, default = [200, 100, 100]
        List of integers that define the input and output units of each block.
        For example: _[200, 100, 100]_ will generate 2 blocks. The first will
        receive a tensor of size 200 and output a tensor of size 100, and the
        second will receive a tensor of size 100 and output a tensor of size
        100. See `pytorch_widedeep.models.tab_resnet._layers` for
        details on the structure of each block.
    blocks_dropout: float, default =  0.1
        Block's internal dropout.
    simplify_blocks: bool, default = False,
        Boolean indicating if the simplest possible residual blocks (`X -> [
        [LIN, BN, ACT]  + X ]`) will be used instead of a standard one
        (`X -> [ [LIN1, BN1, ACT1] -> [LIN2, BN2]  + X ]`).
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        _[64, 32]_. If `None` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s).
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        float with the dropout between the dense layers of the MLP.
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
    mlp: nn.Module
        if `mlp_hidden_dims` is `True`, this attribute will be an mlp
        model that will receive the results of the concatenation of the
        embeddings and the continuous columns -- if present --.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabResnet
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabResnet(blocks_dims=[16,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_deep)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str='batchnorm', embed_continuous: bool=False, cont_embed_dim: int=32, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, blocks_dims: List[int]=[200, 100, 100], blocks_dropout: float=0.1, simplify_blocks: bool=False, mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=False):
        super(TabResnet, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=embed_continuous, cont_embed_dim=cont_embed_dim, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation)
        if len(blocks_dims) < 2:
            raise ValueError("'blocks' must contain at least two elements, e.g. [256, 128]")
        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.simplify_blocks = simplify_blocks
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        cat_out_dim = self.cat_and_cont_embed.cat_out_dim
        cont_out_dim = self.cat_and_cont_embed.cont_out_dim
        dense_resnet_input_dim = cat_out_dim + cont_out_dim
        self.encoder = DenseResnet(dense_resnet_input_dim, blocks_dims, blocks_dropout, self.simplify_blocks)
        if self.mlp_hidden_dims is not None:
            mlp_hidden_dims = [self.blocks_dims[-1]] + mlp_hidden_dims
            self.mlp = MLP(mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.blocks_dims[-1]


class TabResnetDecoder(nn.Module):
    """Companion decoder model for the `TabResnet` model (which can be
    considered an encoder itself)

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). This class will receive the output from the ResNet blocks or the
    MLP(if present) and '_reconstruct_' the embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor to be reconstructed.
    blocks_dims: List, default = [200, 100, 100]
        List of integers that define the input and output units of each block.
        For example: _[200, 100, 100]_ will generate 2 blocks. The first will
        receive a tensor of size 200 and output a tensor of size 100, and the
        second will receive a tensor of size 100 and output a tensor of size
        100. See `pytorch_widedeep.models.tab_resnet._layers` for
        details on the structure of each block.
    blocks_dropout: float, default =  0.1
        Block's internal dropout.
    simplify_blocks: bool, default = False,
        Boolean indicating if the simplest possible residual blocks (`X -> [
        [LIN, BN, ACT]  + X ]`) will be used instead of a standard one
        (`X -> [ [LIN1, BN1, ACT1] -> [LIN2, BN2]  + X ]`).
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        _[64, 32]_. If `None` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s).
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        float with the dropout between the dense layers of the MLP.
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    decoder: nn.Module
        deep dense Resnet model that will receive the output of the encoder IF
        `mlp_hidden_dims` is None
    mlp: nn.Module
        if `mlp_hidden_dims` is not None, the overall decoder will consist
        in an MLP that will receive the output of the encoder followed by the
        deep dense Resnet.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabResnetDecoder
    >>> x_inp = torch.rand(3, 8)
    >>> decoder = TabResnetDecoder(embed_dim=32, blocks_dims=[8, 16, 16])
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(self, embed_dim: int, blocks_dims: List[int]=[100, 100, 200], blocks_dropout: float=0.1, simplify_blocks: bool=False, mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=False):
        super(TabResnetDecoder, self).__init__()
        if len(blocks_dims) < 2:
            raise ValueError("'blocks' must contain at least two elements, e.g. [256, 128]")
        self.embed_dim = embed_dim
        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.simplify_blocks = simplify_blocks
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        if self.mlp_hidden_dims is not None:
            self.mlp = MLP(mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None
        if self.mlp is not None:
            self.decoder = DenseResnet(mlp_hidden_dims[-1], blocks_dims, blocks_dropout, self.simplify_blocks)
        else:
            self.decoder = DenseResnet(blocks_dims[0], blocks_dims, blocks_dropout, self.simplify_blocks)
        self.reconstruction_layer = nn.Linear(blocks_dims[-1], embed_dim, bias=False)

    def forward(self, X: Tensor) ->Tensor:
        x = self.mlp(X) if self.mlp is not None else X
        return self.reconstruction_layer(self.decoder(x))


class CatSingleMlp(nn.Module):
    """Single MLP will be applied to all categorical features"""

    def __init__(self, input_dim: int, cat_embed_input: List[Tuple[str, int]], column_idx: Dict[str, int], activation: str):
        super(CatSingleMlp, self).__init__()
        self.input_dim = input_dim
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.activation = activation
        self.num_class = sum([ei[1] for ei in cat_embed_input if ei[0] != 'cls_token'])
        self.mlp = MLP(d_hidden=[input_dim, self.num_class * 4, self.num_class], activation=activation, dropout=0.0, batchnorm=False, batchnorm_last=False, linear_first=False)

    def forward(self, X: Tensor, r_: Tensor) ->Tuple[Tensor, Tensor]:
        x = torch.cat([X[:, self.column_idx[col]].long() for col, _ in self.cat_embed_input if col != 'cls_token'])
        cat_r_ = torch.cat([r_[:, self.column_idx[col], :] for col, _ in self.cat_embed_input if col != 'cls_token'])
        x_ = self.mlp(cat_r_)
        return x, x_


class CatMlpPerFeature(nn.Module):
    """Dedicated MLP per categorical feature"""

    def __init__(self, input_dim: int, cat_embed_input: List[Tuple[str, int]], column_idx: Dict[str, int], activation: str):
        super(CatMlpPerFeature, self).__init__()
        self.input_dim = input_dim
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.activation = activation
        self.mlp = nn.ModuleDict({('mlp_' + col): MLP(d_hidden=[input_dim, val * 4, val], activation=activation, dropout=0.0, batchnorm=False, batchnorm_last=False, linear_first=False) for col, val in self.cat_embed_input if col != 'cls_token'})

    def forward(self, X: Tensor, r_: Tensor) ->List[Tuple[Tensor, Tensor]]:
        x = [X[:, self.column_idx[col]].long() for col, _ in self.cat_embed_input if col != 'cls_token']
        x_ = [self.mlp['mlp_' + col](r_[:, self.column_idx[col], :]) for col, _ in self.cat_embed_input if col != 'cls_token']
        return list(zip(x, x_))


class ContSingleMlp(nn.Module):
    """Single MLP will be applied to all continuous features"""

    def __init__(self, input_dim: int, continuous_cols: List[str], column_idx: Dict[str, int], activation: str):
        super(ContSingleMlp, self).__init__()
        self.input_dim = input_dim
        self.column_idx = column_idx
        self.continuous_cols = continuous_cols
        self.activation = activation
        self.mlp = MLP(d_hidden=[input_dim, input_dim * 2, 1], activation=activation, dropout=0.0, batchnorm=False, batchnorm_last=False, linear_first=False)

    def forward(self, X: Tensor, r_: Tensor) ->Tuple[Tensor, Tensor]:
        x = torch.cat([X[:, self.column_idx[col]].float() for col in self.continuous_cols]).unsqueeze(1)
        cont_r_ = torch.cat([r_[:, self.column_idx[col], :] for col in self.continuous_cols])
        x_ = self.mlp(cont_r_)
        return x, x_


class ContMlpPerFeature(nn.Module):
    """Dedicated MLP per continuous feature"""

    def __init__(self, input_dim: int, continuous_cols: List[str], column_idx: Dict[str, int], activation: str):
        super(ContMlpPerFeature, self).__init__()
        self.input_dim = input_dim
        self.column_idx = column_idx
        self.continuous_cols = continuous_cols
        self.activation = activation
        self.mlp = nn.ModuleDict({('mlp_' + col): MLP(d_hidden=[input_dim, input_dim * 2, 1], activation=activation, dropout=0.0, batchnorm=False, batchnorm_last=False, linear_first=False) for col in self.continuous_cols})

    def forward(self, X: Tensor, r_: Tensor) ->List[Tuple[Tensor, Tensor]]:
        x = [X[:, self.column_idx[col]].unsqueeze(1).float() for col in self.continuous_cols]
        x_ = [self.mlp['mlp_' + col](r_[:, self.column_idx[col]]) for col in self.continuous_cols]
        return list(zip(x, x_))


class RandomObfuscator(nn.Module):
    """Creates and applies an obfuscation masks

    Note that the class will return a mask tensor with 1s IF the feature value
    is considered for reconstruction

    Parameters:
    ----------
    p: float
        Ratio of features that will be discarded for reconstruction
    """

    def __init__(self, p: float):
        super(RandomObfuscator, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.bernoulli(self.p * torch.ones(x.shape))
        masked_input = torch.mul(1 - mask, x)
        return masked_input, mask


DenoiseMlp = Union[CatSingleMlp, ContSingleMlp, CatMlpPerFeature, ContMlpPerFeature]


class FeedForward(nn.Module):

    def __init__(self, input_dim: int, dropout: float, activation: str, mult: float=4.0):
        super(FeedForward, self).__init__()
        ff_hidden_dim = int(input_dim * mult)
        self.w_1 = nn.Linear(input_dim, ff_hidden_dim * 2 if activation.endswith('glu') else ff_hidden_dim)
        self.w_2 = nn.Linear(ff_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, X: Tensor) ->Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(X))))


class LinearAttention(nn.Module):

    def __init__(self, input_dim: int, n_feats: int, n_heads: int, use_bias: bool, dropout: float, kv_compression_factor: float, kv_sharing: bool):
        super(LinearAttention, self).__init__()
        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        self.n_feats = n_feats
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.kv_compression_factor = kv_compression_factor
        self.share_kv = kv_sharing
        dim_k = int(self.kv_compression_factor * self.n_feats)
        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(input_dim, input_dim * 3, bias=use_bias)
        self.E = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(n_feats, dim_k)))
        if not kv_sharing:
            self.F = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(n_feats, dim_k)))
        else:
            self.F = self.E
        self.out_proj = nn.Linear(input_dim, input_dim, bias=use_bias) if n_heads > 1 else None

    def forward(self, X: Tensor) ->Tensor:
        q, k, v = self.qkv_proj(X).chunk(3, dim=-1)
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        k = einsum('b s i, s k -> b k i', k, self.E)
        v = einsum('b s i, s k -> b k i', v, self.F)
        k = einops.rearrange(k, 'b k (h d) -> b h k d', d=self.head_dim)
        v = einops.rearrange(v, 'b k (h d) -> b h k d', d=self.head_dim)
        scores = einsum('b h s d, b h k d -> b h s k', q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        output = einsum('b h s k, b h k d -> b h s d', attn_weights, v)
        output = einops.rearrange(output, 'b h s d -> b s (h d)')
        if self.out_proj is not None:
            output = self.out_proj(output)
        return output


class NormAdd(nn.Module):
    """aka PreNorm"""

    def __init__(self, input_dim: int, dropout: float):
        super(NormAdd, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, sublayer: nn.Module) ->Tensor:
        return X + self.dropout(sublayer(self.ln(X)))


class FTTransformerEncoder(nn.Module):

    def __init__(self, input_dim: int, n_feats: int, n_heads: int, use_bias: bool, attn_dropout: float, ff_dropout: float, kv_compression_factor: float, kv_sharing: bool, activation: str, ff_factor: float, first_block: bool):
        super(FTTransformerEncoder, self).__init__()
        self.first_block = first_block
        self.attn = LinearAttention(input_dim, n_feats, n_heads, use_bias, attn_dropout, kv_compression_factor, kv_sharing)
        self.ff = FeedForward(input_dim, ff_dropout, activation, ff_factor)
        self.attn_normadd = NormAdd(input_dim, attn_dropout)
        self.ff_normadd = NormAdd(input_dim, ff_dropout)

    def forward(self, X: Tensor) ->Tensor:
        if self.first_block:
            x = X + self.attn(X)
        else:
            x = self.attn_normadd(X, self.attn)
        return self.ff_normadd(x, self.ff)


class FTTransformer(BaseTabularModelWithAttention):
    """Defines a [FTTransformer model](https://arxiv.org/abs/2106.11959) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.


    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name and number of unique values for
        each categorical component e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 64
        The so-called *dimension of the model*. Is the number of embeddings used to encode
        the categorical and/or continuous columns.
    kv_compression_factor: int, default = 0.5
        By default, the FTTransformer uses Linear Attention
        (See [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768>) ).
        The compression factor that will be used to reduce the input sequence
        length. If we denote the resulting sequence length as
        $k = int(kv_{compression \\space factor} \\times s)$
        where $s$ is the input sequence length.
    kv_sharing: bool, default = False
        Boolean indicating if the $E$ and $F$ projection matrices
        will share weights.  See [Linformer: Self-Attention with Linear
        Complexity](https://arxiv.org/abs/2006.04768) for details
    n_heads: int, default = 8
        Number of attention heads per FTTransformer block
    use_qkv_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 4
        Number of FTTransformer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Linear-Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    ff_factor: float, default = 4 / 3
        Multiplicative factor applied to the first layer of the FF network in
        each Transformer block, This is normally set to 4, but they use 4/3
        in the paper.
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided no MLP on top of the final
        FTTransformer block will be used
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of FTTransformer blocks
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import FTTransformer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = FTTransformer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=64, kv_compression_factor: float=0.5, kv_sharing: bool=False, use_qkv_bias: bool=False, n_heads: int=8, n_blocks: int=4, attn_dropout: float=0.2, ff_dropout: float=0.1, transformer_activation: str='reglu', ff_factor: float=1.33, mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=True):
        super(FTTransformer, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.kv_compression_factor = kv_compression_factor
        self.kv_sharing = kv_sharing
        self.use_qkv_bias = use_qkv_bias
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.transformer_activation = transformer_activation
        self.ff_factor = ff_factor
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.n_feats = self.n_cat + self.n_cont
        is_first = True
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module('fttransformer_block' + str(i), FTTransformerEncoder(input_dim, self.n_feats, n_heads, use_qkv_bias, attn_dropout, ff_dropout, kv_compression_factor, kv_sharing, transformer_activation, ff_factor, is_first))
            is_first = False
        self.mlp_first_hidden_dim = self.input_dim if self.with_cls_token else self.n_feats * self.input_dim
        if mlp_hidden_dims is not None:
            self.mlp = MLP([self.mlp_first_hidden_dim] + mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.mlp_first_hidden_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights per block

        The shape of the attention weights is: $(N, H, F, k)$, where $N$ is
        the batch size, $H$ is the number of attention heads, $F$ is the
        number of features/columns and $k$ is the reduced sequence length or
        dimension, i.e. $k = int(kv_{compression \\space factor} \\times s)$
        """
        return [blk.attn.attn_weights for blk in self.encoder]


class MultiHeadedAttention(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, dropout: float, query_dim: Optional[int]=None):
        super(MultiHeadedAttention, self).__init__()
        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        query_dim = query_dim if query_dim is not None else input_dim
        self.q_proj = nn.Linear(query_dim, input_dim, bias=use_bias)
        self.kv_proj = nn.Linear(input_dim, input_dim * 2, bias=use_bias)
        self.out_proj = nn.Linear(input_dim, query_dim, bias=use_bias) if n_heads > 1 else None

    def forward(self, X_Q: Tensor, X_KV: Optional[Tensor]=None) ->Tensor:
        q = self.q_proj(X_Q)
        X_KV = X_KV if X_KV is not None else X_Q
        k, v = self.kv_proj(X_KV).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b m (h d) -> b h m d', h=self.n_heads), (q, k, v))
        scores = einsum('b h s d, b h l d -> b h s l', q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum('b h s l, b h l d -> b h s d', attn_weights, v)
        output = einops.rearrange(attn_output, 'b h s d -> b s (h d)', h=self.n_heads)
        if self.out_proj is not None:
            output = self.out_proj(output)
        return output


class SaintEncoder(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, attn_dropout: float, ff_dropout: float, activation: str, n_feat: int):
        super(SaintEncoder, self).__init__()
        self.n_feat = n_feat
        self.col_attn = MultiHeadedAttention(input_dim, n_heads, use_bias, attn_dropout)
        self.col_attn_ff = FeedForward(input_dim, ff_dropout, activation)
        self.col_attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.col_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)
        self.row_attn = MultiHeadedAttention(n_feat * input_dim, n_heads, use_bias, attn_dropout)
        self.row_attn_ff = FeedForward(n_feat * input_dim, ff_dropout, activation)
        self.row_attn_addnorm = AddNorm(n_feat * input_dim, attn_dropout)
        self.row_attn_ff_addnorm = AddNorm(n_feat * input_dim, ff_dropout)

    def forward(self, X: Tensor) ->Tensor:
        x = self.col_attn_addnorm(X, self.col_attn)
        x = self.col_attn_ff_addnorm(x, self.col_attn_ff)
        x = einops.rearrange(x, 'b n d -> 1 b (n d)')
        x = self.row_attn_addnorm(x, self.row_attn)
        x = self.row_attn_ff_addnorm(x, self.row_attn_ff)
        x = einops.rearrange(x, '1 b (n d) -> b n d', n=self.n_feat)
        return x


class SAINT(BaseTabularModelWithAttention):
    """Defines a [SAINT model](https://arxiv.org/abs/2106.01342) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.

    :information_source: **NOTE**: This is an slightly modified and enhanced
     version of the model described in the paper,

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name and number of unique values and
        embedding dimension. e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    use_qkv_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 2
        Number of SAINT-Transformer blocks.
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention column and
        row layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l, 4
        \\times l, 2 \\times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of SAINT-Transformer blocks
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import SAINT
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = SAINT(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, use_qkv_bias: bool=False, n_heads: int=8, n_blocks: int=2, attn_dropout: float=0.1, ff_dropout: float=0.2, transformer_activation: str='gelu', mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=True):
        super(SAINT, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.use_qkv_bias = use_qkv_bias
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.n_feats = self.n_cat + self.n_cont
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module('saint_block' + str(i), SaintEncoder(input_dim, n_heads, use_qkv_bias, attn_dropout, ff_dropout, transformer_activation, self.n_feats))
        self.mlp_first_hidden_dim = self.input_dim if self.with_cls_token else self.n_feats * self.input_dim
        if mlp_hidden_dims is not None:
            self.mlp = MLP([self.mlp_first_hidden_dim] + mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.mlp_first_hidden_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights. Each element of the list is a tuple
        where the first and the second elements are the column and row
        attention weights respectively

        The shape of the attention weights is:

        - column attention: $(N, H, F, F)$

        - row attention: $(1, H, N, N)$

        where $N$ is the batch size, $H$ is the number of heads and $F$ is the
        number of features/columns in the dataset
        """
        attention_weights = []
        for blk in self.encoder:
            attention_weights.append((blk.col_attn.attn_weights, blk.row_attn.attn_weights))
        return attention_weights


class AdditiveAttention(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, dropout: float, share_qv_weights: bool):
        super(AdditiveAttention, self).__init__()
        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.share_qv_weights = share_qv_weights
        self.dropout = nn.Dropout(dropout)
        if share_qv_weights:
            self.qv_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
        else:
            self.q_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
        self.W_q = nn.Linear(input_dim, n_heads)
        self.W_k = nn.Linear(input_dim, n_heads)
        self.r_out = nn.Linear(input_dim, input_dim)

    def forward(self, X: Tensor) ->Tensor:
        q = self.qv_proj(X) if self.share_qv_weights else self.q_proj(X)
        v = self.qv_proj(X) if self.share_qv_weights else self.v_proj(X)
        k = self.k_proj(X)
        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=1)
        q_r = einops.rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        global_query = einsum(' b s h, b s h d -> b h d', alphas, q_r)
        global_query = einops.rearrange(global_query, 'b h d -> b () (h d)')
        p = k * global_query
        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=1)
        p_r = einops.rearrange(p, 'b s (h d) -> b s h d', h=self.n_heads)
        global_key = einsum(' b s h, b s h d -> b h d', betas, p_r)
        global_key = einops.rearrange(global_key, 'b h d -> b () (h d)')
        u = v * global_key
        self.attn_weights = einops.rearrange(alphas, 'b s h -> b h s'), einops.rearrange(betas, 'b s h -> b h s')
        output = q + self.dropout(self.r_out(u))
        return output


class FastFormerEncoder(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, attn_dropout: float, ff_dropout: float, share_qv_weights: bool, activation: str):
        super(FastFormerEncoder, self).__init__()
        self.attn = AdditiveAttention(input_dim, n_heads, use_bias, attn_dropout, share_qv_weights)
        self.ff = FeedForward(input_dim, ff_dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.ff_addnorm = AddNorm(input_dim, ff_dropout)

    def forward(self, X: Tensor) ->Tensor:
        x = self.attn_addnorm(X, self.attn)
        return self.ff_addnorm(x, self.ff)


class TabFastFormer(BaseTabularModelWithAttention):
    """Defines an adaptation of a [FastFormer](https://arxiv.org/abs/2108.09084)
    that can be used as the `deeptabular` component of a Wide & Deep model
    or independently by itself.

    :information_source: **NOTE**: while there are scientific publications for
     the `TabTransformer`, `SAINT` and `FTTransformer`, the `TabPerceiver`
     and the `TabFastFormer` are our own adaptations of the
     [Perceiver](https://arxiv.org/abs/2103.03206) and the
     [FastFormer](https://arxiv.org/abs/2108.09084) for tabular data.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabFastFormer` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the goal of
        having column embedding is to enable the model to distinguish the
        classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded at
        the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per FastFormer block
    use_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 4
        Number of FastFormer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Additive Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    share_qv_weights: bool, default = False
        Following the paper, this is a boolean indicating if the Value ($V$) and
        the Query ($Q$) transformation parameters will be shared.
    share_weights: bool, default = False
        In addition to sharing the $V$ and $Q$ transformation parameters, the
        parameters across different Fastformer layers can also be shared.
        Please, see
        `pytorch_widedeep/models/tabular/transformers/tab_fastformer.py` for
        details
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l, 4
        \\times l, 2 \\times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of FasFormer blocks.
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabFastFormer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabFastFormer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, n_heads: int=8, use_bias: bool=False, n_blocks: int=4, attn_dropout: float=0.1, ff_dropout: float=0.2, share_qv_weights: bool=False, share_weights: bool=False, transformer_activation: str='relu', mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=True):
        super(TabFastFormer, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.n_heads = n_heads
        self.use_bias = use_bias
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.share_qv_weights = share_qv_weights
        self.share_weights = share_weights
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.n_feats = self.n_cat + self.n_cont
        self.encoder = nn.Sequential()
        first_fastformer_block = FastFormerEncoder(input_dim, n_heads, use_bias, attn_dropout, ff_dropout, share_qv_weights, transformer_activation)
        self.encoder.add_module('fastformer_block0', first_fastformer_block)
        for i in range(1, n_blocks):
            if share_weights:
                self.encoder.add_module('fastformer_block' + str(i), first_fastformer_block)
            else:
                self.encoder.add_module('fastformer_block' + str(i), FastFormerEncoder(input_dim, n_heads, use_bias, attn_dropout, ff_dropout, share_qv_weights, transformer_activation))
        self.mlp_first_hidden_dim = self.input_dim if self.with_cls_token else self.n_feats * self.input_dim
        if mlp_hidden_dims is not None:
            self.mlp = MLP([self.mlp_first_hidden_dim] + mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.mlp_first_hidden_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights. Each element of the list is a
        tuple where the first and second elements are the $\\alpha$
        and $\\beta$ attention weights in the paper.

        The shape of the attention weights is $(N, H, F)$ where $N$ is the
        batch size, $H$ is the number of attention heads and $F$ is the
        number of features/columns in the dataset
        """
        if self.share_weights:
            attention_weights = [self.encoder[0].attn.attn_weight]
        else:
            attention_weights = [blk.attn.attn_weights for blk in self.encoder]
        return attention_weights


class PerceiverEncoder(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, attn_dropout: float, ff_dropout: float, activation: str, query_dim: Optional[int]=None):
        super(PerceiverEncoder, self).__init__()
        self.attn = MultiHeadedAttention(input_dim, n_heads, use_bias, attn_dropout, query_dim)
        attn_dim_out = query_dim if query_dim is not None else input_dim
        self.ff = FeedForward(attn_dim_out, ff_dropout, activation)
        self.ln_q = nn.LayerNorm(attn_dim_out)
        self.ln_kv = nn.LayerNorm(input_dim)
        self.norm_attn_dropout = nn.Dropout(attn_dropout)
        self.ff_norm = nn.LayerNorm(attn_dim_out)
        self.norm_ff_dropout = nn.Dropout(ff_dropout)

    def forward(self, X_Q: Tensor, X_KV: Optional[Tensor]=None) ->Tensor:
        x = self.ln_q(X_Q)
        y = None if X_KV is None else self.ln_kv(X_KV)
        x = x + self.norm_attn_dropout(self.attn(x, y))
        return x + self.norm_ff_dropout(self.ff(self.ff_norm(x)))


class TabPerceiver(BaseTabularModelWithAttention):
    """Defines an adaptation of a [Perceiver](https://arxiv.org/abs/2103.03206)
     that can be used as the `deeptabular` component of a Wide & Deep model
     or independently by itself.

    :information_source: **NOTE**: while there are scientific publications for
     the `TabTransformer`, `SAINT` and `FTTransformer`, the `TabPerceiver`
     and the `TabFastFormer` are our own adaptations of the
     [Perceiver](https://arxiv.org/abs/2103.03206) and the
     [FastFormer](https://arxiv.org/abs/2108.09084) for tabular data.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name and number of unique values for
        each categorical component e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings
        used to encode the categorical and/or continuous columns.
    n_cross_attns: int, default = 1
        Number of times each perceiver block will cross attend to the input
        data (i.e. number of cross attention components per perceiver block).
        This should normally be 1. However, in the paper they describe some
        architectures (normally computer vision-related problems) where the
        Perceiver attends multiple times to the input array. Therefore, maybe
        multiple cross attention to the input array is also useful in some
        cases for tabular data :shrug: .
    n_cross_attn_heads: int, default = 4
        Number of attention heads for the cross attention component
    n_latents: int, default = 16
        Number of latents. This is the $N$ parameter in the paper. As
        indicated in the paper, this number should be significantly lower
        than $M$ (the number of columns in the dataset). Setting $N$ closer
        to $M$ defies the main purpose of the Perceiver, which is to overcome
        the transformer quadratic bottleneck
    latent_dim: int, default = 128
        Latent dimension.
    n_latent_heads: int, default = 4
        Number of attention heads per Latent Transformer
    n_latent_blocks: int, default = 4
        Number of transformer encoder blocks (normalised MHA + normalised FF)
        per Latent Transformer
    n_perceiver_blocks: int, default = 4
        Number of Perceiver blocks defined as [Cross Attention + Latent
        Transformer]
    share_weights: Boolean, default = False
        Boolean indicating if the weights will be shared between Perceiver
        blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l, 4
        \\times l, 2 \\times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.ModuleDict
        ModuleDict with the Perceiver blocks
    latents: nn.Parameter
        Latents that will be used for prediction
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabPerceiver
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabPerceiver(column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols=continuous_cols, n_latents=2, latent_dim=16,
    ... n_perceiver_blocks=2)
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, n_cross_attns: int=1, n_cross_attn_heads: int=4, n_latents: int=16, latent_dim: int=128, n_latent_heads: int=4, n_latent_blocks: int=4, n_perceiver_blocks: int=4, share_weights: bool=False, attn_dropout: float=0.1, ff_dropout: float=0.1, transformer_activation: str='geglu', mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=True):
        super(TabPerceiver, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=True, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.n_cross_attns = n_cross_attns
        self.n_cross_attn_heads = n_cross_attn_heads
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.n_latent_heads = n_latent_heads
        self.n_latent_blocks = n_latent_blocks
        self.n_perceiver_blocks = n_perceiver_blocks
        self.share_weights = share_weights
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.latents = nn.init.trunc_normal_(nn.Parameter(torch.empty(n_latents, latent_dim)))
        self.encoder = nn.ModuleDict()
        first_perceiver_block = self._build_perceiver_block()
        self.encoder['perceiver_block0'] = first_perceiver_block
        if share_weights:
            for n in range(1, n_perceiver_blocks):
                self.encoder['perceiver_block' + str(n)] = first_perceiver_block
        else:
            for n in range(1, n_perceiver_blocks):
                self.encoder['perceiver_block' + str(n)] = self._build_perceiver_block()
        self.mlp_first_hidden_dim = self.latent_dim
        if mlp_hidden_dims is not None:
            self.mlp = MLP([self.mlp_first_hidden_dim] + mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        x_emb = self._get_embeddings(X)
        x = einops.repeat(self.latents, 'n d -> b n d', b=X.shape[0])
        for n in range(self.n_perceiver_blocks):
            cross_attns = self.encoder['perceiver_block' + str(n)]['cross_attns']
            latent_transformer = self.encoder['perceiver_block' + str(n)]['latent_transformer']
            for cross_attn in cross_attns:
                x = cross_attn(x, x_emb)
            x = latent_transformer(x)
        x = x.mean(dim=1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.mlp_first_hidden_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights. If the weights are not shared
        between perceiver blocks each element of the list will be a list
        itself containing the Cross Attention and Latent Transformer
        attention weights respectively

        The shape of the attention weights is:

        - Cross Attention: $(N, C, L, F)$

        - Latent Attention: $(N, T, L, L)$

        WHere $N$ is the batch size, $C$ is the number of Cross Attention
        heads, $L$ is the number of Latents, $F$ is the number of
        features/columns in the dataset and $T$ is the number of Latent
        Attention heads
        """
        if self.share_weights:
            cross_attns = self.encoder['perceiver_block0']['cross_attns']
            latent_transformer = self.encoder['perceiver_block0']['latent_transformer']
            attention_weights = self._extract_attn_weights(cross_attns, latent_transformer)
        else:
            attention_weights = []
            for n in range(self.n_perceiver_blocks):
                cross_attns = self.encoder['perceiver_block' + str(n)]['cross_attns']
                latent_transformer = self.encoder['perceiver_block' + str(n)]['latent_transformer']
                attention_weights.append(self._extract_attn_weights(cross_attns, latent_transformer))
        return attention_weights

    def _build_perceiver_block(self) ->nn.ModuleDict:
        perceiver_block = nn.ModuleDict()
        cross_attns = nn.ModuleList()
        for _ in range(self.n_cross_attns):
            cross_attns.append(PerceiverEncoder(self.input_dim, self.n_cross_attn_heads, False, self.attn_dropout, self.ff_dropout, self.transformer_activation, self.latent_dim))
        perceiver_block['cross_attns'] = cross_attns
        latent_transformer = nn.Sequential()
        for i in range(self.n_latent_blocks):
            latent_transformer.add_module('latent_block' + str(i), PerceiverEncoder(self.latent_dim, self.n_latent_heads, False, self.attn_dropout, self.ff_dropout, self.transformer_activation))
        perceiver_block['latent_transformer'] = latent_transformer
        return perceiver_block

    @staticmethod
    def _extract_attn_weights(cross_attns, latent_transformer) ->List:
        attention_weights = []
        for cross_attn in cross_attns:
            attention_weights.append(cross_attn.attn.attn_weights)
        for latent_block in latent_transformer:
            attention_weights.append(latent_block.attn.attn_weights)
        return attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, input_dim: int, n_heads: int, use_bias: bool, attn_dropout: float, ff_dropout: float, activation: str):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadedAttention(input_dim, n_heads, use_bias, attn_dropout)
        self.ff = FeedForward(input_dim, ff_dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.ff_addnorm = AddNorm(input_dim, ff_dropout)

    def forward(self, X: Tensor) ->Tensor:
        x = self.attn_addnorm(X, self.attn)
        return self.ff_addnorm(x, self.ff)


class TabTransformer(BaseTabularModelWithAttention):
    """Defines a [TabTransformer model](https://arxiv.org/abs/2012.06678) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.

    :information_source: **NOTE**:
    This is an enhanced adaptation of the model described in the paper,
    containing a series of additional features.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name and number of unique values for
        each categorical component e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded at
        the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    embed_continuous: bool, default = False
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    use_qkv_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers.
    n_blocks: int, default = 4
        Number of Transformer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l,
        4\\times l, 2 \\times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of Transformer blocks
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabTransformer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabTransformer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, full_embed_dropout: bool=False, shared_embed: bool=False, add_shared_embed: bool=False, frac_shared_embed: float=0.25, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, embed_continuous: bool=False, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, input_dim: int=32, n_heads: int=8, use_qkv_bias: bool=False, n_blocks: int=4, attn_dropout: float=0.2, ff_dropout: float=0.1, transformer_activation: str='gelu', mlp_hidden_dims: Optional[List[int]]=None, mlp_activation: str='relu', mlp_dropout: float=0.1, mlp_batchnorm: bool=False, mlp_batchnorm_last: bool=False, mlp_linear_first: bool=True):
        super(TabTransformer, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, full_embed_dropout=full_embed_dropout, shared_embed=shared_embed, add_shared_embed=add_shared_embed, frac_shared_embed=frac_shared_embed, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=embed_continuous, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation, input_dim=input_dim)
        self.n_heads = n_heads
        self.use_qkv_bias = use_qkv_bias
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first
        self.with_cls_token = 'cls_token' in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        if self.n_cont and not self.n_cat and not self.embed_continuous:
            raise ValueError("If only continuous features are used 'embed_continuous' must be set to 'True'")
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module('transformer_block' + str(i), TransformerEncoder(input_dim, n_heads, use_qkv_bias, attn_dropout, ff_dropout, transformer_activation))
        self.mlp_first_hidden_dim = self._mlp_first_hidden_dim()
        if mlp_hidden_dims is not None:
            self.mlp = MLP([self.mlp_first_hidden_dim] + mlp_hidden_dims, mlp_activation, mlp_dropout, mlp_batchnorm, mlp_batchnorm_last, mlp_linear_first)
        else:
            self.mlp = None

    def forward(self, X: Tensor) ->Tensor:
        if not self.embed_continuous:
            x_cat, x_cont = self.cat_and_cont_embed(X)
            if x_cat is not None:
                x = self.cat_embed_act_fn(x_cat) if self.cat_embed_act_fn is not None else x_cat
        else:
            x = self._get_embeddings(X)
            x_cont = None
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
        if x_cont is not None and not self.embed_continuous:
            x = torch.cat([x, x_cont], 1)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    def _mlp_first_hidden_dim(self) ->int:
        if self.with_cls_token:
            if self.embed_continuous:
                attn_output_dim = self.input_dim
            else:
                attn_output_dim = self.input_dim + self.n_cont
        elif self.embed_continuous:
            attn_output_dim = (self.n_cat + self.n_cont) * self.input_dim
        else:
            attn_output_dim = self.n_cat * self.input_dim + self.n_cont
        return attn_output_dim

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.mlp_hidden_dims[-1] if self.mlp_hidden_dims is not None else self.mlp_first_hidden_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights per block

        The shape of the attention weights is $(N, H, F, F)$, where $N$ is the
        batch size, $H$ is the number of attention heads and $F$ is the
        number of features/columns in the dataset
        """
        return [blk.attn.attn_weights for blk in self.encoder]

    def _compute_attn_output_dim(self) ->int:
        if self.with_cls_token:
            if self.embed_continuous:
                attn_output_dim = self.input_dim
            else:
                attn_output_dim = self.input_dim + self.n_cont
        elif self.embed_continuous:
            attn_output_dim = (self.n_cat + self.n_cont) * self.input_dim
        else:
            attn_output_dim = self.n_cat * self.input_dim + self.n_cont
        return attn_output_dim


ModelWithAttention = Union[TabTransformer, SAINT, FTTransformer, TabFastFormer, TabPerceiver, ContextAttentionMLP, SelfAttentionMLP]


class BasePreprocessor:
    """Base Class of All Preprocessors."""

    def __init__(self, *args):
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError('Preprocessor must implement this method')


def check_is_fitted(estimator: BasePreprocessor, attributes: List[str]=None, all_or_any: str='all', condition: bool=True):
    """Checks if an estimator is fitted

    Parameters
    ----------
    estimator: ``BasePreprocessor``,
        An object of type ``BasePreprocessor``
    attributes: List, default = None
        List of strings with the attributes to check for
    all_or_any: str, default = "all"
        whether all or any of the attributes in the list must be present
    condition: bool, default = True,
        If not attribute list is passed, this condition that must be True for
        the estimator to be considered as fitted
    """
    estimator_name: str = estimator.__class__.__name__
    error_msg = "This {} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.".format(estimator_name)
    if attributes is not None and all_or_any == 'all':
        if not all([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif attributes is not None and all_or_any == 'any':
        if not any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif not condition:
        raise NotFittedError(error_msg)


def cut_mix(x: Tensor, lam: float=0.8) ->Tensor:
    batch_size = x.size()[0]
    mask = torch.from_numpy(np.random.choice(2, x.shape, p=[lam, 1 - lam]))
    rand_idx = torch.randperm(batch_size)
    x_ = x[rand_idx].clone()
    x_[mask == 0] = x[mask == 0]
    return x_


def mix_up(p: Tensor, lam: float=0.8) ->Tensor:
    batch_size = p.size()[0]
    rand_idx = torch.randperm(batch_size)
    p_ = lam * p + (1 - lam) * p[rand_idx, ...]
    return p_


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim: int, virtual_batch_size: int=128, momentum: float=0.01):
        super(GBN, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, X: Tensor) ->Tensor:
        chunks = X.chunk(int(np.ceil(X.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


def initialize_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GLU_Layer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float, fc: nn.Module=None, ghost_bn: bool=True, virtual_batch_size: int=128, momentum: float=0.02):
        super(GLU_Layer, self).__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)
        if ghost_bn:
            self.bn: Union[GBN, nn.BatchNorm1d] = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        else:
            self.bn = nn.BatchNorm1d(2 * output_dim, momentum=momentum)
        self.dp = nn.Dropout(dropout)

    def forward(self, X: Tensor) ->Tensor:
        return self.dp(F.glu(self.bn(self.fc(X))))


class GLU_Block(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float, n_glu: int=2, first: bool=False, shared_layers: nn.ModuleList=None, ghost_bn: bool=True, virtual_batch_size: int=128, momentum: float=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        if shared_layers is not None and n_glu != len(shared_layers):
            self.n_glu = len(shared_layers)
            warnings.warn("If 'shared_layers' is nor None, 'n_glu' must be equal to the number of shared_layers.Got n_glu = {} and n shared_layers = {}. 'n_glu' has been set to {}".format(n_glu, len(shared_layers), len(shared_layers)), UserWarning)
        else:
            self.n_glu = n_glu
        glu_dim = [input_dim] + [output_dim] * self.n_glu
        self.glu_layers = nn.ModuleList()
        for i in range(self.n_glu):
            fc = shared_layers[i] if shared_layers else None
            self.glu_layers.append(GLU_Layer(glu_dim[i], glu_dim[i + 1], dropout, fc=fc, ghost_bn=ghost_bn, virtual_batch_size=virtual_batch_size, momentum=momentum))

    def forward(self, X: Tensor) ->Tensor:
        scale = torch.sqrt(torch.FloatTensor([0.5]))
        if self.first:
            x = self.glu_layers[0](X)
            layers_left = range(1, self.n_glu)
        else:
            x = nn.Identity()(X)
            layers_left = range(self.n_glu)
        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x)) * scale
        return x


class FeatTransformer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float, shared_layers: nn.ModuleList, n_glu_step_dependent: int, ghost_bn=True, virtual_batch_size=128, momentum=0.02):
        super(FeatTransformer, self).__init__()
        params = {'ghost_bn': ghost_bn, 'virtual_batch_size': virtual_batch_size, 'momentum': momentum}
        self.shared = GLU_Block(input_dim, output_dim, dropout, n_glu=len(shared_layers), first=True, shared_layers=shared_layers, **params)
        self.step_dependent = GLU_Block(output_dim, output_dim, dropout, n_glu=n_glu_step_dependent, first=False, **params)

    def forward(self, X: Tensor) ->Tensor:
        return self.step_dependent(self.shared(X))


def initialize_non_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class TabNetDecoder(nn.Module):
    """Companion decoder model for the `TabNet` model (which can be
    considered an encoder itself)

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). This class will receive the output from the `TabNet` encoder
    (i.e. the output from the so called 'steps') and '_reconstruct_' the
    embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor to be reconstructed.
    n_steps: int, default = 3
        number of decision steps. For a better understanding of the function
        of `n_steps` and the upcoming parameters, please see the
        [paper](https://arxiv.org/abs/1908.07442).
    step_dim: int, default = 8
        Step's output dimension. This is the output dimension that
        `WideDeep` will collect and connect to the output neuron(s).
    dropout: float, default = 0.0
        GLU block's internal dropout
    n_glu_step_dependent: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that are step dependent
    n_glu_shared: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that will be shared
        across decision steps
    ghost_bn: bool, default=True
        Boolean indicating if [Ghost Batch Normalization](https://arxiv.org/abs/1705.08741)
        will be used.
    virtual_batch_size: int, default = 128
        Batch size when using Ghost Batch Normalization
    momentum: float, default = 0.02
        Ghost Batch Normalization's momentum. The dreamquark-ai advises for
        very low values. However high values are used in the original
        publication. During our tests higher values lead to better results

    Attributes
    ----------
    decoder: nn.Module
        decoder that will receive the output from the encoder's steps and will
        reconstruct the embeddings

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabNetDecoder
    >>> x_inp = [torch.rand(3, 8), torch.rand(3, 8), torch.rand(3, 8)]
    >>> decoder = TabNetDecoder(embed_dim=32, ghost_bn=False)
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(self, embed_dim: int, n_steps: int=3, step_dim: int=8, dropout: float=0.0, n_glu_step_dependent: int=2, n_glu_shared: int=2, ghost_bn: bool=True, virtual_batch_size: int=128, momentum: float=0.02):
        super(TabNetDecoder, self).__init__()
        self.n_steps = n_steps
        self.step_dim = step_dim
        self.dropout = dropout
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        shared_layers = nn.ModuleList()
        for i in range(n_glu_shared):
            if i == 0:
                shared_layers.append(nn.Linear(step_dim, 2 * step_dim, bias=False))
            else:
                shared_layers.append(nn.Linear(step_dim, 2 * step_dim, bias=False))
        self.decoder = nn.ModuleList()
        for step in range(n_steps):
            transformer = FeatTransformer(step_dim, step_dim, dropout, shared_layers, n_glu_step_dependent, ghost_bn, virtual_batch_size, momentum=momentum)
            self.decoder.append(transformer)
        self.reconstruction_layer = nn.Linear(step_dim, embed_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, step_dim, embed_dim)

    def forward(self, X: List[Tensor]) ->Tensor:
        out = torch.tensor(0.0)
        for i, x in enumerate(X):
            x = self.decoder[i](x)
            out = torch.add(out, x)
        out = self.reconstruction_layer(out)
        return out


DecoderWithoutAttention = Union[TabMlpDecoder, TabNetDecoder, TabResnetDecoder]


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class AttentiveTransformer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, mask_type: str='sparsemax', ghost_bn=True, virtual_batch_size=128, momentum=0.02):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        if ghost_bn:
            self.bn: Union[GBN, nn.BatchNorm1d] = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        else:
            self.bn = nn.BatchNorm1d(output_dim, momentum=momentum)
        if mask_type == 'sparsemax':
            self.mask: Union[sparsemax.Sparsemax, sparsemax.Entmax15] = sparsemax.Sparsemax(dim=-1)
        elif mask_type == 'entmax':
            self.mask = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError("Please choose either 'sparsemax' or 'entmax' as masktype")

    def forward(self, priors: Tensor, processed_feat: Tensor) ->Tensor:
        x = self.bn(self.fc(processed_feat))
        x = torch.mul(x, priors)
        return self.mask(x)


class TabNetEncoder(nn.Module):

    def __init__(self, input_dim: int, n_steps: int=3, step_dim: int=8, attn_dim: int=8, dropout: float=0.0, n_glu_step_dependent: int=2, n_glu_shared: int=2, ghost_bn: bool=True, virtual_batch_size: int=128, momentum: float=0.02, gamma: float=1.3, epsilon: float=1e-15, mask_type: str='sparsemax'):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        params = {'ghost_bn': ghost_bn, 'virtual_batch_size': virtual_batch_size, 'momentum': momentum}
        shared_layers = nn.ModuleList()
        for i in range(n_glu_shared):
            if i == 0:
                shared_layers.append(nn.Linear(input_dim, 2 * (step_dim + attn_dim), bias=False))
            else:
                shared_layers.append(nn.Linear(step_dim + attn_dim, 2 * (step_dim + attn_dim), bias=False))
        self.initial_splitter = FeatTransformer(input_dim, step_dim + attn_dim, dropout, shared_layers, n_glu_step_dependent, **params)
        self.feat_transformers = nn.ModuleList()
        self.attn_transformers = nn.ModuleList()
        for step in range(n_steps):
            feat_transformer = FeatTransformer(input_dim, step_dim + attn_dim, dropout, shared_layers, n_glu_step_dependent, **params)
            attn_transformer = AttentiveTransformer(attn_dim, input_dim, mask_type, **params)
            self.feat_transformers.append(feat_transformer)
            self.attn_transformers.append(attn_transformer)

    def forward(self, X: Tensor, prior: Optional[Tensor]=None) ->Tuple[List[Tensor], Tensor]:
        x = self.initial_bn(X)
        if prior is None:
            prior = torch.ones(x.shape)
        M_loss = torch.FloatTensor([0.0])
        attn = self.initial_splitter(x)[:, self.step_dim:]
        steps_output = []
        for step in range(self.n_steps):
            M = self.attn_transformers[step](prior, attn)
            prior = torch.mul(self.gamma - M, prior)
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1))
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            attn = out[:, self.step_dim:]
            d_out = nn.ReLU()(out[:, :self.step_dim])
            steps_output.append(d_out)
        M_loss /= self.n_steps
        return steps_output, M_loss

    def forward_masks(self, X: Tensor) ->Tuple[Tensor, Dict[int, Tensor]]:
        x = self.initial_bn(X)
        prior = torch.ones(x.shape)
        M_explain = torch.zeros(x.shape)
        attn = self.initial_splitter(x)[:, self.step_dim:]
        masks = {}
        for step in range(self.n_steps):
            M = self.attn_transformers[step](prior, attn)
            masks[step] = M
            prior = torch.mul(self.gamma - M, prior)
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            attn = out[:, self.step_dim:]
            d_out = nn.ReLU()(out[:, :self.step_dim])
            agg_decision_contrib = torch.sum(d_out, dim=1)
            M_explain += torch.mul(M, agg_decision_contrib.unsqueeze(dim=1))
        return M_explain, masks


class TabNet(BaseTabularModelWithoutAttention):
    """Defines a [TabNet model](https://arxiv.org/abs/1908.07442) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.

    The implementation in this library is fully based on that
    [here](https://github.com/dreamquark-ai/tabnet) by the dreamquark-ai team,
    simply adapted so that it can work within the `WideDeep` frame.
    Therefore, **ALL CREDIT TO THE DREAMQUARK-AI TEAM**.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabNet` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or `None`.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    n_steps: int, default = 3
        number of decision steps. For a better understanding of the function
        of `n_steps` and the upcoming parameters, please see the
        [paper](https://arxiv.org/abs/1908.07442).
    step_dim: int, default = 8
        Step's output dimension. This is the output dimension that
        `WideDeep` will collect and connect to the output neuron(s).
    attn_dim: int, default = 8
        Attention dimension
    dropout: float, default = 0.0
        GLU block's internal dropout
    n_glu_step_dependent: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that are step dependent
    n_glu_shared: int, default = 2
        number of GLU Blocks (`[FC -> BN -> GLU]`) that will be shared
        across decision steps
    ghost_bn: bool, default=True
        Boolean indicating if [Ghost Batch Normalization](https://arxiv.org/abs/1705.08741)
        will be used.
    virtual_batch_size: int, default = 128
        Batch size when using Ghost Batch Normalization
    momentum: float, default = 0.02
        Ghost Batch Normalization's momentum. The dreamquark-ai advises for
        very low values. However high values are used in the original
        publication. During our tests higher values lead to better results
    gamma: float, default = 1.3
        Relaxation parameter in the paper. When gamma = 1, a feature is
        enforced to be used only at one decision step. As gamma increases,
        more flexibility is provided to use a feature at multiple decision
        steps
    epsilon: float, default = 1e-15
        Float to avoid log(0). Always keep low
    mask_type: str, default = "sparsemax"
        Mask function to use. Either _'sparsemax'_ or _'entmax'_

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        the TabNet encoder. For details see the [original publication](https://arxiv.org/abs/1908.07442).

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabNet
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabNet(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(self, column_idx: Dict[str, int], cat_embed_input: Optional[List[Tuple[str, int, int]]]=None, cat_embed_dropout: float=0.1, use_cat_bias: bool=False, cat_embed_activation: Optional[str]=None, continuous_cols: Optional[List[str]]=None, cont_norm_layer: str=None, embed_continuous: bool=False, cont_embed_dim: int=32, cont_embed_dropout: float=0.1, use_cont_bias: bool=True, cont_embed_activation: Optional[str]=None, n_steps: int=3, step_dim: int=8, attn_dim: int=8, dropout: float=0.0, n_glu_step_dependent: int=2, n_glu_shared: int=2, ghost_bn: bool=True, virtual_batch_size: int=128, momentum: float=0.02, gamma: float=1.3, epsilon: float=1e-15, mask_type: str='sparsemax'):
        super(TabNet, self).__init__(column_idx=column_idx, cat_embed_input=cat_embed_input, cat_embed_dropout=cat_embed_dropout, use_cat_bias=use_cat_bias, cat_embed_activation=cat_embed_activation, continuous_cols=continuous_cols, cont_norm_layer=cont_norm_layer, embed_continuous=embed_continuous, cont_embed_dim=cont_embed_dim, cont_embed_dropout=cont_embed_dropout, use_cont_bias=use_cont_bias, cont_embed_activation=cont_embed_activation)
        self.n_steps = n_steps
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask_type = mask_type
        self.embed_out_dim = self.cat_and_cont_embed.output_dim
        self.encoder = TabNetEncoder(self.embed_out_dim, n_steps, step_dim, attn_dim, dropout, n_glu_step_dependent, n_glu_shared, ghost_bn, virtual_batch_size, momentum, gamma, epsilon, mask_type)

    def forward(self, X: Tensor, prior: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        x = self._get_embeddings(X)
        steps_output, M_loss = self.encoder(x, prior)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return res, M_loss

    def forward_masks(self, X: Tensor) ->Tuple[Tensor, Dict[int, Tensor]]:
        x = self._get_embeddings(X)
        return self.encoder.forward_masks(x)

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.step_dim


ModelWithoutAttention = Union[TabMlp, TabNet, TabResnet]


class EncoderDecoderModel(nn.Module):
    """This Class, which is referred as a 'Model', implements an Encoder-Decoder
    Self Supervised 'routine' inspired by `TabNet: Attentive Interpretable
    Tabular Learning <https://arxiv.org/abs/1908.07442>_`

    This class is in principle not exposed to the user and its documentation
    is detailed in its corresponding Trainer:  see
    ``pytorch_widedeep.self_supervised_training.EncoderDecoderTrainer``
    """

    def __init__(self, encoder: ModelWithoutAttention, decoder: Optional[DecoderWithoutAttention], masked_prob: float):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        if decoder is None:
            self.decoder = self._build_decoder(encoder)
        else:
            self.decoder = decoder
        self.masker = RandomObfuscator(p=masked_prob)
        self.is_tabnet = isinstance(self.encoder, TabNet)

    def forward(self, X: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        if self.is_tabnet:
            return self._forward_tabnet(X)
        else:
            return self._forward(X)

    def _forward(self, X: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        x_embed = self.encoder._get_embeddings(X)
        if self.training:
            masked_x, mask = self.masker(x_embed)
            x_embed_rec = self.decoder(self.encoder(X))
        else:
            x_embed_rec = self.decoder(self.encoder(X))
            mask = torch.ones(x_embed.shape)
        return x_embed, x_embed_rec, mask

    def _forward_tabnet(self, X: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        x_embed = self.encoder._get_embeddings(X)
        if self.training:
            masked_x, mask = self.masker(x_embed)
            prior = 1 - mask
            steps_out, _ = self.encoder.encoder(masked_x, prior=prior)
            x_embed_rec = self.decoder(steps_out)
        else:
            steps_out, _ = self.encoder(x_embed)
            x_embed_rec = self.decoder(steps_out)
            mask = torch.ones(x_embed.shape)
        return x_embed_rec, x_embed, mask

    def _build_decoder(self, encoder: ModelWithoutAttention) ->DecoderWithoutAttention:
        if isinstance(encoder, TabMlp):
            decoder = self._build_tabmlp_decoder()
        if isinstance(encoder, TabResnet):
            decoder = self._build_tabresnet_decoder()
        if isinstance(encoder, TabNet):
            decoder = self._build_tabnet_decoder()
        return decoder

    def _build_tabmlp_decoder(self) ->DecoderWithoutAttention:
        common_params = inspect.signature(TabMlp).parameters.keys() & inspect.signature(TabMlpDecoder).parameters.keys()
        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)
        decoder_param['mlp_hidden_dims'] = decoder_param['mlp_hidden_dims'][::-1]
        decoder_param['embed_dim'] = self.encoder.cat_and_cont_embed.output_dim
        return TabMlpDecoder(**decoder_param)

    def _build_tabresnet_decoder(self) ->DecoderWithoutAttention:
        common_params = inspect.signature(TabResnet).parameters.keys() & inspect.signature(TabResnetDecoder).parameters.keys()
        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)
        decoder_param['blocks_dims'] = decoder_param['blocks_dims'][::-1]
        if decoder_param['mlp_hidden_dims'] is not None:
            decoder_param['mlp_hidden_dims'] = decoder_param['mlp_hidden_dims'][::-1]
        decoder_param['embed_dim'] = self.encoder.cat_and_cont_embed.output_dim
        return TabResnetDecoder(**decoder_param)

    def _build_tabnet_decoder(self) ->DecoderWithoutAttention:
        common_params = inspect.signature(TabNet).parameters.keys() & inspect.signature(TabNetDecoder).parameters.keys()
        decoder_param = {}
        for cpn in common_params:
            decoder_param[cpn] = getattr(self.encoder, cpn)
        decoder_param['embed_dim'] = self.encoder.cat_and_cont_embed.output_dim
        return TabNetDecoder(**decoder_param)


class Sparsemax(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2
        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)
        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


entmax15 = Entmax15Function.apply


class Entmax15(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)


class TabNetPredLayer(nn.Module):

    def __init__(self, inp, out):
        """This class is a 'hack' required because TabNet is a very particular
        model within `WideDeep`.

        TabNet's forward method within `WideDeep` outputs two tensors, one
        with the last layer's activations and the sparse regularization
        factor. Since the output needs to be collected by `WideDeep` to then
        Sequentially build the output layer (connection to the output
        neuron(s)) I need to code a custom TabNetPredLayer that accepts two
        inputs. This will be used by the `WideDeep` class.
        """
        super(TabNetPredLayer, self).__init__()
        self.pred_layer = nn.Linear(inp, out, bias=False)
        initialize_non_glu(self.pred_layer, inp, out)

    def forward(self, tabnet_tuple: Tuple[Tensor, Tensor]) ->Tuple[Tensor, Tensor]:
        res, M_loss = tabnet_tuple[0], tabnet_tuple[1]
        return self.pred_layer(res), M_loss


class BasicRNN(nn.Module):
    """Standard text classifier/regressor comprised by a stack of RNNs
    (LSTMs or GRUs) that can be used as the `deeptext` component of a Wide &
    Deep model or independently by itself.

    In addition, there is the option to add a Fully Connected (FC) set of
    dense layers on top of the stack of RNNs

    Parameters
    ----------
    vocab_size: int
        Number of words in the vocabulary
    embed_dim: int, Optional, default = None
        Dimension of the word embeddings if non-pretained word vectors are
        used
    embed_matrix: np.ndarray, Optional, default = None
        Pretrained word embeddings
    embed_trainable: bool, default = True
        Boolean indicating if the pretrained embeddings are trainable
    rnn_type: str, default = 'lstm'
        String indicating the type of RNN to use. One of _'lstm'_ or _'gru'_
    hidden_dim: int, default = 64
        Hidden dim of the RNN
    n_layers: int, default = 3
        Number of recurrent layers
    rnn_dropout: float, default = 0.1
        Dropout for each RNN layer except the last layer
    bidirectional: bool, default = True
        Boolean indicating whether the staked RNNs are bidirectional
    use_hidden_state: str, default = True
        Boolean indicating whether to use the final hidden state or the RNN's
        output as predicting features. Typically the former is used.
    padding_idx: int, default = 1
        index of the padding token in the padded-tokenised sequences. The
        `TextPreprocessor` class within this library uses fastai's tokenizer
        where the token index 0 is reserved for the _'unknown'_ word token.
        Therefore, the default value is set to 1.
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: _[128, 64]_
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in
        the dense layers that form the _'rnn_mlp'_
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    word_embed: nn.Module
        word embedding matrix
    rnn: nn.Module
        Stack of RNNs
    rnn_mlp: nn.Module
        Stack of dense layers on top of the RNN. This will only exists if
        `head_layers_dim` is not None

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import BasicRNN
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = BasicRNN(vocab_size=4, hidden_dim=4, n_layers=2, padding_idx=0, embed_dim=4)
    >>> out = model(X_text)
    """

    def __init__(self, vocab_size: int, embed_dim: Optional[int]=None, embed_matrix: Optional[np.ndarray]=None, embed_trainable: bool=True, rnn_type: str='lstm', hidden_dim: int=64, n_layers: int=3, rnn_dropout: float=0.1, bidirectional: bool=False, use_hidden_state: bool=True, padding_idx: int=1, head_hidden_dims: Optional[List[int]]=None, head_activation: str='relu', head_dropout: Optional[float]=None, head_batchnorm: bool=False, head_batchnorm_last: bool=False, head_linear_first: bool=False):
        super(BasicRNN, self).__init__()
        if embed_dim is not None and embed_matrix is not None and not embed_dim == embed_matrix.shape[1]:
            warnings.warn('the input embedding dimension {} and the dimension of the pretrained embeddings {} do not match. The pretrained embeddings dimension ({}) will be used'.format(embed_dim, embed_matrix.shape[1], embed_matrix.shape[1]), UserWarning)
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError(f"'rnn_type' must be 'lstm' or 'gru', got {rnn_type} instead")
        self.vocab_size = vocab_size
        self.embed_trainable = embed_trainable
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.use_hidden_state = use_hidden_state
        self.padding_idx = padding_idx
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first
        self.word_embed, self.embed_dim = self._set_embeddings(embed_matrix)
        rnn_params = {'input_size': self.embed_dim, 'hidden_size': hidden_dim, 'num_layers': n_layers, 'bidirectional': bidirectional, 'dropout': rnn_dropout, 'batch_first': True}
        if self.rnn_type.lower() == 'lstm':
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.LSTM(**rnn_params)
        elif self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(**rnn_params)
        self.rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if self.head_hidden_dims is not None:
            head_hidden_dims = [self.rnn_output_dim] + head_hidden_dims
            self.rnn_mlp: Union[MLP, nn.Identity] = MLP(head_hidden_dims, head_activation, head_dropout, head_batchnorm, head_batchnorm_last, head_linear_first)
        else:
            self.rnn_mlp = nn.Identity()

    def forward(self, X: Tensor) ->Tensor:
        embed = self.word_embed(X.long())
        if self.rnn_type.lower() == 'lstm':
            o, (h, c) = self.rnn(embed)
        elif self.rnn_type.lower() == 'gru':
            o, h = self.rnn(embed)
        processed_outputs = self._process_rnn_outputs(o, h)
        return self.rnn_mlp(processed_outputs)

    @property
    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.head_hidden_dims[-1] if self.head_hidden_dims is not None else self.rnn_output_dim

    def _set_embeddings(self, embed_matrix: Union[Any, np.ndarray]) ->Tuple[nn.Module, int]:
        if isinstance(embed_matrix, np.ndarray):
            assert embed_matrix.dtype == 'float32', "'embed_matrix' must be of dtype 'float32', got dtype '{}'".format(str(embed_matrix.dtype))
            word_embed = nn.Embedding(self.vocab_size, embed_matrix.shape[1], padding_idx=self.padding_idx)
            if self.embed_trainable:
                word_embed.weight = nn.Parameter(torch.tensor(embed_matrix), requires_grad=True)
            else:
                word_embed.weight = nn.Parameter(torch.tensor(embed_matrix), requires_grad=False)
            embed_dim = embed_matrix.shape[1]
        else:
            word_embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
            embed_dim = self.embed_dim
        return word_embed, embed_dim

    def _process_rnn_outputs(self, output: Tensor, hidden: Tensor) ->Tensor:
        output = output.permute(1, 0, 2)
        if self.bidirectional:
            processed_outputs = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.use_hidden_state else output[-1]
        else:
            processed_outputs = hidden[-1] if self.use_hidden_state else output[-1]
        return processed_outputs


class StackedAttentiveRNN(nn.Module):
    """Text classifier/regressor comprised by a stack of blocks:
    `[RNN + Attention]`. This can be used as the `deeptext` component of a
    Wide & Deep model or independently by itself.

    In addition, there is the option to add a Fully Connected (FC) set of
    dense layers on top of the attentiob blocks

    Parameters
    ----------
    vocab_size: int
        Number of words in the vocabulary
    embed_dim: int, Optional, default = None
        Dimension of the word embeddings if non-pretained word vectors are
        used
    embed_matrix: np.ndarray, Optional, default = None
        Pretrained word embeddings
    embed_trainable: bool, default = True
        Boolean indicating if the pretrained embeddings are trainable
    rnn_type: str, default = 'lstm'
        String indicating the type of RNN to use. One of 'lstm' or 'gru'
    hidden_dim: int, default = 64
        Hidden dim of the RNN
    bidirectional: bool, default = True
        Boolean indicating whether the staked RNNs are bidirectional
    padding_idx: int, default = 1
        index of the padding token in the padded-tokenised sequences. The
        `TextPreprocessor` class within this library uses fastai's
        tokenizer where the token index 0 is reserved for the _'unknown'_
        word token. Therefore, the default value is set to 1.
    n_blocks: int, default = 3
        Number of attention blocks. Each block is comprised by an RNN and a
        Context Attention Encoder
    attn_concatenate: bool, default = True
        Boolean indicating if the input to the attention mechanism will be the
        output of the RNN or the output of the RNN concatenated with the last
        hidden state or simply
    attn_dropout: float, default = 0.1
        Internal dropout for the attention mechanism
    with_addnorm: bool, default = False
        Boolean indicating if the output of each block will be added to the
        input and normalised
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: [128, 64]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in
        the dense layers that form the _'rnn_mlp'_
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    word_embed: nn.Module
        word embedding matrix
    rnn: nn.Module
        Stack of RNNs
    rnn_mlp: nn.Module
        Stack of dense layers on top of the RNN. This will only exists if
        `head_layers_dim` is not `None`

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import StackedAttentiveRNN
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = StackedAttentiveRNN(vocab_size=4, hidden_dim=4, padding_idx=0, embed_dim=4)
    >>> out = model(X_text)
    """

    def __init__(self, vocab_size: int, embed_dim: Optional[int]=None, embed_matrix: Optional[np.ndarray]=None, embed_trainable: bool=True, rnn_type: str='lstm', hidden_dim: int=64, bidirectional: bool=False, padding_idx: int=1, n_blocks: int=3, attn_concatenate: bool=False, attn_dropout: float=0.1, with_addnorm: bool=False, head_hidden_dims: Optional[List[int]]=None, head_activation: str='relu', head_dropout: Optional[float]=None, head_batchnorm: bool=False, head_batchnorm_last: bool=False, head_linear_first: bool=False):
        super(StackedAttentiveRNN, self).__init__()
        if embed_dim is not None and embed_matrix is not None and not embed_dim == embed_matrix.shape[1]:
            warnings.warn('the input embedding dimension {} and the dimension of the pretrained embeddings {} do not match. The pretrained embeddings dimension ({}) will be used'.format(embed_dim, embed_matrix.shape[1], embed_matrix.shape[1]), UserWarning)
        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError(f"'rnn_type' must be 'lstm' or 'gru', got {rnn_type} instead")
        self.vocab_size = vocab_size
        self.embed_trainable = embed_trainable
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.n_blocks = n_blocks
        self.attn_concatenate = attn_concatenate
        self.attn_dropout = attn_dropout
        self.with_addnorm = with_addnorm
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first
        self.word_embed, self.embed_dim = self._set_embeddings(embed_matrix)
        if bidirectional and attn_concatenate:
            self.rnn_output_dim = hidden_dim * 4
        elif bidirectional or attn_concatenate:
            self.rnn_output_dim = hidden_dim * 2
        else:
            self.rnn_output_dim = hidden_dim
        if self.rnn_output_dim != self.embed_dim:
            self.embed_proj: Union[nn.Linear, nn.Identity] = nn.Linear(self.embed_dim, self.rnn_output_dim)
        else:
            self.embed_proj = nn.Identity()
        rnn_params = {'input_size': self.rnn_output_dim, 'hidden_size': hidden_dim, 'bidirectional': bidirectional, 'batch_first': True}
        if self.rnn_type.lower() == 'lstm':
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.LSTM(**rnn_params)
        elif self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(**rnn_params)
        self.attention_blks = nn.ModuleList()
        for i in range(n_blocks):
            self.attention_blks.append(ContextAttentionEncoder(self.rnn, self.rnn_output_dim, attn_dropout, attn_concatenate, with_addnorm=with_addnorm if i != n_blocks - 1 else False, sum_along_seq=i == n_blocks - 1))
        if self.head_hidden_dims is not None:
            head_hidden_dims = [self.rnn_output_dim] + head_hidden_dims
            self.rnn_mlp: Union[MLP, nn.Identity] = MLP(head_hidden_dims, head_activation, head_dropout, head_batchnorm, head_batchnorm_last, head_linear_first)
        else:
            self.rnn_mlp = nn.Identity()

    def forward(self, X: Tensor) ->Tensor:
        x = self.embed_proj(self.word_embed(X.long()))
        h = nn.init.zeros_(torch.Tensor(2 if self.bidirectional else 1, X.shape[0], self.hidden_dim))
        if self.rnn_type == 'lstm':
            c = nn.init.zeros_(torch.Tensor(2 if self.bidirectional else 1, X.shape[0], self.hidden_dim))
        else:
            c = None
        for blk in self.attention_blks:
            x, h, c = blk(x, h, c)
        return self.rnn_mlp(x)

    def output_dim(self) ->int:
        """The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return self.head_hidden_dims[-1] if self.head_hidden_dims is not None else self.rnn_output_dim

    @property
    def attention_weights(self) ->List:
        """List with the attention weights per block

        The shape of the attention weights is $(N, S)$ Where $N$ is the batch
        size and $S$ is the length of the sequence
        """
        return [blk.attn.attn_weights for blk in self.attention_blks]

    def _set_embeddings(self, embed_matrix: Union[Any, np.ndarray]) ->Tuple[nn.Module, int]:
        if isinstance(embed_matrix, np.ndarray):
            assert embed_matrix.dtype == 'float32', "'embed_matrix' must be of dtype 'float32', got dtype '{}'".format(str(embed_matrix.dtype))
            word_embed = nn.Embedding(self.vocab_size, embed_matrix.shape[1], padding_idx=self.padding_idx)
            if self.embed_trainable:
                word_embed.weight = nn.Parameter(torch.tensor(embed_matrix), requires_grad=True)
            else:
                word_embed.weight = nn.Parameter(torch.tensor(embed_matrix), requires_grad=False)
            embed_dim = embed_matrix.shape[1]
        else:
            word_embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
            embed_dim = self.embed_dim
        return word_embed, embed_dim


class WideDeep(nn.Module):
    """Main collector class that combines all `wide`, `deeptabular`
    `deeptext` and `deepimage` models.

    Note that all models described so far in this library must be passed to
    the `WideDeep` class once constructed. This is because the models output
    the last layer before the prediction layer. Such prediction layer is
    added by the `WideDeep` class as it collects the components for every
    data mode.

    There are two options to combine these models that correspond to the
    two main architectures that `pytorch-widedeep` can build.

    - Directly connecting the output of the model components to an ouput neuron(s).

    - Adding a `Fully-Connected Head` (FC-Head) on top of the deep models.
      This FC-Head will combine the output form the `deeptabular`, `deeptext` and
      `deepimage` and will be then connected to the output neuron(s).

    Parameters
    ----------
    wide: nn.Module, Optional, default = None
        `Wide` model. This is a linear model where the non-linearities are
        captured via crossed-columns.
    deeptabular: nn.Module, Optional, default = None
        Currently this library implements a number of possible architectures
        for the `deeptabular` component. See the documenation of the
        package.
    deeptext: nn.Module, Optional, default = None
        Currently this library implements a number of possible architectures
        for the `deeptext` component. See the documenation of the
        package.
    deepimage: nn.Module, Optional, default = None
        Currently this library uses `torchvision` and implements a number of
        possible architectures for the `deepimage` component. See the
        documenation of the package.
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: [128, 64]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in
        the dense layers that form the `'rnn_mlp'`
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`
    deephead: nn.Module, Optional, default = None
        Alternatively, the user can pass a custom model that will receive the
        output of the deep component. If `deephead` is not None all the
        previous fc-head parameters will be ignored
    enforce_positive: bool, default = False
        Boolean indicating if the output from the final layer must be
        positive. This is important if you are using loss functions with
        non-negative input restrictions, e.g. RMSLE, or if you know your
        predictions are bounded in between 0 and inf
    enforce_positive_activation: str, default = "softplus"
        Activation function to enforce that the final layer has a positive
        output. `'softplus'` or `'relu'` are supported.
    pred_dim: int, default = 1
        Size of the final wide and deep output layer containing the
        predictions. `1` for regression and binary classification or number
        of classes for multiclass classification.
    with_fds: bool, default = False
        Boolean indicating if Feature Distribution Smoothing (FDS) will be
        applied before the final prediction layer. Only available for
        regression problems.
        See [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554) for details.

    Other Parameters
    ----------------
    **fds_config: dict, default = None
        Dictionary with the parameters to be used when using Feature
        Distribution Smoothing. Please, see the docs for the `FDSLayer`.
        <br/>
        :information_source: **NOTE**: Feature Distribution Smoothing
         is available when using **ONLY** a `deeptabular` component
        <br/>
        :information_source: **NOTE**: We consider this feature absolutely
        experimental and we recommend the user to not use it unless the
        corresponding [publication](https://arxiv.org/abs/2102.09554) is
        well understood


    Examples
    --------

    >>> from pytorch_widedeep.models import TabResnet, Vision, BasicRNN, Wide, WideDeep
    >>> embed_input = [(u, i, j) for u, i, j in zip(["a", "b", "c"][:4], [4] * 3, [8] * 3)]
    >>> column_idx = {k: v for v, k in enumerate(["a", "b", "c"])}
    >>> wide = Wide(10, 1)
    >>> deeptabular = TabResnet(blocks_dims=[8, 4], column_idx=column_idx, cat_embed_input=embed_input)
    >>> deeptext = BasicRNN(vocab_size=10, embed_dim=4, padding_idx=0)
    >>> deepimage = Vision()
    >>> model = WideDeep(wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage)


    :information_source: **NOTE**: It is possible to use custom components to
     build Wide & Deep models. Simply, build them and pass them as the
     corresponding parameters. Note that the custom models MUST return a last
     layer of activations(i.e. not the final prediction) so that  these
     activations are collected by `WideDeep` and combined accordingly. In
     addition, the models MUST also contain an attribute `output_dim` with
     the size of these last layers of activations. See for example
     `pytorch_widedeep.models.tab_mlp.TabMlp`
    """

    def __init__(self, wide: Optional[nn.Module]=None, deeptabular: Optional[nn.Module]=None, deeptext: Optional[nn.Module]=None, deepimage: Optional[nn.Module]=None, deephead: Optional[nn.Module]=None, head_hidden_dims: Optional[List[int]]=None, head_activation: str='relu', head_dropout: float=0.1, head_batchnorm: bool=False, head_batchnorm_last: bool=False, head_linear_first: bool=False, enforce_positive: bool=False, enforce_positive_activation: str='softplus', pred_dim: int=1, with_fds: bool=False, **fds_config):
        super(WideDeep, self).__init__()
        self._check_inputs(wide, deeptabular, deeptext, deepimage, deephead, head_hidden_dims, pred_dim, with_fds)
        self.wd_device: str = None
        self.pred_dim = pred_dim
        self.wide = wide
        self.deeptabular = deeptabular
        self.deeptext = deeptext
        self.deepimage = deepimage
        self.deephead = deephead
        self.enforce_positive = enforce_positive
        self.with_fds = with_fds
        if self.deeptabular is not None:
            self.is_tabnet = deeptabular.__class__.__name__ == 'TabNet'
        else:
            self.is_tabnet = False
        if self.deephead is None and head_hidden_dims is not None:
            self._build_deephead(head_hidden_dims, head_activation, head_dropout, head_batchnorm, head_batchnorm_last, head_linear_first)
        elif self.deephead is not None:
            pass
        else:
            self._add_pred_layer()
        if self.with_fds:
            self.fds_layer = FDSLayer(feature_dim=self.deeptabular.output_dim, **fds_config)
        if self.enforce_positive:
            self.enf_pos = get_activation_fn(enforce_positive_activation)

    def forward(self, X: Dict[str, Tensor], y: Optional[Tensor]=None, epoch: Optional[int]=None):
        if self.with_fds:
            return self._forward_deep_with_fds(X, y, epoch)
        wide_out = self._forward_wide(X)
        if self.deephead:
            deep = self._forward_deephead(X, wide_out)
        else:
            deep = self._forward_deep(X, wide_out)
        if self.enforce_positive:
            return self.enf_pos(deep)
        else:
            return deep

    def _build_deephead(self, head_hidden_dims, head_activation, head_dropout, head_batchnorm, head_batchnorm_last, head_linear_first):
        deep_dim = 0
        if self.deeptabular is not None:
            deep_dim += self.deeptabular.output_dim
        if self.deeptext is not None:
            deep_dim += self.deeptext.output_dim
        if self.deepimage is not None:
            deep_dim += self.deepimage.output_dim
        head_hidden_dims = [deep_dim] + head_hidden_dims
        self.deephead = MLP(head_hidden_dims, head_activation, head_dropout, head_batchnorm, head_batchnorm_last, head_linear_first)
        self.deephead.add_module('head_out', nn.Linear(head_hidden_dims[-1], self.pred_dim))

    def _add_pred_layer(self):
        if self.deeptabular is not None and not self.with_fds:
            if self.is_tabnet:
                self.deeptabular = nn.Sequential(self.deeptabular, TabNetPredLayer(self.deeptabular.output_dim, self.pred_dim))
            else:
                self.deeptabular = nn.Sequential(self.deeptabular, nn.Linear(self.deeptabular.output_dim, self.pred_dim))
        if self.deeptext is not None:
            self.deeptext = nn.Sequential(self.deeptext, nn.Linear(self.deeptext.output_dim, self.pred_dim))
        if self.deepimage is not None:
            self.deepimage = nn.Sequential(self.deepimage, nn.Linear(self.deepimage.output_dim, self.pred_dim))

    def _forward_wide(self, X):
        if self.wide is not None:
            out = self.wide(X['wide'])
        else:
            batch_size = X[list(X.keys())[0]].size(0)
            out = torch.zeros(batch_size, self.pred_dim)
        return out

    def _forward_deephead(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out = self.deeptabular(X['deeptabular'])
                deepside, M_loss = tab_out[0], tab_out[1]
            else:
                deepside = self.deeptabular(X['deeptabular'])
        else:
            deepside = torch.FloatTensor()
        if self.deeptext is not None:
            deepside = torch.cat([deepside, self.deeptext(X['deeptext'])], axis=1)
        if self.deepimage is not None:
            deepside = torch.cat([deepside, self.deepimage(X['deepimage'])], axis=1)
        deephead_out = self.deephead(deepside)
        deepside_out = nn.Linear(deephead_out.size(1), self.pred_dim)
        if self.is_tabnet:
            res = wide_out.add_(deepside_out(deephead_out)), M_loss
        else:
            res = wide_out.add_(deepside_out(deephead_out))
        return res

    def _forward_deep(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out, M_loss = self.deeptabular(X['deeptabular'])
                wide_out.add_(tab_out)
            else:
                wide_out.add_(self.deeptabular(X['deeptabular']))
        if self.deeptext is not None:
            wide_out.add_(self.deeptext(X['deeptext']))
        if self.deepimage is not None:
            wide_out.add_(self.deepimage(X['deepimage']))
        if self.is_tabnet:
            res = wide_out, M_loss
        else:
            res = wide_out
        return res

    def _forward_deep_with_fds(self, X: Dict[str, Tensor], y: Optional[Tensor]=None, epoch: Optional[int]=None):
        res = self.fds_layer(self.deeptabular(X['deeptabular']), y, epoch)
        if self.enforce_positive:
            if isinstance(res, Tuple):
                out = res[0], self.enf_pos(res[1])
            else:
                out = self.enf_pos(res)
        else:
            out = res
        return out

    @staticmethod
    def _check_inputs(wide, deeptabular, deeptext, deepimage, deephead, head_hidden_dims, pred_dim, with_fds):
        if wide is not None:
            assert wide.wide_linear.weight.size(1) == pred_dim, "the 'pred_dim' of the wide component ({}) must be equal to the 'pred_dim' of the deep component and the overall model itself ({})".format(wide.wide_linear.weight.size(1), pred_dim)
        if deeptabular is not None and not hasattr(deeptabular, 'output_dim'):
            raise AttributeError("deeptabular model must have an 'output_dim' attribute or property. See pytorch-widedeep.models.deep_text.DeepText")
        if deeptabular is not None:
            is_tabnet = deeptabular.__class__.__name__ == 'TabNet'
            has_wide_text_or_image = wide is not None or deeptext is not None or deepimage is not None
            if is_tabnet and has_wide_text_or_image:
                warnings.warn("'WideDeep' is a model comprised by multiple components and the 'deeptabular' component is 'TabNet'. We recommend using 'TabNet' in isolation. The reasons are: i)'TabNet' uses sparse regularization which partially losses its purpose when used in combination with other components. If you still want to use a multiple component model with 'TabNet', consider setting 'lambda_sparse' to 0 during training. ii) The feature importances will be computed only for TabNet but the model will comprise multiple components. Therefore, such importances will partially lose their 'meaning'.", UserWarning)
        if deeptext is not None and not hasattr(deeptext, 'output_dim'):
            raise AttributeError("deeptext model must have an 'output_dim' attribute or property. See pytorch-widedeep.models.deep_text.DeepText")
        if deepimage is not None and not hasattr(deepimage, 'output_dim'):
            raise AttributeError("deepimage model must have an 'output_dim' attribute or property. See pytorch-widedeep.models.deep_text.DeepText")
        if deephead is not None and head_hidden_dims is not None:
            raise ValueError("both 'deephead' and 'head_hidden_dims' are not None. Use one of the other, but not both")
        if head_hidden_dims is not None and not deeptabular and not deeptext and not deepimage:
            raise ValueError("if 'head_hidden_dims' is not None, at least one deep component must be used")
        if deephead is not None:
            deephead_inp_feat = next(deephead.parameters()).size(1)
            output_dim = 0
            if deeptabular is not None:
                output_dim += deeptabular.output_dim
            if deeptext is not None:
                output_dim += deeptext.output_dim
            if deepimage is not None:
                output_dim += deepimage.output_dim
            assert deephead_inp_feat == output_dim, "if a custom 'deephead' is used its input features ({}) must be equal to the output features of the deep component ({})".format(deephead_inp_feat, output_dim)
        if with_fds and ((wide is not None or deeptext is not None or deepimage is not None or deephead is not None) or pred_dim != 1):
            raise ValueError('Feature Distribution Smoothing (FDS) is supported when using only a deeptabular component and for regression problems.')


class TextModeTestClass(nn.Module):

    def __init__(self):
        super(TextModeTestClass, self).__init__()
        self.word_embed = nn.Embedding(5, 16, padding_idx=0)
        self.rnn = nn.LSTM(16, 8, batch_first=True)
        self.linear = nn.Linear(8, 1)

    def forward(self, X):
        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        return self.linear(h).view(-1, 1)


class ImageModeTestClass(nn.Module):

    def __init__(self):
        super(ImageModeTestClass, self).__init__()
        self.conv_block = nn.Sequential(conv_layer(3, 64, 3), conv_layer(64, 128, 1, maxpool=False, adaptiveavgpool=True))
        self.linear = nn.Linear(128, 1)

    def forward(self, X):
        x = self.conv_block(X)
        x = x.view(x.size(0), -1)
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddNorm,
     lambda: ([], {'input_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BayesianContEmbeddings,
     lambda: ([], {'n_cont_cols': 4, 'embed_dim': 4, 'prior_sigma_1': 4, 'prior_sigma_2': 4, 'prior_pi': 4, 'posterior_mu_init': 4, 'posterior_rho_init': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BayesianLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BayesianRegressionLoss,
     lambda: ([], {'noise_tolerance': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BayesianSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContEmbeddings,
     lambda: ([], {'n_cont_cols': 4, 'embed_dim': 4, 'embed_dropout': 0.5, 'use_bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextAttention,
     lambda: ([], {'input_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderDecoderLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Entmax15,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GBN,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GLU_Block,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GLU_Layer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (HuberLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageModeTestClass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (InfoNCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSLELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormAdd,
     lambda: ([], {'input_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
    (REGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RMSLELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sparsemax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TabNetPredLayer,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TextModeTestClass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (TweedieLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Wide,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jrzaurin_pytorch_widedeep(_paritybench_base):
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

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

