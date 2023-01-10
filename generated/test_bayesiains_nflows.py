import sys
_module = sys.modules[__name__]
del sys
nflows = _module
distributions = _module
base = _module
discrete = _module
mixture = _module
normal = _module
uniform = _module
flows = _module
autoregressive = _module
base = _module
realnvp = _module
nn = _module
nde = _module
made = _module
nets = _module
mlp = _module
resnet = _module
MonotonicNormalizer = _module
UMNN = _module
transforms = _module
autoregressive = _module
base = _module
conv = _module
coupling = _module
linear = _module
lu = _module
made = _module
nonlinearities = _module
normalization = _module
orthogonal = _module
permutations = _module
qr = _module
reshape = _module
splines = _module
cubic = _module
linear = _module
quadratic = _module
rational_quadratic = _module
standard = _module
svd = _module
utils = _module
torchutils = _module
typechecks = _module
version = _module
setup = _module
tests = _module
discrete_test = _module
normal_test = _module
autoregressive_test = _module
base_test = _module
realnvp_test = _module
autoregressive_test = _module
base_test = _module
conv_test = _module
coupling_test = _module
linear_test = _module
lu_test = _module
made_test = _module
nonlinearities_test = _module
normalization_test = _module
orthogonal_test = _module
permutations_test = _module
qr_test = _module
reshape_test = _module
cubic_test = _module
linear_test = _module
quadratic_test = _module
rational_quadratic_test = _module
standard_test = _module
svd_test = _module
transform_test = _module
torchutils_test = _module

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


from torch import nn


from torch.nn import functional as F


import numpy as np


from typing import Union


from torch import distributions


import torch.nn


from inspect import signature


from matplotlib import pyplot as plt


from torch.nn import init


import torch.nn as nn


import warnings


from torch.nn.functional import softplus


import math


from typing import Iterable


from typing import Optional


from typing import Tuple


from torch import Tensor


class NoMeanException(Exception):
    """Exception to be thrown when a mean function doesn't exist."""
    pass


class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def forward(self, *args):
        raise RuntimeError('Forward method cannot be called for a Distribution object.')

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError('Number of input items must be equal to number of context items.')
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given, where ... represents a feature
            vector of arbitrary shape.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError('Number of samples must be a positive integer.')
        if context is not None:
            context = torch.as_tensor(context)
        if batch_size is None:
            return self._sample(num_samples, context)
        else:
            if not check.is_positive_int(batch_size):
                raise TypeError('Batch size must be a positive integer.')
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given, where ... represents a 
                  feature vector of arbitrary shape.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, features if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)
        if context is not None:
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]
        log_prob = self.log_prob(samples, context=context)
        if context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])
        return samples, log_prob

    def mean(self, context=None):
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def _mean(self, context):
        raise NoMeanException()


class ConditionalIndependentBernoulli(Distribution):
    """An independent Bernoulli whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

    def _compute_params(self, context):
        """Compute the logits from context."""
        if context is None:
            raise ValueError("Context can't be None.")
        logits = self._context_encoder(context)
        if logits.shape[0] != context.shape[0]:
            raise RuntimeError('The batch dimension of the parameters is inconsistent with the input.')
        return logits.reshape(logits.shape[0], *self._shape)

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError('Expected input of shape {}, got {}'.format(self._shape, inputs.shape[1:]))
        logits = self._compute_params(context)
        assert logits.shape == inputs.shape
        log_prob = -inputs * F.softplus(-logits) - (1.0 - inputs) * F.softplus(logits)
        log_prob = torchutils.sum_except_batch(log_prob, num_batch_dims=1)
        return log_prob

    def _sample(self, num_samples, context):
        logits = self._compute_params(context)
        probs = torch.sigmoid(logits)
        probs = torchutils.repeat_rows(probs, num_samples)
        context_size = context.shape[0]
        noise = torch.rand(context_size * num_samples, *self._shape)
        samples = (noise < probs).float()
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        logits = self._compute_params(context)
        return torch.sigmoid(logits)


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self, in_degrees, out_features, autoregressive_features, random_mask, is_output, bias=True):
        super().__init__(in_features=len(in_degrees), out_features=out_features, bias=bias)
        mask, degrees = self._get_mask_and_degrees(in_degrees=in_degrees, out_features=out_features, autoregressive_features=autoregressive_features, random_mask=random_mask, is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

    @classmethod
    def _get_mask_and_degrees(cls, in_degrees, out_features, autoregressive_features, random_mask, is_output):
        if is_output:
            out_degrees = torchutils.tile(_get_input_degrees(autoregressive_features), out_features // autoregressive_features)
            mask = (out_degrees[..., None] > in_degrees).float()
        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(low=min_in_degree, high=autoregressive_features, size=[out_features], dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()
        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self, in_degrees, autoregressive_features, context_features=None, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=0.001)
        else:
            self.batch_norm = None
        self.linear = MaskedLinear(in_degrees=in_degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=random_mask, is_output=False)
        self.degrees = self.linear.degrees
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            temps = self.batch_norm(inputs)
        else:
            temps = inputs
        temps = self.linear(temps)
        temps = self.activation(temps)
        outputs = self.dropout(temps)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self, in_degrees, autoregressive_features, context_features=None, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        features = len(in_degrees)
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=0.001) for _ in range(2)])
        linear_0 = MaskedLinear(in_degrees=in_degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=False, is_output=False)
        linear_1 = MaskedLinear(in_degrees=linear_0.degrees, out_features=features, autoregressive_features=autoregressive_features, random_mask=False, is_output=False)
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError("In a masked residual block, the output degrees can't be less than the corresponding input degrees.")
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-0.001, b=0.001)
            init.uniform_(self.linear_layers[-1].bias, a=-0.001, b=0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if context is not None:
            temps += self.context_layer(context)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self, features, hidden_features, context_features=None, num_blocks=2, output_multiplier=1, use_residual_blocks=True, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()
        self.initial_layer = MaskedLinear(in_degrees=_get_input_degrees(features), out_features=hidden_features, autoregressive_features=features, random_mask=random_mask, is_output=False)
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)
        self.use_residual_blocks = use_residual_blocks
        self.activation = activation
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(block_constructor(in_degrees=prev_out_degrees, autoregressive_features=features, context_features=context_features, random_mask=random_mask, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm))
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = MaskedLinear(in_degrees=prev_out_degrees, out_features=features * output_multiplier, autoregressive_features=features, random_mask=random_mask, is_output=True)

    def forward(self, inputs, context=None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.activation(self.context_layer(context))
        if not self.use_residual_blocks:
            temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class MixtureOfGaussiansMADE(MADE):

    def __init__(self, features, hidden_features, context_features=None, num_blocks=2, num_mixture_components=5, use_residual_blocks=True, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, epsilon=0.01, custom_initialization=True):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__(features, hidden_features, context_features=context_features, num_blocks=num_blocks, output_multiplier=3 * num_mixture_components, use_residual_blocks=use_residual_blocks, random_mask=random_mask, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm)
        self.num_mixture_components = num_mixture_components
        self.features = features
        self.hidden_features = hidden_features
        self.epsilon = epsilon
        if custom_initialization:
            self._initialize()

    def forward(self, inputs, context=None):
        return super().forward(inputs, context=context)

    def log_prob(self, inputs, context=None):
        outputs = self.forward(inputs, context=context)
        outputs = outputs.reshape(*inputs.shape, self.num_mixture_components, 3)
        logits, means, unconstrained_stds = outputs[..., 0], outputs[..., 1], outputs[..., 2]
        log_mixture_coefficients = torch.log_softmax(logits, dim=-1)
        stds = F.softplus(unconstrained_stds) + self.epsilon
        log_prob = torch.sum(torch.logsumexp(log_mixture_coefficients - 0.5 * (np.log(2 * np.pi) + 2 * torch.log(stds) + ((inputs[..., None] - means) / stds) ** 2), dim=-1), dim=-1)
        return log_prob

    def sample(self, num_samples, context=None):
        if context is not None:
            context = torchutils.repeat_rows(context, num_samples)
        with torch.no_grad():
            samples = torch.zeros(context.shape[0], self.features)
            for feature in range(self.features):
                outputs = self.forward(samples, context)
                outputs = outputs.reshape(*samples.shape, self.num_mixture_components, 3)
                logits, means, unconstrained_stds = outputs[:, feature, :, 0], outputs[:, feature, :, 1], outputs[:, feature, :, 2]
                logits = torch.log_softmax(logits, dim=-1)
                stds = F.softplus(unconstrained_stds) + self.epsilon
                component_distribution = distributions.Categorical(logits=logits)
                components = component_distribution.sample((1,)).reshape(-1, 1)
                means, stds = means.gather(1, components).reshape(-1), stds.gather(1, components).reshape(-1)
                samples[:, feature] = (means + torch.randn(context.shape[0]) * stds).detach()
        return samples.reshape(-1, num_samples, self.features)

    def _initialize(self):
        self.final_layer.weight.data[::3, :] = self.epsilon * torch.randn(self.features * self.num_mixture_components, self.hidden_features)
        self.final_layer.bias.data[::3] = self.epsilon * torch.randn(self.features * self.num_mixture_components)
        self.final_layer.weight.data[2::3] = self.epsilon * torch.randn(self.features * self.num_mixture_components, self.hidden_features)
        self.final_layer.bias.data[2::3] = torch.log(torch.exp(torch.Tensor([1 - self.epsilon])) - 1) * torch.ones(self.features * self.num_mixture_components) + self.epsilon * torch.randn(self.features * self.num_mixture_components)


class MADEMoG(Distribution):

    def __init__(self, features, hidden_features, context_features, num_blocks=2, num_mixture_components=1, use_residual_blocks=True, random_mask=False, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, custom_initialization=False):
        super().__init__()
        self._made = MixtureOfGaussiansMADE(features=features, hidden_features=hidden_features, context_features=context_features, num_blocks=num_blocks, num_mixture_components=num_mixture_components, use_residual_blocks=use_residual_blocks, random_mask=random_mask, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm, custom_initialization=custom_initialization)

    def _log_prob(self, inputs, context=None):
        return self._made.log_prob(inputs, context=context)

    def _sample(self, num_samples, context=None):
        return self._made.sample(num_samples, context=context)


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self.register_buffer('_log_z', torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64), persistent=False)

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError('Expected input of shape {}, got {}'.format(self._shape, inputs.shape[1:]))
        neg_energy = -0.5 * torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape, device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            return context.new_zeros(context.shape[0], *self._shape)


class ConditionalDiagonalNormal(Distribution):
    """A diagonal multivariate Normal whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        self.register_buffer('_log_z', torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64), persistent=False)

    def _compute_params(self, context):
        """Compute the means and log stds form the context."""
        if context is None:
            raise ValueError("Context can't be None.")
        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError('The context encoder must return a tensor whose last dimension is even.')
        if params.shape[0] != context.shape[0]:
            raise RuntimeError('The batch dimension of the parameters is inconsistent with the input.')
        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_stds

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError('Expected input of shape {}, got {}'.format(self._shape, inputs.shape[1:]))
        means, log_stds = self._compute_params(context)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        means, log_stds = self._compute_params(context)
        stds = torch.exp(log_stds)
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(stds, num_samples)
        context_size = context.shape[0]
        noise = torch.randn(context_size * num_samples, *self._shape, device=means.device)
        samples = means + stds * noise
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return means


class DiagonalNormal(Distribution):
    """A diagonal multivariate Normal with trainable parameters."""

    def __init__(self, shape):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        self.mean_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.log_std_ = nn.Parameter(torch.zeros(shape).reshape(1, -1))
        self.register_buffer('_log_z', torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64), persistent=False)

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError('Expected input of shape {}, got {}'.format(self._shape, inputs.shape[1:]))
        means = self.mean_
        log_stds = self.log_std_
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs ** 2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def _mean(self, context):
        return self.mean


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        distribution_signature = signature(self._distribution.log_prob)
        distribution_arguments = distribution_signature.parameters.keys()
        self._context_used_in_base = 'context' in distribution_arguments
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), 'embedding_net is not a nn.Module. If you want to use hard-coded summary features, please simply pass the encoded features and pass embedding_net=None'
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        if self._context_used_in_base:
            log_prob = self._distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise = self._distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self._distribution.sample(num_samples * embedded_context.shape[0])
            noise = torch.reshape(repeat_noise, (embedded_context.shape[0], -1, repeat_noise.shape[1]))
        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)
        samples, _ = self._transform.inverse(noise, context=embedded_context)
        if embedded_context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=embedded_context)
        else:
            noise, log_prob = self._distribution.sample_and_log_prob(num_samples)
        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=num_samples)
        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)
        if embedded_context is not None:
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])
        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""
    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class CouplingTransform(Transform):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None):
        """
        Constructor.

        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")
        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)
        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))
        assert self.num_identity_features + self.num_transform_features == self.features
        self.transform_net = transform_net_create_fn(self.num_identity_features, self.num_transform_features * self._transform_dim_multiplier())
        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(features=self.num_identity_features)

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(self.features, inputs.shape[1]))
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]
        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)
        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(identity_split, context)
            logabsdet += logabsdet_identity
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')
        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(self.features, inputs.shape[1]))
        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]
        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(identity_split, context)
        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
        logabsdet += logabsdet_split
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split
        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class AffineCouplingTransform(CouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    The user should supply `scale_activation`, the final activation function in the neural network producing the scale tensor.
    Two options are predefined in the class.
    `DEFAULT_SCALE_ACTIVATION` preserves backwards compatibility but only produces scales <= 1.001.
    `GENERAL_SCALE_ACTIVATION` produces scales <= 3, which is more useful in general applications.
    """
    DEFAULT_SCALE_ACTIVATION = lambda x: torch.sigmoid(x + 2) + 0.001
    GENERAL_SCALE_ACTIVATION = lambda x: (softplus(x) + 0.001).clamp(0, 3)

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, scale_activation=DEFAULT_SCALE_ACTIVATION):
        self.scale_activation = scale_activation
        super().__init__(mask, transform_net_create_fn, unconditional_transform)

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, :self.num_transform_features, ...]
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


class AdditiveCouplingTransform(AffineCouplingTransform):
    """An additive coupling layer, i.e. an affine coupling layer without scaling.

    Reference:
    > L. Dinh et al., NICE:  Non-linear  Independent  Components  Estimation,
    > arXiv:1410.8516, 2014.
    """

    def _transform_dim_multiplier(self):
        return 1

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.ones_like(shift)
        return scale, shift


class BatchNorm(Transform):
    """Transform that performs batch normalization.

    Limitations:
        * It works only for 1-dim inputs.
        * Inverse is not available in training mode, only in eval mode.
    """

    def __init__(self, features, eps=1e-05, momentum=0.1, affine=True):
        if not check.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        constant = np.log(np.exp(1 - eps) - 1)
        self.unconstrained_weight = nn.Parameter(constant * torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer('running_mean', torch.zeros(features))
        self.register_buffer('running_var', torch.zeros(features))

    @property
    def weight(self):
        return F.softplus(self.unconstrained_weight) + self.eps

    def forward(self, inputs, context=None):
        if inputs.dim() != 2:
            raise ValueError('Expected 2-dim inputs, got inputs of shape: {}'.format(inputs.shape))
        if self.training:
            mean, var = inputs.mean(0), inputs.var(0)
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        outputs = self.weight * ((inputs - mean) / torch.sqrt(var + self.eps)) + self.bias
        logabsdet_ = torch.log(self.weight) - 0.5 * torch.log(var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if self.training:
            raise InverseNotAvailable('Batch norm inverse is only available in eval mode, not in training mode.')
        if inputs.dim() != 2:
            raise ValueError('Expected 2-dim inputs, got inputs of shape: {}'.format(inputs.shape))
        outputs = torch.sqrt(self.running_var + self.eps) * ((inputs - self.bias) / self.weight) + self.running_mean
        logabsdet_ = -torch.log(self.weight) + 0.5 * torch.log(self.running_var + self.eps)
        logabsdet = torch.sum(logabsdet_) * inputs.new_ones(inputs.shape[0])
        return outputs, logabsdet


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)


class SimpleRealNVP(Flow):
    """An simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, features, hidden_features, num_layers, num_blocks_per_layer, use_volume_preserving=False, activation=F.relu, dropout_probability=0.0, batch_norm_within_layers=False, batch_norm_between_layers=False):
        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform
        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(in_features, out_features, hidden_features=hidden_features, num_blocks=num_blocks_per_layer, activation=activation, dropout_probability=dropout_probability, use_batch_norm=batch_norm_within_layers)
        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(mask=mask, transform_net_create_fn=create_resnet)
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))
        super().__init__(transform=CompositeTransform(layers), distribution=StandardNormal([features]))


class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(self, in_shape, out_shape, hidden_sizes, activation=F.relu, activate_output=False):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output
        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")
        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs):
        if inputs.shape[1:] != self._in_shape:
            raise ValueError('Expected inputs of shape {}, got {}.'.format(self._in_shape, inputs.shape[1:]))
        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)
        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self._activation(outputs)
        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)
        return outputs


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self, features, context_features, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=0.001) for _ in range(2)])
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList([nn.Linear(features, features) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -0.001, 0.001)
            init.uniform_(self.linear_layers[-1].bias, -0.001, 0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self, in_features, out_features, hidden_features, context_features=None, num_blocks=2, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([ResidualBlock(features=hidden_features, context_features=context_features, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


class ConvResidualBlock(nn.Module):

    def __init__(self, channels, context_channels=None, activation=F.relu, dropout_probability=0.0, use_batch_norm=False, zero_initialization=True):
        super().__init__()
        self.activation = activation
        if context_channels is not None:
            self.context_layer = nn.Conv2d(in_channels=context_channels, out_channels=channels, kernel_size=1, padding=0)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(channels, eps=0.001) for _ in range(2)])
        self.conv_layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -0.001, 0.001)
            init.uniform_(self.conv_layers[-1].bias, -0.001, 0.001)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ConvResidualNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, context_channels=None, num_blocks=2, activation=F.relu, dropout_probability=0.0, use_batch_norm=False):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(in_channels=in_channels + context_channels, out_channels=hidden_channels, kernel_size=1, padding=0)
        else:
            self.initial_layer = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, padding=0)
        self.blocks = nn.ModuleList([ConvResidualBlock(channels=hidden_channels, context_channels=context_channels, activation=activation, dropout_probability=dropout_probability, use_batch_norm=use_batch_norm) for _ in range(num_blocks)])
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class ELUPlus(nn.Module):

    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.0


class IntegrandNet(nn.Module):

    def __init__(self, hidden, cond_in):
        super(IntegrandNet, self).__init__()
        l1 = [1 + cond_in] + hidden
        l2 = hidden + [1]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x, h):
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(nb_batch, -1, in_d).transpose(1, 2).contiguous().view(nb_batch * in_d, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class MonotonicNormalizer(nn.Module):

    def __init__(self, integrand_net, cond_size, nb_steps=20, solver='CC'):
        super(MonotonicNormalizer, self).__init__()
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        else:
            self.integrand_net = integrand_net
        self.solver = solver
        self.nb_steps = nb_steps

    def forward(self, x, h, context=None):
        x0 = torch.zeros(x.shape)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == 'CC':
            z = NeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()), h, self.nb_steps) + z0
        elif self.solver == 'CCParallel':
            z = ParallelNeuralIntegral.apply(x0, xT, self.integrand_net, _flatten(self.integrand_net.parameters()), h, self.nb_steps) + z0
        else:
            return None
        return z, self.integrand_net(x, h)

    def inverse_transform(self, z, h, context=None):
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)
        for i in range(25):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min
        return (x_max + x_min) / 2


class MultiscaleCompositeTransform(Transform):
    """A multiscale composite transform as described in the RealNVP paper.

    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.

    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms, split_dim=1):
        """Constructor.

        Args:
            num_transforms: int, total number of transforms to be added.
            split_dim: dimension along which to split.
        """
        if not check.is_positive_int(split_dim):
            raise TypeError('Split dimension must be a positive integer.')
        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim

    def add_transform(self, transform, transform_output_shape):
        """Add a transform. Must be called exactly `num_transforms` times.

        Parameters:
            transform: the `Transform` object to be added.
            transform_output_shape: tuple, shape of transform's outputs, excl. the first batch
                dimension.

        Returns:
            Input shape for the next transform, or None if adding the last transform.
        """
        assert len(self._transforms) <= self._num_transforms
        if len(self._transforms) == self._num_transforms:
            raise RuntimeError('Adding more than {} transforms is not allowed.'.format(self._num_transforms))
        if self._split_dim - 1 >= len(transform_output_shape):
            raise ValueError('No split_dim in output shape')
        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError('Size of dimension {} must be at least 2.'.format(self._split_dim))
        self._transforms.append(transform)
        if len(self._transforms) != self._num_transforms:
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (output_shape[self._split_dim - 1] + 1) // 2
            output_shape = tuple(output_shape)
            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            output_shape = transform_output_shape
            hidden_shape = None
        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(self, inputs, context=None):
        if self._split_dim >= inputs.dim():
            raise ValueError('No split_dim in inputs.')
        if self._num_transforms != len(self._transforms):
            raise RuntimeError('Expecting exactly {} transform(s) to be added.'.format(self._num_transforms))
        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs
            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context)
                outputs, hiddens = torch.chunk(transform_outputs, chunks=2, dim=self._split_dim)
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet
            outputs, logabsdet = self._transforms[-1](hiddens, context)
            yield outputs, logabsdet
        all_outputs = []
        total_logabsdet = inputs.new_zeros(batch_size)
        for outputs, logabsdet in cascade():
            all_outputs.append(outputs.reshape(batch_size, -1))
            total_logabsdet += logabsdet
        all_outputs = torch.cat(all_outputs, dim=-1)
        return all_outputs, total_logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() != 2:
            raise ValueError('Expecting NxD inputs')
        if self._num_transforms != len(self._transforms):
            raise RuntimeError('Expecting exactly {} transform(s) to be added.'.format(self._num_transforms))
        batch_size = inputs.shape[0]
        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]
        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)
        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i]:split_indices[i + 1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]
        total_logabsdet = inputs.new_zeros(batch_size)
        hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
        total_logabsdet += logabsdet
        for inv_transform, input_chunk in zip(rev_inv_transforms[1:], rev_split_inputs[1:]):
            tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
            hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
            total_logabsdet += logabsdet
        outputs = hiddens
        return outputs, total_logabsdet


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.

        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)


class UMNNCouplingTransform(CouplingTransform):
    """An unconstrained monotonic neural networks coupling layer that transforms the variables.

    Reference:
    > A. Wehenkel and G. Louppe, Unconstrained Monotonic Neural Networks, NeurIPS2019.

    ---- Specific arguments ----
        integrand_net_layers: the layers dimension to put in the integrand network.
        cond_size: The embedding size for the conditioning factors.
        nb_steps: The number of integration steps.
        solver: The quadrature algorithm - CC or CCParallel. Both implements Clenshaw-Curtis quadrature with
        Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps) at once, it is faster
        but requires more memory.

    """

    def __init__(self, mask, transform_net_create_fn, integrand_net_layers=[50, 50, 50], cond_size=20, nb_steps=20, solver='CCParallel', apply_unconditional_transform=False):
        if apply_unconditional_transform:
            unconditional_transform = lambda features: MonotonicNormalizer(integrand_net_layers, 0, nb_steps, solver)
        else:
            unconditional_transform = None
        self.cond_size = cond_size
        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)
        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)

    def _transform_dim_multiplier(self):
        return self.cond_size

    def _coupling_transform_forward(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            z, jac = self.transformer(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = jac.log().sum(1)
            return z, log_det_jac
        else:
            B, C, H, W = inputs.shape
            z, jac = self.transformer(inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]), transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = jac.log().reshape(B, -1).sum(1)
            return z.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac

    def _coupling_transform_inverse(self, inputs, transform_params):
        if len(inputs.shape) == 2:
            x = self.transformer.inverse_transform(inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            z, jac = self.transformer(x, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1))
            log_det_jac = -jac.log().sum(1)
            return x, log_det_jac
        else:
            B, C, H, W = inputs.shape
            x = self.transformer.inverse_transform(inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]), transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            z, jac = self.transformer(x, transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]))
            log_det_jac = -jac.log().reshape(B, -1).sum(1)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac


class PiecewiseCouplingTransform(CouplingTransform):

    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            transform_params = transform_params.reshape(b, d, -1)
        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)
        return outputs, torchutils.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()


def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


class PiecewiseLinearCDF(Transform):

    def __init__(self, shape, num_bins=10, tails=None, tail_bound=1.0):
        super().__init__()
        self.tail_bound = tail_bound
        self.tails = tails
        self.unnormalized_pdf = nn.Parameter(torch.randn(*shape, num_bins))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]
        unnormalized_pdf = _share_across_batch(self.unnormalized_pdf, batch_size)
        if self.tails is None:
            outputs, logabsdet = splines.linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse)
        else:
            outputs, logabsdet = splines.unconstrained_linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tails=self.tails, tail_bound=self.tail_bound)
        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


class PiecewiseLinearCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    def __init__(self, mask, transform_net_create_fn, num_bins=10, tails=None, tail_bound=1.0, apply_unconditional_transform=False, img_shape=None):
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseLinearCDF(shape=[features] + (img_shape if img_shape else []), num_bins=num_bins, tails=tails, tail_bound=tail_bound)
        else:
            unconditional_transform = None
        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_pdf = transform_params
        if self.tails is None:
            return splines.linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse)
        else:
            return splines.unconstrained_linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tails=self.tails, tail_bound=self.tail_bound)


class LinearCache:
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None


class Linear(Transform):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        if not check.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()
        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))
        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()
        elif self.cache.weight is None:
            self.cache.weight = self.weight()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = -self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            self.cache.inverse, self.cache.logabsdet = self.weight_inverse_and_logabsdet()
        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()
        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        if not check.is_bool(mode):
            raise TypeError('Mode must be boolean.')
        self.using_cache = mode

    def weight_and_logabsdet(self):
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class NaiveLinear(Linear):
    """A general linear transform that uses an unconstrained weight matrix.

    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.

    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.
    """

    def __init__(self, features, orthogonal_initialization=True, using_cache=False):
        """Constructor.

        Args:
            features: int, number of input features.
            orthogonal_initialization: bool, if True initialize weights to be a random
                orthogonal matrix.

        Raises:
            TypeError: if `features` is not a positive integer.
        """
        super().__init__(features, using_cache)
        if orthogonal_initialization:
            self._weight = nn.Parameter(torchutils.random_orthogonal(features))
        else:
            self._weight = nn.Parameter(torch.empty(features, features))
            stdv = 1.0 / np.sqrt(features)
            init.uniform_(self._weight, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet = torchutils.logabsdet(self._weight)
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        lu, lu_pivots = torch.lu(self._weight)
        outputs = torch.lu_solve(outputs.t(), lu, lu_pivots).t()
        logabsdet = -torch.sum(torch.log(torch.abs(torch.diag(lu))))
        logabsdet = logabsdet * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(1)
        """
        return self._weight

    def weight_inverse(self):
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        return torch.inverse(self._weight)

    def weight_inverse_and_logabsdet(self):
        """
        Cost:
            inverse = O(D^3)
            logabsdet = O(D)
        where:
            D = num of features
        """
        identity = torch.eye(self.features, self.features)
        lu, lu_pivots = torch.lu(self._weight)
        weight_inv = torch.lu_solve(identity, lu, lu_pivots)
        logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
        return weight_inv, logabsdet

    def logabsdet(self):
        """Cost:
            logabsdet = O(D^3)
        where:
            D = num of features
        """
        return torchutils.logabsdet(self._weight)


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=0.001):
        super().__init__(features, using_cache)
        self.eps = eps
        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)
        n_triangular_entries = (features - 1) * features // 2
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))
        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)
        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0
        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag
        return lower, upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs = torch.linalg.solve_triangular(lower, outputs.t(), upper=False, unitriangular=True)
        outputs = torch.linalg.solve_triangular(upper, outputs, upper=True, unitriangular=False)
        outputs = outputs.t()
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features, device=self.lower_entries.device)
        lower_inverse = torch.linalg.solve_triangular(lower, identity, upper=False, unitriangular=True)
        weight_inverse = torch.linalg.solve_triangular(upper, lower_inverse, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass


class Exp(Transform):

    def forward(self, inputs, context=None):
        outputs = torch.exp(inputs)
        logabsdet = torchutils.sum_except_batch(inputs, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) <= 0.0:
            raise InputOutsideDomain()
        outputs = torch.log(inputs)
        logabsdet = -torchutils.sum_except_batch(outputs, num_batch_dims=1)
        return outputs, logabsdet


class Tanh(Transform):

    def forward(self, inputs, context=None):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class LogTanh(Transform):
    """Tanh with unbounded output. 

    Constructed by selecting a cut_point, and replacing values to the right of cut_point
    with alpha * log(beta * x), and to the left of -cut_point with -alpha * log(-beta *
    x). alpha and beta are set to match the value and the first derivative of tanh at
    cut_point."""

    def __init__(self, cut_point=1):
        if cut_point <= 0:
            raise ValueError('Cut point must be positive.')
        super().__init__()
        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)
        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp((np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha)

    def forward(self, inputs, context=None):
        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)
        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])
        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)
        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log((1 + inputs[mask_middle]) / (1 - inputs[mask_middle]))
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta
        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        logabsdet[mask_left] = -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class LeakyReLU(Transform):

    def __init__(self, negative_slope=0.01):
        if negative_slope <= 0:
            raise ValueError('Slope must be positive.')
        super().__init__()
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.log(torch.as_tensor(self.negative_slope))

    def forward(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = F.leaky_relu(inputs, negative_slope=1 / self.negative_slope)
        mask = (inputs < 0).type(torch.Tensor)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class Sigmoid(Transform):

    def __init__(self, temperature=1, eps=1e-06, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            temperature = torch.Tensor([temperature])
            self.register_buffer('temperature', temperature)

    def forward(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()
        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)
        outputs = 1 / self.temperature * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(torch.log(self.temperature) - F.softplus(-self.temperature * outputs) - F.softplus(self.temperature * outputs))
        return outputs, logabsdet


class Logit(InverseTransform):

    def __init__(self, temperature=1, eps=1e-06):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))


class GatedLinearUnit(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, context=None):
        gate = torch.sigmoid(context)
        return inputs * gate, torch.log(gate).reshape(-1)

    def inverse(self, inputs, context=None):
        gate = torch.sigmoid(context)
        return inputs / gate, -torch.log(gate).reshape(-1)


class CauchyCDF(Transform):

    def __init__(self, location=None, scale=None, features=None):
        super().__init__()

    def forward(self, inputs, context=None):
        outputs = 1 / np.pi * torch.atan(inputs) + 0.5
        logabsdet = torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + inputs ** 2))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()
        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + outputs ** 2))
        return outputs, logabsdet


class CauchyCDFInverse(InverseTransform):

    def __init__(self, location=None, scale=None, features=None):
        super().__init__(CauchyCDF(location=location, scale=scale, features=features))


class CompositeCDFTransform(CompositeTransform):

    def __init__(self, squashing_transform, cdf_transform):
        super().__init__([squashing_transform, cdf_transform, InverseTransform(squashing_transform)])


class ActNorm(Transform):

    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not check.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        super().__init__()
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Expecting inputs to be a 2D or a 4D tensor.')
        if self.training and not self.initialized:
            self._initialize(inputs)
        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = scale * inputs + shift
        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Expecting inputs to be a 2D or a 4D tensor.')
        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale
        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = -h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = -torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class HouseholderSequence(Transform):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, features, num_transforms):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        if not check.is_positive_int(features):
            raise TypeError('Number of features must be a positive integer.')
        if not check.is_positive_int(num_transforms):
            raise TypeError('Number of transforms must be a positive integer.')
        super().__init__()
        self.features = features
        self.num_transforms = num_transforms
        import numpy as np

        def tile(a, dim, n_tile):
            if a.nelement() == 0:
                return a
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(np.concatenate([(init_dim * np.arange(n_tile) + i) for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index)
        qv = tile(torch.eye(num_transforms // 2, features), 0, 2)
        if np.mod(num_transforms, 2) != 0:
            qv = torch.cat((qv, torch.zeros(1, features)))
            qv[-1, num_transforms // 2] = 1
        self.q_vectors = nn.Parameter(qv)

    @staticmethod
    def _apply_transforms(inputs, q_vectors):
        """Apply the sequence of transforms parameterized by given q_vectors to inputs.

        Costs O(KDN), where:
        - K is number of transforms
        - D is dimensionality of inputs
        - N is number of inputs

        Args:
            inputs: Tensor of shape [N, D]
            q_vectors: Tensor of shape [K, D]

        Returns:
            A tuple of:
            - A Tensor of shape [N, D], the outputs.
            - A Tensor of shape [N], the log absolute determinants of the total transform.
        """
        squared_norms = torch.sum(q_vectors ** 2, dim=-1)
        outputs = inputs
        for q_vector, squared_norm in zip(q_vectors, squared_norms):
            temp = outputs @ q_vector
            temp = torch.ger(temp, 2.0 / squared_norm * q_vector)
            outputs = outputs - temp
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._apply_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return self._apply_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self):
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.features, self.features)
        outputs, _ = self.inverse(identity)
        return outputs


class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError('Permutation must be a 1D tensor.')
        if not check.is_positive_int(dim):
            raise ValueError('dim must be a positive integer.')
        super().__init__()
        self._dim = dim
        self.register_buffer('_permutation', permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError('No dimension {} in inputs.'.format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError('Dimension {} in inputs must be of size {}.'.format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.arange(features - 1, -1, -1), dim)


class QRLinear(Linear):
    """A linear module using the QR decomposition for the weight matrix."""

    def __init__(self, features, num_householder, using_cache=False):
        super().__init__(features, using_cache)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)
        n_triangular_entries = (features - 1) * features // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.log_upper_diag = nn.Parameter(torch.zeros(features))
        self.orthogonal = HouseholderSequence(features=features, num_transforms=num_householder)
        self._initialize()

    def _initialize(self):
        stdv = 1.0 / np.sqrt(self.features)
        init.uniform_(self.upper_entries, -stdv, stdv)
        init.uniform_(self.log_upper_diag, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def _create_upper(self):
        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.exp(self.log_upper_diag)
        return upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()
        outputs = F.linear(inputs, upper)
        outputs, _ = self.orthogonal(outputs)
        outputs += self.bias
        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal.inverse(outputs)
        outputs = torch.linalg.solve_triangular(upper, outputs.t(), upper=True)
        outputs = outputs.t()
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        upper = self._create_upper()
        weight, _ = self.orthogonal(upper.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3 + KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        upper = self._create_upper()
        identity = torch.eye(self.features, self.features)
        upper_inv = torch.linalg.solve_triangular(upper, identity, upper=True)
        weight_inv, _ = self.orthogonal(upper_inv)
        return weight_inv

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_upper_diag)


class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()
        if not check.is_int(factor) or factor <= 1:
            raise ValueError('Factor must be an integer > 1.')
        self.factor = factor

    def get_output_shape(self, c, h, w):
        return c * self.factor * self.factor, h // self.factor, w // self.factor

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')
        batch_size, c, h, w = inputs.size()
        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError('Input image size not compatible with the factor.')
        inputs = inputs.view(batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor)
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.view(batch_size, c * self.factor * self.factor, h // self.factor, w // self.factor)
        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')
        batch_size, c, h, w = inputs.size()
        if c < 4 or c % 4 != 0:
            raise ValueError('Invalid number of channel dimensions.')
        inputs = inputs.view(batch_size, c // self.factor ** 2, self.factor, self.factor, h, w)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(batch_size, c // self.factor ** 2, h * self.factor, w * self.factor)
        return inputs, inputs.new_zeros(batch_size)


class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs: Tensor, context=Optional[Tensor]):
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]):
        return self(inputs, context)


class PointwiseAffineTransform(Transform):
    """Forward transform X = X * scale + shift."""

    def __init__(self, shift: Union[Tensor, float]=0.0, scale: Union[Tensor, float]=1.0):
        super().__init__()
        shift, scale = map(torch.as_tensor, (shift, scale))
        if (scale == 0.0).any():
            raise ValueError('Scale must be non-zero.')
        self.register_buffer('_shift', shift)
        self.register_buffer('_scale', scale)

    @property
    def _log_abs_scale(self) ->Tensor:
        return torch.log(torch.abs(self._scale))

    def _batch_logabsdet(self, batch_shape: Iterable[int]) ->Tensor:
        """Return log abs det with input batch shape."""
        if self._log_abs_scale.numel() > 1:
            return self._log_abs_scale.expand(batch_shape).sum()
        else:
            return self._log_abs_scale * torch.Size(batch_shape).numel()

    def forward(self, inputs: Tensor, context=Optional[Tensor]) ->Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()
        outputs = inputs * self._scale + self._shift
        logabsdet = self._batch_logabsdet(batch_shape).expand(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]) ->Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()
        outputs = (inputs - self._shift) / self._scale
        logabsdet = -self._batch_logabsdet(batch_shape).expand(batch_size)
        return outputs, logabsdet


class AffineTransform(PointwiseAffineTransform):

    def __init__(self, shift: Union[Tensor, float]=0.0, scale: Union[Tensor, float]=1.0):
        warnings.warn('Use PointwiseAffineTransform', DeprecationWarning)
        if shift is None:
            shift = 0.0
            warnings.warn(f'`shift=None` deprecated; default is {shift}')
        if scale is None:
            scale = 1.0
            warnings.warn(f'`scale=None` deprecated; default is {scale}.')
        super().__init__(shift, scale)


class SVDLinear(Linear):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(self, features, num_householder, using_cache=False, identity_init=True, eps=0.001):
        super().__init__(features, using_cache)
        assert num_householder % 2 == 0
        self.eps = eps
        self.orthogonal_1 = HouseholderSequence(features=features, num_transforms=num_householder)
        self.unconstrained_diagonal = nn.Parameter(torch.zeros(features))
        self.orthogonal_2 = HouseholderSequence(features=features, num_transforms=num_householder)
        self.identity_init = identity_init
        self._initialize()

    @property
    def diagonal(self):
        return self.eps + F.softplus(self.unconstrained_diagonal)

    @property
    def log_diagonal(self):
        return torch.log(self.diagonal)

    def _initialize(self):
        init.zeros_(self.bias)
        if self.identity_init:
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diagonal, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.unconstrained_diagonal, -stdv, stdv)

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs, _ = self.orthogonal_2(inputs)
        outputs *= self.diagonal
        outputs, _ = self.orthogonal_1(outputs)
        outputs += self.bias
        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(outputs)
        outputs /= self.diagonal
        outputs, _ = self.orthogonal_2.inverse(outputs)
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal = torch.diag(self.diagonal)
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal_inv = torch.diag(torch.reciprocal(self.diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
        return weight_inv.t()

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_diagonal)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvResidualNet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ELUPlus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IdentityTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'in_shape': [4, 4], 'out_shape': [4, 4], 'hidden_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PointwiseAffineTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'features': 4, 'context_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualNet,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'hidden_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_bayesiains_nflows(_paritybench_base):
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

