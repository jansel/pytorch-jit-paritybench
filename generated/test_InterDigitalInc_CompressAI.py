import sys
_module = sys.modules[__name__]
del sys
compressai = _module
datasets = _module
image = _module
pregenerated = _module
rawvideo = _module
video = _module
vimeo90k = _module
entropy_models = _module
entropy_models = _module
layers = _module
gdn = _module
layers = _module
losses = _module
rate_distortion = _module
models = _module
base = _module
google = _module
priors = _module
utils = _module
google = _module
waseda = _module
ops = _module
bound_ops = _module
ops = _module
parametrizers = _module
optimizers = _module
net_aux = _module
registry = _module
torchvision = _module
dataset2latent = _module
extract_codec = _module
extract_quantizers = _module
transforms = _module
functional = _module
transforms = _module
typing = _module
bench = _module
codecs = _module
eval_model = _module
find_close = _module
plot = _module
update_model = _module
zoo = _module
image = _module
pretrained = _module
video = _module
conf = _module
generate_cli_help = _module
codec = _module
train = _module
train_video = _module
setup = _module
test_bench_codec = _module
test_bench_codec_video = _module
test_codec = _module
test_coder = _module
test_datasets = _module
test_datasets_video = _module
test_entropy_models = _module
test_eval_model = _module
test_eval_model_video = _module
test_find_close = _module
test_init = _module
test_layers = _module
test_models = _module
test_ops = _module
test_plot = _module
test_scripting = _module
test_train = _module
test_transforms = _module
test_update_model = _module
test_waseda = _module
test_zoo = _module

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


from torch.utils.data import Dataset


from typing import Tuple


from typing import Union


import numpy as np


import random


import torch


import warnings


from typing import Any


from typing import Callable


from typing import List


from typing import Optional


import scipy.stats


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torch.autograd import Function


import math


from typing import cast


from torch.cuda import amp


from typing import Dict


from typing import Mapping


import torch.optim as optim


from typing import Type


from typing import TypeVar


from torch import optim


from torch.optim import lr_scheduler


from torch.optim import Optimizer


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import DataLoader


import abc


import time


from collections import defaultdict


from torchvision import transforms


from itertools import starmap


from torch.utils.model_zoo import tqdm


from torch.hub import load_state_dict_from_url


from enum import Enum


from typing import IO


from typing import NamedTuple


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


import itertools


import copy


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


def lower_bound_fwd(x: Tensor, bound: Tensor) ->Tensor:
    return torch.max(x, bound)


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer('bound', torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')
        if method not in available_entropy_coders():
            methods = ', '.join(available_entropy_coders())
            raise ValueError(f'Unknown entropy coder "{method}" (available: {methods})')
        if method == 'ans':
            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == 'rangecoder':
            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()
        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def _forward(self, *args: Any) ->Any:
    raise NotImplementedError()


def default_entropy_coder():
    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int=16) ->Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class EntropyModel(nn.Module):
    """Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(self, likelihood_bound: float=1e-09, entropy_coder: Optional[str]=None, entropy_coder_precision: int=16):
        super().__init__()
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.register_buffer('_offset', torch.IntTensor())
        self.register_buffer('_quantized_cdf', torch.IntTensor())
        self.register_buffer('_cdf_length', torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes['entropy_coder'] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop('entropy_coder'))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length
    forward: Callable[..., Any] = _forward

    def quantize(self, inputs: Tensor, mode: str, means: Optional[Tensor]=None) ->Tensor:
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        if mode == 'noise':
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs
        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs

    def _quantize(self, inputs: Tensor, mode: str, means: Optional[Tensor]=None) ->Tensor:
        warnings.warn('_quantize is deprecated. Use quantize instead.')
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(inputs: Tensor, means: Optional[Tensor]=None, dtype: torch.dtype=torch.float) ->Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor]=None) ->Tensor:
        warnings.warn('_dequantize. Use dequantize instead.')
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() first')
        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f'Invalid CDF size {self._quantized_cdf.size()}')

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError('Uninitialized offsets. Run update() first')
        if len(self._offset.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._offset.size()}')

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError('Uninitialized CDF lengths. Run update() first')
        if len(self._cdf_length.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._cdf_length.size()}')

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, 'symbols', means)
        if len(inputs.size()) < 2:
            raise ValueError('Invalid `inputs` size. Expected a tensor with at least 2 dimensions.')
        if inputs.size() != indexes.size():
            raise ValueError('`inputs` and `indexes` should have the same size.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(symbols[i].reshape(-1).int().tolist(), indexes[i].reshape(-1).int().tolist(), self._quantized_cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            strings.append(rv)
        return strings

    def decompress(self, strings: str, indexes: torch.IntTensor, dtype: torch.dtype=torch.float, means: torch.Tensor=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """
        if not isinstance(strings, (tuple, list)):
            raise ValueError('Invalid `strings` parameter type.')
        if not len(strings) == indexes.size(0):
            raise ValueError('Invalid strings or indexes parameters')
        if len(indexes.size()) < 2:
            raise ValueError('Invalid `indexes` size. Expected a tensor with at least 2 dimensions.')
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()
        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError('Invalid means or indexes parameters')
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError('Invalid means parameters')
        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())
        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(s, indexes[i].reshape(-1).int().tolist(), cdf.tolist(), self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype)
        return outputs


class EntropyBottleneck(EntropyModel):
    """Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """
    _offset: Tensor

    def __init__(self, channels: int, *args: Any, tail_mass: float=1e-09, init_scale: float=10, filters: Tuple[int, ...]=(3, 3, 3, 3), **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f'_matrix{i:d}', nn.Parameter(matrix))
            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f'_bias{i:d}', nn.Parameter(bias))
            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f'_factor{i:d}', nn.Parameter(factor))
        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer('target', torch.Tensor([-target, 0, target]))

    def _get_medians(self) ->Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool=False) ->bool:
        if self._offset.numel() > 0 and not force:
            return False
        medians = self.quantiles[:, 0, 1]
        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)
        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)
        self._offset = -minima
        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]
        half = float(0.5)
        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) ->Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) ->Tensor:
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f'_matrix{i:d}')
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)
            bias = getattr(self, f'_bias{i:d}')
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self.filters):
                factor = getattr(self, f'_factor{i:d}')
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs: Tensor) ->Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return likelihood

    def forward(self, x: Tensor, training: Optional[bool]=None) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if not torch.jit.is_scripting():
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)
        outputs = self.quantize(values, 'noise' if training else 'dequantize', self._get_medians())
        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]
        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()
        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = len(strings), self._quantized_cdf.size(0), *size
        indexes = self._build_indexes(output_size)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditional(EntropyModel):
    """Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(self, scale_table: Optional[Union[List, Tuple]], *args: Any, scale_bound: float=0.11, tail_mass: float=1e-09, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')
        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')
        if scale_table and (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')
        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError('Invalid parameters')
        self.lower_bound_scale = LowerBound(scale_bound)
        self.register_buffer('scale_table', self._prepare_scale_table(scale_table) if scale_table else torch.Tensor())
        self.register_buffer('scale_bound', torch.Tensor([float(scale_bound)]) if scale_bound is not None else None)

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) ->Tensor:
        half = float(0.5)
        const = float(-2 ** -0.5)
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()
        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower
        tail_mass = 2 * lower[:, :1]
        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor]=None) ->Tensor:
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = self.lower_bound_scale(scales)
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def forward(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor]=None, training: Optional[bool]=None) ->Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, 'noise' if training else 'dequantize', means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales: Tensor) ->Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """
    pedestal: Tensor

    def __init__(self, minimum: float=0, reparam_offset: float=2 ** -18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset ** 2
        self.register_buffer('pedestal', torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) ->Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) ->Tensor:
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out


class GDN(nn.Module):
    """Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \\frac{x[i]}{\\sqrt{\\beta[i] + \\sum_j(\\gamma[j, i] * x[j]^2)}}

    """

    def __init__(self, in_channels: int, inverse: bool=False, beta_min: float=1e-06, gamma_init: float=0.1):
        super().__init__()
        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) ->Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)
        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        out = x * norm
        return out


class GDN1(GDN):
    """Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \\frac{x[i]}{\\beta[i] + \\sum_j(\\gamma[j, i] * |x[j]|}

    """

    def forward(self, x: Tensor) ->Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)
        if not self.inverse:
            norm = 1.0 / norm
        out = x * norm
        return out


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str='A', **kwargs: Any):
        super().__init__(*args, **kwargs)
        if mask_type not in ('A', 'B'):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: Tensor) ->Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


def conv1x1(in_ch: int, out_ch: int, stride: int=1) ->nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int=1) ->nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        if self.skip is not None:
            identity = self.skip(x)
        out += identity
        return out


def subpel_conv3x3(in_ch: int, out_ch: int, r: int=1) ->nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r))


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int=2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()


        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(conv1x1(N, N // 2), nn.ReLU(inplace=True), conv3x3(N // 2, N // 2), nn.ReLU(inplace=True), conv1x1(N // 2, N))
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) ->Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out
        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(N, N))

    def forward(self, x: Tensor) ->Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0
    for i, frame_likelihoods in enumerate(likelihoods_list):
        frame_bpp = 0
        for label, likelihoods in frame_likelihoods.items():
            label_bpp = 0
            for field, v in likelihoods.items():
                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)
                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp
                bpp_info_dict[f'bpp_loss.{label}'] += bpp.sum()
                bpp_info_dict[f'bpp_loss.{label}.{i}.{field}'] = bpp.sum()
            bpp_info_dict[f'bpp_loss.{label}.{i}'] = label_bpp.sum()
        bpp_info_dict[f'bpp_loss.{i}'] = frame_bpp.sum()
    return bpp_loss, bpp_info_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, return_details: bool=False, bitdepth: int=8):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2 ** bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) for frame_likelihoods in likelihoods_list for likelihoods in frame_likelihoods)

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f'len(x)={len(x)} != len(target)={len(target)})')
        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError('number of channels mismatches while computing distortion')
        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)
        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)
        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) ->bool:
        return isinstance(x, torch.Tensor) and x.ndimension() == 4 or isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)

    @classmethod
    def _check_tensors_list(cls, lst):
        if not isinstance(lst, (tuple, list)) or len(lst) < 1 or any(not cls._check_tensor(x) for x in lst):
            raise ValueError('Expected a list of 4D torch.Tensor (or tuples of) as input')

    def forward(self, output, target):
        assert isinstance(target, type(output['x_hat']))
        assert len(output['x_hat']) == len(target)
        self._check_tensors_list(target)
        self._check_tensors_list(output['x_hat'])
        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output['x_hat'], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)
            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)
            if self.return_details:
                out[f'frame{i}.mse_loss'] = distortion
        out['mse_loss'] = torch.stack(distortions).mean()
        scaled_distortions = sum(scaled_distortions) / num_frames
        assert isinstance(output['likelihoods'], list)
        likelihoods_list = output.pop('likelihoods')
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        if self.return_details:
            out.update(bpp_info_dict)
        lambdas = torch.full_like(bpp_loss, self.lmbda)
        bpp_loss = bpp_loss.mean()
        out['loss'] = (lambdas * scaled_distortions).mean() + bpp_loss
        out['distortion'] = scaled_distortions.mean()
        out['bpp_loss'] = bpp_loss
        return out


SCALES_LEVELS = 64


SCALES_MAX = 256


SCALES_MIN = 0.11


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(module, buffer_name, state_dict_key, state_dict, policy='resize_if_empty', dtype=torch.int):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)
    if policy in ('resize_if_empty', 'resize'):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == 'resize' or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == 'register':
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(module, module_name, buffer_names, state_dict, policy='resize_if_empty', dtype=torch.int):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        _update_registered_buffer(module, buffer_name, f'{module_name}.{buffer_name}', state_dict, policy, dtype)


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with any number of
    EntropyBottleneck or GaussianConditional modules.
    """

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue
            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(module, name, ['_quantized_cdf', '_offset', '_cdf_length'], state_dict)
            if isinstance(module, GaussianConditional):
                update_registered_buffers(module, name, ['_quantized_cdf', '_offset', '_cdf_length', 'scale_table'], state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) ->Tensor:
        """Returns the total auxiliary loss over all ``EntropyBottleneck``\\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)


TModel = nn.Module


TModel_b = TypeVar('TModel_b', bound=TModel)


def register_model(name: str):
    """Decorator for registering a model."""

    def decorator(cls: Type[TModel_b]) ->Type[TModel_b]:
        MODELS[name] = cls
        return cls
    return decorator


class FactorizedPrior(CompressionModel):
    """Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y
            x ──►─┤g_a├──►─┐
                  └───┘    │
                           ▼
                         ┌─┴─┐
                         │ Q │
                         └─┬─┘
                           │
                     y_hat ▼
                           │
                           ·
                        EB :
                           ·
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N), conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, 3))
        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) ->int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {'strings': [y_strings], 'shape': y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class FactorizedPriorReLU(FactorizedPrior):
    """Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.
    GDN activations are replaced by ReLU.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(conv(3, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), deconv(N, 3))


class ScaleHyperprior(CompressionModel):
    """Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N), conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True), deconv(N, 3))
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.ReLU(inplace=True), conv(N, N), nn.ReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, N), nn.ReLU(inplace=True), deconv(N, N), nn.ReLU(inplace=True), conv(N, M, stride=1, kernel_size=3), nn.ReLU(inplace=True))
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) ->int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class MeanScaleHyperprior(ScaleHyperprior):
    """Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.LeakyReLU(inplace=True), conv(N, N), nn.LeakyReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, M), nn.LeakyReLU(inplace=True), deconv(M, M * 3 // 2), nn.LeakyReLU(inplace=True), conv(M * 3 // 2, M * 2, stride=1, kernel_size=3))

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}


class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    """Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                   params ▼
                         └─┬─┘                                          │
                     y_hat ▼                  ┌─────┐                   │
                           ├──────────►───────┤  CP ├────────►──────────┤
                           │                  └─────┘                   │
                           ▼                                            ▼
                           │                                            │
                           ·                  ┌─────┐                   │
                        GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                           ·     scales_hat   └─────┘
                           │      means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional
        EP = Entropy parameters network
        CP = Context prediction (masked convolution)

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(conv(3, N, kernel_size=5, stride=2), GDN(N), conv(N, N, kernel_size=5, stride=2), GDN(N), conv(N, N, kernel_size=5, stride=2), GDN(N), conv(N, M, kernel_size=5, stride=2))
        self.g_s = nn.Sequential(deconv(M, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, N, kernel_size=5, stride=2), GDN(N, inverse=True), deconv(N, 3, kernel_size=5, stride=2))
        self.h_a = nn.Sequential(conv(M, N, stride=1, kernel_size=3), nn.LeakyReLU(inplace=True), conv(N, N, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), conv(N, N, stride=2, kernel_size=5))
        self.h_s = nn.Sequential(deconv(N, M, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), deconv(M, M * 3 // 2, stride=2, kernel_size=5), nn.LeakyReLU(inplace=True), conv(M * 3 // 2, M * 2, stride=1, kernel_size=3))
        self.entropy_parameters = nn.Sequential(nn.Conv2d(M * 12 // 3, M * 10 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(M * 10 // 3, M * 8 // 3, 1), nn.LeakyReLU(inplace=True), nn.Conv2d(M * 8 // 3, M * 6 // 3, 1))
        self.context_prediction = MaskedConv2d(M, 2 * M, kernel_size=5, padding=2, stride=1)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) ->int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        y_hat = self.gaussian_conditional.quantize(y, 'noise' if self.training else 'dequantize')
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}}

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.weight'].size(0)
        M = state_dict['g_a.6.weight'].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).')
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding)
            y_strings.append(string)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, bias=self.context_prediction.bias)
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, 'symbols', means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat
                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        if next(self.parameters()).device != torch.device('cpu'):
            warnings.warn('Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU).')
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)
        s = 4
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = torch.zeros((z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding), device=z_hat.device)
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(y_string, y_hat[i:i + 1], params[i:i + 1], y_height, y_width, kernel_size, padding)
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {'x_hat': x_hat}

    def _decompress_ar(self, y_string, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, self.context_prediction.weight, bias=self.context_prediction.bias)
                p = params[:, :, h:h + 1, w:w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)
                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp:hp + 1, wp:wp + 1] = rv


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2 ** bit_depth - 1
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_sub = torch.exp(-ctx.alpha ** ctx.beta * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta) * grad_output.clone()
        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]
        return grad_input, None, None


def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError('Missing kernel_size or sigma parameters')
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)
    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode='replicate')
    x = torch.nn.functional.conv2d(x, kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)), groups=x.size(1))
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)


def quantize_ste(x: Tensor) ->Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


class ScaleSpaceFlow(CompressionModel):
    """Google's first end-to-end optimized video compression from E.
    Agustsson, D. Minnen, N. Johnston, J. Balle, S. J. Hwang, G. Toderici: `"Scale-space flow for end-to-end
    optimized video compression" <https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html>`_,
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020).

    Args:
        num_levels (int): Number of Scale-space
        sigma0 (float): standard deviation for gaussian kernel of the first space scale.
        scale_field_shift (float):
    """

    def __init__(self, num_levels: int=5, sigma0: float=1.5, scale_field_shift: float=1.0):
        super().__init__()


        class Encoder(nn.Sequential):

            def __init__(self, in_planes: int, mid_planes: int=128, out_planes: int=192):
                super().__init__(conv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, out_planes, kernel_size=5, stride=2))


        class Decoder(nn.Sequential):

            def __init__(self, out_planes: int, in_planes: int=192, mid_planes: int=128):
                super().__init__(deconv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, out_planes, kernel_size=5, stride=2))


        class HyperEncoder(nn.Sequential):

            def __init__(self, in_planes: int=192, mid_planes: int=192, out_planes: int=192):
                super().__init__(conv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), conv(mid_planes, mid_planes, kernel_size=5, stride=2))


        class HyperDecoder(nn.Sequential):

            def __init__(self, in_planes: int=192, mid_planes: int=192, out_planes: int=192):
                super().__init__(deconv(in_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, mid_planes, kernel_size=5, stride=2), nn.ReLU(inplace=True), deconv(mid_planes, out_planes, kernel_size=5, stride=2))


        class HyperDecoderWithQReLU(nn.Module):

            def __init__(self, in_planes: int=192, mid_planes: int=192, out_planes: int=192):
                super().__init__()

                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)
                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))
                return x


        class Hyperprior(CompressionModel):

            def __init__(self, planes: int=192, mid_planes: int=192):
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(mid_planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
                self.gaussian_conditional = GaussianConditional(None)

            def forward(self, y):
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                return y_hat, {'y': y_likelihoods, 'z': z_likelihoods}

            def compress(self, y):
                z = self.hyper_encoder(y)
                z_string = self.entropy_bottleneck.compress(z)
                z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_string = self.gaussian_conditional.compress(y, indexes, means)
                y_hat = self.gaussian_conditional.quantize(y, 'dequantize', means)
                return y_hat, {'strings': [y_string, z_string], 'shape': z.size()[-2:]}

            def decompress(self, strings, shape):
                assert isinstance(strings, list) and len(strings) == 2
                z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                indexes = self.gaussian_conditional.build_indexes(scales)
                y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)
                return y_hat
        self.img_encoder = Encoder(3)
        self.img_decoder = Decoder(3)
        self.img_hyperprior = Hyperprior()
        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3, in_planes=384)
        self.res_hyperprior = Hyperprior()
        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1)
        self.motion_hyperprior = Hyperprior()
        self.sigma0 = sigma0
        self.num_levels = num_levels
        self.scale_field_shift = scale_field_shift

    def forward(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f'Invalid number of frames: {len(frames)}.')
        reconstructions = []
        frames_likelihoods = []
        x_hat, likelihoods = self.forward_keyframe(frames[0])
        reconstructions.append(x_hat)
        frames_likelihoods.append(likelihoods)
        x_ref = x_hat.detach()
        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, likelihoods = self.forward_inter(x, x_ref)
            reconstructions.append(x_ref)
            frames_likelihoods.append(likelihoods)
        return {'x_hat': reconstructions, 'likelihoods': frames_likelihoods}

    def forward_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, likelihoods = self.img_hyperprior(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, {'keyframe': likelihoods}

    def encode_keyframe(self, x):
        y = self.img_encoder(x)
        y_hat, out_keyframe = self.img_hyperprior.compress(y)
        x_hat = self.img_decoder(y_hat)
        return x_hat, out_keyframe

    def decode_keyframe(self, strings, shape):
        y_hat = self.img_hyperprior.decompress(strings, shape)
        x_hat = self.img_decoder(y_hat)
        return x_hat

    def forward_inter(self, x_cur, x_ref):
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec, {'motion': motion_likelihoods, 'residual': res_likelihoods}

    def encode_inter(self, x_cur, x_ref):
        x = torch.cat((x_cur, x_ref), dim=1)
        y_motion = self.motion_encoder(x)
        y_motion_hat, out_motion = self.motion_hyperprior.compress(y_motion)
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, out_res = self.res_hyperprior.compress(y_res)
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec, {'strings': {'motion': out_motion['strings'], 'residual': out_res['strings']}, 'shape': {'motion': out_motion['shape'], 'residual': out_res['shape']}}

    def decode_inter(self, x_ref, strings, shapes):
        key = 'motion'
        y_motion_hat = self.motion_hyperprior.decompress(strings[key], shapes[key])
        motion_info = self.motion_decoder(y_motion_hat)
        x_pred = self.forward_prediction(x_ref, motion_info)
        key = 'residual'
        y_res_hat = self.res_hyperprior.decompress(strings[key], shapes[key])
        y_combine = torch.cat((y_res_hat, y_motion_hat), dim=1)
        x_res_hat = self.res_decoder(y_combine)
        x_rec = x_pred + x_res_hat
        return x_rec

    @staticmethod
    def gaussian_volume(x, sigma: float, num_levels: int):
        """Efficient gaussian volume construction.

        From: "Generative Video Compression as Hierarchical Variational Inference",
        by Yang et al.
        """
        k = 2 * int(math.ceil(3 * sigma)) + 1
        device = x.device
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        kernel = gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        volume = [x.unsqueeze(2)]
        x = gaussian_blur(x, kernel=kernel)
        volume += [x.unsqueeze(2)]
        for i in range(1, num_levels):
            x = F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
            x = gaussian_blur(x, kernel=kernel)
            interp = x
            for _ in range(0, i):
                interp = F.interpolate(interp, scale_factor=2, mode='bilinear', align_corners=False)
            volume.append(interp.unsqueeze(2))
        return torch.cat(volume, dim=2)

    @amp.autocast(enabled=False)
    def warp_volume(self, volume, flow, scale_field, padding_mode: str='border'):
        """3D volume warping."""
        if volume.ndimension() != 5:
            raise ValueError(f'Invalid number of dimensions for volume {volume.ndimension()}')
        N, C, _, H, W = volume.size()
        grid = meshgrid2d(N, C, H, W, volume.device)
        update_grid = grid + flow.permute(0, 2, 3, 1).float()
        update_scale = scale_field.permute(0, 2, 3, 1).float()
        volume_grid = torch.cat((update_grid, update_scale), dim=-1).unsqueeze(1)
        out = F.grid_sample(volume.float(), volume_grid, padding_mode=padding_mode, align_corners=False)
        return out.squeeze(2)

    def forward_prediction(self, x_ref, motion_info):
        flow, scale_field = motion_info.chunk(2, dim=1)
        volume = self.gaussian_volume(x_ref, self.sigma0, self.num_levels)
        x_pred = self.warp_volume(volume, flow, scale_field)
        return x_pred

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel) and m is not self:
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def compress(self, frames):
        if not isinstance(frames, List):
            raise RuntimeError(f'Invalid number of frames: {len(frames)}.')
        frame_strings = []
        shape_infos = []
        x_ref, out_keyframe = self.encode_keyframe(frames[0])
        frame_strings.append(out_keyframe['strings'])
        shape_infos.append(out_keyframe['shape'])
        for i in range(1, len(frames)):
            x = frames[i]
            x_ref, out_interframe = self.encode_inter(x, x_ref)
            frame_strings.append(out_interframe['strings'])
            shape_infos.append(out_interframe['shape'])
        return frame_strings, shape_infos

    def decompress(self, strings, shapes):
        if not isinstance(strings, List) or not isinstance(shapes, List):
            raise RuntimeError(f'Invalid number of frames: {len(strings)}.')
        assert len(strings) == len(shapes), f'Number of information should match {len(strings)} != {len(shapes)}.'
        dec_frames = []
        x_ref = self.decode_keyframe(strings[0], shapes[0])
        dec_frames.append(x_ref)
        for i in range(1, len(strings)):
            string = strings[i]
            shape = shapes[i]
            x_ref = self.decode_inter(x_ref, string, shape)
            dec_frames.append(x_ref)
        return dec_frames

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net


class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(ResidualBlockWithStride(3, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), conv3x3(N, N, stride=2))
        self.h_a = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2), nn.LeakyReLU(inplace=True), conv3x3(N, N), nn.LeakyReLU(inplace=True), conv3x3(N, N, stride=2))
        self.h_s = nn.Sequential(conv3x3(N, N), nn.LeakyReLU(inplace=True), subpel_conv3x3(N, N, 2), nn.LeakyReLU(inplace=True), conv3x3(N, N * 3 // 2), nn.LeakyReLU(inplace=True), subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2), nn.LeakyReLU(inplace=True), conv3x3(N * 3 // 2, N * 2))
        self.g_s = nn.Sequential(ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), subpel_conv3x3(N, 3, 2))

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict['g_a.0.conv1.weight'].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)
        self.g_a = nn.Sequential(ResidualBlockWithStride(3, N, stride=2), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), AttentionBlock(N), ResidualBlock(N, N), ResidualBlockWithStride(N, N, stride=2), ResidualBlock(N, N), conv3x3(N, N, stride=2), AttentionBlock(N))
        self.g_s = nn.Sequential(AttentionBlock(N), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), AttentionBlock(N), ResidualBlock(N, N), ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N), subpel_conv3x3(N, 3, 2))


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class DummyCompressionModel(CompressionModel):

    def __init__(self, entropy_bottleneck_channels):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)


class Foo(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionBlock,
     lambda: ([], {'N': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GDN,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GDN1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LowerBound,
     lambda: ([], {'bound': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NonNegativeParametrizer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlockUpsample,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlockWithStride,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_InterDigitalInc_CompressAI(_paritybench_base):
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

