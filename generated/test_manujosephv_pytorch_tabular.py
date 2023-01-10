import sys
_module = sys.modules[__name__]
del sys
adhoc_scaffold = _module
to_test_classification = _module
to_test_node = _module
to_test_regression = _module
to_test_regression_custom_models = _module
pytorch_tabular = _module
augmentations = _module
categorical_encoders = _module
config = _module
config = _module
feature_extractor = _module
models = _module
autoint = _module
autoint = _module
base_model = _module
category_embedding = _module
category_embedding_model = _module
common = _module
ft_transformer = _module
ft_transformer = _module
mixture_density = _module
mdn = _module
node = _module
architecture_blocks = _module
node_model = _module
odst = _module
utils = _module
tab_transformer = _module
tab_transformer = _module
tabnet = _module
tabnet_model = _module
ssl = _module
tabular_datamodule = _module
tabular_model = _module
utils = _module
setup = _module
tests = _module
conftest = _module
test_augmentations = _module
test_autoint = _module
test_categorical_embedding = _module
test_common = _module
test_datamodule = _module
test_ft_transformer = _module
test_mdn = _module
test_node = _module
test_tabnet = _module
test_tabtransformer = _module

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


import numpy as np


from torch.functional import norm


from sklearn.datasets import fetch_covtype


import pandas as pd


from sklearn.preprocessing import PowerTransformer


from sklearn.model_selection import train_test_split


from sklearn.datasets import fetch_california_housing


from torch.utils import data


import torch.nn as nn


import torch.nn.functional as F


from typing import Dict


from typing import List


from typing import Optional


import logging


import math


from torch.autograd import Variable


from typing import Any


from typing import Tuple


from collections import defaultdict


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


from abc import ABCMeta


from abc import abstractmethod


from typing import Callable


from torch import Tensor


from torch import einsum


from torch import nn


from collections import OrderedDict


from torch.distributions import Categorical


from warnings import warn


from torch.autograd import Function


from torch.jit import script


import re


from typing import Iterable


from typing import Union


from pandas.tseries import offsets


from pandas.tseries.frequencies import to_offset


from sklearn.base import copy


from sklearn.preprocessing import FunctionTransformer


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import QuantileTransformer


from sklearn.preprocessing import StandardScaler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import inspect


from sklearn.cluster import KMeans


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PositionWiseFeedForward(nn.Module):
    """
    title: Position-wise Feed-Forward Network (FFN)
    summary: Documented reusable implementation of the position wise feedforward network.

    # Position-wise Feed-Forward Network (FFN)
    This is a [PyTorch](https://pytorch.org)  implementation
    of position-wise feedforward network used in transformer.
    FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$.
    So it is sometime also called the expand-and-contract network.
    There is an activation at the hidden layer, which is
    usually set to ReLU (Rectified Linear Unit) activation, $$\\max(0, x)$$
    That is, the FFN function is,
    $$FFN(x, W_1, W_2, b_1, b_2) = \\max(0, x W_1 + b_1) W_2 + b_2$$
    where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.
    Sometimes the
    GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
    $$x \\Phi(x)$$ where $\\Phi(x) = P(X \\le x), X \\sim \\mathcal{N}(0,1)$
    ### Gated Linear Units
    This is a generic implementation that supports different variants including
    [Gated Linear Units](https://arxiv.org/abs/2002.05202) (GLU).
    We have also implemented experiments on these:
    * [experiment that uses `labml.configs`](glu_variants/experiment.html)
    * [simpler version from scratch](glu_variants/simple.html)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1, activation=nn.ReLU(), is_gated: bool=False, bias1: bool=True, bias2: bool=True, bias_gate: bool=True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class GEGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.GELU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class ReGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.ReLU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.SiLU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class AddNorm(nn.Module):
    """
    Applies LayerNorm, Dropout and adds to input. Standard AddNorm operations in Transformers
    """

    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) ->torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    """
    Multi Headed Attention Block in Transformers
    """

    def __init__(self, input_dim: int, num_heads: int=8, head_dim: int=16, dropout: int=0.1, keep_attn: bool=True):
        super().__init__()
        assert input_dim % num_heads == 0, "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim ** -0.5
        self.keep_attn = keep_attn
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if self.keep_attn:
            self.attn_weights = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class SharedEmbeddings(nn.Module):
    """
    Enables different values in a categorical feature to share some embeddings across
    """

    def __init__(self, num_embed: int, embed_dim: int, add_shared_embed: bool=False, frac_shared_embed: float=0.25):
        super(SharedEmbeddings, self).__init__()
        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: torch.Tensor) ->torch.Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, :shared_embed.shape[1]] = shared_embed
        return out

    @property
    def weight(self):
        w = self.embed.weight.detach()
        if self.add_shared_embed:
            w += self.shared_embed
        else:
            w[:, :self.shared_embed.shape[1]] = self.shared_embed
        return w


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block
    """

    def __init__(self, input_embed_dim: int, num_heads: int=8, ff_hidden_multiplier: int=4, ff_activation: str='GEGLU', attn_dropout: float=0.1, keep_attn: bool=True, ff_dropout: float=0.1, add_norm_dropout: float=0.1, transformer_head_dim: Optional[int]=None):
        super().__init__()
        self.mha = MultiHeadedAttention(input_embed_dim, num_heads, head_dim=input_embed_dim if transformer_head_dim is None else transformer_head_dim, dropout=attn_dropout, keep_attn=keep_attn)
        try:
            self.pos_wise_ff = getattr(common, ff_activation)(d_model=input_embed_dim, d_ff=input_embed_dim * ff_hidden_multiplier, dropout=ff_dropout)
        except AttributeError:
            self.pos_wise_ff = getattr(common, 'PositionWiseFeedForward')(d_model=input_embed_dim, d_ff=input_embed_dim * ff_hidden_multiplier, dropout=ff_dropout, activation=getattr(nn, self.hparams.ff_activation))
        self.attn_add_norm = AddNorm(input_embed_dim, add_norm_dropout)
        self.ff_add_norm = AddNorm(input_embed_dim, add_norm_dropout)

    def forward(self, x):
        y = self.mha(x)
        x = self.attn_add_norm(x, y)
        y = self.pos_wise_ff(y)
        return self.ff_add_norm(x, y)


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == 'kaiming_uniform':
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == 'kaiming_normal':
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError('initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`')


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int, initialization: str) ->None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


LOG2PI = math.log(2 * math.pi)


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class ModuleWithInit(nn.Module):
    """ Base class for pytorch module with data-aware initializer on first batch """

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None

    def initialize(self, *args, **kwargs):
        """ initialize module tensors using first batch of data """
        raise NotImplementedError('Please implement ')

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


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

        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax

        Returns:
            output (Tensor): same shape as input
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

        Args:
            input: any dimension
            dim: dimension along which to apply the sparsemax

        Returns:
            the threshold value
        """
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size
        return tau, support_size


def sparsemax(input, dim=-1):
    return SparsemaxFunction.apply(input, dim)


def sparsemoid(input):
    return (0.5 * input + 0.5).clamp_(0, 1)


class ODST(ModuleWithInit):

    def __init__(self, in_features, num_trees, depth=6, tree_output_dim=1, flatten_output=True, choice_function=sparsemax, bin_function=sparsemoid, initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_, threshold_init_beta=1.0, threshold_init_cutoff=1.0):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_output_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.response = nn.Parameter(torch.zeros([num_trees, tree_output_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)
        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)
        self.feature_thresholds = nn.Parameter(torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True)
        self.log_temperatures = nn.Parameter(torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True)
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = indices.view(1, -1) // offsets.view(-1, 1) % 2
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)
        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        bins = self.bin_function(threshold_logits)
        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        response_weights = torch.prod(bin_matches, dim=-2)
        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-06):
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn('Data-aware initialization is performed on less than 1000 data points. This may cause instability.To avoid potential problems, run this model on a data batch with at least 1000 data samples.You can do so manually before training. Use with torch.no_grad() for memory efficiency.')
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta, size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())), dtype=feature_values.dtype, device=feature_values.device).view(self.num_trees, self.depth)
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)), q=100 * min(1.0, self.threshold_init_cutoff), axis=0)
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return '{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})'.format(self.__class__.__name__, self.feature_selection_logits.shape[0], self.num_trees, self.depth, self.tree_dim, self.flatten_output)


class DenseODSTBlock(nn.Sequential):

    def __init__(self, input_dim, num_trees, num_layers, tree_output_dim=1, max_features=None, input_dropout=0.0, flatten_output=False, Module=ODST, **kwargs):
        layers = []
        for i in range(num_layers):
            oddt = Module(input_dim, num_trees, tree_output_dim=tree_output_dim, flatten_output=True, **kwargs)
            input_dim = min(input_dim + num_trees * tree_output_dim, max_features or float('inf'))
            layers.append(oddt)
        super().__init__(*layers)
        self.num_layers, self.layer_dim, self.tree_dim = num_layers, num_trees, tree_output_dim
        self.max_features, self.flatten_output = max_features, flatten_output
        self.input_dropout = input_dropout

    def forward(self, x):
        initial_features = x.shape[-1]
        for layer in self:
            layer_inp = x
            if self.max_features is not None:
                tail_features = min(self.max_features, layer_inp.shape[-1]) - initial_features
                if tail_features != 0:
                    layer_inp = torch.cat([layer_inp[..., :initial_features], layer_inp[..., -tail_features:]], dim=-1)
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)
        outputs = x[..., initial_features:]
        if not self.flatten_output:
            outputs = outputs.view(*outputs.shape[:-1], self.num_layers * self.layer_dim, self.tree_dim)
        return outputs


class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddNorm,
     lambda: ([], {'input_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseODSTBlock,
     lambda: ([], {'input_dim': 4, 'num_trees': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GEGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'func': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (ODST,
     lambda: ([], {'in_features': 4, 'num_trees': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PositionWiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwiGLU,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_manujosephv_pytorch_tabular(_paritybench_base):
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

