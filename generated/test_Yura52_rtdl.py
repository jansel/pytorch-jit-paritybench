import sys
_module = sys.modules[__name__]
del sys
conf = _module
rtdl = _module
_utils = _module
data = _module
exceptions = _module
functional = _module
modules = _module
nn = _module
_attention = _module
_backbones = _module
_embeddings = _module
_models = _module
_utils = _module
optim = _module
tests = _module
test_data = _module
test_modules = _module
test_vs_paper = _module

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


import math


import warnings


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import TypeVar


from typing import Union


from typing import cast


from typing import overload


import numpy as np


import scipy.sparse


import torch


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


from sklearn.preprocessing import QuantileTransformer


from sklearn.preprocessing import StandardScaler


from sklearn.tree import DecisionTreeClassifier


from sklearn.tree import DecisionTreeRegressor


from sklearn.utils import check_random_state


from torch import Tensor


from torch import as_tensor


import torch.nn.functional as F


import enum


import time


from typing import Tuple


from typing import Type


import torch.nn as nn


import torch.optim


from collections import OrderedDict


import collections.abc


import itertools


from torch.nn.parameter import Parameter


from typing import Generic


from typing import Set


import random


import typing as ty


class ReGLU(nn.Module):

    def forward(self, x: Tensor) ->Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError('The size of the last dimension must be even.')
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


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
        return rtdlF.geglu(x)


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) ->'_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) ->None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    """Transforms continuous features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    For one feature, the transformation consists of two steps:

    * the feature is multiplied by a trainable vector
    * another trainable vector is added

    Note that each feature has its separate pair of trainable vectors, i.e. the vectors
    are not shared between features.

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = NumericalFeatureTokenizer(n_features, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(self, n_features: int, d_token: int, bias: bool, initialization: str) ->None:
        """
        Args:
            n_features: the number of continuous (scalar) features
            d_token: the size of one token
            bias: if `False`, then the transformation will include only multiplication.
                **Warning**: :code:`bias=False` leads to significantly worse results for
                Transformer-like (token-based) architectures.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
                In [gorishniy2021revisiting], the 'uniform' initialization was used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) ->Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CategoricalFeatureTokenizer(nn.Module):
    """Transforms categorical features to tokens (embeddings).

    See `FeatureTokenizer` for the illustration.

    The module efficiently implements a collection of `torch.nn.Embedding` (with
    optional biases).

    Examples:
        .. testcode::

            # the input must contain integers. For example, if the first feature can
            # take 3 distinct values, then its cardinality is 3 and the first column
            # must contain values from the range `[0, 1, 2]`.
            cardinalities = [3, 10]
            x = torch.tensor([
                [0, 5],
                [1, 7],
                [0, 2],
                [2, 4]
            ])
            n_objects, n_features = x.shape
            d_token = 3
            tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token, True, 'uniform')
            tokens = tokenizer(x)
            assert tokens.shape == (n_objects, n_features, d_token)
    """
    category_offsets: Tensor

    def __init__(self, cardinalities: List[int], d_token: int, bias: bool, initialization: str) ->None:
        """
        Args:
            cardinalities: the number of distinct values for each feature. For example,
                :code:`cardinalities=[3, 4]` describes two features: the first one can
                take values in the range :code:`[0, 1, 2]` and the second one can take
                values in the range :code:`[0, 1, 2, 3]`.
            d_token: the size of one token.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of feature value. The bias vectors are not shared
                between features.
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        assert d_token > 0, 'd_token must be positive'
        initialization_ = _TokenInitialization.from_str(initialization)
        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) ->Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


class FeatureTokenizer(nn.Module):
    """Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.

    The "Feature Tokenizer" module from [gorishniy2021revisiting]. The module transforms
    continuous and categorical features to tokens (embeddings).

    In the illustration below, the red module in the upper brackets represents
    `NumericalFeatureTokenizer` and the green module in the lower brackets represents
    `CategoricalFeatureTokenizer`.

    .. image:: ../images/feature_tokenizer.png
        :scale: 33%
        :alt: Feature Tokenizer

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities fr
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    def __init__(self, n_num_features: int, cat_cardinalities: List[int], d_token: int) ->None:
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, 'n_num_features must be non-negative'
        assert n_num_features or cat_cardinalities, 'at least one of n_num_features or cat_cardinalities must be positive/non-empty'
        self.initialization = 'uniform'
        self.num_tokenizer = NumericalFeatureTokenizer(n_features=n_num_features, d_token=d_token, bias=True, initialization=self.initialization) if n_num_features else None
        self.cat_tokenizer = CategoricalFeatureTokenizer(cat_cardinalities, d_token, True, self.initialization) if cat_cardinalities else None

    @property
    def n_tokens(self) ->int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) ->int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) ->Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0`
                was passed to the constructor.
            x_cat: categorical features (see `CategoricalFeatureTokenizer.forward` for
                details). Must be presented if non-empty :code:`cat_cardinalities` was
                passed to the constructor.
        Returns:
            tokens
        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        assert x_num is not None or x_cat is not None, 'At least one of x_num and x_cat must be presented'
        assert _all_or_none([self.num_tokenizer, x_num]), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none([self.cat_tokenizer, x_cat]), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) ->None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) ->Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) ->Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


ModuleType0 = Union[str, Callable[[], nn.Module]]


ModuleType = Union[str, Callable[..., nn.Module]]


def make_nn_module(module_type: ModuleType, *args) ->nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            cls = ReGLU
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError:
                raise ValueError(f'There is no such module as {module_type} in torch.nn')
        return cls(*args)
    else:
        return module_type(*args)


class MLP(nn.Module):
    """The MLP model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    **Input shape**: ``(n_objects, n_features)``.

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Head -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
        Head == Linear

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `MLP.Block`)
        head: (optional) the last layer (`MLP.Head`)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            model = MLP.make_baseline(
                d_in=x.shape[1],
                d_out=1,
                n_blocks=2,
                d_layer=3,
                dropout=0.1,
            )
            assert model(x).shape == (len(x), 1)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """


    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(self, *, d_in: int, d_out: int, bias: bool, activation: ModuleType0, dropout: float) ->None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) ->Tensor:
            return self.dropout(self.activation(self.linear(x)))
    Head = nn.Linear
    """The output module of `MLP`."""

    def __init__(self, *, d_in: int, d_out: Optional[int], d_layers: List[int], dropouts: Union[float, List[float]], activation: ModuleType0) ->None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need more
            control over the architecture.
        """
        if not d_layers:
            raise ValueError('d_layers must be non-empty')
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        if len(dropouts) != len(d_layers):
            raise ValueError('if dropouts is a list, then its size must be equal to the size of d_layers')
        super().__init__()
        self.blocks = nn.Sequential(*[MLP.Block(d_in=d_layers[i - 1] if i else d_in, d_out=d, bias=True, activation=activation, dropout=dropout) for i, (d, dropout) in enumerate(zip(d_layers, dropouts))])
        self.head = None if d_out is None else MLP.Head(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(cls, *, d_in: int, d_out: Optional[int], n_blocks: int, d_layer: int, dropout: float) ->'MLP':
        """A simplified constructor for building baseline MLPs.

        Features:

        * all linear layers have the same dimension
        * all dropout layers have the same dropout rate
        * all activations are ``ReLU``

        Args:
            d_in: the input size.
            d_out: the output size of `MLP.Head`. If `None`, then the output of MLP
                will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks.
            d_layer: the dimension of each linear layer.
            dropout: the dropout rate for all hidden layers.
        Returns:
            mlp
        """
        if n_blocks <= 0:
            raise ValueError('n_blocks must be positive')
        if not isinstance(dropout, float):
            raise ValueError('In this constructor, dropout must be float')
        return MLP(d_in=d_in, d_out=d_out, d_layers=[d_layer] * n_blocks if n_blocks else [], dropouts=dropout, activation='ReLU')

    def forward(self, x: Tensor) ->Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


class ResNet(nn.Module):
    """The ResNet model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    **Input shape**: ``(n_objects, n_features)``.

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `ResNet.Block`)
        head: (optional) the last module (`ResNet.Head`)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                d_out=1,
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """


    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(self, *, d_main: int, d_hidden: int, bias_first: bool, bias_second: bool, dropout_first: float, dropout_second: float, normalization: ModuleType0, activation: ModuleType0, skip_connection: bool) ->None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_nn_module(activation)
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
        """The output module of `ResNet`."""

        def __init__(self, *, d_in: int, d_out: int, bias: bool, normalization: ModuleType0, activation: ModuleType0) ->None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) ->Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(self, *, d_in: int, d_out: Optional[int], n_blocks: int, d_main: int, d_hidden: int, dropout_first: float, dropout_second: float, normalization: ModuleType0, activation: ModuleType0) ->None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need
            more control over the architecture.
        """
        super().__init__()
        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(*[ResNet.Block(d_main=d_main, d_hidden=d_hidden, bias_first=True, bias_second=True, dropout_first=dropout_first, dropout_second=dropout_second, normalization=normalization, activation=activation, skip_connection=True) for _ in range(n_blocks)])
        self.head = None if d_out is None else ResNet.Head(d_in=d_main, d_out=d_out, bias=True, normalization=normalization, activation=activation)

    @classmethod
    def make_baseline(cls, *, d_in: int, d_out: Optional[int], n_blocks: int, d_main: int, d_hidden: int, dropout_first: float, dropout_second: float) ->'ResNet':
        """A simplified constructor for building baseline ResNets.

        Features:

        * all activations are ``ReLU``
        * all normalizations are ``BatchNorm1d``

        Args:
            d_in: the input size
            d_out: the output size of `ResNet.Head`. If `None`, then the output of
                ResNet will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks
            d_main: the input size (or, equivalently, the output size) of each block
            d_hidden: the output size of the first linear layer in each block
            dropout_first: the dropout rate of the first dropout layer in each block.
            dropout_second: the dropout rate of the second dropout layer in each block.
                The value `0.0` is a good starting point.
        Return:
            resnet
        """
        return cls(d_in=d_in, d_out=d_out, n_blocks=n_blocks, d_main=d_main, d_hidden=d_hidden, dropout_first=dropout_first, dropout_second=dropout_second, normalization='BatchNorm1d', activation='ReLU')

    def forward(self, x: Tensor) ->Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue here: https://github.com/Yura52/rtdl/issues/new'


def all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' (fast) attention.

    To learn more about Multihead Attention, see [1]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [2] to speed up the module when the number of tokens is large.

    Examples:
        .. testcode::

            batch_size, n_tokens, d = 2, 3, 12
            n_heads = 6
            a = torch.randn(batch_size, n_tokens, d)
            b = torch.randn(batch_size, n_tokens * 2, d)
            module = MultiheadAttention(d_embedding=d, n_heads=n_heads, dropout=0.2)

            # self-attention
            x, attention_stats = module(a, a)
            assert x.shape == a.shape
            assert attention_stats['attention_probs'].shape == (batch_size, n_heads, n_tokens, n_tokens)
            assert attention_stats['attention_logits'].shape == (batch_size, n_heads, n_tokens, n_tokens)

            # cross-attention
            module(a, b)

            # Linformer (fast) self-attention
            module = MultiheadAttention(
                **kwargs,
                linformer_compression_ratio=0.25,
                linformer_sharing_policy='headwise',
                n_tokens=n_tokens,
            )
            module(a, a)

    References:
        * [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [2] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(self, *, d_embedding: int, n_heads: int, dropout: float, d_key: Optional[int]=None, d_value: Optional[int]=None, share_key_query_projection: bool=False, bias: bool=True, initialization: str='kaiming', linformer_compression_ratio: Optional[float]=None, linformer_sharing_policy: Optional[str]=None, n_tokens: Optional[int]=None) ->None:
        """
        Args:
            d_embedding: the input embedding size. Must be a multiple of ``n_heads```
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            d_key: the key projection size. Must be a multiple of ``n_heads``. If `None`,
                then ``d_embedding`` is used instead.
            d_value: the value (output) projection size. Must be a multiple of ``n_heads``.
                  If `None`, then ``d_embedding`` is used instead.
            share_key_query_projection: if `True`, then the projections for keys and
                queries are shared.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                ``['kaiming', 'xavier']``. ``'kaiming'`` is a reasonable default choice.
            linformer_compression_ratio: apply the technique from [1] to speed
                up the attention operation when the number of tokens is large. Can
                actually slow things down if the number of tokens is too low. This
                option can affect task metrics in an unpredictable way, use it with caution.
            linformer_sharing_policy: weight sharing policy for the Linformer compression.
                Must be `None` if ``linformer_compression_ratio`` is None. Otherwise,
                must either ``'headwise'`` or ``'key-value'`` (both policies are
                described in [1]). The first one leads to more parameters. The effect
                on the task performance depends on the task.
            n_tokens: the number of tokens (features). Must be provided if
                ``linformer_compression_ratio`` is not `None`.
        Raises:
            ValueError: if input arguments are not valid.
        """
        super().__init__()
        if d_key is None:
            d_key = d_embedding
        if d_value is None:
            d_value = d_embedding
        if n_heads > 1 and any(d % n_heads != 0 for d in [d_embedding, d_key, d_value]):
            raise ValueError('d_embedding, d_key and d_value must be multiples of n_heads')
        if initialization not in ['kaiming', 'xavier']:
            raise ValueError('initialization must be "kaiming" or "xavier"')
        if not all_or_none([n_tokens, linformer_compression_ratio, linformer_sharing_policy]):
            raise ValueError('The arguments n_tokens, linformer_compression_ratio and linformer_sharing_policy must be either all None or all not-None')
        linformer_sharing_policy_valid_values = [None, 'headwise', 'key-value']
        if linformer_sharing_policy not in linformer_sharing_policy_valid_values:
            raise ValueError(f'linformer_sharing_policy must be one of: {linformer_sharing_policy_valid_values}')
        self.W_k = nn.Linear(d_embedding, d_key, bias)
        self.W_q = None if share_key_query_projection else nn.Linear(d_embedding, d_key, bias)
        self.W_v = nn.Linear(d_embedding, d_value, bias)
        self.W_out = nn.Linear(d_value, d_value, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        self._initialization = initialization
        self.logits_handler = nn.Identity()
        self.probs_handler = nn.Identity()

        def make_linformer_compression():
            assert n_tokens and linformer_compression_ratio, INTERNAL_ERROR_MESSAGE
            return nn.Linear(n_tokens, int(n_tokens * linformer_compression_ratio), bias=False)
        if linformer_compression_ratio is not None:
            self.linformer_key_compression = make_linformer_compression()
            self.linformer_value_compression = None if linformer_sharing_policy == 'key-value' else make_linformer_compression()
        else:
            self.linformer_key_compression = None
            self.linformer_value_compression = None
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.W_q, self.W_k, self.W_v]:
            if m is None:
                continue
            if self._initialization == 'xavier' and (m is not self.W_v or self.W_out is not None):
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) ->Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)

    def forward(self, x_q: Tensor, x_kv: Tensor) ->Tensor:
        """Perform the forward pass.

        Args:
            x_q: query token embeddings. Shape: ``(batch_size, n_q_tokens, d_embedding)``.
            x_kv: key-value token embeddings. Shape: ``(batch_size, n_kv_tokens, d_embedding)``.
        Returns:
            (new_token_embeddings, attention_stats)
        """
        W_q = self.W_k if self.W_q is None else self.W_q
        q, k, v = W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, INTERNAL_ERROR_MESSAGE
        if self.linformer_key_compression is not None:
            k = self.linformer_key_compression(k.transpose(1, 2)).transpose(1, 2)
            value_compression = self.linformer_key_compression if self.linformer_value_compression is None else self.linformer_value_compression
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        n_k_tokens = k.shape[1]
        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        _attention_shape = batch_size, self.n_heads, n_k_tokens, n_q_tokens
        _ = self.logits_handler(attention_logits.reshape(*_attention_shape))
        _ = self.probs_handler(attention_probs.reshape(*_attention_shape))
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        if self.W_out is not None:
            x = self.W_out(x)
        return x


def _initialize_embeddings(weight: Tensor, d: int) ->None:
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


class CLSEmbedding(nn.Module):
    """Embedding of the [CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert].

    In the forward pass, the module appends [CLS]-embedding **to the beginning** of each
    item in the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d = 4
            cls_embedding = CLSEmbedding(d)
            x = torch.randn(batch_size, n_tokens, d)
            x = cls_embedding(x)
            assert x.shape == (batch_size, n_tokens + 1, d)
            assert (x[:, 0, :] == cls_embedding.weight.expand(len(x), -1)).all()

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_embedding: int) ->None:
        """
        Args:
            d_embedding: the size of the embedding
        """
        super().__init__()
        self.weight = Parameter(Tensor(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) ->None:
        _initialize_embeddings(self.weight, self.weight.shape[-1])

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != len(self.weight):
            raise ValueError('The last dimension of x must be equal to the embedding size')
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


def _is_reglu(module: ModuleType) ->bool:
    return isinstance(module, str) and module == 'ReGLU' or module is ReGLU


_INTERNAL_ERROR_MESSAGE = 'Internal error. Please, open an issue.'


def _is_glu_activation(activation: ModuleType):
    return isinstance(activation, str) and activation.endswith('GLU') or activation in [ReGLU, GEGLU]


class OneHotEncoder(nn.Module):
    """One hot encoding for categorical features.

    * **Input shape**: ``(batch_size, n_categorical_features)``
    * **Input data type**: ``integer``

    Examples::

        # three categorical features
        cardinalities = [3, 4, 5]
        ohe = OneHotEncoder(cardinalities)
        batch_size = 2
        x_cat = torch.stack([torch.randint(0, c, (batch_size,)) for c in cardinalities], 1)
        assert ohe(x_cat).shape == (batch_size, sum(cardinalities))
        assert (x_cat.sum(1) == len(cardinalities)).all()
    """
    cardinalities: Tensor

    def __init__(self, cardinalities: List[int]) ->None:
        """
        Args:
            cardinalities: ``cardinalities[i]`` is the number of unique values for the
                i-th categorical feature.
        """
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        encoded_columns = [F.one_hot(column, cardinality) for column, cardinality in zip(x.T, self.cardinalities)]
        return torch.cat(encoded_columns, 1)


class CatEmbeddings(nn.Module):
    """Embeddings for categorical features.

    * **Input shape**: ``(batch_size, n_categorical_features)``
    * **Input data type**: ``integer``

    To obtain embeddings for the i-th feature, use `get_embeddings`.

    Examples:
        .. testcode::

            # three categorical features
            cardinalities = [3, 4, 5]
            embedding_sizes = [6, 7, 8]
            m_cat = CatEmbeddings(list(zip(cardinalities, embedding_sizes)))
            batch_size = 2
            x_cat = torch.stack([
                torch.randint(0, c, (batch_size,))
                for c in cardinalities
            ], 1)
            assert m_cat(x_cat).shape == (batch_size, sum(embedding_sizes))
            i = 1
            assert m_cat.get_embeddings(i).shape == (cardinalities[i], embedding_sizes[i])

            d_embedding = 9
            m_cat = CatEmbeddings(cardinalities, d_embedding, stack=True)
            m_cat(x_cat).shape == (batch_size, len(cardinalities), d_embedding)
    """

    def __init__(self, _cardinalities_and_maybe_dimensions: Union[List[int], List[Tuple[int, int]]], d_embedding: Optional[int]=None, *, stack: bool=False, bias: bool=False) ->None:
        """
        Args:
            _cardinalities_and_maybe_dimensions: (positional-only argument!) either a
                list of cardinalities or a list of ``(cardinality, embedding_size)`` pairs.
            d_embedding: if not `None`, then (1) the first argument must be a list of
                cardinalities, (2) all the features will have the same embedding size,
                (3) ``stack=True`` becomes allowed.
            stack: if `True`, then ``d_embedding`` must be provided, and the module will
                produce outputs of the shape ``(batch_size, n_cat_features, d_embedding)``.
            bias: this argument is presented for historical reasons, just keep it `False`
                (when it is `True`, then for a each feature one more trainable vector is
                allocated, and it is added to the main embedding regardless of the
                feature values).
        """
        spec = _cardinalities_and_maybe_dimensions
        if not spec:
            raise ValueError('The first argument must be non-empty')
        if not (isinstance(spec[0], tuple) and d_embedding is None or isinstance(spec[0], int) and d_embedding is not None):
            raise ValueError('Invalid arguments. Valid combinations are: (1) the first argument is a list of (cardinality, embedding)-tuples AND d_embedding is None (2) the first argument is a list of cardinalities AND d_embedding is an integer')
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')
        super().__init__()
        spec_ = cast(List[Tuple[int, int]], spec if d_embedding is None else [(x, d_embedding) for x in spec])
        self._embeddings = nn.ModuleList()
        for cardinality, d_embedding in spec_:
            self._embeddings.append(nn.Embedding(cardinality, d_embedding))
        self._biases = nn.ParameterList(Parameter(Tensor(d)) for _, d in spec_) if bias else None
        self.stack = stack
        self.reset_parameters()

    def reset_parameters(self) ->None:
        for module in self._embeddings:
            _initialize_embeddings(module.weight, module.weight.shape[-1])
        if self._biases is not None:
            for x in self._biases:
                _initialize_embeddings(x, x.shape[-1])

    def get_embeddings(self, feature_idx: int) ->Tensor:
        """Get embeddings for the i-th feature.

        This method is needed because of the ``bias`` option (when it is set to `True`,
        the embeddings provided by the underlying `torch.nn.Embedding` are not "complete").

        Args:
            feature_idx: the feature index
        Return:
            embeddings for the feature ``feature_idx``
        Raises:
            ValueError: for invalid inputs
        """
        if feature_idx < 0 or feature_idx >= len(self._embeddings):
            raise ValueError(f'feature_idx must be in the range(0, {len(self._embeddings)}). The provided value is {feature_idx}.')
        x = self._embeddings[feature_idx].weight
        if self._biases is not None:
            x = x + self._biases[feature_idx][None]
        return x

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 2:
            raise ValueError('x must have two dimensions')
        if x.shape[1] != len(self._embeddings):
            raise ValueError(f'x has {x.shape[1]} columns, but it must have {len(self._embeddings)} columns.')
        out = []
        biases = itertools.repeat(None) if self._biases is None else self._biases
        assert isinstance(biases, collections.abc.Iterable)
        for module, bias, column in zip(self._embeddings, biases, x.T):
            x = module(column)
            if bias is not None:
                x = x + bias[None]
            out.append(x)
        return torch.stack(out, 1) if self.stack else torch.cat(out, 1)


class LinearEmbeddings(nn.Module):
    """Linear embeddings for numerical features.

    * **Input shape**: ``(batch_size, n_features)``
    * **Output shape**: ``(batch_size, n_features, d_embedding)``

    For each feature, a separate linear layer is allocated (``n_features`` layers in total).
    One such layer can be represented as ``torch.nn.Linear(1, d_embedding)``

    The embedding process is illustrated in the following pseudocode::

        layers = [nn.Linear(1, d_embedding) for _ in range(n_features)]
        x = torch.randn(batch_size, n_features)
        x_embeddings = torch.stack(
            [layers[i](x[:, i:i+1]) for i in range(n_features)],
            1,
        )

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = torch.randn(batch_size, n_features)
            m = LinearEmbeddings(n_features, d_embedding)
            assert m(x).shape == (batch_size, n_features, d_embedding)
    """

    def __init__(self, n_features: int, d_embedding: int, bias: bool=True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) ->None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


def compute_bin_indices(X, bin_edges):
    """Compute bin indices for the given feature values.

    The output of this function can be passed as input to:

        * `compute_bin_linear_ratios`
        * `piecewise_linear_encoding`
        * `rtdl.nn.PiecewiseLinearEncoder` (to the forward method)

    For ``X[i][j]``, compute the index ``k`` of the bin in ``bin_edges[j]`` such that
    ``bin_edges[j][k] <= X[i][j] < bin_edges[j][k + 1]``. If the value is less than the
    leftmost bin edge, ``0`` is returned. If the value is greater or equal than the rightmost
    bin edge, ``len(bin_edges[j]) - 1`` is returned.

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges for each features. Can be obtained from
            `compute_quantile_bin_edges` or `compute_decision_tree_bin_edges`.
    Return:
        bin indices: Shape: ``(n_objects, n_features)``.

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
    """
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)
    bin_edges = [as_tensor(x) for x in bin_edges]
    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape[1] != len(bin_edges):
        raise ValueError('The number of columns in X must be equal to the size of the `bin_edges` list')
    inf = torch.tensor([math.inf], dtype=X.dtype, device=X.device)
    bin_indices_list = [(torch.bucketize(column, torch.cat((-inf, column_bin_edges[1:-1], inf)), right=True) - 1) for column, column_bin_edges in zip(X.T, bin_edges)]
    bin_indices = torch.stack(bin_indices_list, 1)
    return bin_indices if is_torch else bin_indices.numpy()


def compute_bin_linear_ratios(X, bin_edges, bin_indices):
    """Compute the ratios for piecewise linear encoding as described in [1].

    The output of this function can be passed as input to:

        * `piecewise_linear_encoding`
        * `rtdl.nn.PiecewiseLinearEncoder` (to the forward method)

    For details, see the section "Piecewise linear encoding" in [1].

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges for each features. Size: ``n_features``. Can be obtained from
            `compute_quantile_bin_edges` or `compute_decision_tree_bin_edges`.
        bin_indices: the bin indices (can be computed via `compute_bin_indices`)
    Return:
        ratios

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
            bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
    """
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)
    bin_indices = as_tensor(bin_indices)
    bin_edges = [as_tensor(x) for x in bin_edges]
    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape != bin_indices.shape:
        raise ValueError('X and bin_indices must be of the same shape')
    if X.shape[1] != len(bin_edges):
        raise ValueError('The number of columns in X must be equal to the number of items in bin_edges')
    inf = torch.tensor([math.inf], dtype=X.dtype, device=X.device)
    values_list = []
    for c_i, (c_values, c_indices, c_bin_edges) in enumerate(zip(X.T, bin_indices.T, bin_edges)):
        if (c_indices + 1 >= len(c_bin_edges)).any():
            raise ValueError(f'The indices in indices[:, {c_i}] are not compatible with bin_edges[{c_i}]')
        effective_c_bin_edges = torch.cat((-inf, c_bin_edges[1:-1], inf))
        if (c_values < effective_c_bin_edges[c_indices]).any() or (c_values > effective_c_bin_edges[c_indices + 1]).any():
            raise ValueError('Values in X are not consistent with the provided bin indices and edges.')
        c_left_edges = c_bin_edges[c_indices]
        c_right_edges = c_bin_edges[c_indices + 1]
        values_list.append((c_values - c_left_edges) / (c_right_edges - c_left_edges))
    values = torch.stack(values_list, 1)
    return values if is_torch else values.numpy()


Number = TypeVar('Number', int, float)


def _LVR_encoding(indices, values, d_encoding: Union[int, List[int]], left: Number, right: Number, *, stack: bool):
    """Left-Value-Right encoding

    For one feature:
    f(x) = [left, left, ..., left, <value at the given index>, right, right, ... right]
    """
    is_torch = isinstance(values, Tensor)
    values = as_tensor(values)
    indices = as_tensor(indices)
    if type(left) is not type(right):
        raise ValueError('left and right must be of the same type')
    if type(left).__name__ not in str(values.dtype):
        raise ValueError('The `values` array has dtype incompatible with left and right')
    if values.ndim != 2:
        raise ValueError('values must have two dimensions')
    if values.shape != indices.shape:
        raise ValueError('values and indices must be of the same shape')
    if stack and not isinstance(d_encoding, int):
        raise ValueError('stack can be True only if d_encoding is an integer')
    if isinstance(d_encoding, int):
        if (indices >= d_encoding).any():
            raise ValueError('All indices must be less than d_encoding')
    else:
        if values.shape[1] != len(d_encoding):
            raise ValueError('If d_encoding is a list, then its size must be equal to `values.shape[1]`')
        if (indices >= torch.tensor(d_encoding)[None]).any():
            raise ValueError('All indices must be less than the corresponding d_encoding')
    dtype = values.dtype
    device = values.device
    n_objects, n_features = values.shape
    left_tensor = torch.tensor(left, dtype=dtype, device=device)
    right_tensor = torch.tensor(right, dtype=dtype, device=device)
    shared_d_encoding = d_encoding if isinstance(d_encoding, int) else d_encoding[0] if all(d == d_encoding[0] for d in d_encoding) else None
    if shared_d_encoding is None:
        encoding_list = []
        for c_values, c_indices, c_d_encoding in zip(values.T, indices.T, cast(List[int], d_encoding)):
            c_left_mask = torch.arange(c_d_encoding, device=device)[None] < c_indices[:, None]
            c_encoding = torch.where(c_left_mask, left_tensor, right_tensor)
            c_encoding[torch.arange(n_objects, device=device), c_indices] = c_values
            encoding_list.append(c_encoding)
        encoding = torch.cat(encoding_list, 1)
    else:
        left_mask = torch.arange(shared_d_encoding, device=device)[None, None] < indices[:, :, None]
        encoding = torch.where(left_mask, left_tensor, right_tensor)
        object_indices = torch.arange(n_objects, device=device)[:, None].repeat(1, n_features).reshape(-1)
        feature_indices = torch.arange(n_features, device=device).repeat(n_objects)
        encoding[object_indices, feature_indices, indices.flatten()] = values.flatten()
        if not stack:
            encoding = encoding.reshape(n_objects, -1)
    return encoding if is_torch else encoding.numpy()


def piecewise_linear_encoding(bin_edges, bin_indices, bin_ratios, d_encoding: Union[int, List[int]], *, stack: bool):
    """Construct piecewise linear encoding as described in [1].

    See `compute_piecewise_linear_encoding` for details.

    Note:
        To compute the encoding from the original feature valies, see
        `compute_piecewise_linear_encoding`.

    Args:
        bin_ratios: linear ratios (can be computed via `compute_bin_linear_ratios`).
            Shape: ``(n_objects, n_features)``.
        bin_indices: bin indices (can be computed via `compute_bin_indices`).
            Shape: ``(n_objects, n_features)``.
        d_encoding: the encoding sizes for all features (if an integer, it is used for
            all the features)
        stack: if `True`, then d_encoding must be an integer, and the output shape is
            ``(n_objects, n_features, d_encoding)``. Otherwise, the output shape is
            ``(n_objects, sum(d_encoding))``.
    Returns:
        encoded input
    Raises:
        ValueError: for invalid input

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
            bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
            bin_counts = [len(x) - 1 for x in bin_edges]
            X_ple = piecewise_linear_encoding(bin_edges, bin_indices, bin_ratios, bin_counts, stack=True)
    """
    is_torch = isinstance(bin_ratios, Tensor)
    bin_edges = torch.as_tensor(bin_ratios)
    bin_ratios = torch.as_tensor(bin_ratios)
    bin_indices = torch.as_tensor(bin_indices)
    if bin_ratios.ndim != 2:
        raise ValueError('bin_ratios must have two dimensions')
    if bin_ratios.shape != bin_indices.shape:
        raise ValueError('rations and bin_indices must be of the same shape')
    if isinstance(d_encoding, list) and bin_ratios.shape[1] != len(d_encoding):
        raise ValueError('the number of columns in bin_ratios must be equal to the size of d_encoding')
    message = 'bin_ratios do not satisfy requirements for the piecewise linear encoding. Use rtdl.data.compute_bin_linear_ratios to obtain valid values.'
    lower_bounds = torch.zeros_like(bin_ratios)
    is_first_bin = bin_indices == 0
    lower_bounds[is_first_bin] = -math.inf
    if (bin_ratios < lower_bounds).any():
        raise ValueError(message)
    del lower_bounds
    upper_bounds = torch.ones_like(bin_ratios)
    is_last_bin = bin_indices + 1 == as_tensor(list(map(len, bin_edges)))
    upper_bounds[is_last_bin] = math.inf
    if (bin_ratios > upper_bounds).any():
        raise ValueError(message)
    del upper_bounds
    encoding = _LVR_encoding(bin_indices, bin_ratios, d_encoding, 1.0, 0.0, stack=stack)
    return encoding if is_torch else encoding.numpy()


def compute_piecewise_linear_encoding(X, bin_edges, *, stack: bool):
    """Compute piecewise linear encoding as described in [1].

    .. image:: ../images/piecewise_linear_encoding_figure.png
        :scale: 25%
        :alt: obtaining bins from decision trees (figure)

    .. image:: ../images/piecewise_linear_encoding_equation.png
        :scale: 25%
        :alt: obtaining bins from decision trees (equation)

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges. Size: ``n_features``. Can be computed via
            `compute_quantile_bin_edges` and `compute_decision_tree_bin_edges`.
        stack: (let ``bin_counts = [len(x) - 1 for x in bin_edges]``) if `True`, then
            the output shape is ``(n_objects, n_features, max(bin_counts))``, otherwise
            the output shape is ``(n_objects, sum(bin_counts))``.
    Returns:
        encoded input
    Raises:
        ValueError: for invalid input

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            X_ple = compute_piecewise_linear_encoding(X, bin_edges, stack=False)
    """
    bin_indices = compute_bin_indices(X, bin_edges)
    bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
    bin_counts = [(len(x) - 1) for x in bin_edges]
    return piecewise_linear_encoding(bin_edges, bin_indices, bin_ratios, d_encoding=max(bin_counts) if stack else bin_counts, stack=stack)


class PiecewiseLinearEncoder(nn.Module):
    """Piecewise linear encoding for numerical features described in [1].

    See `rtdl.nn.compute_piecewise_linear_encoding` for details.

    Examples:
        .. testcode::

            train_size = 100
            n_features = 4
            X = torch.randn(train_size, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_counts = [len(x) - 1 for x in bin_edges]
            batch_size = 3
            x = X[:batch_size]

            m_ple = PiecewiseLinearEncoder(bin_edges, stack=False)
            assert m_ple(x).shape == (n_objects, sum(bin_counts))

            m_ple = PiecewiseLinearEncoder(bin_edges, stack=True)
            assert m_ple(x).shape == (n_objects, n_features, max(bin_counts))

            x_bin_indices = compute_bin_indices(x, bin_edges)
            x_bin_ratios = compute_bin_linear_ratios(x, x_bin_indices, bin_edges)
            m_ple = PiecewiseLinearEncoder(
                bin_edges, stack=True, expect_ratios_and_indices=True
            )
            assert m_ple(x_bin_ratios, x_bin_indices).shape == (
                n_objects, n_features, max(bin_counts)
            )

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    """
    bin_edges: Tensor
    d_encoding: Union[int, List[int]]

    def __init__(self, bin_edges: List[Tensor], *, stack: bool, expect_ratios_and_indices: bool=False) ->None:
        """
        Args:
            bin_edges: the bin edges. Can be obtained via
                `rtdl.data.compute_quantile_bin_edges` or `rtdl.data.compute_decision_tree_bin_edges`
            stack: the argument for `rtdl.data.compute_piecewise_linear_encoding`.
            expect_ratios_and_indices: if `True`, then the module will expect two arguments
                in its forward pass: bin ratios (produced by `rtdl.data.compute_bin_linear_ratios`)
                and indices (produced by `rtdl.data.compute_bin_indices`). Otherwise,
                the modules will expect one argument (raw numerical feature values).
                This option can be usefull if computing ratios and indices on-the-fly
                is a bottleneck and you want to use precomputed values.
        """
        super().__init__()
        self.register_buffer('bin_edges', torch.cat(bin_edges), False)
        self.edge_counts = [len(x) for x in bin_edges]
        self.stack = stack
        self.d_encoding = max(self.edge_counts) - 1 if self.stack else [(x - 1) for x in self.edge_counts]
        self.expect_ratios_and_indices = expect_ratios_and_indices

    def forward(self, x: Tensor, indices: Optional[Tensor]=None) ->Tensor:
        bin_edges = self.bin_edges.split(self.edge_counts)
        if indices is None:
            if self.expect_ratios_and_indices:
                raise ValueError('The module expects two arguments (ratios and indices), because the argument expect_ratios_and_indices was set to `True` in the constructor')
            return compute_piecewise_linear_encoding(x, bin_edges, stack=self.stack)
        else:
            if not self.expect_ratios_and_indices:
                raise ValueError('The module expects one arguments (raw numerical feature values), because the argument expect_ratios_and_indices was set to `False` in the constructor')
            ratios = x
            return piecewise_linear_encoding(bin_edges, ratios, indices, self.d_encoding, stack=self.stack)


class PeriodicEmbeddings(nn.Module):
    """Periodic embeddings for numerical features described in [1].

    Warning:
        For better performance and to avoid some failure modes, it is recommended
        to insert `NLinear` after this module (even if the next module after that is the
        first linear layer of the model's backbone). Alternatively, you can use
        `make_plr_embeddings`.

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = torch.randn(batch_size, n_features)
            sigma = 0.1  # THIS HYPERPARAMETER MUST BE TUNED CAREFULLY
            m = PeriodicEmbeddings(n_features, d_embedding, sigma)
            # for better performance: m = nn.Sequantial(PeriodicEmbeddings(...), NLinear(...))
            assert m(x).shape == (batch_size, n_features, d_embedding)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    """

    def __init__(self, n_features: int, d_embedding: int, sigma: float) ->None:
        """
        Args:
            n_features: the number of numerical features
            d_embedding: the embedding size, must be an even positive integer.
            sigma: the scale of the weight initialization.
                **This is a super important parameter which significantly affects performance**.
                Its optimal value can be dramatically different for different datasets, so
                no "default value" can exist for this parameter, and it must be tuned for
                each dataset. In the original paper, during hyperparameter tuning, this
                parameter was sampled from the distribution ``LogUniform[1e-2, 1e2]``.
                A similar grid would be ``[1e-2, 1e-1, 1e0, 1e1, 1e2]``.
                If possible, add more intermidiate values to this grid.
        """
        if d_embedding % 2:
            raise ValueError('d_embedding must be even')
        super().__init__()
        self.sigma = sigma
        self.coefficients = Parameter(Tensor(n_features, d_embedding // 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.coefficients, 0.0, self.sigma)

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = 2 * math.pi * self.coefficients[None] * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], -1)


class NLinear(nn.Module):
    """N linear layers for N token (feature) embeddings.

    To understand this module, let's revise `torch.nn.Linear`. When `torch.nn.Linear` is
    applied to three-dimensional inputs of the shape
    ``(batch_size, n_tokens, d_embedding)``, then the same linear transformation is
    applied to each of ``n_tokens`` token (feature) embeddings.

    By contrast, `NLinear` allocates one linear layer per token (``n_tokens`` layers in total).
    One such layer can be represented as ``torch.nn.Linear(d_in, d_out)``.
    So, the i-th linear transformation is applied to the i-th token embedding, as
    illustrated in the following pseudocode::

        layers = [nn.Linear(d_in, d_out) for _ in range(n_tokens)]
        x = torch.randn(batch_size, n_tokens, d_in)
        result = torch.stack([layers[i](x[:, i]) for i in range(n_tokens)], 1)

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding_in = 4
            d_embedding_out = 5
            x = torch.randn(batch_size, n_features, d_embedding_in)
            m = NLinear(n_features, d_embedding_in, d_embedding_out)
            assert m(x).shape == (batch_size, n_features, d_embedding_out)
    """

    def __init__(self, n_tokens: int, d_in: int, d_out: int, bias: bool=True) ->None:
        """
        Args:
            n_tokens: the number of tokens (features)
            d_in: the input dimension
            d_out: the output dimension
            bias: indicates if the underlying linear layers have biases
        """
        super().__init__()
        self.weight = Parameter(Tensor(n_tokens, d_in, d_out))
        self.bias = Parameter(Tensor(n_tokens, d_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d_in = self.weight.shape[1]
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) ->Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions (batch_size, n_tokens, d_embedding)')
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class _Lambda(nn.Module):

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


MainModule = TypeVar('MainModule', bound=nn.Module)


class SimpleModel(nn.Module, Generic[MainModule]):
    """
    Warning:
        Do not instantiate this class directly, use `make_simple_model` instead.
    """

    def __init__(self, input: Dict[str, Union[Tuple, List[Tuple]]], main: MainModule, main_input_ndim: int, output: Optional[Dict[str, nn.Module]]=None) ->None:
        assert main_input_ndim in (2, 3), INTERNAL_ERROR_MESSAGE
        super().__init__()
        input_modules = {}
        input_args: Dict[str, Union[Tuple[str, ...], List[Tuple[str, ...]]]] = {}
        for name, spec in input.items():
            if isinstance(spec, list):
                input_modules[name] = nn.ModuleList()
                input_args[name] = []
                for module, *args in spec:
                    input_modules[name].append(module)
                    assert isinstance(input_args[name], list)
                    cast(list, input_args[name]).append(tuple(args))
            else:
                input_modules[name] = spec[0]
                input_args[name] = spec[1:]
        self.input = nn.ModuleDict(input_modules)
        self._input_args = input_args
        self.main = main
        self._main_input_ndim = main_input_ndim
        self.output = None if output is None else nn.ModuleDict(output)

    def _get_forward_kwarg_names(self) ->Set[str]:
        kwargs: Set[str] = set()
        for args in self._input_args.values():
            if isinstance(args, tuple):
                args = [args]
            kwargs.update(itertools.chain.from_iterable(args))
        return kwargs

    def usage(self) ->str:
        return f"forward(*, {', '.join(self._get_forward_kwarg_names())})"

    def forward(self, **kwargs) ->Any:
        required_kwarg_names = self._get_forward_kwarg_names()
        if required_kwarg_names != set(kwargs):
            raise TypeError(f'The expected arguments are: {required_kwarg_names}. The provided arguments are: {set(kwargs)}.')
        input_results = []
        for name in self.input:
            module = self.input[name]
            input_args = self._input_args[name]
            if isinstance(module, nn.ModuleList):
                assert isinstance(input_args, list), INTERNAL_ERROR_MESSAGE
                outputs = []
                for i_mod, (mod, args) in enumerate(zip(module, input_args)):
                    out = mod(**{arg: kwargs[arg] for arg in args})
                    if out.ndim != 3:
                        raise RuntimeError(f'The output of the input module {name}[{i_mod}] has {out.ndim} dimensions, but when there are multiple input modules under the same name, they must output three-dimensional tensors')
                    outputs.append(out)
                    if outputs[-1].shape[:2] != outputs[0].shape[:2]:
                        raise RuntimeError(f'The input modules {name}[{0}] and {name}[{i_mod}] produced tensors with different two dimensions: {outputs[-1].shape} VS  {outputs[0].shape}')
                input_results.append(torch.cat(outputs, 2))
            else:
                assert isinstance(input_args, tuple), INTERNAL_ERROR_MESSAGE
                output = module(**{arg: kwargs[arg] for arg in input_args})
                if output.ndim < 2 or output.ndim > 3:
                    raise RuntimeError(f'The input module {name} produced tensor with {output.ndim} dimensions, but it must be 2 or 3.')
                input_results.append(output)
        assert self._main_input_ndim in (2, 3), INTERNAL_ERROR_MESSAGE
        x = torch.cat([t.flatten(self._main_input_ndim - 1, -1) for t in input_results], dim=1)
        x = self.main(x)
        return x if self.output is None else {k: v(x) for k, v in self.output.items()}


class Model(nn.Module):

    def __init__(self, cat_input_module: nn.Module, model: nn.Module):
        super().__init__()
        self.cat_input_module = cat_input_module
        self.model = model

    def forward(self, x_num, x_cat):
        return self.model(torch.cat([x_num, self.cat_input_module(x_cat).flatten(1, -1)], dim=1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CLSEmbedding,
     lambda: ([], {'d_embedding': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearEmbeddings,
     lambda: ([], {'n_features': 4, 'd_embedding': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'d_in': 4, 'd_out': 4, 'd_layers': [4, 4], 'dropouts': 0.5, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Model,
     lambda: ([], {'cat_input_module': _mock_layer(), 'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'d_embedding': 4, 'n_heads': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (NLinear,
     lambda: ([], {'n_tokens': 4, 'd_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PeriodicEmbeddings,
     lambda: ([], {'n_features': 4, 'd_embedding': 4, 'sigma': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ReGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'d_in': 4, 'd_out': 4, 'n_blocks': 4, 'd_main': 4, 'd_hidden': 4, 'dropout_first': 0.5, 'dropout_second': 0.5, 'normalization': _mock_layer, 'activation': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Lambda,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_Yura52_rtdl(_paritybench_base):
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

