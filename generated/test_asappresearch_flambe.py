import sys
_module = sys.modules[__name__]
del sys
conf = _module
flambe = _module
cluster = _module
aws = _module
const = _module
errors = _module
instance = _module
instance = _module
ssh = _module
utils = _module
compile = _module
component = _module
downloader = _module
extensions = _module
registrable = _module
serialization = _module
utils = _module
dataset = _module
tabular = _module
experiment = _module
options = _module
progress = _module
tune_adapter = _module
utils = _module
webapp = _module
app = _module
wording = _module
export = _module
builder = _module
exporter = _module
field = _module
bow = _module
field = _module
label = _module
text = _module
learn = _module
distillation = _module
eval = _module
script = _module
train = _module
utils = _module
logging = _module
datatypes = _module
handler = _module
contextual_file = _module
tensorboard = _module
utils = _module
logo = _module
metric = _module
dev = _module
accuracy = _module
auc = _module
binary = _module
bpc = _module
perplexity = _module
recall = _module
loss = _module
cross_entropy = _module
nll_loss = _module
metric = _module
model = _module
logistic_regression = _module
nlp = _module
classification = _module
datasets = _module
model = _module
fewshot = _module
model = _module
language_modeling = _module
fields = _module
model = _module
sampler = _module
transformers = _module
field = _module
model = _module
nn = _module
cnn = _module
distance = _module
cosine = _module
distance = _module
euclidean = _module
hyperbolic = _module
embedding = _module
mlp = _module
module = _module
mos = _module
pooling = _module
rnn = _module
sequential = _module
softmax = _module
transformer = _module
transformer_sru = _module
optim = _module
linear = _module
noam = _module
radam = _module
scheduler = _module
runnable = _module
cluster_runnable = _module
context = _module
environment = _module
error = _module
runner = _module
report_site_run = _module
run = _module
base = _module
episodic = _module
sampler = _module
tokenizer = _module
char = _module
word = _module
config = _module
version = _module
vision = _module
datasets = _module
model = _module
deploy_documentation = _module
setup = _module
tests = _module
conftest = _module
flambe_inference = _module
obj = _module
flambe_runnable = _module
flambe_script = _module
test_builder = _module
test_examples = _module
test_resources_experiment = _module
unit = _module
test_aws = _module
test_cluster = _module
test_instance = _module
test_compilable = _module
test_downloader = _module
test_extensions = _module
test_registrable = _module
test_serialization = _module
test_utils = _module
test_tabular = _module
test_experiment = _module
test_experiment_preprocess = _module
test_options = _module
test_utils = _module
test_label_field = _module
test_text_field = _module
test_trainer = _module
metrics = _module
test_dev = _module
test_loss = _module
test_logistic_regression = _module
test_tc_datasets = _module
test_lm_datasets = _module
test_cnn = _module
test_mlp = _module
test_pooling = _module
test_rnn = _module
test_scheduler = _module
remote = _module
test_runnable = _module
test_args = _module
test_base = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import logging


import uuid


from typing import Optional


from typing import Type


from typing import Generator


from typing import TypeVar


from typing import List


from typing import Dict


from types import TracebackType


import inspect


from warnings import warn


from typing import Any


from typing import Mapping


from typing import Union


from typing import MutableMapping


from typing import Callable


from typing import Set


from typing import Tuple


from typing import Sequence


from functools import WRAPPER_ASSIGNMENTS


from collections import OrderedDict


import copy


import torch


import re


from typing import Iterable


from abc import abstractmethod


from abc import ABC


from collections import defaultdict


import functools


from typing import NamedTuple


from collections import abc


from collections import OrderedDict as odict


import numpy as np


from itertools import chain


import warnings


import torch.nn.functional as F


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import math


from typing import Iterator


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn.utils.clip_grad import clip_grad_norm_


from torch.nn.utils.clip_grad import clip_grad_value_


import numpy


from types import SimpleNamespace


import random


import sklearn.metrics


from torch import Tensor


from torch.nn import Sigmoid


import torch.nn as nn


from torch.utils.data import DataLoader


from torch import nn


from typing import cast


from functools import partial


from torch.nn.utils.rnn import pad_sequence


from sklearn.model_selection import train_test_split


from torch.nn import NLLLoss


from torch.optim import Adam


from numpy import isclose


import torch.testing


from torch import allclose


class MLPEncoder(Module):
    """Implements a multi layer feed forward network.

    This module can be used to create output layers, or
    more complex multi-layer feed forward networks.

    Attributes
    ----------
    seq: nn.Sequential
        the sequence of layers and activations

    """

    def __init__(self, input_size: int, output_size: int, n_layers: int=1, dropout: float=0.0, output_activation: Optional[nn.Module]=None, hidden_size: Optional[int]=None, hidden_activation: Optional[nn.Module]=None) ->None:
        """Initializes the FullyConnected object.

        Parameters
        ----------
        input_size: int
            Input_dimension
        output_size: int
            Output dimension
        n_layers: int, optional
            Number of layers in the network, defaults to 1
        dropout: float, optional
            Dropout to be used before each MLP layer.
            Only used if n_layers > 1.
        output_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None
        hidden_size: int, optional
            Hidden dimension, used only if n_layers > 1.
            If not given, defaults to the input_size
        hidden_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None

        """
        super().__init__()
        layers = []
        if n_layers == 1 or hidden_size is None:
            hidden_size = input_size
        if n_layers > 1:
            layers.append(nn.Linear(input_size, hidden_size))
            if hidden_activation is not None:
                layers.append(hidden_activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(1, n_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if hidden_activation is not None:
                    layers.append(hidden_activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) ->torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model of shape (batch_size, input_size)

        Returns
        -------
        output: torch.Tensor
            output of the model of shape (batch_size, output_size)

        """
        return self.seq(data)


class LogisticRegression(Module):
    """
    Logistic regression model given an input vector v
    the forward calculation is sigmoid(Wv+b), where
    W is a weight vector and b a bias term. The result
    is then passed to a sigmoid function, which maps it
    as a real number in [0,1]. This is typically interpreted
    in classification settings as the probability of belonging
    to a given class.

    Attributes
    ----------
    input_size : int
        Dimension (number of features) of the input vector.
    """

    def __init__(self, input_size: int) ->None:
        """
        Initialize the Logistic Regression Model.
        Parameters
        ----------
        input_size: int
            The dimension of the input vector
        """
        super().__init__()
        self.encoder = MLPEncoder(input_size, output_size=1, n_layers=1, output_activation=Sigmoid())

    def forward(self, data: Tensor, target: Optional[Tensor]=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass that encodes data
        Parameters
        ----------
        data : Tensor
            input data to encode
        target: Optional[Tensor]
            target value, will be casted to a float tensor.
        """
        encoding = self.encoder(data)
        return (encoding, target.float()) if target is not None else encoding


class Embedder(Module):
    """Implements an Embedder module.

    An Embedder takes as input a sequence of index tokens,
    and computes the corresponding embedded representations, and
    padding mask. The encoder may be initialized using a pretrained
    embedding matrix.

    Attributes
    ----------
    embeddings: Module
        The embedding module
    encoder: Module
        The sub-encoder that this object is wrapping
    pooling: Module
        An optional pooling module
    drop: nn.Dropout
        The dropout layer

    """

    def __init__(self, embedding: Module, encoder: Module, pooling: Optional[Module]=None, embedding_dropout: float=0, padding_idx: Optional[int]=0, return_mask: bool=False) ->None:
        """Initializes the TextEncoder module.

        Extra arguments are passed to the nn.Embedding module.

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer
        encoder: Module
            The encoder
        pooling: Module, optional
            An optioonal pooling module, takes a sequence of Tensor and
            reduces them to a single Tensor.
        embedding_dropout: float, optional
            Amount of dropout between the embeddings and the encoder
        padding_idx: int, optional
            Passed the nn.Embedding object. See pytorch documentation.
        return_mask: bool
            If enabled, the forward call returns a tuple of
            (encoding, mask)

        """
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(embedding_dropout)
        self.encoder = encoder
        self.pooling = pooling
        self.padding_idx = padding_idx
        self.return_mask = return_mask

    def forward(self, data: Tensor) ->Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tuple[Tensor, Tensor], Tensor]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor of shape [S x B]

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor],
                Tuple[Tuple[Tensor, Tensor], Tensor]
            The encoded output, as a float tensor. May return a state
            if the encoder is an RNN and no pooling is provided.
            May also return a tuple if `return_mask` was passed in as
            a constructor argument.

        """
        embedded = self.embedding(data)
        embedded = self.dropout(embedded)
        padding_mask: Optional[Tensor]
        if self.padding_idx is not None:
            padding_mask = data != self.padding_idx
            encoding = self.encoder(embedded, padding_mask=padding_mask)
        else:
            padding_mask = None
            encoding = self.encoder(embedded)
        if self.pooling is not None:
            encoding = encoding[0] if isinstance(encoding, tuple) else encoding
            encoding = self.pooling(encoding, padding_mask)
        if self.return_mask:
            return encoding, padding_mask
        else:
            return encoding


class TextClassifier(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    embedder: Embedder
        The embedder layer
    output_layer : Module
        The output layer, yields a probability distribution over targets
    drop: nn.Dropout
        the dropout layer
    loss: Metric
        the loss function to optimize the model with
    metric: Metric
        the dev metric to evaluate the model on

    """

    def __init__(self, embedder: Embedder, output_layer: Module, dropout: float=0) ->None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer
        output_layer : Module
            The output layer, yields a probability distribution
        dropout : float, optional
            Amount of dropout to include between layers (defaults to 0)

        """
        super().__init__()
        self.embedder = embedder
        self.output_layer = output_layer
        self.drop = nn.Dropout(dropout)

    def forward(self, data: Tensor, target: Optional[Tensor]=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data
        target: Tensor, optional
            The input targets, optional

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]
            The output predictions, and optionally the targets

        """
        outputs = self.embedder(data)
        if isinstance(outputs, tuple):
            encoding = outputs[0]
        else:
            encoding = outputs
        pred = self.output_layer(self.drop(encoding))
        return (pred, target) if target is not None else pred


class DistanceModule(Module):
    """Implement a DistanceModule object.

    """

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        raise NotImplementedError


class CosineDistance(DistanceModule):
    """Implement a CosineDistance object.

    """

    def __init__(self, eps: float=1e-08) ->None:
        """Initialize the CosineDistance module.

        Parameters
        ----------
        eps : float, optional
            Used for numerical stability

        """
        super().__init__()
        self.eps = eps

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Returns the cosine distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        w1 = mat_1.norm(p=2, dim=1, keepdim=True)
        w2 = mat_2.norm(p=2, dim=1, keepdim=True)
        return 1 - torch.mm(mat_1, mat_2.t()) / (w1 * w2.t()).clamp(min=self.eps)


class EuclideanDistance(DistanceModule):
    """Implement a EuclideanDistance object."""

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        dist = [torch.sum((mat_1 - mat_2[i]) ** 2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(dist, dim=1)
        return dist


EPSILON = 1e-05


def arccosh(x):
    """Compute the arcosh, numerically stable."""
    x = torch.clamp(x, min=1 + EPSILON)
    a = torch.log(x)
    b = torch.log1p(torch.sqrt(x * x - 1) / x)
    return a + b


class HyperbolicDistance(DistanceModule):
    """Implement a HyperbolicDistance object.

    """

    def forward(self, mat_1: Tensor, mat_2: Tensor) ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        mat_1_x_0 = torch.sqrt(1 + mat_1.pow(2).sum(dim=1, keepdim=True))
        mat_2_x_0 = torch.sqrt(1 + mat_2.pow(2).sum(dim=1, keepdim=True))
        left = mat_1_x_0.mm(mat_2_x_0.t())
        right = mat_1[:, 1:].mm(mat_2[:, 1:].t())
        return arccosh(left - right).pow(2)


def get_distance_module(metric: str) ->DistanceModule:
    """Get the distance module from a string alias.

    Currently available:
    . `euclidean`
    . `cosine`
    . `hyperbolic`

    Parameters
    ----------
    metric : str
        The distance metric to use

    Raises
    ------
    ValueError
        Unvalid distance string alias provided

    Returns
    -------
    DistanceModule
        The instantiated distance module

    """
    if metric == 'euclidean':
        module = EuclideanDistance()
    elif metric == 'cosine':
        module = CosineDistance()
    elif metric == 'hyperbolic':
        module = HyperbolicDistance()
    else:
        raise ValueError(f'Unknown distance alias: {metric}')
    return module


class MeanModule(Module):
    """Implement a MeanModule object.

    """

    def __init__(self, detach_mean: bool=False) ->None:
        """Initilaize the MeanModule.

        Parameters
        ----------
        detach_mean : bool, optional
            Set to detach the mean computation, this is useful when the
            mean computation does not admit a closed form.

        """
        super().__init__()
        self.detach_mean = detach_mean

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        raise NotImplementedError


class CosineMean(MeanModule):
    """Implement a CosineMean object.

    """

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        data = data / data.norm(dim=1, keepdim=True)
        return data.mean(0)


class EuclideanMean(MeanModule):
    """Implement a EuclideanMean object."""

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        return data.mean(0)


def mdot(x, y):
    """Compute the inner product."""
    m = x.new_ones(1, x.size(1))
    m[0, 0] = -1
    return torch.sum(m * x * y, 1, keepdim=True)


def norm(x):
    """Compute the norm"""
    n = torch.sqrt(torch.abs(mdot(x, x)))
    return n


def exp_map(x, y):
    """Perform the exp step."""
    n = torch.clamp(norm(y), min=EPSILON)
    return torch.cosh(n) * x + torch.sinh(n) / n * y


def dist(x, y):
    """Get the hyperbolic distance between x and y."""
    return arccosh(-mdot(x, y))


def log_map(x, y):
    """Perform the log step."""
    d = dist(x, y)
    return d / torch.sinh(d) * (y - torch.cosh(d) * x)


def project(x):
    """Project onto the hyeprboloid embedded in in n+1 dimensions."""
    return torch.cat([torch.sqrt(1.0 + torch.sum(x * x, 1, keepdim=True)), x], 1)


class HyperbolicMean(MeanModule):
    """Compute the mean point in the hyperboloid model."""

    def forward(self, data: Tensor) ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        n_iter = 5 if self.training else 100
        projected = project(data)
        mean = torch.mean(projected, 0, keepdim=True)
        mean = mean / norm(mean)
        r = 0.01
        for i in range(n_iter):
            g = -2 * torch.mean(log_map(mean, projected), 0, keepdim=True)
            mean = exp_map(mean, -r * g)
            mean = mean / norm(mean)
        return mean.squeeze()[1:]


def get_mean_module(metric: str) ->MeanModule:
    """Get the mean module from a string alias.

    Currently available:
    . `euclidean`
    . `cosine`
    . `hyperbolic`

    Parameters
    ----------
    metric : str
        The distance metric to use

    Raises
    ------
    ValueError
        Unvalid distance string alias provided

    Returns
    -------
    DistanceModule
        The instantiated distance module

    """
    if metric == 'euclidean':
        module = EuclideanMean()
    elif metric == 'cosine':
        module = CosineMean()
    elif metric == 'hyperbolic':
        module = HyperbolicMean()
    else:
        raise ValueError(f'Unknown distance alias: {metric}')
    return module


class PrototypicalTextClassifier(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    encoder: Module
        the encoder object
    decoder: Decoder
        the decoder layer
    drop: nn.Dropout
        the dropout layer
    loss: Metric
        the loss function to optimize the model with
    metric: Metric
        the dev metric to evaluate the model on

    """

    def __init__(self, embedder: Embedder, distance: str='euclidean', detach_mean: bool=False) ->None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer

        """
        super().__init__()
        self.embedder = embedder
        self.distance_module = get_distance_module(distance)
        self.mean_module = get_mean_module(distance)
        self.detach_mean = detach_mean

    def compute_prototypes(self, support: Tensor, label: Tensor) ->Tensor:
        """Set the current prototypes used for classification.

        Parameters
        ----------
        data : torch.Tensor
            Input encodings
        label : torch.Tensor
            Corresponding labels

        """
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])
        means = []
        n_means = len(means_dict)
        for i in range(n_means):
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = self.mean_module(supports).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)
        prototypes = torch.stack(means, dim=0)
        return prototypes

    def forward(self, query: Tensor, query_label: Optional[Tensor]=None, support: Optional[Tensor]=None, support_label: Optional[Tensor]=None, prototypes: Optional[Tensor]=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            The output predictions

        """
        query_encoding = self.embedder(query)
        if isinstance(query_encoding, tuple):
            query_encoding = query_encoding[0]
        if prototypes is not None:
            prototypes = prototypes
        elif support is not None and support_label is not None:
            if self.detach_mean:
                support = support.detach()
                support_label = support_label.detach()
            support_encoding = self.embedder(support)
            if isinstance(support_encoding, tuple):
                support_encoding = support_encoding[0]
            prototypes = self.compute_prototypes(support_encoding, support_label)
        else:
            raise ValueError('No prototypes set or provided')
        dist = self.distance_module(query_encoding, prototypes)
        if query_label is not None:
            return -dist, query_label
        else:
            return -dist


class LanguageModel(Module):
    """Implement an LanguageModel model for sequential classification.

    This model can be used to language modeling, as well as other
    sequential classification tasks. The full sequence predictions
    are produced by the model, effectively making the number of
    examples the batch size multiplied by the sequence length.

    """

    def __init__(self, embedder: Embedder, output_layer: Module, dropout: float=0, pad_index: int=0, tie_weights: bool=False, tie_weight_attr: str='embedding') ->None:
        """Initialize the LanguageModel model.

        Parameters
        ----------
        embedder: Embedder
            The embedder layer
        output_layer : Decoder
            Output layer to use
        dropout : float, optional
            Amount of droput between the encoder and decoder,
            defaults to 0.
        pad_index: int, optional
            Index used for padding, defaults to 0
        tie_weights : bool, optional
            If true, the input and output layers share the same weights
        tie_weight_attr: str, optional
            The attribute to call on the embedder to get the weight
            to tie. Only used if tie_weights is ``True``. Defaults
            to ``embedding``. Multiple attributes can also be called
            by adding another dot: ``embeddings.word_embedding``.

        """
        super().__init__()
        self.embedder = embedder
        self.output_layer = output_layer
        self.drop = nn.Dropout(dropout)
        self.pad_index = pad_index
        self.tie_weights = tie_weights
        if tie_weights:
            module = self.embedder
            for attr in tie_weight_attr.split('.'):
                module = getattr(module, attr)
            self.output_layer.weight = module.weight

    def forward(self, data: Tensor, target: Optional[Tensor]=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            The output predictions of shape seq_len x batch_size x n_out

        """
        outputs = self.embedder(data)
        if isinstance(outputs, tuple):
            encoding = outputs[0]
        else:
            encoding = outputs
        if target is not None:
            mask = (target != self.pad_index).float()
            flat_mask = mask.view(-1).bool()
            flat_encodings = encoding.view(-1, encoding.size(2))[flat_mask]
            flat_targets = target.contiguous().view(-1)[flat_mask]
            flat_pred = self.output_layer(self.drop(flat_encodings))
            return flat_pred, flat_targets
        else:
            pred = self.output_layer(self.drop(encoding))
            return pred


class PretrainedTransformerEmbedder(Module):
    """Embedder intergation of the transformers library.

    Instantiate this object using any alias available in the
    `transformers` library. More information can be found here:

    https://huggingface.co/transformers/

    """

    def __init__(self, alias: str, cache_dir: Optional[str]=None, padding_idx: Optional[int]=None, pool: bool=False, **kwargs) ->None:
        """Initialize from a pretrained model.

        Parameters
        ----------
        alias: str
            Alias of a pretrained model.
        cache_dir: str, optional
            A directory where to cache the downloaded vocabularies.
        padding_idx: int, optional
            The padding index used to compute the attention mask.
        pool: optional, optional
            Whether to return the pooled output or the full sequence
            encoding. Default ``False``.

        """
        super().__init__()
        if 'gpt2' in alias and pool:
            raise ValueError('GPT2 does not support pooling.')
        embedder = AutoModel.from_pretrained(alias, cache_dir=cache_dir, **kwargs)
        self.config = embedder.config
        self._embedder = embedder
        self.padding_idx = padding_idx
        self.pool = pool

    def forward(self, data: torch.Tensor, token_type_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Perform a forward pass through the network.

        If pool was provided, will only return the pooled output
        of shape [B x H]. Otherwise, returns the full sequence encoding
        of shape [S x B x H].

        Parameters
        ----------
        data : torch.Tensor
            The input data of shape [B x S]
        token_type_ids : Optional[torch.Tensor], optional
            Segment token indices to indicate first and second portions
            of the inputs. Indices are selected in ``[0, 1]``: ``0``
            corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token. Has shape [B x S]
        attention_mask : Optional[torch.Tensor], optional
            FloatTensor of shape [B x S]. Masked values should
            be 0 for padding tokens, 1 otherwise.
        position_ids : Optional[torch.Tensor], optional
            Indices of positions of each input sequence tokens
            in the position embedding. Defaults to the order given
            in the input. Has shape [B x S].
        head_mask : Optional[torch.Tensor], optional
            Mask to nullify selected heads of the self-attention
            modules. Should be 0 for heads to mask, 1 otherwise.
            Has shape [num_layers x num_heads]

        Returns
        -------
        torch.Tensor
            If pool is True, returns a tneosr of shape [B x H],
            else returns an encoding for each token in the sequence
            of shape [B x S x H].

        """
        if attention_mask is None and self.padding_idx is not None:
            attention_mask = (data != self.padding_idx).float()
        outputs = self._embedder(data, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)
        output = outputs[0] if not self.pool else outputs[1]
        return output

    def __getattr__(self, name: str) ->Any:
        """Override getattr to inspect config.

        Parameters
        ----------
        name : str
            The attribute to fetch

        Returns
        -------
        Any
            The attribute

        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            config = self.__dict__['config']
            if hasattr(config, name):
                return getattr(config, name)
            else:
                raise e


def conv_block(conv_mod: nn.Module, activation: nn.Module, pooling: nn.Module, dropout: float, batch_norm: Optional[nn.Module]=None) ->nn.Module:
    """Return a convolutional block.

    """
    mods = [conv_mod]
    if pooling:
        mods.append(pooling)
    if batch_norm is None:
        mods.append(batch_norm)
    mods.append(activation)
    mods.append(nn.Dropout(dropout))
    return nn.Sequential(*mods)


class CNNEncoder(Module):
    """Implements a multi-layer n-dimensional CNN.

    This module can be used to create multi-layer CNN models.

    Attributes
    ----------
    cnn: nn.Module
        The cnn submodule

    """

    def __init__(self, input_channels: int, channels: List[int], conv_dim: int=2, kernel_size: Union[int, List[Union[Tuple[int, ...], int]]]=3, activation: nn.Module=None, pooling: nn.Module=None, dropout: float=0, batch_norm: bool=True, stride: int=1, padding: int=0) ->None:
        """Initializes the CNNEncoder object.

        Parameters
        ----------
        input_channels: int
            The input's channels. For example, 3 for RGB images.
        channels: List[int]
            A list to specify the channels of the convolutional layers.
            The length of this list will be the amount of convolutions
            in the encoder.
        conv_dim: int, optional
            The dimension of the convolutions. Can be 1, 2 or 3.
            Defaults to 2.
        kernel_size: Union[int, List[Union[Tuple[int], int]]], optional
            The kernel size for the convolutions. This could be an int
            (the same kernel size for all convolutions and dimensions),
            or a List where for each convolution you can specify an int
            or a tuple (for different sizes per dimension, in which case
            the length of the tuple must match the dimension of the
            convolution). Defaults to 3.
        activation: nn.Module, optional
           The activation function to use in all layers.
           Defaults to nn.ReLU
        pooling: nn.Module, optional
            The pooling function to use after all layers.
            Defaults to None
        dropout: float, optional
            Amount of dropout to use between CNN layers, defaults to 0
        batch_norm: bool, optional
            Wether to user Batch Normalization or not. Defaults to True
        stride: int, optional
            The stride to use when doing convolutions. Defaults to 1
        padding: int, optional
            The padding to use when doing convolutions. Defaults to 0

        Raises
        ------
        ValueError
            The conv_dim should be 1, 2, 3.

        """
        super().__init__()
        dim2mod = {(1): (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d), (2): (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d), (3): (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d)}
        if conv_dim not in dim2mod:
            raise ValueError(f'Invalid conv_dim value {conv_dim}. Values 1, 2, 3 supported')
        if isinstance(kernel_size, List) and len(kernel_size) != len(channels):
            raise ValueError('Kernel size list should have same length as channels list')
        conv, bn, pool = dim2mod[conv_dim]
        activation = activation or nn.ReLU()
        layers = []
        prev_c = input_channels
        for i, c in enumerate(channels):
            k: Union[int, Tuple]
            if isinstance(kernel_size, int):
                k = kernel_size
            else:
                k = kernel_size[i]
                if not isinstance(k, int) and len(k) != conv_dim:
                    raise ValueError('Kernel size tuple should have same length as conv_dim')
            layer = conv_block(conv(prev_c, c, k, stride, padding), activation, pooling, dropout, bn(c))
            layers.append(layer)
            prev_c = c
        self.cnn = nn.Sequential(*layers)

    def forward(self, data: Tensor) ->Union[Tensor, Tuple[Tensor, ...]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...]]
            The encoded output, as a float tensor

        """
        return self.cnn(data)


class RegistrationError(Exception):
    """Error thrown when acessing yaml tag on a non-registered class

    Thrown when trying to access the default yaml tag for a class
    typically occurs when called on an abstract class
    """
    pass


class registrable_factory:
    """Decorate Registrable factory method for use in the config

    This Descriptor class will set properties that allow the factory
    method to be specified directly in the config as a suffix to the
    tag; for example:

    .. code-block:: python

        class MyModel(Component):

            @registrable_factory
            def from_file(cls, path):
                # load instance from path
                ...
                return instance

    defines the factory, which can then be used in yaml:

    .. code-block:: yaml

        model: !MyModel.from_file
            path: some/path/to/file.pt

    """

    def __init__(self, fn: Any) ->None:
        self.fn = fn

    def __set_name__(self, owner: type, name: str) ->None:
        if not hasattr(owner, '_yaml_registered_factories'):
            raise RegistrationError(f"class {owner} doesn't have property _yaml_registered_factories; {owner} should subclass Registrable or Component")
        owner._yaml_registered_factories.add(name)
        setattr(owner, name, self.fn)


C = TypeVar('C', bound='Component')


FLAMBE_CLASS_KEY = '_flambe_class'


FLAMBE_CONFIG_KEY = '_flambe_config'


FLAMBE_DIRECTORIES_KEY = '_flambe_directories'


FLAMBE_SOURCE_KEY = '_flambe_source'


FLAMBE_STASH_KEY = '_flambe_stash'


KEEP_VARS_KEY = 'keep_vars'


R = TypeVar('R', bound='Registrable')


RT = TypeVar('RT', bound=Type['Registrable'])


logger = logging.getLogger(__name__)


def make_from_yaml_with_metadata(from_yaml_fn: Callable[..., Any], tag: str, factory_name: Optional[str]=None) ->Callable[..., Any]:

    @functools.wraps(from_yaml_fn)
    def wrapped(constructor: Any, node: Any) ->Any:
        obj = from_yaml_fn(constructor, node, factory_name=factory_name)
        obj.__dict__['_created_with_tag'] = tag
        return obj
    return wrapped


def make_to_yaml_with_metadata(to_yaml_fn: Callable[..., Any]) ->Callable[..., Any]:

    @functools.wraps(to_yaml_fn)
    def wrapped(representer: Any, node: Any) ->Any:
        if hasattr(node, '_created_with_tag'):
            tag = node._created_with_tag
        else:
            tag = Registrable.get_default_tag(type(node))
        return to_yaml_fn(representer, node, tag=tag)
    return wrapped


class Registrable(ABC):
    """Subclasses automatically registered as yaml tags

    Automatically registers subclasses with the yaml loader by
    adding a constructor and representer which can be overridden
    """
    _yaml_tags: Dict[Any, List[str]] = defaultdict(list)
    _yaml_tag_namespace: Dict[Type, str] = defaultdict(str)
    _yaml_registered_factories: Set[str] = set()

    def __init_subclass__(cls: Type[R], should_register: Optional[bool]=True, tag_override: Optional[str]=None, tag_namespace: Optional[str]=None, **kwargs: Mapping[str, Any]) ->None:
        super().__init_subclass__(**kwargs)
        cls._yaml_registered_factories = set(cls._yaml_registered_factories)
        if should_register:
            default_tag = cls.__name__ if tag_override is None else tag_override
            Registrable.register_tag(cls, default_tag, tag_namespace)

    @staticmethod
    def register_tag(class_: RT, tag: str, tag_namespace: Optional[str]=None) ->None:
        modules = class_.__module__.split('.')
        top_level_module_name = modules[0] if len(modules) > 0 else None
        global _reg_prefix
        if _reg_prefix is not None:
            tag_namespace = _reg_prefix
        elif tag_namespace is not None:
            tag_namespace = tag_namespace
        elif (tag_namespace is None and top_level_module_name is not None) and (top_level_module_name != 'flambe' and top_level_module_name != 'tests'):
            tag_namespace = top_level_module_name
        else:
            tag_namespace = None
        if tag_namespace is not None:
            full_tag = f'!{tag_namespace}.{tag}'
        else:
            full_tag = f'!{tag}'
        if class_ in class_._yaml_tag_namespace:
            if tag_namespace != class_._yaml_tag_namespace[class_]:
                msg = f'You are trying to register class {class_} with namespace {tag_namespace} != {class_._yaml_tag_namespace[class_]} so ignoring'
                warn(msg)
                return
        elif tag_namespace is not None:
            class_._yaml_tag_namespace[class_] = tag_namespace
        class_._yaml_tags[class_].append(full_tag)

        def registration_helper(factory_name: Optional[str]=None) ->None:
            from_yaml_tag = full_tag if factory_name is None else full_tag + '.' + factory_name
            logger.debug(f'Registering tag: {from_yaml_tag}')
            try:
                to_yaml = class_.to_yaml
            except AttributeError:

                def t_y(representer: Any, node: Any, tag: str) ->Any:
                    return representer.represent_yaml_object(tag, node, class_, flow_style=representer.default_flow_style)
                to_yaml = t_y
            finally:
                yaml.representer.add_representer(class_, make_to_yaml_with_metadata(to_yaml))
            try:
                from_yaml = class_.from_yaml
            except AttributeError:

                def f_y(constructor: Any, node: Any, factory_name: str) ->Any:
                    return constructor.construct_yaml_object(node, class_)
                from_yaml = f_y
            finally:
                yaml.constructor.add_constructor(from_yaml_tag, make_from_yaml_with_metadata(from_yaml, from_yaml_tag, factory_name))
        registration_helper()
        for factory_name in class_._yaml_registered_factories:
            factory_full_tag = f'{full_tag}.{factory_name}'
            class_._yaml_tags[class_, factory_name] = [factory_full_tag]
            registration_helper(factory_name)

    @staticmethod
    def get_default_tag(class_: RT, factory_name: Optional[str]=None) ->str:
        """Retrieve default yaml tag for class `cls`

        Retrieve the default tag (aka the last one, which will
        be the only one, or the alias if it exists) for use in
        yaml representation
        """
        if class_ in class_._yaml_tags:
            tag = class_._yaml_tags[class_][-1]
            if factory_name is not None and factory_name not in class_._yaml_registered_factories:
                raise RegistrationError(f'This class has no factory {factory_name}')
            elif factory_name is not None:
                tag = tag + '.' + factory_name
            return tag
        raise RegistrationError('This class has no registered tags')

    @classmethod
    @abstractmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) ->Any:
        """Use representer to create yaml representation of node

        See Component class, and experiment/options for examples

        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) ->Any:
        """Use constructor to create an instance of cls

        See Component class, and experiment/options for examples

        """
        pass


def alias(tag: str, tag_namespace: Optional[str]=None) ->Callable[[RT], RT]:
    """Decorate a Registrable subclass with a new tag

    Can be added multiple times to give a class multiple aliases,
    however the top most alias tag will be the default tag which means
    it will be used when representing the class in YAML

    """

    def decorator(cls: RT) ->RT:
        Registrable.register_tag(cls, tag, tag_namespace)
        return cls
    return decorator


class LinkError(Exception):
    pass


class UnpreparedLinkError(LinkError):
    pass


class MalformedLinkError(LinkError):
    pass


def create_link_str(schematic_path: Sequence[str], attr_path: Optional[Sequence[str]]=None) ->str:
    """Create a string representation of the specified link

    Performs the reverse operation of
    :func:`~flambe.compile.component.parse_link_str`

    Parameters
    ----------
    schematic_path : Sequence[str]
        List of entries corresponding to dictionary keys in a nested
        :class:`~flambe.compile.Schema`
    attr_path : Optional[Sequence[str]]
        List of attributes to access on the target object
        (the default is None).

    Returns
    -------
    str
        The string representation of the schematic + attribute paths

    Raises
    -------
    MalformedLinkError
        If the schematic_path is empty

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> create_link_str(['obj', 'key1', 'key2'], ['attr1', 'attr2'])
    'obj[key1][key2].attr1.attr2'

    """
    if len(schematic_path) == 0:
        raise MalformedLinkError("Can't create link without schematic path")
    root, schematic_path = schematic_path[0], schematic_path[1:]
    schematic_str = ''
    attr_str = ''
    if len(schematic_path) > 0:
        schematic_str = '[' + ']['.join(schematic_path) + ']'
    if attr_path is not None and len(attr_path) > 0:
        attr_str = '.' + '.'.join(attr_path)
    return root + schematic_str + attr_str


def parse_link_str(link_str: str) ->Tuple[Sequence[str], Sequence[str]]:
    """Parse link to extract schematic and attribute paths

    Links should be of the format ``obj[key1][key2].attr1.attr2`` where
    obj is the entry point; in a pipeline, obj would be the stage name,
    in a single-object config obj would be the target keyword at the
    top level. The following keys surrounded in brackets traverse
    the nested dictionary structure that appears in the config; this
    is intentonally analagous to how you would access properties in the
    dictionary when loaded into python. Then, you can use the dot
    notation to access the runtime instance attributes of the object
    at that location.

    Parameters
    ----------
    link_str : str
        Link to earlier object in the config of the format
        ``obj[key1][key2].attr1.attr2``

    Returns
    -------
    Tuple[Sequence[str], Sequence[str]]
        Tuple of the schematic and attribute paths respectively

    Raises
    -------
    MalformedLinkError
        If the link is written incorrectly

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> parse_link_str('obj[key1][key2].attr1.attr2')
    (['obj', 'key1', 'key2'], ['attr1', 'attr2'])

    """
    schematic_path: List[str] = []
    attr_path: List[str] = []
    temp: List[str] = []
    x = link_str
    bracket_open = False
    root_extracted = False
    while '[' in x or ']' in x:
        if bracket_open:
            temp = x.split(']', 1)
            if '[' in temp[0]:
                raise MalformedLinkError(f'Previous bracket unclosed in {link_str}')
            if len(temp) != 2:
                raise MalformedLinkError(f"Open bracket '[' not closed in {link_str}")
            schematic_path.append(temp[0])
            bracket_open = False
        else:
            temp = x.split('[', 1)
            if ']' in temp[0]:
                raise MalformedLinkError(f"Close ']' before open in {link_str}")
            if len(temp) != 2:
                raise MalformedLinkError(f"']' encountered before '[' in {link_str}")
            if len(temp[0]) != 0:
                if len(schematic_path) != 0:
                    raise MalformedLinkError(f'Text between brackets in {link_str}')
                schematic_path.append(temp[0])
                root_extracted = True
            elif len(schematic_path) == 0:
                raise MalformedLinkError(f'No top level object in {link_str}')
            bracket_open = True
        x = temp[1]
    attr_path = x.split('.')
    if not root_extracted:
        if len(attr_path[0]) == 0:
            raise MalformedLinkError(f'No top level object in {link_str}')
        schematic_path.append(attr_path[0])
    elif len(attr_path) > 1:
        if attr_path[0] != '':
            raise MalformedLinkError(f'Attribute without preceeding dot notation in {link_str}')
        if attr_path[-1] == '':
            raise MalformedLinkError(f'Trailing dot in {link_str}')
    attr_path = attr_path[1:]
    return schematic_path, attr_path


class LoadError(Exception):
    """Error thrown because of fatal error when loading"""


STATE_DICT_DELIMETER = '.'


class State(OrderedDict):
    """A state object for Flambe."""
    _metadata: Dict[str, Any]


VERSION_KEY = '_flambe_version'


class contextualized_linking:
    """Context manager used to change the representation of links

    Links are always defined in relation to some root object and an
    attribute path, so when representing some piece of a larger object
    all the links need to be redefined in relation to the target object

    """

    def __init__(self, root_obj: Any, prefix: str) ->None:
        self.root_obj = root_obj
        self.prefix = prefix
        self.old_root: Optional['Component'] = None
        self.old_active = False
        self.old_stash: Dict[str, Any] = {}

    def __enter__(self) ->'contextualized_linking':
        global _link_root_obj
        global _link_context_active
        global _link_obj_stash
        self.old_root = _link_root_obj
        self.old_active = _link_context_active
        self.old_stash = _link_obj_stash
        _link_root_obj = self.root_obj
        _link_context_active = True
        _link_obj_stash = {}
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) ->None:
        global _link_root_obj
        global _link_context_active
        global _link_obj_stash
        _link_root_obj = self.old_root
        _link_context_active = self.old_active
        _link_obj_stash = self.old_stash


_EMPTY = inspect.Parameter.empty


def fill_defaults(kwargs: Dict[str, Any], function: Callable[..., Any]) ->Dict[str, Any]:
    """Use function signature to add missing kwargs to a dictionary"""
    signature = inspect.signature(function)
    kwargs_with_defaults = kwargs.copy()
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        default = param.default
        if name not in kwargs and default != _EMPTY:
            kwargs_with_defaults[name] = default
    return kwargs_with_defaults


CONFIG_FILE_NAME = 'config.yaml'


HIGHEST_SERIALIZATION_PROTOCOL_VERSION = 1


PROTOCOL_VERSION_FILE_NAME = 'protocol_version.txt'


SOURCE_FILE_NAME = 'source.py'


STASH_FILE_NAME = 'stash.pkl'


STATE_FILE_NAME = 'state.pt'


VERSION_FILE_NAME = 'version.txt'


def _extract_prefix(root, directory):
    if directory.startswith(root):
        return directory[len(root):].lstrip(os.sep).replace(os.sep, STATE_DICT_DELIMETER)
    else:
        raise Exception()


def _prefix_keys(state, prefix):
    for key in set(state.keys()):
        val = state[key]
        del state[key]
        state[prefix + key] = val
    return state


def download_http_file(url: str, destination: str) ->None:
    """Download an HTTP/HTTPS file.

    Parameters
    ----------
    url: str
        The HTTP/HTTPS URL.
    destination: str
        The output file where to copy the content. Needs to support
        binary writing.

    """
    r = requests.get(url, allow_redirects=True)
    with open(destination, 'wb') as f:
        f.write(r.content)


def download_s3_file(url: str, destination: str) ->None:
    """Download an S3 file.

    Parameters
    ----------
    url: str
        The S3 URL. Should follow the format:
        's3://<bucket-name>[/path/to/file]'
    destination: str
        The output file where to copy the content

    """
    try:
        parsed_url = urlparse(url)
        s3 = boto3.client('s3')
        s3.download_file(parsed_url.netloc, parsed_url.path[1:], destination)
    except botocore.client.ClientError:
        raise ValueError(f'Error downlaoding artifact from s3.')


def download_s3_folder(url: str, destination: str) ->None:
    """Download an S3 folder.

    Parameters
    ----------
    url: str
        The S3 URL. Should follow the format:
        's3://<bucket-name>[/path/to/folder]'
    destination: str
        The output folder where to copy the content

    """
    try:
        subprocess.check_output(f'aws s3 cp --recursive {url} {destination}'.split(), stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as exc:
        logger.debug(exc.output)
        raise ValueError(f'Error downlaoding artifacts from s3. ' + 'Check logs for more information')


def http_exists(url: str) ->bool:
    """Check if an HTTP/HTTPS file exists.

    Parameters
    ----------
    url: str
        The HTTP/HTTPS URL.

    Returns
    -------
    bool
        True if the HTTP file exists

    """
    try:
        r = requests.head(url, allow_redirects=True)
        return r.status_code != 404
    except requests.ConnectionError:
        return False


class CompilationError(Exception):
    pass


def merge_kwargs(kwargs: Dict[str, Any], compiled_kwargs: Dict[str, Any]) ->Dict[str, Any]:
    """Replace non links in kwargs with corresponding compiled values

    For every key in `kwargs` if the value is NOT a link and IS a
    Schema, replace with the corresponding value in `compiled_kwargs`

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Original kwargs containing Links and Schemas
    compiled_kwargs : Dict[str, Any]
        Processes kwargs containing no links and no Schemas

    Returns
    -------
    Dict[str, Any]
        kwargs with links, but with Schemas replaced by compiled
        objects

    """
    merged_kwargs = {}
    for kw in kwargs:
        if not isinstance(kwargs[kw], Link) and isinstance(kwargs[kw], Schema):
            if kw not in compiled_kwargs:
                raise CompilationError('Non matching kwargs and compiled_kwargs')
            merged_kwargs[kw] = compiled_kwargs[kw]
        else:
            merged_kwargs[kw] = kwargs[kw]
    return merged_kwargs


class MixtureOfSoftmax(Module):
    """Implement the MixtureOfSoftmax output layer.

    Attributes
    ----------
    pi: FullyConnected
        softmax layer over the different softmax
    layers: [FullyConnected]
        list of the k softmax layers

    """

    def __init__(self, input_size: int, output_size: int, k: int=1, take_log: bool=True) ->None:
        """Initialize the MOS layer.

        Parameters
        ----------
        input_size: int
            input dimension
        output_size: int
            output dimension
        k: int (Default: 1)
            number of softmax in the mixture

        """
        super().__init__()
        self.pi_w = MLPEncoder(input_size, k)
        self.softmax = nn.Softmax()
        self.layers = [MLPEncoder(input_size, output_size) for _ in range(k)]
        self.tanh = nn.Tanh()
        self.activation = nn.LogSoftmax() if take_log else nn.Softmax()

    def forward(self, data: Tensor) ->Tensor:
        """Implement mixture of softmax for language modeling.

        Parameters
        ----------
        data: torch.Tensor
            seq_len x batch_size x hidden_size

        Return
        -------
        out: Variable
            output matrix of shape seq_len x batch_size x out_size

        """
        w = self.softmax(self.pi_w(data))
        out = [(w[:, :, (i)] * self.tanh(W(data))) for i, W in enumerate(self.layers)]
        out = torch.cat(out, dim=0).sum(dim=0)
        return self.activation(out)


class FirstPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        return data[:, (0), :]


class LastPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        if padding_mask is None:
            lengths = torch.tensor([data.size(1)] * data.size(0)).long()
        else:
            lengths = padding_mask.long().sum(dim=1)
        return data[(torch.arange(data.size(0)).long()), (lengths - 1), :]


def _default_padding_mask(data: torch.Tensor) ->torch.Tensor:
    """
    Builds a 1s padding mask taking into account initial 2 dimensions
    of input data.

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]

    Returns
    ----------
    torch.Tensor
        A padding mask , as a tensor of shape [B x S]
    """
    return torch.ones((data.size(0), data.size(1)))


def _sum_with_padding_mask(data: torch.Tensor, padding_mask: torch.Tensor) ->torch.Tensor:
    """
    Applies padding_mask and performs summation over the data

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]
    padding_mask: torch.Tensor
        The input mask, as a tensor of shape [B X S]
    Returns
    ----------
    torch.Tensor
        The result of the summation, as a tensor of shape [B x H]

    """
    return (data * padding_mask.unsqueeze(2)).sum(dim=1)


class SumPooling(Module):
    """Get the sum of the hidden state of a sequence."""

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        if padding_mask is None:
            padding_mask = _default_padding_mask(data)
        return _sum_with_padding_mask(data, padding_mask)


class AvgPooling(Module):
    """Get the average of the hidden state of a sequence."""

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        if padding_mask is None:
            padding_mask = _default_padding_mask(data)
        value_count = padding_mask.sum(dim=1).unsqueeze(1)
        data = _sum_with_padding_mask(data, padding_mask)
        return data / value_count


class StructuredSelfAttentivePooling(Module):
    """Structured Self Attentive Pooling."""

    def __init__(self, input_size: int, attention_heads: int=16, attention_units: Sequence[int]=(300,), output_activation: Optional[torch.nn.Module]=None, hidden_activation: Optional[torch.nn.Module]=None, is_biased: bool=False, input_dropout: float=0.0, attention_dropout: float=0.0):
        """Initialize a self attention pooling layer

        A generalized implementation of:
        `A Structured Self-attentive Sentence Embedding`
        https://arxiv.org/pdf/1703.03130.pdf

        cite:
        @article{lin2017structured,
            title={A structured self-attentive sentence embedding},
            author={Lin, Zhouhan and Feng, Minwei and
            Santos, Cicero Nogueira dos and Yu, Mo and
            Xiang, Bing and Zhou, Bowen and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1703.03130},
        year={2017}
        }

        Parameters
        ----------
        input_size : int
            The input data dim
        attention_heads: int
            the number of attn heads
        attention_units: Iterable[int]
            the list of hidden dimensions of the MLP computing the attn
        output_activation: Optional[torch.nn.Module]
            The output activation to the attention weights.
            Defaults to nn.Softmax, in accordance with the paper.
        hidden_activation: Optional[torch.nn.Module]
            The hidden activation to the attention weight computation.
            Defaults to nn.Tanh, in accordance with the paper.
        is_biased: bool
            Whether the MLP should be biased. Defaults to false,
            as in the paper.
        input_dropout: float
            dropout applied to the data argument of the forward method.
        attention_dropout: float
            dropout applied to the attention output before applying it
            to the input for reduction. decouples the attn dropout
            from the input dropout
        """
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0.0 else nn.Identity()
        dimensions = [input_size, *attention_units, attention_heads]
        layers = []
        for l in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[l], dimensions[l + 1], bias=is_biased))
            layers.append(nn.Tanh() if hidden_activation is None else hidden_activation)
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        if attention_dropout > 0.0:
            layers.append(nn.Dropout(attention_dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_activation = nn.Softmax(dim=1) if output_activation is None else output_activation

    def _compute_attention(self, data: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        """Computes the attention

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The attention, as a tensor of shape [B x S x HEADS]

        """
        batch_size, num_encs, dim = data.shape
        data = self.in_drop(data)
        attention_logits = self.mlp(data.reshape(-1, dim)).reshape(batch_size, num_encs, -1)
        if mask is not None:
            mask = mask.unsqueeze(2).float()
            attention_logits = attention_logits * mask + (1.0 - mask) * -1e+20
        attention = self.output_activation(attention_logits)
        return attention

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        attention = self._compute_attention(data, mask)
        attended = torch.bmm(attention.transpose(1, 2), data)
        return attended.mean(dim=1)


class GeneralizedPooling(StructuredSelfAttentivePooling):
    """Self attention pooling."""

    def __init__(self, input_size: int, attention_units: Sequence[int]=(300,), output_activation: Optional[torch.nn.Module]=None, hidden_activation: Optional[torch.nn.Module]=None, is_biased: bool=True, input_dropout: float=0.0, attention_dropout: float=0.0):
        """Initialize a self attention pooling layer

        A generalized implementation of:
        `Enhancing Sentence Embedding with Generalized Pooling`
        https://arxiv.org/pdf/1806.09828.pdf

        cite:
        @article{chen2018enhancing,
            title={Enhancing sentence embedding with
                   generalized pooling},
            author={Chen, Qian and Ling, Zhen-Hua and Zhu, Xiaodan},
            journal={arXiv preprint arXiv:1806.09828},
            year={2018}
        }

        Parameters
        ----------
        input_size : int
            The input data dim
        attention_units: Iterable[int]
            the list of hidden dimensions of the MLP computing the attn
        output_activation: Optional[torch.nn.Module]
            The output activation to the attention weights.
            Defaults to nn.Softmax, in accordance with the paper.
        hidden_activation: Optional[torch.nn.Module]
            The hidden activation to the attention weight computation.
            Defaults to nn.Tanh, in accordance with the paper.
        is_biased: bool
            Whether the MLP should be biased. Defaults to true,
            as in the paper.
        input_dropout: float
            dropout applied to the data argument of the forward method.
        attention_dropout: float
            dropout applied to the attention output before applying it
            to the input for reduction. decouples the attn dropout
            from the input dropout
        """
        super().__init__(input_size=input_size, attention_heads=input_size, attention_units=attention_units, output_activation=output_activation, hidden_activation=nn.ReLU() if hidden_activation is None else hidden_activation, is_biased=is_biased, input_dropout=input_dropout, attention_dropout=attention_dropout)

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        attention = self._compute_attention(data, mask)
        attended = attention * data
        return attended.mean(dim=1)


class RNNEncoder(Module):
    """Implements a multi-layer RNN.

    This module can be used to create multi-layer RNN models, and
    provides a way to reduce to output of the RNN to a single hidden
    state by pooling the encoder states either by taking the maximum,
    average, or by taking the last hidden state before padding.

    Padding is dealt with by using torch's PackedSequence.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule

    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int=1, rnn_type: str='lstm', dropout: float=0, bidirectional: bool=False, layer_norm: bool=False, highway_bias: float=0, rescale: bool=True, enforce_sorted: bool=False, **kwargs) ->None:
        """Initializes the RNNEncoder object.

        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        enforce_sorted: bool
            Whether rnn should enforce that sequences are ordered by
            length. Requires True for ONNX support. Defaults to False.
        kwargs
            Additional parameters to be passed to SRU when building
            the rnn.

        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`

        """
        super().__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        if rnn_type in ['lstm', 'gru']:
            if kwargs:
                logger.warn(f"The following '{kwargs}' will be ignored " + "as they are only considered when using 'sru' as " + "'rnn_type'")
            rnn_fn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_fn(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == 'sru':
            try:
                self.rnn = SRU(input_size, hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional, layer_norm=layer_norm, rescale=rescale, highway_bias=highway_bias, **kwargs)
            except TypeError:
                raise ValueError(f'Unkown kwargs passed to SRU: {kwargs}')
        else:
            raise ValueError(f'Unkown rnn type: {rnn_type}, use of of: gru, sru, lstm')

    def forward(self, data: Tensor, state: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S], dtype should be bool

        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x S x H]
        Tensor
            The encoded state, as a float tensor of shape [L x B x H]

        """
        data = data.transpose(0, 1)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1)
        if padding_mask is None:
            output, state = self.rnn(data, state)
        elif self.rnn_type == 'sru':
            output, state = self.rnn(data, state, mask_pad=~padding_mask)
        else:
            lengths = padding_mask.long().sum(dim=0)
            packed = nn.utils.rnn.pack_padded_sequence(data, lengths, enforce_sorted=self.enforce_sorted)
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return output.transpose(0, 1).contiguous(), state


class PooledRNNEncoder(Module):
    """Implement an RNNEncoder with additional pooling.

    This class can be used to obtan a single encoded output for
    an input sequence. It also ignores the state of the RNN.

    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int=1, rnn_type: str='lstm', dropout: float=0, bidirectional: bool=False, layer_norm: bool=False, highway_bias: float=0, rescale: bool=True, pooling: str='last') ->None:
        """Initializes the PooledRNNEncoder object.

        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        pooling : Optional[str], optional
            If given, the output is pooled into a single hidden state,
            through the given pooling routine. Should be one of:
            "first", last", "average", or "sum". Defaults to "last"

        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`

        """
        super().__init__()
        warnings.warn('PooledRNNEncoder is deprecated, please use the Pooling                        module in the Embedder object', DeprecationWarning)
        self.pooling = pooling
        self.rnn = RNNEncoder(input_size=input_size, hidden_size=hidden_size, n_layers=n_layers, rnn_type=rnn_type, dropout=dropout, bidirectional=bidirectional, layer_norm=layer_norm, highway_bias=highway_bias, rescale=rescale)
        self.output_size = 2 * hidden_size if bidirectional else hidden_size

    def forward(self, data: Tensor, state: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None) ->Tensor:
        """Perform a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S]

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor of shape [B x H]

        """
        output, _ = self.rnn(data, state=state, padding_mask=padding_mask)
        if padding_mask is None:
            padding_mask = torch.ones_like(output)
        cast(torch.Tensor, padding_mask)
        if self.pooling == 'average':
            output = (output * padding_mask.unsqueeze(2)).sum(dim=1)
            output = output / padding_mask.sum(dim=1)
        elif self.pooling == 'sum':
            output = (output * padding_mask.unsqueeze(2)).sum(dim=1)
        elif self.pooling == 'last':
            lengths = padding_mask.long().sum(dim=1)
            output = output[(torch.arange(output.size(0)).long()), (lengths - 1), :]
        elif self.pooling == 'first':
            output = output[(torch.arange(output.size(0)).long()), (0), :]
        else:
            raise ValueError(f'Invalid pooling type: {self.pooling}')
        return output


class Sequential(Module):
    """Implement a Sequential module.

    This class can be used in the same way as torch's nn.Sequential,
    with the difference that it accepts kwargs arguments.

    """

    def __init__(self, **kwargs: Dict[str, Union[Module, torch.nn.Module]]) ->None:
        """Initialize the Sequential module.

        Parameters
        ----------
        kwargs: Dict[str, Union[Module, torch.nn.Module]]
            The list of modules.

        """
        super().__init__()
        modules = []
        for name, module in kwargs.items():
            setattr(self, name, module)
            modules.append(module)
        self.seq = torch.nn.Sequential(modules)

    def forward(self, data: torch.Tensor) ->torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model

        Returns
        -------
        output: torch.Tensor
            output of the model

        """
        return self.seq(data)


class SoftmaxLayer(Module):
    """Implement an SoftmaxLayer module.

    Can be used to form a classifier out of any encoder.
    Note: by default takes the log_softmax so that it can be fed to
    the NLLLoss module. You can disable this behavior through the
    `take_log` argument.

    """

    def __init__(self, input_size: int, output_size: int, mlp_layers: int=1, mlp_dropout: float=0.0, mlp_hidden_activation: Optional[nn.Module]=None, take_log: bool=True) ->None:
        """Initialize the SoftmaxLayer.

        Parameters
        ----------
        input_size : int
            Input size of the decoder, usually the hidden size of
            some encoder.
        output_size : int
            The output dimension, usually the number of target labels
        mlp_layers : int
            The number of layers in the MLP
        mlp_dropout: float, optional
            Dropout to be used before each MLP layer
        mlp_hidden_activation: nn.Module, optional
            Any PyTorch activation layer, defaults to None
        take_log: bool, optional
            If ``True``, compute the LogSoftmax to be fed in NLLLoss.
            Defaults to ``False``.

        """
        super().__init__()
        softmax = nn.LogSoftmax(dim=-1) if take_log else nn.Softmax()
        self.mlp = MLPEncoder(input_size=input_size, output_size=output_size, n_layers=mlp_layers, dropout=mlp_dropout, hidden_activation=mlp_hidden_activation, output_activation=softmax)

    def forward(self, data: torch.Tensor) ->torch.Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data: torch.Tensor
            input to the model of shape (*, input_size)

        Returns
        -------
        output: torch.Tensor
            output of the model of shape (*, output_size)

        """
        return self.mlp(data)


class TransformerDecoderLayer(Module):
    """A TransformerDecoderLayer.

    A TransformerDecoderLayer is made up of self-attn, multi-head-attn
    and feedforward network. This standard decoder layer is based on the
    paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz
    Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
    In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way
    during application.

    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1) ->None:
        """Initialize a TransformerDecoder.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequnce from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            the mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            the mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            the mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor
            Output tensor of shape [T x B x H]

        """
        if padding_mask is not None:
            padding_mask = ~padding_mask
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers"""

    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int=2048, dropout: float=0.1) ->None:
        """Initialize the TransformerDecoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            The number of expected features in encoder/decoder inputs.
        nhead : int, optional
            The number of heads in the multiheadattention.
        num_layers : int
            The number of sub-encoder-layers in the encoder (required).
        dim_feedforward : int, optional
            The inner feedforard dimension, by default 2048.
        dropout : float, optional
            The dropout percentage, by default 0.1.

        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)
        layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            The mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor

        """
        output = tgt.transpose(0, 1)
        if self.input_size != self.d_model:
            output = self.proj(output)
        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, padding_mask=padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward.

    This standard encoder layer is based on the paper "Attention Is
    All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may
    modify or implement in a different way during application.

    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1) ->None:
        """Initialize a TransformerEncoderLayer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, memory: Optional[torch.Tensor]=None, src_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the input through the endocder layer.

        Parameters
        ----------
        src: torch.Tensor
            The seqeunce to the encoder layer (required).
        memory: torch.Tensor, optional
            Optional memory from previous sequence, unused by default.
        src_mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B x S x H]

        """
        if padding_mask is not None:
            padding_mask = ~padding_mask.bool()
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8, num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1) ->None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.

        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, src: torch.Tensor, memory: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the input through the endocder layers in turn.

        Parameters
        ----------
        src: torch.Tensor
            The sequence to the encoder (required).
        memory: torch.Tensor, optional
            Optional memory, unused by default.
        mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        """
        output = src.transpose(0, 1)
        if self.input_size != self.d_model:
            output = self.proj(output)
        for i in range(self.num_layers):
            output = self.layers[i](output, memory=memory, src_mask=mask, padding_mask=padding_mask)
        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Transformer(Module):
    """A Transformer model

    User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    """

    def __init__(self, input_size, d_model: int=512, nhead: int=8, num_encoder_layers: int=6, num_decoder_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1) ->None:
        """Initialize the Transformer Model.

        Parameters
        ----------
        input_size : int, optional
            dimension of embeddings. If different from
            d_model, then a linear layer is added to project from
            input_size to d_model.
        d_model : int, optional
            the number of expected features in the
            encoder/decoder inputs (default=512).
        nhead : int, optional
            the number of heads in the multiheadattention
            models (default=8).
        num_encoder_layers : int, optional
            the number of sub-encoder-layers in the encoder
            (default=6).
        num_decoder_layers : int, optional
            the number of sub-decoder-layers in the decoder
            (default=6).
        dim_feedforward : int, optional
            the dimension of the feedforward network model
            (default=2048).
        dropout : float, optional
            the dropout value (default=0.1).

        """
        super().__init__()
        self.encoder = TransformerEncoder(input_size, d_model, nhead, dim_feedforward, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(input_size, d_model, nhead, dim_feedforward, num_encoder_layers, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, src_key_padding_mask: Optional[torch.Tensor]=None, tgt_key_padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Take in and process masked source/target sequences.

        Parameters
        ----------
        src: torch.Tensor
            the sequence to the encoder (required).
            shape: :math:`(N, S, E)`.
        tgt: torch.Tensor
            the sequence to the decoder (required).
            shape: :math:`(N, T, E)`.
        src_mask: torch.Tensor, optional
            the additive mask for the src sequence (optional).
            shape: :math:`(S, S)`.
        tgt_mask: torch.Tensor, optional
            the additive mask for the tgt sequence (optional).
            shape: :math:`(T, T)`.
        memory_mask: torch.Tensor, optional
            the additive mask for the encoder output (optional).
            shape: :math:`(T, S)`.
        src_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for src keys per batch (optional).
            shape: :math:`(N, S)`
        tgt_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for tgt keys per batch (optional).
            shape: :math:`(N, T)`.
        memory_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for memory keys per batch (optional).
            shape" :math:`(N, S)`.

        Returns
        -------
        output: torch.Tensor
            The output sequence, shape: :math:`(N, T, E)`.

        Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else.
            These masks ensure that predictions for position i depend
            only on the unmasked positions j and are applied identically
            for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor
            where False values are positions that should be masked with
            float('-inf') and True values will be unchanged.
            This mask ensures that no information will be taken from
            position i if it is masked, and has a separate mask for each
            sequence in a batch.
        Note: Due to the multi-head attention architecture in the
            transformer model, the output sequence length of a
            transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target
            sequence length, N is the batchsize, E is the feature number

        """
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        memory = self.encoder(src, mask=src_mask, padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerSRUDecoderLayer(Module):
    """A TransformerSRUDecoderLayer with an SRU replacing the FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, sru_dropout: Optional[float]=None, **kwargs: Dict[str, Any]) ->None:
        """Initialize a TransformerDecoder.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sru = SRUCell(d_model, dim_feedforward, dropout, sru_dropout or dropout, bidirectional=False, has_skip_term=False, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, state: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            the mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor, optional
            the mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor
            Output Tensor of shape [S x B x H]

        """
        reversed_mask = None
        if padding_mask is not None:
            reversed_mask = ~padding_mask
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=reversed_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2, _ = self.sru(tgt, state, mask_pad=padding_mask)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerSRUDecoder(Module):
    """A TransformerSRUDecoderwith an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8, num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1, sru_dropout: Optional[float]=None, **kwargs: Dict[str, Any]) ->None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)
        layer = TransformerSRUDecoderLayer(d_model, nhead, dim_feedforward, dropout, sru_dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, state: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the tgt keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.
        memory_key_padding_mask: torch.Tensor, optional
            The mask for the memory keys per batch (optional).

        Returns
        -------
        torch.Tensor

        """
        output = tgt.transpose(0, 1)
        state = state or [None] * self.num_layers
        if self.input_size != self.d_model:
            output = self.proj(output)
        for i in range(self.num_layers):
            output = self.layers[i](output, memory, state=state[i], tgt_mask=tgt_mask, memory_mask=memory_mask, padding_mask=padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerSRUEncoderLayer(Module):
    """A TransformerSRUEncoderLayer with an SRU replacing the FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, sru_dropout: Optional[float]=None, bidirectional: bool=False, **kwargs: Dict[str, Any]) ->None:
        """Initialize a TransformerSRUEncoderLayer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.sru = SRUCell(d_model, dim_feedforward, dropout, sru_dropout or dropout, bidirectional=bidirectional, has_skip_term=False, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, state: Optional[torch.Tensor]=None, src_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the endocder layer.

        Parameters
        ----------
        src: torch.Tensor
            The sequence to the encoder layer (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        src_mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        Returns
        -------
        torch.Tensor
            Output Tensor of shape [S x B x H]
        torch.Tensor
            Output state of the SRU of shape [N x B x H]

        """
        reversed_mask = None
        if padding_mask is not None:
            reversed_mask = ~padding_mask
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=reversed_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2, state = self.sru(src, state, mask_pad=padding_mask)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, state


class TransformerSRUEncoder(Module):
    """A TransformerSRUEncoder with an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8, num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1, sru_dropout: Optional[float]=None, bidirectional: bool=False, **kwargs: Dict[str, Any]) ->None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        input_size : int
            The embedding dimension of the model.  If different from
            d_model, a linear projection layer is added.
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        if input_size != d_model:
            self.proj = nn.Linear(input_size, d_model)
        layer = TransformerSRUEncoderLayer(d_model, nhead, dim_feedforward, dropout, sru_dropout, bidirectional)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, src: torch.Tensor, state: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the endocder layers in turn.

        Parameters
        ----------
        src: torch.Tensor
            The sequnce to the encoder (required).
        state: Optional[torch.Tensor]
            Optional state from previous sequence encoding.
            Only passed to the SRU (not used to perform multihead
            attention).
        mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).
            Should be True for tokens to leave untouched, and False
            for padding tokens.

        """
        output = src.transpose(0, 1)
        if self.input_size != self.d_model:
            output = self.proj(output)
        new_states = []
        for i in range(self.num_layers):
            input_state = state[i] if state is not None else None
            output, new_state = self.layers[i](output, state=input_state, src_mask=mask, padding_mask=padding_mask)
            new_states.append(new_state)
        new_states = torch.stack(new_states, dim=0)
        return output.transpose(0, 1), new_states

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerSRU(Module):
    """A Transformer with an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8, num_encoder_layers: int=6, num_decoder_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1, sru_dropout: Optional[float]=None, bidrectional: bool=False, **kwargs: Dict[str, Any]) ->None:
        """Initialize the TransformerSRU Model.

        Parameters
        ----------
        input_size : int, optional
            dimension of embeddings (default=512). if different from
            d_model, then a linear layer is added to project from
            input_size to d_model.
        d_model : int, optional
            the number of expected features in the
            encoder/decoder inputs (default=512).
        nhead : int, optional
            the number of heads in the multiheadattention
            models (default=8).
        num_encoder_layers : int, optional
            the number of sub-encoder-layers in the encoder
            (default=6).
        num_decoder_layers : int, optional
            the number of sub-decoder-layers in the decoder
            (default=6).
        dim_feedforward : int, optional
            the dimension of the feedforward network model
            (default=2048).
        dropout : float, optional
            the dropout value (default=0.1).
        sru_dropout: float, optional
            Dropout for the SRU cell. If not given, uses the same
            dropout value as the rest of the transformer.
        bidrectional: bool, optional
            Whether the SRU Encoder module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super().__init__()
        self.encoder = TransformerSRUEncoder(input_size, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, sru_dropout, bidrectional, **kwargs)
        self.decoder = TransformerSRUDecoder(input_size, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, sru_dropout, **kwargs)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None, src_key_padding_mask: Optional[torch.Tensor]=None, tgt_key_padding_mask: Optional[torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Take in and process masked source/target sequences.

        Parameters
        ----------
        src: torch.Tensor
            the sequence to the encoder (required).
            shape: :math:`(N, S, E)`.
        tgt: torch.Tensor
            the sequence to the decoder (required).
            shape: :math:`(N, T, E)`.
        src_mask: torch.Tensor, optional
            the additive mask for the src sequence (optional).
            shape: :math:`(S, S)`.
        tgt_mask: torch.Tensor, optional
            the additive mask for the tgt sequence (optional).
            shape: :math:`(T, T)`.
        memory_mask: torch.Tensor, optional
            the additive mask for the encoder output (optional).
            shape: :math:`(T, S)`.
        src_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for src keys per batch (optional).
            shape: :math:`(N, S)`.
        tgt_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for tgt keys per batch (optional).
            shape: :math:`(N, T)`.
        memory_key_padding_mask: torch.Tensor, optional
            the ByteTensor mask for memory keys per batch (optional).
            shape" :math:`(N, S)`.

        Returns
        -------
        output: torch.Tensor
            The output sequence, shape: :math:`(T, N, E)`.

        Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else.
            These masks ensure that predictions for position i depend
            only on the unmasked positions j and are applied identically
            for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor
            where False values are positions that should be masked with
            float('-inf') and True values will be unchanged.
            This mask ensures that no information will be taken from
            position i if it is masked, and has a separate mask for each
            sequence in a batch.
        Note: Due to the multi-head attention architecture in the
            transformer model, the output sequence length of a
            transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target
            sequence length, N is the batchsize, E is the feature number

        """
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        memory, state = self.encoder(src, mask=src_mask, padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, state=state, tgt_mask=tgt_mask, memory_mask=memory_mask, padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class ImageClassifier(Module):
    """Implements a simple image classifier.

    This classifier consists of an encocder module, followed by
    a fully connected output layer that outputs a probability
    distribution.

    Attributes
    ----------
    encoder: Moodule
        The encoder layer
    output_layer: Module
        The output layer, yields a probability distribution over targets
    """

    def __init__(self, encoder: Module, output_layer: Module) ->None:
        super().__init__()
        self.encoder = encoder
        self.output_layer = output_layer

    def forward(self, data: Tensor, target: Optional[Tensor]=None) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.

        Parameters
        ----------
        data: Tensor
            The input data
        target: Tensor, optional
            The input targets, optional

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]
            The output predictions, and optionally the targets

        """
        encoded = self.encoder(data)
        pred = self.output_layer(torch.flatten(encoded, 1))
        return (pred, target) if target is not None else pred


class IntermediateTorchOnly(torch.nn.Module):

    def __init__(self, component):
        super().__init__()
        self.child = component
        self.linear = torch.nn.Linear(2, 2)


class DummyModel(Module):
    pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AvgPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CNNEncoder,
     lambda: ([], {'input_channels': 4, 'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (CosineDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (CosineMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EuclideanDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EuclideanMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FirstPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedPooling,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (HyperbolicDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (HyperbolicMean,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 5])], {}),
     False),
    (ImageClassifier,
     lambda: ([], {'encoder': _mock_layer(), 'output_layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LanguageModel,
     lambda: ([], {'embedder': _mock_layer(), 'output_layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LastPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogisticRegression,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPEncoder,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixtureOfSoftmax,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PooledRNNEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RNNEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SoftmaxLayer,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StructuredSelfAttentivePooling,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SumPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TextClassifier,
     lambda: ([], {'embedder': _mock_layer(), 'output_layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerDecoder,
     lambda: ([], {'input_size': 4, 'd_model': 4, 'nhead': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_asappresearch_flambe(_paritybench_base):
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

