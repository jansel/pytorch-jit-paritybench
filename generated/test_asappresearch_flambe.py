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
ssh = _module
utils = _module
compile = _module
component = _module
downloader = _module
extensions = _module
registrable = _module
serialization = _module
dataset = _module
tabular = _module
experiment = _module
options = _module
progress = _module
tune_adapter = _module
webapp = _module
app = _module
wording = _module
export = _module
builder = _module
exporter = _module
field = _module
bow = _module
label = _module
text = _module
learn = _module
distillation = _module
eval = _module
script = _module
train = _module
logging = _module
datatypes = _module
handler = _module
contextual_file = _module
tensorboard = _module
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
model = _module
logistic_regression = _module
nlp = _module
classification = _module
datasets = _module
model = _module
fewshot = _module
language_modeling = _module
fields = _module
model = _module
sampler = _module
transformers = _module
nn = _module
cnn = _module
distance = _module
cosine = _module
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
tokenizer = _module
char = _module
word = _module
config = _module
version = _module
vision = _module
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

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import inspect


import logging


from warnings import warn


from typing import Type


from typing import TypeVar


from typing import Any


from typing import Mapping


from typing import Dict


from typing import Optional


from typing import List


from typing import Union


from typing import Generator


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


from typing import NamedTuple


from collections import OrderedDict as odict


import torch.nn.functional as F


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import math


from typing import Iterator


import numpy as np


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn.utils.clip_grad import clip_grad_norm_


from torch.nn.utils.clip_grad import clip_grad_value_


import time


import numpy


from torch import Tensor


from torch.nn import Sigmoid


import torch.nn as nn


from torch import nn


from typing import cast


import warnings


from collections import defaultdict


from itertools import chain


from functools import partial


from torch.utils.data import DataLoader


from torch.nn.utils.rnn import pad_sequence


from collections import abc


from torch.nn import NLLLoss


from torch.optim import Adam


import random


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
        self.encoder = MLPEncoder(input_size, output_size=1, n_layers=1,
            output_activation=Sigmoid())

    def forward(self, data: Tensor, target: Optional[Tensor]=None) ->Union[
        Tensor, Tuple[Tensor, Tensor]]:
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


def conv_block(conv_mod: nn.Module, activation: nn.Module, pooling: nn.
    Module, dropout: float, batch_norm: Optional[nn.Module]=None) ->nn.Module:
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

    def __init__(self, input_channels: int, channels: List[int], conv_dim:
        int=2, kernel_size: Union[int, List[Union[Tuple[int, ...], int]]]=3,
        activation: nn.Module=None, pooling: nn.Module=None, dropout: float
        =0, batch_norm: bool=True, stride: int=1, padding: int=0) ->None:
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
        dim2mod = {(1): (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d), (2): (nn
            .Conv2d, nn.BatchNorm2d, nn.MaxPool2d), (3): (nn.Conv3d, nn.
            BatchNorm3d, nn.MaxPool3d)}
        if conv_dim not in dim2mod:
            raise ValueError(
                f'Invalid conv_dim value {conv_dim}. Values 1, 2, 3 supported')
        if isinstance(kernel_size, List) and len(kernel_size) != len(channels):
            raise ValueError(
                'Kernel size list should have same length as channels list')
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
                    raise ValueError(
                        'Kernel size tuple should have same length as conv_dim'
                        )
            layer = conv_block(conv(prev_c, c, k, stride, padding),
                activation, pooling, dropout, bn(c))
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


class MLPEncoder(Module):
    """Implements a multi layer feed forward network.

    This module can be used to create output layers, or
    more complex multi-layer feed forward networks.

    Attributes
    ----------
    seq: nn.Sequential
        the sequence of layers and activations

    """

    def __init__(self, input_size: int, output_size: int, n_layers: int=1,
        dropout: float=0.0, output_activation: Optional[nn.Module]=None,
        hidden_size: Optional[int]=None, hidden_activation: Optional[nn.
        Module]=None) ->None:
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


C = TypeVar('C', bound='Component')


FLAMBE_CLASS_KEY = '_flambe_class'


FLAMBE_CONFIG_KEY = '_flambe_config'


FLAMBE_DIRECTORIES_KEY = '_flambe_directories'


FLAMBE_SOURCE_KEY = '_flambe_source'


FLAMBE_STASH_KEY = '_flambe_stash'


KEEP_VARS_KEY = 'keep_vars'


class RegistrationError(Exception):
    """Error thrown when acessing yaml tag on a non-registered class

    Thrown when trying to access the default yaml tag for a class
    typically occurs when called on an abstract class
    """
    pass


logger = logging.getLogger(__name__)


def make_from_yaml_with_metadata(from_yaml_fn: Callable[..., Any], tag: str,
    factory_name: Optional[str]=None) ->Callable[..., Any]:

    @functools.wraps(from_yaml_fn)
    def wrapped(constructor: Any, node: Any) ->Any:
        obj = from_yaml_fn(constructor, node, factory_name=factory_name)
        obj.__dict__['_created_with_tag'] = tag
        return obj
    return wrapped


def make_to_yaml_with_metadata(to_yaml_fn: Callable[..., Any]) ->Callable[
    ..., Any]:

    @functools.wraps(to_yaml_fn)
    def wrapped(representer: Any, node: Any) ->Any:
        if hasattr(node, '_created_with_tag'):
            tag = node._created_with_tag
        else:
            tag = Registrable.get_default_tag(type(node))
        return to_yaml_fn(representer, node, tag=tag)
    return wrapped


class MixtureOfSoftmax(Module):
    """Implement the MixtureOfSoftmax output layer.

    Attributes
    ----------
    pi: FullyConnected
        softmax layer over the different softmax
    layers: [FullyConnected]
        list of the k softmax layers

    """

    def __init__(self, input_size: int, output_size: int, k: int=1,
        take_log: bool=True) ->None:
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
        out = [(w[:, :, (i)] * self.tanh(W(data))) for i, W in enumerate(
            self.layers)]
        out = torch.cat(out, dim=0).sum(dim=0)
        return self.activation(out)


class FirstPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.
        Tensor]=None) ->torch.Tensor:
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

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.
        Tensor]=None) ->torch.Tensor:
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
    return torch.ones((data.size(0), data.size(1))).to(data)


def _sum_with_padding_mask(data: torch.Tensor, padding_mask: torch.Tensor
    ) ->torch.Tensor:
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

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.
        Tensor]=None) ->torch.Tensor:
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

    def forward(self, data: torch.Tensor, padding_mask: Optional[torch.
        Tensor]=None) ->torch.Tensor:
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

    def __init__(self, input_size: int, attention_heads: int=16,
        attention_units: Sequence[int]=(300,), output_activation: Optional[
        torch.nn.Module]=None, hidden_activation: Optional[torch.nn.Module]
        =None, is_biased: bool=False, input_dropout: float=0.0,
        attention_dropout: float=0.0):
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
        self.in_drop = nn.Dropout(input_dropout
            ) if input_dropout > 0.0 else nn.Identity()
        dimensions = [input_size, *attention_units, attention_heads]
        layers = []
        for l in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[l], dimensions[l + 1], bias=
                is_biased))
            layers.append(nn.Tanh() if hidden_activation is None else
                hidden_activation)
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        if attention_dropout > 0.0:
            layers.append(nn.Dropout(attention_dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_activation = nn.Softmax(dim=1
            ) if output_activation is None else output_activation

    def _compute_attention(self, data: torch.Tensor, mask: torch.Tensor
        ) ->torch.Tensor:
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
        attention_logits = self.mlp(data.reshape(-1, dim)).reshape(batch_size,
            num_encs, -1)
        if mask is not None:
            mask = mask.unsqueeze(2).float()
            attention_logits = attention_logits * mask + (1.0 - mask) * -1e+20
        attention = self.output_activation(attention_logits)
        return attention

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor]=None
        ) ->torch.Tensor:
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


class PooledRNNEncoder(Module):
    """Implement an RNNEncoder with additional pooling.

    This class can be used to obtan a single encoded output for
    an input sequence. It also ignores the state of the RNN.

    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int=1,
        rnn_type: str='lstm', dropout: float=0, bidirectional: bool=False,
        layer_norm: bool=False, highway_bias: float=0, rescale: bool=True,
        pooling: str='last') ->None:
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
        warnings.warn(
            'PooledRNNEncoder is deprecated, please use the Pooling                        module in the Embedder object'
            , DeprecationWarning)
        self.pooling = pooling
        self.rnn = RNNEncoder(input_size=input_size, hidden_size=
            hidden_size, n_layers=n_layers, rnn_type=rnn_type, dropout=
            dropout, bidirectional=bidirectional, layer_norm=layer_norm,
            highway_bias=highway_bias, rescale=rescale)
        self.output_size = 2 * hidden_size if bidirectional else hidden_size

    def forward(self, data: Tensor, state: Optional[Tensor]=None,
        padding_mask: Optional[Tensor]=None) ->Tensor:
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
            output = output[(torch.arange(output.size(0)).long()), (lengths -
                1), :]
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

    def __init__(self, **kwargs: Dict[str, Union[Module, torch.nn.Module]]
        ) ->None:
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

    def __init__(self, input_size: int, output_size: int, mlp_layers: int=1,
        mlp_dropout: float=0.0, mlp_hidden_activation: Optional[nn.Module]=
        None, take_log: bool=True) ->None:
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
        self.mlp = MLPEncoder(input_size=input_size, output_size=
            output_size, n_layers=mlp_layers, dropout=mlp_dropout,
            hidden_activation=mlp_hidden_activation, output_activation=softmax)

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


class Transformer(Module):
    """A Transformer model

    User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    """

    def __init__(self, input_size, d_model: int=512, nhead: int=8,
        num_encoder_layers: int=6, num_decoder_layers: int=6,
        dim_feedforward: int=2048, dropout: float=0.1) ->None:
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
        self.encoder = TransformerEncoder(input_size, d_model, nhead,
            dim_feedforward, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(input_size, d_model, nhead,
            dim_feedforward, num_encoder_layers, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask:
        Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None,
        memory_mask: Optional[torch.Tensor]=None, src_key_padding_mask:
        Optional[torch.Tensor]=None, tgt_key_padding_mask: Optional[torch.
        Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None
        ) ->torch.Tensor:
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
            raise RuntimeError(
                'the feature number of src and tgt must be equal to d_model')
        memory = self.encoder(src, mask=src_mask, padding_mask=
            src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=
            memory_mask, padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8,
        num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1
        ) ->None:
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
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
            dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, src: torch.Tensor, memory: Optional[torch.Tensor]=
        None, mask: Optional[torch.Tensor]=None, padding_mask: Optional[
        torch.Tensor]=None) ->torch.Tensor:
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
            output = self.layers[i](output, memory=memory, src_mask=mask,
                padding_mask=padding_mask)
        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers"""

    def __init__(self, input_size: int, d_model: int, nhead: int,
        num_layers: int, dim_feedforward: int=2048, dropout: float=0.1) ->None:
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
        layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
            dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask:
        Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=
        None, padding_mask: Optional[torch.Tensor]=None,
        memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
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
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask, padding_mask=padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)
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

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048,
        dropout: float=0.1) ->None:
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

    def forward(self, src: torch.Tensor, memory: Optional[torch.Tensor]=
        None, src_mask: Optional[torch.Tensor]=None, padding_mask: Optional
        [torch.Tensor]=None) ->torch.Tensor:
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
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
            key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


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

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048,
        dropout: float=0.1) ->None:
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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout
            =dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask:
        Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=
        None, padding_mask: Optional[torch.Tensor]=None,
        memory_key_padding_mask: Optional[torch.Tensor]=None) ->torch.Tensor:
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
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=
            memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerSRU(Module):
    """A Transformer with an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8,
        num_encoder_layers: int=6, num_decoder_layers: int=6,
        dim_feedforward: int=2048, dropout: float=0.1, sru_dropout:
        Optional[float]=None, bidrectional: bool=False, **kwargs: Dict[str,
        Any]) ->None:
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
        self.encoder = TransformerSRUEncoder(input_size, d_model, nhead,
            dim_feedforward, num_encoder_layers, dropout, sru_dropout,
            bidrectional, **kwargs)
        self.decoder = TransformerSRUDecoder(input_size, d_model, nhead,
            dim_feedforward, num_encoder_layers, dropout, sru_dropout, **kwargs
            )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask:
        Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None,
        memory_mask: Optional[torch.Tensor]=None, src_key_padding_mask:
        Optional[torch.Tensor]=None, tgt_key_padding_mask: Optional[torch.
        Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]=None
        ) ->torch.Tensor:
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
            raise RuntimeError(
                'the feature number of src and tgt must be equal to d_model')
        memory, state = self.encoder(src, mask=src_mask, padding_mask=
            src_key_padding_mask)
        output = self.decoder(tgt, memory, state=state, tgt_mask=tgt_mask,
            memory_mask=memory_mask, padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerSRUEncoder(Module):
    """A TransformerSRUEncoder with an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8,
        num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1,
        sru_dropout: Optional[float]=None, bidirectional: bool=False, **
        kwargs: Dict[str, Any]) ->None:
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
        layer = TransformerSRUEncoderLayer(d_model, nhead, dim_feedforward,
            dropout, sru_dropout, bidirectional)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, src: torch.Tensor, state: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch.
        Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
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
            output, new_state = self.layers[i](output, state=input_state,
                src_mask=mask, padding_mask=padding_mask)
            new_states.append(new_state)
        new_states = torch.stack(new_states, dim=0)
        return output.transpose(0, 1), new_states

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerSRUDecoder(Module):
    """A TransformerSRUDecoderwith an SRU replacing the FFN."""

    def __init__(self, input_size: int=512, d_model: int=512, nhead: int=8,
        num_layers: int=6, dim_feedforward: int=2048, dropout: float=0.1,
        sru_dropout: Optional[float]=None, **kwargs: Dict[str, Any]) ->None:
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
        layer = TransformerSRUDecoderLayer(d_model, nhead, dim_feedforward,
            dropout, sru_dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(
            num_layers)])
        self.num_layers = num_layers
        self._reset_parameters()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, state:
        Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None,
        memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[
        torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]
        =None) ->torch.Tensor:
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
            output = self.layers[i](output, memory, state=state[i],
                tgt_mask=tgt_mask, memory_mask=memory_mask, padding_mask=
                padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output.transpose(0, 1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerSRUEncoderLayer(Module):
    """A TransformerSRUEncoderLayer with an SRU replacing the FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048,
        dropout: float=0.1, sru_dropout: Optional[float]=None,
        bidirectional: bool=False, **kwargs: Dict[str, Any]) ->None:
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
        self.sru = SRUCell(d_model, dim_feedforward, dropout, sru_dropout or
            dropout, bidirectional=bidirectional, has_skip_term=False, **kwargs
            )
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, state: Optional[torch.Tensor]=None,
        src_mask: Optional[torch.Tensor]=None, padding_mask: Optional[torch
        .Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
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
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
            key_padding_mask=reversed_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2, state = self.sru(src, state, mask_pad=padding_mask)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, state


class TransformerSRUDecoderLayer(Module):
    """A TransformerSRUDecoderLayer with an SRU replacing the FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048,
        dropout: float=0.1, sru_dropout: Optional[float]=None, **kwargs:
        Dict[str, Any]) ->None:
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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout
            =dropout)
        self.sru = SRUCell(d_model, dim_feedforward, dropout, sru_dropout or
            dropout, bidirectional=False, has_skip_term=False, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, state:
        Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None,
        memory_mask: Optional[torch.Tensor]=None, padding_mask: Optional[
        torch.Tensor]=None, memory_key_padding_mask: Optional[torch.Tensor]
        =None) ->torch.Tensor:
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
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=reversed_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=
            memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2, _ = self.sru(tgt, state, mask_pad=padding_mask)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class IntermediateTorchOnly(torch.nn.Module):

    def __init__(self, component):
        super().__init__()
        self.child = component
        self.linear = torch.nn.Linear(2, 2)


class DummyModel(Module):
    pass


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_asappresearch_flambe(_paritybench_base):
    pass
    def test_000(self):
        self._check(AvgPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(CNNEncoder(*[], **{'input_channels': 4, 'channels': [4, 4]}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(FirstPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(LastPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(LogisticRegression(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(MLPEncoder(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(MixtureOfSoftmax(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(SoftmaxLayer(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(StructuredSelfAttentivePooling(*[], **{'input_size': 4}), [torch.rand([4, 4, 4])], {})

    def test_009(self):
        self._check(SumPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(TransformerDecoder(*[], **{'input_size': 4, 'd_model': 4, 'nhead': 4, 'num_layers': 1}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_011(self):
        self._check(TransformerDecoderLayer(*[], **{'d_model': 4, 'nhead': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(TransformerEncoderLayer(*[], **{'d_model': 4, 'nhead': 4}), [torch.rand([4, 4, 4])], {})

