import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
core = _module
test_config = _module
data = _module
test_graph = _module
test_molecule = _module
test_split = _module
layers = _module
test_common = _module
test_conv = _module
test_pool = _module
test_readout = _module
test_sampler = _module
test_spmm = _module
test_variadic = _module
test = _module
utils = _module
test_comm = _module
test_torch = _module
torchdrug = _module
core = _module
engine = _module
logger = _module
meter = _module
constant = _module
dataloader = _module
dataset = _module
dictionary = _module
feature = _module
graph = _module
molecule = _module
protein = _module
rdkit = _module
draw = _module
datasets = _module
alphafolddb = _module
bace = _module
bbbp = _module
beta_lactamase = _module
binary_localization = _module
bindingdb = _module
cep = _module
chembl_filtered = _module
citeseer = _module
clintox = _module
cora = _module
delaney = _module
enzyme_commission = _module
fb15k = _module
fluorescence = _module
fold = _module
freesolv = _module
gene_ontology = _module
hetionet = _module
hiv = _module
human_ppi = _module
lipophilicity = _module
malaria = _module
moses = _module
muv = _module
opv = _module
pcqm4m = _module
pdbbind = _module
ppi_affinity = _module
proteinnet = _module
pubchem110m = _module
pubmed = _module
qm8 = _module
qm9 = _module
secondary_structure = _module
sider = _module
solubility = _module
stability = _module
subcellular_localization = _module
tox21 = _module
toxcast = _module
uspto50k = _module
wn18 = _module
yeast_ppi = _module
zinc250k = _module
zinc2m = _module
block = _module
common = _module
conv = _module
distribution = _module
flow = _module
functional = _module
embedding = _module
extension = _module
functional = _module
spmm = _module
geometry = _module
function = _module
graph = _module
pool = _module
readout = _module
sampler = _module
metrics = _module
metric = _module
sascorer = _module
models = _module
bert = _module
chebnet = _module
cnn = _module
embedding = _module
esm = _module
flow = _module
gat = _module
gcn = _module
gearnet = _module
gin = _module
infograph = _module
kbgat = _module
lstm = _module
mpnn = _module
neuralfp = _module
neurallp = _module
physicochemical = _module
schnet = _module
statistic = _module
patch = _module
tasks = _module
contact_prediction = _module
generation = _module
pretrain = _module
property_prediction = _module
reasoning = _module
retrosynthesis = _module
task = _module
transforms = _module
transform = _module
utils = _module
comm = _module
decorator = _module
doc = _module
file = _module
io = _module
plot = _module
pretty = _module

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


import math


from torch.nn import functional as F


from itertools import product


from torch import multiprocessing as mp


import re


import types


import inspect


from collections import defaultdict


import logging


from itertools import islice


from torch import distributed as dist


from torch.utils import data as torch_data


import time


import numpy as np


from collections import deque


from collections.abc import Mapping


from collections.abc import Sequence


import warnings


from collections import Sequence


from functools import reduce


from matplotlib import pyplot as plt


from copy import copy


import copy


import functools


from torch.utils import checkpoint


from torch import autograd


import random


from torch import optim


from torch.optim import lr_scheduler as scheduler


from torch.utils.data import dataset


from torch.utils import cpp_extension


import torch.nn.functional as F


from collections import Mapping


class ProteinResNetBlock(nn.Module):
    """
    Convolutional block with residual connection from `Deep Residual Learning for Image Recognition`_.

    .. _Deep Residual Learning for Image Recognition:
        https://arxiv.org/pdf/1512.03385.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation='gelu'):
        super(ProteinResNetBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, input, mask):
        """
        Perform 1D convolutions over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length, dim)`
        """
        identity = input
        input = input * mask
        out = self.conv1(input.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm1(out)
        out = self.activation(out)
        out = out * mask
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm2(out)
        out += identity
        out = self.activation(out)
        return out


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        dropout (float, optional): dropout ratio of attention maps
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, input, mask):
        """
        Perform self attention over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        query = self.query(input).transpose(0, 1)
        key = self.key(input).transpose(0, 1)
        value = self.value(input).transpose(0, 1)
        mask = (~mask.bool()).squeeze(-1)
        output = self.attn(query, key, value, key_padding_mask=mask)[0].transpose(0, 1)
        return output


class ProteinBERTBlock(nn.Module):
    """
    Transformer encoding block from
    `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        num_heads (int): number of attention heads
        attention_dropout (float, optional): dropout ratio of attention maps
        hidden_dropout (float, optional): dropout ratio of hidden features
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, hidden_dim, num_heads, attention_dropout=0, hidden_dropout=0, activation='relu'):
        super(ProteinBERTBlock, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.hidden_dim = hidden_dim
        self.attention = SelfAttentionBlock(input_dim, num_heads, attention_dropout)
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.intermediate = layers.MultiLayerPerceptron(input_dim, hidden_dim, activation=activation)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, input, mask):
        """
        Perform a BERT-block transformation over the input.

        Parameters:
            input (Tensor): input representations of shape `(..., length, dim)`
            mask (Tensor): bool mask of shape `(..., length)`
        """
        x = self.attention(input, mask)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x + input)
        hidden = self.intermediate(x)
        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        output = self.layer_norm2(hidden + x)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation='relu', dropout=0):
        super(MultiLayerPerceptron, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input
        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden


class GaussianSmearing(nn.Module):
    """
    Gaussian smearing from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    There are two modes for Gaussian smearing.

    Non-centered mode:

    .. math::

        \\mu = [0, 1, ..., n], \\sigma = [1, 1, ..., 1]

    Centered mode:

    .. math::

        \\mu = [0, 0, ..., 0], \\sigma = [0, 1, ..., n]

    .. _SchNet\\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        start (int, optional): minimal input value
        stop (int, optional): maximal input value
        num_kernel (int, optional): number of RBF kernels
        centered (bool, optional): centered mode or not
        learnable (bool, optional): learnable gaussian parameters or not
    """

    def __init__(self, start=0, stop=5, num_kernel=100, centered=False, learnable=False):
        super(GaussianSmearing, self).__init__()
        if centered:
            mu = torch.zeros(num_kernel)
            sigma = torch.linspace(start, stop, num_kernel)
        else:
            mu = torch.linspace(start, stop, num_kernel)
            sigma = torch.ones(num_kernel) * (mu[1] - mu[0])
        if learnable:
            self.mu = nn.Parameter(mu)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('sigma', sigma)

    def forward(self, x, y):
        """
        Compute smeared gaussian features between data.

        Parameters:
            x (Tensor): data of shape :math:`(..., d)`
            y (Tensor): data of shape :math:`(..., d)`
        Returns:
            Tensor: features of shape :math:`(..., num\\_kernel)`
        """
        distance = (x - y).norm(2, dim=-1, keepdim=True)
        z = (distance - self.mu) / self.sigma
        prob = torch.exp(-0.5 * z * z)
        return prob


class PairNorm(nn.Module):
    """
    Pair normalization layer proposed in `PairNorm: Tackling Oversmoothing in GNNs`_.

    .. _PairNorm\\: Tackling Oversmoothing in GNNs:
        https://openreview.net/pdf?id=rkecl1rtwB

    Parameters:
        scale_individual (bool, optional): additionally normalize each node representation to have the same L2-norm
    """
    eps = 1e-08

    def __init__(self, scale_individual=False):
        super(PairNorm, self).__init__()
        self.scale_individual = scale_individual

    def forward(self, graph, input):
        """"""
        if graph.batch_size > 1:
            warnings.warn('PairNorm is proposed for a single graph, but now applied to a batch of graphs.')
        x = input.flatten(1)
        x = x - x.mean(dim=0)
        if self.scale_individual:
            output = x / (x.norm(dim=-1, keepdim=True) + self.eps)
        else:
            output = x * x.shape[0] ** 0.5 / (x.norm() + self.eps)
        return output.view_as(input)


class InstanceNorm(nn.modules.instancenorm._InstanceNorm):
    """
    Instance normalization for graphs. This layer follows the definition in
    `GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training`_.

    .. _GraphNorm\\: A Principled Approach to Accelerating Graph Neural Network Training:
        https://arxiv.org/pdf/2009.03294.pdf

    Parameters:
        input_dim (int): input dimension
        eps (float, optional): epsilon added to the denominator
        affine (bool, optional): use learnable affine parameters or not
    """

    def __init__(self, input_dim, eps=1e-05, affine=False):
        super(InstanceNorm, self).__init__(input_dim, eps, affine=affine)

    def forward(self, graph, input):
        """"""
        assert (graph.num_nodes >= 1).all()
        mean = scatter_mean(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        centered = input - mean[graph.node2graph]
        var = scatter_mean(centered ** 2, graph.node2graph, dim=0, dim_size=graph.batch_size)
        std = (var + self.eps).sqrt()
        output = centered / std[graph.node2graph]
        if self.affine:
            output = torch.addcmul(self.bias, self.weight, output)
        return output


class MutualInformation(nn.Module):
    """
    Mutual information estimator from
    `Learning deep representations by mutual information estimation and maximization`_.

    .. _Learning deep representations by mutual information estimation and maximization:
        https://arxiv.org/pdf/1808.06670.pdf

    Parameters:
        input_dim (int): input dimension
        num_mlp_layer (int, optional): number of MLP layers
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, num_mlp_layer=2, activation='relu'):
        super(MutualInformation, self).__init__()
        self.x_mlp = MultiLayerPerceptron(input_dim, [input_dim] * num_mlp_layer, activation=activation)
        self.y_mlp = MultiLayerPerceptron(input_dim, [input_dim] * num_mlp_layer, activation=activation)

    def forward(self, x, y, pair_index=None):
        """"""
        x = self.x_mlp(x)
        y = self.y_mlp(y)
        score = x @ y.t()
        score = score.flatten()
        if pair_index is None:
            assert len(x) == len(y)
            pair_index = torch.arange(len(x), device=x.device).unsqueeze(-1).expand(-1, 2)
        index = pair_index[:, 0] * len(y) + pair_index[:, 1]
        positive = torch.zeros_like(score, dtype=torch.bool)
        positive[index] = 1
        negative = ~positive
        mutual_info = -functional.shifted_softplus(-score[positive]).mean() - functional.shifted_softplus(score[negative]).mean()
        return mutual_info


class Sequential(nn.Sequential):
    """
    Improved sequential container.
    Modules will be called in the order they are passed to the constructor.

    Compared to the vanilla nn.Sequential, this layer additionally supports the following features.

    1. Multiple input / output arguments.

    >>> # layer1 signature: (...) -> (a, b)
    >>> # layer2 signature: (a, b) -> (...)
    >>> layer = layers.Sequential(layer1, layer2)

    2. Global arguments.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: (graph, b) -> c
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    Note the global arguments don't need to be present in every layer.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: b -> c
    >>> # layer3 signature: (graph, c) -> d
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    3. Dict outputs.

    >>> # layer1 signature: a -> {"b": b, "c": c}
    >>> # layer2 signature: b -> d
    >>> layer = layers.Sequential(layer1, layer2, allow_unused=True)

    When dict outputs are used with global arguments, the global arguments can be explicitly
    overwritten by any layer outputs.

    >>> # layer1 signature: (graph, a) -> {"graph": graph, "b": b}
    >>> # layer2 signature: (graph, b) -> c
    >>> # layer2 takes in the graph output by layer1
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))
    """

    def __init__(self, *args, global_args=None, allow_unused=False):
        super(Sequential, self).__init__(*args)
        if global_args is not None:
            self.global_args = set(global_args)
        else:
            self.global_args = {}
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        """"""
        global_kwargs = {}
        for i, module in enumerate(self._modules.values()):
            sig = inspect.signature(module.forward)
            parameters = list(sig.parameters.values())
            param_names = [param.name for param in parameters]
            j = 0
            for name in param_names:
                if j == len(args):
                    break
                if name in kwargs:
                    continue
                if name in global_kwargs and name not in kwargs:
                    kwargs[name] = global_kwargs[name]
                    continue
                kwargs[name] = args[j]
                j += 1
            if self.allow_unused:
                param_names = set(param_names)
                kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            if j < len(args):
                raise TypeError('too many positional arguments')
            output = module(**kwargs)
            global_kwargs.update({k: v for k, v in kwargs.items() if k in self.global_args})
            args = []
            kwargs = {}
            if isinstance(output, dict):
                kwargs.update(output)
            elif isinstance(output, Sequence):
                args += list(output)
            else:
                args.append(output)
        return output


class SinusoidalPositionEmbedding(nn.Module):
    """
    Positional embedding based on sine and cosine functions, proposed in `Attention Is All You Need`_.

    .. _Attention Is All You Need:
        https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        output_dim (int): output dimension
    """

    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        inverse_frequency = 1 / 10000 ** (torch.arange(0.0, output_dim, 2.0) / output_dim)
        self.register_buffer('inverse_frequency', inverse_frequency)

    def forward(self, input):
        """"""
        positions = torch.arange(input.shape[1] - 1, -1, -1.0, dtype=input.dtype, device=input.device)
        sinusoidal_input = torch.outer(positions, self.inverse_frequency)
        position_embedding = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos()], -1)
        return position_embedding


class MessagePassingBase(nn.Module):
    """
    Base module for message passing.

    Any custom message passing module should be derived from this class.
    """
    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.

        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update):
        """
        Combine node input and node update.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output


class GraphConv(MessagePassingBase):
    """
    Graph convolution operator from `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation='relu'):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= degree_in[node_in].sqrt() + 1e-10
        return message

    def aggregate(self, graph, message):
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        edge_weight = edge_weight / ((degree_in[node_in] * degree_out[node_out]).sqrt() + 1e-10)
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = torch.cat([self.edge_linear(edge_input), torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            edge_weight = edge_weight.unsqueeze(-1)
            node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update += edge_update
        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GraphAttentionConv(MessagePassingBase):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """
    eps = 1e-10

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True, batch_norm=False, activation='relu'):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if output_dim % num_head != 0:
            raise ValueError('Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`' % (output_dim, num_head))
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, output_dim)
        else:
            self.edge_linear = None
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode='fan_in')

    def message(self, graph, input):
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input)
        key = torch.stack([hidden[node_in], hidden[node_out]], dim=-1)
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.output_dim, device=graph.device)])
            key += edge_input.unsqueeze(-1)
        key = key.view(-1, *self.query.shape)
        weight = torch.einsum('hd, nhd -> nh', self.query, key)
        weight = self.leaky_relu(weight)
        weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
        attention = weight.exp() * edge_weight
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
        attention = attention / (normalizer + self.eps)
        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GraphIsomorphismConv(MessagePassingBase):
    """
    Graph isomorphism convolution operator from `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dims (list of int, optional): hidden dimensions
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False, batch_norm=False, activation='relu'):
        super(GraphIsomorphismConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        eps = torch.tensor([eps], dtype=torch.float32)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer('eps', eps)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(input_dim, list(hidden_dims) + [output_dim], activation)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0, dim_size=graph.num_node)
            update += edge_update
        return update

    def combine(self, input, update):
        output = self.mlp((1 + self.eps) * input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.

    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation='relu'):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) / (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) + self.eps)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight, (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0, dim_size=graph.num_node * graph.num_relation)
            update += edge_update
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class NeuralFingerprintConv(MessagePassingBase):
    """
    Graph neural network operator from `Convolutional Networks on Graphs for Learning Molecular Fingerprints`_.

    Note this operator doesn't include the sparsifying step of the original paper.

    .. _Convolutional Networks on Graphs for Learning Molecular Fingerprints:
        https://arxiv.org/pdf/1509.09292.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation='relu'):
        super(NeuralFingerprintConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0, dim_size=graph.num_node)
            update += edge_update
        return update

    def combine(self, input, update):
        output = self.linear(input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class ContinuousFilterConv(MessagePassingBase):
    """
    Continuous filter operator from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    .. _SchNet\\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dim (int, optional): hidden dimension. By default, same as :attr:`output_dim`
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dim=None, cutoff=5, num_gaussian=100, batch_norm=False, activation='shifted_softplus'):
        super(ContinuousFilterConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        if hidden_dim is None:
            hidden_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rbf = layers.RBF(stop=cutoff, num_kernel=num_gaussian)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if activation == 'shifted_softplus':
            self.activation = functional.shifted_softplus
        elif isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.rbf_layer = nn.Linear(num_gaussian, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, hidden_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        position = graph.node_position
        message = self.input_layer(input)[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        message *= weight
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        position = graph.node_position
        rbf_weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        indices = torch.stack([node_out, node_in, torch.arange(graph.num_edge, device=graph.device)])
        adjacency = utils.sparse_coo_tensor(indices, graph.edge_weight, (graph.num_node, graph.num_node, graph.num_edge))
        update = functional.generalized_rspmm(adjacency, rbf_weight, self.input_layer(input))
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1) * rbf_weight
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0, dim_size=graph.num_node)
            update += edge_update
        return update

    def combine(self, input, update):
        output = self.output_layer(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class MessagePassing(MessagePassingBase):
    """
    Message passing operator from `Neural Message Passing for Quantum Chemistry`_.

    This implements the edge network variant in the original paper.

    .. _Neural Message Passing for Quantum Chemistry:
        https://arxiv.org/pdf/1704.01212.pdf

    Parameters:
        input_dim (int): input dimension
        edge_input_dim (int): dimension of edge features
        hidden_dims (list of int, optional): hidden dims of edge network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, edge_input_dim, hidden_dims=None, batch_norm=False, activation='relu'):
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if hidden_dims is None:
            hidden_dims = []
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.edge_mlp = layers.MLP(edge_input_dim, list(hidden_dims) + [input_dim * input_dim], activation)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        transform = self.edge_mlp(graph.edge_feature.float()).view(-1, self.input_dim, self.input_dim)
        if graph.num_edge:
            message = torch.einsum('bed, bd -> be', transform, input[node_in])
        else:
            message = torch.zeros(0, self.input_dim, device=graph.device)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class ChebyshevConv(MessagePassingBase):
    """
    Chebyshev spectral graph convolution operator from
    `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering`_.

    .. _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering:
        https://arxiv.org/pdf/1606.09375.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        k (int, optional): number of Chebyshev polynomials.
            This also corresponds to the radius of the receptive field.
        hidden_dims (list of int, optional): hidden dims of edge network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, k=1, batch_norm=False, activation='relu'):
        super(ChebyshevConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.edge_input_dim = edge_input_dim
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.linear = nn.Linear((k + 1) * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        degree_in = graph.degree_in.unsqueeze(-1)
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        message /= degree_in[node_in].sqrt() + 1e-10
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1)
        update = -scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = -graph.edge_weight / ((graph.degree_in[node_in] * graph.degree_out[node_out]).sqrt() + 1e-10)
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0, dim_size=graph.num_node)
            update += edge_update
        return update

    def forward(self, graph, input):
        bases = [input]
        for i in range(self.k):
            x = super(ChebyshevConv, self).forward(graph, bases[-1])
            if i > 0:
                x = 2 * x - bases[-2]
            bases.append(x)
        bases = torch.cat(bases, dim=-1)
        output = self.linear(bases)
        if self.batch_norm:
            x = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def combine(self, input, update):
        output = input + update
        return output


class GeometricRelationalGraphConv(RelationalGraphConv):
    """
    Geometry-aware relational graph convolution operator from
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation='relu'):
        super(GeometricRelationalGraphConv, self).__init__(input_dim, output_dim, num_relation, edge_input_dim, batch_norm, activation)

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)
        return update

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation
        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight, (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0, dim_size=graph.num_node * graph.num_relation)
            update += edge_update
        return update.view(graph.num_node, self.num_relation * self.input_dim)


class IndependentGaussian(nn.Module):
    """
    Independent Gaussian distribution.

    Parameters:
        mu (Tensor): mean of shape :math:`(N,)`
        sigma2 (Tensor): variance of shape :math:`(N,)`
        learnable (bool, optional): learnable parameters or not
    """

    def __init__(self, mu, sigma2, learnable=False):
        super(IndependentGaussian, self).__init__()
        if learnable:
            self.mu = nn.Parameter(torch.as_tensor(mu))
            self.sigma2 = nn.Parameter(torch.as_tensor(sigma2))
        else:
            self.register_buffer('mu', torch.as_tensor(mu))
            self.register_buffer('sigma2', torch.as_tensor(sigma2))
        self.dim = len(mu)

    def forward(self, input):
        """
        Compute the likelihood of input data.

        Parameters:
            input (Tensor): input data of shape :math:`(..., N)`
        """
        log_likelihood = -0.5 * (math.log(2 * math.pi) + self.sigma2.log() + (input - self.mu) ** 2 / self.sigma2)
        return log_likelihood

    def sample(self, *size):
        """
        Draw samples from the distribution.

        Parameters:
            size (tuple of int): shape of the samples
        """
        if len(size) == 1 and isinstance(size[0], Sequence):
            size = size[0]
        size = list(size) + [self.dim]
        sample = torch.randn(size, device=self.mu.device) * self.sigma2.sqrt() + self.mu
        return sample


class ConditionalFlow(nn.Module):
    """
    Conditional flow transformation from `Masked Autoregressive Flow for Density Estimation`_.

    .. _Masked Autoregressive Flow for Density Estimation:
        https://arxiv.org/pdf/1705.07057.pdf

    Parameters:
        input_dim (int): input & output dimension
        condition_dim (int): condition dimension
        hidden_dims (list of int, optional): hidden dimensions
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, condition_dim, hidden_dims=None, activation='relu'):
        super(ConditionalFlow, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(condition_dim, list(hidden_dims) + [input_dim * 2], activation)
        self.rescale = nn.Parameter(torch.zeros(1))

    def forward(self, input, condition):
        """
        Transform data into latent representations.

        Parameters:
            input (Tensor): input representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): latent representations, log-likelihood of the transformation
        """
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = F.tanh(scale) * self.rescale
        output = (input + bias) * scale.exp()
        log_det = scale
        return output, log_det

    def reverse(self, latent, condition):
        """
        Transform latent representations into data.

        Parameters:
            latent (Tensor): latent representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): input representations, log-likelihood of the transformation
        """
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = F.tanh(scale) * self.rescale
        output = latent / scale.exp() - bias
        log_det = scale
        return output, log_det


class DiffPool(nn.Module):
    """
    Differentiable pooling operator from `Hierarchical Graph Representation Learning with Differentiable Pooling`_

    .. _Hierarchical Graph Representation Learning with Differentiable Pooling:
        https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf

    Parameter
        input_dim (int): input dimension
        output_node (int): number of nodes after pooling
        feature_layer (Module, optional): graph convolution layer for embedding
        pool_layer (Module, optional): graph convolution layer for pooling assignment
        loss_weight (float, optional): weight of entropy regularization
        zero_diagonal (bool, optional): remove self loops in the pooled graph or not
        sparse (bool, optional): use sparse assignment or not
    """
    tau = 1
    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=False, sparse=False):
        super(DiffPool, self).__init__()
        self.input_dim = input_dim
        self.output_dim = feature_layer.output_dim
        self.output_node = output_node
        self.feature_layer = feature_layer
        self.pool_layer = pool_layer
        self.loss_weight = loss_weight
        self.zero_diagonal = zero_diagonal
        self.sparse = sparse
        if pool_layer is not None:
            self.linear = nn.Linear(pool_layer.output_dim, output_node)
        else:
            self.linear = nn.Linear(input_dim, output_node)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            (PackedGraph, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = input
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)
        x = input
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = F.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = F.softmax(x, dim=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)
        if all_loss is not None:
            prob = scatter_mean(assignment, graph.node2graph, dim=0, dim_size=graph.batch_size)
            entropy = -(prob * (prob + self.eps).log()).sum(dim=-1)
            entropy = entropy.mean()
            metric['assignment entropy'] = entropy
            if self.loss_weight > 0:
                all_loss -= entropy * self.loss_weight
        if self.zero_diagonal:
            edge_list = new_graph.edge_list[:, :2]
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            new_graph = new_graph.edge_mask(~is_diagonal)
        return new_graph, output, assignment

    def dense_pool(self, graph, input, assignment):
        node_in, node_out = graph.edge_list.t()[:2]
        x = graph.edge_weight.unsqueeze(-1) * assignment[node_out]
        x = scatter_add(x, node_in, dim=0, dim_size=graph.num_node)
        x = torch.einsum('np, nq -> npq', assignment, x)
        adjacency = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
        x = torch.einsum('na, nd -> nad', assignment, input)
        output = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size).flatten(0, 1)
        index = torch.arange(self.output_node, device=graph.device).expand(len(graph), self.output_node, -1)
        edge_list = torch.stack([index.transpose(-1, -2), index], dim=-1).flatten(0, -2)
        edge_weight = adjacency.flatten()
        if isinstance(graph, data.PackedGraph):
            num_nodes = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node
            num_edges = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node ** 2
            graph = data.PackedGraph(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        else:
            graph = data.Graph(edge_list, edge_weight=edge_weight, num_node=self.output_node)
        return graph, output

    def sparse_pool(self, graph, input, assignment):
        assignment = assignment.argmax(dim=-1)
        edge_list = graph.edge_list[:, :2]
        edge_list = assignment[edge_list]
        pooled_node = graph.node2graph * self.output_node + assignment
        output = scatter_add(input, pooled_node, dim=0, dim_size=graph.batch_size * self.output_node)
        edge_weight = graph.edge_weight
        if isinstance(graph, data.PackedGraph):
            num_nodes = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node
            num_edges = graph.num_edges
            graph = data.PackedGraph(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        else:
            graph = data.Graph(edge_list, edge_weight=edge_weight, num_node=self.output_node)
        return graph, output


class MinCutPool(DiffPool):
    """
    Min cut pooling operator from `Spectral Clustering with Graph Neural Networks for Graph Pooling`_

    .. _Spectral Clustering with Graph Neural Networks for Graph Pooling:
        http://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf

    Parameters:
        input_dim (int): input dimension
        output_node (int): number of nodes after pooling
        feature_layer (Module, optional): graph convolution layer for embedding
        pool_layer (Module, optional): graph convolution layer for pooling assignment
        loss_weight (float, optional): weight of entropy regularization
        zero_diagonal (bool, optional): remove self loops in the pooled graph or not
        sparse (bool, optional): use sparse assignment or not
    """
    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=True, sparse=False):
        super(MinCutPool, self).__init__(input_dim, output_node, feature_layer, pool_layer, loss_weight, zero_diagonal, sparse)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            (PackedGraph, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = input
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)
        x = input
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = F.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = F.softmax(x, dim=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)
        if all_loss is not None:
            edge_list = new_graph.edge_list
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            num_intra = scatter_add(new_graph.edge_weight[is_diagonal], new_graph.edge2graph[is_diagonal], dim=0, dim_size=new_graph.batch_size)
            x = torch.einsum('na, n, nc -> nac', assignment, graph.degree_in, assignment)
            x = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
            num_all = torch.einsum('baa -> b', x)
            cut_loss = (1 - num_intra / (num_all + self.eps)).mean()
            metric['normalized cut loss'] = cut_loss
            x = torch.einsum('na, nc -> nac', assignment, assignment)
            x = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
            x = x / x.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
            x = x - torch.eye(self.output_node, device=x.device) / self.output_node ** 0.5
            regularization = x.flatten(-2).norm(dim=-1).mean()
            metric['orthogonal regularization'] = regularization
            if self.loss_weight > 0:
                all_loss += (cut_loss + regularization) * self.loss_weight
        if self.zero_diagonal:
            edge_list = new_graph.edge_list[:, :2]
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            new_graph = new_graph.edge_mask(~is_diagonal)
        return new_graph, output, assignment


class Readout(nn.Module):

    def __init__(self, type='node'):
        super(Readout, self).__init__()
        self.type = type

    def get_index2graph(self, graph):
        if self.type == 'node':
            input2graph = graph.node2graph
        elif self.type == 'edge':
            input2graph = graph.edge2graph
        elif self.type == 'residue':
            input2graph = graph.residue2graph
        else:
            raise ValueError('Unknown input type `%s` for readout functions' % self.type)
        return input2graph


class MeanReadout(Readout):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_mean(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class SumReadout(Readout):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_add(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class MaxReadout(Readout):
    """Max readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_max(input, input2graph, dim=0, dim_size=graph.batch_size)[0]
        return output


class AttentionReadout(Readout):
    """Attention readout operator over graphs with variadic sizes."""

    def __init__(self, input_dim, type='node'):
        super(AttentionReadout, self).__init__(type)
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, graph, input):
        index2graph = self.get_index2graph(graph)
        weight = self.linear(input)
        attention = scatter_softmax(weight, index2graph, dim=0)
        output = scatter_add(attention * input, index2graph, dim=0, dim_size=graph.batch_size)
        return output


class Softmax(Readout):
    """Softmax operator over graphs with variadic sizes."""
    eps = 1e-10

    def forward(self, graph, input):
        """
        Perform softmax over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node logits

        Returns:
            Tensor: node probabilities
        """
        input2graph = self.get_index2graph(graph)
        x = input - scatter_max(input, input2graph, dim=0, dim_size=graph.batch_size)[0][input2graph]
        x = x.exp()
        normalizer = scatter_add(x, input2graph, dim=0, dim_size=graph.batch_size)[input2graph]
        return x / (normalizer + self.eps)


class Sort(Readout):
    """
    Sort operator over graphs with variadic sizes.

    Parameters:
        descending (bool, optional): use descending sort order or not
    """

    def __init__(self, type='node', descending=False):
        super(Sort, self).__init__(type)
        self.descending = descending

    def forward(self, graph, input):
        """
        Perform sort over graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node values

        Returns:
            (Tensor, LongTensor): sorted values, sorted indices
        """
        input2graph = self.get_index2graph(graph)
        step = input.max(dim=0) - input.min(dim=0) + 1
        if self.descending:
            step = -step
        x = input + input2graph * step
        sorted, index = x.sort(dim=0, descending=self.descending)
        sorted = sorted - input2graph * step
        return sorted, index


class Set2Set(Readout):
    """
    Set2Set operator from `Order Matters: Sequence to sequence for sets`_.

    .. _Order Matters\\: Sequence to sequence for sets:
        https://arxiv.org/pdf/1511.06391.pdf

    Parameters:
        input_dim (int): input dimension
        num_step (int, optional): number of process steps
        num_lstm_layer (int, optional): number of LSTM layers
    """

    def __init__(self, input_dim, type='node', num_step=3, num_lstm_layer=1):
        super(Set2Set, self).__init__(type)
        self.input_dim = input_dim
        self.output_dim = self.input_dim * 2
        self.num_step = num_step
        self.lstm = nn.LSTM(input_dim * 2, input_dim, num_lstm_layer)
        self.softmax = Softmax(type)

    def forward(self, graph, input):
        """
        Perform Set2Set readout over graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        hx = (torch.zeros(self.lstm.num_layers, graph.batch_size, self.lstm.hidden_size, device=input.device),) * 2
        query_star = torch.zeros(graph.batch_size, self.output_dim, device=input.device)
        for i in range(self.num_step):
            query, hx = self.lstm(query_star.unsqueeze(0), hx)
            query = query.squeeze(0)
            product = torch.einsum('bd, bd -> b', query[input2graph], input)
            attention = self.softmax(graph, product)
            output = scatter_add(attention.unsqueeze(-1) * input, input2graph, dim=0, dim_size=graph.batch_size)
            query_star = torch.cat([query, output], dim=-1)
        return query_star


class NodeSampler(nn.Module):
    """
    Node sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT\\: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Parameters:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep
    """

    def __init__(self, budget=None, ratio=None):
        super(NodeSampler, self).__init__()
        if budget is None and ratio is None:
            raise ValueError('At least one of `budget` and `ratio` should be provided')
        self.budget = budget
        self.ratio = ratio

    def forward(self, graph):
        """
        Sample a subgraph from the graph.

        Parameters:
            graph (Graph): graph(s)
        """
        num_sample = graph.num_node
        if self.budget:
            num_sample = min(num_sample, self.budget)
        if self.ratio:
            num_sample = min(num_sample, int(self.ratio * graph.num_node))
        prob = scatter_add(graph.edge_weight ** 2, graph.edge_list[:, 1], dim_size=graph.num_node)
        prob /= prob.mean()
        index = functional.multinomial(prob, num_sample)
        new_graph = graph.node_mask(index)
        node_out = new_graph.edge_list[:, 1]
        new_graph._edge_weight /= num_sample * prob[node_out] / graph.num_node
        return new_graph


class EdgeSampler(nn.Module):
    """
    Edge sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT\\: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Parameters:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep
    """

    def __init__(self, budget=None, ratio=None):
        super(EdgeSampler, self).__init__()
        if budget is None and ratio is None:
            raise ValueError('At least one of `budget` and `ratio` should be provided')
        self.budget = budget
        self.ratio = ratio

    def forward(self, graph):
        """
        Sample a subgraph from the graph.

        Parameters:
            graph (Graph): graph(s)
        """
        node_in, node_out = graph.edge_list.t()[:2]
        num_sample = graph.num_edge
        if self.budget:
            num_sample = min(num_sample, self.budget)
        if self.ratio:
            num_sample = min(num_sample, int(self.ratio * graph.num_edge))
        prob = 1 / graph.degree_out[node_out] + 1 / graph.degree_in[node_in]
        prob = prob / prob.mean()
        index = functional.multinomial(prob, num_sample)
        new_graph = graph.edge_mask(index)
        new_graph._edge_weight /= num_sample * prob[index] / graph.num_edge
        return new_graph


class PatchedModule(nn.Module):

    def __init__(self):
        super(PatchedModule, self).__init__()

    def graph_state_dict(self, destination, prefix, local_metadata):
        local_graphs = []
        for name, param in self._buffers.items():
            if isinstance(param, data.Graph):
                local_graphs.append(name)
                destination.pop(prefix + name)
                for t_name, tensor in zip(data.Graph._tensor_names, param.to_tensors()):
                    if tensor is not None:
                        destination[prefix + name + '.' + t_name] = tensor
        if local_graphs:
            local_metadata['graph'] = local_graphs
        return destination

    @classmethod
    def load_graph_state_dict(cls, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if 'graph' not in local_metadata:
            return
        for name in local_metadata['graph']:
            tensors = []
            for t_name in data.Graph._tensor_names:
                key = prefix + name + '.' + t_name
                input_tensor = state_dict.get(key, None)
                tensors.append(input_tensor)
            try:
                state_dict[prefix + name] = data.Graph.from_tensors(tensors)
                None
            except:
                error_msgs.append("Can't construct Graph `%s` from tensors in the state dict" % key)
        return state_dict

    @property
    def device(self):
        try:
            tensor = next(self.parameters())
        except StopIteration:
            tensor = next(self.buffers())
        return tensor.device

    def register_buffer(self, name, tensor, persistent=True):
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError('ScriptModule does not support non-persistent buffers')
        if '_buffers' not in self.__dict__:
            raise AttributeError('cannot assign buffer before Module.__init__() call')
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError('buffer name should be a string. Got {}'.format(torch.typename(name)))
        elif '.' in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == '':
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor) and not isinstance(tensor, data.Graph):
            raise TypeError("cannot assign '{}' object to buffer '{}' (torch.Tensor, torchdrug.data.Graph or None required)".format(torch.typename(tensor), name))
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)


class PatchedDistributedDataParallel(nn.parallel.DistributedDataParallel):

    def _distributed_broadcast_coalesced(self, tensors, buffer_size, *args, **kwargs):
        new_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                new_tensors.append(tensor)
        if new_tensors:
            dist._broadcast_coalesced(self.process_group, new_tensors, buffer_size, *args, **kwargs)


class Task(nn.Module):
    _option_members = set()

    def _standarize_option(self, x, name):
        if x is None:
            x = {}
        elif isinstance(x, str):
            x = {x: 1}
        elif isinstance(x, Sequence):
            x = dict.fromkeys(x, 1)
        elif not isinstance(x, Mapping):
            raise ValueError('Invalid value `%s` for option member `%s`' % (x, name))
        return x

    def __setattr__(self, key, value):
        if key in self._option_members:
            value = self._standarize_option(value, key)
        super(Task, self).__setattr__(key, value)

    def preprocess(self, train_set, valid_set, test_set):
        pass

    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    def predict(self, batch, all_loss=None, metric=None):
        raise NotImplementedError

    def target(self, batch):
        raise NotImplementedError

    def evaluate(self, pred, target):
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GaussianSmearing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IndependentGaussian,
     lambda: ([], {'mu': [4, 4], 'sigma2': 4}),
     lambda: ([torch.rand([4, 4, 4, 2])], {}),
     True),
    (MultiLayerPerceptron,
     lambda: ([], {'input_dim': 4, 'hidden_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProteinBERTBlock,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4])], {}),
     False),
    (ProteinResNetBlock,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (SelfAttentionBlock,
     lambda: ([], {'hidden_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4])], {}),
     False),
    (SinusoidalPositionEmbedding,
     lambda: ([], {'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DeepGraphLearning_torchdrug(_paritybench_base):
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

