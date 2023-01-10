import sys
_module = sys.modules[__name__]
del sys
chemicalx = _module
compat = _module
constants = _module
data = _module
batchgenerator = _module
contextfeatureset = _module
datasetloader = _module
drugfeatureset = _module
drugpairbatch = _module
labeledtriples = _module
utils = _module
loss = _module
models = _module
base = _module
caster = _module
deepddi = _module
deepdds = _module
deepdrug = _module
deepsynergy = _module
epgcnds = _module
gcnbmp = _module
matchmaker = _module
mhcaddi = _module
mrgnn = _module
ssiddi = _module
pipeline = _module
utils = _module
version = _module
drugbank_ddi_cleaner = _module
twosides_cleaner = _module
conf = _module
caster_example = _module
deepddi_example = _module
deepdds_example = _module
deepdrug_example = _module
deepsynergy_example = _module
deepsynergy_synergy_example = _module
epgcnds_example = _module
gcnbmp_example = _module
matchmaker_example = _module
mhcaddi_example = _module
mrgnn_example = _module
ssiddi_example = _module
setup = _module
test_batching = _module
test_compat = _module
test_dataset = _module
test_datastructures = _module
test_models = _module
test_utils = _module

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


from torch.types import Device


import math


from typing import Iterable


from typing import Iterator


from typing import Optional


from typing import Sequence


import numpy as np


import pandas as pd


from collections import UserDict


from typing import Mapping


from abc import ABC


from abc import abstractmethod


from functools import lru_cache


from itertools import chain


from typing import ClassVar


from typing import Dict


from typing import Tuple


from typing import cast


from typing import Union


from typing import TypeVar


from torch.nn.modules.loss import _Loss


from torch import nn


from typing import List


from torch.nn.functional import normalize


from torch.fft import fft


from torch.fft import ifft


from torch.nn import functional as F


import functools


import torch.nn as nn


from typing import Any


import torch.nn.functional


from torch.nn import LayerNorm


from torch.nn.modules.container import ModuleList


import collections.abc


import time


from typing import Type


from sklearn.metrics import mean_absolute_error


from sklearn.metrics import mean_squared_error


from sklearn.metrics import roc_auc_score


from torch.optim.optimizer import Optimizer


import logging


import torch.cuda


import inspect


class CASTERSupervisedLoss(_Loss):
    """An implementation of the custom loss function for the supervised learning stage of the CASTER algorithm.

    The algorithm is described in [huang2020]_. The loss function combines three separate loss functions on
    different model outputs: class prediction loss, input reconstruction loss, and dictionary projection loss.

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702–709.
    """

    def __init__(self, recon_loss_coeff: float=0.1, proj_coeff: float=0.1, lambda1: float=0.01, lambda2: float=0.1):
        """
        Initialize the custom loss function for the supervised learning stage of the CASTER algorithm.

        :param recon_loss_coeff: coefficient for the reconstruction loss
        :param proj_coeff: coefficient for the projection loss
        :param lambda1: regularization coefficient for the projection loss
        :param lambda2: regularization coefficient for the augmented projection loss
        """
        super().__init__(reduction='none')
        self.recon_loss_coeff = recon_loss_coeff
        self.proj_coeff = proj_coeff
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = torch.nn.BCELoss()

    def forward(self, x: Tuple[torch.FloatTensor, ...], target: torch.Tensor) ->torch.FloatTensor:
        """Perform a forward pass of the loss calculation for the supervised learning stage of the CASTER algorithm.

        :param x: a tuple of tensors returned by the model forward pass (see CASTER.forward() method)
        :param target: target labels
        :return: combined loss value
        """
        score, recon, code, dictionary_features_latent, drug_pair_features_latent, drug_pair_features = x
        batch_size, _ = drug_pair_features.shape
        loss_prediction = self.loss(score, target.float())
        loss_reconstruction = self.recon_loss_coeff * self.loss(recon, drug_pair_features)
        loss_projection = self.proj_coeff * (torch.norm(drug_pair_features_latent - torch.matmul(code, dictionary_features_latent)) + self.lambda1 * torch.sum(torch.abs(code)) / batch_size + self.lambda2 * torch.norm(dictionary_features_latent, p='fro') / batch_size)
        loss = loss_prediction + loss_reconstruction + loss_projection
        return loss


class Highway(nn.Module):
    """The Highway update layer from [srivastava2015]_.

    .. [srivastava2015] Srivastava, R. K., *et al.* (2015).
       `Highway Networks <http://arxiv.org/abs/1505.00387>`_.
       *arXiv*, 1505.00387.
    """

    def __init__(self, input_size: int, prev_input_size: int):
        """Instantiate the Highway update layer.

        :param input_size: Current representation size.
        :param prev_input_size: Size of the representation obtained by the previous convolutional layer.
        """
        super().__init__()
        total_size = input_size + prev_input_size
        self.proj = nn.Linear(total_size, input_size)
        self.transform = nn.Linear(total_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, current: torch.Tensor, previous: torch.Tensor) ->torch.Tensor:
        """Compute the gated update.

        :param current: Current layer node representations.
        :param previous: Previous layer node representations.
        :returns: The highway-updated inputs.
        """
        concat_inputs = torch.cat((current, previous), 1)
        proj_result = F.relu(self.proj(concat_inputs))
        proj_gate = F.sigmoid(self.transform(concat_inputs))
        gated = proj_gate * proj_result + (1 - proj_gate) * current
        return gated


class AttentionPooling(nn.Module):
    """The attention pooling layer from [chen2020]_."""

    def __init__(self, molecule_channels: int, hidden_channels: int):
        """Instantiate the attention pooling layer.

        :param molecule_channels: Input node features.
        :param hidden_channels: Final node representation.
        """
        super(AttentionPooling, self).__init__()
        total_features_channels = molecule_channels + hidden_channels
        self.lin = nn.Linear(total_features_channels, hidden_channels)
        self.last_rep = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, input_rep: torch.Tensor, final_rep: torch.Tensor, graph_index: torch.Tensor) ->torch.Tensor:
        """
        Compute an attention-based readout using the input and output layers of the RGCN encoder for one molecule.

        :param input_rep: Input nodes representations.
        :param final_rep: Final nodes representations.
        :param graph_index: Node to graph readout index.
        :returns: Graph-level representation.
        """
        att = torch.sigmoid(self.lin(torch.cat((input_rep, final_rep), dim=1)))
        g = att.mul(self.last_rep(final_rep))
        g = scatter_add(g, graph_index, dim=0)
        return g


def circular_correlation(left: torch.FloatTensor, right: torch.FloatTensor) ->torch.FloatTensor:
    """Compute the circular correlation of two vectors ``left`` and ``right`` via their Fast Fourier Transforms.

    :param left: the left vector
    :param right: the right vector
    :returns: Joint representation by circular correlation.
    """
    left_x_cfft = torch.conj(fft(left))
    right_x_fft = fft(right)
    circ_corr = ifft(torch.mul(left_x_cfft, right_x_fft))
    return circ_corr.real


class MessagePassing(nn.Module):
    """A network for creating node representations based on internal message passing."""

    def __init__(self, node_channels: int, edge_channels: int, hidden_channels: int, dropout: float=0.5):
        """Instantiate the MessagePassing network.

        :param node_channels: Dimension of node features
        :param edge_channels: Dimension of edge features
        :param hidden_channels: Dimension of hidden layer
        :param dropout: Dropout probability
        """
        super().__init__()
        self.node_projection = nn.Sequential(nn.Linear(node_channels, hidden_channels, bias=False), nn.Dropout(dropout))
        self.edge_projection = nn.Sequential(nn.Linear(edge_channels, hidden_channels), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU(), nn.Dropout(dropout))

    def forward(self, nodes: torch.FloatTensor, edges: torch.FloatTensor, segmentation_index: torch.LongTensor, index: torch.LongTensor) ->torch.FloatTensor:
        """Calculate forward pass of message passing network.

        :param nodes: Node feature matrix.
        :param edges: Edge feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :param index: List of node indices from where edges in the molecular graph end.
        :returns: Messages between nodes.
        """
        edges = self.edge_projection(edges)
        messages = self.node_projection(nodes)
        messages = self.message_composing(messages, edges, index)
        messages = self.message_aggregation(nodes, messages, segmentation_index)
        return messages

    def message_composing(self, messages: torch.FloatTensor, edges: torch.FloatTensor, index: torch.LongTensor) ->torch.FloatTensor:
        """Compose message based by elementwise multiplication of edge and node projections.

        :param messages: Message matrix.
        :param edges: Edge feature matrix.
        :param index: Global node indexing.
        :returns: Composed messages.
        """
        messages = messages.index_select(0, index)
        messages = messages * edges
        return messages

    def message_aggregation(self, nodes: torch.FloatTensor, messages: torch.FloatTensor, segmentation_index: torch.LongTensor) ->torch.FloatTensor:
        """Aggregate the messages.

        :param nodes: Node feature matrix.
        :param messages: Message feature matrix.
        :param segmentation_index: List of node indices from where edges in the molecular graph start.
        :returns: Messages between nodes.
        """
        messages = torch.zeros_like(nodes).index_add(0, segmentation_index, messages)
        return messages


def segment_max(logit: torch.FloatTensor, number_of_segments: torch.LongTensor, segmentation_index: torch.LongTensor, index: torch.LongTensor):
    """Segmentation maximal index finder.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :param index: Global index
    :returns: Largest index in each segmentation.
    """
    max_number_of_segments = index.max().item() + 1
    segmentation_max = logit.new_full((number_of_segments, max_number_of_segments), -np.inf)
    segmentation_max = segmentation_max.index_put_((segmentation_index, index), logit).max(dim=1)[0]
    return segmentation_max[segmentation_index]


def segment_sum(logit: torch.FloatTensor, number_of_segments: torch.LongTensor, segmentation_index: torch.LongTensor):
    """Segmentation sum calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segments.
    :returns: Sum of logits on segments.
    """
    norm = logit.new_zeros(number_of_segments).index_add(0, segmentation_index, logit)
    return norm[segmentation_index]


def segment_softmax(logit: torch.FloatTensor, number_of_segments: torch.LongTensor, segmentation_index: torch.LongTensor, index: torch.LongTensor, temperature: torch.FloatTensor):
    """Segmentation softmax calculation.

    :param logit: Logit vector.
    :param number_of_segments: Segment numbers.
    :param segmentation_index: Index of segmentation.
    :param index: Global index.
    :param temperature: Normalization values.
    :returns: Probability scores for attention.
    """
    logit_max = segment_max(logit, number_of_segments, segmentation_index, index).detach()
    logit = torch.exp((logit - logit_max) / temperature)
    logit_norm = segment_sum(logit, number_of_segments, segmentation_index)
    prob = logit / (logit_norm + torch.finfo(logit_norm.dtype).eps)
    return prob


class CoAttention(nn.Module):
    """The co-attention network for MHCADDI model."""

    def __init__(self, input_channels: int, output_channels: int, dropout: float=0.1):
        """Instantiate the co-attention network.

        :param input_channels: The number of atom features.
        :param output_channels: The number of output features.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.temperature = np.sqrt(input_channels)
        self.key_projection = nn.Linear(input_channels, input_channels, bias=False)
        self.value_projection = nn.Linear(input_channels, input_channels, bias=False)
        nn.init.xavier_normal_(self.key_projection.weight)
        nn.init.xavier_normal_(self.value_projection.weight)
        self.attention_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.out_projection = nn.Sequential(nn.Linear(input_channels, output_channels), nn.LeakyReLU(), nn.Dropout(dropout))

    def _calculate_message(self, translation: torch.Tensor, segmentation_number: torch.Tensor, segmentation_index: torch.Tensor, index: torch.Tensor, node: torch.Tensor, node_hidden_channels: torch.Tensor, node_neighbor: torch.Tensor):
        """Calculate the outer message."""
        node_edge = self.attention_dropout(segment_softmax(translation, segmentation_number, segmentation_index, index, self.temperature))
        node_edge = node_edge.view(-1, 1)
        message = node.new_zeros((segmentation_number, node_hidden_channels)).index_add(0, segmentation_index, node_edge * node_neighbor)
        message_graph = self.out_projection(message)
        return message_graph

    def forward(self, node_left: torch.FloatTensor, segmentation_index_left: torch.LongTensor, index_left: torch.LongTensor, node_right: torch.FloatTensor, segmentation_index_right: torch.LongTensor, index_right: torch.LongTensor):
        """Forward pass with the segmentation indices and node features.

        :param node_left: Left side node features.
        :param segmentation_index_left: Left side segmentation index.
        :param index_left: Left side indices.
        :param node_right: Right side node features.
        :param segmentation_index_right: Right side segmentation index.
        :param index_right: Right side indices.
        :returns: Left and right side messages and edge indices.
        """
        node_left_hidden_channels = node_left.size(1)
        node_right_hidden_channels = node_right.size(1)
        segmentation_number_left = node_left.size(0)
        segmentation_number_right = node_right.size(0)
        node_left_center = self.key_projection(node_left).index_select(0, segmentation_index_left)
        node_right_center = self.key_projection(node_right).index_select(0, segmentation_index_right)
        node_left_neighbor = self.value_projection(node_right).index_select(0, segmentation_index_right)
        node_right_neighbor = self.value_projection(node_left).index_select(0, segmentation_index_left)
        translation = (node_left_center * node_right_center).sum(1)
        message_graph_left = self._calculate_message(translation, segmentation_number_left, segmentation_index_left, index_left, node_left, node_left_hidden_channels, node_left_neighbor)
        message_graph_right = self._calculate_message(translation, segmentation_number_right, segmentation_index_right, index_right, node_right, node_right_hidden_channels, node_right_neighbor)
        return message_graph_left, message_graph_right


class CoAttentionMessagePassingNetwork(nn.Module):
    """Coattention message passing layer."""

    def __init__(self, hidden_channels: int, readout_channels: int, dropout: float=0.5):
        """Initialize a co-attention message passing network.

        :param hidden_channels: Input channel number.
        :param readout_channels: Readout channel number.
        :param dropout: Rate of dropout.
        """
        super().__init__()
        self.message_passing = MessagePassing(node_channels=hidden_channels, edge_channels=hidden_channels, hidden_channels=hidden_channels, dropout=dropout)
        self.co_attention = CoAttention(input_channels=hidden_channels, output_channels=hidden_channels, dropout=dropout)
        self.linear = nn.LayerNorm(hidden_channels)
        self.leaky_relu = nn.LeakyReLU()
        self.prediction_readout_projection = nn.Linear(hidden_channels, readout_channels)

    def _get_graph_features(self, atom_features: torch.Tensor, inner_message: torch.Tensor, outer_message: torch.Tensor, segmentation_molecule: torch.Tensor):
        """Get the graph representations."""
        message = atom_features + inner_message + outer_message
        message = self.linear(message)
        graph_features = self.readout(message, segmentation_molecule)
        return graph_features

    def forward(self, segmentation_molecule_left: torch.Tensor, atom_left: torch.Tensor, bond_left: torch.Tensor, inner_segmentation_index_left: torch.Tensor, inner_index_left: torch.Tensor, outer_segmentation_index_left: torch.Tensor, outer_index_left: torch.Tensor, segmentation_molecule_right: torch.Tensor, atom_right: torch.Tensor, bond_right: torch.Tensor, inner_segmentation_index_right: torch.Tensor, inner_index_right: torch.Tensor, outer_segmentation_index_right: torch.Tensor, outer_index_right: torch.Tensor):
        """Make a forward pass with the data.

        :param segmentation_molecule_left: Mapping from node id to graph id for the left drugs.
        :param atom_left: Atom features on the left-hand side.
        :param bond_left: Bond features on the left-hand side.
        :param inner_segmentation_index_left: Heads of edges connecting atoms within the left drug molecules.
        :param inner_index_left: Tails of edges connecting atoms within the left drug molecules.
        :param outer_segmentation_index_left: Heads of edges connecting atoms between left and right drug molecules
        :param outer_index_left: Tails of edges connecting atoms between left and right drug molecules.
        :param segmentation_molecule_right:  Mapping from node id to graph id for the right drugs.
        :param atom_right: Atom features on the right-hand side.
        :param bond_right: Bond features on the right-hand side.
        :param inner_segmentation_index_right: Heads of edges connecting atoms within the right drug molecules.
        :param inner_index_right: Tails of edges connecting atoms within the right drug molecules.
        :param outer_segmentation_index_right: Heads of edges connecting atoms between right and left drug molecules
        :param outer_index_right: Heads of edges connecting atoms between right and left drug molecules
        :returns: Graph level representations.
        """
        outer_message_left, outer_message_right = self.co_attention(atom_left, outer_segmentation_index_left, outer_index_left, atom_right, outer_segmentation_index_right, outer_index_right)
        inner_message_left = self.message_passing(atom_left, bond_left, inner_segmentation_index_left, inner_index_left)
        inner_message_right = self.message_passing(atom_right, bond_right, inner_segmentation_index_right, inner_index_right)
        graph_left = self._get_graph_features(atom_left, inner_message_left, outer_message_left, segmentation_molecule_left)
        graph_right = self._get_graph_features(atom_right, inner_message_right, outer_message_right, segmentation_molecule_right)
        return graph_left, graph_right

    def readout(self, atom_features: torch.Tensor, segmentation_molecule: torch.Tensor):
        """Aggregate node features.

        :param atom_features: Atom embeddings.
        :param segmentation_molecule: Molecular segmentation index.
        :returns: Graph readout vectors.
        """
        segmentation_max = segmentation_molecule.max() + 1
        atom_features = self.leaky_relu(self.prediction_readout_projection(atom_features))
        hidden_channels = atom_features.size(1)
        readout_vectors = atom_features.new_zeros((segmentation_max, hidden_channels)).index_add(0, segmentation_molecule, atom_features)
        return readout_vectors


class EmbeddingLayer(torch.nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: int):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(feature_number, feature_number))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, left_representations: torch.FloatTensor, right_representations: torch.FloatTensor, alpha_scores: torch.FloatTensor):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = torch.nn.functional.normalize(self.weights, dim=-1)
        left_representations = torch.nn.functional.normalize(left_representations, dim=-1)
        right_representations = torch.nn.functional.normalize(right_representations, dim=-1)
        attention = attention.view(-1, self.weights.shape[0], self.weights.shape[1])
        scores = alpha_scores * (left_representations @ attention @ right_representations.transpose(-2, -1))
        scores = scores.sum(dim=(-2, -1)).view(-1, 1)
        return scores


class DrugDrugAttentionLayer(torch.nn.Module):
    """Co-attention layer for drug pairs."""

    def __init__(self, feature_number: int):
        """Initialize the co-attention layer.

        :param feature_number: Number of input features.
        """
        super().__init__()
        self.weight_query = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.weight_key = torch.nn.Parameter(torch.zeros(feature_number, feature_number // 2))
        self.bias = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.attention = torch.nn.Parameter(torch.zeros(feature_number // 2))
        self.tanh = torch.nn.Tanh()
        torch.nn.init.xavier_uniform_(self.weight_query)
        torch.nn.init.xavier_uniform_(self.weight_key)
        torch.nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        torch.nn.init.xavier_uniform_(self.attention.view(*self.attention.shape, -1))

    def forward(self, left_representations: torch.Tensor, right_representations: torch.Tensor):
        """Make a forward pass with the co-attention calculation.

        :param left_representations: Matrix of left hand side representations.
        :param right_representations: Matrix of right hand side representations.
        :returns: Attention scores.
        """
        keys = left_representations @ self.weight_key
        queries = right_representations @ self.weight_query
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = self.tanh(e_activations) @ self.attention
        return attentions


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DrugDrugAttentionLayer,
     lambda: ([], {'feature_number': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EmbeddingLayer,
     lambda: ([], {'feature_number': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Highway,
     lambda: ([], {'input_size': 4, 'prev_input_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_AstraZeneca_chemicalx(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

