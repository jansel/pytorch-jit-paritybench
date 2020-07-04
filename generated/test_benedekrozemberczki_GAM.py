import sys
_module = sys.modules[__name__]
del sys
gam = _module
main = _module
param_parser = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import random


import numpy as np


class StepNetworkLayer(torch.nn.Module):
    """
    Step Network Layer Class for selecting next node to move.
    """

    def __init__(self, args, identifiers):
        """
        Initializing the layer.
        :param args: Arguments object.
        :param identifiers: Node type -- id hash map.
        """
        super(StepNetworkLayer, self).__init__()
        self.identifiers = identifiers
        self.args = args
        self.setup_attention()
        self.create_parameters()

    def setup_attention(self):
        """
        Initial attention generation with uniform attention scores.
        """
        self.attention = torch.ones(len(self.identifiers)) / len(self.
            identifiers)

    def create_parameters(self):
        """
        Creating trainable weights and initlaizing them.
        """
        self.theta_step_1 = torch.nn.Parameter(torch.Tensor(len(self.
            identifiers), self.args.step_dimensions))
        self.theta_step_2 = torch.nn.Parameter(torch.Tensor(len(self.
            identifiers), self.args.step_dimensions))
        self.theta_step_3 = torch.nn.Parameter(torch.Tensor(2 * self.args.
            step_dimensions, self.args.combined_dimensions))
        torch.nn.init.uniform_(self.theta_step_1, -1, 1)
        torch.nn.init.uniform_(self.theta_step_2, -1, 1)
        torch.nn.init.uniform_(self.theta_step_3, -1, 1)

    def sample_node_label(self, orig_neighbors, graph, features):
        """
        Sampling a label from the neighbourhood.
        :param original_neighbors: Neighbours of the source node.
        :param graph: NetworkX graph.
        :param features: Node feature matrix.
        :return label: Label sampled from the neighbourhood with attention.
        """
        neighbor_vector = torch.tensor([(1.0 if n in orig_neighbors else 
            0.0) for n in graph.nodes()])
        neighbor_features = torch.mm(neighbor_vector.view(1, -1), features)
        attention_spread = self.attention * neighbor_features
        normalized_attention_spread = attention_spread / attention_spread.sum()
        normalized_attention_spread = normalized_attention_spread.detach(
            ).numpy().reshape(-1)
        label = np.random.choice(np.arange(len(self.identifiers)), p=
            normalized_attention_spread)
        return label

    def make_step(self, node, graph, features, labels, inverse_labels):
        """
        :param node: Source node for step.
        :param graph: NetworkX graph.
        :param features: Feature matrix.
        :param labels: Node labels hash table.
        :param inverse_labels: Inverse node label hash table.
        """
        orig_neighbors = set(nx.neighbors(graph, node))
        label = self.sample_node_label(orig_neighbors, graph, features)
        labels = list(set(orig_neighbors).intersection(set(inverse_labels[
            str(label)])))
        new_node = random.choice(labels)
        new_node_attributes = torch.zeros((len(self.identifiers), 1))
        new_node_attributes[label, 0] = 1.0
        attention_score = self.attention[label]
        return new_node_attributes, new_node, attention_score

    def forward(self, data, graph, features, node):
        """
        Making a forward propagation step.
        :param data: Data hash table.
        :param graph: NetworkX graph object.
        :param features: Feature matrix of the graph.
        :param node: Base node where the step is taken from.
        :return state: State vector.
        :return node: New node to move to.
        :return attention_score: Attention score of chosen node.
        """
        feature_row, node, attention_score = self.make_step(node, graph,
            features, data['labels'], data['inverse_labels'])
        hidden_attention = torch.mm(self.attention.view(1, -1), self.
            theta_step_1)
        hidden_node = torch.mm(torch.t(feature_row), self.theta_step_2)
        combined_hidden_representation = torch.cat((hidden_attention,
            hidden_node), dim=1)
        state = torch.mm(combined_hidden_representation, self.theta_step_3)
        state = state.view(1, 1, self.args.combined_dimensions)
        return state, node, attention_score


class DownStreamNetworkLayer(torch.nn.Module):
    """
    Neural network layer for attention update and node label assignment.
    """

    def __init__(self, args, target_number, identifiers):
        """
        :param args:
        :param target_number:
        :param identifiers:
        """
        super(DownStreamNetworkLayer, self).__init__()
        self.args = args
        self.target_number = target_number
        self.identifiers = identifiers
        self.create_parameters()

    def create_parameters(self):
        """
        Defining and initializing the classification and attention update weights.
        """
        self.theta_classification = torch.nn.Parameter(torch.Tensor(self.
            args.combined_dimensions, self.target_number))
        self.theta_rank = torch.nn.Parameter(torch.Tensor(self.args.
            combined_dimensions, len(self.identifiers)))
        torch.nn.init.xavier_normal_(self.theta_classification)
        torch.nn.init.xavier_normal_(self.theta_rank)

    def forward(self, hidden_state):
        """
        Making a forward propagation pass with the input from the LSTM layer.
        :param hidden_state: LSTM state used for labeling and attention update.
        """
        predictions = torch.mm(hidden_state.view(1, -1), self.
            theta_classification)
        attention = torch.mm(hidden_state.view(1, -1), self.theta_rank)
        attention = torch.nn.functional.softmax(attention, dim=1)
        return predictions, attention


def read_node_labels(args):
    """
    Reading the graphs from disk.
    :param args: Arguments object.
    :return identifiers: Hash table of unique node labels in the dataset.
    :return class_number: Number of unique graph classes in the dataset.
    """
    print('\nCollecting unique node labels.\n')
    labels = set()
    targets = set()
    graphs = glob.glob(args.train_graph_folder + '*.json')
    try:
        graphs = graphs + glob.glob(args.test_graph_folder + '*.json')
    except:
        pass
    for g in tqdm(graphs):
        data = json.load(open(g))
        labels = labels.union(set(list(data['labels'].values())))
        targets = targets.union(set([data['target']]))
    identifiers = {label: i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print('\n\nThe number of graph classes is: ' + str(class_number) + '.\n')
    return identifiers, class_number


class GAM(torch.nn.Module):
    """
    Graph Attention Machine class.
    """

    def __init__(self, args):
        """
        Initializing the machine.
        :param args: Arguments object.
        """
        super(GAM, self).__init__()
        self.args = args
        self.identifiers, self.class_number = read_node_labels(self.args)
        self.step_block = StepNetworkLayer(self.args, self.identifiers)
        self.recurrent_block = torch.nn.LSTM(self.args.combined_dimensions,
            self.args.combined_dimensions, 1)
        self.down_block = DownStreamNetworkLayer(self.args, self.
            class_number, self.identifiers)
        self.reset_attention()

    def reset_attention(self):
        """
        Resetting the attention and hidden states.
        """
        self.step_block.attention = torch.ones(len(self.identifiers)) / len(
            self.identifiers)
        self.lstm_h_0 = torch.randn(1, 1, self.args.combined_dimensions)
        self.lstm_c_0 = torch.randn(1, 1, self.args.combined_dimensions)

    def forward(self, data, graph, features, node):
        """
        Doing a forward pass on a graph from a given node.
        :param data: Data dictionary.
        :param graph: NetworkX graph.
        :param features: Feature tensor.
        :param node: Source node identifier.
        :return label_predictions: Label prediction.
        :return node: New node to move to.
        :return attention_score: Attention score on selected node.
        """
        self.state, node, attention_score = self.step_block(data, graph,
            features, node)
        lstm_output, (self.h0, self.c0) = self.recurrent_block(self.state,
            (self.lstm_h_0, self.lstm_c_0))
        label_predictions, attention = self.down_block(lstm_output)
        self.step_block.attention = attention.view(-1)
        label_predictions = torch.nn.functional.log_softmax(label_predictions,
            dim=1)
        return label_predictions, node, attention_score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_benedekrozemberczki_GAM(_paritybench_base):
    pass
