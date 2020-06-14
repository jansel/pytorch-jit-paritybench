import sys
_module = sys.modules[__name__]
del sys
generator = _module
grid = _module
helper = _module
hyperparameter = _module
model = _module
olstm_model = _module
olstm_train = _module
test = _module
train = _module
utils = _module
validation = _module
visualize = _module
vlstm_model = _module
vlstm_train = _module

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


import time


import itertools


import torch


from torch.autograd import Variable


import numpy as np


import torch.nn as nn


class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(SocialModel, self).__init__()
        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda
        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.gru = args.gru
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        if self.gru:
            self.cell = nn.GRUCell(2 * self.embedding_size, self.rnn_size)
        self.input_embedding_layer = nn.Linear(self.input_size, self.
            embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.
            grid_size * self.rnn_size, self.embedding_size)
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        """
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        """
        numNodes = grid.size()[0]
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size *
            self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor
        for node in range(numNodes):
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
        social_tensor = social_tensor.view(numNodes, self.grid_size * self.
            grid_size * self.rnn_size)
        return social_tensor

    def forward(self, *args):
        """
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]
        if self.gru:
            cell_states = None
        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]
        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.
            output_size))
        if self.use_cuda:
            outputs = outputs
        for framenum, frame in enumerate(input_data):
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            if len(nodeIDs) == 0:
                continue
            list_of_nodes = [look_up[x] for x in nodeIDs]
            corr_index = Variable(torch.LongTensor(list_of_nodes))
            if self.use_cuda:
                corr_index = corr_index
            nodes_current = frame[(list_of_nodes), :]
            grid_current = grids[framenum]
            hidden_states_current = torch.index_select(hidden_states, 0,
                corr_index)
            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0,
                    corr_index)
            social_tensor = self.getSocialTensor(grid_current,
                hidden_states_current)
            input_embedded = self.dropout(self.relu(self.
                input_embedding_layer(nodes_current)))
            tensor_embedded = self.dropout(self.relu(self.
                tensor_embedding_layer(social_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedded, (
                    hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, hidden_states_current)
            outputs[framenum * numNodes + corr_index.data] = self.output_layer(
                h_nodes)
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes,
            self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[(framenum), (node), :] = outputs[(framenum *
                    numNodes + node), :]
        return outputs_return, hidden_states, cell_states


class OLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(OLSTMModel, self).__init__()
        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda
        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.gru = args.gru
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        if self.gru:
            self.cell = nn.GRUCell(2 * self.embedding_size, self.rnn_size)
        self.input_embedding_layer = nn.Linear(self.input_size, self.
            embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.
            grid_size, self.embedding_size)
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getObsTensor(self, grid):
        """
        Computes the obstacle map tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        """
        numNodes = grid.size()[0]
        Obs_tensor = Variable(torch.zeros(numNodes, self.grid_size * self.
            grid_size))
        if self.use_cuda:
            Obs_tensor = Obs_tensor
        for node in range(numNodes):
            Obs_tensor[node] = grid[node]
        Obs_tensor = Obs_tensor.view(numNodes, self.grid_size * self.grid_size)
        return Obs_tensor

    def forward(self, *args):
        """
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]
        if self.gru:
            cell_states = None
        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]
        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.
            output_size))
        if self.use_cuda:
            outputs = outputs
        for framenum, frame in enumerate(input_data):
            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            if len(nodeIDs) == 0:
                continue
            list_of_nodes = [look_up[x] for x in nodeIDs]
            corr_index = Variable(torch.LongTensor(list_of_nodes))
            if self.use_cuda:
                corr_index = corr_index
            nodes_current = frame[(list_of_nodes), :]
            grid_current = grids[framenum]
            hidden_states_current = torch.index_select(hidden_states, 0,
                corr_index)
            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0,
                    corr_index)
            Obs_tensor = self.getObsTensor(grid_current)
            input_embedded = self.dropout(self.relu(self.
                input_embedding_layer(nodes_current)))
            tensor_embedded = self.dropout(self.relu(self.
                tensor_embedding_layer(Obs_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedded, (
                    hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, hidden_states_current)
            outputs[framenum * numNodes + corr_index.data] = self.output_layer(
                h_nodes)
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes,
            self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[(framenum), (node), :] = outputs[(framenum *
                    numNodes + node), :]
        return outputs_return, hidden_states, cell_states


class VLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(VLSTMModel, self).__init__()
        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda
        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.gru = args.gru
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)
        if self.gru:
            self.cell = nn.GRUCell(self.embedding_size, self.rnn_size)
        self.input_embedding_layer = nn.Linear(self.input_size, self.
            embedding_size)
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, *args):
        """
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        input_data = args[0]
        hidden_states = args[1]
        cell_states = args[2]
        if self.gru:
            cell_states = None
        PedsList = args[3]
        num_pedlist = args[4]
        dataloader = args[5]
        look_up = args[6]
        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.
            output_size))
        if self.use_cuda:
            outputs = outputs
        for framenum, frame in enumerate(input_data):
            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            if len(nodeIDs) == 0:
                continue
            list_of_nodes = [look_up[x] for x in nodeIDs]
            corr_index = Variable(torch.LongTensor(list_of_nodes))
            if self.use_cuda:
                outputs = outputs
            nodes_current = frame[(list_of_nodes), :]
            hidden_states_current = torch.index_select(hidden_states, 0,
                corr_index)
            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0,
                    corr_index)
            input_embedded = self.dropout(self.relu(self.
                input_embedding_layer(nodes_current)))
            if not self.gru:
                h_nodes, c_nodes = self.cell(input_embedded, (
                    hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(input_embedded, hidden_states_current)
            outputs[framenum * numNodes + corr_index.data] = self.output_layer(
                h_nodes)
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes,
            self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[(framenum), (node), :] = outputs[(framenum *
                    numNodes + node), :]
        return outputs_return, hidden_states, cell_states


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_quancore_social_lstm(_paritybench_base):
    pass
