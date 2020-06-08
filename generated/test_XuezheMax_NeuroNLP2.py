import sys
_module = sys.modules[__name__]
del sys
ner = _module
parsing = _module
pos_tagging = _module
neuronlp2 = _module
io = _module
alphabet = _module
common = _module
conll03_data = _module
conllx_data = _module
conllx_stacked_data = _module
instance = _module
logger = _module
reader = _module
utils = _module
writer = _module
models = _module
parsing = _module
sequence_labeling = _module
nn = _module
_functions = _module
rnnFusedBackend = _module
skipconnect_rnn = _module
variational_rnn = _module
crf = _module
init = _module
modules = _module
skip_rnn = _module
utils = _module
variational_rnn = _module
optim = _module
lr_scheduler = _module
tasks = _module
parser = _module

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


import numpy as np


import torch


from torch.optim.adamw import AdamW


from torch.optim import SGD


from torch.nn.utils import clip_grad_norm_


import math


from enum import Enum


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from collections import OrderedDict


import collections


from itertools import repeat


from torch._six import inf


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class BiRecurrentConv(nn.Module):

    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode,
        hidden_size, out_features, num_layers, num_labels, embedd_word=None,
        embedd_char=None, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5),
        activation='elu'):
        super(BiRecurrentConv, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=
            embedd_word, padding_idx=1)
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=
            embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=4 *
            char_dim, activation=activation)
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_out = nn.Dropout(p_out)
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM' or rnn_mode == 'FastLSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        self.rnn = RNN(word_dim + char_dim, hidden_size, num_layers=
            num_layers, batch_first=True, bidirectional=True, dropout=p_rnn[1])
        self.fc = nn.Linear(hidden_size * 2, out_features)
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        self.readout = nn.Linear(out_features, num_labels)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char)

    def reset_parameters(self, embedd_word, embedd_char):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
        for param in self.rnn.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        nn.init.constant_(self.readout.bias, 0.0)

    def _get_rnn_output(self, input_word, input_char, mask=None):
        word = self.word_embed(input_word)
        char = self.char_cnn(self.char_embed(input_char))
        word = self.dropout_in(word)
        char = self.dropout_in(char)
        enc = torch.cat([word, char], dim=2)
        enc = self.dropout_rnn_in(enc)
        if mask is not None:
            length = mask.sum(dim=1).long()
            packed_enc = pack_padded_sequence(enc, length, batch_first=True,
                enforce_sorted=False)
            packed_out, _ = self.rnn(packed_enc)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.rnn(enc)
        output = self.dropout_out(output)
        output = self.dropout_out(self.activation(self.fc(output)))
        return output

    def forward(self, input_word, input_char, mask=None):
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        return output

    def loss(self, input_word, input_char, target, mask=None):
        output = self(input_word, input_char, mask=mask)
        logits = self.readout(output).transpose(1, 2)
        loss = self.criterion(logits, target)
        if mask is not None:
            loss = loss * mask
        loss = loss.sum(dim=1)
        return loss

    def decode(self, input_word, input_char, mask=None, leading_symbolic=0):
        output = self(input_word, input_char, mask=mask)
        logits = self.readout(output).transpose(1, 2)
        _, preds = torch.max(logits[:, leading_symbolic:], dim=1)
        preds += leading_symbolic
        if mask is not None:
            preds = preds * mask.long()
        return preds


class ChainCRF(nn.Module):

    def __init__(self, input_size, num_labels, bigram=True):
        """

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        """
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.state_net = nn.Linear(input_size, self.num_labels)
        if bigram:
            self.transition_net = nn.Linear(input_size, self.num_labels *
                self.num_labels)
            self.register_parameter('transition_matrix', None)
        else:
            self.transition_net = None
            self.transition_matrix = Parameter(torch.Tensor(self.num_labels,
                self.num_labels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_net.bias, 0.0)
        if self.bigram:
            nn.init.xavier_uniform_(self.transition_net.weight)
            nn.init.constant_(self.transition_net.bias, 0.0)
        else:
            nn.init.normal_(self.transition_matrix)

    def forward(self, input, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        """
        batch, length, _ = input.size()
        out_s = self.state_net(input).unsqueeze(2)
        if self.bigram:
            out_t = self.transition_net(input).view(batch, length, self.
                num_labels, self.num_labels)
            output = out_t + out_s
        else:
            output = self.transition_matrix + out_s
        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)
        return output

    def loss(self, input, target, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss [batch]
        """
        batch, length, _ = input.size()
        energy = self(input, mask=mask)
        energy_transpose = energy.transpose(0, 1)
        target_transpose = target.transpose(0, 1)
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)
        partition = None
        batch_index = torch.arange(0, batch).type_as(input).long()
        prev_label = input.new_full((batch,), self.num_labels - 1).long()
        tgt_energy = input.new_zeros(batch)
        for t in range(length):
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, (-1), :]
            else:
                partition_new = torch.logsumexp(curr_energy + partition.
                    unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition
                        ) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label,
                target_transpose[t]]
            prev_label = target_transpose[t]
        return torch.logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch, length]

        """
        energy = self(input, mask=mask)
        energy_transpose = energy.transpose(0, 1)
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1,
            leading_symbolic:-1]
        length, batch_size, num_label, _ = energy_transpose.size()
        batch_index = torch.arange(0, batch_size).type_as(input).long()
        pi = input.new_zeros([length, batch_size, num_label])
        pointer = batch_index.new_zeros(length, batch_size, num_label)
        back_pointer = batch_index.new_zeros(length, batch_size)
        pi[0] = energy[:, (0), (-1), leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev.
                unsqueeze(2), dim=1)
        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        return back_pointer.transpose(0, 1) + leading_symbolic


class TreeCRF(nn.Module):
    """
    Tree CRF layer.
    """

    def __init__(self, model_dim):
        """

        Args:
            model_dim: int
                the dimension of the input.

        """
        super(TreeCRF, self).__init__()
        self.model_dim = model_dim
        self.energy = BiAffine(model_dim, model_dim)

    def forward(self, heads, children, mask=None):
        """

        Args:
            heads: Tensor
                the head input tensor with shape = [batch, length, model_dim]
            children: Tensor
                the child input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            lengths: Tensor or None
                the length tensor with shape = [batch]

        Returns: Tensor
            the energy tensor with shape = [batch, length, length]

        """
        batch, length, _ = heads.size()
        output = self.energy(heads, children, mask_query=mask, mask_key=mask)
        return output

    def loss(self, heads, children, target_heads, mask=None):
        """

        Args:
            heads: Tensor
                the head input tensor with shape = [batch, length, model_dim]
            children: Tensor
                the child input tensor with shape = [batch, length, model_dim]
            target_heads: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        """
        batch, length, _ = heads.size()
        energy = self(heads, children, mask=mask).double()
        A = torch.exp(energy)
        if mask is not None:
            mask = mask.double()
            A = A * mask.unsqueeze(2) * mask.unsqueeze(1)
        diag_mask = 1.0 - torch.eye(length).unsqueeze(0).type_as(energy)
        A = A * diag_mask
        energy = energy * diag_mask
        D = A.sum(dim=1)
        rtol = 0.0001
        atol = 1e-06
        D += atol
        if mask is not None:
            D = D * mask
        D = torch.diag_embed(D)
        L = D - A
        if mask is not None:
            L = L + torch.diag_embed(1.0 - mask)
        L = L[:, 1:, 1:]
        z = torch.logdet(L)
        index = torch.arange(0, length).view(length, 1).expand(length, batch)
        index = index.type_as(energy).long()
        batch_index = torch.arange(0, batch).type_as(index)
        tgt_energy = energy[batch_index, target_heads.t(), index][1:]
        tgt_energy = tgt_energy.sum(dim=0)
        return (z - tgt_energy).float()


class BiLinear(nn.Module):
    """
    Bi-linear layer
    """

    def __init__(self, left_features, right_features, out_features, bias=True):
        """

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features
        self.U = Parameter(torch.Tensor(self.out_features, self.
            left_features, self.right_features))
        self.weight_left = Parameter(torch.Tensor(self.out_features, self.
            left_features))
        self.weight_right = Parameter(torch.Tensor(self.out_features, self.
            right_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.xavier_uniform_(self.weight_right)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        """
        batch_size = input_left.size()[:-1]
        batch = int(np.prod(batch_size))
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.weight_left, None
            ) + F.linear(input_right, self.weight_right, None)
        return output.view(batch_size + (self.out_features,))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'left_features=' + str(self
            .left_features) + ', right_features=' + str(self.right_features
            ) + ', out_features=' + str(self.out_features) + ')'


class CharCNN(nn.Module):
    """
    CNN layers for characters
    """

    def __init__(self, num_layers, in_channels, out_channels,
        hidden_channels=None, activation='elu'):
        super(CharCNN, self).__init__()
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            ACT = nn.ELU
        else:
            ACT = nn.Tanh
        layers = list()
        for i in range(num_layers - 1):
            layers.append(('conv{}'.format(i), nn.Conv1d(in_channels,
                hidden_channels, kernel_size=3, padding=1)))
            layers.append(('act{}'.format(i), ACT()))
            in_channels = hidden_channels
        layers.append(('conv_top', nn.Conv1d(in_channels, out_channels,
            kernel_size=3, padding=1)))
        layers.append(('act_top', ACT()))
        self.act = ACT
        self.net = nn.Sequential(OrderedDict(layers))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            else:
                assert isinstance(layer, self.act)

    def forward(self, char):
        """

        Args:
            char: Tensor
                the input tensor of character [batch, sent_length, char_length, in_channels]

        Returns: Tensor
            output character encoding with shape [batch, sent_length, in_channels]

        """
        char_size = char.size()
        char = char.view(-1, char_size[2], char_size[3]).transpose(1, 2)
        char = self.net(char).max(dim=2)[0]
        return char.view(char_size[0], char_size[1], -1)


class VarSkipRNNBase(nn.Module):

    def __init__(self, Cell, input_size, hidden_size, num_layers=1, bias=
        True, batch_first=False, dropout=(0, 0), bidirectional=False, **kwargs
        ):
        super(VarSkipRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1
        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (input_size if layer == 0 else 
                    hidden_size * num_directions)
                cell = self.Cell(layer_input_size, hidden_size, self.bias,
                    p=dropout, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions +
                    direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, skip_connect, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradSkipConnectRNN(num_layers=self.num_layers,
            batch_first=self.batch_first, bidirectional=self.bidirectional,
            lstm=self.lstm)
        self.reset_noise(batch_size)
        output, hidden = func(input, skip_connect, self.all_cells, hx, None if
            mask is None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, hs=None, mask=None):
        """
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, model_dim): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            hs (batch. hidden_size): tensor containing the skip connection state for each element in the batch.
            mask (batch): the mask tensor of this step.

        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        """
        assert not self.bidirectional, 'step only cannot be applied to bidirectional RNN.'
        batch_size = input.size(0)
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        if hs is None:
            hs = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        func = rnn_F.AutogradSkipConnectStep(num_layers=self.num_layers,
            lstm=self.lstm)
        output, hidden = func(input, self.all_cells, hx, hs, mask)
        return output, hidden


class VarRNNBase(nn.Module):

    def __init__(self, Cell, input_size, hidden_size, num_layers=1, bias=
        True, batch_first=False, dropout=(0, 0), bidirectional=False, **kwargs
        ):
        super(VarRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1
        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (input_size if layer == 0 else 
                    hidden_size * num_directions)
                cell = self.Cell(layer_input_size, hidden_size, self.bias,
                    p=dropout, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions +
                    direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradVarRNN(num_layers=self.num_layers, batch_first
            =self.batch_first, bidirectional=self.bidirectional, lstm=self.lstm
            )
        self.reset_noise(batch_size)
        output, hidden = func(input, self.all_cells, hx, None if mask is
            None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, mask=None):
        """
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, model_dim): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            mask (batch): the mask tensor of this step.

        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        """
        assert not self.bidirectional, 'step only cannot be applied to bidirectional RNN.'
        batch_size = input.size(0)
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradVarRNNStep(num_layers=self.num_layers, lstm=
            self.lstm)
        output, hidden = func(input, self.all_cells, hx, mask)
        return output, hidden


class VarRNNCellBase(nn.Module):

    def __repr__(self):
        s = '{name}({model_dim}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_noise(self, batch_size):
        """
        Should be overriden by all subclasses.
        Args:
            batch_size: (int) batch size of input.
        """
        raise NotImplementedError


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_XuezheMax_NeuroNLP2(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(ChainCRF(*[], **{'input_size': 4, 'num_labels': 4}), [torch.rand([4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(BiLinear(*[], **{'left_features': 4, 'right_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(CharCNN(*[], **{'num_layers': 1, 'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})
