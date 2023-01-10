import sys
_module = sys.modules[__name__]
del sys
elmoformanylangs = _module
biLM = _module
dataloader = _module
elmo = _module
frontend = _module
modules = _module
classify_layer = _module
elmo = _module
embedding_layer = _module
encoder_base = _module
highway = _module
lstm = _module
lstm_cell_with_projection = _module
token_embedder = _module
util = _module
utils = _module
main = _module
preprocess = _module

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


import logging


import torch


import numpy as np


import time


import random


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


from collections import Counter


import math


from typing import Optional


from typing import Tuple


from typing import List


from typing import Callable


from typing import Union


import numpy


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import copy


from collections import defaultdict


from typing import Dict


from typing import Any


import itertools


from torch.utils.tensorboard import SummaryWriter


from scipy.stats import pearsonr


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


class ConvTokenEmbedder(nn.Module):

    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
        super(ConvTokenEmbedder, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.word_emb_layer = word_emb_layer
        self.char_emb_layer = char_emb_layer
        self.output_dim = config['encoder']['projection_dim']
        self.emb_dim = 0
        if word_emb_layer is not None:
            self.emb_dim += word_emb_layer.n_d
        if char_emb_layer is not None:
            self.convolutions = []
            cnn_config = config['token_embedder']
            filters = cnn_config['filters']
            char_embed_dim = cnn_config['char_dim']
            for i, (width, num) in enumerate(filters):
                conv = torch.nn.Conv1d(in_channels=char_embed_dim, out_channels=num, kernel_size=width, bias=True)
                self.convolutions.append(conv)
            self.convolutions = nn.ModuleList(self.convolutions)
            self.n_filters = sum(f[1] for f in filters)
            self.n_highway = cnn_config['n_highway']
            self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)
            self.emb_dim += self.n_filters
        self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)

    def forward(self, word_inp, chars_inp, shape):
        embs = []
        batch_size, seq_len = shape
        if self.word_emb_layer is not None:
            batch_size, seq_len = word_inp.size(0), word_inp.size(1)
            word_emb = self.word_emb_layer(Variable(word_inp) if self.use_cuda else Variable(word_inp))
            embs.append(word_emb)
        if self.char_emb_layer is not None:
            chars_inp = chars_inp.view(batch_size * seq_len, -1)
            character_embedding = self.char_emb_layer(Variable(chars_inp) if self.use_cuda else Variable(chars_inp))
            character_embedding = torch.transpose(character_embedding, 1, 2)
            cnn_config = self.config['token_embedder']
            if cnn_config['activation'] == 'tanh':
                activation = torch.nn.functional.tanh
            elif cnn_config['activation'] == 'relu':
                activation = torch.nn.functional.relu
            else:
                raise Exception('Unknown activation')
            convs = []
            for i in range(len(self.convolutions)):
                convolved = self.convolutions[i](character_embedding)
                convolved, _ = torch.max(convolved, dim=-1)
                convolved = activation(convolved)
                convs.append(convolved)
            char_emb = torch.cat(convs, dim=-1)
            char_emb = self.highways(char_emb)
            embs.append(char_emb.view(batch_size, -1, self.n_filters))
        token_embedding = torch.cat(embs, dim=2)
        return self.projection(token_embedding)


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int], gain: float=1.0) ->None:
    """
        An initializer which allows initializing model parameters in "blocks". This is helpful
        in the case of recurrent models which use multiple gates applied to linear projections,
        which can be computed efficiently if they are concatenated together. However, they are
        separate parameters which should be initialized independently.
        Parameters
        ----------
        tensor : ``torch.Tensor``, required.
            A tensor to initialize.
        split_sizes : List[int], required.
            A list of length ``tensor.ndim()`` specifying the size of the
            blocks along that particular dimension. E.g. ``[10, 20]`` would
            result in the tensor being split into chunks of size 10 along the
            first dimension and 20 along the second.
        gain : float, optional (default = 1.0)
            The gain (scaling) applied to the orthogonal initialization.
        """
    if isinstance(tensor, Variable):
        sizes = list(tensor.size())
        if any([(a % b != 0) for a, b in zip(sizes, split_sizes)]):
            raise ConfigurationError('tensor dimensions must be divisible by their respective split_sizes. Found size: {} and split_sizes: {}'.format(sizes, split_sizes))
        indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]
        for block_start_indices in itertools.product(*indexes):
            index_and_step_tuples = zip(block_start_indices, split_sizes)
            block_slice = tuple([slice(start_index, start_index + step) for start_index, step in index_and_step_tuples])
            tensor[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class LstmCellWithProjection(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.
    Parameters
    ----------
    input_size : ``int``, required.
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required.
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell used for the LSTM.
    go_forward: ``bool``, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    Returns
    -------
    output_accumulator : ``torch.FloatTensor``
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(self, input_size: int, hidden_size: int, cell_size: int, go_forward: bool=True, recurrent_dropout_probability: float=0.0, memory_cell_clip_value: Optional[float]=None, state_projection_clip_value: Optional[float]=None) ->None:
        super(LstmCellWithProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)
        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self, inputs: torch.FloatTensor, batch_lengths: List[int], initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]
        output_accumulator = Variable(inputs.data.new(batch_size, total_timesteps, self.hidden_size).fill_(0))
        if initial_state is None:
            full_batch_previous_memory = Variable(inputs.data.new(batch_size, self.cell_size).fill_(0))
            full_batch_previous_state = Variable(inputs.data.new(batch_size, self.hidden_size).fill_(0))
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_state)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:current_length_index + 1].clone()
            timestep_input = inputs[0:current_length_index + 1, index]
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            input_gate = torch.sigmoid(projected_input[:, 0 * self.cell_size:1 * self.cell_size] + projected_state[:, 0 * self.cell_size:1 * self.cell_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.cell_size:2 * self.cell_size] + projected_state[:, 1 * self.cell_size:2 * self.cell_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.cell_size:3 * self.cell_size] + projected_state[:, 2 * self.cell_size:3 * self.cell_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.cell_size:4 * self.cell_size] + projected_state[:, 3 * self.cell_size:4 * self.cell_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            if self.memory_cell_clip_value:
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)
            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output, -self.state_projection_clip_value, self.state_projection_clip_value)
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0:current_length_index + 1]
            full_batch_previous_memory = Variable(full_batch_previous_memory.data.clone())
            full_batch_previous_state = Variable(full_batch_previous_state.data.clone())
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output
        final_state = full_batch_previous_state.unsqueeze(0), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


RnnStateStorage = Tuple[torch.Tensor, ...]


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : Variable(torch.LongTensor)
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
        raise Exception('Both the tensor and sequence lengths must be torch.autograd.Variables.')
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


class _EncoderBase(torch.nn.Module):
    """
    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`
    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool=False) ->None:
        super(_EncoderBase, self).__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence, Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor], RnnState]], inputs: torch.Tensor, mask: torch.Tensor, hidden_state: Optional[RnnState]=None):
        """
        This function exists because Pytorch RNNs require that their inputs be sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a ``PackedSequence`` and some ``hidden_state``, which can either be a
        tuple of tensors or a tensor.
        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module, which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.
        Parameters
        ----------
        module : ``Callable[[PackedSequence, Optional[RnnState]],
                            Tuple[Union[PackedSequence, torch.Tensor], RnnState]]``, required.
            A function to run on the inputs. In most cases, this is a ``torch.nn.Module``.
        inputs : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length, embedding_size)`` representing
            the inputs to the Encoder.
        mask : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length)``, representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : ``Optional[RnnState]``, (default = None).
            A single tensor of shape (num_layers, batch_size, hidden_size) representing the
            state of an RNN with or a tuple of
            tensors of shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden state and memory
            state of an LSTM-like RNN.
        Returns
        -------
        module_output : ``Union[torch.Tensor, PackedSequence]``.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
            The batch size dimension will be equal to ``num_valid``, as sequences of zero
            length are clipped off before the module is called, as Pytorch cannot handle
            zero length sequences.
        final_states : ``Optional[RnnState]``
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : ``torch.LongTensor``
            A tensor of shape ``(batch_size,)``, describing the re-indexing required to transform
            the outputs back to their original batch order.
        """
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item()
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :], sorted_sequence_lengths[:num_valid].data.tolist(), batch_first=True)
        if not self.stateful:
            if hidden_state is None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:, :num_valid, :] for state in hidden_state]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[:, :num_valid, :]
        else:
            initial_states = self._get_initial_states(batch_size, num_valid, sorting_indices)
        module_output, final_states = module(packed_sequence_input, initial_states)
        return module_output, final_states, restoration_indices

    def _get_initial_states(self, batch_size: int, num_valid: int, sorting_indices: torch.LongTensor) ->Optional[RnnState]:
        """
        Returns an initial state for use in an RNN. Additionally, this method handles
        the batch size changing across calls by mutating the state to append initial states
        for new elements in the batch. Finally, it also handles sorting the states
        with respect to the sequence lengths of elements in the batch and removing rows
        which are completely padded. Importantly, this `mutates` the state if the
        current batch size is larger than when it was previously called.
        Parameters
        ----------
        batch_size : ``int``, required.
            The batch size can change size across calls to stateful RNNs, so we need
            to know if we need to expand or shrink the states before returning them.
            Expanded states will be set to zero.
        num_valid : ``int``, required.
            The batch may contain completely padded sequences which get removed before
            the sequence is passed through the encoder. We also need to clip these off
            of the state too.
        sorting_indices ``torch.LongTensor``, required.
            Pytorch RNNs take sequences sorted by length. When we return the states to be
            used for a given call to ``module.forward``, we need the states to match up to
            the sorted sequences, so before returning them, we sort the states using the
            same indices used to sort the sequences.
        Returns
        -------
        This method has a complex return type because it has to deal with the first time it
        is called, when it has no state, and the fact that types of RNN have heterogeneous
        states.
        If it is the first time the module has been called, it returns ``None``, regardless
        of the type of the ``Module``.
        Otherwise, for LSTMs, it returns a tuple of ``torch.Tensors`` with shape
        ``(num_layers, num_valid, state_size)`` and ``(num_layers, num_valid, memory_size)``
        respectively, or for GRUs, it returns a single ``torch.Tensor`` of shape
        ``(num_layers, num_valid, state_size)``.
        """
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = state.data.new(state.size(0), num_states_to_concat, state.size(2)).fill_(0)
                zeros = Variable(zeros)
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1, sorting_indices)
            return sorted_state[:, :num_valid, :]
        else:
            sorted_states = [state.index_select(1, sorting_indices) for state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :] for state in sorted_states)

    def _update_states(self, final_states: RnnStateStorage, restoration_indices: torch.LongTensor) ->None:
        """
        After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detatches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.
        Parameters
        ----------
        final_states : ``RnnStateStorage``, required.
            The hidden states returned as output from the RNN.
        restoration_indices : ``torch.LongTensor``, required.
            The indices that invert the sorting used in ``sort_and_run_forward``
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        new_unsorted_states = [state.index_select(1, restoration_indices) for state in final_states]
        if self._states is None:
            self._states = tuple([torch.autograd.Variable(state.data) for state in new_unsorted_states])
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [(state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1) for state in new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(torch.autograd.Variable(old_state.data))
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states, new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(torch.autograd.Variable(new_state.data))
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None


class ElmobiLm(_EncoderBase):

    def __init__(self, config, use_cuda=False):
        super(ElmobiLm, self).__init__(stateful=True)
        self.config = config
        self.use_cuda = use_cuda
        input_size = config['encoder']['projection_dim']
        hidden_size = config['encoder']['projection_dim']
        cell_size = config['encoder']['dim']
        num_layers = config['encoder']['n_layers']
        memory_cell_clip_value = config['encoder']['cell_clip']
        state_projection_clip_value = config['encoder']['proj_clip']
        recurrent_dropout_probability = config['dropout']
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        forward_layers = []
        backward_layers = []
        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size, hidden_size, cell_size, go_forward, recurrent_dropout_probability, memory_cell_clip_value, state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size, hidden_size, cell_size, not go_forward, recurrent_dropout_probability, memory_cell_clip_value, state_projection_clip_value)
            lstm_input_size = hidden_size
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs, mask):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(self._lstm_forward, inputs, mask)
        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        if num_valid < batch_size:
            zeros = stacked_sequence_output.data.new(num_layers, batch_size - num_valid, returned_timesteps, encoder_dim).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
                zeros = Variable(zeros)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.data.new(num_layers, batch_size, sequence_length_difference, stacked_sequence_output[0].size(-1)).fill_(0)
            zeros = Variable(zeros)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)
        self._update_states(final_states, restoration_indices)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self, inputs: PackedSequence, initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) ->Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
    Parameters
    ----------
    inputs : ``PackedSequence``, required.
      A batch first ``PackedSequence`` to run the stacked LSTM over.
    initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
      A tuple (state, memory) representing the initial hidden state and memory
      of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
      (num_layers, batch_size, 2 * cell_size) respectively.
    Returns
    -------
    output_sequence : ``torch.FloatTensor``
      The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
    final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
      The per-layer final (state, memory) states of the LSTM, with shape
      (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
      respectively. The last dimension is duplicated because it contains the state/memory
      for both the forward and backward layers.
    """
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise Exception('Initial states were passed to forward() but the number of initial states does not match the number of layers.')
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))
        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs
        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence
            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = forward_hidden_state, forward_memory_state
                backward_state = backward_hidden_state, backward_memory_state
            else:
                forward_state = None
                backward_state = None
            forward_output_sequence, forward_state = forward_layer(forward_output_sequence, batch_lengths, forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence, batch_lengths, backward_state)
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache
            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], -1))
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1), torch.cat([forward_state[1], backward_state[1]], -1)))
        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor, torch.FloatTensor] = (torch.cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple


class LstmTokenEmbedder(nn.Module):

    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
        super(LstmTokenEmbedder, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.word_emb_layer = word_emb_layer
        self.char_emb_layer = char_emb_layer
        self.output_dim = config['encoder']['projection_dim']
        emb_dim = 0
        if word_emb_layer is not None:
            emb_dim += word_emb_layer.n_d
        if char_emb_layer is not None:
            emb_dim += char_emb_layer.n_d * 2
            self.char_lstm = nn.LSTM(char_emb_layer.n_d, char_emb_layer.n_d, num_layers=1, bidirectional=True, batch_first=True, dropout=config['dropout'])
        self.projection = nn.Linear(emb_dim, self.output_dim, bias=True)

    def forward(self, word_inp, chars_inp, shape):
        embs = []
        batch_size, seq_len = shape
        if self.word_emb_layer is not None:
            word_emb = self.word_emb_layer(Variable(word_inp) if self.use_cuda else Variable(word_inp))
            embs.append(word_emb)
        if self.char_emb_layer is not None:
            chars_inp = chars_inp.view(batch_size * seq_len, -1)
            chars_emb = self.char_emb_layer(Variable(chars_inp) if self.use_cuda else Variable(chars_inp))
            _, (chars_outputs, __) = self.char_lstm(chars_emb)
            chars_outputs = chars_outputs.contiguous().view(-1, self.config['token_embedder']['char_dim'] * 2)
            embs.append(chars_outputs)
        token_embedding = torch.cat(embs, dim=2)
        return self.projection(token_embedding)


class LstmbiLm(nn.Module):

    def __init__(self, config, use_cuda=False):
        super(LstmbiLm, self).__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.encoder = nn.LSTM(self.config['encoder']['projection_dim'], self.config['encoder']['dim'], num_layers=self.config['encoder']['n_layers'], bidirectional=True, batch_first=True, dropout=self.config['dropout'])
        self.projection = nn.Linear(self.config['encoder']['dim'], self.config['encoder']['projection_dim'], bias=True)

    def forward(self, inputs):
        forward, backward = self.encoder(inputs)[0].split(self.config['encoder']['dim'], 2)
        return torch.cat([self.projection(forward), self.projection(backward)], dim=2)


class Model(nn.Module):

    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.config = config
        if config['token_embedder']['name'].lower() == 'cnn':
            self.token_embedder = ConvTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
        elif config['token_embedder']['name'].lower() == 'lstm':
            self.token_embedder = LstmTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
        if config['encoder']['name'].lower() == 'elmo':
            self.encoder = ElmobiLm(config, use_cuda)
        elif config['encoder']['name'].lower() == 'lstm':
            self.encoder = LstmbiLm(config, use_cuda)
        self.output_dim = config['encoder']['projection_dim']

    def forward(self, word_inp, chars_package, mask_package):
        """

    :param word_inp:
    :param chars_package:
    :param mask_package:
    :return:
    """
        token_embedding = self.token_embedder(word_inp, chars_package, (mask_package[0].size(0), mask_package[0].size(1)))
        if self.config['encoder']['name'] == 'elmo':
            mask = Variable(mask_package[0]) if self.use_cuda else Variable(mask_package[0])
            encoder_output = self.encoder(token_embedding, mask)
            sz = encoder_output.size()
            token_embedding = torch.cat([token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
            encoder_output = torch.cat([token_embedding, encoder_output], dim=0)
        elif self.config['encoder']['name'] == 'lstm':
            encoder_output = self.encoder(token_embedding)
        else:
            raise ValueError('Unknown encoder: {0}'.format(self.config['encoder']['name']))
        return encoder_output

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'), map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'), map_location=lambda storage, loc: storage))


class SoftmaxLayer(nn.Module):
    """ Naive softmax-layer """

    def __init__(self, output_dim, n_class):
        """

    :param output_dim: int
    :param n_class: int
    """
        super(SoftmaxLayer, self).__init__()
        self.hidden2tag = nn.Linear(output_dim, n_class)
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, x, y):
        """

    :param x: torch.Tensor
    :param y: torch.Tensor
    :return:
    """
        tag_scores = self.hidden2tag(x)
        return self.criterion(tag_scores, y)


class SampledSoftmaxLayer(nn.Module):
    """

  """

    def __init__(self, output_dim, n_class, n_samples, use_cuda):
        """

    :param output_dim:
    :param n_class:
    :param n_samples:
    :param use_cuda:
    """
        super(SampledSoftmaxLayer, self).__init__()
        self.n_samples = n_samples
        self.n_class = n_class
        self.use_cuda = use_cuda
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.negative_samples = []
        self.word_to_column = {(0): 0}
        self.all_word = []
        self.all_word_to_column = {(0): 0}
        self.column_emb = nn.Embedding(n_class, output_dim)
        self.column_emb.weight.data.uniform_(-0.25, 0.25)
        self.column_bias = nn.Embedding(n_class, 1)
        self.column_bias.weight.data.uniform_(-0.25, 0.25)
        self.oov_column = nn.Parameter(torch.Tensor(output_dim, 1))
        self.oov_column.data.uniform_(-0.25, 0.25)

    def forward(self, x, y):
        if self.training:
            for i in range(y.size(0)):
                y[i] = self.word_to_column.get(y[i].tolist())
            samples = torch.LongTensor(len(self.word_to_column)).fill_(0)
            for word in self.negative_samples:
                samples[self.word_to_column[word]] = word
        else:
            for i in range(y.size(0)):
                y[i] = self.all_word_to_column.get(y[i].tolist(), 0)
            samples = torch.LongTensor(len(self.all_word_to_column)).fill_(0)
            for word in self.all_word:
                samples[self.all_word_to_column[word]] = word
        if self.use_cuda:
            samples = samples
        tag_scores = x.matmul(self.embedding_matrix).view(y.size(0), -1) + self.column_bias.forward(samples).view(1, -1)
        return self.criterion(tag_scores, y)

    def update_embedding_matrix(self):
        word_inp, chars_inp = [], []
        if self.training:
            columns = torch.LongTensor(len(self.negative_samples) + 1)
            samples = self.negative_samples
            for i, word in enumerate(samples):
                columns[self.word_to_column[word]] = word
            columns[0] = 0
        else:
            columns = torch.LongTensor(len(self.all_word) + 1)
            samples = self.all_word
            for i, word in enumerate(samples):
                columns[self.all_word_to_column[word]] = word
            columns[0] = 0
        if self.use_cuda:
            columns = columns
        self.embedding_matrix = self.column_emb.forward(columns).transpose(0, 1)

    def update_negative_samples(self, word_inp, chars_inp, mask):
        batch_size, seq_len = word_inp.size(0), word_inp.size(1)
        in_batch = set()
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    continue
                word = word_inp[i][j].tolist()
                in_batch.add(word)
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    continue
                word = word_inp[i][j].tolist()
                if word not in self.all_word_to_column:
                    self.all_word.append(word)
                    self.all_word_to_column[word] = len(self.all_word_to_column)
                if word not in self.word_to_column:
                    if len(self.negative_samples) < self.n_samples:
                        self.negative_samples.append(word)
                        self.word_to_column[word] = len(self.word_to_column)
                    else:
                        while self.negative_samples[0] in in_batch:
                            self.negative_samples = self.negative_samples[1:] + [self.negative_samples[0]]
                        self.word_to_column[word] = self.word_to_column.pop(self.negative_samples[0])
                        self.negative_samples = self.negative_samples[1:] + [word]


class CNNSoftmaxLayer(nn.Module):

    def __init__(self, token_embedder, output_dim, n_class, n_samples, corr_dim, use_cuda):
        super(CNNSoftmaxLayer, self).__init__()
        self.token_embedder = token_embedder
        self.n_samples = n_samples
        self.use_cuda = use_cuda
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.negative_samples = []
        self.word_to_column = {(0): 0}
        self.all_word = []
        self.all_word_to_column = {(0): 0}
        self.M = nn.Parameter(torch.Tensor(output_dim, corr_dim))
        stdv = 1.0 / math.sqrt(self.M.size(1))
        self.M.data.uniform_(-stdv, stdv)
        self.corr = nn.Embedding(n_class, corr_dim)
        self.corr.weight.data.uniform_(-0.25, 0.25)
        self.oov_column = nn.Parameter(torch.Tensor(output_dim, 1))
        self.oov_column.data.uniform_(-0.25, 0.25)

    def forward(self, x, y):
        if self.training:
            for i in range(y.size(0)):
                y[i] = self.word_to_column.get(y[i].tolist())
            samples = torch.LongTensor(len(self.word_to_column)).fill_(0)
            for package in self.negative_samples:
                samples[self.word_to_column[package[0]]] = package[0]
        else:
            for i in range(y.size(0)):
                y[i] = self.all_word_to_column.get(y[i].tolist(), 0)
            samples = torch.LongTensor(len(self.all_word_to_column)).fill_(0)
            for package in self.all_word:
                samples[self.all_word_to_column[package[0]]] = package[0]
        if self.use_cuda:
            samples = samples
        tag_scores = x.matmul(self.embedding_matrix).view(y.size(0), -1) + x.matmul(self.M).matmul(self.corr.forward(samples).transpose(0, 1)).view(y.size(0), -1)
        return self.criterion(tag_scores, y)

    def update_embedding_matrix(self):
        batch_size = 2048
        word_inp, chars_inp = [], []
        if self.training:
            sub_matrices = [self.oov_column]
            samples = self.negative_samples
            id2pack = {}
            for i, package in enumerate(samples):
                id2pack[self.word_to_column[package[0]]] = i
        else:
            sub_matrices = [self.oov_column]
            samples = self.all_word
            id2pack = {}
            for i, package in enumerate(samples):
                id2pack[self.all_word_to_column[package[0]]] = i
        for i in range(len(samples)):
            word_inp.append(samples[id2pack[i + 1]][0])
            chars_inp.append(samples[id2pack[i + 1]][1])
            if len(word_inp) == batch_size or i == len(samples) - 1:
                sub_matrices.append(self.token_embedder.forward(torch.LongTensor(word_inp).view(len(word_inp), 1), None if chars_inp[0] is None else torch.LongTensor(chars_inp).view(len(word_inp), 1, len(package[1])), (len(word_inp), 1)).squeeze(1).transpose(0, 1))
                if not self.training:
                    sub_matrices[-1] = sub_matrices[-1].detach()
                word_inp, chars_inp = [], []
        sum = 0
        for mat in sub_matrices:
            sum += mat.size(1)
        self.embedding_matrix = torch.cat(sub_matrices, dim=1)

    def update_negative_samples(self, word_inp, chars_inp, mask):
        batch_size, seq_len = word_inp.size(0), word_inp.size(1)
        in_batch = set()
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    continue
                word = word_inp[i][j].tolist()
                in_batch.add(word)
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    continue
                package = word_inp[i][j].tolist(), None if chars_inp is None else chars_inp[i][j].tolist()
                if package[0] not in self.all_word_to_column:
                    self.all_word.append(package)
                    self.all_word_to_column[package[0]] = len(self.all_word_to_column)
                if package[0] not in self.word_to_column:
                    if len(self.negative_samples) < self.n_samples:
                        self.negative_samples.append(package)
                        self.word_to_column[package[0]] = len(self.word_to_column)
                    else:
                        while self.negative_samples[0][0] in in_batch:
                            self.negative_samples = self.negative_samples[1:] + [self.negative_samples[0]]
                        self.word_to_column[package[0]] = self.word_to_column.pop(self.negative_samples[0][0])
                        self.negative_samples = self.negative_samples[1:] + [package]


class EmbeddingLayer(nn.Module):

    def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        if embs is not None:
            embwords, embvecs = embs
            logging.info('{} pre-trained word embeddings loaded.'.format(len(word2id)))
            if n_d != len(embvecs[0]):
                logging.warning('[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.'.format(n_d, len(embvecs[0]), len(embvecs[0])))
                n_d = len(embvecs[0])
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            logging.info('embedding shape: {}'.format(weight.size()))
        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))
        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input_):
        return self.embedding(input_)


log = logging.getLogger()


class MLP(nn.Module):
    """
    b: batch_size, n: seq_len, d: embedding_size
    """

    def __init__(self, config):
        super().__init__()
        opt = config['mlp']
        self.max_length = opt['max_length']
        dropout = opt['dropout']
        u = opt['hidden_size']
        self.mlp = nn.Sequential(nn.Linear(self.max_length * config['embedding_size'], u), nn.ReLU(), nn.Dropout(dropout), nn.Linear(u, config['num_labels']))
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        data = torch.stack(embedding)
        output = self.mlp(data.view(data.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


class CNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        opt = config['cnn']
        self.cnn_1 = nn.Sequential(nn.Conv1d(config['embedding_size'], opt['conv_1']['size'], opt['conv_1']['kernel_size'], padding=opt['conv_1']['kernel_size'] // 2), nn.ReLU(), nn.Dropout(opt['conv_1']['dropout']), nn.MaxPool1d(opt['max_pool_1']['kernel_size'], opt['max_pool_1']['stride']))
        """
        self.cnn_2 = nn.Sequential(
            nn.Conv1d(opt['conv_1']['size'], opt['conv_2']['size'], opt['conv_2']['kernel_size'],
                      padding=opt['conv_2']['kernel_size'] // 2),
            nn.ReLU(),
            nn.Dropout(opt['conv_2']['dropout']),
            nn.MaxPool1d(opt['max_pool_2']['kernel_size'], opt['max_pool_2']['stride']),
        )
        """
        mlp_u = opt['fc']['hidden_size']
        self.mlp = nn.Sequential(nn.Linear(opt['conv_1']['size'] * opt['max_length'] // 2, mlp_u), nn.ReLU(), nn.Dropout(opt['fc']['dropout']), nn.Linear(mlp_u, config['num_labels']))
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        data = torch.stack(embedding).transpose(1, 2)
        out_1 = self.cnn_1(data)
        output = self.mlp(out_1.view(out_1.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


class RNN(nn.Module):
    """
    b: batch_size, n: seq_len, u: rnn_hidden_size, da: param_da, r: param_r, d: embedding_size
    """

    def __init__(self, config):
        super().__init__()
        opt = config['rnn']
        u = opt['rnn_hidden_size']
        da = opt['param_da']
        r = opt['param_r']
        d = config['embedding_size']
        num_layers = opt['num_layers']
        bidirectional = opt['bidirectional']
        if opt['type'] == 'lstm':
            self.rnn = nn.LSTM(input_size=d, hidden_size=u, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif opt['type'] == 'gru':
            self.rnn = nn.GRU(input_size=d, hidden_size=u, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            log.fatal('Invalid rnn type. Should be "lstm" or "gru"')
        if bidirectional:
            u = u * 2
        mlp_u = opt['mlp_hidden_size']
        self.mlp = nn.Sequential(nn.Linear(r * u, mlp_u), nn.ReLU(), nn.Dropout(opt['dropout']), nn.Linear(mlp_u, config['num_labels']))
        self.Ws1 = nn.Parameter(torch.randn(da, u))
        self.Ws2 = nn.Parameter(torch.randn(r, da))
        self.p_c = opt['p_coefficient']
        self.loss_type = opt['loss']
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            log.fatal('Invalid loss type. Should be "l1" or "cross_entropy"')

    def forward(self, embedding, gold_labels=None):
        """
        :param embedding: [b, n, d]
        :param gold_labels: [b, num_labels]
        :return: If training, return (loss, predicted labels). Else return predicted labels
        """
        padded = nn.utils.rnn.pad_sequence(embedding, batch_first=True)
        H = self.rnn(padded)[0]
        A = F.softmax(torch.matmul(self.Ws2, torch.tanh(torch.matmul(self.Ws1, H.transpose(1, 2)))), dim=2)
        M = torch.matmul(A, H)
        output = self.mlp(M.view(M.size(0), -1))
        labels = F.softmax(output, dim=1)
        if not self.training:
            return labels.detach()
        I = torch.eye(A.size(1))
        if is_gpu:
            I = I
        tmp = torch.matmul(A, A.transpose(1, 2)) - I
        P = (tmp * tmp).sum() / A.size(0)
        loss = self.p_c * P
        if self.loss_type == 'cross_entropy':
            loss = self.loss(output, torch.argmax(gold_labels, dim=1))
        else:
            loss = self.loss(labels, gold_labels)
        return loss, labels.detach()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LstmCellWithProjection,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'cell_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), [4, 4, 4, 4]], {}),
     False),
    (SoftmaxLayer,
     lambda: ([], {'output_dim': 4, 'n_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_xalanq_chinese_sentiment_classification(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

