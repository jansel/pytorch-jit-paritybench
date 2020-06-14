import sys
_module = sys.modules[__name__]
del sys
jupyter_notebook_config = _module
setup = _module
src = _module
quicknlp = _module
callbacks = _module
data = _module
data_loaders = _module
datasets = _module
dialogue_analysis = _module
dialogue_model_data_loader = _module
hierarchical_model_data_loader = _module
iterators = _module
learners = _module
model_helpers = _module
s2s_model_data_loader = _module
sampler = _module
spacy_tokenizer = _module
torchtext_data_loaders = _module
vocab = _module
metrics = _module
models = _module
cvae = _module
hred = _module
hred_attention = _module
hred_constrained = _module
seq2seq = _module
seq2seq_attention = _module
transformer = _module
modules = _module
attention = _module
attention_decoder = _module
basic_decoder = _module
basic_encoder = _module
cell = _module
embeddings = _module
hred_encoder = _module
projection = _module
rnn_encoder = _module
transformer = _module
stepper = _module
utils = _module
conftest = _module
test_attention = _module
test_attention_projection = _module
test_cell = _module
test_cvae = _module
test_datasets = _module
test_decoder = _module
test_hierarchical_data_loader = _module
test_hierarichal_datasets = _module
test_hred = _module
test_iterator = _module
test_rnn_encoder = _module
test_sentence_data = _module
test_seq2seq = _module
test_spacy_tokenizer = _module
test_transformer = _module
test_transformer_modules = _module
test_utils = _module

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


from functools import partial


import torch


from torch.nn import functional as F


from typing import List


from typing import Union


import torch.nn as nn


import numpy as np


import torch as tr


import torch.nn.functional as F


from torch import nn as nn


import math


from torch.nn import Parameter


from collections import OrderedDict


import warnings


from inspect import signature


from typing import Any


from typing import Callable


from typing import Optional


from typing import Sequence


from typing import Tuple


Array = Union[np.ndarray, torch.Tensor, int, float]


def assert_dims(value: Sequence[Array], dims: List[Optional[int]]) ->Sequence[
    Array]:
    """Given a nested sequence, with possibly torch or nympy tensors inside, assert it agrees with the
        dims provided

    Args:
        value (Sequence[Array]): A sequence of sequences with potentially arrays inside
        dims (List[Optional[int]]: A list with the expected dims. None is used if the dim size can be anything

    Raises:
        AssertionError if the value does not comply with the dims provided
    """
    if isinstance(value, list):
        if dims[0] is not None:
            assert len(value) == dims[0], f'{value} does not match {dims}'
            for row in value:
                assert_dims(row, dims[1:])
    elif hasattr(value, 'shape'):
        shape = value.shape
        assert len(shape) == len(dims), f'{shape} does not match {dims}'
        for actual_dim, expected_dim in zip(shape, dims):
            if expected_dim is not None:
                if isinstance(expected_dim, tuple):
                    assert actual_dim in expected_dim, f'{shape} does not match {dims}'
                else:
                    assert actual_dim == expected_dim, f'{shape} does not match {dims}'
    return value


def get_kwarg(kwargs, name, default_value=None, remove=True):
    """Returns the value for the parameter if it exists in the kwargs otherwise the default value provided"""
    if remove:
        value = kwargs.pop(name) if name in kwargs else default_value
    else:
        value = kwargs.get(name, default_value)
    return value


def get_list(value: Union[List[Any], Any], multiplier: int=1) ->List[Any]:
    if isinstance(value, list):
        assert len(value
            ) == multiplier, f'{value} is not the correct size {multiplier}'
    else:
        value = [value] * multiplier
    return value


States = Union[List[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    torch.Tensor]


def concat_layer_bidir_state(states: States, bidir):
    if isinstance(states, (list, tuple)) and bidir:
        return states[0].transpose(1, 0).contiguous().view(1, -1, 2 *
            states[0].size(-1)), states[1].transpose(1, 0).contiguous().view(
            1, -1, 2 * states[1].size(-1))
    elif bidir:
        return states.transpose(1, 0).contiguous().view(1, -1, 2 * states[0
            ].size(-1))
    else:
        return states


def concat_bidir_state(states: States, bidir: bool, cell_type: str, nlayers:
    int) ->States:
    if isinstance(states, list):
        state = []
        for index in range(len(states)):
            state.append(concat_layer_bidir_state(states[index], bidir=bidir))
    else:
        state = concat_layer_bidir_state(states, bidir=bidir)
    return state


def repeat_cell_state(hidden, num_beams):
    results = []
    for row in hidden:
        if isinstance(row, (list, tuple)):
            state = row[0].repeat(1, num_beams, 1), row[1].repeat(1,
                num_beams, 1)
        else:
            state = row.repeat(1, num_beams, 1)
        results.append(state)
    return results


def reshape_parent_indices(indices, bs, num_beams):
    parent_indices = V((torch.arange(end=bs) * num_beams).unsqueeze_(1).
        repeat(1, num_beams).view(-1).long())
    return indices + parent_indices


class MLPAttention(nn.Module):
    """Multilayer Perceptron Attention Bandandau et al. 2015"""

    def __init__(self, n_in, nhid, p=0.0):
        """

        Args:
            n_in (int):  The input dims of the first linear layer. It should equal
                    the sum of the keys and query dims
            nhid (int): The dimension of the internal prediction.
        """
        super().__init__()
        self.dropout = LockedDropout(p) if p > 0.0 else None
        self.linear1 = nn.Linear(in_features=n_in, out_features=nhid, bias=
            False)
        self.linear2 = nn.Linear(in_features=nhid, out_features=1, bias=False)

    def forward(self, query, keys, values):
        inputs = tr.cat([query.unsqueeze(0).repeat(keys.size(0), 1, 1),
            keys], dim=-1)
        scores = self.linear2(F.tanh(self.linear1(inputs)))
        scores = F.softmax(scores, dim=0)
        if self.dropout is not None:
            scores = self.dropout(scores)
        return (scores * values).sum(dim=0)


class SDPAttention(nn.Module):
    """Scaled Dot Product Attention Vaswani et al. 2017"""

    def __init__(self, n_in, p=0.0):
        super().__init__()
        self.dropout = LockedDropout(p) if p > 0.0 else None
        self.scale = np.sqrt(n_in)

    def forward(self, query, keys, values):
        dot = (query * keys).sum(dim=-1) / self.scale
        weights = F.softmax(dot, dim=0).unsqueeze(-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return (weights * values).sum(0)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, nhid, keys_dim, query_dim, values_dim,
        dropout=0.0, out_dim=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.num_heads = num_heads
        self.nhid = nhid
        self.linear_out_dim = self.nhid * num_heads
        self.out_dim = self.linear_out_dim if out_dim is None else out_dim
        self.keys_linear = nn.Linear(in_features=keys_dim, out_features=
            self.linear_out_dim, bias=False)
        self.query_linear = nn.Linear(in_features=query_dim, out_features=
            self.linear_out_dim, bias=False)
        self.values_linear = nn.Linear(in_features=values_dim, out_features
            =self.linear_out_dim, bias=False)
        self.scale = np.sqrt(self.nhid)
        self.linear = nn.Linear(in_features=self.linear_out_dim,
            out_features=self.out_dim, bias=False)

    def forward(self, query, keys, values, mask=None):
        sl, bs, dimK = keys.size()
        slq = query.size(0)
        query_projection = self.query_linear(query).view(slq, bs, self.
            num_heads, self.nhid).permute(1, 2, 0, 3)
        keys_projection = self.keys_linear(keys).view(sl, bs, self.
            num_heads, self.nhid).permute(1, 2, 3, 0)
        values_projection = self.values_linear(values).view(sl, bs, self.
            num_heads, self.nhid).permute(1, 2, 0, 3)
        scores = query_projection @ keys_projection
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e+20)
        weights = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            weights = self.dropout(weights)
        attention = (weights @ values_projection).permute(2, 0, 1, 3
            ).contiguous().view(slq, bs, self.num_heads * self.nhid)
        output = self.linear(attention)
        return assert_dims(output, [slq, bs, self.out_dim])


class RandomUniform:

    def __init__(self, numbers=1000000):
        self.numbers = numbers
        self.array = np.random.rand(numbers)
        self.count = 0

    def __call__(self, *args, **kwargs):
        if self.count >= self.array.size:
            self.count = 0
            self.array = np.random.rand(self.numbers)
        rand = self.array[self.count]
        self.count += 1
        return rand


def select_hidden_by_index(hidden, indices):
    if hidden is None:
        return hidden
    results = []
    for row in hidden:
        if isinstance(row, (list, tuple)):
            state = torch.index_select(row[0], 1, indices), torch.index_select(
                row[1], 1, indices)
        else:
            state = torch.index_select(row, 1, indices)
        results.append(state)
    return results


class Decoder(nn.Module):
    MAX_STEPS_ALLOWED = 320

    def __init__(self, decoder_layer, projection_layer, max_tokens,
        eos_token, pad_token, embedding_layer: torch.nn.Module):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.nlayers = decoder_layer.nlayers
        self.projection_layer = projection_layer
        self.bs = 1
        self.max_iterations = max_tokens
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.beam_outputs = None
        self.embedding_layer = embedding_layer
        self.emb_size = embedding_layer.emb_size
        self.pr_force = 0.0
        self.random = RandomUniform()

    def reset(self, bs):
        self.decoder_layer.reset(bs)

    def forward(self, inputs, hidden=None, num_beams=0, constraints=None):
        self.bs = inputs.size(1)
        if num_beams == 0:
            return self._train_forward(inputs, hidden, constraints)
        elif num_beams == 1:
            return self._greedy_forward(inputs, hidden, constraints)
        elif num_beams > 1:
            return self._beam_forward(inputs, hidden, num_beams, constraints)

    def _beam_forward(self, inputs, hidden, num_beams, constraints=None):
        return self._topk_forward(inputs, hidden, num_beams, constraints)

    def _train_forward(self, inputs, hidden=None, constraints=None):
        inputs = self.embedding_layer(inputs)
        if constraints is not None:
            inputs = torch.cat([inputs, constraints.repeat(inputs.size(0), 
                1, 1)], dim=-1)
        outputs = self.decoder_layer(inputs, hidden)
        outputs = self.projection_layer(outputs[-1]
            ) if self.projection_layer is not None else outputs[-1]
        return outputs

    def _greedy_forward(self, inputs, hidden=None, constraints=None):
        dec_inputs = inputs
        max_iterations = min(dec_inputs.size(0), self.MAX_STEPS_ALLOWED
            ) if self.training else self.max_iterations
        inputs = V(inputs[:1].data)
        sl, bs = inputs.size()
        finished = to_gpu(torch.zeros(bs).byte())
        iteration = 0
        self.beam_outputs = inputs.clone()
        final_outputs = []
        while not finished.all() and iteration < max_iterations:
            if 0 < iteration and self.training and 0.0 < self.random(
                ) < self.pr_force:
                inputs = dec_inputs[iteration].unsqueeze(0)
            output = self.forward(inputs, hidden=hidden, num_beams=0,
                constraints=constraints)
            hidden = self.decoder_layer.hidden
            final_outputs.append(output)
            inputs = assert_dims(V(output.data.max(dim=-1)[1]), [1, bs])
            iteration += 1
            self.beam_outputs = assert_dims(torch.cat([self.beam_outputs,
                inputs], dim=0), [iteration + 1, bs])
            new_finished = inputs.data == self.eos_token
            finished = finished | new_finished
        self.beam_outputs = self.beam_outputs.view(-1, bs, 1)
        outputs = torch.cat(final_outputs, dim=0)
        return outputs

    def _topk_forward(self, inputs, hidden, num_beams, constraints=None):
        sl, bs = inputs.size()
        logprobs = torch.zeros_like(inputs[:1]).view(1, bs, 1).float()
        inputs = inputs[:1].repeat(1, num_beams)
        finished = to_gpu(torch.zeros(bs * num_beams).byte())
        iteration = 0
        final_outputs = []
        self.beam_outputs = inputs.clone()
        hidden = repeat_cell_state(hidden, num_beams)
        while not finished.all() and iteration < self.max_iterations:
            output = self.forward(inputs, hidden=hidden, num_beams=0,
                constraints=constraints)
            hidden = self.decoder_layer.hidden
            final_outputs.append(output)
            new_logprobs = F.log_softmax(output, dim=-1)
            num_tokens = new_logprobs.size(2)
            new_logprobs = new_logprobs.view(1, bs, num_beams, num_tokens
                ) + logprobs.unsqueeze(-1)
            new_logprobs = self.mask_logprobs(bs, finished, iteration,
                logprobs, new_logprobs, num_beams, num_tokens)
            logprobs, beams = torch.topk(new_logprobs, k=num_beams, dim=-1)
            parents = beams / num_tokens
            inputs = beams % num_tokens
            parent_indices = reshape_parent_indices(parents.view(-1), bs=bs,
                num_beams=num_beams)
            self.decoder_layer.hidden = select_hidden_by_index(self.
                decoder_layer.hidden, indices=parent_indices)
            finished = torch.index_select(finished, 0, parent_indices.data)
            inputs = inputs.view(1, -1).contiguous()
            self.beam_outputs = torch.index_select(self.beam_outputs, dim=1,
                index=parent_indices)
            self.beam_outputs = torch.cat([self.beam_outputs, inputs], dim=0)
            new_finished = (inputs.data == self.eos_token).view(-1)
            finished = finished | new_finished
            iteration += 1
        self.beam_outputs = self.beam_outputs.view(-1, bs, num_beams)
        outputs = torch.cat(final_outputs, dim=0)
        return outputs

    def mask_logprobs(self, bs, finished, iteration, logprobs, new_logprobs,
        num_beams, num_tokens):
        if iteration == 0:
            new_logprobs = new_logprobs[(...), (0), :]
        else:
            mask = torch.zeros_like(new_logprobs).fill_(-1e+32).view(1, bs *
                num_beams, num_tokens)
            f = V(finished.unsqueeze(0))
            mask[..., self.pad_token] = logprobs.view(1, bs * num_beams)
            mask = mask.masked_select(f.unsqueeze(-1)).view(1, -1, num_tokens)
            new_logprobs.masked_scatter_(f.view(1, bs, num_beams, 1), mask)
            new_logprobs = new_logprobs.view(1, bs, -1)
        return new_logprobs

    @property
    def hidden(self):
        return self.decoder_layer.hidden

    @hidden.setter
    def hidden(self, value):
        self.decoder_layer.hidden = value

    @property
    def layers(self):
        return self.decoder_layer.layers

    @property
    def output_size(self):
        return (self.projection_layer.output_size if self.projection_layer
             is not None else self.decoder_layer.output_size)


class Encoder(nn.Module):

    def __init__(self, embedding_layer, encoder_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer

    def forward(self, input_tensor, state=None):
        ed = self.embedding_layer(input_tensor)
        return self.encoder_layer(ed, state)

    def reset(self, bs):
        self.encoder_layer.reset(bs)

    @property
    def hidden(self):
        return self.encoder_layer.hidden

    @hidden.setter
    def hidden(self, value):
        self.encoder_layer.hidden = value

    @property
    def layers(self):
        return self.encoder_layer.layers

    @property
    def output_size(self):
        return self.encoder_layer.output_size


class Cell(nn.Module):
    """GRU or LSTM cell with withdrop. Can also be bidirectional and have trainable initial state"""

    def __init__(self, cell_type, input_size, output_size, dropout=0.0,
        wdrop=0.0, dropoutinit=0.0, bidir=False, train_init=False):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.bidir = bidir
        self.input_size = input_size
        self.output_size = output_size
        self.dropoutinit = dropoutinit
        if self.cell_type == 'lstm':
            self.cell = nn.LSTM(input_size, output_size, num_layers=1,
                bidirectional=bidir, dropout=dropout)
        elif self.cell_type == 'gru':
            self.cell = nn.GRU(input_size, output_size, num_layers=1,
                bidirectional=bidir, dropout=dropout)
        else:
            raise NotImplementedError(f'cell: {cell_type} not supported')
        if wdrop:
            self.cell = WeightDrop(self.cell, wdrop)
        self.train_init = train_init
        self.init_state = None
        self.init_cell_state = None
        if self.train_init:
            ndir = 2 if bidir else 1
            self.init_state = Parameter(torch.Tensor(ndir, 1, self.output_size)
                )
            stdv = 1.0 / math.sqrt(self.init_state.size(1))
            self.init_state.data.uniform_(-stdv, stdv)
            if self.cell_type == 'lstm':
                ndir = 2 if bidir else 1
                self.init_cell_state = Parameter(torch.Tensor(ndir, 1, self
                    .output_size))
                stdv = 1.0 / math.sqrt(self.init_state.size(1))
                self.init_cell_state.data.uniform_(-stdv, stdv)
        self.reset(bs=1)

    def forward(self, inputs, hidden):
        """
        LSTM Inputs: input, (h_0, c_0)
                    - **input** (seq_len, batch, input_size): tensor containing the features
                      of the input sequence.
                      The input can also be a packed variable length sequence.
                      See :func:`torch.nn.utils.rnn.pack_padded_sequence` for details.
                    - **h_0** (num_layers \\* num_directions, batch, hidden_size): tensor
                      containing the initial hidden state for each element in the batch.
                    - **c_0** (num_layers \\* num_directions, batch, hidden_size): tensor
                      containing the initial cell state for each element in the batch.

                      If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.
            Outputs: output, (h_n, c_n)
                - **output** (seq_len, batch, hidden_size * num_directions): tensor
                  containing the output features `(h_t)` from the last layer of the RNN,
                  for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the hidden state for t=seq_len
                - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the cell state for t=seq_len

        GRU: Inputs: input, h_0
                    - **input** (seq_len, batch, input_size): tensor containing the features
                      of the input sequence. The input can also be a packed variable length
                      sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
                      for details.
                    - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
                      containing the initial hidden state for each element in the batch.
                      Defaults to zero if not provided.
            Outputs: output, h_n
                - **output** (seq_len, batch, hidden_size * num_directions): tensor
                  containing the output features h_t from the last layer of the RNN,
                  for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
                  given as the input, the output will also be a packed sequence.
                - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
                  containing the hidden state for t=seq_len


        """
        return self.cell(inputs, hidden)

    def one_hidden(self, bs=1, cell_state=False):
        ndir = 2 if self.bidir else 1
        if not self.train_init:
            init_state = to_gpu(torch.zeros(ndir, bs, self.output_size))
        elif cell_state:
            init_state = F.dropout(self.init_cell_state, p=self.dropoutinit,
                training=self.training)
            init_state.repeat(1, bs, 1)
        else:
            init_state = F.dropout(self.init_state, p=self.dropoutinit,
                training=self.training)
            return init_state.repeat(1, bs, 1)
        return init_state

    def hidden_state(self, bs):
        if self.cell_type == 'gru':
            return self.one_hidden(bs)
        else:
            return self.one_hidden(bs, cell_state=False), self.one_hidden(bs,
                cell_state=True)

    def reset(self, bs=1):
        self.hidden = self.hidden_state(bs=bs)

    def get_hidden_state(self):
        return self.hidden[0] if self.cell_type == 'lstm' else self.hidden


class NormEmbeddings(nn.Module):
    """Normalized embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"""

    def __init__(self, emb_size, tokens, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(tokens, emb_size, padding_idx=padding_idx
            )
        self.in_features = emb_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.in_features)

    @property
    def weight(self):
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """Sinusoid Positional embedding see http://nlp.seas.harvard.edu/2018/04/03/attention.html"""

    def __init__(self, input_size, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * -(math.log(
            10000.0) / input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + V(self.pe[:, :x.size(1)])
        return self.dropout(x)


class DropoutEmbeddings(nn.Module):
    initrange = 0.1

    def __init__(self, ntokens, emb_size, dropoute=0.1, dropouti=0.65,
        pad_token=None):
        """ Default Constructor for the DropoutEmbeddings class

        Args:
            ntokens (int): number of vocabulary (or tokens) in the source dataset
            emb_size (int): the embedding size to use to encode each token
            pad_token (int): the int value used for padding text.
            dropoute (float): dropout to apply to the embedding layer. zeros out tokens
            dropouti (float): dropout to apply to the input layer. zeros out features
        """
        super().__init__()
        self.encoder = nn.Embedding(ntokens, emb_size, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout_embedding = dropoute
        self.dropout_input = LockedDropout(dropouti)
        self.emb_size = emb_size

    def forward(self, input_tensor):
        emb = self.encoder_with_dropout(input_tensor, dropout=self.
            dropout_embedding if self.training else 0)
        return self.dropout_input(emb)

    @property
    def weight(self):
        return self.encoder.weight


class TransformerEmbeddings(nn.Module):

    def __init__(self, ntokens, emb_size, dropout, pad_token=None, max_len=5000
        ):
        super(TransformerEmbeddings, self).__init__()
        self.layers = nn.Sequential(NormEmbeddings(emb_size=emb_size,
            tokens=ntokens, padding_idx=pad_token), PositionalEncoding(
            input_size=emb_size, dropout=dropout, max_len=max_len))
        self.emb_size = emb_size

    def forward(self, input_tensor):
        return self.layers(input_tensor)

    @property
    def weight(self):
        return self.layers[0].weight


class Projection(nn.Module):
    initrange = 0.1

    def __init__(self, output_size: int, input_size: int, dropout: float,
        nhid: int=None, tie_encoder=None):
        super().__init__()
        layers = OrderedDict()
        self.dropout = LockedDropout(dropout)
        if nhid is not None:
            linear1 = nn.Linear(input_size, nhid)
            linear1.weight.data.uniform_(-self.initrange, self.initrange)
            layers['projection1'] = linear1
            dropout1 = nn.Dropout(dropout)
            layers['dropout'] = dropout1
        else:
            nhid = input_size
        linear2 = nn.Linear(nhid, output_size, bias=False)
        if tie_encoder:
            assert linear2.weight.shape == tie_encoder.weight.shape, 'tied encoder {} does not match projection {}'.format(
                tie_encoder.weight.shape, linear2.weight.shape)
            linear2.weight = tie_encoder.weight
        layers['projection2'] = linear2
        self.layers = nn.Sequential(layers)
        self.output_size = output_size

    def forward(self, projection_input):
        output = self.dropout(projection_input)
        decoded = output.view(output.size(0) * output.size(1), output.size(2))
        decoded = self.layers(decoded)
        return decoded.view(-1, projection_input.size(1), decoded.size(1))


class AttentionProjection(nn.Module):

    def __init__(self, output_size, input_size, dropout, att_nhid, att_type
        ='MLP', tie_encoder=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.keys = None
        self._attention_output = None
        self.attention = MLPAttention(n_in=input_size * 2, nhid=att_nhid
            ) if att_type == 'MLP' else SDPAttention(n_in=input_size)
        self.projection1 = Projection(output_size=input_size, input_size=
            input_size * 2, dropout=dropout)
        self.projection2 = Projection(output_size=output_size, input_size=
            input_size, dropout=dropout, tie_encoder=tie_encoder)

    def forward(self, input):
        assert_dims(input, [None, self.input_size])
        self._attention_output = self.attention(query=input, keys=self.keys,
            values=self.keys)
        output = torch.cat([input, self._attention_output], dim=-1).unsqueeze_(
            0)
        assert_dims(output, [1, None, self.input_size * 2])
        output = assert_dims(self.projection1(output), [1, None, self.
            input_size])
        projection = self.projection2(output)
        return assert_dims(projection, [1, None, self.output_size])

    def get_attention_output(self, raw_output):
        if self._attention_output is None:
            return torch.zeros_like(raw_output)
        else:
            return self._attention_output

    def reset(self, keys):
        self._attention_output = None
        self.keys = keys


def get_layer_dims(layer_index, total_layers, input_size, output_size, nhid,
    bidir):
    ndir = 2 if bidir else 1
    input_size = input_size if layer_index == 0 else nhid
    output_size = (nhid if layer_index != total_layers - 1 else output_size
        ) // ndir
    return input_size, output_size


class RNNLayers(nn.Module):
    """
    Wrote this class to allow for a multilayered RNN encoder. It is based the fastai RNN_Encoder class
    """

    def __init__(self, input_size, output_size, nhid, nlayers, dropouth=0.3,
        wdrop=0.5, bidir=False, cell_type='lstm', train_init=False,
        dropoutinit=0.1, **kwargs):
        """ Default Constructor for the RNNLayers class

        Args:
            input_size (int): the dimension of the input vectors
            output_size (int) the dimension of the output vectors
            nhid (int): number of hidden activation per layer
            nlayers (int): number of layers to use in the architecture
            dropouth (float): dropout to apply to the activations going from one  layer to another
            wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            bidir (bool): If true the cell will be bidirectional
            train_init (bool): If true the initial states will be trainable
            dropoutinit (float): The dropout to use in the initial states if trainable
            cell_type (str): Type of cell (default is LSTM)
        """
        super().__init__()
        layers = []
        for layer_index in range(nlayers):
            inp_size, out_size = get_layer_dims(layer_index=layer_index,
                total_layers=nlayers, input_size=input_size, output_size=
                output_size, nhid=nhid, bidir=bidir)
            layers.append(Cell(cell_type=cell_type, input_size=inp_size,
                output_size=out_size, bidir=bidir, wdrop=wdrop, train_init=
                train_init, dropoutinit=dropoutinit))
        self.layers = nn.ModuleList(layers)
        self.input_size, self.output_size, self.nhid, self.nlayers = (
            input_size, output_size, nhid, nlayers)
        self.cell_type, self.bidir = cell_type, bidir
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in
            range(nlayers)])
        self.hidden, self.weights = None, None
        self.reset(1)

    def forward(self, input_tensor, hidden=None):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input_tensor (Tensor): input of shape [sentence_length, batch_size, hidden_dim]
            hidden (List[Tensor]: state  of the encoder

        Returns:
            (Tuple[List[Tensor], List[Tensor]]):
            raw_outputs: list of tensors evaluated from each RNN layer without using dropouth,
            outputs: list of tensors evaluated from each RNN layer using dropouth,
            The outputs should have dims [sl,bs,layer_dims]
        """
        output = input_tensor
        self.hidden = self.hidden if hidden is None else hidden
        new_hidden, outputs = [], []
        for layer_index, (rnn, drop) in enumerate(zip(self.layers, self.
            dropouths)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output, new_h = rnn(output, self.hidden[layer_index])
            new_hidden.append(new_h)
            if layer_index != self.nlayers - 1:
                output = drop(output)
            outputs.append(output)
        self.hidden = new_hidden
        return outputs

    def reset_hidden(self, bs):
        self.hidden = [self.layers[l].hidden_state(bs) for l in range(self.
            nlayers)]

    def reset(self, bs):
        self.reset_hidden(bs)

    def hidden_shape(self, bs):
        if isinstance(self.layers[0].hidden_state(1), tuple):
            return [self.layers[l].hidden_state(bs)[0].shape for l in range
                (self.nlayers)]
        else:
            return [self.layers[l].hidden_state(bs).shape for l in range(
                self.nlayers)]

    def get_last_hidden_state(self):
        return self.hidden[-1][0] if self.cell_type == 'lstm' else self.hidden[
            -1]


class PositionFeedForward(nn.Module):

    def __init__(self, input_size, out_dim, nhid, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = out_dim
        self.nhid = nhid
        self.pff = nn.Sequential(nn.Linear(in_features=self.input_size,
            out_features=self.nhid), nn.ReLU(), nn.Dropout(dropout), nn.
            Linear(in_features=self.nhid, out_features=self.output_size))

    def forward(self, inputs):
        return self.pff(inputs)


class SubLayer(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.input_size = input_size,
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, input_tensor, sublayer):
        return self.layer_norm(input_tensor.add(self.dropout(sublayer(
            input_tensor))))


class AttentionLayer(nn.Module):

    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        self.input_size = input_size
        self.nhid = input_size // num_heads
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, nhid=self.
            nhid, out_dim=self.input_size, keys_dim=self.input_size,
            values_dim=self.input_size, query_dim=self.input_size, dropout=
            dropout)

    def causal_mask(self, bs, sl):
        return T(np.tril(np.ones((bs, self.num_heads, sl, sl)))).float()

    def forward(self, input_tensor, keys_vector, values_vector, mask=False):
        sl, bs, _ = keys_vector.size()
        mask = self.causal_mask(bs=bs, sl=sl) if mask else None
        outputs = self.attention(query=input_tensor, keys=keys_vector,
            values=values_vector, mask=mask)
        return outputs


class TransformerLayer(nn.Module):

    def __init__(self, input_size, num_heads, nhid=2048, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.nhid = input_size // num_heads
        self.attention = AttentionLayer(input_size=input_size, num_heads=
            num_heads, dropout=dropout)
        self.linear = PositionFeedForward(input_size=input_size, out_dim=
            input_size, nhid=nhid, dropout=dropout)
        self.sublayers = nn.ModuleList([SubLayer(input_size=input_size,
            dropout=dropout), SubLayer(input_size=input_size, dropout=dropout)]
            )

    def forward(self, input_tensor):
        attention_output = self.sublayers[0](input_tensor, lambda x: self.
            attention(x, x, x))
        ff_output = self.sublayers[1](attention_output, self.linear)
        return ff_output


class TransformerEncoderLayers(nn.Module):

    def __init__(self, num_layers, input_size, num_heads, nhid, dropout=0.1):
        super().__init__()
        nhid = get_list(nhid, num_layers)
        num_heads = get_list(num_heads, num_layers)
        self.layers = nn.ModuleList([TransformerLayer(input_size=input_size,
            nhid=nhid[i], dropout=dropout, num_heads=num_heads[i]) for i in
            range(num_layers)])

    def forward(self, *input_tensors):
        output_tensors = []
        inputs, *_ = input_tensors
        for layer in self.layers:
            inputs = layer(inputs)
            output_tensors.append(inputs)
        return output_tensors


class TransformerLayerDecoder(TransformerLayer):

    def __init__(self, input_size, num_heads, nhid, dropout=0.1):
        super().__init__(input_size=input_size, num_heads=num_heads, nhid=
            nhid, dropout=dropout)
        self.decoder_attention = AttentionLayer(input_size=input_size,
            num_heads=num_heads, dropout=dropout)
        self.sublayers.append(SubLayer(input_size=input_size, dropout=dropout))

    def forward(self, *inputs):
        encoder_input, decoder_input = assert_dims(inputs, [2, None, None,
            self.input_size])
        att_output = self.sublayers[0](decoder_input, lambda x: self.
            attention(x, x, x, mask=True))
        dec_att_output = self.sublayers[1](att_output, lambda x: self.
            decoder_attention(x, encoder_input, encoder_input))
        return self.sublayers[2](dec_att_output, self.linear)


class TransformerDecoderLayers(nn.Module):

    def __init__(self, nlayers, input_size, num_heads, nhid, dropout=0.1):
        super().__init__()
        self.nlayers = nlayers
        nhid = get_list(nhid, nlayers)
        num_heads = get_list(num_heads, nlayers)
        self.hidden = None
        self.input_size = input_size
        self.layers = nn.ModuleList([TransformerLayerDecoder(input_size=
            input_size, nhid=nhid[i], dropout=dropout, num_heads=num_heads[
            i]) for i in range(nlayers)])

    def forward(self, decoder_inputs, encoder_inputs):
        output_tensors = []
        sl, bs, input_size = decoder_inputs.size()
        dec_inputs = assert_dims(decoder_inputs, [sl, bs, self.input_size])
        encoder_inputs = assert_dims(encoder_inputs, [self.nlayers, None,
            bs, self.input_size])
        for enc_inputs, layer in zip(encoder_inputs, self.layers):
            dec_inputs = layer(enc_inputs, dec_inputs)
            output_tensors.append(dec_inputs)
        assert_dims(output_tensors, [self.nlayers, sl, bs, self.input_size])
        return output_tensors


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_outcastofmusic_quick_nlp(_paritybench_base):
    pass
    def test_000(self):
        self._check(NormEmbeddings(*[], **{'emb_size': 4, 'tokens': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(PositionFeedForward(*[], **{'input_size': 4, 'out_dim': 4, 'nhid': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(SDPAttention(*[], **{'n_in': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(SubLayer(*[], **{'input_size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), ReLU()], {})

    @_fails_compile()
    def test_004(self):
        self._check(TransformerLayer(*[], **{'input_size': 4, 'num_heads': 4}), [torch.rand([4, 4, 4])], {})

