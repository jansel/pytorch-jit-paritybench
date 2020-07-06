import sys
_module = sys.modules[__name__]
del sys
data = _module
generate_rmc = _module
generate_rnn = _module
relational_rnn_general = _module
relational_rnn_models = _module
rnn_models = _module
train_embeddings = _module
train_nth_farthest = _module
train_rmc = _module
train_rnn = _module

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


import torch


import numpy as np


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import time


import math


import torch.onnx


class RelationalMemory(nn.Module):
    """
    Constructs a `RelationalMemory` object.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      input_size: The size of input per step. i.e. the dimension of each input vector
      num_heads: The number of attention heads to use. Defaults to 1.
      num_blocks: Number of times to compute attention per time step. Defaults
        to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.
      name: Name of the module.
    Raises:
      ValueError: gate_style not one of [None, 'memory', 'unit'].
      ValueError: num_blocks is < 1.
      ValueError: attention_mlp_layers is < 1.
    """

    def __init__(self, mem_slots, head_size, input_size, num_tokens, num_heads=1, num_blocks=1, forget_bias=1.0, input_bias=0.0, gate_style='unit', attention_mlp_layers=2, key_size=None, use_adaptive_softmax=False, cutoffs=None):
        super(RelationalMemory, self).__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads
        self.mem_slots_plus_input = self.mem_slots + 1
        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(num_blocks))
        self.num_blocks = num_blocks
        if gate_style not in ['unit', 'memory', None]:
            raise ValueError("gate_style must be one of ['unit', 'memory', None]. got: {}.".format(gate_style))
        self.gate_style = gate_style
        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.format(attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers
        self.key_size = key_size if key_size else self.head_size
        self.value_size = self.head_size
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.total_qkv_size])
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.mem_size)] * self.attention_mlp_layers)
        self.attended_memory_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])
        self.attended_memory_layernorm2 = nn.LayerNorm([self.mem_slots_plus_input, self.mem_size])
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)
        self.num_gates = 2 * self.calculate_gate_size()
        self.input_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))
        self.dropout = nn.Dropout()
        self.num_tokens = num_tokens
        self.token_to_input_encoder = nn.Embedding(self.num_tokens, self.input_size)
        self.output_to_embed_decoder = nn.Linear(self.mem_slots * self.mem_size, self.input_size)
        self.use_adaptive_softmax = use_adaptive_softmax
        if not self.use_adaptive_softmax:
            self.embed_to_logit_decoder = nn.Linear(self.input_size, self.num_tokens)
            self.embed_to_logit_decoder.weight = self.token_to_input_encoder.weight
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(self.input_size, self.num_tokens, cutoffs=cutoffs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initial_state(self, batch_size, trainable=False):
        """
        Creates the initial memory.
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self.mem_slots, self.mem_size).
        Args:
          batch_size: The size of the batch.
          trainable: Whether the initial state is trainable. This is always True.
        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self.mem_slots, self.mem_size).
        """
        init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :, :self.mem_size]
        return init_state

    def multihead_attention(self, memory):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """
        qkv = self.qkv_projector(memory)
        qkv = self.qkv_layernorm(qkv)
        mem_slots = memory.shape[1]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads, self.qkv_size)
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)
        q *= self.key_size ** -0.5
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)
        output = torch.matmul(weights, v)
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))
        return new_memory

    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:
            return 0

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        memory = torch.tanh(memory)
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError('input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1')
            inputs = inputs.view(inputs.shape[0], -1)
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError('input shape of create_gate function is 2, expects 3')
        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        return input_gate, forget_gate

    def attend_over_memory(self, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(memory)
            memory = self.attended_memory_layernorm(memory + attended_memory)
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
        return memory

    def forward_step(self, inputs, memory, treat_input_as_matrix=False):
        """
        Forward step of the relational memory core.
        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.
          treat_input_as_matrix: Optional, whether to treat `input` as a sequence
            of matrices. Default to False, in which case the input is flattened
            into a vector.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """
        inputs_embed = self.dropout(self.token_to_input_encoder(inputs))
        if treat_input_as_matrix:
            inputs_embed = inputs_embed.view(inputs_embed.shape[0], inputs_embed.shape[1], -1)
            inputs_reshape = self.input_projector(inputs_embed)
        else:
            inputs_embed = inputs_embed.view(inputs_embed.shape[0], -1)
            inputs_embed = self.input_projector(inputs_embed)
            inputs_reshape = inputs_embed.unsqueeze(dim=1)
        memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)
        next_memory = self.attend_over_memory(memory_plus_input)
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]
        if self.gate_style == 'unit' or self.gate_style == 'memory':
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory
        output = next_memory.view(next_memory.shape[0], -1)
        output_embed = self.output_to_embed_decoder(output)
        output_embed = self.dropout(output_embed)
        if not self.use_adaptive_softmax:
            logit = self.embed_to_logit_decoder(output_embed)
        else:
            logit = output_embed
        return logit, next_memory

    def forward(self, inputs, memory, targets, require_logits=False):
        memory = self.repackage_hidden(memory)
        logits = []
        for idx_step in range(inputs.shape[1]):
            logit, memory = self.forward_step(inputs[:, (idx_step)], memory)
            logits.append(logit)
        logits = torch.cat(logits)
        if targets is not None:
            if not self.use_adaptive_softmax:
                loss = self.criterion(logits, targets)
            else:
                _, loss = self.criterion_adaptive(logits, targets)
        else:
            loss = None
        if not require_logits:
            return loss, memory
        else:
            return logits, loss, memory


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True, use_adaptive_softmax=False, cutoffs=None):
        super(RNNModel, self).__init__()
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if use_cudnn_version:
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        elif rnn_type in ['LSTM', 'GRU']:
            rnn_type = str(rnn_type) + 'Cell'
            rnn_modulelist = []
            for i in range(nlayers):
                rnn_modulelist.append(getattr(nn, rnn_type)(ninp, nhid))
                if i < nlayers - 1:
                    rnn_modulelist.append(nn.Dropout(dropout))
            self.rnn = nn.ModuleList(rnn_modulelist)
        else:
            raise ValueError('non-cudnn version of (RNNCell) is not implemented. use LSTM or GRU instead')
        if not use_adaptive_softmax:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, ntoken)
            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
        else:
            self.decoder_adaptive = nn.Linear(nhid, nhid)
            self.use_adaptive_softmax = use_adaptive_softmax
            self.cutoffs = cutoffs
            if tie_weights:
                None
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        if self.use_cudnn_version:
            output, hidden = self.rnn(emb, hidden)
        else:
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers + 1, 2):
                output = []
                hx, cx = hidden[0][int(idx_layer / 2)], hidden[1][int(idx_layer / 2)]
                for idx_step in range(input.shape[0]):
                    hx, cx = self.rnn[idx_layer](layer_input[idx_step], (hx, cx))
                    output.append(hx)
                output = torch.stack(output)
                if idx_layer + 1 < self.nlayers:
                    output = self.rnn[idx_layer + 1](output)
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)
        output = self.drop(output)
        if not self.use_adaptive_softmax:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        else:
            decoded = self.decoder_adaptive(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


num_vectors = 8


class RRNN(nn.Module):

    def __init__(self, mlp_size):
        super(RRNN, self).__init__()
        self.mlp_size = mlp_size
        self.memory_size_per_row = args.headsize * args.numheads * args.memslots
        self.relational_memory = RelationalMemory(mem_slots=args.memslots, head_size=args.headsize, input_size=args.input_size, num_heads=args.numheads, num_blocks=args.numblocks, forget_bias=args.forgetbias, input_bias=args.inputbias)
        self.mlp = nn.Sequential(nn.Linear(self.memory_size_per_row, self.mlp_size), nn.ReLU(), nn.Linear(self.mlp_size, self.mlp_size), nn.ReLU(), nn.Linear(self.mlp_size, self.mlp_size), nn.ReLU(), nn.Linear(self.mlp_size, self.mlp_size), nn.ReLU())
        self.out = nn.Linear(self.mlp_size, num_vectors)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, memory):
        logit, memory = self.relational_memory(input, memory)
        mlp = self.mlp(logit)
        out = self.out(mlp)
        out = self.softmax(out)
        return out, memory

