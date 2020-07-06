import sys
_module = sys.modules[__name__]
del sys
main = _module
modules = _module
dropout = _module
gate = _module
pair_encoder = _module
attentions = _module
cells = _module
pair_encoder = _module
pointer_network = _module
pointer_network = _module
rnn = _module
stacked_rnn = _module
utils = _module
qa = _module
squad = _module
dataset = _module
rnet = _module
tests = _module
models = _module

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


from torch import nn


import torch


from torch import Tensor


from typing import Dict


from typing import Optional


from typing import List


from typing import Any


from torch.nn.functional import nll_loss


class RNNDropout(nn.Module):

    def __init__(self, p, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, inputs):
        if not self.training:
            return inputs
        if self.batch_first:
            mask = inputs.new_ones(inputs.size(0), 1, inputs.size(2), requires_grad=False)
        else:
            mask = inputs.new_ones(1, inputs.size(1), inputs.size(2), requires_grad=False)
        return self.dropout(mask) * inputs


class Gate(nn.Module):

    def __init__(self, input_size, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(RNNDropout(dropout), nn.Linear(input_size, input_size, bias=False), nn.Sigmoid())

    def forward(self, inputs):
        return inputs * self.gate(inputs)


class StaticDotAttention(nn.Module):

    def __init__(self, memory_size, input_size, attention_size, batch_first=False, dropout=0.2):
        super().__init__()
        self.input_linear = nn.Sequential(RNNDropout(dropout, batch_first=True), nn.Linear(input_size, attention_size, bias=False), nn.ReLU())
        self.memory_linear = nn.Sequential(RNNDropout(dropout, batch_first=True), nn.Linear(memory_size, attention_size, bias=False), nn.ReLU())
        self.attention_size = attention_size
        self.batch_first = batch_first

    def forward(self, inputs: Tensor, memory: Tensor, memory_mask: Tensor):
        if not self.batch_first:
            inputs = inputs.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_mask = memory_mask.transpose(0, 1)
        input_ = self.input_linear(inputs)
        memory_ = self.memory_linear(memory)
        logits = torch.bmm(input_, memory_.transpose(2, 1)) / self.attention_size ** 0.5
        memory_mask = memory_mask.unsqueeze(1).expand(-1, inputs.size(1), -1)
        score = masked_softmax(logits, memory_mask, dim=-1)
        context = torch.bmm(score, memory)
        new_input = torch.cat([context, inputs], dim=-1)
        if not self.batch_first:
            return new_input.transpose(0, 1)
        return new_input


class _PairEncodeCell(nn.Module):

    def __init__(self, input_size, cell, attention_size, memory_size=None, use_state_in_attention=True, batch_first=False):
        super().__init__()
        if memory_size is None:
            memory_size = input_size
        self.cell = cell
        self.use_state = use_state_in_attention
        attention_input_size = input_size + memory_size
        if use_state_in_attention:
            attention_input_size += cell.hidden_size
        self.attention_w = nn.Sequential(nn.Dropout(), nn.Linear(attention_input_size, attention_size, bias=False), nn.Tanh(), nn.Linear(attention_size, 1, bias=False))
        self.batch_first = batch_first

    def forward(self, inputs: Tensor, memory: Tensor=None, memory_mask: Tensor=None, state: Tensor=None):
        """
        :param inputs:  B x H
        :param memory: T x B x H if not batch_first
        :param memory_mask: T x B if not batch_first
        :param state: B x H
        :return:
        """
        if self.batch_first:
            memory = memory.transpose(0, 1)
            memory_mask = memory_mask.transpose(0, 1)
        assert inputs.size(0) == memory.size(1) == memory_mask.size(1), 'inputs batch size does not match memory batch size'
        memory_time_length = memory.size(0)
        if state is None:
            state = inputs.new_zeros(inputs.size(0), self.cell.hidden_size, requires_grad=False)
        if self.use_state:
            hx = state
            if isinstance(state, tuple):
                hx = state[0]
            attention_input = torch.cat([inputs, hx], dim=-1)
            attention_input = attention_input.unsqueeze(0).expand(memory_time_length, -1, -1)
        else:
            attention_input = inputs.unsqueeze(0).expand(memory_time_length, -1, -1)
        attention_logits = self.attention_w(torch.cat([attention_input, memory], dim=-1)).squeeze(-1)
        attention_scores = masked_softmax(attention_logits, memory_mask, dim=0)
        attention_vector = torch.sum(attention_scores.unsqueeze(-1) * memory, dim=0)
        new_input = torch.cat([inputs, attention_vector], dim=-1)
        return self.cell(new_input, state)


class PairEncodeCell(_PairEncodeCell):

    def __init__(self, input_size, cell, attention_size, memory_size=None, batch_first=False):
        super().__init__(input_size, cell, attention_size, memory_size=memory_size, use_state_in_attention=True, batch_first=batch_first)


class SelfMatchCell(_PairEncodeCell):

    def __init__(self, input_size, cell, attention_size, memory_size=None, batch_first=False):
        super().__init__(input_size, cell, attention_size, memory_size=memory_size, use_state_in_attention=False, batch_first=batch_first)


def unroll_attention_cell(cell, inputs, memory, memory_mask, batch_first=False, initial_state=None, backward=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    output = []
    state = initial_state
    steps = range(inputs.size(0))
    if backward:
        steps = range(inputs.size(0) - 1, -1, -1)
    for t in steps:
        state = cell(inputs[t], memory=memory, memory_mask=memory_mask, state=state)
        output.append(state)
    if backward:
        output = output[::-1]
    output = torch.stack(output, dim=1 if batch_first else 0)
    return output, state


def bidirectional_unroll_attention_cell(cell_fw, cell_bw, inputs, memory, memory_mask, batch_first=False, initial_state=None):
    if initial_state is None:
        initial_state = [None, None]
    output_fw, state_fw = unroll_attention_cell(cell_fw, inputs, memory, memory_mask, batch_first=batch_first, initial_state=initial_state[0], backward=False)
    output_bw, state_bw = unroll_attention_cell(cell_bw, inputs, memory, memory_mask, batch_first=batch_first, initial_state=initial_state[1], backward=True)
    return torch.cat([output_fw, output_bw], dim=-1), (state_fw, state_bw)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Gate,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RNNDropout,
     lambda: ([], {'p': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_matthew_z_R_net(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

