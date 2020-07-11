import sys
_module = sys.modules[__name__]
del sys
dataset = _module
model = _module
nn = _module
optim = _module
train = _module
trainer = _module
plugins = _module
utils = _module

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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader as DataLoaderBase


from torch.nn import functional as F


from torch.nn import init


import numpy as np


from torch import nn


import math


from torch.nn.functional import hardtanh


from functools import reduce


import re


from torch.autograd import Variable


import matplotlib


from matplotlib import pyplot


import time


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm):
        super().__init__()
        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))
        self.input_expand = torch.nn.Conv1d(in_channels=n_frame_samples, out_channels=dim, kernel_size=1)
        init.kaiming_uniform(self.input_expand.weight)
        init.constant(self.input_expand.bias, 0)
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)
        self.rnn = torch.nn.GRU(input_size=dim, hidden_size=dim, num_layers=n_rnn, batch_first=True)
        for i in range(n_rnn):
            nn.concat_init(getattr(self.rnn, 'weight_ih_l{}'.format(i)), [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform])
            init.constant(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)
            nn.concat_init(getattr(self.rnn, 'weight_hh_l{}'.format(i)), [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal])
            init.constant(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)
        self.upsampling = nn.LearnedUpsampling1d(in_channels=dim, out_channels=dim, kernel_size=frame_size)
        init.uniform(self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim))
        init.constant(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(self.upsampling.conv_t)

    def forward(self, prev_samples, upper_tier_conditioning, hidden):
        batch_size, _, _ = prev_samples.size()
        input = self.input_expand(prev_samples.permute(0, 2, 1)).permute(0, 2, 1)
        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning
        reset = hidden is None
        if hidden is None:
            n_rnn, _ = self.h0.size()
            hidden = self.h0.unsqueeze(1).expand(n_rnn, batch_size, self.dim).contiguous()
        output, hidden = self.rnn(input, hidden)
        output = self.upsampling(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output, hidden


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()
        self.q_levels = q_levels
        self.embedding = torch.nn.Embedding(self.q_levels, self.q_levels)
        self.input = torch.nn.Conv1d(in_channels=q_levels, out_channels=dim, kernel_size=frame_size, bias=False)
        init.kaiming_uniform(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)
        self.hidden = torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        init.kaiming_uniform(self.hidden.weight)
        init.constant(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)
        self.output = torch.nn.Conv1d(in_channels=dim, out_channels=q_levels, kernel_size=1)
        nn.lecun_uniform(self.output.weight)
        init.constant(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        batch_size, _, _ = upper_tier_conditioning.size()
        prev_samples = self.embedding(prev_samples.contiguous().view(-1)).view(batch_size, -1, self.q_levels)
        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)
        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()
        return F.log_softmax(x.view(-1, self.q_levels)).view(batch_size, -1, self.q_levels)


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels, weight_norm):
        super().__init__()
        self.dim = dim
        self.q_levels = q_levels
        ns_frame_samples = map(int, np.cumprod(frame_sizes))
        self.frame_level_rnns = torch.nn.ModuleList([FrameLevelRNN(frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm) for frame_size, n_frame_samples in zip(frame_sizes, ns_frame_samples)])
        self.sample_level_mlp = SampleLevelMLP(frame_sizes[0], dim, q_levels, weight_norm)

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        output, new_hidden = rnn(prev_samples, upper_tier_conditioning, self.hidden_states[rnn])
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()
        batch_size, _ = input_sequences.size()
        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(input_sequences[:, from_index:to_index], self.model.q_levels)
            prev_samples = prev_samples.contiguous().view(batch_size, -1, rnn.n_frame_samples)
            upper_tier_conditioning = self.run_rnn(rnn, prev_samples, upper_tier_conditioning)
        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences[:, self.model.lookback - bottom_frame_size:]
        return self.model.sample_level_mlp(mlp_input_sequences, upper_tier_conditioning)


class LearnedUpsampling1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=kernel_size, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels, kernel_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv_t.reset_parameters()
        nn.init.constant(self.bias, 0)

    def forward(self, input):
        batch_size, _, length = input.size()
        kernel_size, = self.conv_t.kernel_size
        bias = self.bias.unsqueeze(0).unsqueeze(2).expand(batch_size, self.conv_t.out_channels, length, kernel_size).contiguous().view(batch_size, self.conv_t.out_channels, length * kernel_size)
        return self.conv_t(input) + bias


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LearnedUpsampling1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_deepsound_project_samplernn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

