import sys
_module = sys.modules[__name__]
del sys
generate_data = _module
preprocess = _module
rnn_preprocess = _module
fix_q18 = _module
symbolic_preprocess = _module
main = _module
model = _module
utils = _module
data = _module
dataloader = _module
dataset = _module
test = _module
train = _module

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


import random


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.autograd import Variable


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()
        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.reset_gate = nn.Sequential(nn.Linear(state_dim * 3, state_dim), nn.Sigmoid())
        self.update_gate = nn.Sequential(nn.Linear(state_dim * 3, state_dim), nn.Sigmoid())
        self.tansform = nn.Sequential(nn.Linear(state_dim * 3, state_dim), nn.Tanh())

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node * self.n_edge_types]
        A_out = A[:, :, self.n_node * self.n_edge_types:]
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, opt):
        super(GGNN, self).__init__()
        assert (opt.state_dim >= opt.annotation_dim, 'state_dim must be no less than annotation_dim')
        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module('in_{}'.format(i), in_fc)
            self.add_module('out_{}'.format(i), out_fc)
        self.in_fcs = AttrProxy(self, 'in_')
        self.out_fcs = AttrProxy(self, 'out_')
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)
        self.out = nn.Sequential(nn.Linear(self.state_dim + self.annotation_dim, self.state_dim), nn.Tanh(), nn.Linear(self.state_dim, 1))
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            prop_state = self.propogator(in_states, out_states, prop_state, A)
        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out(join_state)
        output = output.sum(2)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Propogator,
     lambda: ([], {'state_dim': 4, 'n_node': 4, 'n_edge_types': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 0, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_chingyaoc_ggnn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

