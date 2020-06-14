import sys
_module = sys.modules[__name__]
del sys
core = _module
accessor = _module
accessors = _module
dynamic_accessor = _module
static_accessor = _module
agent = _module
agents = _module
empty_agent = _module
sl_agent = _module
circuit = _module
circuits = _module
dnc_circuit = _module
ntm_circuit = _module
controller = _module
controllers = _module
lstm_controller = _module
env = _module
envs = _module
copy_env = _module
repeat_copy_env = _module
head = _module
heads = _module
dynamic_head = _module
dynamic_read_head = _module
dynamic_write_head = _module
static_head = _module
static_read_head = _module
static_write_head = _module
memory = _module
main = _module
utils = _module
factory = _module
fake_ops = _module
helpers = _module
init_weights = _module
options = _module
similarities = _module

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


import torch.nn as nn


import torch


from torch.autograd import Variable


import numpy as np


import random


import time


import torch.nn.functional as F


import torch.optim as optim


class Accessor(nn.Module):

    def __init__(self, args):
        super(Accessor, self).__init__()
        self.logger = args.logger
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value
        self.write_head_params = args.write_head_params
        self.read_head_params = args.read_head_params
        self.memory_params = args.memory_params
        self.write_head_params.num_heads = self.num_write_heads
        self.write_head_params.batch_size = self.batch_size
        self.write_head_params.hidden_dim = self.hidden_dim
        self.write_head_params.mem_hei = self.mem_hei
        self.write_head_params.mem_wid = self.mem_wid
        self.read_head_params.num_heads = self.num_read_heads
        self.read_head_params.batch_size = self.batch_size
        self.read_head_params.hidden_dim = self.hidden_dim
        self.read_head_params.mem_hei = self.mem_hei
        self.read_head_params.mem_wid = self.mem_wid
        self.memory_params.batch_size = self.batch_size
        self.memory_params.clip_value = self.clip_value
        self.memory_params.mem_hei = self.mem_hei
        self.memory_params.mem_wid = self.mem_wid

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def _reset_states(self):
        raise NotImplementedError('not implemented in base calss')

    def _reset(self):
        raise NotImplementedError('not implemented in base calss')

    def visual(self):
        self.write_heads.visual()
        self.read_heads.visual()
        self.memory.visual()

    def forward(self, lstm_hidden_vb):
        raise NotImplementedError('not implemented in base calss')


class Circuit(nn.Module):

    def __init__(self, args):
        super(Circuit, self).__init__()
        self.logger = args.logger
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value
        self.controller_params = args.controller_params
        self.accessor_params = args.accessor_params
        self.read_vec_dim = self.num_read_heads * self.mem_wid
        self.controller_params.batch_size = self.batch_size
        self.controller_params.input_dim = self.input_dim
        self.controller_params.read_vec_dim = self.read_vec_dim
        self.controller_params.output_dim = self.output_dim
        self.controller_params.hidden_dim = self.hidden_dim
        self.controller_params.mem_hei = self.mem_hei
        self.controller_params.mem_wid = self.mem_wid
        self.controller_params.clip_value = self.clip_value
        self.accessor_params.batch_size = self.batch_size
        self.accessor_params.hidden_dim = self.hidden_dim
        self.accessor_params.num_write_heads = self.num_write_heads
        self.accessor_params.num_read_heads = self.num_read_heads
        self.accessor_params.mem_hei = self.mem_hei
        self.accessor_params.mem_wid = self.mem_wid
        self.accessor_params.clip_value = self.clip_value
        self.logger.warning(
            '<-----------------------------======> Circuit:    {Controller, Accessor}'
            )

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def print_model(self):
        self.logger.warning(
            '<-----------------------------======> Circuit:    {Overall Architecture}'
            )
        self.logger.warning(self)

    def _reset_states(self):
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        self.controller._reset_states()
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim
            ).fill_(1e-06)
        self._reset_states()

    def forward(self, input_vb):
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.
            hidden_dim), self.read_vec_vb.view(-1, self.read_vec_dim)), 1))
        return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=
            self.clip_value)).view(1, self.batch_size, self.output_dim)


class Controller(nn.Module):

    def __init__(self, args):
        super(Controller, self).__init__()
        self.logger = args.logger
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.read_vec_dim = args.read_vec_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def print_model(self):
        self.logger.warning('<--------------------------------===> Controller:'
            )
        self.logger.warning(self)

    def _reset_states(self):
        self.lstm_hidden_vb = Variable(self.lstm_hidden_ts[0]).type(self.dtype
            ), Variable(self.lstm_hidden_ts[1]).type(self.dtype)

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        self.lstm_hidden_ts = []
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.
            hidden_dim))
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.
            hidden_dim))
        self._reset_states()

    def forward(self, input_vb):
        raise NotImplementedError('not implemented in base calss')


class Head(nn.Module):

    def __init__(self, args):
        super(Head, self).__init__()
        self.logger = args.logger
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.refs = args.refs
            self.win_head = None
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.num_allowed_shifts = args.num_allowed_shifts

    def _reset_states(self):
        self.wl_prev_vb = Variable(self.wl_prev_ts).type(self.dtype)

    def _reset(self):
        self.type(self.dtype)
        self.wl_prev_ts = torch.eye(1, self.mem_hei).unsqueeze(0).expand(self
            .batch_size, self.num_heads, self.mem_hei)
        self._reset_states()

    def _visual(self):
        raise NotImplementedError('not implemented in base calss')

    def _access(self, memory_vb):
        raise NotImplementedError('not implemented in base calss')


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jingweiz_pytorch_dnc(_paritybench_base):
    pass
