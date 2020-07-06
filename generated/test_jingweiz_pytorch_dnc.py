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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torch


from torch.autograd import Variable


import torch.optim as optim


import random


import numpy as np


import time


import torch.nn.functional as F


from random import randint


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


class ExternalMemory(object):

    def __init__(self, args):
        self.logger = args.logger
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.refs = args.refs
            self.win_memory = 'win_memory'
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.batch_size = args.batch_size
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.logger.warning('<-----------------------------------> Memory:     {' + str(self.batch_size) + '(batch_size) x ' + str(self.mem_hei) + '(mem_hei) x ' + str(self.mem_wid) + '(mem_wid)}')

    def _save_memory(self):
        raise NotImplementedError('not implemented in base calss')

    def _load_memory(self):
        raise NotImplementedError('not implemented in base calss')

    def _reset_states(self):
        self.memory_vb = Variable(self.memory_ts).type(self.dtype)

    def _reset(self):
        self.memory_ts = torch.zeros(self.batch_size, self.mem_hei, self.mem_wid).fill_(1e-06)
        self._reset_states()

    def visual(self):
        if self.visualize:
            self.win_memory = self.vis.heatmap(self.memory_vb.data[0].clone().cpu().numpy(), env=self.refs, win=self.win_memory, opts=dict(title='memory'))


class DynamicAccessor(Accessor):

    def __init__(self, args):
        super(DynamicAccessor, self).__init__(args)
        self.logger = args.logger
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.read_head_params.num_read_modes = self.write_head_params.num_heads * 2 + 1
        self.logger.warning('<--------------------------------===> Accessor:   {WriteHead, ReadHead, Memory}')
        self.usage_vb = None
        self.link_vb = None
        self.preced_vb = None
        self.write_heads = WriteHead(self.write_head_params)
        self.read_heads = ReadHead(self.read_head_params)
        self.memory = ExternalMemory(self.memory_params)
        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        self.usage_vb = Variable(self.usage_ts).type(self.dtype)
        self.link_vb = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        self.memory._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.usage_ts = torch.zeros(self.batch_size, self.mem_hei)
        self.link_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)
        self._reset_states()

    def forward(self, hidden_vb):
        self.usage_vb = self.write_heads._update_usage(self.usage_vb)
        self.usage_vb = self.read_heads._update_usage(hidden_vb, self.usage_vb)
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb, self.usage_vb)
        self.link_vb, self.preced_vb = self.write_heads._temporal_link(self.link_vb, self.preced_vb)
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb, self.link_vb, self.write_head_params.num_heads)
        return read_vec_vb


class StaticAccessor(Accessor):

    def __init__(self, args):
        super(StaticAccessor, self).__init__(args)
        self.logger = args.logger
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        self.logger.warning('<--------------------------------===> Accessor:   {WriteHead, ReadHead, Memory}')
        self.write_heads = WriteHead(self.write_head_params)
        self.read_heads = ReadHead(self.read_head_params)
        self.memory = ExternalMemory(self.memory_params)
        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        self.memory._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self._reset_states()

    def forward(self, hidden_vb):
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb)
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb)
        return read_vec_vb


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
        self.logger.warning('<-----------------------------======> Circuit:    {Controller, Accessor}')

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def print_model(self):
        self.logger.warning('<-----------------------------======> Circuit:    {Overall Architecture}')
        self.logger.warning(self)

    def _reset_states(self):
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        self.controller._reset_states()
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim).fill_(1e-06)
        self._reset_states()

    def forward(self, input_vb):
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.hidden_dim), self.read_vec_vb.view(-1, self.read_vec_dim)), 1))
        return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(1, self.batch_size, self.output_dim)


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
        self.logger.warning('<--------------------------------===> Controller:')
        self.logger.warning(self)

    def _reset_states(self):
        self.lstm_hidden_vb = Variable(self.lstm_hidden_ts[0]).type(self.dtype), Variable(self.lstm_hidden_ts[1]).type(self.dtype)

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        self.lstm_hidden_ts = []
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.hidden_dim))
        self.lstm_hidden_ts.append(torch.zeros(self.batch_size, self.hidden_dim))
        self._reset_states()

    def forward(self, input_vb):
        raise NotImplementedError('not implemented in base calss')


class DNCCircuit(Circuit):

    def __init__(self, args):
        super(DNCCircuit, self).__init__(args)
        self.controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)
        self._reset()

    def _init_weights(self):
        pass


class NTMCircuit(Circuit):

    def __init__(self, args):
        super(NTMCircuit, self).__init__(args)
        self.controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)
        self._reset()

    def _init_weights(self):
        pass


class LSTMController(Controller):

    def __init__(self, args):
        super(LSTMController, self).__init__(args)
        self.in_2_hid = nn.LSTMCell(self.input_dim + self.read_vec_dim, self.hidden_dim, 1)
        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input_vb, read_vec_vb):
        self.lstm_hidden_vb = self.in_2_hid(torch.cat((input_vb.contiguous().view(-1, self.input_dim), read_vec_vb.contiguous().view(-1, self.read_vec_dim)), 1), self.lstm_hidden_vb)
        self.lstm_hidden_vb = [self.lstm_hidden_vb[0].clamp(min=-self.clip_value, max=self.clip_value), self.lstm_hidden_vb[1]]
        return self.lstm_hidden_vb[0]


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
        self.wl_prev_ts = torch.eye(1, self.mem_hei).unsqueeze(0).expand(self.batch_size, self.num_heads, self.mem_hei)
        self._reset_states()

    def _visual(self):
        raise NotImplementedError('not implemented in base calss')

    def _access(self, memory_vb):
        raise NotImplementedError('not implemented in base calss')


def batch_cosine_sim(u, v, epsilon=1e-06):
    """
    u: content_key: [batch_size x num_heads x mem_wid]
    v: memory:      [batch_size x mem_hei   x mem_wid]
    k: similarity:  [batch_size x num_heads x mem_hei]
    """
    assert u.dim() == 3 and v.dim() == 3
    numerator = torch.bmm(u, v.transpose(1, 2))
    denominator = torch.sqrt(torch.bmm(u.norm(2, 2, keepdim=True).pow(2) + epsilon, v.norm(2, 2, keepdim=True).pow(2).transpose(1, 2) + epsilon))
    k = numerator / (denominator + epsilon)
    return k


class DynamicHead(Head):

    def __init__(self, args):
        super(DynamicHead, self).__init__(args)
        self.hid_2_key = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _update_usage(self, prev_usage_vb):
        raise NotImplementedError('not implemented in base calss')

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \\in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb)
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2)).transpose(0, 2)

    def _location_focus(self):
        raise NotImplementedError('not implemented in base calss')

    def forward(self, hidden_vb, memory_vb):
        self.key_vb = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.beta_vb = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)
        self._content_focus(memory_vb)


class DynamicReadHead(DynamicHead):

    def __init__(self, args):
        super(DynamicReadHead, self).__init__(args)
        if self.visualize:
            self.win_head = 'win_read_head'
        self.num_read_modes = args.num_read_modes
        self.hid_2_free_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_read_mode = nn.Linear(self.hidden_dim, self.num_heads * self.num_read_modes)
        self.logger.warning('<-----------------------------------> ReadHeads:  {' + str(self.num_heads) + ' heads}')
        self.logger.warning(self)
        self._reset()

    def visual(self):
        if self.visualize:
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title='read_head'))

    def _update_usage(self, hidden_vb, prev_usage_vb):
        """
        calculates the new usage after reading and freeing from memory
        variables needed:
            hidden_vb:     [batch_size x hidden_dim]
            prev_usage_vb: [batch_size x mem_hei]
            free_gate_vb:  [batch_size x num_heads x 1]
            wl_prev_vb:    [batch_size x num_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        self.free_gate_vb = F.sigmoid(self.hid_2_free_gate(hidden_vb)).view(-1, self.num_heads, 1)
        free_read_weights_vb = self.free_gate_vb.expand_as(self.wl_prev_vb) * self.wl_prev_vb
        psi_vb = torch.prod(1.0 - free_read_weights_vb, 1)
        return prev_usage_vb * psi_vb

    def _directional_read_weights(self, link_vb, num_write_heads, forward):
        """
        calculates the forward or the backward read weights
        for each read head (at a given address), there are `num_writes` link
        graphs to follow. thus this function computes a read address for each of
        the `num_reads * num_writes` pairs of read and write heads.
        we calculate the forward and backward directions for each pair of read
        and write heads; hence we need to tile the read weights and do a sort of
        "outer product" to get this.
        variables needed:
            link_vb:    [batch_size x num_read_heads x mem_hei x mem_hei]
                     -> {L_t}, current link graph
            wl_prev_vb: [batch_size x num_read_heads x mem_hei]
                     -> containing the previous read weights w_{t-1}^r.
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            forward:    boolean
                     -> indicating whether to follow the "future" (True)
                     -> direction in the link graph or the "past" (False)
                     -> direction
        returns:
            directional_weights_vb: [batch_size x num_read_heads x num_write_heads x mem_hei]
        """
        expanded_read_weights_vb = self.wl_prev_vb.unsqueeze(1).expand_as(torch.Tensor(self.batch_size, num_write_heads, self.num_heads, self.mem_hei)).contiguous()
        if forward:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei).transpose(1, 2))
        else:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei))
        return directional_weights_vb.view(-1, num_write_heads, self.num_heads, self.mem_hei).transpose(1, 2)

    def _location_focus(self, link_vb, num_write_heads):
        """
        calculates the read weights after location focus
        variables needed:
            link_vb:      [batch_size x num_heads x mem_hei x mem_hei]
                       -> {L_t}, current link graph
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            wc_vb:        [batch_size x num_heads x mem_hei]
                       -> containing the focus by content of {t}
            read_mode_vb: [batch_size x num_heads x num_read_modes]
        returns:
            wl_curr_vb:   [batch_size x num_read_heads x num_write_heads x mem_hei]
                       -> focus by content of {t}
        """
        forward_weights_vb = self._directional_read_weights(link_vb, num_write_heads, True)
        backward_weights_vb = self._directional_read_weights(link_vb, num_write_heads, False)
        backward_mode_vb = self.read_mode_vb[:, :, :num_write_heads]
        forward_mode_vb = self.read_mode_vb[:, :, num_write_heads:2 * num_write_heads]
        content_mode_vb = self.read_mode_vb[:, :, 2 * num_write_heads:]
        self.wl_curr_vb = content_mode_vb.expand_as(self.wc_vb) * self.wc_vb + torch.sum(forward_mode_vb.unsqueeze(3).expand_as(forward_weights_vb) * forward_weights_vb, 2) + torch.sum(backward_mode_vb.unsqueeze(3).expand_as(backward_weights_vb) * backward_weights_vb, 2)

    def _access(self, memory_vb):
        """
        variables needed:
            wl_curr_vb:   [batch_size x num_heads x mem_hei]
                       -> location focus of {t}
            memory_vb:    [batch_size x mem_hei   x mem_wid]
        returns:
            read_vec_vb:  [batch_size x num_heads x mem_wid]
                       -> read vector of {t}
        """
        return torch.bmm(self.wl_curr_vb, memory_vb)

    def forward(self, hidden_vb, memory_vb, link_vb, num_write_heads):
        super(DynamicReadHead, self).forward(hidden_vb, memory_vb)
        self.read_mode_vb = F.softmax(self.hid_2_read_mode(hidden_vb).view(-1, self.num_heads, self.num_read_modes).transpose(0, 2)).transpose(0, 2)
        self._location_focus(link_vb, num_write_heads)
        self.wl_prev_vb = self.wl_curr_vb
        return self._access(memory_vb)


def fake_cumprod(vb):
    """
    args:
        vb:  [hei x wid]
          -> NOTE: we are lazy here so now it only supports cumprod along wid
    """
    vb = vb.unsqueeze(0)
    mul_mask_vb = Variable(torch.zeros(vb.size(2), vb.size(1), vb.size(2))).type_as(vb)
    for i in range(vb.size(2)):
        mul_mask_vb[(i), :, :i + 1] = 1
    add_mask_vb = 1 - mul_mask_vb
    vb = vb.expand_as(mul_mask_vb) * mul_mask_vb + add_mask_vb
    vb = torch.prod(vb, 2, keepdim=True).transpose(0, 2)
    return vb


class DynamicWriteHead(DynamicHead):

    def __init__(self, args):
        super(DynamicWriteHead, self).__init__(args)
        if self.visualize:
            self.win_head = 'win_write_head'
        self.hid_2_alloc_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_write_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.logger.warning('<-----------------------------------> WriteHeads: {' + str(self.num_heads) + ' heads}')
        self.logger.warning(self)
        self._reset()

    def visual(self):
        if self.visualize:
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title='write_head'))

    def _update_usage(self, prev_usage_vb):
        """
        calculates the new usage after writing to memory
        variables needed:
            prev_usage_vb: [batch_size x mem_hei]
            wl_prev_vb:    [batch_size x num_write_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        write_weights_vb = 1.0 - torch.prod(1.0 - self.wl_prev_vb, 1)
        return prev_usage_vb + (1.0 - prev_usage_vb) * write_weights_vb

    def _allocation(self, usage_vb, epsilon=1e-06):
        """
        computes allocation by sorting usage, a = a_t[\\phi_t[j]]
        variables needed:
            usage_vb: [batch_size x mem_hei]
                   -> indicating current memory usage, this is equal to u_t in
                      the paper when we only have one write head, but for
                      multiple write heads, one should update the usage while
                      iterating through the write heads to take into account the
                      allocation returned by this function
        returns:
            alloc_vb: [batch_size x num_write_heads x mem_hei]
        """
        usage_vb = epsilon + (1 - epsilon) * usage_vb
        sorted_usage_vb, indices_vb = torch.topk(usage_vb, k=self.mem_hei, dim=1, largest=False)
        cat_sorted_usage_vb = torch.cat((Variable(torch.ones(self.batch_size, 1)).type(self.dtype), sorted_usage_vb), 1)[:, :-1]
        prod_sorted_usage_vb = fake_cumprod(cat_sorted_usage_vb)
        alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb.squeeze()
        _, indices_vb = torch.topk(indices_vb, k=self.mem_hei, dim=1, largest=False)
        alloc_weight_vb = alloc_weight_vb.gather(1, indices_vb)
        return alloc_weight_vb

    def _location_focus(self, usage_vb):
        """
        Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for
        each write head. (For more than one write head, we use a "simulated new
        usage" which takes into account the fact that the previous write head
        will increase the usage in that area of the memory.)
        variables needed:
            usage_vb:         [batch_size x mem_hei]
                           -> representing current memory usage
            write_gate_vb:    [batch_size x num_write_heads x 1]
                           -> /in [0, 1] indicating how much each write head
                              does writing based on the address returned here
                              (and hence how much usage increases)
        returns:
            alloc_weights_vb: [batch_size x num_write_heads x mem_hei]
                            -> containing the freeness-based write locations
                               Note that this isn't scaled by `write_gate`;
                               this scaling must be applied externally.
        """
        alloc_weights_vb = []
        for i in range(self.num_heads):
            alloc_weights_vb.append(self._allocation(usage_vb))
            usage_vb += (1 - usage_vb) * self.write_gate_vb[:, (i), :].expand_as(usage_vb) * alloc_weights_vb[i]
        alloc_weight_vb = torch.stack(alloc_weights_vb, dim=1)
        self.wl_curr_vb = self.write_gate_vb.expand_as(alloc_weight_vb) * (self.alloc_gate_vb.expand_as(self.wc_vb) * alloc_weight_vb + (1.0 - self.alloc_gate_vb.expand_as(self.wc_vb)) * self.wc_vb)

    def _access(self, memory_vb):
        """
        variables needed:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
            erase_vb:   [batch_size x num_heads x mem_wid]
                     -> /in (0, 1)
            add_vb:     [batch_size x num_heads x mem_wid]
                     -> w/ no restrictions in range
            memory_vb:  [batch_size x mem_hei x mem_wid]
        returns:
            memory_vb:  [batch_size x mem_hei x mem_wid]
        NOTE: IMPORTANT: https://github.com/deepmind/dnc/issues/10
        """
        weighted_erase_vb = torch.bmm(self.wl_curr_vb.contiguous().view(-1, self.mem_hei, 1), self.erase_vb.contiguous().view(-1, 1, self.mem_wid)).view(-1, self.num_heads, self.mem_hei, self.mem_wid)
        keep_vb = torch.prod(1.0 - weighted_erase_vb, dim=1)
        memory_vb = memory_vb * keep_vb
        return memory_vb + torch.bmm(self.wl_curr_vb.transpose(1, 2), self.add_vb)

    def forward(self, hidden_vb, memory_vb, usage_vb):
        super(DynamicWriteHead, self).forward(hidden_vb, memory_vb)
        self.alloc_gate_vb = F.sigmoid(self.hid_2_alloc_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self.write_gate_vb = F.sigmoid(self.hid_2_write_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self._location_focus(usage_vb)
        self.wl_prev_vb = self.wl_curr_vb
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb = F.tanh(self.hid_2_add(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _update_link(self, prev_link_vb, prev_preced_vb):
        """
        calculates the new link graphs
        For each write head, the link is a directed graph (represented by a
        matrix with entries in range [0, 1]) whose vertices are the memory
        locations, and an edge indicates temporal ordering of writes.
        variables needed:
            prev_link_vb:   [batch_size x num_heads x mem_hei x mem_wid]
                         -> {L_t-1}, previous link graphs
            prev_preced_vb: [batch_size x num_heads x mem_hei]
                         -> {p_t}, the previous aggregated precedence
                         -> weights for each write head
            wl_curr_vb:     [batch_size x num_heads x mem_hei]
                         -> location focus of {t}
        returns:
            link_vb:        [batch_size x num_heads x mem_hei x mem_hei]
                         -> {L_t}, current link graph
        """
        write_weights_i_vb = self.wl_curr_vb.unsqueeze(3).expand_as(prev_link_vb)
        write_weights_j_vb = self.wl_curr_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_preced_j_vb = prev_preced_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_link_scale_vb = 1 - write_weights_i_vb - write_weights_j_vb
        new_link_vb = write_weights_i_vb * prev_preced_j_vb
        link_vb = prev_link_scale_vb * prev_link_vb + new_link_vb
        diag_mask_vb = Variable(1 - torch.eye(self.mem_hei).unsqueeze(0).unsqueeze(0).expand_as(link_vb)).type(self.dtype)
        link_vb = link_vb * diag_mask_vb
        return link_vb

    def _update_precedence_weights(self, prev_preced_vb):
        """
        calculates the new precedence weights given the current write weights
        the precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the
        precedence weights unchanged, but with sum close to one will replace the
        precedence weights.
        variables needed:
            prev_preced_vb: [batch_size x num_write_heads x mem_hei]
            wl_curr_vb:     [batch_size x num_write_heads x mem_hei]
        returns:
            preced_vb:      [batch_size x num_write_heads x mem_hei]
        """
        write_sum_vb = torch.sum(self.wl_curr_vb, 2, keepdim=True)
        return (1 - write_sum_vb).expand_as(prev_preced_vb) * prev_preced_vb + self.wl_curr_vb

    def _temporal_link(self, prev_link_vb, prev_preced_vb):
        link_vb = self._update_link(prev_link_vb, prev_preced_vb)
        preced_vb = self._update_precedence_weights(prev_preced_vb)
        return link_vb, preced_vb


class StaticHead(Head):

    def __init__(self, args):
        super(StaticHead, self).__init__(args)
        self.hid_2_key = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_shift = nn.Linear(self.hidden_dim, self.num_heads * self.num_allowed_shifts)
        self.hid_2_gamma = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \\in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb)
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2)).transpose(0, 2)

    def _shift(self, wg_vb, shift_vb):
        """
        variables needed:
            wg_vb:    [batch_size x num_heads x mem_hei]
            shift_vb: [batch_size x num_heads x num_allowed_shifts]
                   -> sum=1; the shift weight vector
        returns:
            ws_vb:    [batch_size x num_heads x mem_hei]
                   -> the attention weight by location focus
        """
        batch_size = wg_vb.size(0)
        input_dim = wg_vb.size(2)
        assert input_dim == self.mem_hei
        filter_dim = shift_vb.size(2)
        assert filter_dim == self.num_allowed_shifts
        ws_vb = None
        for i in range(batch_size):
            for j in range(self.num_heads):
                ws_tmp_vb = F.conv1d(wg_vb[i][j].unsqueeze(0).unsqueeze(0).repeat(1, 1, 3), shift_vb[i][j].unsqueeze(0).unsqueeze(0).contiguous(), padding=filter_dim // 2)[:, :, input_dim:2 * input_dim]
                if ws_vb is None:
                    ws_vb = ws_tmp_vb
                else:
                    ws_vb = torch.cat((ws_vb, ws_tmp_vb), 0)
        ws_vb = ws_vb.view(-1, self.num_heads, self.mem_hei)
        return ws_vb

    def _location_focus(self):
        """
        variables needed:
            wl_prev_vb: [batch_size x num_heads x mem_hei]
            wc_vb:      [batch_size x num_heads x mem_hei]
            gate_vb:    [batch_size x num_heads x 1]
                     -> \\in (0, 1); the interpolation gate
            shift_vb:   [batch_size x num_heads x num_allowed_shifts]
                     -> sum=1; the shift weight vector
            gamma_vb:   [batch_size x num_heads x 1]
                     -> >=1; the sharpening vector
        returns:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
                     -> the attention weight by location focus
        """
        self.gate_vb = self.gate_vb.expand_as(self.wc_vb)
        wg_vb = self.wc_vb * self.gate_vb + self.wl_prev_vb * (1.0 - self.gate_vb)
        ws_vb = self._shift(wg_vb, self.shift_vb)
        wp_vb = ws_vb.pow(self.gamma_vb.expand_as(ws_vb))
        self.wl_curr_vb = wp_vb / wp_vb.sum(2, keepdim=True).expand_as(wp_vb)

    def forward(self, hidden_vb, memory_vb):
        self.key_vb = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.beta_vb = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)
        self.gate_vb = F.sigmoid(self.hid_2_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self.shift_vb = F.softmax(self.hid_2_shift(hidden_vb).view(-1, self.num_heads, self.num_allowed_shifts).transpose(0, 2)).transpose(0, 2)
        self.gamma_vb = (1.0 + F.softplus(self.hid_2_gamma(hidden_vb))).view(-1, self.num_heads, 1)
        self._content_focus(memory_vb)
        self._location_focus()


class StaticReadHead(StaticHead):

    def __init__(self, args):
        super(StaticReadHead, self).__init__(args)
        if self.visualize:
            self.win_head = 'win_read_head'
        self.logger.warning('<-----------------------------------> ReadHeads:  {' + str(self.num_heads) + ' heads}')
        self.logger.warning(self)
        self._reset()

    def visual(self):
        if self.visualize:
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title='read_head'))

    def forward(self, hidden_vb, memory_vb):
        super(StaticReadHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        return self._access(memory_vb)

    def _access(self, memory_vb):
        """
        variables needed:
            wl_curr_vb:   [batch_size x num_heads x mem_hei]
                       -> location focus of {t}
            memory_vb:    [batch_size x mem_hei   x mem_wid]
        returns:
            read_vec_vb:  [batch_size x num_heads x mem_wid]
                       -> read vector of {t}
        """
        return torch.bmm(self.wl_curr_vb, memory_vb)


class StaticWriteHead(StaticHead):

    def __init__(self, args):
        super(StaticWriteHead, self).__init__(args)
        if self.visualize:
            self.win_head = 'win_write_head'
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.logger.warning('<-----------------------------------> WriteHeads: {' + str(self.num_heads) + ' heads}')
        self.logger.warning(self)
        self._reset()

    def visual(self):
        if self.visualize:
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title='write_head'))

    def forward(self, hidden_vb, memory_vb):
        super(StaticWriteHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb = F.tanh(self.hid_2_add(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _access(self, memory_vb):
        """
        variables needed:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
            erase_vb:   [batch_size x num_heads x mem_wid]
                     -> /in (0, 1)
            add_vb:     [batch_size x num_heads x mem_wid]
                     -> w/ no restrictions in range
            memory_vb:  [batch_size x mem_hei x mem_wid]
        returns:
            memory_vb:  [batch_size x mem_hei x mem_wid]
        NOTE: IMPORTANT: https://github.com/deepmind/dnc/issues/10
        """
        weighted_erase_vb = torch.bmm(self.wl_curr_vb.contiguous().view(-1, self.mem_hei, 1), self.erase_vb.contiguous().view(-1, 1, self.mem_wid)).view(-1, self.num_heads, self.mem_hei, self.mem_wid)
        keep_vb = torch.prod(1.0 - weighted_erase_vb, dim=1)
        memory_vb = memory_vb * keep_vb
        return memory_vb + torch.bmm(self.wl_curr_vb.transpose(1, 2), self.add_vb)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DynamicHead,
     lambda: ([], {'args': _mock_config(logger=4, visualize=4, vis=4, refs=4, use_cuda=False, dtype=torch.float32, num_heads=4, batch_size=4, hidden_dim=4, mem_hei=4, mem_wid=4, num_allowed_shifts=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([64, 4, 4])], {}),
     False),
]

class Test_jingweiz_pytorch_dnc(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

