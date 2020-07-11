import sys
_module = sys.modules[__name__]
del sys
adept = _module
actor = _module
ac_eval = _module
ac_rollout = _module
base = _module
ac_helper = _module
actor_module = _module
impala = _module
agent = _module
actor_critic = _module
agent_module = _module
app = _module
container = _module
actorlearner = _module
learner_container = _module
rollout_queuer = _module
rollout_worker = _module
container = _module
nccl_optimizer = _module
distrib = _module
evaluation = _module
evaluation_thread = _module
init = _module
local = _module
render = _module
env = _module
_gym_wrappers = _module
_spaces = _module
_env = _module
env_module = _module
deepmind_sc2 = _module
openai_gym = _module
exp = _module
exp_module = _module
spec_builder = _module
replay = _module
rollout = _module
globals = _module
learner = _module
ac_rollout = _module
dm_return_scale = _module
learner_module = _module
impala = _module
manager = _module
manager_module = _module
simple_env_manager = _module
subproc_env_manager = _module
modules = _module
attention = _module
memory = _module
mlp = _module
norm = _module
sequence = _module
spatial = _module
network = _module
base = _module
network_module = _module
submodule = _module
modular_network = _module
net1d = _module
identity_1d = _module
linear = _module
lstm = _module
submodule_1d = _module
net2d = _module
identity_2d = _module
submodule_2d = _module
net3d = _module
_resnets = _module
four_conv = _module
identity_3d = _module
networks = _module
rmc = _module
submodule_3d = _module
net4d = _module
identity_4d = _module
submodule_4d = _module
preprocess = _module
observation = _module
ops = _module
registry = _module
rewardnorm = _module
rewnorm_module = _module
normalizers = _module
scripts = _module
_distrib = _module
distrib = _module
evaluate = _module
replay_gen_sc2 = _module
utils = _module
logging = _module
requires_args = _module
script_helpers = _module
util = _module
connect = _module
custom_agent_stub = _module
custom_environment_stub = _module
custom_network_stub = _module
custom_submodule_stub = _module
setup = _module
tests = _module
allreduce = _module
container_sync = _module
control_flow_zmq = _module
exp_sync_broadcast = _module
hello_ray = _module
launch = _module
multi_group = _module
nccl_typecheck = _module
ray_container = _module
test_rollout = _module
nstep = _module
test_modular_network = _module
test_registry = _module
test_requires_args = _module
test_util = _module

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


from collections import OrderedDict


import torch


import abc


from torch.nn import functional as F


from collections import defaultdict


from functools import reduce


from time import time


import numpy as np


import torch.distributed as dist


from torch.utils.tensorboard import SummaryWriter


import queue


from collections import namedtuple


import time


from torch.optim.lr_scheduler import LambdaLR


from torch import distributed as dist


from torch import multiprocessing as mp


import math


from torch import nn


from torch.nn import Linear


from torch.nn import Softmax


from torch.nn import Parameter


from torch.nn import Module


from torch.nn import LayerNorm


from torch import nn as nn


from torch.nn import LSTMCell


from torch.nn import Conv2d


from torch.nn import BatchNorm2d


from torch.nn import GroupNorm


from torch.nn import init


from torch.nn import BatchNorm1d


from collections import deque


from itertools import chain


import random


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head Self Attention.

    Adapted from:
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_py.py
    Reference implementation (Tensorflow):
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L2674
    """

    def __init__(self, nb_embed, nb_qk_chan, nb_v_chan, nb_head, scale=False):
        super(MultiHeadSelfAttention, self).__init__()
        assert nb_qk_chan % nb_head == 0
        self.register_buffer('b', torch.tril(torch.ones(nb_embed, nb_embed)).view(1, 1, nb_embed, nb_embed))
        self.nb_head = nb_head
        self.split_size = nb_qk_chan
        self.scale = scale
        self.qk_projection = Linear(nb_qk_chan, nb_qk_chan * 2)
        self.v_projection = Linear(nb_qk_chan, nb_v_chan)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1000000000.0 * (1 - self.b)
        w = Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.nb_head, x.size(-1) // self.nb_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        :param x: A tensor with a shape of [batch, nb_embed, nb_channel]
        :return: A tensor with a shape of [batch, nb_embed, nb_channel]
        """
        qk = self.qk_projection(x)
        query, key = qk.split(self.split_size, dim=2)
        value = self.v_projection(x)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        return self.merge_heads(a)


class RMCCell(torch.nn.Module):
    """
    Strict implementation a Relational Memory Core.

    Paper: https://arxiv.org/pdf/1806.01822.pdf
    Reference implementation: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
    """

    def __init__(self, nb_input_embed, nb_memory_embed, nb_channel, nb_head=1, nb_block=1, nb_mlp=2, input_bias=0, forget_bias=1):
        super(RMCCell, self).__init__()
        self._mem_slots = nb_memory_embed
        self._head_size = nb_channel
        self._num_heads = nb_head
        self._nb_block = nb_block
        self._nb_total_mem_chan = nb_channel * nb_head
        self._input_bias = input_bias
        self._forget_bias = forget_bias
        self.input_linear = Linear(nb_channel, self._nb_total_mem_chan)
        self.ih = Linear(nb_channel, 2 * self._nb_total_mem_chan)
        self.hh = Linear(nb_channel, 2 * self._nb_total_mem_chan)
        self.ih.bias.data.fill_(0)
        self.hh.bias.data.fill_(0)
        self.attention = MultiHeadSelfAttention(nb_input_embed + nb_memory_embed, nb_channel, nb_channel, 1, scale=True)
        self.mlp = torch.nn.ModuleList([Linear(self._nb_total_mem_chan, self._nb_total_mem_chan) for _ in range(nb_mlp)])
        self.ln1 = torch.nn.LayerNorm([nb_input_embed + nb_memory_embed, self._nb_total_mem_chan])
        self.ln2 = torch.nn.LayerNorm([nb_input_embed + nb_memory_embed, self._nb_total_mem_chan])

    def _attend(self, memory):
        for _ in range(self._nb_block):
            attended_mem = self.attention(memory)
            memory = self.ln1(memory + attended_mem)
            mlp_mem = memory
            for layer in self.mlp:
                mlp_mem = F.relu(layer(mlp_mem))
            memory = self.ln2(mlp_mem + memory)
        return memory

    def forward(self, input, prev_memory):
        """
        B: Batch length
        E: Embeddings
        C: Channels

        Type{Shape}[Contents]
        :param input: Tensor{B, Ei, Ci}
        :param prev_memory: Tensor{B, Em, Cm}
        :return:
        """
        input = self.input_linear(input)
        memory_plus_input = torch.cat([prev_memory, input], dim=1)
        next_memory = self._attend(memory_plus_input)
        next_memory = next_memory[:, :-prev_memory.size(1), :]
        i2h = self.ih(input)
        h2h = self.hh(prev_memory.tanh())
        preact = i2h + h2h
        input_gate, forget_gate = torch.chunk(preact, 2, dim=2)
        input_gate = (input_gate + self._input_bias).sigmoid()
        forget_gate = (forget_gate + self._forget_bias).sigmoid()
        next_memory = input_gate * next_memory.tanh()
        next_memory += forget_gate * prev_memory
        return next_memory


class RelationalMHDPA(nn.Module):
    """
    Multi-head dot product attention.
    Adapted from:
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_py.py
    Reference implementation (Tensorflow):
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L2674
    """

    def __init__(self, height, width, nb_channel, nb_head, scale=False):
        super(RelationalMHDPA, self).__init__()
        assert nb_channel % nb_head == 0
        seq_len = height * width
        self.register_buffer('b', torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
        self.nb_head = nb_head
        self.split_size = nb_channel
        self.scale = scale
        self.projection = nn.Linear(nb_channel, nb_channel * 3)
        self.mlp = nn.Linear(nb_channel, nb_channel)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        None
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1000000000.0 * (1 - self.b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.nb_head, x.size(-1) // self.nb_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        :param x: A tensor with a shape of [batch, seq_len, nb_channel]
        :return: A tensor with a shape of [batch, seq_len, nb_channel]
        """
        size_out = x.size()[:-1] + (self.split_size * 3,)
        x = self.projection(x.view(-1, x.size(-1)))
        x = x.view(*size_out)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        e = self.merge_heads(a)
        return self.mlp(e)

    def get_parameter_names(self, layer):
        return ['Proj{}_W'.format(layer), 'Proj{}_b'.format(layer), 'MLP{}_W'.format(layer), 'MLP{}_b'.format(layer)]


BATCH_AXIS = 0


CHANNEL_AXIS = 1


class CircularDND(torch.nn.Module):
    """
    Does not support batches
    """

    def __init__(self, nb_key_chan, nb_v_chan, delta=0.001, query_width=50, max_len=1028):
        super(CircularDND, self).__init__()
        self.delta = delta
        self.query_width = query_width
        self.keys = torch.nn.Parameter(torch.zeros(max_len, nb_key_chan, requires_grad=True))
        self.values = torch.nn.Parameter(torch.zeros(max_len, nb_v_chan, requires_grad=True))

    def forward(self, key):
        inds, weights = self._k_nearest(key, self.query_width)
        return torch.sum(self.values[(inds), :] * weights, BATCH_AXIS, keepdim=True)

    def _k_nearest(self, key, k):
        lookup_weights = self._kernel(key, self.keys)
        top_ks, top_k_inds = torch.topk(lookup_weights, k)
        weights = (top_ks / torch.sum(lookup_weights)).unsqueeze(CHANNEL_AXIS)
        return top_k_inds, weights

    def _kernel(self, query_key, all_keys):
        return 1.0 / (torch.pow(query_key - all_keys, 2).sum(CHANNEL_AXIS) + self.delta)

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys.detach_()
        self.values.detach_()


class PruningDND(torch.nn.Module):
    """
    Does not support batches.
    """

    def __init__(self, nb_key_chan, nb_v_chan, delta=0.001, query_width=50, max_len=1024):
        super(PruningDND, self).__init__()
        self.delta = delta
        self.query_width = query_width
        self.keys = torch.nn.Parameter(torch.rand(max_len, nb_key_chan))
        self.values = torch.nn.Parameter(torch.zeros(max_len, nb_v_chan))
        self.register_buffer('weight_buff', torch.zeros(max_len))

    def forward(self, key):
        inds, weights = self._k_nearest(key, self.query_width)
        return torch.sum(self.values[(inds), :] * weights.unsqueeze(CHANNEL_AXIS), BATCH_AXIS, keepdim=True), inds, weights

    def _k_nearest(self, key, k):
        lookup_weights = self._kernel(key, self.keys)
        top_ks, top_k_inds = torch.topk(lookup_weights, k)
        weights = top_ks / torch.sum(lookup_weights)
        return top_k_inds, weights

    def _kernel(self, query_key, all_keys):
        return 1.0 / (torch.pow(query_key - all_keys, 2).sum(CHANNEL_AXIS) + self.delta)

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys.detach_()
        self.values.detach_()

    def update_buff(self, inds, weights):
        self.weight_buff[inds] = weights

    def append(self, new_k, new_v):
        min_idx = torch.argmin(self.weight_buff).item()
        self.keys[(min_idx), :] = new_k
        self.values[(min_idx), :] = new_v
        self.weight_buff[min_idx] = torch.mean(self.weight_buff)


class FreqPruningLTM(torch.nn.Module):

    def __init__(self, nb_key_chan, nb_v_chan, query_breadth=50, max_len=1024):
        super(FreqPruningLTM, self).__init__()
        self.query_breadth = query_breadth
        self.keys = torch.nn.Parameter(torch.randn(max_len, nb_key_chan))
        self.values = torch.nn.Parameter(torch.randn(max_len, nb_v_chan))
        self.register_buffer('weight_buff', torch.zeros(max_len))

    def forward(self, queries):
        """
        :param queries: expecting a [Batch Size, Num Key Channel] matrix
        :return: a [Batch Size, Num Value Channel] matrix
        """
        inds, weights = self._k_nearest(queries, self.query_breadth)
        values = self.values.unsqueeze(0)
        values = values.expand(inds.size(0), values.size(1), values.size(2))
        inds_tmp = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), values.size(2))
        selected_values = values.gather(1, inds_tmp)
        weighted_selection = (selected_values * weights.unsqueeze(2)).sum(1)
        return weighted_selection, inds, weights

    def _k_nearest(self, queries, query_width):
        lookup_weights = torch.mm(queries, torch.t(self.keys))
        top_ks, top_k_inds = torch.topk(lookup_weights, query_width)
        weights = F.softmax(top_ks, dim=CHANNEL_AXIS)
        return top_k_inds, weights

    def sync_from_shared(self, shared_dnd):
        self.load_state_dict(shared_dnd.state_dict())

    def sync_to_shared(self, shared_dnd):
        is_cpu = self.keys.device.type == 'cpu'
        if shared_dnd.keys.grad is not None and is_cpu:
            return
        elif is_cpu:
            shared_dnd.keys._grad = self.keys.grad
            shared_dnd.values._grad = self.values.grad
        else:
            shared_dnd.keys._grad = self.keys.grad.cpu()
            shared_dnd.values._grad = self.values.grad.cpu()

    def detach(self):
        self.keys = self.keys.detach_()
        self.values = self.values.detach_()

    def update_buff(self, inds, weights):
        self.weight_buff[inds] = weights

    def append(self, new_k, new_v):
        """
        :param new_k: expecting a vector of dimensionality [Num Key Chan]
        :param new_v: expecting a vector of dimensionality [Num Value Chan]
        :return:
        """
        min_idx = torch.argmin(self.weight_buff).item()
        self.keys[(min_idx), :] = new_k
        self.values[(min_idx), :] = new_v
        self.weight_buff[min_idx] = torch.mean(self.weight_buff)


class GaussianLinear(nn.Module):

    def __init__(self, fan_in, nodes):
        super().__init__()
        self.mu = nn.Linear(fan_in, nodes)
        self.std = nn.Linear(fan_in, nodes)

    def forward(self, x):
        mu = self.mu(x)
        if self.training:
            std = self.std(x)
            std = torch.exp(0.5 * std)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_parameter_names(self):
        return ['Mu_W', 'Mu_b', 'Std_W', 'Std_b']


class NoisyLinear(nn.Linear):
    """
    Reference implementation:
    https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))
        self.init_params()

    def init_params(self):
        limit = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-limit, limit)
        self.bias.data.uniform_(-limit, limit)
        self.sigma_weight.data.fill_(self.sigma_init)
        self.sigma_bias.data.fill_(self.sigma_init)

    def forward(self, x, internals):
        if self.training:
            w = self.weight + self.sigma_weight * internals[0]
            b = self.bias + self.sigma_bias * internals[1]
        else:
            w = self.weight + self.sigma_weight
            b = self.bias + self.sigma_bias
        return F.linear(x, w, b)

    def batch_forward(self, x, internals, batch_size=None):
        None
        batch_size = batch_size if batch_size is not None else x.shape[0]
        x = x.unsqueeze(1)
        eps_w, eps_b = zip(*internals)
        eps_w = torch.stack(eps_w)
        eps_b = torch.stack(eps_b)
        batch_w = self.weight.unsqueeze(0).expand(batch_size, -1, -1) + self.sigma_weight.unsqueeze(0).expand(batch_size, -1, -1)
        batch_w += eps_w
        batch_w = batch_w.permute(0, 2, 1)
        batch_b = self.bias.expand(batch_size, -1) + self.sigma_bias.expand(batch_size, -1)
        batch_b += eps_b
        bmm = torch.bmm(x, batch_w).squeeze(1)
        return bmm + batch_b

    def reset(self, gpu=False, device=None):
        if not gpu:
            return torch.randn(self.out_features, self.in_features).detach(), torch.randn(self.out_features).detach()
        else:
            return torch.randn(self.out_features, self.in_features).detach(), torch.randn(self.out_features).detach()

    def get_parameter_names(self):
        return ['W', 'b', 'sigma_W', 'sigma_b']


class RequiresArgsMixin(metaclass=abc.ABCMeta):
    """
    This mixin makes it so that subclasses must implement an args class
    attribute. These arguments are parsed at runtime and the user is offered a
    chance to change any desired args. Classes the use this mixin must
    implement the from_args() class method. from_args() is essentially a
    secondary constructor.
    """
    args = None

    @classmethod
    def check_args_implemented(cls):
        if cls.args is None:
            raise NotImplementedError('Subclass must define class attribute "args"')

    @classmethod
    def prompt(cls, provided=None):
        """
        Display defaults as JSON, prompt user for changes.

        :param provided: Dict[str, Any], Override default prompts.
        :return: Dict[str, Any] Updated config dictionary.
        """
        if provided is not None:
            overrides = {k: v for k, v in provided.items() if k in cls.args}
            args = {**cls.args, **overrides}
        else:
            args = cls.args
        return cls._prompt(cls.__name__, args)

    @staticmethod
    def _prompt(name, args):
        """
        Display defaults as JSON, prompt user for changes.

        :param name: str Name of class
        :param args: Dict[str, Any]
        :return: Dict[str, Any] Updated config dictionary.
        """
        if not args:
            return args
        user_input = input('\n{} Defaults:\n{}\nPress ENTER to use defaults. Otherwise, modify JSON keys then press ENTER.\n'.format(name, json.dumps(args, indent=2, sort_keys=True)) + 'Example: {"x": True, "gamma": 0.001}\n')
        if user_input == '':
            return args
        updates = json.loads(user_input)
        return {**args, **updates}

    @classmethod
    @abc.abstractmethod
    def from_args(cls, *argss, **kwargs):
        raise NotImplementedError


class RewardNormModule(RequiresArgsMixin, metaclass=abc.ABCMeta):

    def __call__(self, reward):
        """
        Normalizes a reward tensor.

        :param reward: torch.Tensor (1D)
        :return:
        """
        raise NotImplementedError


class Identity(RewardNormModule):
    args = {}

    @classmethod
    def from_args(cls, args):
        return cls()

    def __call__(self, item):
        return item


class LSTMCellLayerNorm(Module):
    """
    A lstm cell that layer norms the cell state
    https://github.com/seba-1511/lstms.pth/blob/master/lstms/lstm.py for reference.
    Original License Apache 2.0
    """

    def __init__(self, input_size, hidden_size, bias=True, forget_bias=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = Linear(hidden_size, 4 * hidden_size, bias=bias)
        if bias:
            self.ih.bias.data.fill_(0)
            self.hh.bias.data.fill_(0)
            self.ih.bias.data[hidden_size:hidden_size * 2].fill_(forget_bias)
            self.hh.bias.data[hidden_size:hidden_size * 2].fill_(forget_bias)
        self.ln_cell = LayerNorm(hidden_size)

    def forward(self, x, hidden):
        """
        LSTM Cell that layer normalizes the cell state.
        :param x: Tensor{B, C}
        :param hidden: A Tuple[Tensor{B, C}, Tensor{B, C}] of (previous output, cell state)
        :return:
        """
        h, c = hidden
        i2h = self.ih(x)
        h2h = self.hh(h)
        preact = i2h + h2h
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        c_t = self.ln_cell(c_t)
        h_t = torch.mul(o_t, c_t.tanh())
        return h_t, c_t


class Residual2DPreact(nn.Module):

    def __init__(self, nb_in_chan, nb_out_chan, stride=1):
        super(Residual2DPreact, self).__init__()
        self.nb_in_chan = nb_in_chan
        self.nb_out_chan = nb_out_chan
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(nb_in_chan)
        self.conv1 = nn.Conv2d(nb_in_chan, nb_out_chan, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_out_chan)
        self.conv2 = nn.Conv2d(nb_out_chan, nb_out_chan, 3, stride=1, padding=1, bias=False)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.do_projection = self.nb_in_chan != self.nb_out_chan or self.stride > 1
        if self.do_projection:
            self.projection = nn.Conv2d(nb_in_chan, nb_out_chan, 3, stride=stride, padding=1)
            self.projection.weight.data.mul_(relu_gain)

    def forward(self, x):
        first = F.relu(self.bn1(x))
        if self.do_projection:
            projection = self.projection(first)
        else:
            projection = x
        x = self.conv1(first)
        x = self.conv2(F.relu(self.bn2(x)))
        return x + projection


class BaseNetwork(torch.nn.Module):

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, observation_space, output_space, gpu_preprocessor, net_reg):
        raise NotImplementedError

    @abc.abstractmethod
    def new_internals(self, device):
        """
        :return: Dict[InternalKey, torch.Tensor (ND)]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, observation, internals):
        raise NotImplementedError

    def internal_space(self):
        return {k: t.shape for k, t in self.new_internals('cpu').items()}

    def sync(self, src, grp=None, async_op=False):
        keys = []
        handles = []
        for k, t in self.state_dict().items():
            if grp is None:
                h = dist.broadcast(t, src, async_op=True)
            else:
                h = dist.broadcast(t, src, grp, async_op=True)
            keys.append(k)
            handles.append(h)
        if not async_op:
            for k, h in zip(keys, handles):
                h.wait()
        return handles


class SubModule(torch.nn.Module, RequiresArgsMixin, metaclass=abc.ABCMeta):
    """
    SubModule of a ModularNetwork.
    """
    dim = None

    def __init__(self, input_shape, id):
        """
        Parameters
        ----------
        input_shape : tuple[int]
            Input shape excluding batch dimension
        id : str
            Unique identifier for this instance
        """
        super(SubModule, self).__init__()
        self._input_shape = input_shape
        self._id = id

    @classmethod
    @abc.abstractmethod
    def from_args(cls, args, input_shape, id):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _output_shape(self):
        """Output shape excluding batch dimension

        Returns
        -------
        tuple[int]
            Output shape exlcuding batch dimension
        """
        raise NotImplementedError

    def output_shape(self, dim=None):
        """Output shape casted to requested dimension

        Parameters
        ----------
        dim : int, optional
            Desired dimensionality, defaults to native

        Returns
        -------
        tuple[int]
            Output shape
        """
        if dim is None:
            dim = len(self._output_shape)
        if dim == 1:
            return self._to_1d_shape()
        elif dim == 2 or dim is None:
            return self._to_2d_shape()
        elif dim == 3:
            return self._to_3d_shape()
        elif dim == 4:
            return self._to_4d_shape()
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    @abc.abstractmethod
    def _forward(self, input, internals, **kwargs):
        """
        :param input: torch.Tensor (B+1D | B+2D | B+3D | B+4D)
        :return: Tuple[Result, Internals]
        """
        raise NotImplementedError

    def _to_1d(self, submodule_output):
        """Convert to Batch + 1D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor

        Returns
        -------
        torch.Tensor
            Batch + 1D Tensor
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_1d_shape())

    def _to_2d(self, submodule_output):
        """Convert to Batch + 2D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 2D Tensor (B, S, F)
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_2d_shape())

    def _to_3d(self, submodule_output):
        """Convert to Batch + 3D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 3D Tensor
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_3d_shape())

    def _to_4d(self, submodule_output):
        """Convert to Batch + 4D

        Parameters
        ----------
        submodule_output : torch.Tensor
            Batch + 2D Tensor (B, S, F)

        Returns
        -------
        torch.Tensor
            Batch + 4D Tensor (B, F, S, H, W)
        """
        b = submodule_output.size()[0]
        return submodule_output.view(b, *self._to_4d_shape())

    @abc.abstractmethod
    def _to_1d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_2d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_3d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _to_4d_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _new_internals(self):
        """
        :return: Dict[InternalKey, List[torch.Tensor (ND)]]
        """
        raise NotImplementedError

    @property
    def id(self):
        return self._id

    @property
    def input_shape(self):
        return self._input_shape

    def new_internals(self, device):
        return {(self.id + k): v for k, v in self._new_internals().items()}

    def stacked_internals(self, key, internals):
        return torch.stack(internals[self.id + key])

    def to_dim(self, submodule_output, dim):
        """
        :param submodule_output: torch.Tensor (1D | 2D | 3D | 4D)
        Output of a forward pass to be converted.
        :param dim: int Desired dimensionality
        :return:
        """
        if dim <= 0 or dim > 4:
            raise ValueError('Invalid dim: {}'.format(dim))
        elif dim == 1:
            return self._to_1d(submodule_output)
        elif dim == 2:
            return self._to_2d(submodule_output)
        elif dim == 3:
            return self._to_3d(submodule_output)
        elif dim == 4:
            return self._to_4d(submodule_output)

    def forward(self, *input, dim=None):
        submodule_output, internals = self._forward(*input)
        if dim is None:
            return submodule_output, self._id_internals(internals)
        if dim == 1:
            return self._to_1d(submodule_output), self._id_internals(internals)
        elif dim == 2:
            return self._to_2d(submodule_output), self._id_internals(internals)
        elif dim == 3:
            return self._to_3d(submodule_output), self._id_internals(internals)
        elif dim == 4:
            return self._to_4d(submodule_output), self._id_internals(internals)
        else:
            raise ValueError('Invalid dim: {}'.format(dim))

    def _id_internals(self, internals):
        return {(self.id + k): v for k, v in internals.items()}


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    """
    A neural network comprised of SubModules. Tries to be smart about
    converting dimensionality. Does not need or build submodules for unused
    source nets or heads.
    """

    def __init__(self, source_nets, body_submodule, head_submodules, output_space, gpu_preprocessor):
        """
        :param source_nets: Dict[ObsKey, SubModule]
        :param body_submodule: SubModule
        :param head_submodules: Dict[Dim, SubModule]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        """
        super().__init__()
        self.gpu_preprocessor = gpu_preprocessor
        self.source_nets = torch.nn.ModuleDict([(key, net) for key, net in source_nets.items()])
        self.body = body_submodule
        self.heads = torch.nn.ModuleDict(head_submodules)
        self.output_layers = self._build_out_layers(output_space, self.heads)
        self._obs_keys = list(source_nets.keys())
        self._output_keys = list(output_space.keys())
        self._output_space = output_space
        self._output_dims = set([len(shape) for shape in self._output_space.values()])
        self._check_outputs_have_heads()
        self._validate_shapes()

    @staticmethod
    def _build_out_layers(output_space, heads):
        """
        Build output_layers to match the desired output space.
        * For 1D outputs, converts uses a Linear layer
        * For 2D outputs, uses a Conv1D, kernel size 1
        * For 3D outputs, uses a 1x1 Conv
        * For 4D outputs, uses a 1x1x1 Conv

        :param output_space: Dict[OutputKey, Shape]
        :param heads: Dict[DimStr, SubModule]
        :return: ModuleDict[OutputKey, torch.nn.Module]
        """
        outputs = []
        for output_name, shape in output_space.items():
            dim = len(shape)
            if dim == 1:
                layer = torch.nn.Linear(heads[str(dim)].output_shape(dim)[0], shape[0])
            elif dim == 2:
                layer = torch.nn.Conv1d(heads[str(dim)].output_shape(dim)[0], shape[0], kernel_size=1)
            elif dim == 3:
                layer = torch.nn.Conv2d(heads[str(dim)].output_shape(dim)[0], shape[0], kernel_size=1)
            elif dim == 4:
                layer = torch.nn.Conv3d(heads[str(dim)].output_shape(dim)[0], shape[0], kernel_size=1)
            else:
                raise ValueError('Invalid dim {}'.format(dim))
            outputs.append((output_name, layer))
        return torch.nn.ModuleDict(outputs)

    def _validate_shapes(self):
        """
        Ensures SubModule graph is valid.
        :return:
        """
        if self.body.dim > 1:
            for submod in self.source_nets.values():
                if submod.dim > 1:
                    shape = submod.output_shape(dim=self.body.dim)
                    for a, b in zip(shape[1:], self.body.input_shape[1:]):
                        assert a == b or a == 1 or b == 1, 'Source-Body conflict: {} {}'.format(shape, self.body.input_shape)
            for submod in self.heads.values():
                if submod.dim > 1:
                    shape = self.body.output_shape(dim=submod.dim)
                    for a, b in zip(shape[1:], submod.input_shape[1:]):
                        assert a == b or a == 1 or b == 1, 'Body-Head conflict: {} {}'.format(shape, submod.input_shape)
        for shape in self._output_space.values():
            dim = len(shape)
            if dim > 1:
                submod = self.heads[str(dim)]
                head_shp = submod.output_shape(dim)
                for a, b in zip(shape[1:], head_shp[1:]):
                    assert a == b, 'Head-Output conflict: {}-{}'.format(head_shp, shape)

    def _check_outputs_have_heads(self):
        for dim in self._output_dims:
            assert str(dim) in self.heads

    @classmethod
    def from_args(cls, args, observation_space, output_space, gpu_preprocessor, net_reg):
        """
        Construct a Modular Network from arguments.

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        :param net_reg: NetworkRegistry
        :return: ModularNetwork
        """
        obs_key_to_submod = {}
        for obs_key, shape in observation_space.items():
            dim = len(shape)
            if dim == 1:
                submod = net_reg.lookup_submodule(args.net1d).from_args(args, shape, obs_key)
            elif dim == 2:
                submod = net_reg.lookup_submodule(args.net2d).from_args(args, shape, obs_key)
            elif dim == 3:
                submod = net_reg.lookup_submodule(args.net3d).from_args(args, shape, obs_key)
            elif dim == 4:
                submod = net_reg.lookup_submodule(args.net4d).from_args(args, shape, obs_key)
            else:
                raise ValueError('Invalid dim: {}'.format(dim))
            obs_key_to_submod[obs_key] = submod
        body_cls = net_reg.lookup_submodule(args.netbody)
        nb_body_feature = sum([submod.output_shape(dim=body_cls.dim)[0] for submod in obs_key_to_submod.values()])
        if body_cls.dim > 1:
            other_dims = [submod.output_shape(dim=body_cls.dim)[1:] for submod in obs_key_to_submod.values() if submod.dim == body_cls.dim][0]
        else:
            other_dims = []
        input_shape = [nb_body_feature] + list(other_dims)
        body_submod = body_cls.from_args(args, input_shape, 'body')
        head_submodules = {}
        for output_key, shape in output_space.items():
            dim = len(shape)
            if dim in head_submodules:
                continue
            elif dim == 1:
                submod_cls = net_reg.lookup_submodule(args.head1d)
            elif dim == 2:
                submod_cls = net_reg.lookup_submodule(args.head2d)
            elif dim == 3:
                submod_cls = net_reg.lookup_submodule(args.head3d)
            elif dim == 4:
                submod_cls = net_reg.lookup_submodule(args.head4d)
            else:
                raise ValueError('Invalid dim: {}'.format(dim))
            submod = submod_cls.from_args(args, body_submod.output_shape(submod_cls.dim), 'head' + str(dim) + 'd')
            head_submodules[str(dim)] = submod
        return cls(obs_key_to_submod, body_submod, head_submodules, output_space, gpu_preprocessor)

    def forward(self, observation, internals):
        """

        :param observation: Dict[str, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[str, torch.Tensor (ND)]
        :return: Tuple[
            Dict[str, torch.Tensor (1D | 2D | 3D | 4D)],
            Dict[str, torch.Tensor (ND)]
        ]
        """
        proc_obs = self.gpu_preprocessor(observation)
        nxt_internals = []
        processed_inputs = []
        for key in self._obs_keys:
            result, nxt_internal = self.source_nets[key].forward(proc_obs[key], internals, dim=self.body.dim)
            processed_inputs.append(result)
            nxt_internals.append(nxt_internal)
        processed_inputs = self._expand_dims(processed_inputs)
        body_out, nxt_internal = self.body.forward(torch.cat(processed_inputs, dim=1), internals)
        nxt_internals.append(nxt_internal)
        head_out_by_dim = {}
        for dim in self._output_dims:
            cur_head = self.heads[str(dim)]
            head_out, next_internal = cur_head.forward(self.body.to_dim(body_out, cur_head.dim), internals, dim=dim)
            head_out_by_dim[dim] = head_out
            nxt_internals.append(nxt_internal)
        output_by_key = {}
        for key in self._output_keys:
            output = self.output_layers[key].forward(head_out_by_dim[len(self._output_space[key])])
            output_by_key[key] = output
        merged_internals = {}
        for internal in nxt_internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return output_by_key, merged_internals, proc_obs

    @staticmethod
    def _expand_dims(inputs):
        """
        Expands dimensions when input dimension is 1.

        :param inputs: List[torch.Tensor]
        :return: List[torch.Tensor]
        """
        if len(inputs[0].shape) <= 2:
            return inputs
        target_shape = max([inpt.shape[2:] for inpt in inputs])
        processed_inputs = []
        for inpt in inputs:
            if inpt.shape[2:] < target_shape:
                processed_inputs.append(inpt.expand(-1, -1, *target_shape))
            else:
                processed_inputs.append(inpt)
        return processed_inputs

    def new_internals(self, device):
        """

        :param device:
        :return: Dict[
        """
        internals = [submod.new_internals(device) for submod in self.source_nets.values()]
        internals.append(self.body.new_internals(device))
        internals += [submod.new_internals(device) for submod in self.heads.values()]
        merged_internals = {}
        for internal in internals:
            for k, v in internal.items():
                merged_internals[k] = v
        return merged_internals


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, nb_input_channel, nb_output_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(nb_input_channel, nb_output_channel, stride)
        self.bn1 = nn.BatchNorm2d(nb_output_channel)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(nb_output_channel, nb_output_channel)
        self.bn2 = nn.BatchNorm2d(nb_output_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, nb_input_channel, nb_output_channel, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(nb_input_channel)
        self.conv1 = conv3x3(nb_input_channel, nb_output_channel, stride)
        self.bn2 = nn.BatchNorm2d(nb_output_channel)
        self.conv2 = conv3x3(nb_output_channel, nb_output_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, nb_in_channel, nb_out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(nb_in_channel, nb_out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_out_channel)
        self.conv2 = nn.Conv2d(nb_out_channel, nb_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_out_channel)
        self.conv3 = nn.Conv2d(nb_out_channel, nb_out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_out_channel * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, nb_in_channel, nb_out_channel, stride=1, downsample=None):
        super(BottleneckV2, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(nb_in_channel)
        self.conv1 = nn.Conv2d(nb_in_channel, nb_out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_out_channel)
        self.conv2 = nn.Conv2d(nb_out_channel, nb_out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_out_channel)
        self.conv3 = nn.Conv2d(nb_out_channel, nb_out_channel * self.expansion, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResNet(nn.Module):

    def __init__(self, block, layer_sizes):
        self.nb_input_channel = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layer_sizes[0])
        self.layer2 = self._make_layer(block, 128, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_sizes[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.nb_output_channel = 512 * block.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.nb_input_channel != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.nb_input_channel, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.nb_input_channel, planes, stride, downsample)]
        self.nb_input_channel = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.nb_input_channel, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class Nature(SubModule):
    """
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    """

    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 3136
        self.conv1 = Conv2d(in_shape[0], 32, 8, stride=4, padding=0, bias=bias)
        self.conv2 = Conv2d(32, 64, 4, stride=2, padding=0, bias=bias)
        self.conv3 = Conv2d(64, 64, 3, stride=1, padding=0, bias=bias)
        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(64)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = xs.view(xs.size(0), -1)
        return xs


class Mnih2013(SubModule):

    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 2592
        self.conv1 = Conv2d(in_shape[0], 16, 8, stride=4, padding=0, bias=bias)
        self.conv2 = Conv2d(16, 32, 4, stride=2, padding=0, bias=bias)
        if normalize:
            self.bn1 = BatchNorm2d(16)
            self.bn2 = BatchNorm2d(32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = xs.view(xs.size(0), -1)
        return xs


class FourConvSpatialAttention(SubModule):
    """
    https://arxiv.org/pdf/1806.01830.pdf
    """

    def __init__(self, in_shape, nb_head, normalize):
        self._nb_output_channel = 800
        super().__init__()
        self.normalize = normalize
        bias = not normalize
        self.conv1 = Conv2d(in_shape[0], 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.attention = MultiHeadSelfAttention(20 * 20, 34, 34, nb_head)
        self.mlp = Linear(34, 34)
        self.conv3 = Conv2d(34, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=bias)
        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(32)
            self.bn3 = BatchNorm2d(32)
            self.bn4 = BatchNorm2d(32)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.nb_head, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, input, internals):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        xs_chan = torch.linspace(-1, 1, 20).view(1, 1, 1, 20).expand(input.size(0), 1, 20, 20)
        ys_chan = torch.linspace(-1, 1, 20).view(1, 1, 20, 1).expand(input.size(0), 1, 20, 20)
        x = torch.cat([x, xs_chan, ys_chan], dim=1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0), x.size(1), h * w).transpose(1, 2)
        x = self.attention(x.contiguous())
        x = F.relu(self.mlp(x))
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), h, w)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        return x


class FourConvLarger(SubModule):

    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self._nb_output_channel = 3200
        self.conv1 = Conv2d(in_shape[0], 32, 7, stride=2, padding=1, bias=bias)
        self.conv2 = Conv2d(32, 64, 3, stride=2, padding=1, bias=bias)
        self.conv3 = Conv2d(64, 64, 3, stride=2, padding=1, bias=bias)
        self.conv4 = Conv2d(64, 128, 3, stride=2, padding=1, bias=bias)
        if normalize:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)
            self.bn3 = BatchNorm2d(64)
            self.bn4 = BatchNorm2d(128)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            self.bn4 = Identity()
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self._nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = F.relu(self.bn2(self.conv2(xs)))
        xs = F.relu(self.bn3(self.conv3(xs)))
        xs = F.relu(self.bn4(self.conv4(xs)))
        xs = xs.view(xs.size(0), -1)
        return xs


class BaseResNet(SubModule, metaclass=abc.ABCMeta):

    def __init__(self, in_shape, normalize):
        super().__init__()
        bias = not normalize
        self.conv1 = Conv2d(in_shape[0], 64, 7, stride=2, padding=1, bias=bias)
        relu_gain = init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        if normalize:
            self.bn1 = BatchNorm2d(64)
        else:
            self.bn1 = Identity()

    @property
    @abc.abstractmethod
    def resnet(self):
        raise NotImplementedError()

    @classmethod
    def from_args(cls, in_shape, args):
        return cls(in_shape, args.normalize)

    @property
    def nb_output_channel(self):
        return self.resnet.nb_output_channel

    def forward(self, xs):
        xs = F.relu(self.bn1(self.conv1(xs)))
        xs = self.resnet(xs)
        xs = xs.view(xs.size(0), -1)
        return xs


def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


class ResNet18(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet18()

    @property
    def resnet(self):
        return self._resnet


def resnet18v2():
    model = ResNet(BasicBlockV2, [2, 2, 2, 2])
    return model


class ResNet18V2(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet18v2()

    @property
    def resnet(self):
        return self._resnet


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


class ResNet34(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet34()

    @property
    def resnet(self):
        return self._resnet


def resnet34v2():
    model = ResNet(BasicBlockV2, [3, 4, 6, 3])
    return model


class ResNet34V2(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet34v2()

    @property
    def resnet(self):
        return self._resnet


def resnet50v2():
    model = ResNet(BottleneckV2, [3, 4, 6, 3])
    return model


class ResNet50(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet50v2()

    @property
    def resnet(self):
        return self._resnet


class ResNet50V2(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet50v2()

    @property
    def resnet(self):
        return self._resnet


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


class ResNet101(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet101()

    @property
    def resnet(self):
        return self._resnet


def resnet101v2():
    model = ResNet(BottleneckV2, [3, 4, 23, 3])
    return model


class ResNet101V2(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet101v2()

    @property
    def resnet(self):
        return self._resnet


def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


class ResNet152(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet152()

    @property
    def resnet(self):
        return self._resnet


def resnet152v2():
    model = ResNet(BottleneckV2, [3, 8, 36, 3])
    return model


class ResNet152V2(BaseResNet):

    def __init__(self, in_shape, normalize):
        super().__init__(in_shape, normalize)
        self._resnet = resnet152v2()

    @property
    def resnet(self):
        return self._resnet


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'nb_input_channel': 4, 'nb_output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockV2,
     lambda: ([], {'nb_input_channel': 4, 'nb_output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CircularDND,
     lambda: ([], {'nb_key_chan': 4, 'nb_v_chan': 4}),
     lambda: ([torch.rand([1028, 4])], {}),
     False),
    (FreqPruningLTM,
     lambda: ([], {'nb_key_chan': 4, 'nb_v_chan': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GaussianLinear,
     lambda: ([], {'fan_in': 4, 'nodes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadSelfAttention,
     lambda: ([], {'nb_embed': 4, 'nb_qk_chan': 4, 'nb_v_chan': 4, 'nb_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PruningDND,
     lambda: ([], {'nb_key_chan': 4, 'nb_v_chan': 4}),
     lambda: ([torch.rand([1024, 4])], {}),
     False),
    (RMCCell,
     lambda: ([], {'nb_input_embed': 4, 'nb_memory_embed': 4, 'nb_channel': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Residual2DPreact,
     lambda: ([], {'nb_in_chan': 4, 'nb_out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_heronsystems_adeptRL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

