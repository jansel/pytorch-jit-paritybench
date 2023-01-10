import sys
_module = sys.modules[__name__]
del sys
atari_data = _module
nest_test = _module
setup = _module
setup = _module
batching_queue_test = _module
contiguous_arrays_env = _module
contiguous_arrays_test = _module
core_agent_state_env = _module
core_agent_state_test = _module
dynamic_batcher_test = _module
inference_speed_profiling = _module
polybeast_inference_test = _module
polybeast_learn_function_test = _module
polybeast_loss_functions_test = _module
polybeast_net_test = _module
vtrace_test = _module
atari_wrappers = _module
environment = _module
file_writer = _module
prof = _module
vtrace = _module
fast_transformers = _module
fast_weight = _module
fast_weight_rnn_v2 = _module
layer = _module
model = _module
noneg_polybeast_learner = _module
polybeast = _module
polybeast_env = _module
polybeast_learner = _module
rec_update_fwm_tanh = _module
self_ref_v0 = _module
self_ref_v1 = _module
model = _module
polybeast_learner = _module
procgen_wrappers = _module
eval_delay_multi_sequential = _module
eval_sync = _module
fast_weight = _module
layer = _module
main_few_shot_delayed_multi_sequential = _module
main_few_shot_sync = _module
model_few_shot = _module
resnet_impl = _module
self_ref_v0 = _module
torchmeta_local = _module
datasets = _module
bach = _module
cifar100 = _module
base = _module
cifar_fs = _module
fc100 = _module
cub = _module
doublemnist = _module
helpers = _module
helpers_tabular = _module
letter = _module
miniimagenet = _module
omniglot = _module
one_hundred_plants_margin = _module
one_hundred_plants_shape = _module
one_hundred_plants_texture = _module
pascal5i = _module
tcga = _module
tieredimagenet = _module
triplemnist = _module
utils = _module
transforms = _module
augmentations = _module
categorical = _module
splitters = _module
tabular_transforms = _module
target_transforms = _module
data = _module
dataloader = _module
dataset = _module
sampler = _module
task = _module
wrappers = _module
version = _module
utils_few_shot = _module
warmup_lr = _module

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


import numpy as np


from torch.utils import cpp_extension


import time


from torch import nn


import logging


import warnings


import copy


from torch.nn import functional as F


import collections


import torch.nn.functional as F


from torch.utils.cpp_extension import load


import math


import random


import torch.nn as nn


from collections import defaultdict


from collections import OrderedDict


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataset import Dataset as TorchDataset


from itertools import combinations


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data import ConcatDataset


from torch.utils.data import Subset


from torch.utils.data import Dataset as Dataset_


from torchvision.transforms import Compose


from torch.utils.data import Dataset


class Net(nn.Module):

    def __init__(self, num_actions, conv_scale=1, use_lstm=False):
        super(Net, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        base_num_ch = [16, 32, 32]
        scaled_num_ch = [(c * conv_scale) for c in base_num_ch]
        for num_ch in scaled_num_ch:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048 * conv_scale, 256)
        core_output_size = self.fc.out_features + 1
        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state):
        x = inputs['frame']
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs['done']).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class TransformerFFlayers(nn.Module):

    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True):
        super(TransformerFFlayers, self).__init__()
        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.ff_layers = nn.Sequential(nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ff_dim, res_dim), nn.Dropout(dropout))
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        out = self.ff_layers(out) + x
        return out


@torch.jit.script
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1.0, False) + 1.0
    return y / (y.sum(-1, keepdim=True) + 1e-05)


class AdditiveFastFFlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(AdditiveFastFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_sum
        self.slow_net = nn.Linear(in_dim, num_head * (3 * dim_head), bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)
        qkv = self.slow_net(out)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, -1)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)
        if state is not None:
            fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
        assert torch.isnan(fast_weights).sum().item() == 0, f'Before NaN: fast weights'
        out = self.fw_layer(head_q, head_k, head_v, fast_weights)
        assert torch.isnan(fast_weights).sum().item() == 0, f'NaN: fast weights'
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out
        return out, fast_weights.clone()


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1.0, False) + 1.0


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


class FastFFlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_delta
        self.slow_net = nn.Linear(in_dim, num_head * (3 * dim_head + 1), bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)
        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)
        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)
        head_k = sum_norm(head_k)
        head_q = sum_norm(head_q)
        fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out
        return out


class FastFastFFlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFastFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_memory
        self.very_fw_layer = fast_weight_memory
        self.cached_fast_weights = nn.Parameter(torch.zeros(1, self.num_head, self.dim_head, 3 * self.dim_head + 1), requires_grad=False)
        self.slow_net = nn.Linear(in_dim, num_head * (5 * dim_head + 2), bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)
        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 5 * self.dim_head + 2)
        head_q, head_k, head_v, head_beta = torch.split(qkvb, (self.dim_head, self.dim_head, 3 * self.dim_head + 1, 1), -1)
        head_beta = torch.sigmoid(head_beta)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)
        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)
        if state is not None:
            fast_weights, very_fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, 3 * self.dim_head + 1, device=head_k.device)
            very_fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
        assert torch.isnan(fast_weights).sum().item() == 0, 'Before NaN: fast weights'
        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        assert torch.isnan(fast_weights).sum().item() == 0, 'NaN: fast weights'
        fast_head_q, fast_head_k, fast_head_v, fast_beta = torch.split(out, (self.dim_head,) * 3 + (1,), -1)
        fast_head_q = elu_p1_sum_norm_eps(fast_head_q)
        fast_head_k = elu_p1_sum_norm_eps(fast_head_k)
        fast_beta = torch.sigmoid(fast_beta)
        out = self.very_fw_layer(fast_head_q, fast_head_k, fast_head_v, fast_beta, very_fast_weights)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out
        return out, (fast_weights.clone(), very_fast_weights.clone())


class PseudoSRlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(PseudoSRlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        n_head = num_head
        d_head = dim_head
        self.W_y = nn.Parameter(torch.Tensor(n_head, d_head, d_head), requires_grad=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0.0, std=std)

    def forward(self, h, state=None):
        slen, bsz, _ = h.size()
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = x.reshape(slen * bsz, self.num_head, self.dim_head)
        x = x.permute(1, 0, 2)
        out = torch.bmm(x, self.W_y)
        out = out.permute(1, 0, 2)
        out = out.reshape(slen, bsz, self.num_head, self.dim_head)
        out = out.reshape(slen, bsz, self.num_head * self.dim_head)
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        return out, state


class SRlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(SRlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head
        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4), requires_grad=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0.0, std=std)
        nn.init.normal_(self.W_q, mean=0.0, std=std)
        nn.init.normal_(self.W_k, mean=0.0, std=std)
        nn.init.normal_(self.w_b, mean=0.0, std=std)

    def forward(self, h, state=None):
        slen, bsz, _ = h.size()
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        x = x.permute(1, 2, 0, 3)
        if state is not None:
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
        else:
            assert False
        W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)
        out = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)
        state = W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(), w_b_bc.detach()
        return out, state


class NoCarryOverSRlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(NoCarryOverSRlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head
        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4), requires_grad=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0.0, std=std)
        nn.init.normal_(self.W_q, mean=0.0, std=std)
        nn.init.normal_(self.W_k, mean=0.0, std=std)
        nn.init.normal_(self.w_b, mean=0.0, std=std)

    def forward(self, h, state=None):
        slen, bsz, _ = h.size()
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        x = x.permute(1, 2, 0, 3)
        W_y_bc = self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = self.w_b.repeat(bsz, 1, 1, 1)
        out = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        return out, state


class SMFWPlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(SMFWPlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = self_ref_v1
        n_head = num_head
        d_head = dim_head
        y_d_head = 3 * dim_head + 1
        self.y_d_head = y_d_head
        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, y_d_head), requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4), requires_grad=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0.0, std=std)
        nn.init.normal_(self.W_q, mean=0.0, std=std)
        nn.init.normal_(self.W_k, mean=0.0, std=std)
        nn.init.normal_(self.w_b, mean=0.0, std=std)

    def forward(self, h, state=None):
        slen, bsz, _ = h.size()
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        x = x.permute(1, 2, 0, 3)
        if state is not None:
            W_y_bc, W_q_bc, W_k_bc, w_b_bc, fast_weights = state
        else:
            assert False
        assert torch.isnan(fast_weights).sum().item() == 0, 'Before NaN: fast weights'
        W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)
        fast_qkvb = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
        fast_head_q, fast_head_k, fast_head_v, fast_beta = torch.split(fast_qkvb, (self.dim_head,) * 3 + (1,), -1)
        fast_head_q = F.softmax(fast_head_q, dim=-1)
        fast_head_k = F.softmax(fast_head_k, dim=-1)
        fast_beta = torch.sigmoid(fast_beta)
        out = fast_weight_memory(fast_head_q, fast_head_k, fast_head_v, fast_beta, fast_weights)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)
        state = W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(), w_b_bc.detach(), fast_weights.detach()
        return out, state


class FastRNNlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastRNNlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_delta
        self.rec_fw_layer = fast_rnn_v2
        self.slow_net = nn.Linear(in_dim, num_head * (5 * dim_head + 2), bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)
        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 5 * self.dim_head + 2)
        head_q, head_k, head_v, rec_head_k, rec_head_v, head_beta, rec_beta = torch.split(qkvb, (self.dim_head,) * 5 + (1,) * 2, -1)
        head_beta = torch.sigmoid(head_beta)
        rec_beta = torch.sigmoid(rec_beta)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)
        rec_head_k = rec_head_k.permute(1, 2, 0, 3)
        rec_head_v = rec_head_v.permute(1, 2, 0, 3)
        rec_beta = rec_beta.permute(1, 2, 0, 3)
        head_q = F.softmax(head_q, dim=-1)
        head_k = F.softmax(head_k, dim=-1)
        rec_head_k = F.softmax(rec_head_k, dim=-1)
        if state is None:
            fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
            rec_fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
            state0 = torch.zeros(bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, rec_fast_weights, state0 = state
        assert torch.isnan(fast_weights).sum().item() == 0, f'Before NaN: fast weights'
        z_out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        out = self.rec_fw_layer(z_out, rec_head_k, rec_head_v, rec_fast_weights, rec_beta, state0)
        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out
        return out, (fast_weights.clone(), rec_fast_weights.clone(), state0_next)


class RecUpdateTanhFastFFlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(RecUpdateTanhFastFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = rec_update_fwm_tanh
        self.slow_net = nn.Linear(in_dim, num_head * (3 * dim_head + 1), bias=False)
        self.R_q = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head), requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head), requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head), requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, num_head, 1, dim_head), requires_grad=True)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.R_q, mean=0.0, std=std)
        nn.init.normal_(self.R_k, mean=0.0, std=std)
        nn.init.normal_(self.R_v, mean=0.0, std=std)
        nn.init.normal_(self.r_b, mean=0.0, std=std)

    def forward(self, x, state=None):
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)
        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)
        if state is None:
            fast_weights = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head, device=head_k.device)
            state0 = torch.zeros(bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, state0 = state
        out = self.fw_layer(head_q, head_k, head_v, head_beta, self.R_q, self.R_k, self.R_v, self.r_b, fast_weights, state0)
        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out
        return out, (fast_weights.clone(), state0_next)


class LinearTransformerLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(LinearTransformerLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(AdditiveFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        state_tuple = tuple(state_list)
        return out, state_tuple


class DeeperNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, dim_ff, dropout):
        super(DeeperNetLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        ff_layers = []
        for _ in range(num_layers):
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x):
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out = self.ff_layers[i](out)
        return out


class DeltaNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(DeltaNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(FastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        state_tuple = tuple(state_list)
        return out, state_tuple


class FastFFRecUpdateTanhLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(FastFFRecUpdateTanhLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(RecUpdateTanhFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rnn_states = core_state
        fw_state_list = []
        rnn_state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=(fw_states[i].squeeze(0), rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rnn_state_list.append(out_state[1].unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        fw_state_tuple = tuple(fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = fw_state_tuple, rnn_state_tuple
        return out, state_tuple


class FastRNNModelLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(FastRNNModelLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(FastRNNlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rec_fw_states, rnn_states = core_state
        fw_state_list = []
        rec_fw_state_list = []
        rnn_state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=(fw_states[i].squeeze(0), rec_fw_states[i].squeeze(0), rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rec_fw_state_list.append(out_state[1].unsqueeze(0).clone())
            rnn_state_list.append(out_state[2].unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        fw_state_tuple = tuple(fw_state_list)
        rec_fw_state_tuple = tuple(rec_fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = fw_state_tuple, rec_fw_state_tuple, rnn_state_tuple
        return out, state_tuple


class DeltaDeltaNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(DeltaDeltaNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(FastFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, very_fw_states = core_state
        fw_state_list = []
        very_fw_state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=(fw_states[i].squeeze(0), very_fw_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            very_fw_state_list.append(out_state[1].unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        fw_state_tuple = tuple(fw_state_list)
        very_fw_state_tuple = tuple(very_fw_state_list)
        state_tuple = fw_state_tuple, very_fw_state_tuple
        return out, state_tuple


class SRNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(SRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(SRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        Wy_states, Wq_states, Wk_states, wb_states = core_state
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=(Wy_states[i].squeeze(0), Wq_states[i].squeeze(0), Wk_states[i].squeeze(0), wb_states[i].squeeze(0)))
            Wy_state_list.append(out_state[0].unsqueeze(0).clone())
            Wq_state_list.append(out_state[1].unsqueeze(0).clone())
            Wk_state_list.append(out_state[2].unsqueeze(0).clone())
            wb_state_list.append(out_state[3].unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)
        state_tuple = Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple
        return out, state_tuple


class PseudoSRNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(PseudoSRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(PseudoSRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, _ = self.fwm_layers[i](out)
            out = self.ff_layers[i](out)
        return out, core_state


class NoCarryOverSRNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(NoCarryOverSRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(NoCarryOverSRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, _ = self.fwm_layers[i](out, state=core_state)
            out = self.ff_layers[i](out)
        return out, core_state


class SMFWPNetLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout):
        super(SMFWPNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_size)
        fwm_layers = []
        ff_layers = []
        for _ in range(num_layers):
            fwm_layers.append(SMFWPlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        Wy_states, Wq_states, Wk_states, wb_states, fw_states = core_state
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []
        fw_state_list = []
        out = self.input_proj(x)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=(Wy_states[i].squeeze(0), Wq_states[i].squeeze(0), Wk_states[i].squeeze(0), wb_states[i].squeeze(0), fw_states[i].squeeze(0)))
            Wy_state_list.append(out_state[0].unsqueeze(0).clone())
            Wq_state_list.append(out_state[1].unsqueeze(0).clone())
            Wk_state_list.append(out_state[2].unsqueeze(0).clone())
            wb_state_list.append(out_state[3].unsqueeze(0).clone())
            fw_state_list.append(out_state[4].unsqueeze(0).clone())
            out = self.ff_layers[i](out)
        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)
        fw_state_tuple = tuple(fw_state_list)
        state_tuple = Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple, fw_state_tuple
        return out, state_tuple


class DeltaNetModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(DeltaNetModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = DeltaNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        return state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1, 1, 1)
            core_state = nest.map(nd.mul, core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class LinearTransformerModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(LinearTransformerModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = LinearTransformerLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        return state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1, 1, 1)
            core_state = nest.map(nd.mul, core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class RecDeltaModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(RecDeltaModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = FastFFRecUpdateTanhLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        fw_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        rnn_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, 1, self.dim_head) for _ in range(self.num_layers))
        return fw_state_tuple, rnn_state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            fw_state, rnn_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            fw_state = nest.map(nd.mul, fw_state)
            rnn_state = nest.map(nd.mul, rnn_state)
            core_state = fw_state, rnn_state
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class FastRNNModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(FastRNNModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = FastRNNModelLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        fw_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        rec_fw_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        rnn_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, 1, self.dim_head) for _ in range(self.num_layers))
        return fw_state_tuple, rec_fw_state_tuple, rnn_state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            fw_state, rec_fw_state, rnn_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            fw_state = nest.map(nd.mul, fw_state)
            rec_fw_state = nest.map(nd.mul, rec_fw_state)
            rnn_state = nest.map(nd.mul, rnn_state)
            core_state = fw_state, rec_fw_state, rnn_state
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class DeltaDeltaNetModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=32, dim_ff=512, dropout=0.0, use_xem=False):
        super(DeltaDeltaNetModel, self).__init__()
        self.num_actions = num_actions
        self.use_xem = use_xem
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = DeltaDeltaNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        fw_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, 3 * self.dim_head + 1) for _ in range(self.num_layers))
        very_fw_state_tuple = tuple(torch.zeros(1, batch_size, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        return fw_state_tuple, very_fw_state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            fw_state, very_fw_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            if not self.use_xem:
                fw_state = nest.map(nd.mul, fw_state)
            else:
                layer_id = 0
                for fw_layer in self.core.fwm_layers:
                    fw_layer.cached_fast_weights = fw_state[layer_id][0]
                    layer_id += 1
            very_fw_state = nest.map(nd.mul, very_fw_state)
            core_state = fw_state, very_fw_state
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class DeeperNet(nn.Module):

    def __init__(self, num_actions, hidden_size, num_layers, dim_ff, dropout, use_lstm=False):
        super(DeeperNet, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        core_output_size = self.fc.out_features + 1
        self.trafo_ff_block = DeeperNetLayer(core_output_size, hidden_size, num_layers, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state):
        x = inputs['frame']
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_output = self.trafo_ff_block(core_input)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class SRModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=16, dim_ff=512, dropout=0.0, use_xem=False):
        super(SRModel, self).__init__()
        self.num_actions = num_actions
        self.use_xem = use_xem
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = SRNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):
        Wy_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        Wq_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        Wk_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        wb_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, 4) for _ in range(self.num_layers))
        return Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1, 1, 1)
            Wy_s, Wq_s, Wk_s, wb_s = core_state
            if not self.use_xem:
                Wy_s = nest.map(nd.mul, Wy_s)
                Wq_s = nest.map(nd.mul, Wq_s)
                Wk_s = nest.map(nd.mul, Wk_s)
                wb_s = nest.map(nd.mul, wb_s)
            core_state = Wy_s, Wq_s, Wk_s, wb_s
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class PseudoSRModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=16, dim_ff=512, dropout=0.0, use_xem=False):
        super(PseudoSRModel, self).__init__()
        self.num_actions = num_actions
        self.use_xem = use_xem
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = PseudoSRNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        core_output, _ = self.core(core_input, core_state=None)
        core_output = torch.flatten(core_output, 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class NoCarryOverSRModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=16, dim_ff=512, dropout=0.0):
        super(NoCarryOverSRModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = NoCarryOverSRNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):
        return tuple()

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class SMFWPModel(nn.Module):

    def __init__(self, num_actions, hidden_size=128, num_layers=2, num_head=4, dim_head=16, dim_ff=512, dropout=0.0):
        super(SMFWPModel, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.y_d_head = 3 * dim_head + 1
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(2048, 256)
        self.core = SMFWPNetLayer(self.fc.out_features + 1, hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):
        Wy_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.y_d_head) for _ in range(self.num_layers))
        Wq_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        Wk_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        wb_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, 4) for _ in range(self.num_layers))
        fw_state_tuple = tuple(torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head) for _ in range(self.num_layers))
        return Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple, fw_state_tuple

    def forward(self, inputs, core_state):
        x = inputs['frame']
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1, 1, 1)
            Wy_s, Wq_s, Wk_s, wb_s, fw_s = core_state
            Wy_s = nest.map(nd.mul, Wy_s)
            Wq_s = nest.map(nd.mul, Wq_s)
            Wk_s = nest.map(nd.mul, Wk_s)
            wb_s = nest.map(nd.mul, wb_s)
            fw_s = nest.map(nd.mul, fw_s)
            core_state = Wy_s, Wq_s, Wk_s, wb_s, fw_s
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class SRWMlayer(nn.Module):

    def __init__(self, num_head, dim_head, in_dim, dropout, use_ln=True, use_input_softmax=False, beta_init=-1.0):
        super(SRWMlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.use_ln = use_ln
        self.use_input_softmax = use_input_softmax
        self.sr_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head
        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head), requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4), requires_grad=True)
        if use_ln:
            self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0.0, std=std)
        nn.init.normal_(self.W_q, mean=0.0, std=std)
        nn.init.normal_(self.W_k, mean=0.0, std=std)
        nn.init.normal_(self.w_b, mean=beta_init, std=std)

    def forward(self, h, state=None, get_state=False):
        slen, bsz, _ = h.size()
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        if self.use_input_softmax:
            x = F.softmax(x, dim=-1)
        x = x.permute(1, 2, 0, 3)
        if state is not None:
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
            W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)
        else:
            W_y_bc = self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = self.w_b.repeat(bsz, 1, 1, 1)
        out = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        if self.use_ln:
            out = self.layer_norm(h) + out
        else:
            out = h + out
        if get_state:
            W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)
            state = W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(), w_b_bc.detach()
            return out, state
        return out


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            None


class ConvLSTMModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layer=1, imagenet=False, fc100=False, vision_dropout=0.0, bn_momentum=0.1):
        super(ConvLSTMModel, self).__init__()
        num_conv_blocks = 4
        if imagenet:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2
        else:
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64
        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []
        for i in range(num_conv_blocks):
            conv_block = []
            conv_block.append(nn.Conv2d(in_channels=input_channels, out_channels=out_num_channel, kernel_size=3, stride=1, padding=1))
            conv_block.append(nn.BatchNorm2d(out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel
        self.conv_layers = nn.ModuleList(list_conv_layers)
        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes, hidden_size, num_layers=num_layer)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out, _ = self.rnn(out, state)
        out = self.out_layer(out)
        return out, None


class ConvDeltaModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layers, num_head, dim_head, dim_ff, dropout, vision_dropout=0.0, imagenet=False, fc100=False, bn_momentum=0.1):
        super(ConvDeltaModel, self).__init__()
        num_conv_blocks = 4
        if imagenet:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2
        else:
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64
        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []
        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(nn.Conv2d(in_channels=input_channels, out_channels=out_num_channel, kernel_size=3, stride=1, padding=1))
            conv_block.append(nn.BatchNorm2d(out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel
        self.conv_layers = nn.ModuleList(list_conv_layers)
        self.input_proj = nn.Linear(self.conv_feature_final_size + num_classes, hidden_size)
        layers = []
        for _ in range(num_layers):
            layers.append(FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)
        return out, None


class ConvSRWMModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layers, num_head, dim_head, dim_ff, dropout, vision_dropout=0.0, use_ln=True, use_input_softmax=False, beta_init=0.0, imagenet=False, fc100=False, bn_momentum=0.1):
        super(ConvSRWMModel, self).__init__()
        num_conv_blocks = 4
        if imagenet:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2
        else:
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64
        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []
        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(nn.Conv2d(in_channels=input_channels, out_channels=out_num_channel, kernel_size=3, stride=1, padding=1))
            conv_block.append(nn.BatchNorm2d(out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel
        self.conv_layers = nn.ModuleList(list_conv_layers)
        self.input_proj = nn.Linear(self.conv_feature_final_size + num_classes, hidden_size)
        layers = []
        for _ in range(num_layers):
            layers.append(SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln, use_input_softmax, beta_init))
            layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)
        return out, None


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class Drop2dBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class Drop2dInBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.dropout3 = nn.Dropout2d(dropout)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout3(out)
        return out


class DropInBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample, dropout=0.0):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout3(out)
        return out


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


class ResNet12(nn.Module):

    def __init__(self, channels, dropout, dropout_type='base'):
        super().__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(channels[0], dropout, dropout_type)
        self.layer2 = self._make_layer(channels[1], dropout, dropout_type)
        self.layer3 = self._make_layer(channels[2], dropout, dropout_type)
        self.layer4 = self._make_layer(channels[3], dropout, dropout_type)
        self.out_dim = channels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, dropout, dropout_type='base'):
        downsample = nn.Sequential(conv1x1(self.inplanes, planes), norm_layer(planes))
        if dropout_type == 'base':
            block = Block(self.inplanes, planes, downsample, dropout)
        elif dropout_type == 'inblock':
            block = DropInBlock(self.inplanes, planes, downsample, dropout)
        elif dropout_type == '2d':
            block = Drop2dBlock(self.inplanes, planes, downsample, dropout)
        elif dropout_type == '2d_inblock':
            block = Drop2dInBlock(self.inplanes, planes, downsample, dropout)
        else:
            assert False, f'Unknown dropout_type: {dropout_type}'
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


def resnet12_base(dropout=0.0, use_big=False, dropout_type='base'):
    if use_big:
        return ResNet12([64, 128, 256, 512], dropout, dropout_type)
    else:
        return ResNet12([64, 96, 128, 256], dropout, dropout_type)


class Res12LSTMModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layers, dropout, vision_dropout=0.0, use_big=False, input_dropout=0.0, dropout_type='base'):
        super(Res12LSTMModel, self).__init__()
        self.stem_resnet12 = resnet12_base(vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256
        self.input_drop = nn.Dropout(input_dropout)
        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes, hidden_size, num_layers=num_layers, dropout=dropout)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)
        x = self.stem_resnet12(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out, _ = self.rnn(out, state)
        out = self.out_layer(out)
        return out, None


class Res12DeltaModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layers, num_head, dim_head, dim_ff, dropout, vision_dropout=0.0, use_big=False, input_dropout=0.0, dropout_type='base'):
        super(Res12DeltaModel, self).__init__()
        self.stem_resnet12 = resnet12_base(vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256
        self.input_drop = nn.Dropout(input_dropout)
        self.input_proj = nn.Linear(self.conv_feature_final_size + num_classes, hidden_size)
        layers = []
        for _ in range(num_layers):
            layers.append(FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)
        x = self.stem_resnet12(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)
        return out, None


class Res12SRWMModel(BaseModel):

    def __init__(self, hidden_size, num_classes, num_layers, num_head, dim_head, dim_ff, dropout, vision_dropout=0.0, use_big=False, use_ln=True, use_input_softmax=False, input_dropout=0.0, dropout_type='base', beta_init=0.0):
        super(Res12SRWMModel, self).__init__()
        self.stem_resnet12 = resnet12_base(vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256
        self.input_drop = nn.Dropout(input_dropout)
        self.input_proj = nn.Linear(self.conv_feature_final_size + num_classes, hidden_size)
        layers = []
        for _ in range(num_layers):
            layers.append(SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln, use_input_softmax, beta_init))
            layers.append(TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)
        x = self.stem_resnet12(x)
        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)
        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)
        return out, None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'downsample': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeeperNetLayer,
     lambda: ([], {'input_dim': 4, 'hidden_size': 4, 'num_layers': 1, 'dim_ff': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Drop2dBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'downsample': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Drop2dInBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'downsample': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropInBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'downsample': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet12,
     lambda: ([], {'channels': [4, 4, 4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (TransformerFFlayers,
     lambda: ([], {'ff_dim': 4, 'res_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_IDSIA_modern_srwm(_paritybench_base):
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

