import sys
_module = sys.modules[__name__]
del sys
RNNBackend = _module
RNN = _module
cells = _module
models = _module
apex = _module
_autocast_utils = _module
amp = _module
__version__ = _module
_amp_state = _module
_initialize = _module
_process_optimizer = _module
amp = _module
compat = _module
frontend = _module
handle = _module
lists = _module
functional_overrides = _module
tensor_overrides = _module
torch_overrides = _module
opt = _module
rnn_compat = _module
scaler = _module
utils = _module
wrap = _module
contrib = _module
bottleneck = _module
bottleneck = _module
halo_exchangers = _module
test = _module
clip_grad = _module
clip_grad = _module
conv_bias_relu = _module
conv_bias_relu = _module
cudnn_gbn = _module
batch_norm = _module
func_test_multihead_attn = _module
perf_test_multihead_attn = _module
fmha = _module
fmha = _module
focal_loss = _module
focal_loss = _module
groupbn = _module
batch_norm = _module
index_mul_2d = _module
index_mul_2d = _module
layer_norm = _module
layer_norm = _module
multihead_attn = _module
encdec_multihead_attn = _module
encdec_multihead_attn_func = _module
fast_encdec_multihead_attn_func = _module
fast_encdec_multihead_attn_norm_add_func = _module
fast_self_multihead_attn_func = _module
fast_self_multihead_attn_norm_add_func = _module
mask_softmax_dropout_func = _module
self_multihead_attn = _module
self_multihead_attn_func = _module
optimizers = _module
distributed_fused_adam = _module
distributed_fused_lamb = _module
fp16_optimizer = _module
fused_adam = _module
fused_lamb = _module
fused_sgd = _module
peer_memory = _module
peer_halo_exchanger_1d = _module
peer_memory = _module
sparsity = _module
asp = _module
permutation_lib = _module
permutation_search_kernels = _module
call_permutation_search_kernels = _module
channel_swap = _module
exhaustive_search = _module
permutation_utilities = _module
permutation_test = _module
sparse_masklib = _module
checkpointing_test_part1 = _module
checkpointing_test_part2 = _module
checkpointing_test_reference = _module
toy_problem = _module
test_bottleneck_module = _module
test_clip_grad = _module
test_conv_bias_relu = _module
test_cudnn_gbn_with_two_gpus = _module
test_fmha = _module
test_focal_loss = _module
test_fused_dense = _module
test_index_mul_2d = _module
test_fast_layer_norm = _module
test_encdec_multihead_attn = _module
test_encdec_multihead_attn_norm_add = _module
test_fast_self_multihead_attn_bias = _module
test_mha_fused_softmax = _module
test_self_multihead_attn = _module
test_self_multihead_attn_norm_add = _module
test_dist_adam = _module
test_distributed_fused_lamb = _module
test_peer_halo_exchange_module = _module
transducer = _module
test_transducer_joint = _module
test_transducer_loss = _module
xentropy = _module
test_label_smoothing = _module
_transducer_ref = _module
transducer = _module
softmax_xentropy = _module
fp16_utils = _module
fp16_optimizer = _module
fp16util = _module
loss_scaler = _module
fused_dense = _module
fused_dense = _module
mlp = _module
mlp = _module
multi_tensor_apply = _module
multi_tensor_apply = _module
normalization = _module
fused_layer_norm = _module
fused_adagrad = _module
fused_adam = _module
fused_lamb = _module
fused_mixed_precision_lamb = _module
fused_novograd = _module
fused_sgd = _module
LARC = _module
parallel = _module
distributed = _module
multiproc = _module
optimized_sync_batchnorm = _module
optimized_sync_batchnorm_kernel = _module
sync_batchnorm = _module
sync_batchnorm_kernel = _module
transformer = _module
_data = _module
_batchsampler = _module
_ucc_util = _module
grad_scaler = _module
enums = _module
functional = _module
fused_softmax = _module
layers = _module
layer_norm = _module
log_util = _module
microbatches = _module
parallel_state = _module
pipeline_parallel = _module
_timers = _module
p2p_communication = _module
schedules = _module
common = _module
fwd_bwd_no_pipelining = _module
fwd_bwd_pipelining_with_interleaving = _module
fwd_bwd_pipelining_without_interleaving = _module
utils = _module
tensor_parallel = _module
cross_entropy = _module
data = _module
layers = _module
mappings = _module
memory = _module
random = _module
utils = _module
testing = _module
arguments = _module
commons = _module
distributed_test_base = _module
global_vars = _module
standalone_bert = _module
standalone_gpt = _module
standalone_transformer_lm = _module
utils = _module
conf = _module
main_amp = _module
main_amp = _module
distributed_data_parallel = _module
setup = _module
run_amp = _module
test_add_param_group = _module
test_basic_casts = _module
test_cache = _module
test_checkpointing = _module
test_fused_sgd = _module
test_larc = _module
test_multi_tensor_axpby = _module
test_multi_tensor_l2norm = _module
test_multi_tensor_scale = _module
test_multiple_models_optimizers_losses = _module
test_promotion = _module
test_rnn = _module
utils = _module
test_deprecated_warning = _module
run_fp16util = _module
test_fp16util = _module
test_fused_layer_norm = _module
test_mlp = _module
run_optimizers = _module
test_fused_novograd = _module
test_fused_optimizer = _module
test_lamb = _module
run_test = _module
run_transformer = _module
gpt_scaling_test = _module
test_batch_sampler = _module
test_bert_minimal = _module
test_cross_entropy = _module
test_data = _module
test_dynamic_batchsize = _module
test_fused_softmax = _module
test_gpt_minimal = _module
test_layers = _module
test_mapping = _module
test_microbatches = _module
test_p2p_comm = _module
test_parallel_state = _module
test_pipeline_parallel_fwd_bwd = _module
test_random = _module
test_transformer_utils = _module
compare = _module
main_amp = _module
pipeline_parallel_fwd_bwd_ucc_async = _module
ddp_race_condition_test = _module
amp_master_params = _module
compare = _module
python_single_gpu_unit_test = _module
single_gpu_unit_test = _module
test_batchnorm1d = _module
test_groups = _module
two_gpu_test_different_batch_size = _module
two_gpu_unit_test = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import math


import logging


import warnings


from typing import Optional


from typing import Sequence


from torch._six import string_classes


import functools


import numpy as np


from types import MethodType


import types


import itertools


from collections import OrderedDict


import torch.nn.functional


from itertools import product


import functools as func


import torch.distributed as dist


from torch import nn


from torch._six import inf


from typing import Union


from typing import Iterable


from torch.autograd import gradcheck


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn import functional as F


from torch import Tensor


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


from torch.nn import init


from torch.nn import Parameter


import collections


import enum


import inspect


from torch.distributed.distributed_c10d import _get_default_group


import torch.distributed.distributed_c10d as c10d


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import string


import time


from itertools import permutations


from torch.testing._internal import common_utils


import random


import copy


import typing


from torch.cuda.amp import GradScaler


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from copy import copy


import numbers


from copy import deepcopy


from itertools import chain


from collections import defaultdict


from collections import abc as container_abcs


from torch.nn.modules import Module


from torch.autograd.function import Function


import abc


from torch import distributed as dist


from typing import Tuple


from functools import reduce


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from torch.autograd.variable import Variable


from torch.nn.parallel import DistributedDataParallel


import torch.nn.init as init


from torch import _C


from torch.cuda import _lazy_call


from torch.cuda import device as device_ctx_manager


from torch.utils.checkpoint import detach_variable


import numpy


from torch.utils import collect_env


from torch.testing._internal import common_distributed


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


import torch.optim


import torch.utils.data.distributed


import torchvision.datasets as datasets


import torchvision.models as models


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import load


import functools as ft


import itertools as it


from math import floor


from time import time


from torch.optim import Optimizer


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.testing


from functools import partial


import re


from torch.nn import Module


from torch.nn.parallel import DistributedDataParallel as DDP


class RNNCell(nn.Module):
    """ 
    RNNCell 
    gate_multiplier is related to the architecture you're working with
    For LSTM-like it will be 4 and GRU-like will be 3.
    Always assumes input is NOT batch_first.
    Output size that's not hidden size will use output projection
    Hidden_states is number of hidden states that are needed for cell
    if one will go directly to cell as tensor, if more will go as list
    """

    def __init__(self, gate_multiplier, input_size, hidden_size, cell, n_hidden_states=2, bias=False, output_size=None):
        super(RNNCell, self).__init__()
        self.gate_multiplier = gate_multiplier
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.bias = bias
        self.output_size = output_size
        if output_size is None:
            self.output_size = hidden_size
        self.gate_size = gate_multiplier * self.hidden_size
        self.n_hidden_states = n_hidden_states
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))
        if self.output_size != self.hidden_size:
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.b_ih = self.b_hh = None
        if self.bias:
            self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.hidden = [None for states in range(self.n_hidden_states)]
        self.reset_parameters()

    def new_like(self, new_input_size=None):
        """
        new_like()
        """
        if new_input_size is None:
            new_input_size = self.input_size
        return type(self)(self.gate_multiplier, new_input_size, self.hidden_size, self.cell, self.n_hidden_states, self.bias, self.output_size)

    def reset_parameters(self, gain=1):
        """
        reset_parameters()
        """
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdev, stdev)
    """
    Xavier reset:
    def reset_parameters(self, gain=1):
        stdv = 1.0 / math.sqrt(self.gate_size)

        for param in self.parameters():
            if (param.dim() > 1):
                torch.nn.init.xavier_normal(param, gain)
            else:
                param.data.uniform_(-stdv, stdv)
    """

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for param in self.parameters():
            if param is not None:
                a_param = param
                break
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz:
                if i == 0:
                    hidden_size = self.output_size
                else:
                    hidden_size = self.hidden_size
                tens = a_param.data.new(bsz, hidden_size).zero_()
                self.hidden[i] = Variable(tens, requires_grad=False)

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = None
        self.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None:
                raise RuntimeError('Must initialize hidden state before you can detach it')
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()

    def forward(self, input):
        """
        forward()
        if not inited or bsz has changed this will create hidden states
        """
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        if self.n_hidden_states > 1:
            self.hidden = list(self.hidden)
        else:
            self.hidden = [self.hidden]
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)


def is_iterable(maybe_iterable):
    return isinstance(maybe_iterable, list) or isinstance(maybe_iterable, tuple)


def flatten_list(tens_list):
    """
    flatten_list
    """
    if not is_iterable(tens_list):
        return tens_list
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].size())


class stackedRNN(nn.Module):
    """
    stackedRNN
    """

    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(stackedRNN, self).__init__()
        self.dropout = dropout
        if isinstance(inputRNN, RNNCell):
            self.rnns = [inputRNN]
            for i in range(num_layers - 1):
                self.rnns.append(inputRNN.new_like(inputRNN.output_size))
        elif isinstance(inputRNN, list):
            assert len(inputRNN) == num_layers, 'RNN list length must be equal to num_layers'
            self.rnns = inputRNN
        else:
            raise RuntimeError()
        self.nLayers = len(self.rnns)
        self.rnns = nn.ModuleList(self.rnns)
    """
    Returns output as hidden_state[0] Tensor([sequence steps][batch size][features])
    If collect hidden will also return Tuple(
        [n_hidden_states][sequence steps] Tensor([layer][batch size][features])
    )
    If not collect hidden will also return Tuple(
        [n_hidden_states] Tensor([layer][batch size][features])
    """

    def forward(self, input, collect_hidden=False, reverse=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        inp_iter = reversed(range(seq_len)) if reverse else range(seq_len)
        hidden_states = [[] for i in range(self.nLayers)]
        outputs = []
        for seq in inp_iter:
            for layer in range(self.nLayers):
                if layer == 0:
                    prev_out = input[seq]
                outs = self.rnns[layer](prev_out)
                if collect_hidden:
                    hidden_states[layer].append(outs)
                elif seq == seq_len - 1:
                    hidden_states[layer].append(outs)
                prev_out = outs[0]
            outputs.append(prev_out)
        if reverse:
            outputs = list(reversed(outputs))
        """
        At this point outputs is in format:
        list( [seq_length] x Tensor([bsz][features]) )
        need to convert it to:
        list( Tensor([seq_length][bsz][features]) )
        """
        output = flatten_list(outputs)
        """
        hidden_states at this point is in format:
        list( [layer][seq_length][hidden_states] x Tensor([bsz][features]) )
        need to convert it to:
          For not collect hidden:
            list( [hidden_states] x Tensor([layer][bsz][features]) )
          For collect hidden:
            list( [hidden_states][seq_length] x Tensor([layer][bsz][features]) )
        """
        if not collect_hidden:
            seq_len = 1
        n_hid = self.rnns[0].n_hidden_states
        new_hidden = [[[None for k in range(self.nLayers)] for j in range(seq_len)] for i in range(n_hid)]
        for i in range(n_hid):
            for j in range(seq_len):
                for k in range(self.nLayers):
                    new_hidden[i][j][k] = hidden_states[k][j][i]
        hidden_states = new_hidden
        if reverse:
            hidden_states = list(list(reversed(list(entry))) for entry in hidden_states)
        hiddens = list(list(flatten_list(seq) for seq in hidden) for hidden in hidden_states)
        if not collect_hidden:
            hidden_states = list(entry[0] for entry in hidden_states)
        return output, hidden_states

    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detach_hidden()

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        """ 
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)


class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN
    """

    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout=dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers, dropout=dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])

    def forward(self, input, collect_hidden=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        fwd_out, fwd_hiddens = list(self.fwd(input, collect_hidden=collect_hidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True, collect_hidden=collect_hidden))
        output = torch.cat([fwd_out, bckwrd_out], -1)
        hiddens = tuple(torch.cat(hidden, -1) for hidden in zip(fwd_hiddens, bckwrd_hiddens))
        return output, hiddens

    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detachHidden()

    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        """
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)


def mLSTMCell(input, hidden, w_ih, w_hh, w_mih, w_mhh, b_ih=None, b_hh=None):
    """
    mLSTMCell
    """
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
        hgates = F.linear(m, w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1], b_ih, b_hh)
    hx, cx = hidden
    m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * F.tanh(cy)
    return hy, cy


class mLSTMRNNCell(RNNCell):
    """
    mLSTMRNNCell
    """

    def __init__(self, input_size, hidden_size, bias=False, output_size=None):
        gate_multiplier = 4
        super(mLSTMRNNCell, self).__init__(gate_multiplier, input_size, hidden_size, mLSTMCell, n_hidden_states=2, bias=bias, output_size=output_size)
        self.w_mih = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        self.w_mhh = nn.Parameter(torch.Tensor(self.output_size, self.output_size))
        self.reset_parameters()

    def forward(self, input):
        """
        mLSTMRNNCell.forward()
        """
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = list(self.cell(input, hidden_state, self.w_ih, self.w_hh, self.w_mih, self.w_mhh, b_ih=self.b_ih, b_hh=self.b_hh))
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)

    def new_like(self, new_input_size=None):
        if new_input_size is None:
            new_input_size = self.input_size
        return type(self)(new_input_size, self.hidden_size, self.bias, self.output_size)


class FrozenBatchNorm2d(torch.jit.ScriptModule):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    @torch.jit.script_method
    def get_scale_bias(self, nhwc):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        if nhwc:
            scale = scale.reshape(1, 1, 1, -1)
            bias = bias.reshape(1, 1, 1, -1)
        else:
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
        return scale, bias

    @torch.jit.script_method
    def forward(self, x):
        scale, bias = self.get_scale_bias(False)
        return x * scale + bias


@torch.jit.script
def drelu_dscale1(grad_o, output, scale1):
    relu_mask = output > 0
    dx_relu = relu_mask * grad_o
    g1 = dx_relu * scale1
    return g1, dx_relu


@torch.jit.script
def drelu_dscale2(grad_o, output, scale1, scale2):
    relu_mask = output > 0
    dx_relu = relu_mask * grad_o
    g1 = dx_relu * scale1
    g2 = dx_relu * scale2
    return g1, g2


class BottleneckFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, nhwc, stride_1x1, scale, bias, x, *conv):
        args = [x, *conv[0:3], *scale[0:3], *bias[0:3]]
        ctx.downsample = len(conv) > 3
        if ctx.downsample:
            args.append(conv[3])
            args.append(scale[3])
            args.append(bias[3])
        outputs = fast_bottleneck.forward(nhwc, stride_1x1, args)
        ctx.save_for_backward(*(args + outputs))
        ctx.nhwc = nhwc
        ctx.stride_1x1 = stride_1x1
        return outputs[2]

    @staticmethod
    def backward(ctx, grad_o):
        outputs = ctx.saved_tensors[-3:]
        if ctx.downsample:
            grad_conv3, grad_conv4 = drelu_dscale2(grad_o, outputs[2], ctx.saved_tensors[6], ctx.saved_tensors[11])
        else:
            grad_conv3, grad_conv4 = drelu_dscale1(grad_o, outputs[2], ctx.saved_tensors[6])
        t_list = [*ctx.saved_tensors[0:10]]
        t_list.append(grad_conv3)
        t_list.append(grad_conv4)
        t_list.append(outputs[0])
        t_list.append(outputs[1])
        if ctx.downsample:
            t_list.append(ctx.saved_tensors[10])
        grads = fast_bottleneck.backward(ctx.nhwc, ctx.stride_1x1, t_list)
        return None, None, None, None, *grads


bottleneck_function = BottleneckFunction.apply


def compute_scale_bias_one(nhwc, weight, bias, running_mean, running_var, w_scale, w_bias):
    scale = weight * running_var.rsqrt()
    bias = bias - running_mean * scale
    w_scale.copy_(scale)
    w_bias.copy_(bias)


def compute_scale_bias_method(nhwc, args):
    for arg in args:
        compute_scale_bias_one(nhwc, *arg)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    weight_tensor_nchw = tensor
    nn.init.kaiming_uniform_(weight_tensor_nchw, a=a, mode=mode, nonlinearity=nonlinearity)


class Bottleneck(torch.nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, groups=1, dilation=1, norm_func=None, use_cudnn=False, explicit_nhwc=False):
        super(Bottleneck, self).__init__()
        if groups != 1:
            raise RuntimeError('Only support groups == 1')
        if dilation != 1:
            raise RuntimeError('Only support dilation == 1')
        if norm_func == None:
            norm_func = FrozenBatchNorm2d
        else:
            raise RuntimeError('Only support frozen BN now.')
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride), norm_func(out_channels))
        else:
            self.downsample = None
        self.conv1 = conv1x1(in_channels, bottleneck_channels, stride)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.bn1 = norm_func(bottleneck_channels)
        self.bn2 = norm_func(bottleneck_channels)
        self.bn3 = norm_func(out_channels)
        self.w_scale = None
        self.use_cudnn = use_cudnn
        self.w_conv = [self.conv1.weight, self.conv2.weight, self.conv3.weight]
        if self.downsample is not None:
            self.w_conv.append(self.downsample[0].weight)
        for w in self.w_conv:
            kaiming_uniform_(w, a=1)
        self.explicit_nhwc = explicit_nhwc
        if self.explicit_nhwc:
            for p in self.parameters():
                with torch.no_grad():
                    p.data = p.data.permute(0, 2, 3, 1).contiguous()
        return

    def get_scale_bias_callable(self):
        self.w_scale, self.w_bias, args = [], [], []
        batch_norms = [self.bn1, self.bn2, self.bn3]
        if self.downsample is not None:
            batch_norms.append(self.downsample[1])
        for bn in batch_norms:
            s = torch.empty_like(bn.weight)
            b = torch.empty_like(s)
            args.append((bn.weight, bn.bias, bn.running_mean, bn.running_var, s, b))
            if self.explicit_nhwc:
                self.w_scale.append(s.reshape(1, 1, 1, -1))
                self.w_bias.append(b.reshape(1, 1, 1, -1))
            else:
                self.w_scale.append(s.reshape(1, -1, 1, 1))
                self.w_bias.append(b.reshape(1, -1, 1, 1))
        return func.partial(compute_scale_bias_method, self.explicit_nhwc, args)

    def forward(self, x):
        if self.use_cudnn:
            if self.w_scale is None:
                s1, b1 = self.bn1.get_scale_bias(self.explicit_nhwc)
                s2, b2 = self.bn2.get_scale_bias(self.explicit_nhwc)
                s3, b3 = self.bn3.get_scale_bias(self.explicit_nhwc)
                w_scale = [s1, s2, s3]
                w_bias = [b1, b2, b3]
                if self.downsample is not None:
                    s4, b4 = self.downsample[1].get_scale_bias(self.explicit_nhwc)
                    w_scale.append(s4)
                    w_bias.append(b4)
                out = bottleneck_function(self.explicit_nhwc, self.stride, w_scale, w_bias, x, *self.w_conv)
            else:
                out = bottleneck_function(self.explicit_nhwc, self.stride, self.w_scale, self.w_bias, x, *self.w_conv)
            return out
        if self.explicit_nhwc:
            raise RuntimeError('explicit nhwc with native ops is not supported.')
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SpatialBottleneckFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spatial_group_size, spatial_group_rank, spatial_communicator, spatial_halo_exchanger, spatial_method, use_delay_kernel, explicit_nhwc, stride_1x1, scale, bias, thresholdTop, thresholdBottom, x, *conv):
        if spatial_group_size > 1:
            stream1 = spatial_halo_exchanger.stream1
            stream2 = spatial_halo_exchanger.stream2
            stream3 = spatial_halo_exchanger.stream3
        args = [x, *conv[0:3], *scale[0:3], *bias[0:3]]
        ctx.downsample = len(conv) > 3
        if ctx.downsample:
            args.append(conv[3])
            args.append(scale[3])
            args.append(bias[3])
        outputs = fast_bottleneck.forward_init(explicit_nhwc, stride_1x1, args)
        fast_bottleneck.forward_out1(explicit_nhwc, stride_1x1, args, outputs)
        if spatial_group_size > 1:
            out1 = outputs[0]
            if explicit_nhwc:
                N, Hs, W, C = list(out1.shape)
                memory_format = torch.contiguous_format
                out1_pad = torch.empty([N, Hs + 2, W, C], dtype=out1.dtype, device='cuda')
            else:
                N, C, Hs, W = list(out1.shape)
                memory_format = torch.channels_last if out1.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
                out1_pad = torch.empty([N, C, Hs + 2, W], dtype=out1.dtype, device='cuda', memory_format=memory_format)
            stream1.wait_stream(torch.cuda.current_stream())
            if spatial_method != 2:
                stream3.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream1):
                if explicit_nhwc:
                    top_out1_halo = out1_pad[:, :1, :, :]
                    btm_out1_halo = out1_pad[:, Hs + 1:Hs + 2, :, :]
                    spatial_halo_exchanger.left_right_halo_exchange(out1[:, :1, :, :], out1[:, Hs - 1:, :, :], top_out1_halo, btm_out1_halo)
                else:
                    top_out1_halo = out1_pad[:, :, :1, :]
                    btm_out1_halo = out1_pad[:, :, Hs + 1:Hs + 2, :]
                    spatial_halo_exchanger.left_right_halo_exchange(out1[:, :, :1, :], out1[:, :, Hs - 1:, :], top_out1_halo, btm_out1_halo)
            if spatial_method == 1:
                if spatial_group_rank < spatial_group_size - 1:
                    stream2.wait_stream(stream1)
                    with torch.cuda.stream(stream2):
                        if explicit_nhwc:
                            btm_fat_halo = torch.empty((N, 3, W, C), dtype=out1.dtype, device=out1.device)
                            btm_fat_halo[:, 0:2, :, :].copy_(out1[:, Hs - 2:, :, :])
                            btm_fat_halo[:, 2:, :, :].copy_(btm_out1_halo)
                        else:
                            btm_fat_halo = torch.empty((N, C, 3, W), dtype=out1.dtype, device=out1.device)
                            btm_fat_halo[:, :, 0:2, :].copy_(out1[:, :, Hs - 2:, :])
                            btm_fat_halo[:, :, 2:, :].copy_(btm_out1_halo)
                        btm_out2 = fast_bottleneck.forward_out2_halo(explicit_nhwc, btm_fat_halo, args)
                if spatial_group_rank > 0:
                    with torch.cuda.stream(stream1):
                        if explicit_nhwc:
                            top_fat_halo = torch.empty((N, 3, W, C), dtype=out1.dtype, device=out1.device)
                            top_fat_halo[:, :1, :, :].copy_(top_out1_halo)
                            top_fat_halo[:, 1:3, :, :].copy_(out1[:, :2, :, :])
                        else:
                            top_fat_halo = torch.empty((N, C, 3, W), dtype=out1.dtype, device=out1.device)
                            top_fat_halo[:, :, :1, :].copy_(top_out1_halo)
                            top_fat_halo[:, :, 1:3, :].copy_(out1[:, :, :2, :])
                        top_out2 = fast_bottleneck.forward_out2_halo(explicit_nhwc, top_fat_halo, args)
                if use_delay_kernel:
                    inc.add_delay(10)
            elif spatial_method != 2 and spatial_method != 3:
                assert False, 'spatial_method must be 1, 2 or 3'
        if spatial_group_size <= 1:
            fast_bottleneck.forward_out2(explicit_nhwc, stride_1x1, args, outputs)
        elif spatial_method == 1:
            fast_bottleneck.forward_out2(explicit_nhwc, stride_1x1, args, outputs)
            with torch.cuda.stream(stream3):
                if explicit_nhwc:
                    out1_pad[:, 1:Hs + 1, :, :].copy_(out1)
                else:
                    out1_pad[:, :, 1:Hs + 1, :].copy_(out1)
        elif spatial_method == 2:
            if explicit_nhwc:
                out1_pad[:, 1:Hs + 1, :, :].copy_(out1)
            else:
                out1_pad[:, :, 1:Hs + 1, :].copy_(out1)
            torch.cuda.current_stream().wait_stream(stream1)
            fast_bottleneck.forward_out2_pad(explicit_nhwc, stride_1x1, args, outputs, out1_pad)
        elif spatial_method == 3:
            fast_bottleneck.forward_out2_mask(explicit_nhwc, stride_1x1, args, outputs, thresholdTop, thresholdBottom)
            with torch.cuda.stream(stream3):
                if explicit_nhwc:
                    out1_pad[:, 1:Hs + 1, :, :].copy_(out1)
                else:
                    out1_pad[:, :, 1:Hs + 1, :].copy_(out1)
        if spatial_group_size > 1:
            out2 = outputs[1]
            if explicit_nhwc:
                top_out2_halo = out2[:, :1, :, :]
                btm_out2_halo = out2[:, Hs - 1:, :, :]
            else:
                top_out2_halo = out2[:, :, :1, :]
                btm_out2_halo = out2[:, :, Hs - 1:, :]
            if spatial_method == 1:
                if spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(stream1)
                    top_out2_halo.copy_(top_out2)
                if spatial_group_rank < spatial_group_size - 1:
                    torch.cuda.current_stream().wait_stream(stream2)
                    btm_out2_halo.copy_(btm_out2)
            elif spatial_method == 3:
                if spatial_group_rank < spatial_group_size - 1:
                    stream2.wait_stream(stream1)
                    stream2.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream2):
                        w1by3 = args[2][:, 2:3, :, :].clone()
                        btm_out1_halo = btm_out1_halo.clone()
                        btm_out2 = fast_bottleneck.forward_out2_halo_corr(explicit_nhwc, btm_out1_halo, args, w1by3, btm_out2_halo.clone())
                        btm_out2_halo.copy_(btm_out2)
                if spatial_group_rank > 0:
                    stream1.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream1):
                        w1by3 = args[2][:, :1, :, :].clone()
                        top_out1_halo = top_out1_halo.clone()
                        top_out2 = fast_bottleneck.forward_out2_halo_corr(explicit_nhwc, top_out1_halo, args, w1by3, top_out2_halo.clone())
                        top_out2_halo.copy_(top_out2)
                if spatial_group_rank < spatial_group_size - 1:
                    torch.cuda.current_stream().wait_stream(stream2)
                if spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(stream1)
        fast_bottleneck.forward_rest(explicit_nhwc, stride_1x1, args, outputs)
        if spatial_group_size > 1:
            if spatial_method != 2:
                torch.cuda.current_stream().wait_stream(stream3)
            ctx.save_for_backward(*(args + outputs + [out1_pad]))
        else:
            ctx.save_for_backward(*(args + outputs))
        ctx.explicit_nhwc = explicit_nhwc
        ctx.stride_1x1 = stride_1x1
        ctx.spatial_group_size = spatial_group_size
        if spatial_group_size > 1:
            ctx.spatial_group_rank = spatial_group_rank
            ctx.spatial_halo_exchanger = spatial_halo_exchanger
            ctx.spatial_method = spatial_method
            ctx.use_delay_kernel = use_delay_kernel
            ctx.thresholdTop = thresholdTop
            ctx.thresholdBottom = thresholdBottom
            ctx.stream1 = stream1
            ctx.stream2 = stream2
            ctx.stream3 = stream3
        return outputs[2]

    @staticmethod
    def backward(ctx, grad_o):
        if ctx.spatial_group_size > 1:
            out1_pad = ctx.saved_tensors[-1]
            outputs = ctx.saved_tensors[-4:-1]
        else:
            outputs = ctx.saved_tensors[-3:]
        if ctx.downsample:
            grad_conv3, grad_conv4 = drelu_dscale2(grad_o, outputs[2], ctx.saved_tensors[6], ctx.saved_tensors[11])
        else:
            grad_conv3, grad_conv4 = drelu_dscale1(grad_o, outputs[2], ctx.saved_tensors[6])
        t_list = [*ctx.saved_tensors[0:10]]
        t_list.append(grad_conv3)
        t_list.append(grad_conv4)
        t_list.append(outputs[0])
        t_list.append(outputs[1])
        if ctx.downsample:
            t_list.append(ctx.saved_tensors[10])
        grads = fast_bottleneck.backward_init(ctx.explicit_nhwc, ctx.stride_1x1, t_list)
        wgrad3_stream = torch.Stream()
        wgrad3_stream.wait_stream(torch.cuda.current_stream())
        grad_out2 = fast_bottleneck.backward_grad_out2(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads)
        wgrad2_stream = torch.Stream()
        wgrad2_stream.wait_stream(torch.cuda.current_stream())
        if ctx.spatial_group_size > 1:
            if ctx.explicit_nhwc:
                N, Hs, W, C = list(grad_out2.shape)
            else:
                N, C, Hs, W = list(grad_out2.shape)
            relu1 = t_list[12]
            ctx.stream1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ctx.stream1):
                top_halo, btm_halo = ctx.spatial_halo_exchanger.left_right_halo_exchange(grad_out2[:, :1, :, :], grad_out2[:, Hs - 1:, :, :])
            if ctx.spatial_method == 1 or ctx.spatial_method == 2:
                if ctx.spatial_group_rank < ctx.spatial_group_size - 1:
                    ctx.stream2.wait_stream(ctx.stream1)
                    with torch.cuda.stream(ctx.stream2):
                        if ctx.explicit_nhwc:
                            btm_fat_halo = torch.empty((N, 3, W, C), dtype=grad_out2.dtype, device=grad_out2.device)
                            btm_fat_halo[:, :2, :, :].copy_(grad_out2[:, Hs - 2:, :, :])
                            btm_fat_halo[:, 2:, :, :].copy_(btm_halo)
                            btm_fat_relu_halo = torch.empty((N, 3, W, C), dtype=grad_out2.dtype, device=grad_out2.device)
                            btm_fat_relu_halo[:, :2, :, :].copy_(relu1[:, Hs - 2:, :, :])
                            btm_fat_relu_halo[:, 2:, :, :].zero_()
                        else:
                            btm_fat_halo = torch.empty((N, C, 3, W), dtype=grad_out2.dtype, device=grad_out2.device)
                            btm_fat_halo[:, :, :2, :].copy_(grad_out2[:, :, Hs - 2:, :])
                            btm_fat_halo[:, :, 2:, :].copy_(btm_halo)
                            btm_fat_relu_halo = torch.empty((N, C, 3, W), dtype=grad_out2.dtype, device=grad_out2.device)
                            btm_fat_relu_halo[:, :, :2, :].copy_(relu1[:, :, Hs - 2:, :])
                            btm_fat_relu_halo[:, :, 2:, :].zero_()
                        btm_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, btm_fat_halo, btm_fat_relu_halo)
                        if ctx.explicit_nhwc:
                            btm_grad_out1_halo = btm_grad_out1_halo[:, 1:2, :, :]
                        else:
                            btm_grad_out1_halo = btm_grad_out1_halo[:, :, 1:2, :]
                if ctx.spatial_group_rank > 0:
                    with torch.cuda.stream(ctx.stream1):
                        if ctx.explicit_nhwc:
                            top_fat_halo = torch.empty((N, 3, W, C), dtype=grad_out2.dtype, device=grad_out2.device)
                            top_fat_halo[:, :1, :, :].copy_(top_halo)
                            top_fat_halo[:, 1:, :, :].copy_(grad_out2[:, :2, :, :])
                            top_fat_relu_halo = torch.empty((N, 3, W, C), dtype=grad_out2.dtype, device=grad_out2.device)
                            top_fat_relu_halo[:, :1, :, :].zero_()
                            top_fat_relu_halo[:, 1:, :, :].copy_(relu1[:, :2, :, :])
                        else:
                            top_fat_halo = torch.empty((N, C, 3, W), dtype=grad_out2.dtype, device=grad_out2.device)
                            top_fat_halo[:, :, :1, :].copy_(top_halo)
                            top_fat_halo[:, :, 1:, :].copy_(grad_out2[:, :, :2, :])
                            top_fat_relu_halo = torch.empty((N, C, 3, W), dtype=grad_out2.dtype, device=grad_out2.device)
                            top_fat_relu_halo[:, :, :1, :].zero_()
                            top_fat_relu_halo[:, :, 1:, :].copy_(relu1[:, :, :2, :])
                        top_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, top_fat_halo, top_fat_relu_halo)
                        if ctx.explicit_nhwc:
                            top_grad_out1_halo = top_grad_out1_halo[:, 1:2, :, :]
                        else:
                            top_grad_out1_halo = top_grad_out1_halo[:, :, 1:2, :]
                if ctx.use_delay_kernel:
                    inc.add_delay(10)
            elif ctx.spatial_method != 3:
                assert False, 'spatial_method must be 1, 2 or 3'
        if ctx.spatial_group_size <= 1 or ctx.spatial_method == 1 or ctx.spatial_method == 2:
            grad_out1 = fast_bottleneck.backward_grad_out1(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2)
        elif ctx.spatial_group_size > 1 and ctx.spatial_method == 3:
            grad_out1 = fast_bottleneck.backward_grad_out1_mask(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2, ctx.thresholdTop, ctx.thresholdBottom)
        if ctx.spatial_group_size > 1:
            w = t_list[2]
            z = t_list[4]
            relu1 = t_list[12]
            if ctx.spatial_method == 1 or ctx.spatial_method == 2:
                if ctx.spatial_group_rank < ctx.spatial_group_size - 1:
                    torch.cuda.current_stream().wait_stream(ctx.stream2)
                    if ctx.explicit_nhwc:
                        grad_out1[:, Hs - 1:, :, :].copy_(btm_grad_out1_halo)
                    else:
                        grad_out1[:, :, Hs - 1:, :].copy_(btm_grad_out1_halo)
                if ctx.spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(ctx.stream1)
                    if ctx.explicit_nhwc:
                        grad_out1[:, :1, :, :].copy_(top_grad_out1_halo)
                    else:
                        grad_out1[:, :, :1, :].copy_(top_grad_out1_halo)
            elif ctx.spatial_method == 3:
                if ctx.spatial_group_rank < ctx.spatial_group_size - 1:
                    if ctx.explicit_nhwc:
                        btm_relu_halo = relu1[:, Hs - 1:, :, :].clone()
                        btm_grad_out1 = grad_out1[:, Hs - 1:, :, :]
                    else:
                        btm_relu_halo = relu1[:, :, Hs - 1:, :].clone()
                        btm_grad_out1 = grad_out1[:, :, Hs - 1:, :]
                    w1by3 = w[:, :1, :, :].clone()
                    ctx.stream2.wait_stream(ctx.stream1)
                    ctx.stream2.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(ctx.stream2):
                        btm_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo_corr(ctx.explicit_nhwc, ctx.stride_1x1, t_list, w1by3, grads, btm_halo, btm_relu_halo, btm_grad_out1.clone())
                        btm_grad_out1.copy_(btm_grad_out1_halo)
                if ctx.spatial_group_rank > 0:
                    if ctx.explicit_nhwc:
                        top_relu_halo = relu1[:, :1, :, :].clone()
                        top_grad_out1 = grad_out1[:, :1, :, :]
                    else:
                        top_relu_halo = relu1[:, :, :1, :].clone()
                        top_grad_out1 = grad_out1[:, :, :1, :]
                    w1by3 = w[:, 2:, :, :].clone()
                    ctx.stream1.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(ctx.stream1):
                        top_grad_out1_halo = fast_bottleneck.backward_grad_out1_halo_corr(ctx.explicit_nhwc, ctx.stride_1x1, t_list, w1by3, grads, top_halo, top_relu_halo, top_grad_out1.clone())
                        top_grad_out1.copy_(top_grad_out1_halo)
                if ctx.spatial_group_rank < ctx.spatial_group_size - 1:
                    torch.cuda.current_stream().wait_stream(ctx.stream2)
                if ctx.spatial_group_rank > 0:
                    torch.cuda.current_stream().wait_stream(ctx.stream1)
        wgrad1_stream = torch.Stream()
        wgrad1_stream.wait_stream(torch.cuda.current_stream())
        fast_bottleneck.backward_rest(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2, grad_out1)
        with torch.cuda.stream(wgrad3_stream):
            fast_bottleneck.backward_wgrad3(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads)
        with torch.cuda.stream(wgrad2_stream):
            if ctx.spatial_group_size > 1:
                fast_bottleneck.backward_wgrad2_pad(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, out1_pad, grad_out2)
            else:
                fast_bottleneck.backward_wgrad2(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out2)
        with torch.cuda.stream(wgrad1_stream):
            fast_bottleneck.backward_wgrad1(ctx.explicit_nhwc, ctx.stride_1x1, t_list, grads, grad_out1)
        torch.cuda.current_stream().wait_stream(wgrad3_stream)
        torch.cuda.current_stream().wait_stream(wgrad2_stream)
        torch.cuda.current_stream().wait_stream(wgrad1_stream)
        return None, None, None, None, None, None, None, None, None, None, None, None, *grads


spatial_bottleneck_function = SpatialBottleneckFunction.apply


class SpatialBottleneck(torch.nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, groups=1, dilation=1, norm_func=None, use_cudnn=False, explicit_nhwc=False, spatial_parallel_args=None):
        super(SpatialBottleneck, self).__init__()
        if groups != 1:
            raise RuntimeError('Only support groups == 1')
        if dilation != 1:
            raise RuntimeError('Only support dilation == 1')
        if norm_func == None:
            norm_func = FrozenBatchNorm2d
        else:
            raise RuntimeError('Only support frozen BN now.')
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride), norm_func(out_channels))
        else:
            self.downsample = None
        self.conv1 = conv1x1(in_channels, bottleneck_channels, stride)
        self.conv2 = conv3x3(bottleneck_channels, bottleneck_channels)
        self.conv3 = conv1x1(bottleneck_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.bn1 = norm_func(bottleneck_channels)
        self.bn2 = norm_func(bottleneck_channels)
        self.bn3 = norm_func(out_channels)
        self.w_scale = None
        self.use_cudnn = use_cudnn
        self.w_conv = [self.conv1.weight, self.conv2.weight, self.conv3.weight]
        if self.downsample is not None:
            self.w_conv.append(self.downsample[0].weight)
        for w in self.w_conv:
            kaiming_uniform_(w, a=1)
        self.thresholdTop, self.thresholdBottom = None, None
        self.explicit_nhwc = explicit_nhwc
        if self.explicit_nhwc:
            for p in self.parameters():
                with torch.no_grad():
                    p.data = p.data.permute(0, 2, 3, 1).contiguous()
        if spatial_parallel_args is None:
            self.spatial_parallel_args = 1, 0, None, None, 0, False
        else:
            self.spatial_parallel_args = spatial_parallel_args
        return

    def get_scale_bias_callable(self):
        self.w_scale, self.w_bias, args = [], [], []
        batch_norms = [self.bn1, self.bn2, self.bn3]
        if self.downsample is not None:
            batch_norms.append(self.downsample[1])
        for bn in batch_norms:
            s = torch.empty_like(bn.weight)
            b = torch.empty_like(s)
            args.append((bn.weight, bn.bias, bn.running_mean, bn.running_var, s, b))
            if self.explicit_nhwc:
                self.w_scale.append(s.reshape(1, 1, 1, -1))
                self.w_bias.append(b.reshape(1, 1, 1, -1))
            else:
                self.w_scale.append(s.reshape(1, -1, 1, 1))
                self.w_bias.append(b.reshape(1, -1, 1, 1))
        return func.partial(compute_scale_bias_method, self.explicit_nhwc, args)

    def forward(self, x):
        if self.use_cudnn:
            if self.thresholdTop is None:
                spatial_group_size, spatial_group_rank, _, _, _, _ = self.spatial_parallel_args
                if self.explicit_nhwc:
                    N, H, W, C = list(x.shape)
                else:
                    N, C, H, W = list(x.shape)
                self.thresholdTop = torch.tensor([1 if spatial_group_rank > 0 else 0], dtype=torch.int32, device='cuda')
                self.thresholdBottom = torch.tensor([H - 2 if spatial_group_rank < spatial_group_size - 1 else H - 1], dtype=torch.int32, device='cuda')
            if self.w_scale is None:
                s1, b1 = self.bn1.get_scale_bias(self.explicit_nhwc)
                s2, b2 = self.bn2.get_scale_bias(self.explicit_nhwc)
                s3, b3 = self.bn3.get_scale_bias(self.explicit_nhwc)
                w_scale = [s1, s2, s3]
                w_bias = [b1, b2, b3]
                if self.downsample is not None:
                    s4, b4 = self.downsample[1].get_scale_bias(self.explicit_nhwc)
                    w_scale.append(s4)
                    w_bias.append(b4)
                out = spatial_bottleneck_function(*self.spatial_parallel_args, self.explicit_nhwc, self.stride, w_scale, w_bias, self.thresholdTop, self.thresholdBottom, x, *self.w_conv)
            else:
                out = spatial_bottleneck_function(*self.spatial_parallel_args, self.explicit_nhwc, self.stride, self.w_scale, self.w_bias, self.thresholdTop, self.thresholdBottom, x, *self.w_conv)
            return out
        if self.explicit_nhwc:
            raise RuntimeError('explicit nhwc with native ops is not supported.')
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class _GroupBatchNorm2d(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, running_mean, running_variance, minibatch_mean, minibatch_inv_var, momentum, eps, group_size, group_rank, fwd_buffers, bwd_buffers):
        ctx.save_for_backward(input, weight, minibatch_mean, minibatch_inv_var)
        ctx.eps = eps
        ctx.bn_group = group_size
        ctx.rank_id = group_rank
        ctx.peer_buffers = bwd_buffers
        return cudnn_gbn_lib.forward(input, weight, bias, running_mean, running_variance, minibatch_mean, minibatch_inv_var, momentum, eps, group_size, group_rank, fwd_buffers)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, scale, minibatch_mean, minibatch_inv_var = ctx.saved_variables
        eps = ctx.eps
        bn_group = ctx.bn_group
        rank_id = ctx.rank_id
        peer_buffers = ctx.peer_buffers
        dx, dscale, dbias = cudnn_gbn_lib.backward(x, grad_output, scale, minibatch_mean, minibatch_inv_var, eps, bn_group, rank_id, peer_buffers)
        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None


class GroupBatchNorm2d(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.

    When running in training mode, the layer reduces stats across process groups
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model.

    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Example::

        >>> sbn = apex.contrib.GroupBatchNorm2d(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    def __init__(self, num_features, group_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(GroupBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.group_size = group_size
        rank = torch.distributed.get_rank()
        self.group_id = rank // group_size
        self.group_rank = rank % group_size
        self.fwd_peer_buffers = self.get_peer_buffers(num_features)
        self.bwd_peer_buffers = self.get_peer_buffers(num_features)
        self.minibatch_mean = torch.FloatTensor(num_features)
        self.minibatch_inv_var = torch.FloatTensor(num_features)

    def get_peer_buffers(self, num_features):
        peer_size = self.group_size * 4 * num_features * 4
        raw = pm.allocate_raw(peer_size)
        world_size = torch.distributed.get_world_size()
        raw_ipc = pm.get_raw_ipc_address(raw)
        raw_ipcs = [torch.empty_like(raw_ipc) for _ in range(world_size)]
        torch.distributed.all_gather(raw_ipcs, raw_ipc)
        group_ipcs = [raw_ipcs[x] for x in range(self.group_id * self.group_size, self.group_id * self.group_size + self.group_size)]
        peer_raw_ipcs = torch.stack(group_ipcs).cpu()
        return pm.get_raw_peers(peer_raw_ipcs, self.group_rank, raw)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def _check_input_channels(self, input):
        if input.size(1) % 8 != 0:
            raise ValueError('GroupBatchNorm2d number of input channels should be a multiple of 8')

    def forward(self, input: Tensor) ->Tensor:
        if not input.is_cuda:
            raise ValueError('GroupBatchNorm2d expected input tensor to be on GPU')
        if not input.is_contiguous(memory_format=torch.channels_last):
            raise ValueError('GroupBatchNorm2d expected input tensor to be in channels last memory format')
        if torch.is_autocast_enabled():
            input = input
        if input.dtype != torch.float16:
            raise ValueError('GroupBatchNorm2d expected input tensor in float16')
        self._check_input_dim(input)
        self._check_input_channels(input)
        if not self.training:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)
        return _GroupBatchNorm2d.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.minibatch_mean, self.minibatch_inv_var, self.momentum, self.eps, self.group_size, self.group_rank, self.fwd_peer_buffers, self.bwd_peer_buffers)


class FMHAFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, p_dropout, max_s, is_training, zero_tensors):
        batch_size = cu_seqlens.numel() - 1
        if batch_size < 4:
            max_s = 512
            context, S_dmask = mha.fwd_nl(qkv, cu_seqlens, p_dropout, max_s, is_training, True, zero_tensors, None)
        else:
            context, S_dmask = mha.fwd(qkv, cu_seqlens, p_dropout, max_s, is_training, False, zero_tensors, None)
        ctx.save_for_backward(qkv, S_dmask)
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.zero_tensors = zero_tensors
        return context

    @staticmethod
    def backward(ctx, dout):
        qkv, S_dmask = ctx.saved_tensors
        batch_size = ctx.cu_seqlens.numel() - 1
        if batch_size < 4:
            dqkv, dp, _ = mha.bwd_nl(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.zero_tensors)
        else:
            dqkv, dp = mha.bwd(dout, qkv, S_dmask, ctx.cu_seqlens, ctx.p_dropout, ctx.max_s, ctx.zero_tensors)
        return dqkv, None, None, None, None, None


class FMHA(torch.nn.Module):

    def __init__(self, config):
        super(FMHA, self).__init__()
        self.p_dropout = config.attention_probs_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        assert self.d * self.h == self.hidden_size, 'Invalid hidden size/num_heads'

    def forward(self, qkv, cu_seqlens, max_s, is_training=True, zero_tensors=False):
        ctx = FMHAFun.apply(qkv.view(-1, 3, self.h, self.d), cu_seqlens, self.p_dropout, max_s, is_training, zero_tensors)
        return ctx.view(-1, self.hidden_size)


class bn_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.fuse_relu = fuse_relu
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream
            res = bnp.bn_fwd_nhwc(x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_fwd_eval_nhwc(x, s, b, rm, riv, ret_cta, bn_group, mom, epsilon, fuse_relu)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        fuse_relu = ctx.fuse_relu
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream
        dx, dscale, dbias = bnp.bn_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)
        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class bn_addrelu_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z, s, b, rm, riv, mini_m, mini_riv, grid_dim_y, ret_cta, mom, epsilon, is_train, bn_group, my_data, pair_data, magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup, bwd_grid_x, multi_stream):
        if is_train:
            bitmask = torch.IntTensor((x.numel() + 31) // 32 * 2 * grid_dim_y)
            ctx.save_for_backward(x, s, b, rm, riv, mini_m, mini_riv, bitmask)
            ctx.epsilon = epsilon
            ctx.momentum = mom
            ctx.ret_cta = ret_cta
            ctx.my_data = my_data
            ctx.pair_data = pair_data
            ctx.magic = magic
            ctx.pair_data2 = pair_data2
            ctx.pair_data3 = pair_data3
            ctx.bn_group = bn_group
            ctx.bwd_occup = bwd_occup
            ctx.bwd_grid_x = bwd_grid_x
            ctx.multi_stream = multi_stream
            res = bnp.bn_addrelu_fwd_nhwc(x, z, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_addrelu_fwd_eval_nhwc(x, z, s, b, rm, riv, ret_cta, bn_group, mom, epsilon)

    @staticmethod
    def backward(ctx, grad_y):
        x, s, b, rm, riv, mini_m, mini_riv, bitmask = ctx.saved_variables
        epsilon = ctx.epsilon
        mom = ctx.momentum
        ret_cta = ctx.ret_cta
        my_data = ctx.my_data
        pair_data = ctx.pair_data
        magic = ctx.magic
        pair_data2 = ctx.pair_data2
        pair_data3 = ctx.pair_data3
        bn_group = ctx.bn_group
        bwd_occup = ctx.bwd_occup
        bwd_grid_x = ctx.bwd_grid_x
        multi_stream = ctx.multi_stream
        dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm, riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data, pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup, bwd_grid_x, multi_stream)
        return dx, dz, dscale, dbias, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class BatchNorm2d_NHWC(_BatchNorm):

    def __init__(self, num_features, fuse_relu=False, bn_group=1, max_cta_per_sm=2, cta_launch_margin=12, multi_stream=False):
        super(BatchNorm2d_NHWC, self).__init__(num_features)
        self.fuse_relu = fuse_relu
        self.multi_stream = multi_stream
        self.minibatch_mean = torch.FloatTensor(num_features)
        self.minibatch_riv = torch.FloatTensor(num_features)
        self.bn_group = bn_group
        self.max_cta_per_sm = max_cta_per_sm
        self.cta_launch_margin = cta_launch_margin
        self.my_data = None
        self.pair_data = None
        self.pair_data2 = None
        self.pair_data3 = None
        self.local_rank = 0
        self.magic = torch.IntTensor([0])
        assert max_cta_per_sm > 0
        self.fwd_occupancy = min(bnp.bn_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.bwd_occupancy = min(bnp.bn_bwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_fwd_occupancy = min(bnp.bn_addrelu_fwd_nhwc_occupancy(), max_cta_per_sm)
        self.addrelu_bwd_occupancy = min(bnp.bn_addrelu_bwd_nhwc_occupancy(), max_cta_per_sm)
        mp_count = torch.cuda.get_device_properties(None).multi_processor_count
        self.fwd_grid_dim_x = max(mp_count * self.fwd_occupancy - cta_launch_margin, 1)
        self.bwd_grid_dim_x = max(mp_count * self.bwd_occupancy - cta_launch_margin, 1)
        self.addrelu_fwd_grid_dim_x = max(mp_count * self.addrelu_fwd_occupancy - cta_launch_margin, 1)
        self.addrelu_bwd_grid_dim_x = max(mp_count * self.addrelu_bwd_occupancy - cta_launch_margin, 1)
        self.grid_dim_y = (num_features + 63) // 64
        self.ret_cta = torch.ByteTensor(8192).fill_(0)
        if bn_group > 1:
            local_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            assert world_size >= bn_group
            assert world_size % bn_group == 0
            bn_sync_steps = 1
            if bn_group == 4:
                bn_sync_steps = 2
            if bn_group == 8:
                bn_sync_steps = 3
            self.ipc_buffer = torch.ByteTensor(bnp.get_buffer_size(bn_sync_steps))
            self.my_data = bnp.get_data_ptr(self.ipc_buffer)
            self.storage = self.ipc_buffer.storage()
            self.share_cuda = self.storage._share_cuda_()
            internal_cuda_mem = self.share_cuda
            my_handle = torch.ByteTensor(np.frombuffer(internal_cuda_mem[1], dtype=np.uint8))
            my_offset = torch.IntTensor([internal_cuda_mem[3]])
            handles_all = torch.empty(world_size, my_handle.size(0), dtype=my_handle.dtype, device=my_handle.device)
            handles_l = list(handles_all.unbind(0))
            torch.distributed.all_gather(handles_l, my_handle)
            offsets_all = torch.empty(world_size, my_offset.size(0), dtype=my_offset.dtype, device=my_offset.device)
            offsets_l = list(offsets_all.unbind(0))
            torch.distributed.all_gather(offsets_l, my_offset)
            self.pair_handle = handles_l[local_rank ^ 1].cpu().contiguous()
            pair_offset = offsets_l[local_rank ^ 1].cpu()
            self.pair_data = bnp.get_remote_data_ptr(self.pair_handle, pair_offset)
            if bn_group > 2:
                self.pair_handle2 = handles_l[local_rank ^ 2].cpu().contiguous()
                pair_offset2 = offsets_l[local_rank ^ 2].cpu()
                self.pair_data2 = bnp.get_remote_data_ptr(self.pair_handle2, pair_offset2)
            if bn_group > 4:
                self.pair_handle3 = handles_l[local_rank ^ 4].cpu().contiguous()
                pair_offset3 = offsets_l[local_rank ^ 4].cpu()
                self.pair_data3 = bnp.get_remote_data_ptr(self.pair_handle3, pair_offset3)
            self.magic = torch.IntTensor([2])
            self.local_rank = local_rank

    def forward(self, x, z=None):
        if z is not None:
            assert self.fuse_relu == True
            return bn_addrelu_NHWC_impl.apply(x, z, self.weight, self.bias, self.running_mean, self.running_var, self.minibatch_mean, self.minibatch_riv, self.grid_dim_y, self.ret_cta, self.momentum, self.eps, self.training, self.bn_group, self.my_data, self.pair_data, self.magic, self.pair_data2, self.pair_data3, self.addrelu_fwd_occupancy, self.addrelu_fwd_grid_dim_x, self.addrelu_bwd_occupancy, self.addrelu_bwd_grid_dim_x, self.multi_stream)
        else:
            return bn_NHWC_impl.apply(x, self.weight, self.bias, self.running_mean, self.running_var, self.minibatch_mean, self.minibatch_riv, self.ret_cta, self.momentum, self.eps, self.fuse_relu, self.training, self.bn_group, self.my_data, self.pair_data, self.magic, self.pair_data2, self.pair_data3, self.fwd_occupancy, self.fwd_grid_dim_x, self.bwd_occupancy, self.bwd_grid_dim_x, self.multi_stream)

    def __del__(self):
        if self.bn_group > 1:
            bnp.close_remote_data(self.pair_handle)
            if self.bn_group > 2:
                bnp.close_remote_data(self.pair_handle2)
                if self.bn_group > 4:
                    bnp.close_remote_data(self.pair_handle3)


class FastLayerNormFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon)
        ctx.save_for_backward(x, gamma, mu, rsigma)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, gamma, mu, rsigma = ctx.saved_tensors
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        dymat = dy.view(xmat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, xmat, mu, rsigma, gamma)
        dx = dxmat.view(x.shape)
        return dx, dgamma, dbeta, None


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())


def _fast_layer_norm(x, weight, bias, epsilon):
    args = _cast_if_autocast_enabled(x, weight, bias, epsilon)
    with torch.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FastLayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-05):
        super().__init__()
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return _fast_layer_norm(x, self.weight, self.bias, self.epsilon)


def _set_sequence_parallel_enabled(param: torch.Tensor, sequence_parallel_enabled: bool) ->None:
    setattr(param, 'sequence_parallel_enabled', sequence_parallel_enabled)


class EncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, input_biases_q, input_biases_kv, output_biases, mask, dropout_prob):
        use_biases_t = torch.tensor([input_biases_q is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        if use_biases_t[0]:
            input_lin_q_results = torch.addmm(input_biases_q, inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), input_weights_q.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_q_results = torch.mm(inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), input_weights_q.transpose(0, 1))
        input_lin_q_results = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1), input_weights_q.size(0))
        if use_biases_t[0]:
            input_lin_kv_results = torch.addmm(input_biases_kv, inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)), input_weights_kv.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_kv_results = torch.mm(inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)), input_weights_kv.transpose(0, 1))
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1), input_weights_kv.size(0))
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads, head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads, 2, head_dim)
        keys = input_lin_kv_results[:, :, 0, :]
        values = input_lin_kv_results[:, :, 1, :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2), out=matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1), 'Sequence length should match!'
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)), dtype=dropout_results.dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)), output_weights.transpose(0, 1))
        outputs = outputs.view(inputs_q.size(0), inputs_q.size(1), output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        head_dim = inputs_q.size(2) // heads_t[0]
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(1) * heads_t[0], head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0), inputs_kv.size(1) * heads_t[0], 2, head_dim)
        keys = input_lin_kv_results[:, :, 0, :]
        values = input_lin_kv_results[:, :, 1, :]
        input_lin_kv_results_grads = torch.empty_like(input_lin_kv_results)
        queries_grads = torch.empty_like(queries)
        keys_grads = input_lin_kv_results_grads[:, :, 0, :]
        values_grads = input_lin_kv_results_grads[:, :, 1, :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1), matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results.dtype)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1), out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        queries_grads = queries_grads.transpose(0, 1).view(inputs_q.size(0) * inputs_q.size(1), heads_t[0] * head_dim)
        input_q_grads = torch.mm(queries_grads, input_weights_q)
        input_q_grads = input_q_grads.view(inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        input_lin_kv_results_grads = input_lin_kv_results_grads.view(inputs_kv.size(0) * inputs_kv.size(1), heads_t[0] * 2 * head_dim)
        input_kv_grads = torch.mm(input_lin_kv_results_grads, input_weights_kv)
        input_kv_grads = input_kv_grads.view(inputs_kv.size(0), inputs_kv.size(1), inputs_kv.size(2))
        input_weight_q_grads = torch.mm(queries_grads.transpose(0, 1), inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)))
        input_weight_kv_grads = torch.mm(input_lin_kv_results_grads.transpose(0, 1), inputs_kv.view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(2)))
        if use_biases_t[0]:
            input_bias_grads_q = torch.sum(queries_grads, 0)
            input_bias_grads_kv = torch.sum(input_lin_kv_results_grads, 0)
        else:
            input_bias_grads_q = None
            input_bias_grads_kv = None
        return None, None, None, None, input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, input_bias_grads_q, input_bias_grads_kv, output_bias_grads, None, None


encdec_attn_func = EncdecAttnFunc.apply


class FastEncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        input_lin_q_results, input_lin_kv_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_multihead_attn.encdec_multihead_attn_forward(use_mask, use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads = fast_multihead_attn.encdec_multihead_attn_backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_prob_t[0])
        return None, None, None, input_q_grads, input_kv_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, None, None


fast_encdec_attn_func = FastEncdecAttnFunc.apply


class FastEncdecAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_q_results, input_lin_kv_results, softmax_results, dropout_results, dropout_mask, matmul2_results, dropout_add_mask, outputs = fast_multihead_attn.encdec_multihead_attn_norm_add_forward(use_mask, use_time_mask, is_training, heads, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t = ctx.saved_tensors
        input_q_grads, input_kv_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads = fast_multihead_attn.encdec_multihead_attn_norm_add_backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results, input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t[0])
        return None, None, None, input_q_grads, input_kv_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_q_grads, input_weight_kv_grads, output_weight_grads, None, None


fast_encdec_attn_norm_add_func = FastEncdecAttnNormAddFunc.apply


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False, include_norm_add=False, impl='fast'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.bias = bias
        self.include_norm_add = include_norm_add
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if self.bias:
            assert impl != 'fast', 'ERROR! The Fast implementation does not support biases!'
            self.in_proj_bias_q = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_kv = Parameter(torch.Tensor(2 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter('in_proj_bias_q', None)
            self.register_parameter('in_proj_bias_kv', None)
            self.in_proj_bias_q = None
            self.in_proj_bias_kv = None
            self.out_proj_bias = None
        if self.include_norm_add:
            if impl == 'fast':
                self.lyr_nrm_gamma_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm_beta_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm = None
            else:
                self.register_parameter('lyr_norm_gamma_weights', None)
                self.register_parameter('lyr_norm_beta_weights', None)
                self.lyr_nrm_gamma_weights = None
                self.lyr_nrm_beta_weights = None
                self.lyr_nrm = FusedLayerNorm(embed_dim)
        self.reset_parameters()
        if self.include_norm_add:
            if impl == 'fast':
                self.attn_func = fast_encdec_attn_norm_add_func
            elif impl == 'default':
                self.attn_func = encdec_attn_func
            else:
                assert False, 'Unsupported impl: {} !'.format(impl)
        elif impl == 'fast':
            self.attn_func = fast_encdec_attn_func
        elif impl == 'default':
            self.attn_func = encdec_attn_func
        else:
            assert False, 'Unsupported impl: {} !'.format(impl)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight_q)
        nn.init.xavier_uniform_(self.in_proj_weight_kv, gain=math.sqrt(1.5))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias_q, 0.0)
            nn.init.constant_(self.in_proj_bias_kv, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)
        if self.include_norm_add:
            if self.impl == 'fast':
                nn.init.ones_(self.lyr_nrm_gamma_weights)
                nn.init.zeros_(self.lyr_nrm_beta_weights)
            else:
                self.lyr_nrm.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if key_padding_mask is not None:
            assert attn_mask is None, 'ERROR attn_mask and key_padding_mask should not be both defined!'
            mask = key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        else:
            mask = None
        if self.include_norm_add:
            if self.impl == 'fast':
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, key, self.lyr_nrm_gamma_weights, self.lyr_nrm_beta_weights, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, mask, self.dropout)
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, lyr_nrm_results, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, self.in_proj_bias_q, self.in_proj_bias_kv, self.out_proj_bias, mask, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout, is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, mask, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query, key, self.in_proj_weight_q, self.in_proj_weight_kv, self.out_proj_weight, self.in_proj_bias_q, self.in_proj_bias_kv, self.out_proj_bias, mask, self.dropout)
        return outputs, None


class FastSelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask, mask_additive, dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        mask_additive_t = torch.tensor([mask_additive])
        if use_biases_t[0]:
            if not mask_additive:
                input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_multihead_attn.self_attn_bias_forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask if use_mask else null_tensor, dropout_prob)
                ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, null_tensor, null_tensor, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
            else:
                input_lin_results, bmm1_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_multihead_attn.self_attn_bias_additive_mask_forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases, pad_mask if use_mask else null_tensor, dropout_prob)
                ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, null_tensor, bmm1_results, pad_mask, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        else:
            input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, outputs = fast_multihead_attn.self_attn_forward(use_mask, use_time_mask, is_training, heads, inputs, input_weights, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
            ctx.save_for_backward(use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, null_tensor, null_tensor, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, matmul2_results, dropout_results, softmax_results, bmm1_results, pad_mask, mask_additive_t, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        if use_biases_t[0]:
            if not mask_additive_t[0]:
                input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads = fast_multihead_attn.self_attn_bias_backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
            else:
                input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads = fast_multihead_attn.self_attn_bias_additive_mask_backward(heads_t[0], output_grads, matmul2_results, dropout_results, bmm1_results, pad_mask, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
        else:
            input_bias_grads = None
            output_bias_grads = None
            input_grads, input_weight_grads, output_weight_grads = fast_multihead_attn.self_attn_backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t[0])
        return None, None, None, input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads, None, None, None


fast_self_attn_func = FastSelfAttnFunc.apply


class FastSelfAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_results, softmax_results, dropout_results, dropout_mask, matmul2_results, dropout_add_mask, outputs = fast_multihead_attn.self_attn_norm_add_forward(use_mask, use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, pad_mask if use_mask else null_tensor, dropout_prob)
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        heads_t, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t = ctx.saved_tensors
        input_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_grads, output_weight_grads = fast_multihead_attn.self_attn_norm_add_backward(heads_t[0], output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob_t[0])
        return None, None, None, input_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_grads, output_weight_grads, None, None


fast_self_attn_norm_add_func = FastSelfAttnNormAddFunc.apply


class SelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs, input_weights, output_weights, input_biases, output_biases, mask, is_additive_mask, dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads
        if use_biases_t[0]:
            input_lin_results = torch.addmm(input_biases, inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_results = torch.mm(inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0), keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2), out=matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1), 'Sequence length should match!'
                mask = mask
                matmul1_results = matmul1_results.masked_fill_(mask, float('-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q, seql_k)
                if is_additive_mask:
                    matmul1_results = matmul1_results + mask.unsqueeze(1).unsqueeze(2)
                else:
                    mask = mask
                    matmul1_results = matmul1_results.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q, seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)), dtype=dropout_results.dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1), inputs.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)), output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs.size(0) * inputs.size(1), inputs.size(2)), output_weights.transpose(0, 1))
        outputs = outputs.view(inputs.size(0), inputs.size(1), output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        use_biases_t, heads_t, scale_t, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs, input_weights, output_weights, dropout_mask, dropout_prob_t = ctx.saved_tensors
        head_dim = inputs.size(2) // heads_t[0]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1) * heads_t[0], 3, head_dim)
        queries = input_lin_results[:, :, 0, :]
        keys = input_lin_results[:, :, 1, :]
        values = input_lin_results[:, :, 2, :]
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads = input_lin_results_grads[:, :, 0, :]
        keys_grads = input_lin_results_grads[:, :, 1, :]
        values_grads = input_lin_results_grads[:, :, 2, :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1), matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1).transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2), output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results.dtype)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1), softmax_grads, keys.transpose(0, 1), out=queries_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1), softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0) * inputs.size(1), heads_t[0] * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))
        input_weight_grads = torch.mm(input_lin_results_grads.transpose(0, 1), inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)))
        if use_biases_t[0]:
            input_bias_grads = torch.sum(input_lin_results_grads, 0)
        else:
            input_bias_grads = None
        return None, None, None, None, input_grads, input_weight_grads, output_weight_grads, input_bias_grads, output_bias_grads, None, None, None


self_attn_func = SelfAttnFunc.apply


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False, include_norm_add=False, impl='fast', separate_qkv_params=False, mask_additive=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.bias = bias
        self.include_norm_add = include_norm_add
        self.impl = impl
        self.scaling = self.head_dim ** -0.5
        self.separate_qkv_params = separate_qkv_params
        self.mask_additive = mask_additive
        if mask_additive:
            assert self.include_norm_add == False, 'additive mask not supported with layer norm'
            assert impl == 'default' or impl == 'fast' and bias, 'additive mask not supported for fast mode without bias'
        if separate_qkv_params:
            self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.v_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        else:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if self.bias:
            if separate_qkv_params:
                self.q_bias = Parameter(torch.Tensor(embed_dim))
                self.k_bias = Parameter(torch.Tensor(embed_dim))
                self.v_bias = Parameter(torch.Tensor(embed_dim))
            else:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            if separate_qkv_params:
                self.register_parameter('q_bias', None)
                self.register_parameter('k_bias', None)
                self.register_parameter('v_bias', None)
                self.q_bias = None
                self.k_bias = None
                self.v_bias = None
            else:
                self.register_parameter('in_proj_bias', None)
                self.in_proj_bias = None
            self.register_parameter('out_proj_bias', None)
            self.out_proj_bias = None
        if self.include_norm_add:
            if impl == 'fast':
                self.lyr_nrm_gamma_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm_beta_weights = Parameter(torch.Tensor(embed_dim))
                self.lyr_nrm = None
            else:
                self.register_parameter('lyr_norm_gamma_weights', None)
                self.register_parameter('lyr_norm_beta_weights', None)
                self.lyr_nrm_gamma_weights = None
                self.lyr_nrm_beta_weights = None
                self.lyr_nrm = FusedLayerNorm(embed_dim)
        self.reset_parameters()
        if self.include_norm_add:
            if impl == 'fast':
                self.attn_func = fast_self_attn_norm_add_func
            elif impl == 'default':
                self.attn_func = self_attn_func
            else:
                assert False, 'Unsupported impl: {} !'.format(impl)
        elif impl == 'fast':
            self.attn_func = fast_self_attn_func
        elif impl == 'default':
            self.attn_func = self_attn_func
        else:
            assert False, 'Unsupported impl: {} !'.format(impl)

    def reset_parameters(self):
        if self.separate_qkv_params:
            nn.init.xavier_uniform_(self.q_weight)
            nn.init.xavier_uniform_(self.k_weight)
            nn.init.xavier_uniform_(self.v_weight)
        else:
            nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            if self.separate_qkv_params:
                nn.init.constant_(self.q_bias, 0.0)
                nn.init.constant_(self.k_bias, 0.0)
                nn.init.constant_(self.v_bias, 0.0)
            else:
                nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)
        if self.include_norm_add:
            if self.impl == 'fast':
                nn.init.ones_(self.lyr_nrm_gamma_weights)
                nn.init.zeros_(self.lyr_nrm_beta_weights)
            else:
                self.lyr_nrm.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if self.separate_qkv_params:
            input_weights = torch.cat([self.q_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim), self.k_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim), self.v_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim)], dim=1).reshape(3 * self.embed_dim, self.embed_dim).contiguous()
        else:
            input_weights = self.in_proj_weight
        if self.bias:
            if self.separate_qkv_params:
                input_bias = torch.cat([self.q_bias.view(self.num_heads, 1, self.head_dim), self.k_bias.view(self.num_heads, 1, self.head_dim), self.v_bias.view(self.num_heads, 1, self.head_dim)], dim=1).reshape(3 * self.embed_dim).contiguous()
            else:
                input_bias = self.in_proj_bias
        else:
            input_bias = None
        if key_padding_mask is not None:
            assert attn_mask is None, 'ERROR attn_mask and key_padding_mask should not be both defined!'
            mask = key_padding_mask
        elif attn_mask is not None:
            assert self.mask_additive == False, 'additive mask not supported for time mask'
            mask = attn_mask
        else:
            mask = None
        if self.include_norm_add:
            if self.impl == 'fast':
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, self.lyr_nrm_gamma_weights, self.lyr_nrm_beta_weights, input_weights, self.out_proj_weight, mask, self.dropout)
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, lyr_nrm_results, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.mask_additive, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout, is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, query, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.mask_additive, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training, self.num_heads, self.scaling, query, input_weights, self.out_proj_weight, input_bias, self.out_proj_bias, mask, self.mask_additive, self.dropout)
        return outputs, None


class BNModelRef(nn.Module):

    def __init__(self, num_features, num_layers=1000):
        super().__init__()
        self.fwd = nn.Sequential(*[nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in range(num_layers)])

    def forward(self, x):
        return self.fwd(x)


class BNModel(nn.Module):

    def __init__(self, num_features, num_layers=1000):
        super().__init__()
        self.fwd = nn.Sequential(*[GBN(num_features, group_size=2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) for _ in range(num_layers)])

    def forward(self, x):
        return self.fwd(x)


class SimpleModel(torch.nn.Module):

    def __init__(self, num_layers, size):
        super().__init__()
        self.params = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1, size) + 1) for _ in range(num_layers)])

    def forward(self, x):
        y = 0
        for i, param in enumerate(self.params):
            y += (i + 1) * param * x
        return y


class ModelFoo(torch.nn.Module):

    def __init__(self):
        super(ModelFoo, self).__init__()
        self.linear = torch.nn.Linear(128, 128, bias=False)
        self.loss = torch.nn.MSELoss()

    def forward(self, input_tensor, gt):
        y = self.linear(input_tensor)
        loss = self.loss(y, gt)
        return loss


class TransducerJointFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, g, f_len, g_len, pack_output, relu, dropout, batch_offset, packed_batch, opt, fwd_tile_size, dropout_prob, mask_probe):
        h = transducer_joint_cuda.forward(f, g, f_len, g_len, batch_offset, packed_batch, opt, pack_output, relu, dropout, dropout_prob, fwd_tile_size)
        masked = relu or dropout
        if masked:
            ctx.save_for_backward(h[1], f_len, g_len, batch_offset)
            if mask_probe is not None:
                mask_probe.append(h[1])
        else:
            ctx.save_for_backward(f_len, g_len, batch_offset)
        ctx.pack_output = pack_output
        ctx.masked = relu or dropout
        ctx.max_f_len = f.size(1)
        ctx.max_g_len = g.size(1)
        ctx.scale = 1 / (1 - dropout_prob) if dropout and dropout_prob != 1 else 1
        return h[0]

    @staticmethod
    def backward(ctx, loss_grad):
        if ctx.masked:
            mask, f_len, g_len, batch_offset = ctx.saved_tensors
            inp = [loss_grad, mask]
        else:
            f_len, g_len, batch_offset = ctx.saved_tensors
            inp = [loss_grad]
        f_grad, g_grad = transducer_joint_cuda.backward(inp, f_len, g_len, batch_offset, ctx.max_f_len, ctx.max_g_len, ctx.pack_output, ctx.scale)
        return f_grad, g_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class TransducerJoint(torch.nn.Module):
    """Transducer joint
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        pack_output (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
        relu (bool, optional): apply ReLU to the output of the joint operation. Requires opt=1  
        (default: False)
        dropout (bool, optional): apply dropout to the output of the joint operation. Requires opt=1  
        (default: False)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a tiled algorithm. 
            (default: 1)
        fwd_tile_size (int, optional): tile size used in forward operation. This argument will be 
        ignored if opt != 1. (default: 4) 
        dropout_prob (float, optional): dropout probability. (default: 0.0)
        probe_mask (bool, optional): a flag used to probe the mask generated by ReLU and/or dropout
        operation. When this argument is set to True, the mask can be accessed through 
        self.mask_probe. (default: false)
    """

    def __init__(self, pack_output=False, relu=False, dropout=False, opt=1, fwd_tile_size=4, dropout_prob=0, probe_mask=False):
        super(TransducerJoint, self).__init__()
        self.pack_output = pack_output
        self.relu = relu
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.opt = opt
        self.fwd_tile_size = fwd_tile_size
        self.dummy_batch_offset = torch.empty(0)
        masked = self.relu or self.dropout
        self.mask_probe = [] if masked and probe_mask else None
        if masked and opt != 1:
            raise NotImplementedError('ReLU and dropout fusion is only supported with opt=1')

    def forward(self, f, g, f_len, g_len, batch_offset=None, packed_batch=0):
        """Forward operation of transducer joint

        Arguments:
            f (tensor): transcription vector from encode block of shape (B, T, H).
            g (tensor): prediction vector form predict block of shape (B, U, H).
            f_len (tensor): length of transcription vector for each batch.
            g_len (tensor): length of prediction vector minus 1 for each batch.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the results. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*g_len, dim=0)
                This argument is required if pack_output == True, and is ignored if 
                pack_output == False. (default: None)
            packed_batch (int, optional): the batch size after packing. This argument is 
                ignored if pack_output == False. (default: 0)
        """
        my_batch_offset = batch_offset if self.pack_output else self.dummy_batch_offset
        if self.pack_output and (batch_offset is None or packed_batch == 0):
            raise Exception('Please specify batch_offset and packed_batch when packing is enabled')
        dropout = self.dropout and self.training
        return TransducerJointFunc.apply(f, g, f_len, g_len, self.pack_output, self.relu, dropout, my_batch_offset, packed_batch, self.opt, self.fwd_tile_size, self.dropout_prob, self.mask_probe)


class TransducerLossFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, fuse_softmax_backward, debug_list, opt, packed_input):
        if fuse_softmax_backward == False:
            with torch.enable_grad():
                x = torch.nn.functional.log_softmax(x, dim=-1)
        else:
            x = torch.nn.functional.log_softmax(x, dim=-1)
        alpha, beta, loss = transducer_loss_cuda.forward(x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, opt, packed_input)
        if debug_list == []:
            debug_list += [alpha, beta]
        ctx.save_for_backward(x, alpha, beta, f_len, y_len, label, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.fuse_softmax_backward = fuse_softmax_backward
        ctx.opt = opt
        ctx.packed_input = packed_input
        ctx.max_f_len = max_f_len
        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        x, alpha, beta, f_len, y_len, label, batch_offset = ctx.saved_tensors
        x_grad = transducer_loss_cuda.backward(x, loss_grad, alpha, beta, f_len, y_len, label, batch_offset, ctx.max_f_len, ctx.blank_idx, ctx.opt, ctx.fuse_softmax_backward, ctx.packed_input)
        if ctx.fuse_softmax_backward == False:
            x_grad = x.backward(x_grad)
        return x_grad, None, None, None, None, None, None, None, None, None, None


class TransducerLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        fuse_softmax_backward (bool, optional) whether to fuse the backward of transducer loss with
            softmax. (default: True)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a more optimized 
            algorithm. In some cases, opt=1 might fall back to opt=0. (default: 1)
        packed_input (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
    """

    def __init__(self, fuse_softmax_backward=True, opt=1, packed_input=False):
        super(TransducerLoss, self).__init__()
        self.fuse_softmax_backward = fuse_softmax_backward
        self.opt = opt
        self.packed_input = packed_input
        self.dummy_batch_offset = torch.empty(0)

    def forward(self, x, label, f_len, y_len, blank_idx, batch_offset=None, max_f_len=None, debug_list=None):
        """Forward operation of transducer joint

        Arguments:
            x (tensor): input tensor to the loss function with a shape of (B, T, U, H).
            label (tensor): labels for the input data.
            f_len (tensor): lengths of the inputs in the time dimension for each batch.
            y_len (tensor): lengths of the labels for each batch.
            blank_idx (int): index for the null symbol.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the input. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*(y_len+1), dim=0)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
            max_f_len (int, optional): maximum length of the input in the time dimension.
                For example, it can be obtained as 
                max_f_len = max(f_len)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
                (default: None)
            debug_list (list, optional): when an empty list is supplied, Alpha and Beta generated 
                in the forward operation will be attached to this list for debug purpose. 
                (default: None)
        """
        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception('Please specify batch_offset and max_f_len when packing is                                     enabled')
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len
        else:
            my_batch_offset = self.dummy_batch_offset
            my_max_f_len = x.size(1)
        return TransducerLossFunc.apply(x, label, f_len, y_len, my_batch_offset, my_max_f_len, blank_idx, self.fuse_softmax_backward, debug_list, self.opt, self.packed_input)


class tofp16(nn.Module):
    """
    Utility module that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        deprecated_warning('apex.fp16_utils is deprecated and will be removed by the end of February 2023. Use [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)')
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class DenseNoBiasFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight


def _dense_no_bias(input, weight):
    args = _cast_if_autocast_enabled(input, weight)
    with torch.amp.autocast(enabled=False):
        return DenseNoBiasFunc.apply(*args)


class FusedDenseFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = fused_dense_cuda.linear_bias_forward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_backward(input, weight, grad_output)
        return grad_input, grad_weight, grad_bias


def _fused_dense(input, weight, bias):
    args = _cast_if_autocast_enabled(input, weight, bias)
    with torch.amp.autocast(enabled=False):
        return FusedDenseFunc.apply(*args)


class FusedDense(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(FusedDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        if self.bias is not None:
            return _fused_dense(input, self.weight, self.bias)
        else:
            return _dense_no_bias(input, self.weight)


class FusedDenseGeluDenseFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight1, bias1, weight2, bias2):
        ctx.save_for_backward(input, weight1, weight2)
        output1, output2, gelu_in = fused_dense_cuda.linear_gelu_linear_forward(input, weight1, bias1, weight2, bias2)
        ctx.save_for_backward(input, weight1, weight2, gelu_in, output1)
        return output2

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, weight2, gelu_in, output1 = ctx.saved_tensors
        grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2 = fused_dense_cuda.linear_gelu_linear_backward(input, gelu_in, output1, weight1, weight2, grad_output)
        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2


def _fused_dense_gelu_dense(input, weight1, bias1, weight2, bias2):
    args = _cast_if_autocast_enabled(input, weight1, bias1, weight2, bias2)
    with torch.amp.autocast(enabled=False):
        return FusedDenseGeluDenseFunc.apply(*args)


class FusedDenseGeluDense(nn.Module):

    def __init__(self, in_features, intermediate_features, out_features, bias=True):
        super(FusedDenseGeluDense, self).__init__()
        assert bias == True, 'DenseGeluDense module without bias is currently not supported'
        self.in_features = in_features
        self.intermediate_features = intermediate_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.Tensor(intermediate_features, in_features))
        self.bias1 = nn.Parameter(torch.Tensor(intermediate_features))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, intermediate_features))
        self.bias2 = nn.Parameter(torch.Tensor(out_features))

    def forward(self, input):
        return _fused_dense_gelu_dense(input, self.weight1, self.bias1, self.weight2, self.bias2)


class MlpFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bias, activation, *args):
        output = mlp_cuda.forward(bias, activation, args)
        ctx.save_for_backward(*args)
        ctx.outputs = output
        ctx.bias = bias
        ctx.activation = activation
        return output[0]

    @staticmethod
    def backward(ctx, grad_o):
        grads = mlp_cuda.backward(ctx.bias, ctx.activation, grad_o, ctx.outputs, ctx.saved_tensors)
        del ctx.outputs
        return None, None, *grads


def mlp_function(bias, activation, *args):
    autocast_args = _cast_if_autocast_enabled(bias, activation, *args)
    return MlpFunction.apply(*autocast_args)


class MLP(torch.nn.Module):
    """Launch MLP in C++

    Args:
        mlp_sizes (list of int): MLP sizes. Example: [1024,1024,1024] will create 2 MLP layers with shape 1024x1024
        bias (bool): Default True:
        relu (bool): Default True
    """

    def __init__(self, mlp_sizes, bias=True, activation='relu'):
        super().__init__()
        self.num_layers = len(mlp_sizes) - 1
        self.mlp_sizes = copy(mlp_sizes)
        self.bias = 1 if bias else 0
        if activation == 'none':
            self.activation = 0
        elif activation == 'relu':
            self.activation = 1
        elif activation == 'sigmoid':
            self.activation = 2
        else:
            raise TypeError('activation must be relu or none.')
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1], mlp_sizes[i]))
            self.weights.append(w)
            name = 'weight_{}'.format(i)
            setattr(self, name, w)
            if self.bias:
                b = torch.nn.Parameter(torch.empty(mlp_sizes[i + 1]))
                self.biases.append(b)
                name = 'bias_{}'.format(i)
                setattr(self, name, b)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            dimsum = weight.size(0) + weight.size(1)
            std = math.sqrt(2.0 / float(dimsum))
            nn.init.normal_(weight, 0.0, std)
        if self.bias:
            for bias in self.biases:
                std = math.sqrt(1.0 / float(bias.size(0)))
                nn.init.normal_(bias, 0.0, std)

    def forward(self, input):
        return mlp_function(self.bias, self.activation, input, *self.weights, *self.biases)

    def extra_repr(self):
        s = f'MLP sizes: {self.mlp_sizes}, Bias={self.bias}, activation={self.activation}'
        return s


class FusedRMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward(input_, ctx.normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.rms_backward(grad_output.contiguous(), invvar, input_, ctx.normalized_shape, ctx.eps)
        return grad_input, None, None


def fused_rms_norm(input, normalized_shape, eps=1e-06):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.amp.autocast(enabled=False):
        return FusedRMSNormFunction.apply(*args)


class FusedRMSNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine(input_, ctx.normalized_shape, weight_, ctx.eps)
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(grad_output.contiguous(), invvar, input_, ctx.normalized_shape, weight_, ctx.eps)
        return grad_input, grad_weight, None, None


def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-06):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.amp.autocast(enabled=False):
        return FusedRMSNormAffineFunction.apply(*args)


def manual_rms_norm(input, normalized_shape, weight, eps):
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    if weight is None:
        return input
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input
    return weight * input


class FusedRMSNorm(torch.nn.Module):
    """Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \\frac{x}{\\mathrm{RMS}[x]} * \\gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \\times \\text{normalized}\\_\\text{shape}[0] \\times \\text{normalized}\\_\\text{shape}[1]
                    \\times \\ldots \\times \\text{normalized}\\_\\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        if not input.is_cuda:
            return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)
        if self.elementwise_affine:
            return fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)
        else:
            return fused_rms_norm(input, self.normalized_shape, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class FusedRMSNormAffineMixedDtypesFunction(FusedRMSNormAffineFunction):

    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine_mixed_dtypes(input_, ctx.normalized_shape, weight_, ctx.eps)
        ctx.save_for_backward(input_, weight_, invvar)
        return output


def mixed_dtype_fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-06):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.amp.autocast(enabled=False):
        return FusedRMSNormAffineMixedDtypesFunction.apply(*args)


class MixedFusedRMSNorm(FusedRMSNorm):

    def __init__(self, normalized_shape, eps=1e-05, **kwargs):
        if 'elementwise_affine' in kwargs:
            import warnings
            warnings.warn('MixedFusedRMSNorm does not support `elementwise_affine` argument')
            elementwise_affine = kwargs.pop('elementwise_affine')
            if not elementwise_affine:
                raise RuntimeError('MixedFusedRMSNorm does not support `elementwise_affine = False`')
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=True)

    def forward(self, input: torch.Tensor):
        if not input.is_cuda:
            return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)
        return mixed_dtype_fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)


def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        None
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True


imported_flatten_impl = False


def flatten(bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return flatten_impl(bucket)


def unflatten(coalesced, bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return unflatten_impl(coalesced, bucket)


def apply_flat_dist_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)
    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)
    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
    for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
        buf.copy_(synced)


def split_by_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets


def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)
    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, call, extra_args)


class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def check_avail(self):
        if MultiTensorApply.available == False:
            raise RuntimeError('Attempted to call MultiTensorApply method, but MultiTensorApply is not available, possibly because Apex was installed without --cpp_ext --cuda_ext.  Original import error message:', MultiTensorApply.import_err)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.check_avail()
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


def split_half_float_double(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor']
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps, process_group, world_size):
        torch.cuda.nvtx.range_push('sync_BN_fw')
        c_last_input = input.transpose(1, -1).contiguous().clone()
        ctx.save_for_backward(c_last_input, weight, bias, running_mean, running_variance)
        ctx.eps = eps
        ctx.process_group = process_group
        ctx.world_size = world_size
        c_last_input = (c_last_input - running_mean) / torch.sqrt(running_variance + eps)
        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias
        torch.cuda.nvtx.range_pop()
        return c_last_input.transpose(1, -1).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.nvtx.range_push('sync_BN_bw')
        c_last_input, weight, bias, running_mean, running_variance = ctx.saved_tensors
        eps = ctx.eps
        process_group = ctx.process_group
        world_size = ctx.world_size
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]
        torch.cuda.nvtx.range_push('carilli field')
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        c_grad = c_last_grad.view(-1, num_features).contiguous()
        torch.cuda.nvtx.range_pop()
        if ctx.needs_input_grad[0]:
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input - running_mean)).view(-1, num_features).mean(0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_dy, ReduceOp.SUM, process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM, process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            c_last_grad_input = (c_last_grad - mean_dy - (c_last_input - running_mean) / (running_variance + eps) * mean_dy_xmu) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()
        grad_weight = None
        if weight is not None and ctx.needs_input_grad[1]:
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = c_grad.sum(0)
        torch.cuda.nvtx.range_pop()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    ``DistributedDataParallel``.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from ``torch.distributed``.

    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Example::

        >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """
    warned = False

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        deprecated_warning('apex.parallel.SyncBatchNorm is deprecated and will be removed by the end of February 2023. Use `torch.nn.SyncBatchNorm`.')
        if channel_last == True:
            raise AttributeError('channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.')
        if not SyncBatchNorm.warned:
            if hasattr(self, 'syncbn_import_error'):
                None
            else:
                None
            SyncBatchNorm.warned = True
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def forward(self, input):
        torch.cuda.nvtx.range_push('sync_bn_fw_with_mean_var')
        mean = None
        var = None
        cast = None
        out = None
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input
                cast = input.dtype
        if not self.training and self.track_running_stats:
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(channel_first_input.size(0), -1)
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(local_mean, ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(local_sqr_mean, ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                var = sqr_mean - mean.pow(2)
                if self.running_mean is not None:
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                if self.running_var is not None:
                    self.running_var = m / (m - 1) * self.momentum * var + (1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
        return out


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class ScaledMaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    if mask is not None:
        args = _cast_if_autocast_enabled(inputs, mask, scale)
        with torch.amp.autocast(enabled=False):
            return ScaledMaskedSoftmax.apply(*args)
    else:
        args = _cast_if_autocast_enabled(inputs, scale)
        with torch.amp.autocast(enabled=False):
            return ScaledSoftmax.apply(*args)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None


def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, 'causal mask is only for self attention'
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.amp.autocast(enabled=False):
        probs = ScaledUpperTriangMaskedSoftmax.apply(*args)
    return probs.view(b, np, sq, sk)


class FusedScaleMaskSoftmax(torch.nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(self, input_in_fp16, input_in_bf16, attn_mask_type, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise RuntimeError('both fp16 and bf16 flags cannot be active at the same time.')
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        if not (self.scale is None or softmax_in_fp32):
            raise RuntimeError('softmax should be in fp32 when scaled')
        if self.scaled_masked_softmax_fusion:
            if self.attn_mask_type == AttnMaskType.causal:
                self.fused_softmax_func = scaled_upper_triang_masked_softmax
            elif self.attn_mask_type == AttnMaskType.padding:
                self.fused_softmax_func = scaled_masked_softmax
            else:
                raise ValueError('Invalid attn_mask_type.')

    def forward(self, input, mask):
        assert input.dim() == 4
        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if self.scaled_masked_softmax_fusion and self.input_in_float16 and (self.attn_mask_type == AttnMaskType.causal or self.attn_mask_type == AttnMaskType.padding) and 16 < sk <= 4096 and sq % 4 == 0 and sk % 4 == 0 and attn_batches % 4 == 0:
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                elif sq % batch_per_block == 0:
                    return True
        return False

    def forward_fused_softmax(self, input, mask):
        scale = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()
        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)
        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()
        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)


class GenericScaledMaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = generic_scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = generic_scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def generic_scaled_masked_softmax(inputs, mask, scale):
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.amp.autocast(enabled=False):
        return GenericScaledMaskedSoftmax.apply(*args)


class GenericFusedScaleMaskSoftmax(FusedScaleMaskSoftmax):
    """
    Generic version of FusedSacleMaskSoftmax.
    It removes the seq-len limitations and has slight performance degragation compared with FusedScaleMaskSoftmax

    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(self, input_in_fp16, input_in_bf16, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__(input_in_fp16, input_in_bf16, AttnMaskType.padding, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale)
        self.scaled_masked_softmax_fusion = generic_scaled_masked_softmax

    def is_kernel_available(self, mask, b, np, sq, sk):
        if self.scaled_masked_softmax_fusion and 0 < sk:
            return True
        return False


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank, world_size: int) ->Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) ->Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


_TENSOR_MODEL_PARALLEL_GROUP = None


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, 'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False, 'partition_dim': -1, 'partition_stride': 1}


def set_tensor_model_parallel_attributes(tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int) ->None:
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def _initialize_affine_weight_cpu(weight, output_size, input_size, per_partition_size, partition_dim, init_method, stride=1, return_master_weight=False, *, params_dtype=torch.float32):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, dim=partition_dim, stride=stride)
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)
    _lazy_call(cb)


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
        stride (int):
    """
    set_tensor_model_parallel_attributes(tensor=weight, is_parallel=True, dim=partition_dim, stride=stride)
    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _reduce(input_: torch.Tensor) ->torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the tensor model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, init_method=init.xavier_normal_, *, params_dtype: torch.dtype=torch.float32, use_cpu_initialization: bool=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, dtype=params_dtype))
            _initialize_affine_weight_cpu(self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method, params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=torch.cuda.current_device(), dtype=params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


_grad_accum_fusion_available = True


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the tensor model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


def copy_to_tensor_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def _gather_along_last_dim(input_: torch.Tensor) ->torch.Tensor:
    """Gather tensors and concatenate along the last dimension."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool=False) ->List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _split_along_last_dim(input_: torch.Tensor) ->torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()
    return output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from tensor model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication and gradient accumulation fusion in backprop."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], gradient_accumulation_fusion: bool, async_grad_allreduce: bool, sequence_parallel_enabled: bool, use_16bit_in_wgrad_accum_fusion: bool=False):
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.use_16bit_in_wgrad_accum_fusion = use_16bit_in_wgrad_accum_fusion
        ctx.compute_weight_gradient = weight.requires_grad
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(input, weight)
        else:
            ctx.save_for_backward(weight)
        if ctx.sequence_parallel_enabled:
            world_size = get_tensor_model_parallel_world_size()
            shape = list(input.shape)
            shape[0] *= world_size
            all_gather_buffer = torch.empty(shape, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False)
            torch.distributed.all_gather_into_tensor(all_gather_buffer, input, group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None
        use_bias = ctx.use_bias
        handle = None
        if ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                world_size = get_tensor_model_parallel_world_size()
                shape = list(input.shape)
                shape[0] *= world_size
                all_gather_buffer = torch.empty(shape, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False)
                handle = torch.distributed.all_gather_into_tensor(all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True)
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)
        if handle is not None:
            handle.wait()
        if ctx.async_grad_allreduce:
            handle = torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group(), async_op=True)
        if not ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                assert not ctx.async_grad_allreduce
                world_size = get_tensor_model_parallel_world_size()
                shape = list(grad_input.shape)
                shape[0] //= world_size
                sub_grad_input = torch.empty(torch.Size(shape), dtype=grad_input.dtype, device=torch.cuda.current_device(), requires_grad=False)
                handle = torch.distributed.reduce_scatter_tensor(sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True)
                handle.wait()
                return sub_grad_input, None, None, None, None, None, None
            if ctx.async_grad_allreduce:
                handle.wait()
            return grad_input, None, None, None, None, None, None
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])
        if ctx.sequence_parallel_enabled:
            assert not ctx.async_grad_allreduce
            sub_grad_input = torch.empty(input.shape, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False)
            handle = torch.distributed.reduce_scatter_tensor(sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True)
        if ctx.gradient_accumulation_fusion:
            if not ctx.use_16bit_in_wgrad_accum_fusion:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            else:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        if ctx.sequence_parallel_enabled:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None
        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], gradient_accumulation_fusion: bool, async_grad_allreduce: bool, sequence_parallel_enabled: bool) ->torch.Tensor:
    args = _cast_if_autocast_enabled(input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce, sequence_parallel_enabled, False)
    with torch.amp.autocast(enabled=False):
        return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


def linear_with_grad_accumulation_and_async_allreduce_in16bit(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], gradient_accumulation_fusion: bool, async_grad_allreduce: bool, sequence_parallel_enabled: bool) ->torch.Tensor:
    args = _cast_if_autocast_enabled(input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce, sequence_parallel_enabled, True)
    with torch.amp.autocast(enabled=False):
        return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.

    Keyword Arguments:
        no_async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        accumulation_in_fp16:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False, skip_bias_add=False, *, no_async_tensor_model_parallel_allreduce=False, params_dtype=torch.float32, use_cpu_initialization=False, gradient_accumulation_fusion=False, accumulation_in_fp16: bool=False, sequence_parallel_enabled: bool=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(self.weight, self.output_size, self.input_size, self.output_size_per_partition, 0, init_method, stride=stride, return_master_weight=keep_master_weight_for_test, params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, device=torch.cuda.current_device(), dtype=params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(self.output_size_per_partition, device=torch.cuda.current_device(), dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = not no_async_tensor_model_parallel_allreduce and world_size > 1
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f'`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}')
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                warnings.warn('`gradient_accumulation_fusion` is set to `True` but the custom CUDA extension of `fused_weight_gradient_mlp_cuda` module not found. Thus `gradient_accumulation_fusion` set to `False`. Note that the extension requires CUDA>=11.')
                gradient_accumulation_fusion = False
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError('`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time.')
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce_in16bit if accumulation_in_fp16 else linear_with_grad_accumulation_and_async_allreduce

    def forward(self, input_: torch.Tensor) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None
        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        output_parallel = self._forward_impl(input=input_parallel, weight=self.weight, bias=bias, gradient_accumulation_fusion=self.gradient_accumulation_fusion, async_grad_allreduce=self.async_tensor_model_parallel_allreduce, sequence_parallel_enabled=self.sequence_parallel_enabled)
        if self.gather_output:
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def _gather_along_first_dim(input_: torch.Tensor) ->torch.Tensor:
    """Gather tensors and concatenate along the first dimension."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    shape = list(input_.shape)
    shape[0] *= world_size
    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=get_tensor_model_parallel_group())
    return output


def _reduce_scatter_along_first_dim(input_: torch.Tensor) ->torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    shape = list(input_.shape)
    assert shape[0] % world_size == 0
    shape[0] //= world_size
    output = torch.empty(shape, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.reduce_scatter_tensor(output, input_.contiguous(), group=get_tensor_model_parallel_group())
    return output


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


def reduce_scatter_to_sequence_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be sequence, batch, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    Keyword Arguments:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        accumulation_in_fp16:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, init_method=init.xavier_normal_, stride=1, keep_master_weight_for_test=False, skip_bias_add=False, *, params_dtype=torch.float32, use_cpu_initialization=False, gradient_accumulation_fusion=False, accumulation_in_fp16: bool=False, sequence_parallel_enabled: bool=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError('To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`')
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(self.weight, self.output_size, self.input_size, self.input_size_per_partition, 1, init_method, stride=stride, return_master_weight=keep_master_weight_for_test, params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, device=torch.cuda.current_device(), dtype=params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(self.output_size, device=torch.cuda.current_device(), dtype=params_dtype))
            with torch.no_grad():
                self.bias.zero_()
            setattr(self.bias, 'sequence_parallel_enabled', sequence_parallel_enabled)
        else:
            self.register_parameter('bias', None)
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce_in16bit if accumulation_in_fp16 else linear_with_grad_accumulation_and_async_allreduce

    def forward(self, input_: torch.Tensor) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        output_parallel = self._forward_impl(input=input_parallel, weight=self.weight, bias=None, gradient_accumulation_fusion=self.gradient_accumulation_fusion, async_grad_allreduce=False, sequence_parallel_enabled=False)
        if self.sequence_parallel_enabled:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class MyLayer(nn.Module):

    def __init__(self, hidden_size: int, pre_process: bool, post_process: bool):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda', dtype=torch.float32))
        self.weight1 = Parameter(1.0 + unique + torch.arange(2, device='cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return (input * weight0.float() * weight1.float()).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


class ToyParallelMLP(nn.Module):

    def __init__(self, hidden_size: int, pre_process: bool=False, post_process: bool=False, *, sequence_parallel_enabled: bool=False, add_encoder: bool=False, add_decoder: bool=False) ->None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel_enabled = sequence_parallel_enabled
        ffn_hidden_size = 4 * hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, ffn_hidden_size, gather_output=False, skip_bias_add=True, bias=True, sequence_parallel_enabled=sequence_parallel_enabled, no_async_tensor_model_parallel_allreduce=True)
        self.dense_4h_to_h = RowParallelLinear(ffn_hidden_size, hidden_size, input_is_parallel=True, skip_bias_add=False, bias=True, sequence_parallel_enabled=sequence_parallel_enabled)
        self.activation_func = torch.nn.GELU()

    def set_input_tensor(self, input_tensor: Union[torch.Tensor, List[torch.Tensor]]) ->None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.input_tensor = input_tensor[0]

    def forward(self, x: Optional[torch.Tensor]) ->torch.Tensor:
        """Forward of Simplified ParallelMLP.

        Args:
            x: :obj:`None` if pipeline rank != pippeline first rank. When :obj:`None`,
                `self.input_tensor` is taken care of by `forward_step` defined in
                apex/transformer/pipeline_parallel/schedules/common.py
        """
        if self.input_tensor is None:
            input = x
        else:
            input = self.input_tensor
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(input)
        if bias_parallel is not None:
            intermediate_parallel += bias_parallel
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output


class IdentityLayer(torch.nn.Module):

    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


_GLOBAL_ARGS = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support for pipelining."""

    def __init__(self, share_word_embeddings: bool=True) ->None:
        super().__init__()
        self.share_word_embeddings = share_word_embeddings

    def word_embeddings_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last stage, but share_word_embeddings is false')
            return self.word_embeddings.weight

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but share_word_embeddings is false')
        if args.pipeline_model_parallel_size == 1:
            return
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            self.word_embeddings = VocabParallelEmbedding(args.padded_vocab_size, args.hidden_size, init_method=init_method_normal(args.init_method_std))
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and self.pre_process:
            self.language_model.embedding.zero_parameters()
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                torch.distributed.all_reduce(self.word_embeddings_weight(), group=parallel_state.get_embedding_group())
            if parallel_state.is_rank_in_position_embedding_group() and args.pipeline_model_parallel_split_rank is not None:
                self.language_model.embedding
                position_embeddings = self.language_model.embedding.position_embeddings
                torch.distributed.all_reduce(position_embeddings.weight, group=parallel_state.get_position_embedding_group())
        else:
            None


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super().__init__()
        args = get_args()
        self.dense_h_to_4h = ColumnParallelLinear(args.hidden_size, args.ffn_hidden_size, gather_output=False, init_method=init_method, skip_bias_add=True, no_async_tensor_model_parallel_allreduce=not args.async_tensor_model_parallel_allreduce, sequence_parallel_enabled=args.sequence_parallel)
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        self.dense_4h_to_h = RowParallelLinear(args.ffn_hidden_size, args.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True, sequence_parallel_enabled=args.sequence_parallel)

    def forward(self, hidden_states):
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


class CoreAttention(MegatronModule):

    def __init__(self, layer_number, attn_mask_type=AttnMaskType.padding):
        super().__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel
        projection_size = args.kv_channels * args.num_attention_heads
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = apex.transformer.utils.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = apex.transformer.utils.divide(projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = apex.transformer.utils.divide(args.num_attention_heads, world_size)
        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.scale_mask_softmax = FusedScaleMaskSoftmax(self.fp16, self.bf16, self.attn_mask_type, args.masked_softmax_fusion, attention_mask_func, self.attention_softmax_in_fp32, coeff)
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        output_size = query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0)
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        matmul_input_buffer = torch.empty(output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype, device=torch.cuda.current_device())
        matmul_result = torch.baddbmm(matmul_input_buffer, query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2), beta=0.0, alpha=1.0 / self.norm_factor)
        attention_scores = matmul_result.view(*output_size)
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)
        output_size = value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3)
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        context_layer = context_layer.view(*output_size)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method, layer_number, attention_type=AttnType.self_attn, attn_mask_type=AttnMaskType.padding):
        super().__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        projection_size = args.kv_channels * args.num_attention_heads
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = apex.transformer.utils.divide(projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = apex.transformer.utils.divide(args.num_attention_heads, world_size)
        if attention_type == AttnType.self_attn:
            self.query_key_value = ColumnParallelLinear(args.hidden_size, 3 * projection_size, gather_output=False, init_method=init_method, no_async_tensor_model_parallel_allreduce=not args.async_tensor_model_parallel_allreduce, sequence_parallel_enabled=args.sequence_parallel)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = ColumnParallelLinear(args.hidden_size, projection_size, gather_output=False, init_method=init_method, no_async_tensor_model_parallel_allreduce=not args.async_tensor_model_parallel_allreduce, sequence_parallel_enabled=args.sequence_parallel)
            self.key_value = ColumnParallelLinear(args.hidden_size, 2 * projection_size, gather_output=False, init_method=init_method, no_async_tensor_model_parallel_allreduce=not args.async_tensor_model_parallel_allreduce, sequence_parallel_enabled=args.sequence_parallel)
        self.core_attention = CoreAttention(self.layer_number, self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'
        self.dense = RowParallelLinear(projection_size, args.hidden_size, input_is_parallel=True, init_method=output_layer_init_method, skip_bias_add=True, sequence_parallel_enabled=args.sequence_parallel)

    def _checkpointed_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
            return output_
        hidden_states = tensor_parallel.checkpoint(custom_forward, False, query_layer, key_layer, value_layer, attention_mask)
        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(inference_max_sequence_len, batch_size, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head, dtype=self.params_dtype, device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask, encoder_output=None, inference_params=None):
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = inference_key_memory, inference_value_memory
            else:
                inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[self.layer_number]
        if self.attention_type == AttnType.self_attn:
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            query_layer, key_layer, value_layer = tensor_parallel.utils.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            mixed_kv_layer, _ = self.key_value(encoder_output)
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (self.num_attention_heads_per_partition, 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            key_layer, value_layer = tensor_parallel.utils.split_tensor_along_last_dim(mixed_kv_layer, 2)
            query_layer, _ = self.query(hidden_states)
            new_tensor_shape = query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)
        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        output, bias = self.dense(context_layer)
        return output, bias


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


def bias_dropout_add(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) ->torch.Tensor:
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):

    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method, layer_number, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding, drop_path_rate=0.0):
        args = get_args()
        super().__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, sequence_parallel_enabled=args.sequence_parallel)
        self.self_attention = ParallelAttention(init_method, output_layer_init_method, layer_number, attention_type=AttnType.self_attn, attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        assert drop_path_rate <= 0.0
        self.drop_path = None
        self.post_attention_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, sequence_parallel_enabled=args.sequence_parallel)
        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(init_method, output_layer_init_method, layer_number, attention_type=AttnType.cross_attn)
            self.post_inter_attention_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, sequence_parallel_enabled=args.sequence_parallel)
        assert args.num_experts is None
        self.mlp = ParallelMLP(init_method, output_layer_init_method)
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or TORCH_MAJOR == 1 and TORCH_MINOR >= 10
        self.bias_dropout_add_exec_handler = contextlib.nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None):
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias = self.self_attention(layernorm_output, attention_mask, inference_params=inference_params)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        if self.drop_path is None:
            bias_dropout_add_func = get_bias_dropout_add(self.training)
            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias, p=self.hidden_dropout, training=self.training)
            layernorm_input = residual + self.drop_path(out)
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = self.inter_attention(layernorm_output, enc_dec_attn_mask, encoder_output=encoder_output)
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input
            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout)
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias, p=self.hidden_dropout, training=self.training)
            output = residual + self.drop_path(out)
        return output


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None):
        return hidden_states.clone()


def get_num_layers(args, is_encoder_and_decoder_model):
    """Compute the number of transformer layers resident on the current rank."""
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert args.pipeline_model_parallel_split_rank is not None
            num_ranks_in_encoder = args.pipeline_model_parallel_split_rank - 1 if args.standalone_embedding_stage else args.pipeline_model_parallel_split_rank
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert args.num_layers % num_ranks_in_encoder == 0, 'num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.num_layers, num_ranks_in_encoder)
            assert args.num_layers % num_ranks_in_decoder == 0, 'num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.num_layers, num_ranks_in_decoder)
            if parallel_state.is_pipeline_stage_before_split():
                num_layers = 0 if args.standalone_embedding_stage and parallel_state.get_pipeline_model_parallel_rank() == 0 else args.num_layers // num_ranks_in_encoder
            else:
                num_layers = args.num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, 'num_layers must be divisible by transformer_pipeline_model_parallel_size'
            num_layers = 0 if args.standalone_embedding_stage and parallel_state.get_pipeline_model_parallel_rank() == 0 else args.num_layers // args.transformer_pipeline_model_parallel_size
    else:
        num_layers = args.num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding, post_layer_norm=True, pre_process=True, post_process=True, drop_path_rate=0.0):
        super().__init__()
        args = get_args()
        self.layer_type = layer_type
        self.model_type = args.model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = args.distribute_saved_activations and not args.sequence_parallel
        self.sequence_parallel = args.sequence_parallel
        self.num_layers = get_num_layers(args, args.model_type == ModelType.encoder_and_decoder)
        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        def build_layer(layer_number):
            return ParallelTransformerLayer(init_method, output_layer_init_method, layer_number, layer_type=layer_type, self_attn_mask_type=self_attn_mask_type, drop_path_rate=self.drop_path_rates[layer_number - 1])
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, 'num_layers_per_stage must be divisible by virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (args.num_layers // args.virtual_pipeline_model_parallel_size) + parallel_state.get_pipeline_model_parallel_rank() * self.num_layers
        elif args.model_type == ModelType.encoder_and_decoder and parallel_state.get_pipeline_model_parallel_world_size() > 1:
            pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
            if layer_type == LayerType.encoder:
                offset = pipeline_rank * self.num_layers
            else:
                num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
        else:
            offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers
        if self.num_layers == 0:
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])
        if self.post_process and self.post_layer_norm:
            self.final_layernorm = LayerNorm(args.hidden_size, eps=args.layernorm_epsilon, sequence_parallel_enabled=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask, encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):

            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward
        if self.recompute_method == 'uniform':
            l = 0
            while l < self.num_layers:
                hidden_states = tensor_parallel.random.checkpoint(custom(l, l + self.recompute_num_layers), self.distribute_saved_activations, hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                l += self.recompute_num_layers
        elif self.recompute_method == 'block':
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    hidden_states = tensor_parallel.random.checkpoint(custom(l, l + 1), self.distribute_saved_activations, hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = custom(l, l + 1)(hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError('Invalid activation recompute method.')
        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None):
        if inference_params:
            assert self.recompute_granularity is None, 'inference does not work with activation checkpointing'
        if not self.pre_process:
            hidden_states = self.input_tensor
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = contextlib.nullcontext()
        with rng_context:
            if self.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
            else:
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(hidden_states, attention_mask, encoder_output=encoder_output, enc_dec_attn_mask=enc_dec_attn_mask, inference_params=inference_params)
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super().__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states, sequence_index=0):
        if self.sequence_parallel:
            hidden_states = tensor_parallel.mappings.gather_from_sequence_parallel_region(hidden_states)
        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


def _split_along_first_dim(input_: torch.Tensor) ->torch.Tensor:
    """Split the tensor along its first dimension and keep the corresponding slice."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = input_.size(0)
    assert dim_size % world_size == 0
    local_dim_size = dim_size // world_size
    dim_offset = get_tensor_model_parallel_rank() * local_dim_size
    output = input_[dim_offset:dim_offset + local_dim_size].contiguous()
    return output


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


def scatter_to_sequence_parallel_region(input_: torch.Tensor) ->torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_)


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, hidden_size, vocab_size, max_sequence_length, embedding_dropout_prob, init_method, num_tokentypes=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        args = get_args()
        self.word_embeddings = VocabParallelEmbedding(vocab_size, self.hidden_size, init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        self.init_method(self.position_embeddings.weight)
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None
        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            None
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None
        embeddings = embeddings.transpose(0, 1).contiguous()
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        if self.sequence_parallel:
            embeddings = scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)
        return embeddings


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, init_method, output_layer_init_method, encoder_attn_mask_type, num_tokentypes=0, add_encoder=True, add_decoder=False, decoder_attn_mask_type=AttnMaskType.causal, add_pooler=False, pre_process=True, post_process=True):
        super().__init__()
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size, args.padded_vocab_size, args.max_position_embeddings, args.hidden_dropout, self.init_method, self.num_tokentypes)
            self._embedding_key = 'embedding'
        if self.add_encoder:
            self.encoder = ParallelTransformer(self.init_method, output_layer_init_method, self_attn_mask_type=self.encoder_attn_mask_type, pre_process=self.pre_process, post_process=self.post_process)
            self._encoder_key = 'encoder'
        else:
            self.encoder = None
        if self.add_decoder:
            self.decoder = ParallelTransformer(self.init_method, output_layer_init_method, layer_type=LayerType.decoder, self_attn_mask_type=self.decoder_attn_mask_type, pre_process=self.pre_process, post_process=self.post_process)
            self._decoder_key = 'decoder'
        else:
            self.decoder = None
        if self.post_process:
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, 'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, 'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask, dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None, enc_dec_attn_mask=None, tokentype_ids=None, inference_params=None, pooling_sequence_index=0, enc_hidden_states=None, output_enc_hidden=False):
        args = get_args()
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids)
        else:
            encoder_input = None
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(encoder_input, enc_attn_mask, inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states
        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids, dec_position_ids)
        else:
            decoder_input = None
        decoder_output = self.decoder(decoder_input, dec_attn_mask, encoder_output=encoder_output, enc_dec_attn_mask=enc_dec_attn_mask, inference_params=inference_params)
        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class Generator(nn.Module):

    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class WhitelistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(WhitelistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(8 * 8, device='cuda', dtype=dtype).view(8, 8))

    @staticmethod
    def ops(input, weight):
        return input.mm(weight).mm(weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class BlacklistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(BlacklistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda', dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input + torch.pow(weight, 2) + torch.pow(weight, 2)).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class PromoteModule(torch.nn.Module):

    def __init__(self, dtype):
        super(PromoteModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda', dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input * weight * weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class DummyBlock(nn.Module):

    def __init__(self):
        super(DummyBlock, self).__init__()
        self.conv = nn.Conv2d(10, 10, 2)
        self.bn = nn.BatchNorm2d(10, affine=True)

    def forward(self, x):
        return self.conv(self.bn(x))


class DummyNet(nn.Module):

    def __init__(self):
        super(DummyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 2)
        self.bn1 = nn.BatchNorm2d(10, affine=False)
        self.db1 = DummyBlock()
        self.db2 = DummyBlock()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.db2(out)
        return out


class DummyNetWrapper(nn.Module):

    def __init__(self):
        super(DummyNetWrapper, self).__init__()
        self.bn = nn.BatchNorm2d(3, affine=True)
        self.dn = DummyNet()

    def forward(self, x):
        return self.dn(self.bn(x))


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.a = Parameter(torch.FloatTensor(4096 * 4096).fill_(1.0))
        self.b = Parameter(torch.FloatTensor(4096 * 4096).fill_(2.0))

    def forward(self, input):
        return input * self.a * self.b


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNModelRef,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (IdentityLayer,
     lambda: ([], {'size': 4}),
     lambda: ([], {}),
     True),
    (Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 16777216])], {}),
     True),
    (MyLayer,
     lambda: ([], {'hidden_size': 4, 'pre_process': 4, 'post_process': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoopTransformerLayer,
     lambda: ([], {'layer_number': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimpleModel,
     lambda: ([], {'num_layers': 1, 'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (mLSTMRNNCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIA_apex(_paritybench_base):
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

