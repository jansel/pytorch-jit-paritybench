import sys
_module = sys.modules[__name__]
del sys
RNNBackend = _module
RNN = _module
cells = _module
models = _module
apex = _module
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
func_test_multihead_attn = _module
perf_test_multihead_attn = _module
groupbn = _module
batch_norm = _module
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
distributed_fused_adam_v2 = _module
distributed_fused_adam_v3 = _module
distributed_fused_lamb = _module
fp16_optimizer = _module
fused_adam = _module
fused_lamb = _module
fused_sgd = _module
sparsity = _module
asp = _module
sparse_masklib = _module
checkpointing_test_part1 = _module
checkpointing_test_part2 = _module
checkpointing_test_reference = _module
toy_problem = _module
test_encdec_multihead_attn = _module
test_encdec_multihead_attn_norm_add = _module
test_mha_fused_softmax = _module
test_self_multihead_attn = _module
test_self_multihead_attn_norm_add = _module
test_label_smoothing = _module
xentropy = _module
softmax_xentropy = _module
fp16_utils = _module
fp16_optimizer = _module
fp16util = _module
loss_scaler = _module
mlp = _module
mlp = _module
multi_tensor_apply = _module
normalization = _module
fused_layer_norm = _module
fused_adagrad = _module
fused_novograd = _module
LARC = _module
parallel = _module
distributed = _module
multiproc = _module
optimized_sync_batchnorm = _module
optimized_sync_batchnorm_kernel = _module
sync_batchnorm = _module
sync_batchnorm_kernel = _module
pyprof = _module
fused_adam = _module
custom_function = _module
custom_module = _module
imagenet = _module
jit_script_function = _module
jit_script_method = _module
jit_trace_function = _module
jit_trace_method = _module
lenet = _module
operators = _module
simple = _module
resnet = _module
nvtx = _module
nvmarker = _module
parse = _module
__main__ = _module
db = _module
kernel = _module
nvvp = _module
prof = _module
activation = _module
base = _module
blas = _module
conv = _module
convert = _module
data = _module
dropout = _module
embedding = _module
index_slice_join_mutate = _module
linear = _module
loss = _module
misc = _module
optim = _module
output = _module
pointwise = _module
pooling = _module
prof = _module
randomSample = _module
recurrentCell = _module
reduction = _module
softmax = _module
usage = _module
utility = _module
reparameterization = _module
reparameterization = _module
weight_norm = _module
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
run_fp16util = _module
test_fp16util = _module
test_fused_layer_norm = _module
test_mlp = _module
run_optimizers = _module
test_adagrad = _module
test_adam = _module
run_pyprof_data = _module
test_pyprof_data = _module
run_pyprof_nvtx = _module
test_pyprof_nvtx = _module
run_test = _module
compare = _module
main_amp = _module
ddp_race_condition_test = _module
amp_master_params = _module
python_single_gpu_unit_test = _module
single_gpu_unit_test = _module
test_batchnorm1d = _module
test_groups = _module
two_gpu_unit_test = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import math


from torch._six import string_classes


import functools


import numpy as np


from types import MethodType


import warnings


import itertools


from collections import OrderedDict


import torch.nn.functional


from torch.nn.modules.batchnorm import _BatchNorm


from torch import nn


from torch.nn import Parameter


import types


import random


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from copy import copy


from torch.nn import init


from torch.nn import functional as F


import torch.distributed as dist


from torch.nn.modules import Module


from itertools import chain


import copy


import torch.cuda.profiler as profiler


import torch.optim as optim


import torch.cuda.nvtx as nvtx


import numpy


import inspect as ins


from abc import ABC


from abc import abstractmethod


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.utils.data


import torch.optim


import torch.utils.data.distributed


import functools as ft


import itertools as it


from math import floor


import inspect


from torch.nn import Module


class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN
    """

    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout=dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers,
            dropout=dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])

    def forward(self, input, collect_hidden=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        fwd_out, fwd_hiddens = list(self.fwd(input, collect_hidden=
            collect_hidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True,
            collect_hidden=collect_hidden))
        output = torch.cat([fwd_out, bckwrd_out], -1)
        hiddens = tuple(torch.cat(hidden, -1) for hidden in zip(fwd_hiddens,
            bckwrd_hiddens))
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


def hasTileSize(name):
    if 'sgemm' in name or '884gemm' in name or 'hgemm' in name:
        return True
    else:
        return False


def ctaTile(name):
    name = name.split('_')
    name = list(filter(lambda x: 'x' in x, name))
    name = list(filter(lambda x: 'slice' not in x, name))
    assert len(name) == 1
    name = name[0].split('x')
    assert len(name) == 2
    name = list(map(int, name))
    return name[0], name[1]


class OperatorLayerBase(ABC):
    """
	Base class for all layers and operators.
	Every derived class should have the following functions.
	"""

    @abstractmethod
    def tc(self):
        """
		Tensor core usage by the kernel.
		Return "1" (yes), "0" (no, but possible), "-" (not applicable)
		"""
        pass

    @abstractmethod
    def params(self):
        """
		Kernel parameters to be printed.
		"""
        pass

    @abstractmethod
    def flops(self):
        """
		Note that 1 FMA = 2 flops.
		"""
        pass

    @abstractmethod
    def bytes(self):
        pass

    @abstractmethod
    def mod(self):
        """
		Name of the module/class e.g. torch.nn.functional.
		"""
        pass

    @abstractmethod
    def op(self):
        """
		Name of the operator e.g. sigmoid.
		"""
        pass


class Utility(object):

    @staticmethod
    def numElems(shape):
        assert type(shape) == tuple
        return reduce(lambda x, y: x * y, shape, 1)

    @staticmethod
    def typeToBytes(t):
        if t in ['uint8', 'int8', 'byte', 'char', 'bool']:
            return 1
        elif t in ['float16', 'half', 'int16', 'short']:
            return 2
        elif t in ['float32', 'float', 'int32', 'int']:
            return 4
        elif t in ['int64', 'long', 'float64', 'double']:
            return 8
        assert False

    @staticmethod
    def typeToString(t):
        if t in ['uint8', 'byte', 'char']:
            return 'uint8'
        elif t in ['int8']:
            return 'int8'
        elif t in ['int16', 'short']:
            return 'int16'
        elif t in ['float16', 'half']:
            return 'fp16'
        elif t in ['float32', 'float']:
            return 'fp32'
        elif t in ['int32', 'int']:
            return 'int32'
        elif t in ['int64', 'long']:
            return 'int64'
        elif t in ['float64', 'double']:
            return 'fp64'
        elif t in ['bool']:
            return 'bool'
        assert False

    @staticmethod
    def hasNVTX(marker):
        if type(marker) is str:
            try:
                marker = eval(marker)
            except:
                return False
        if type(marker) is dict:
            keys = marker.keys()
            return 'mod' in keys and 'op' in keys and 'args' in keys
        else:
            return False

    @staticmethod
    def isscalar(t):
        return t in ['float', 'int']


class RNNCell(OperatorLayerBase):
    """
	This class supports RNNCell, LSTMCell and GRUCell.
	"""

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']
        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args
        self.name = d.name
        self.dir = d.dir
        self.sub = d.sub
        self.grid = d.grid
        assert op == 'forward'
        assert mod in ['LSTMCell', 'GRUCell', 'RNNCell']
        assert len(args) in [2, 3]
        x, h = args[0], args[1]
        b1, ii = x['shape']
        b2, hh = h['shape']
        assert b1 == b2
        assert x['dtype'] == h['dtype']
        t = x['dtype']
        self.cell = mod
        self.inp = ii
        self.hid = hh
        self.b = b1
        self.type = t
        self.multiple = 1
        if self.cell == 'LSTMCell':
            self.multiple = 4
        elif self.cell == 'GRUCell':
            self.multiple = 3
        self.gemm = None
        self.m = None
        self.n = None
        self.k = None
        self.elems = 0
        self.bar()

    def params(self):
        if self.gemm is None:
            p = OrderedDict([('cell', self.cell), ('X', self.inp), ('H',
                self.hid), ('B', self.b), ('type', self.type)])
        else:
            assert self.m is not None
            assert self.n is not None
            assert self.k is not None
            p = OrderedDict([('gemm', self.gemm), ('M', self.m), ('N', self
                .n), ('K', self.k), ('type', self.type)])
        return p

    def tc(self):
        if 'gemm' in self.name:
            return 1 if '884gemm' in self.name else 0
        else:
            return '-'

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        if self.gemm is not None:
            m, n, k, t = self.m, self.n, self.k, self.type
            b = (m * k + k * n + m * n) * Utility.typeToBytes(t)
        elif self.elems != 0:
            b = self.elems * Utility.typeToBytes(self.type)
        else:
            b = 0
        return b

    def flops(self):
        if self.gemm is not None:
            m, n, k = self.m, self.n, self.k
            f = 2 * m * n * k
        elif self.elems != 0:
            f = 0
        else:
            f = 0
        return f

    def bar(self):
        cell = self.cell
        X = self.inp
        H = self.hid
        B = self.b
        t = self.type
        subseqId = self.sub
        direc = self.dir
        name = self.name
        grid = self.grid
        multiple = self.multiple
        if direc == 'fprop':
            subseqId = subseqId % 3
            if subseqId == 0:
                self.gemm = 'layer'
                self.m = multiple * H
                self.n = B
                self.k = X
            elif subseqId == 1:
                self.gemm = 'recur'
                self.m = multiple * H
                self.n = B
                self.k = H
            else:
                layerGemmElems = multiple * H * B
                recurGemmElems = multiple * H * B
                cElems = H * B
                hElems = H * B
                totElems = (layerGemmElems + recurGemmElems + 2 * cElems +
                    hElems)
                self.elems = totElems
        elif 'gemm' in name and hasTileSize(name):
            tileX, tileY = ctaTile(name)
            grid = grid.split(',')
            gridX, gridY, gridZ = map(lambda x: int(x), grid)
            gemmM = tileX * gridX
            gemmN = tileY * gridY
            if name[-3:] == '_nn':
                if gemmM == H:
                    gemmN = B
                    gemmK = multiple * H
                    self.gemm = 'recur'
                    self.m = gemmM
                    self.n = gemmN
                    self.k = gemmK
                elif gemmM == X:
                    gemmK = multiple * H
                    self.gemm = 'layer'
                    self.m = gemmM
                    self.n = gemmN
                    self.k = gemmK
                else:
                    pass
            elif name[-3:] == '_nt':
                if gemmM == H:
                    assert gemmN == multiple * H
                    gemmK = B
                    self.gemm = 'recur'
                    self.m = gemmM
                    self.n = gemmN
                    self.k = gemmK
                elif gemmM == X:
                    assert gemmN == multiple * H
                    gemmK = B
                    self.gemm = 'layer'
                    self.m = gemmM
                    self.n = gemmN
                    self.k = gemmK
                else:
                    pass
            else:
                pass
        else:
            pass
        return


def is_iterable(maybe_iterable):
    return isinstance(maybe_iterable, list) or isinstance(maybe_iterable, tuple
        )


def flatten_list(tens_list):
    """
    flatten_list
    """
    if not is_iterable(tens_list):
        return tens_list
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].
        size())


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
            assert len(inputRNN
                ) == num_layers, 'RNN list length must be equal to num_layers'
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
        new_hidden = [[[None for k in range(self.nLayers)] for j in range(
            seq_len)] for i in range(n_hid)]
        for i in range(n_hid):
            for j in range(seq_len):
                for k in range(self.nLayers):
                    new_hidden[i][j][k] = hidden_states[k][j][i]
        hidden_states = new_hidden
        if reverse:
            hidden_states = list(list(reversed(list(entry))) for entry in
                hidden_states)
        hiddens = list(list(flatten_list(seq) for seq in hidden) for hidden in
            hidden_states)
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

    def __init__(self, gate_multiplier, input_size, hidden_size, cell,
        n_hidden_states=2, bias=False, output_size=None):
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
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size)
            )
        if self.output_size != self.hidden_size:
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.
                hidden_size))
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
        return type(self)(self.gate_multiplier, new_input_size, self.
            hidden_size, self.cell, self.n_hidden_states, self.bias, self.
            output_size)

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
                raise RuntimeError(
                    'Must initialize hidden state before you can detach it')
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()

    def forward(self, input):
        """
        forward()
        if not inited or bsz has changed this will create hidden states
        """
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0
            ] if self.n_hidden_states == 1 else self.hidden
        self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh,
            b_ih=self.b_ih, b_hh=self.b_hh)
        if self.n_hidden_states > 1:
            self.hidden = list(self.hidden)
        else:
            self.hidden = [self.hidden]
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)


class bn_addrelu_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z, s, b, rm, riv, mini_m, mini_riv, grid_dim_y,
        ret_cta, mom, epsilon, is_train, bn_group, my_data, pair_data,
        magic, pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup,
        bwd_grid_x, multi_stream):
        if is_train:
            bitmask = torch.cuda.IntTensor((x.numel() + 31) // 32 * 2 *
                grid_dim_y)
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
            res = bnp.bn_addrelu_fwd_nhwc(x, z, s, b, rm, riv, mini_m,
                mini_riv, bitmask, ret_cta, mom, epsilon, my_data,
                pair_data, pair_data2, pair_data3, bn_group, magic,
                fwd_occup, fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_addrelu_fwd_eval_nhwc(x, z, s, b, rm, riv,
                ret_cta, bn_group, mom, epsilon)

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
        dx, dz, dscale, dbias = bnp.bn_addrelu_bwd_nhwc(x, grad_y, s, b, rm,
            riv, mini_m, mini_riv, bitmask, ret_cta, mom, epsilon, my_data,
            pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup,
            bwd_grid_x, multi_stream)
        return (dx, dz, dscale, dbias, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None)


class bn_NHWC_impl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, b, rm, riv, mini_m, mini_riv, ret_cta, mom,
        epsilon, fuse_relu, is_train, bn_group, my_data, pair_data, magic,
        pair_data2, pair_data3, fwd_occup, fwd_grid_x, bwd_occup,
        bwd_grid_x, multi_stream):
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
            res = bnp.bn_fwd_nhwc(x, s, b, rm, riv, mini_m, mini_riv,
                ret_cta, mom, epsilon, fuse_relu, my_data, pair_data,
                pair_data2, pair_data3, bn_group, magic, fwd_occup,
                fwd_grid_x, multi_stream)
            return res
        else:
            return bnp.bn_fwd_eval_nhwc(x, s, b, rm, riv, ret_cta, bn_group,
                mom, epsilon, fuse_relu)

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
        dx, dscale, dbias = bnp.bn_bwd_nhwc(x, grad_y, s, b, rm, riv,
            mini_m, mini_riv, ret_cta, mom, epsilon, fuse_relu, my_data,
            pair_data, pair_data2, pair_data3, bn_group, magic, bwd_occup,
            bwd_grid_x, multi_stream)
        return (dx, dscale, dbias, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None)


class BatchNorm2d_NHWC(_BatchNorm):

    def __init__(self, num_features, fuse_relu=False, bn_group=1,
        max_cta_per_sm=2, cta_launch_margin=12, multi_stream=False):
        super(BatchNorm2d_NHWC, self).__init__(num_features)
        self.fuse_relu = fuse_relu
        self.multi_stream = multi_stream
        self.minibatch_mean = torch.cuda.FloatTensor(num_features)
        self.minibatch_riv = torch.cuda.FloatTensor(num_features)
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
        self.addrelu_fwd_occupancy = min(bnp.bn_addrelu_fwd_nhwc_occupancy(
            ), max_cta_per_sm)
        self.addrelu_bwd_occupancy = min(bnp.bn_addrelu_bwd_nhwc_occupancy(
            ), max_cta_per_sm)
        mp_count = torch.cuda.get_device_properties(None).multi_processor_count
        self.fwd_grid_dim_x = max(mp_count * self.fwd_occupancy -
            cta_launch_margin, 1)
        self.bwd_grid_dim_x = max(mp_count * self.bwd_occupancy -
            cta_launch_margin, 1)
        self.addrelu_fwd_grid_dim_x = max(mp_count * self.
            addrelu_fwd_occupancy - cta_launch_margin, 1)
        self.addrelu_bwd_grid_dim_x = max(mp_count * self.
            addrelu_bwd_occupancy - cta_launch_margin, 1)
        self.grid_dim_y = (num_features + 63) // 64
        self.ret_cta = torch.cuda.ByteTensor(8192).fill_(0)
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
            self.ipc_buffer = torch.cuda.ByteTensor(bnp.get_buffer_size(
                bn_sync_steps))
            self.my_data = bnp.get_data_ptr(self.ipc_buffer)
            self.storage = self.ipc_buffer.storage()
            self.share_cuda = self.storage._share_cuda_()
            internal_cuda_mem = self.share_cuda
            my_handle = torch.cuda.ByteTensor(np.frombuffer(
                internal_cuda_mem[1], dtype=np.uint8))
            my_offset = torch.cuda.IntTensor([internal_cuda_mem[3]])
            handles_all = torch.empty(world_size, my_handle.size(0), dtype=
                my_handle.dtype, device=my_handle.device)
            handles_l = list(handles_all.unbind(0))
            torch.distributed.all_gather(handles_l, my_handle)
            offsets_all = torch.empty(world_size, my_offset.size(0), dtype=
                my_offset.dtype, device=my_offset.device)
            offsets_l = list(offsets_all.unbind(0))
            torch.distributed.all_gather(offsets_l, my_offset)
            self.pair_handle = handles_l[local_rank ^ 1].cpu().contiguous()
            pair_offset = offsets_l[local_rank ^ 1].cpu()
            self.pair_data = bnp.get_remote_data_ptr(self.pair_handle,
                pair_offset)
            if bn_group > 2:
                self.pair_handle2 = handles_l[local_rank ^ 2].cpu().contiguous(
                    )
                pair_offset2 = offsets_l[local_rank ^ 2].cpu()
                self.pair_data2 = bnp.get_remote_data_ptr(self.pair_handle2,
                    pair_offset2)
            if bn_group > 4:
                self.pair_handle3 = handles_l[local_rank ^ 4].cpu().contiguous(
                    )
                pair_offset3 = offsets_l[local_rank ^ 4].cpu()
                self.pair_data3 = bnp.get_remote_data_ptr(self.pair_handle3,
                    pair_offset3)
            self.magic = torch.IntTensor([2])
            self.local_rank = local_rank

    def forward(self, x, z=None):
        if z is not None:
            assert self.fuse_relu == True
            return bn_addrelu_NHWC_impl.apply(x, z, self.weight, self.bias,
                self.running_mean, self.running_var, self.minibatch_mean,
                self.minibatch_riv, self.grid_dim_y, self.ret_cta, self.
                momentum, self.eps, self.training, self.bn_group, self.
                my_data, self.pair_data, self.magic, self.pair_data2, self.
                pair_data3, self.addrelu_fwd_occupancy, self.
                addrelu_fwd_grid_dim_x, self.addrelu_bwd_occupancy, self.
                addrelu_bwd_grid_dim_x, self.multi_stream)
        else:
            return bn_NHWC_impl.apply(x, self.weight, self.bias, self.
                running_mean, self.running_var, self.minibatch_mean, self.
                minibatch_riv, self.ret_cta, self.momentum, self.eps, self.
                fuse_relu, self.training, self.bn_group, self.my_data, self
                .pair_data, self.magic, self.pair_data2, self.pair_data3,
                self.fwd_occupancy, self.fwd_grid_dim_x, self.bwd_occupancy,
                self.bwd_grid_dim_x, self.multi_stream)

    def __del__(self):
        if self.bn_group > 1:
            bnp.close_remote_data(self.pair_handle)
            if self.bn_group > 2:
                bnp.close_remote_data(self.pair_handle2)
                if self.bn_group > 4:
                    bnp.close_remote_data(self.pair_handle3)


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    out = F.dropout(x, p=prob, training=True)
    out = residual + out
    return out


class EncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs_q,
        inputs_kv, input_weights_q, input_weights_kv, output_weights,
        input_biases_q, input_biases_kv, output_biases, mask, dropout_prob):
        use_biases_t = torch.tensor([input_biases_q is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs_q.size(2) // heads
        if use_biases_t[0]:
            input_lin_q_results = torch.addmm(input_biases_q, inputs_q.view
                (inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                input_weights_q.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_q_results = torch.mm(inputs_q.view(inputs_q.size(0) *
                inputs_q.size(1), inputs_q.size(2)), input_weights_q.
                transpose(0, 1))
        input_lin_q_results = input_lin_q_results.view(inputs_q.size(0),
            inputs_q.size(1), input_weights_q.size(0))
        if use_biases_t[0]:
            input_lin_kv_results = torch.addmm(input_biases_kv, inputs_kv.
                view(inputs_kv.size(0) * inputs_kv.size(1), inputs_kv.size(
                2)), input_weights_kv.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_kv_results = torch.mm(inputs_kv.view(inputs_kv.size(0
                ) * inputs_kv.size(1), inputs_kv.size(2)), input_weights_kv
                .transpose(0, 1))
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0),
            inputs_kv.size(1), input_weights_kv.size(0))
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(
            1) * heads, head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0),
            inputs_kv.size(1) * heads, 2, head_dim)
        keys = input_lin_kv_results[:, :, (0), :]
        values = input_lin_kv_results[:, :, (1), :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0),
            keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(
            0, 1), keys.transpose(0, 1).transpose(1, 2), out=
            matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1
                    ), 'Sequence length should match!'
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask, float(
                    '-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q,
                    seql_k)
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask.
                    unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q,
                    seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(
                softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1),
            dropout_results.size(0), values.size(2)), dtype=dropout_results
            .dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1),
            out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(
            inputs_q.size(0), inputs_q.size(1), inputs_q.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(
                inputs_q.size(0) * inputs_q.size(1), inputs_q.size(2)),
                output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs_q.size(0) *
                inputs_q.size(1), inputs_q.size(2)), output_weights.
                transpose(0, 1))
        outputs = outputs.view(inputs_q.size(0), inputs_q.size(1),
            output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t,
            matmul2_results, dropout_results, softmax_results,
            input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv,
            input_weights_q, input_weights_kv, output_weights, dropout_mask,
            dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (use_biases_t, heads_t, scale_t, matmul2_results, dropout_results,
            softmax_results, input_lin_q_results, input_lin_kv_results,
            inputs_q, inputs_kv, input_weights_q, input_weights_kv,
            output_weights, dropout_mask, dropout_prob_t) = ctx.saved_tensors
        head_dim = inputs_q.size(2) // heads_t[0]
        queries = input_lin_q_results.view(inputs_q.size(0), inputs_q.size(
            1) * heads_t[0], head_dim)
        input_lin_kv_results = input_lin_kv_results.view(inputs_kv.size(0),
            inputs_kv.size(1) * heads_t[0], 2, head_dim)
        keys = input_lin_kv_results[:, :, (0), :]
        values = input_lin_kv_results[:, :, (1), :]
        input_lin_kv_results_grads = torch.empty_like(input_lin_kv_results)
        queries_grads = torch.empty_like(queries)
        keys_grads = input_lin_kv_results_grads[:, :, (0), :]
        values_grads = input_lin_kv_results_grads[:, :, (1), :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) *
            output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0),
            output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(
            0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1
            ), matmul2_results.view(matmul2_results.size(0) *
            matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(output_grads.size(0), 
            output_grads.size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.
                size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1)
            .transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2),
            output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 
            1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads,
            softmax_results, -1, softmax_results)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1),
            softmax_grads, keys.transpose(0, 1), out=queries_grads.
            transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1),
            softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=
            keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        queries_grads = queries_grads.transpose(0, 1).view(inputs_q.size(0) *
            inputs_q.size(1), heads_t[0] * head_dim)
        input_q_grads = torch.mm(queries_grads, input_weights_q)
        input_q_grads = input_q_grads.view(inputs_q.size(0), inputs_q.size(
            1), inputs_q.size(2))
        input_lin_kv_results_grads = input_lin_kv_results_grads.view(
            inputs_kv.size(0) * inputs_kv.size(1), heads_t[0] * 2 * head_dim)
        input_kv_grads = torch.mm(input_lin_kv_results_grads, input_weights_kv)
        input_kv_grads = input_kv_grads.view(inputs_kv.size(0), inputs_kv.
            size(1), inputs_kv.size(2))
        input_weight_q_grads = torch.mm(queries_grads.transpose(0, 1),
            inputs_q.view(inputs_q.size(0) * inputs_q.size(1), inputs_q.
            size(2)))
        input_weight_kv_grads = torch.mm(input_lin_kv_results_grads.
            transpose(0, 1), inputs_kv.view(inputs_kv.size(0) * inputs_kv.
            size(1), inputs_kv.size(2)))
        if use_biases_t[0]:
            input_bias_grads_q = torch.sum(queries_grads, 0)
            input_bias_grads_kv = torch.sum(input_lin_kv_results_grads, 0)
        else:
            input_bias_grads_q = None
            input_bias_grads_kv = None
        return (None, None, None, None, input_q_grads, input_kv_grads,
            input_weight_q_grads, input_weight_kv_grads,
            output_weight_grads, input_bias_grads_q, input_bias_grads_kv,
            output_bias_grads, None, None)


encdec_attn_func = EncdecAttnFunc.apply


class FastEncdecAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv,
        lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q,
        input_weights_kv, output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        (lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_q_results,
            input_lin_kv_results, softmax_results, dropout_results,
            dropout_mask, matmul2_results, dropout_add_mask, outputs) = (
            fast_encdec_multihead_attn_norm_add.forward(use_mask,
            use_time_mask, is_training, heads, inputs_q, inputs_kv,
            lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q,
            input_weights_kv, output_weights, pad_mask if use_mask else
            null_tensor, dropout_prob))
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results,
            softmax_results, input_lin_q_results, input_lin_kv_results,
            lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q,
            inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights,
            input_weights_q, input_weights_kv, output_weights, dropout_mask,
            dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (heads_t, matmul2_results, dropout_results, softmax_results,
            input_lin_q_results, input_lin_kv_results, lyr_nrm_results,
            lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv,
            lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q,
            input_weights_kv, output_weights, dropout_mask,
            dropout_add_mask, dropout_prob_t) = ctx.saved_tensors
        (input_q_grads, input_kv_grads, lyr_nrm_gamma_grads,
            lyr_nrm_beta_grads, input_weight_q_grads, input_weight_kv_grads,
            output_weight_grads) = (fast_encdec_multihead_attn_norm_add.
            backward(heads_t[0], output_grads, matmul2_results,
            dropout_results, softmax_results, input_lin_q_results,
            input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean,
            lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights, input_weights_q, input_weights_kv,
            output_weights, dropout_mask, dropout_add_mask, dropout_prob_t[0]))
        return (None, None, None, input_q_grads, input_kv_grads,
            lyr_nrm_gamma_grads, lyr_nrm_beta_grads, input_weight_q_grads,
            input_weight_kv_grads, output_weight_grads, None, None)


fast_encdec_attn_norm_add_func = FastEncdecAttnNormAddFunc.apply


class FastEncdecAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs_q, inputs_kv,
        input_weights_q, input_weights_kv, output_weights, pad_mask,
        dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        (input_lin_q_results, input_lin_kv_results, softmax_results,
            dropout_results, dropout_mask, matmul2_results, outputs) = (
            fast_encdec_multihead_attn.forward(use_mask, use_time_mask,
            is_training, heads, inputs_q, inputs_kv, input_weights_q,
            input_weights_kv, output_weights, pad_mask if use_mask else
            null_tensor, dropout_prob))
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results,
            softmax_results, input_lin_q_results, input_lin_kv_results,
            inputs_q, inputs_kv, input_weights_q, input_weights_kv,
            output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (heads_t, matmul2_results, dropout_results, softmax_results,
            input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv,
            input_weights_q, input_weights_kv, output_weights, dropout_mask,
            dropout_prob_t) = ctx.saved_tensors
        (input_q_grads, input_kv_grads, input_weight_q_grads,
            input_weight_kv_grads, output_weight_grads) = (
            fast_encdec_multihead_attn.backward(heads_t[0], output_grads,
            matmul2_results, dropout_results, softmax_results,
            input_lin_q_results, input_lin_kv_results, inputs_q, inputs_kv,
            input_weights_q, input_weights_kv, output_weights, dropout_mask,
            dropout_prob_t[0]))
        return (None, None, None, input_q_grads, input_kv_grads,
            input_weight_q_grads, input_weight_kv_grads,
            output_weight_grads, None, None)


fast_encdec_attn_func = FastEncdecAttnFunc.apply


class EncdecMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False,
        include_norm_add=False, impl='fast'):
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
        self.in_proj_weight_kv = Parameter(torch.Tensor(2 * embed_dim,
            embed_dim))
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
        nn.init.xavier_uniform_(self.in_proj_weight_kv)
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

    def forward(self, query, key, value, key_padding_mask=None,
        need_weights=False, attn_mask=None, is_training=True):
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
                outputs = self.attn_func(attn_mask is not None, is_training,
                    self.num_heads, query, key, self.lyr_nrm_gamma_weights,
                    self.lyr_nrm_beta_weights, self.in_proj_weight_q, self.
                    in_proj_weight_kv, self.out_proj_weight, mask, self.dropout
                    )
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training,
                    self.num_heads, self.scaling, lyr_nrm_results, key,
                    self.in_proj_weight_q, self.in_proj_weight_kv, self.
                    out_proj_weight, self.in_proj_bias_q, self.
                    in_proj_bias_kv, self.out_proj_bias, mask, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout,
                        is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training,
                self.num_heads, query, key, self.in_proj_weight_q, self.
                in_proj_weight_kv, self.out_proj_weight, mask, self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training,
                self.num_heads, self.scaling, query, key, self.
                in_proj_weight_q, self.in_proj_weight_kv, self.
                out_proj_weight, self.in_proj_bias_q, self.in_proj_bias_kv,
                self.out_proj_bias, mask, self.dropout)
        return outputs, None


class FastSelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs,
        input_weights, output_weights, input_biases, output_biases,
        pad_mask, mask_additive, dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        if use_biases_t[0]:
            if not mask_additive:
                (input_lin_results, softmax_results, dropout_results,
                    dropout_mask, matmul2_results, outputs) = (
                    fast_self_multihead_attn_bias.forward(use_mask,
                    use_time_mask, is_training, heads, inputs,
                    input_weights, output_weights, input_biases,
                    output_biases, pad_mask if use_mask else null_tensor,
                    dropout_prob))
            else:
                (input_lin_results, softmax_results, dropout_results,
                    dropout_mask, matmul2_results, outputs) = (
                    fast_self_multihead_attn_bias_additive_mask.forward(
                    use_mask, use_time_mask, is_training, heads, inputs,
                    input_weights, output_weights, input_biases,
                    output_biases, pad_mask if use_mask else null_tensor,
                    dropout_prob))
        else:
            (input_lin_results, softmax_results, dropout_results,
                dropout_mask, matmul2_results, outputs) = (
                fast_self_multihead_attn.forward(use_mask, use_time_mask,
                is_training, heads, inputs, input_weights, output_weights, 
                pad_mask if use_mask else null_tensor, dropout_prob))
        ctx.save_for_backward(use_biases_t, heads_t, matmul2_results,
            dropout_results, softmax_results, input_lin_results, inputs,
            input_weights, output_weights, dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (use_biases_t, heads_t, matmul2_results, dropout_results,
            softmax_results, input_lin_results, inputs, input_weights,
            output_weights, dropout_mask, dropout_prob_t) = ctx.saved_tensors
        if use_biases_t[0]:
            (input_grads, input_weight_grads, output_weight_grads,
                input_bias_grads, output_bias_grads) = (
                fast_self_multihead_attn_bias.backward(heads_t[0],
                output_grads, matmul2_results, dropout_results,
                softmax_results, input_lin_results, inputs, input_weights,
                output_weights, dropout_mask, dropout_prob_t[0]))
        else:
            input_bias_grads = None
            output_bias_grads = None
            input_grads, input_weight_grads, output_weight_grads = (
                fast_self_multihead_attn.backward(heads_t[0], output_grads,
                matmul2_results, dropout_results, softmax_results,
                input_lin_results, inputs, input_weights, output_weights,
                dropout_mask, dropout_prob_t[0]))
        return (None, None, None, input_grads, input_weight_grads,
            output_weight_grads, input_bias_grads, output_bias_grads, None,
            None, None)


fast_self_attn_func = FastSelfAttnFunc.apply


class SelfAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs,
        input_weights, output_weights, input_biases, output_biases, mask,
        dropout_prob):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads
        if use_biases_t[0]:
            input_lin_results = torch.addmm(input_biases, inputs.view(
                inputs.size(0) * inputs.size(1), inputs.size(2)),
                input_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            input_lin_results = torch.mm(inputs.view(inputs.size(0) *
                inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.
            size(1), input_weights.size(0))
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.
            size(1) * heads, 3, head_dim)
        queries = input_lin_results[:, :, (0), :]
        keys = input_lin_results[:, :, (1), :]
        values = input_lin_results[:, :, (2), :]
        matmul1_results = torch.empty((queries.size(1), queries.size(0),
            keys.size(0)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(
            0, 1), keys.transpose(0, 1).transpose(1, 2), out=
            matmul1_results, beta=0.0, alpha=scale_t[0])
        if mask is not None:
            if use_time_mask:
                assert len(mask.size()) == 2, 'Timing mask is not 2D!'
                assert mask.size(0) == mask.size(1
                    ), 'Sequence length should match!'
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask, float(
                    '-inf'))
            else:
                batches, seql_q, seql_k = matmul1_results.size()
                seqs = int(batches / heads)
                matmul1_results = matmul1_results.view(seqs, heads, seql_q,
                    seql_k)
                mask = mask.to(torch.bool)
                matmul1_results = matmul1_results.masked_fill_(mask.
                    unsqueeze(1).unsqueeze(2), float('-inf'))
                matmul1_results = matmul1_results.view(seqs * heads, seql_q,
                    seql_k)
        softmax_results = F.softmax(matmul1_results, dim=-1)
        if is_training:
            dropout_results, dropout_mask = torch._fused_dropout(
                softmax_results, p=1.0 - dropout_prob_t[0])
        else:
            dropout_results = softmax_results
            dropout_mask = null_tensor
        matmul2_results = torch.empty((dropout_results.size(1),
            dropout_results.size(0), values.size(2)), dtype=dropout_results
            .dtype, device=torch.device('cuda')).transpose(1, 0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0, 1),
            out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(
            inputs.size(0), inputs.size(1), inputs.size(2))
        if use_biases_t[0]:
            outputs = torch.addmm(output_biases, matmul2_results.view(
                inputs.size(0) * inputs.size(1), inputs.size(2)),
                output_weights.transpose(0, 1), beta=1.0, alpha=1.0)
        else:
            outputs = torch.mm(matmul2_results.view(inputs.size(0) * inputs
                .size(1), inputs.size(2)), output_weights.transpose(0, 1))
        outputs = outputs.view(inputs.size(0), inputs.size(1),
            output_weights.size(0))
        ctx.save_for_backward(use_biases_t, heads_t, scale_t,
            matmul2_results, dropout_results, softmax_results,
            input_lin_results, inputs, input_weights, output_weights,
            dropout_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (use_biases_t, heads_t, scale_t, matmul2_results, dropout_results,
            softmax_results, input_lin_results, inputs, input_weights,
            output_weights, dropout_mask, dropout_prob_t) = ctx.saved_tensors
        head_dim = inputs.size(2) // heads_t[0]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.
            size(1) * heads_t[0], 3, head_dim)
        queries = input_lin_results[:, :, (0), :]
        keys = input_lin_results[:, :, (1), :]
        values = input_lin_results[:, :, (2), :]
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads = input_lin_results_grads[:, :, (0), :]
        keys_grads = input_lin_results_grads[:, :, (1), :]
        values_grads = input_lin_results_grads[:, :, (2), :]
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) *
            output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0),
            output_grads.size(1), output_weights.size(1))
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(
            0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1
            ), matmul2_results.view(matmul2_results.size(0) *
            matmul2_results.size(1), matmul2_results.size(2)))
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.
            size(1) * heads_t[0], head_dim).transpose(0, 1)
        if use_biases_t[0]:
            output_bias_grads = torch.sum(output_grads.view(output_grads.
                size(0) * output_grads.size(1), output_grads.size(2)), 0)
        else:
            output_bias_grads = None
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0, 1)
            .transpose(1, 2))
        values_grads = torch.bmm(dropout_results.transpose(1, 2),
            output_lin_grads, out=values_grads.transpose(0, 1))
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, 
            1.0 / (1.0 - dropout_prob_t[0]))
        softmax_grads = torch._softmax_backward_data(dropout_grads,
            softmax_results, -1, softmax_results)
        queries_grads = torch.baddbmm(queries_grads.transpose(0, 1),
            softmax_grads, keys.transpose(0, 1), out=queries_grads.
            transpose(0, 1), beta=0.0, alpha=scale_t[0])
        keys_grads = torch.baddbmm(keys_grads.transpose(0, 1),
            softmax_grads.transpose(1, 2), queries.transpose(0, 1), out=
            keys_grads.transpose(0, 1), beta=0.0, alpha=scale_t[0])
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(
            0) * inputs.size(1), heads_t[0] * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1),
            inputs.size(2))
        input_weight_grads = torch.mm(input_lin_results_grads.transpose(0, 
            1), inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)))
        if use_biases_t[0]:
            input_bias_grads = torch.sum(input_lin_results_grads, 0)
        else:
            input_bias_grads = None
        return (None, None, None, None, input_grads, input_weight_grads,
            output_weight_grads, input_bias_grads, output_bias_grads, None,
            None)


self_attn_func = SelfAttnFunc.apply


class FastSelfAttnNormAddFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, inputs,
        lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights,
        output_weights, pad_mask, dropout_prob):
        heads_t = torch.tensor([heads])
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        use_mask = pad_mask is not None
        (lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, input_lin_results,
            softmax_results, dropout_results, dropout_mask, matmul2_results,
            dropout_add_mask, outputs) = (fast_self_multihead_attn_norm_add
            .forward(use_mask, use_time_mask, is_training, heads, inputs,
            lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights,
            output_weights, pad_mask if use_mask else null_tensor,
            dropout_prob))
        ctx.save_for_backward(heads_t, matmul2_results, dropout_results,
            softmax_results, input_lin_results, lyr_nrm_results,
            lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights, input_weights, output_weights,
            dropout_mask, dropout_add_mask, dropout_prob_t)
        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (heads_t, matmul2_results, dropout_results, softmax_results,
            input_lin_results, lyr_nrm_results, lyr_nrm_mean,
            lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights, input_weights, output_weights,
            dropout_mask, dropout_add_mask, dropout_prob_t) = ctx.saved_tensors
        (input_grads, lyr_nrm_gamma_grads, lyr_nrm_beta_grads,
            input_weight_grads, output_weight_grads) = (
            fast_self_multihead_attn_norm_add.backward(heads_t[0],
            output_grads, matmul2_results, dropout_results, softmax_results,
            input_lin_results, lyr_nrm_results, lyr_nrm_mean,
            lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights,
            lyr_nrm_beta_weights, input_weights, output_weights,
            dropout_mask, dropout_add_mask, dropout_prob_t[0]))
        return (None, None, None, input_grads, lyr_nrm_gamma_grads,
            lyr_nrm_beta_grads, input_weight_grads, output_weight_grads,
            None, None)


fast_self_attn_norm_add_func = FastSelfAttnNormAddFunc.apply


class SelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False,
        include_norm_add=False, impl='fast', separate_qkv_params=False,
        mask_additive=False):
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
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim,
                embed_dim))
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
            nn.init.xavier_uniform_(self.in_proj_weight)
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

    def forward(self, query, key, value, key_padding_mask=None,
        need_weights=False, attn_mask=None, is_training=True):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if self.separate_qkv_params:
            input_weights = torch.cat([self.q_weight.view(self.num_heads, 1,
                self.head_dim, self.embed_dim), self.k_weight.view(self.
                num_heads, 1, self.head_dim, self.embed_dim), self.v_weight
                .view(self.num_heads, 1, self.head_dim, self.embed_dim)], dim=1
                ).reshape(3 * self.embed_dim, self.embed_dim).contiguous()
        else:
            input_weights = self.in_proj_weight
        if self.bias:
            if self.separate_qkv_params:
                input_bias = torch.cat([self.q_bias.view(self.num_heads, 1,
                    self.head_dim), self.k_bias.view(self.num_heads, 1,
                    self.head_dim), self.v_bias.view(self.num_heads, 1,
                    self.head_dim)], dim=1).reshape(3 * self.embed_dim
                    ).contiguous()
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
                outputs = self.attn_func(attn_mask is not None, is_training,
                    self.num_heads, query, self.lyr_nrm_gamma_weights, self
                    .lyr_nrm_beta_weights, input_weights, self.
                    out_proj_weight, mask, self.dropout)
            else:
                lyr_nrm_results = self.lyr_nrm(query)
                outputs = self.attn_func(attn_mask is not None, is_training,
                    self.num_heads, self.scaling, lyr_nrm_results,
                    input_weights, self.out_proj_weight, input_bias, self.
                    out_proj_bias, mask, self.dropout)
                if is_training:
                    outputs = jit_dropout_add(outputs, query, self.dropout,
                        is_training)
                else:
                    outputs = outputs + query
        elif self.impl == 'fast':
            outputs = self.attn_func(attn_mask is not None, is_training,
                self.num_heads, query, input_weights, self.out_proj_weight,
                input_bias, self.out_proj_bias, mask, self.mask_additive,
                self.dropout)
        else:
            outputs = self.attn_func(attn_mask is not None, is_training,
                self.num_heads, self.scaling, query, input_weights, self.
                out_proj_weight, input_bias, self.out_proj_bias, mask, self
                .mask_additive, self.dropout)
        return outputs, None


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
                param.data = param.data.to(dtype=dtype)
            if (param._grad is not None and param._grad.data.dtype.
                is_floating_point):
                param._grad.data = param._grad.data.to(dtype=dtype)
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm
            ) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch
            .nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network


class FP16Model(nn.Module):
    """
    Convert model to half precision in a batchnorm-safe way.
    """

    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


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
        grads = mlp_cuda.backward(ctx.bias, ctx.activation, grad_o, ctx.
            outputs, ctx.saved_tensors)
        del ctx.outputs
        return None, None, *grads


class FusedLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module(
                'fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(input_, ctx.
            normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, mean, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.backward(grad_output.contiguous(
            ), mean, invvar, input_, ctx.normalized_shape, ctx.eps)
        return grad_input, None, None


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module(
                'fused_layer_norm_cuda')
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(input_,
            ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = (fused_layer_norm_cuda.
            backward_affine(grad_output.contiguous(), mean, invvar, input_,
            ctx.normalized_shape, weight_, bias_, ctx.eps))
        return grad_input, grad_weight, grad_bias, None, None


class FusedLayerNorm(torch.nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    Currently only runs on cuda() tensors.

    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

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
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedLayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedLayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(FusedLayerNorm, self).__init__()
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda'
            )
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if not input.is_cuda:
            return F.layer_norm(input, self.normalized_shape, self.weight,
                self.bias, self.eps)
        if self.elementwise_affine:
            return FusedLayerNormAffineFunction.apply(input, self.weight,
                self.bias, self.normalized_shape, self.eps)
        else:
            return FusedLayerNormFunction.apply(input, self.
                normalized_shape, self.eps)

    def extra_repr(self):
        return (
            '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'
            .format(**self.__dict__))


def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        import apex_C
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        print(
            'Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.'
            )
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
            import amp_C
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def check_avail(self):
        if MultiTensorApply.available == False:
            raise RuntimeError(
                'Attempted to call MultiTensorApply method, but MultiTensorApply is not available, possibly because Apex was installed without --cpp_ext --cuda_ext.  Original import error message:'
                , MultiTensorApply.import_err)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.check_avail()
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


def split_half_float_double(tensors):
    dtypes = ['torch.cuda.HalfTensor', 'torch.cuda.FloatTensor',
        'torch.cuda.DoubleTensor']
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, z, weight, bias, running_mean, running_variance,
        eps, track_running_stats=True, momentum=1.0, process_group=None,
        channel_last=False, fuse_relu=False):
        input = input.contiguous()
        world_size = 0
        mean = None
        var_biased = None
        inv_std = None
        var = None
        out = None
        count = None
        if track_running_stats:
            if channel_last:
                count = int(input.numel() / input.size(-1))
                mean, var_biased = syncbn.welford_mean_var_c_last(input)
            else:
                count = int(input.numel() / input.size(1))
                mean, var_biased = syncbn.welford_mean_var(input)
            if torch.distributed.is_initialized():
                if not process_group:
                    process_group = torch.distributed.group.WORLD
                world_size = torch.distributed.get_world_size(process_group)
                mean_all = torch.empty(world_size, mean.size(0), dtype=mean
                    .dtype, device=mean.device)
                var_all = torch.empty(world_size, var_biased.size(0), dtype
                    =var_biased.dtype, device=var_biased.device)
                mean_l = [mean_all.narrow(0, i, 1) for i in range(world_size)]
                var_l = [var_all.narrow(0, i, 1) for i in range(world_size)]
                torch.distributed.all_gather(mean_l, mean, process_group)
                torch.distributed.all_gather(var_l, var_biased, process_group)
                mean, var, inv_std = syncbn.welford_parallel(mean_all,
                    var_all, count, eps)
            else:
                inv_std = 1.0 / torch.sqrt(var_biased + eps)
                var = var_biased * count / (count - 1)
            if count == 1 and world_size < 2:
                raise ValueError(
                    'Expected more than 1 value per channel when training, got input size{}'
                    .format(input.size()))
            r_m_inc = (mean if running_mean.dtype != torch.float16 else
                mean.half())
            r_v_inc = (var if running_variance.dtype != torch.float16 else
                var.half())
            running_mean.data = running_mean.data * (1 - momentum
                ) + momentum * r_m_inc
            running_variance.data = running_variance.data * (1 - momentum
                ) + momentum * r_v_inc
        else:
            mean = running_mean.data
            inv_std = 1.0 / torch.sqrt(running_variance.data + eps)
        ctx.save_for_backward(input, weight, mean, inv_std, z, bias)
        ctx.process_group = process_group
        ctx.channel_last = channel_last
        ctx.world_size = world_size
        ctx.fuse_relu = fuse_relu
        if channel_last:
            out = syncbn.batchnorm_forward_c_last(input, z, mean, inv_std,
                weight, bias, fuse_relu)
        else:
            out = syncbn.batchnorm_forward(input, mean, inv_std, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, inv_std, z, bias = ctx.saved_tensors
        process_group = ctx.process_group
        channel_last = ctx.channel_last
        world_size = ctx.world_size
        fuse_relu = ctx.fuse_relu
        grad_input = grad_z = grad_weight = grad_bias = None
        if fuse_relu:
            grad_output = syncbn.relu_bw_c_last(grad_output, saved_input, z,
                mean, inv_std, weight, bias)
        if isinstance(z, torch.Tensor) and ctx.needs_input_grad[1]:
            grad_z = grad_output.clone()
        if channel_last:
            mean_dy, mean_dy_xmu, grad_weight, grad_bias = (syncbn.
                reduce_bn_c_last(grad_output, saved_input, mean, inv_std,
                weight))
        else:
            mean_dy, mean_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(
                grad_output, saved_input, mean, inv_std, weight)
        if ctx.needs_input_grad[0]:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_dy, ReduceOp.SUM,
                    process_group)
                mean_dy = mean_dy / world_size
                torch.distributed.all_reduce(mean_dy_xmu, ReduceOp.SUM,
                    process_group)
                mean_dy_xmu = mean_dy_xmu / world_size
            if channel_last:
                grad_input = syncbn.batchnorm_backward_c_last(grad_output,
                    saved_input, mean, inv_std, weight, mean_dy, mean_dy_xmu)
            else:
                grad_input = syncbn.batchnorm_backward(grad_output,
                    saved_input, mean, inv_std, weight, mean_dy, mean_dy_xmu)
        if weight is None or not ctx.needs_input_grad[2]:
            grad_weight = None
        if weight is None or not ctx.needs_input_grad[3]:
            grad_bias = None
        return (grad_input, grad_z, grad_weight, grad_bias, None, None,
            None, None, None, None, None, None)


class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from `torch.nn.BatchNormNd`
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    `DistributedDataParallel`.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from `torch.distributed`.

    When running in evaluation mode, the layer falls back to
    `torch.nn.functional.batch_norm`

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
        process_group: pass in a process group within which the stats of the
            mini-batch is being synchronized. ``None`` for using default process
            group
        channel_last: a boolean value that when set to ``True``, this module
            take the last dimension of the input tensor to be the channel
            dimension. Default: False

    Examples::
        >>> # channel first tensor
        >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
        >>> # channel last tensor
        >>> sbn = apex.parallel.SyncBatchNorm(100, channel_last=True).cuda()
        >>> inp = torch.randn(10, 14, 14, 100).cuda()
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, process_group=None, channel_last=False,
        fuse_relu=False):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum
            =momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group
        self.channel_last = channel_last
        self.fuse_relu = fuse_relu

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def _specify_channel_last(self, channel_last):
        self.channel_last = channel_last

    def forward(self, input, z=None):
        channel_last = self.channel_last if input.dim() != 2 else True
        if (not self.training and self.track_running_stats and not
            channel_last and not self.fuse_relu and z == None):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, False, 0.0, self.eps)
        else:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.
                        num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
            return SyncBatchnormFunction.apply(input, z, self.weight, self.
                bias, self.running_mean, self.running_var, self.eps, self.
                training or not self.track_running_stats,
                exponential_average_factor, self.process_group,
                channel_last, self.fuse_relu)


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

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, process_group=None, channel_last=False):
        if channel_last == True:
            raise AttributeError(
                'channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.'
                )
        if not SyncBatchNorm.warned:
            if hasattr(self, 'syncbn_import_error'):
                None
            else:
                None
            SyncBatchNorm.warned = True
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum
            =momentum, affine=affine, track_running_stats=track_running_stats)
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
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype
        if not self.training and self.track_running_stats:
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(
                    1)
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group
                        )
                    torch.distributed.all_reduce(local_mean, ReduceOp.SUM,
                        process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(local_sqr_mean, ReduceOp.
                        SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                var = sqr_mean - mean.pow(2)
                if self.running_mean is not None:
                    self.running_mean = self.momentum * mean + (1 - self.
                        momentum) * self.running_mean
                if self.running_var is not None:
                    self.running_var = m / (m - 1) * self.momentum * var + (
                        1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias,
                mean, var, self.eps, process_group, world_size)
        return out.to(cast)


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


class Foo(torch.jit.ScriptModule):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    @torch.jit.script_method
    def forward(self, input):
        return self.n * input + self.m


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    count = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.id = Bottleneck.count
        Bottleneck.count += 1

    def forward(self, x):
        identity = x
        nvtx.range_push('layer:Bottleneck_{}'.format(self.id))
        nvtx.range_push('layer:Conv1')
        out = self.conv1(x)
        nvtx.range_pop()
        nvtx.range_push('layer:BN1')
        out = self.bn1(out)
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_push('layer:Conv2')
        out = self.conv2(out)
        nvtx.range_pop()
        nvtx.range_push('layer:BN2')
        out = self.bn2(out)
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_push('layer:Conv3')
        out = self.conv3(out)
        nvtx.range_pop()
        nvtx.range_push('layer:BN3')
        out = self.bn3(out)
        nvtx.range_pop()
        if self.downsample is not None:
            nvtx.range_push('layer:Downsample')
            identity = self.downsample(x)
            nvtx.range_pop()
        nvtx.range_push('layer:Residual')
        out += identity
        nvtx.range_pop()
        nvtx.range_push('layer:ReLU')
        out = self.relu(out)
        nvtx.range_pop()
        nvtx.range_pop()
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, groups=1,
        width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        nvtx.range_push('layer:conv1_x')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv2_x')
        x = self.layer1(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv3_x')
        x = self.layer2(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv4_x')
        x = self.layer3(x)
        nvtx.range_pop()
        nvtx.range_push('layer:conv5_x')
        x = self.layer4(x)
        nvtx.range_pop()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        nvtx.range_push('layer:FC')
        x = self.fc(x)
        nvtx.range_pop()
        return x


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda',
            dtype=torch.float32))
        self.weight1 = Parameter(1.0 + unique + torch.arange(2, device=
            'cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return (input * weight0.float() * weight1.float()).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


class WhitelistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(WhitelistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(8 * 8, device='cuda',
            dtype=dtype).view(8, 8))

    @staticmethod
    def ops(input, weight):
        return input.mm(weight).mm(weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class BlacklistModule(torch.nn.Module):

    def __init__(self, dtype):
        super(BlacklistModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda',
            dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input + torch.pow(weight, 2) + torch.pow(weight, 2)).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class PromoteModule(torch.nn.Module):

    def __init__(self, dtype):
        super(PromoteModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda',
            dtype=dtype).view(2, 8))

    @staticmethod
    def ops(input, weight):
        return (input * weight * weight).sum()

    def forward(self, input):
        return self.ops(input, self.weight)


class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x * self.param
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        return x


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda',
            dtype=torch.float32))
        self.weight1 = Parameter(1.0 + unique + torch.arange(2, device=
            'cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return (input * weight0.float() * weight1.float()).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda',
            dtype=torch.float32))

    def forward(self, input):
        return (input * self.weight0).sum()


class MyModel(torch.nn.Module):

    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique + torch.arange(2, device='cuda',
            dtype=torch.float32))
        self.weight1 = Parameter(1.0 + unique + torch.arange(2, device=
            'cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return (input * weight0.float() * weight1.float()).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


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
        self.a = Parameter(torch.cuda.FloatTensor(4096 * 4096).fill_(1.0))
        self.b = Parameter(torch.cuda.FloatTensor(4096 * 4096).fill_(2.0))

    def forward(self, input):
        return input * self.a * self.b


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_NVIDIA_apex(_paritybench_base):
    pass
    def test_000(self):
        self._check(DummyNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(Foo(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(SyncBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(tofp16(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

