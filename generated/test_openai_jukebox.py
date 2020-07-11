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
fp16_utils = _module
fp16_optimizer = _module
fp16util = _module
loss_scaler = _module
multi_tensor_apply = _module
multi_tensor_apply = _module
normalization = _module
fused_layer_norm = _module
optimizers = _module
fp16_optimizer = _module
fused_adam = _module
LARC = _module
parallel = _module
distributed = _module
multiproc = _module
optimized_sync_batchnorm = _module
optimized_sync_batchnorm_kernel = _module
sync_batchnorm = _module
sync_batchnorm_kernel = _module
reparameterization = _module
reparameterization = _module
weight_norm = _module
conf = _module
main_amp = _module
distributed_data_parallel = _module
setup = _module
run_amp = _module
test_add_param_group = _module
test_basic_casts = _module
test_cache = _module
test_multi_tensor_axpby = _module
test_multi_tensor_l2norm = _module
test_multi_tensor_scale = _module
test_multiple_models_optimizers_losses = _module
test_promotion = _module
test_rnn = _module
utils = _module
run_fp16util = _module
test_fp16util = _module
test_fused_layer_norm = _module
run_mixed_adam = _module
test_fp16_optimizer = _module
test_mixed_adam = _module
run_test = _module
compare = _module
main_amp = _module
ddp_race_condition_test = _module
amp_master_params = _module
compare = _module
single_gpu_unit_test = _module
test_groups = _module
two_gpu_unit_test = _module
jukebox = _module
align = _module
data = _module
artist_genre_processor = _module
data_processor = _module
files_dataset = _module
labels = _module
text_processor = _module
hparams = _module
lyricdict = _module
make_models = _module
prior = _module
autoregressive = _module
conditioners = _module
prior = _module
sample = _module
save_html = _module
test_sample = _module
train = _module
transformer = _module
factored_attention = _module
ops = _module
transformer = _module
audio_utils = _module
checkpoint = _module
dist_adapter = _module
dist_utils = _module
ema = _module
fp16 = _module
gcs_utils = _module
io = _module
logger = _module
sample_utils = _module
torch_utils = _module
vqvae = _module
bottleneck = _module
encdec = _module
resnet = _module
vqvae = _module
conf = _module
examples = _module
net = _module
train_dcgan = _module
updater = _module
visualize = _module
writetensorboard = _module
train_vae = _module
demo = _module
demo_beholder = _module
demo_caffe2 = _module
demo_custom_scalars = _module
demo_embedding = _module
demo_graph = _module
demo_hparams = _module
demo_matplotlib = _module
demo_multiple_embedding = _module
demo_nvidia_smi = _module
demo_onnx = _module
demo_purge = _module
tensorboardX = _module
beholder = _module
file_system_tools = _module
shared_config = _module
video_writing = _module
caffe2_graph = _module
crc32c = _module
embedding = _module
event_file_writer = _module
onnx_graph = _module
proto = _module
api_pb2 = _module
attr_value_pb2 = _module
event_pb2 = _module
graph_pb2 = _module
layout_pb2 = _module
node_def_pb2 = _module
plugin_hparams_pb2 = _module
plugin_mesh_pb2 = _module
plugin_pr_curve_pb2 = _module
plugin_text_pb2 = _module
resource_handle_pb2 = _module
step_stats_pb2 = _module
summary_pb2 = _module
tensor_pb2 = _module
tensor_shape_pb2 = _module
types_pb2 = _module
versions_pb2 = _module
proto_graph = _module
pytorch_graph = _module
record_writer = _module
summary = _module
torchvis = _module
visdom_writer = _module
writer = _module
x2num = _module
tests = _module
event_file_writer_test = _module
expect_reader = _module
record_writer_test = _module
test_beholder = _module
test_caffe2 = _module
test_chainer_np = _module
test_crc32c = _module
test_embedding = _module
test_figure = _module
test_numpy = _module
test_onnx_graph = _module
test_pr_curve = _module
test_pytorch_graph = _module
test_pytorch_np = _module
test_record_writer = _module
test_summary = _module
test_summary_writer = _module
test_test = _module
test_utils = _module
test_visdom = _module
test_writer = _module

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


from torch._six import string_classes


import functools


import numpy as np


import warnings


import types


import itertools


import torch.nn.functional


from itertools import product


from torch import nn


from torch.nn.parameter import Parameter


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import numbers


from torch.nn import init


from torch.nn import functional as F


import torch.distributed as dist


from torch.nn.modules import Module


from collections import OrderedDict


from itertools import chain


import copy


from torch.nn.modules.batchnorm import _BatchNorm


from torch.autograd.function import Function


import time


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import functools as ft


import itertools as it


from torch.nn import Parameter


import random


from torch.nn import Module


import torch.optim as optim


import torch as t


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import BatchSampler


from torch.utils.data import RandomSampler


from torch.nn.parallel import DistributedDataParallel


from enum import Enum


from time import sleep


from torch.optim import Optimizer


import torchvision.utils as vutils


from torchvision import datasets


from torch.autograd.variable import Variable


from torch.utils.data import TensorDataset


import torchvision


import logging


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
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)

    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)


class FusedLayerNormAffineFunction(torch.autograd.Function):

    def __init__(self, normalized_shape, eps=1e-06):
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input, weight, bias):
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(input_, self.normalized_shape, weight_, bias_, self.eps)
        self.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    def backward(self, grad_output):
        input_, weight_, bias_, mean, invvar = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(grad_output.contiguous(), mean, invvar, input_, self.normalized_shape, weight_, bias_, self.eps)
        return grad_input, grad_weight, grad_bias


class FusedLayerNormFunction(torch.autograd.Function):

    def __init__(self, normalized_shape, eps=1e-06):
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input):
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(input_, self.normalized_shape, self.eps)
        self.save_for_backward(input_, mean, invvar)
        return output

    def backward(self, grad_output):
        input_, mean, invvar = self.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.backward(grad_output.contiguous(), mean, invvar, input_, self.normalized_shape, self.eps)
        return grad_input


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
        fused_layer_norm_cuda = importlib.import_module('fused_layer_norm_cuda')
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
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.elementwise_affine:
            return FusedLayerNormAffineFunction(self.normalized_shape, self.eps)(input, self.weight, self.bias)
        else:
            return FusedLayerNormFunction(self.normalized_shape, self.eps)(input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)


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


class ReduceOp(Enum):
    SUM = 0,
    PRODUCT = 1,
    MIN = 2,
    MAX = 3

    def ToDistOp(self):
        return {self.SUM: dist.ReduceOp.SUM, self.PRODUCT: dist.ReduceOp.PRODUCT, self.MIN: dist.ReduceOp.MIN, self.MAX: dist.ReduceOp.MAX}[self]


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
        if channel_last == True:
            raise AttributeError('channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.')
        if not SyncBatchNorm.warned:
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
        out = out


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


def get_normal(*shape, std=0.01):
    w = t.empty(shape)
    nn.init.normal_(w, std=std)
    return w


class PositionEmbedding(nn.Module):

    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer('pos', t.tensor(get_pos_idx(input_shape)).long())
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        if self.pos_init:
            pos_emb = sum([self._pos_embs[i](self.pos[:, (i)]) for i in range(len(self.input_shape))])
        else:
            pos_emb = self.pos_emb
        return pos_emb


class Conv1D(nn.Module):

    def __init__(self, n_in, n_out, zero_out=False, init_scale=1.0):
        super(Conv1D, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if zero_out:
            w = t.zeros(n_in, n_out)
        else:
            w = t.empty(n_in, n_out)
            nn.init.normal_(w, std=0.02 * init_scale)
        b = t.zeros(n_out)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        size_out = *x.size()[:-1], self.n_out
        x = t.addmm(self.b.type_as(x), x.view(-1, x.size(-1)), self.w.type_as(x))
        x = x.view(*size_out)
        return x


class CheckpointFunction(t.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with t.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with t.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = t.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    if flag:
        args = inputs + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def get_mask(mask, q_l, kv_l, blocks, spread, device, sample, sample_t):
    if mask is None or q_l == 1:
        return None
    offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
    if mask == 'autoregressive':
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    elif mask == 'summary':
        mask = t.nn.functional.pad(t.ones(q_l, q_l, device=device).tril().view(q_l, blocks, q_l // blocks)[:, :-1, -kv_l // blocks:], (0, 0, 1, 0), value=1).contiguous().view(q_l, kv_l)
    elif mask == 'prime':
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    return mask.view(1, 1, q_l, kv_l)


class FactoredAttention(nn.Module):

    def __init__(self, n_in, n_ctx, n_state, n_head, attn_dropout=0.0, resid_dropout=0.0, scale=True, mask=False, zero_out=False, init_scale=1.0, checkpoint_attn=0, attn_func=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.n_state = n_state
        assert n_state % n_head == 0
        self.n_head = n_head
        self.scale = scale
        self.mask = mask
        if attn_func == 6:
            self.c_attn = Conv1D(n_in, n_state, init_scale=init_scale)
            self.c_enc_kv = Conv1D(n_in, n_state * 2, init_scale=init_scale)
        else:
            self.c_attn = Conv1D(n_in, n_state * 3, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else lambda x: x
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x
        self.attn_func = attn_func
        self.qkv, self.attn, self.attn_mask = {(0): (self.factored_qkv, self.dense_attn, 'autoregressive'), (1): (self.factored_qkv, self.block_attn, 'autoregressive'), (2): (self.factored_qkv, self.transpose_block_attn, 'autoregressive'), (3): (self.factored_qkv, self.prev_block_attn, None), (4): (self.factored_qkv, self.summary_attn, 'summary'), (5): (self.factored_qkv, self.summary_spread_attn, 'summary'), (6): (self.decode_qkv, self.decode_attn, None), (7): (self.prime_qkv, self.prime_attn, 'prime')}[attn_func]
        self.blocks = blocks
        self.spread = spread
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.checkpoint_attn = checkpoint_attn
        self.sample_t = 0
        self.cache = {}
        self.encoder_dims = encoder_dims
        self.prime_len = prime_len
        self.record_attn = False
        self.w = None

    def _attn(self, q, k, v, sample):
        scale = 1.0 / math.sqrt(math.sqrt(self.n_state // self.n_head))
        if self.training:
            w = t.matmul(q * scale, k * scale)
        else:
            w = t.matmul(q, k)
            w.mul_(scale * scale)
        wtype = w.dtype
        w = w.float()
        if self.mask:
            mask = get_mask(self.attn_mask, q.size(-2), k.size(-1), self.blocks, self.spread, w.device, sample, self.sample_t)
            if mask is not None:
                w = w * mask + -1000000000.0 * (1 - mask)
            w = F.softmax(w, dim=-1).type(wtype)
        else:
            w = F.softmax(w, dim=-1).type(wtype)
        if self.record_attn:
            self.w = w
            if self.attn_func == 7:
                self.w = self.w[:, :, self.prime_len:, :self.prime_len]
        w = self.attn_dropout(w)
        a = t.matmul(w, v)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = *x.size()[:-2], x.size(-2) * x.size(-1)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = *x.size()[:-1], self.n_head, x.size(-1) // self.n_head
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.checkpoint_attn == 1 and not sample:
            a = checkpoint(lambda q, k, v, s=sample: self._attn(q, k, v, s), (query, key, value), (), True)
        else:
            a = self._attn(query, key, value, sample)
        a = self.merge_heads(a)
        return a

    def block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            assert l == self._suff_cache_len(), f'{l} != {self._suff_cache_len()}'
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            if ql < l:
                l = ql
                k = k[:, -l:].contiguous()
                v = v[:, -l:].contiguous()
            k = k.view(bs * l // block_ctx, block_ctx, d)
            v = v.view(bs * l // block_ctx, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def transpose_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            block_l = (l - 1) % block_ctx
            k = k[:, block_l::block_ctx, :]
            v = v[:, block_l::block_ctx, :]
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs, ql // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, ql // block_ctx, d)
            k = k.view(bs, l // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, l // block_ctx, d)
            v = v.view(bs, l // block_ctx, block_ctx, d).transpose(1, 2).contiguous().view(bs * block_ctx, l // block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, block_ctx, ql // block_ctx, d).transpose(1, 2).contiguous().view(bs, ql, d)

    def prev_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            assert l == self._suff_cache_len(), f'{l} != {self._suff_cache_len()}'
            block = (l - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                assert prev_l == 0
                k = k[:, prev_l:prev_l + block_ctx, :]
                v = v[:, prev_l:prev_l + block_ctx, :]
            else:
                k = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
                v = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            k = t.nn.functional.pad(k.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)).view(bs * l // block_ctx, block_ctx, d)
            v = t.nn.functional.pad(v.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)).view(bs * l // block_ctx, block_ctx, d)
            if ql < l:
                qb = ql // block_ctx
                kb = l // block_ctx
                l = ql
                k = k.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
                v = v.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx
        bs, l, d = v.shape
        if sample:
            k = t.nn.functional.pad(k[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :], (0, 0, 1, 0))
            v = t.nn.functional.pad(v[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :], (0, 0, 1, 0))
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, (-1), :], (0, 0, 1, 0))
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, (-1), :], (0, 0, 1, 0))
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_spread_attn(self, q, k, v, sample):
        blocks, block_ctx, spread = self.blocks, self.block_ctx, self.spread
        bs, l, d = v.shape
        if sample:
            assert False, 'Not yet implemented'
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0)).contiguous().view(bs, blocks * spread, d)
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0)).contiguous().view(bs, blocks * spread, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def prime_attn(self, q, k, v, sample):
        prime_len = self._prime_len
        k = k[:, :prime_len]
        v = v[:, :prime_len]
        return self.dense_attn(q, k, v, sample)

    def decode_attn(self, q, k, v, sample):
        assert k.shape[1] == v.shape[1] == self.encoder_dims, f'k: {k.shape}, v: {v.shape}, enc_dims: {self.encoder_dims}'
        return self.dense_attn(q, k, v, sample)

    def factored_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 0:
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                    assert key.shape[1] % self.block_ctx == 0
                    assert query.shape[1] % self.block_ctx == 0
                assert key.shape[1] == value.shape[1]
                assert query.shape[1] <= key.shape[1]
                sample = False
            else:
                key = self.cache['key']
                value = self.cache['value']
        return query, key, value, sample

    def prime_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._prime_len:
                self._append_cache(key, value)
            if self._cache_len() > self._prime_len:
                self._slice_cache(0, self._prime_len)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
            assert key.shape[1] == value.shape[1] == self._suff_cache_len(), f'k: {key.shape}, v: {value.shape}, prime_dims: {self._suff_cache_len()}'
        else:
            assert key.shape[1] == value.shape[1] == self.n_ctx, f'k: {key.shape}, v: {value.shape}, prime_dims: {self.n_ctx}'
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def decode_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is not None
        query = x
        if sample:
            if self.sample_t == 0:
                self.cache['key'], self.cache['value'] = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[1] == value.shape[1] == self.encoder_dims, f'k: {key.shape}, v: {value.shape}, enc_dims: {self.encoder_dims}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def forward(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        x = self.c_attn(x)
        query, key, value, sample = self.qkv(x, encoder_kv=encoder_kv, sample=sample)
        if self.checkpoint_attn == 2 and not sample:
            a = checkpoint(lambda q, k, v, s=sample: self.attn(q, k, v, s), (query, key, value), (), True)
        else:
            a = self.attn(query, key, value, sample)
        if a.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            a = a[:, offset:offset + curr_ctx, :].contiguous()
        a = self.c_proj(a)
        return self.resid_dropout(a)

    @property
    def _prime_len(self):
        prime_len = self.prime_len
        assert prime_len is not None
        prime_blocks = prime_len // self.blocks + 1
        return prime_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 0:
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, x, query=False):
        l = x.shape[1]
        offset = self._offset(l) if query else 0
        n_blocks = (l + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - l - offset
        if pad == 0 and offset == 0:
            return x
        else:
            return F.pad(x, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if 'key' not in self.cache else self.cache['key'].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and
            self.sample_t reflects the 1-indexed sample location in the
            context.
        """
        if self.attn_func == 0:
            return self.sample_t
        elif self.attn_func == 1:
            return (self.sample_t - 1) % self.block_ctx + 1
        elif self.attn_func == 2:
            return self.sample_t
        elif self.attn_func == 3:
            if self.sample_t <= self.block_ctx:
                return self.sample_t
            else:
                curr_block = (self.sample_t - 1) % self.block_ctx + 1
                prev_block = self.block_ctx
                return curr_block + prev_block
        elif self.attn_func == 6:
            return self.encoder_dims
        elif self.attn_func == 7:
            return min(self.sample_t, self._prime_len)
        else:
            raise NotImplementedError()

    def _slice_cache(self, start, end=None):
        self.cache['key'] = self.cache['key'][:, start:end]
        self.cache['value'] = self.cache['value'][:, start:end]

    def _append_cache(self, key, value):
        if 'key' not in self.cache:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            old_key, old_value = key, value
            key = t.cat([self.cache['key'], key], dim=1)
            value = t.cat([self.cache['value'], value], dim=1)
            del self.cache['key']
            del self.cache['value']
            del old_key
            del old_value
            self.cache['key'] = key
            self.cache['value'] = value
        return self.cache['key'], self.cache['value']

    def del_cache(self):
        self.sample_t = 0
        if 'key' in self.cache:
            del self.cache['key']
        if 'value' in self.cache:
            del self.cache['value']
        self.cache = {}

    def check(self):
        blocks = self.blocks or 1
        spread = self.spread or 1
        bs, l, d = 4, self.n_ctx, self.n_in
        x = t.randn(bs, l, d)
        x.requires_grad = True
        x_out = self.forward(x)
        loss = x_out.mean(dim=-1)
        pos = 60
        grad = t.autograd.grad(loss[2, pos], x)[0]
        assert grad.shape == (bs, l, d)
        assert (grad[:2] == 0).all()
        assert (grad[3:] == 0).all()
        assert (grad[(2), pos + 1:] == 0).all()
        pos_grad = (t.sum(grad[2] ** 2, dim=-1) > 0).nonzero().view(-1).cpu()
        block_pos = pos - pos % (l // blocks)
        exp_pos_grad = {(0): t.arange(pos), (1): t.arange(block_pos, pos), (2): t.arange(pos % (l // blocks), pos, l // blocks), (3): t.arange(block_pos - l // blocks, block_pos), (4): t.arange(l // blocks - 1, pos, l // blocks), (5): ((t.arange(pos) % (l // blocks) >= l // blocks - spread) & (t.arange(pos) < block_pos)).nonzero().view(-1)}[self.attn_func]
        exp_pos_grad = t.cat([exp_pos_grad, t.tensor([pos])], dim=-1)
        assert len(pos_grad) == len(exp_pos_grad) and (pos_grad == exp_pos_grad).all(), f'Expected pos grad {exp_pos_grad} got {pos_grad} for attn_func {self.attn_func} pos {pos} l {l} blocks {blocks}'

    def check_cache(self, n_samples, sample_t, fp16):
        assert self.sample_t == sample_t, f'{self.sample_t} != {sample_t}'
        if sample_t == 0:
            assert self.cache == {}
        else:
            dtype = {(True): t.float16, (False): t.float32}[fp16]
            l_cache = self._suff_cache_len()
            assert self.cache['key'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['value'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['key'].dtype == dtype, f"Expected {dtype}, got {self.cache['key'].dtype}"
            assert self.cache['value'].dtype == dtype, f"Expected {dtype}, got {self.cache['value'].dtype}"

    def check_sample(self):
        t.manual_seed(42)
        bs, l, d = 4, self.n_ctx, self.n_in
        prime = 5
        x = t.randn(bs, l, d)
        xs = t.chunk(x, l, dim=1)
        assert self.sample_t == 0
        assert self.cache == {}
        with t.no_grad():
            enc_l = self.encoder_dims
            encoder_kv = None
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d)
            x_out_normal = self.forward(x, encoder_kv=encoder_kv)
            x_out_sample = t.cat([self.forward(xs[i], encoder_kv=encoder_kv, sample=True) for i in range(l)], dim=1)
        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-08, f'Max sampling err is {max_err} {[i for i in range(l) if t.max(t.abs(x_out_sample - x_out_normal)[:, (i), :]) > 1e-08]}'
        with t.no_grad():
            x_out_normal = x_out_normal[:, :prime, :]
            self.del_cache()
            x_out_sample = self.forward(x[:, :prime, :].contiguous(), encoder_kv=encoder_kv, sample=True)
            self.check_cache(bs, prime, False)
        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-08, f'Max prime sampling err is {max_err} {[i for i in range(prime) if t.max(t.abs(x_out_sample - x_out_normal)[:, (i), :]) > 1e-08]}'

    def check_chunks(self, chunk_size):
        t.manual_seed(42)
        bs, l, d = 4, self.n_ctx, self.n_in
        enc_l = self.encoder_dims
        assert l % chunk_size == 0
        n_chunks = l // chunk_size
        with t.no_grad():
            encoder_kv = None
            x = t.randn(bs, l, d)
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d)
            self.del_cache()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=False)
            self.del_cache()
            y_forw_sample = self.forward(x, encoder_kv=encoder_kv, sample=True)
            max_err = t.max(t.abs(y_forw - y_forw_sample))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_sample)[:, (i), :]) > 1e-06]}'
            self.del_cache()
            x_chunks = t.chunk(x, n_chunks, dim=1)
            y_chunks = []
            total_len = 0
            for x_chunk in x_chunks:
                y_chunk = self.forward(x_chunk.contiguous(), encoder_kv=encoder_kv, sample=True)
                total_len += x_chunk.shape[1]
                self.check_cache(bs, total_len, False)
                y_chunks.append(y_chunk)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)
            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, (i), :]) > 1e-06]}'


class LayerNorm(FusedLayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(input.float(), self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            return super(LayerNorm, self).forward(input.float()).type_as(input)


def gelu(x):
    return 0.5 * x * (1 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))


@t.jit.script
def quick_gelu(x):
    return x * t.sigmoid(1.702 * x)


@t.jit.script
def quick_gelu_bwd(x, grad_output):
    sig = t.sigmoid(1.702 * x)
    return grad_output * sig * (1.702 * x * (1 - sig) + 1.0)


class QuickGelu(t.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return quick_gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return quick_gelu_bwd(ctx.saved_tensors[0], grad_output)


def memory_efficient_quick_gelu(x):
    return QuickGelu.apply(x)


def swish(x):
    return x * t.sigmoid(x)


ACT_FNS = {'relu': t.nn.functional.relu, 'swish': swish, 'gelu': gelu, 'quick_gelu': memory_efficient_quick_gelu}


class MLP(nn.Module):

    def __init__(self, n_in, n_state, resid_dropout=0.0, afn='quick_gelu', zero_out=False, init_scale=1.0):
        super().__init__()
        self.c_fc = Conv1D(n_in, n_state, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.act = ACT_FNS[afn]
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x

    def forward(self, x):
        m = self.act(self.c_fc(x))
        m = self.c_proj(m)
        return self.resid_dropout(m)


class ResAttnBlock(nn.Module):

    def __init__(self, n_in, n_ctx, n_head, attn_dropout=0.0, resid_dropout=0.0, afn='quick_gelu', scale=True, mask=False, zero_out=False, init_scale=1.0, res_scale=1.0, m_attn=0.25, m_mlp=1.0, checkpoint_attn=0, checkpoint_mlp=0, attn_func=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.attn = FactoredAttention(n_in=n_in, n_ctx=n_ctx, n_state=int(m_attn * n_in), n_head=n_head, attn_dropout=attn_dropout, resid_dropout=resid_dropout, scale=scale, mask=mask, zero_out=zero_out, init_scale=init_scale, checkpoint_attn=checkpoint_attn, attn_func=attn_func, blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.ln_0 = LayerNorm(n_in)
        self.mlp = MLP(n_in=n_in, n_state=int(m_mlp * n_in), resid_dropout=resid_dropout, afn=afn, zero_out=zero_out, init_scale=init_scale)
        self.ln_1 = LayerNorm(n_in)
        self.res_scale = res_scale
        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_mlp = checkpoint_mlp
        self.n_in = n_in
        self.attn_func = attn_func

    def forward(self, x, encoder_kv, sample=False):
        if sample:
            a = self.attn(self.ln_0(x), encoder_kv, sample)
            m = self.mlp(self.ln_1(x + a))
        else:
            if self.attn_func == 6:
                assert encoder_kv is not None
                a = checkpoint(lambda _x, _enc_kv, _s=sample: self.attn(self.ln_0(_x), _enc_kv, _s), (x, encoder_kv), (*self.attn.parameters(), *self.ln_0.parameters()), self.checkpoint_attn == 3)
            else:
                assert encoder_kv is None
                a = checkpoint(lambda _x, _enc_kv=None, _s=sample: self.attn(self.ln_0(_x), _enc_kv, _s), (x,), (*self.attn.parameters(), *self.ln_0.parameters()), self.checkpoint_attn == 3)
            m = checkpoint(lambda _x: self.mlp(self.ln_1(_x)), (x + a,), (*self.mlp.parameters(), *self.ln_1.parameters()), self.checkpoint_mlp == 1)
        if self.res_scale == 1.0:
            h = x + a + m
        else:
            h = x + self.res_scale * (a + m)
        return h


class Transformer(nn.Module):

    def __init__(self, n_in, n_ctx, n_head, n_depth, attn_dropout=0.0, resid_dropout=0.0, afn='quick_gelu', scale=True, mask=False, zero_out=False, init_scale=1.0, res_scale=False, m_attn=0.25, m_mlp=1.0, checkpoint_attn=0, checkpoint_mlp=0, checkpoint_res=0, attn_order=0, blocks=None, spread=None, encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.encoder_dims = encoder_dims
        self.blocks = blocks
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.prime_len = prime_len
        self.n_head = n_head
        res_scale = 1.0 / n_depth if res_scale else 1.0
        attn_func = {(0): lambda d: 0, (1): lambda d: [1, 2][d % 2], (2): lambda d: [1, 2, 3][d % 3], (3): lambda d: [1, 4][d % 2], (4): lambda d: [1, 5][d % 2], (5): lambda d: [1, 4, 1, 1][d % 4], (6): lambda d: [1, 2, 3, 6][d % 4], (7): lambda d: [*([1, 2, 3] * 5), 6][d % 16], (8): lambda d: [1, 2, 3, 1, 2, 3, 1, 2, 3, 6][d % 10], (9): lambda d: [1, 2, 3, 0][d % 4], (10): lambda d: [*[1, 2, 3, 1, 2, 3, 1, 2, 3], *([1, 2, 3, 1, 2, 3, 1, 2, 3, 6] * 7)][d % 79], (11): lambda d: [6, 6, 0][d % 3] if d % 16 == 15 else [1, 2, 3][d % 3], (12): lambda d: [7, 7, 0][d % 3] if d % 16 == 15 else [1, 2, 3][d % 3]}[attn_order]
        attn_cycle = {(0): 1, (1): 2, (2): 3, (3): 2, (4): 2, (5): 4, (6): 4, (7): 16, (8): 10, (9): 4, (10): 79, (11): 16, (12): 16}[attn_order]
        attn_block = lambda d: ResAttnBlock(n_in=n_in, n_ctx=n_ctx, n_head=n_head, attn_dropout=attn_dropout, resid_dropout=resid_dropout, afn=afn, scale=scale, mask=mask, zero_out=zero_out if attn_func(d) != 6 else True, init_scale=init_scale, res_scale=res_scale, m_attn=m_attn, m_mlp=m_mlp, checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, attn_func=attn_func(d), blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.checkpoint_res = checkpoint_res
        self._attn_mods = nn.ModuleList()
        for d in range(n_depth):
            self._attn_mods.append(attn_block(d))
        self.ws = []

    def set_record_attn(self, record_attn):
        """
        Arguments:
            record_attn (bool or set): Makes forward prop dump self-attention
                softmaxes to self.ws. Either a set of layer indices indicating
                which layers to store, or a boolean value indicating whether to
                dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn
        for i, l in enumerate(self._attn_mods):
            l.attn.record_attn = _should_record_attn(i)
        if record_attn:
            assert self.ws == []
            for l in self._attn_mods:
                assert l.attn.w == None
        else:
            self.ws = []
            for l in self._attn_mods:
                l.attn.w = None

    def forward(self, x, encoder_kv=None, sample=False, fp16=False, fp16_out=False):
        if fp16:
            x = x.half()
        for i, l in enumerate(self._attn_mods):
            if self.checkpoint_res == 1 and not sample:
                if l.attn_func == 6:
                    assert encoder_kv is not None
                    f = functools.partial(l, sample=sample)
                    x = checkpoint(f, (x, encoder_kv), l.parameters(), True)
                else:
                    f = functools.partial(l, encoder_kv=None, sample=sample)
                    x = checkpoint(f, (x,), l.parameters(), True)
            elif l.attn_func == 6:
                x = l(x, encoder_kv=encoder_kv, sample=sample)
            else:
                x = l(x, encoder_kv=None, sample=sample)
            if l.attn.record_attn:
                self.ws.append(l.attn.w)
        if not fp16_out:
            x = x.float()
        return x

    def check_cache(self, n_samples, sample_t, fp16):
        for l in self._attn_mods:
            l.attn.check_cache(n_samples, sample_t, fp16)

    def del_cache(self):
        for l in self._attn_mods:
            l.attn.del_cache()

    def check_sample(self):
        bs, l, s, d = 4, self.n_ctx, self.encoder_dims, self.n_in
        prime = 5
        with t.no_grad():
            encoder_kv = t.randn(bs, s, d)
            x = t.randn(bs, l, d)
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=True)
            self.del_cache()
            x_chunks = t.chunk(x, 4, dim=1)
            y_chunks = []
            n = 0
            for x_chunk in x_chunks:
                self.check_cache(bs, n, False)
                y_chunk = self.forward(x_chunk, encoder_kv=encoder_kv, sample=True)
                y_chunks.append(y_chunk)
                n += x_chunk.shape[1]
            self.check_cache(bs, n, False)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)
            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, (i), :]) > 1e-06]}'


def empty_cache():
    gc.collect()
    t.cuda.empty_cache()


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))
    assert top_k == 0 or top_p == 0.0
    if top_k > 0:
        indices_to_remove = logits < t.topk(logits, top_k, dim=-1)[0][(...), -1:]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = t.sort(logits, descending=True, dim=-1)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[(...), 1:] = sorted_indices_to_remove[(...), :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = t.zeros_like(logits, dtype=t.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def def_tqdm(x):
    return tqdm(x, leave=True, file=sys.stdout, bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')


def get_range(x):
    if dist.get_rank() == 0:
        return def_tqdm(x)
    else:
        return x


def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)


def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [*([chunk_size] * (n_passes - 1)), (length - 1) % chunk_size + 1]
    assert sum(chunk_sizes) == length
    return chunk_sizes


class ConditionalAutoregressive2D(nn.Module):

    def __init__(self, input_shape, bins, width=128, depth=2, heads=1, attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, mask=True, zero_out=False, init_scale=1.0, res_scale=False, pos_init=False, m_attn=0.25, m_mlp=1, checkpoint_res=0, checkpoint_attn=0, checkpoint_mlp=0, attn_order=0, blocks=None, spread=None, x_cond=False, y_cond=False, encoder_dims=0, only_encode=False, merged_decoder=False, prime_len=None):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth
        self.x_emb = nn.Embedding(bins, width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond:
            self.start_token = nn.Parameter(get_normal(1, width, std=0.01 * init_scale))
        self.pos_emb = PositionEmbedding(input_shape=input_shape, width=width, init_scale=init_scale, pos_init=pos_init)
        self.pos_emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(n_in=width, n_ctx=input_dims, n_head=heads, n_depth=depth, attn_dropout=attn_dropout, resid_dropout=resid_dropout, afn='quick_gelu', scale=True, mask=mask, zero_out=zero_out, init_scale=init_scale, res_scale=res_scale, m_attn=m_attn, m_mlp=m_mlp, checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, checkpoint_res=checkpoint_res, attn_order=attn_order, blocks=blocks, spread=spread, encoder_dims=encoder_dims, prime_len=prime_len)
        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            self.add_cond_after_transformer = False
            self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_x_emb_x_out = True
        if not only_encode:
            self.x_out = nn.Linear(width, bins, bias=False)
            if self.share_x_emb_x_out:
                self.x_out.weight = self.x_emb.weight
            self.loss = t.nn.CrossEntropyLoss()

    def preprocess(self, x):
        N = x.shape[0]
        return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        N = x.shape[0]
        assert (0 <= x).all() and (x < self.bins).all()
        if sample_tokens is None or sample_tokens == self.input_dims:
            return x.view(N, *self.input_shape)
        else:
            return x.view(N, -1)

    def forward(self, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False, encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        with t.no_grad():
            x = self.preprocess(x)
        N, D = x.shape
        assert isinstance(x, t.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'{x_cond.shape} != {N, D, self.width} nor {N, 1, self.width}. Did you pass the correct --sample_length?'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), device=x.device, dtype=t.float)
        x_t = x
        x = self.x_emb(x)
        x = roll(x, 1)
        if self.y_cond:
            x[:, (0)] = y_cond.view(N, self.width)
        else:
            x[:, (0)] = self.start_token
        x = self.x_emb_dropout(x) + self.pos_emb_dropout(self.pos_emb()) + x_cond
        x = self.transformer(x, encoder_kv=encoder_kv, fp16=fp16)
        if self.add_cond_after_transformer:
            x = x + x_cond
        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x)
        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, :self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len:].reshape(-1, self.bins)
            prime_loss = F.cross_entropy(x_prime, x_t[:, :self.prime_len].reshape(-1)) / np.log(2.0)
            gen_loss = F.cross_entropy(x_gen, x_t[:, self.prime_len:].reshape(-1)) / np.log(2.0)
            loss = prime_loss, gen_loss
        else:
            loss = F.cross_entropy(x.view(-1, self.bins), x_t.view(-1)) / np.log(2.0)
        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, D = n_samples, self.input_dims
        if sample_t == 0:
            x = t.empty(n_samples, 1, self.width)
            if self.y_cond:
                x[:, (0)] = y_cond.view(N, self.width)
            else:
                x[:, (0)] = self.start_token
        else:
            assert isinstance(x, t.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape == (N, D, self.width):
            cond = x_cond[:, sample_t:sample_t + 1, :]
        else:
            cond = x_cond
        x = x + self.pos_emb()[sample_t:sample_t + 1] + cond
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def sample(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, get_preds=False, sample_tokens=None):
        assert self.training == False
        if sample_tokens is None:
            sample_tokens = self.input_dims
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float)
        with t.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            for sample_t in get_range(range(0, sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16)
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x)
                if get_preds:
                    preds.append(x.clone())
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = t.distributions.Categorical(logits=x).sample()
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())
            del x
            self.transformer.del_cache()
            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def primed_sample(self, n_samples, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None):
        assert self.training == False
        if sample_tokens is None:
            sample_tokens = self.input_dims
        with t.no_grad():
            x = self.preprocess(x)
        assert isinstance(x, t.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        assert x.shape[0] == n_samples
        xs = t.split(x, 1, dim=1)
        xs = list(xs)
        assert len(xs) < sample_tokens
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None
        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f'Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})'
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float)
        with t.no_grad():
            if get_preds:
                preds = []
            if chunk_size is None:
                chunk_size = len(xs)
            chunk_sizes = split_chunks(len(xs), chunk_size)
            x_primes = []
            start = 0
            x = None
            for current_chunk_size in get_range(chunk_sizes):
                xs_prime, conds_prime = [], []
                for sample_t in range(start, start + current_chunk_size):
                    x_prime, cond_prime = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                    x = xs[sample_t]
                    xs_prime.append(x_prime)
                    conds_prime.append(cond_prime)
                start = start + current_chunk_size
                x_prime, cond_prime = t.cat(xs_prime, dim=1), t.cat(conds_prime, dim=1)
                assert x_prime.shape == (n_samples, current_chunk_size, self.width)
                assert cond_prime.shape == (n_samples, current_chunk_size, self.width)
                del xs_prime
                del conds_prime
                if not get_preds:
                    del cond_prime
                x_prime = self.transformer(x_prime, encoder_kv=encoder_kv, sample=True, fp16=fp16)
                if get_preds:
                    if self.add_cond_after_transformer:
                        x_prime = x_prime + cond_prime
                    assert x_prime.shape == (n_samples, current_chunk_size, self.width)
                    del cond_prime
                    x_primes.append(x_prime)
                else:
                    del x_prime
            if get_preds:
                x_prime = t.cat(x_primes, dim=1)
                assert x_prime.shape == (n_samples, len(xs), self.width)
                x_prime = self.x_out(x_prime)
                preds.append(x_prime)
            empty_cache()
            self.transformer.check_cache(n_samples, len(xs), fp16)
            x = xs[-1]
            assert x.shape == (n_samples, 1)
            empty_cache()
            for sample_t in get_range(range(len(xs), sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16)
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x)
                if get_preds:
                    preds.append(x)
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = t.distributions.Categorical(logits=x).sample()
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())
            del x
            self.transformer.del_cache()
            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def check_sample(self, chunk_size):
        bs, l, d = 4, self.input_dims, self.width
        prime = int(self.input_dims // 8 * 7)
        enc_l = self.encoder_dims
        with t.no_grad():
            y_cond = t.randn(bs, 1, d) if self.y_cond else None
            x_cond = t.randn(bs, l, d) if self.x_cond else None
            encoder_kv = t.randn(bs, enc_l, d)
            x, preds_sample = self.sample(bs, x_cond, y_cond, encoder_kv, get_preds=True)
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, (i), :]) > 1e-06]}'
            x_prime = x.view(bs, -1)[:, :prime]
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True)
            assert (x.view(bs, -1)[:, :prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, (i), :]) > 1e-06]}'
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True, chunk_size=chunk_size)
            assert (x.view(bs, -1)[:, :prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-06, f'Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, (i), :]) > 1e-06]}'


class ResConv1DBlock(nn.Module):

    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(nn.ReLU(), nn.Conv1d(n_in, n_state, 3, 1, padding, dilation), nn.ReLU(), nn.Conv1d(n_state, n_in, 1, 1, 0))
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):

    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), dilation=dilation_growth_rate ** _get_depth(depth), zero_out=zero_out, res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth)) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                None
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x,), block.parameters(), True)
            return x
        else:
            return self.model(x)


class DecoderConvBock(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res), nn.ConvTranspose1d(width, input_emb_width if i == down_t - 1 else width, filter_t, stride_t, pad_t))
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f'Expected {exp_shape} got {x.shape}'


class Conditioner(nn.Module):

    def __init__(self, input_shape, bins, down_t, stride_t, out_width, init_scale, zero_out, res_scale, **block_kwargs):
        super().__init__()
        self.x_shape = input_shape
        self.width = out_width
        self.x_emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.cond = DecoderConvBock(self.width, self.width, down_t, stride_t, **block_kwargs, zero_out=zero_out, res_scale=res_scale)
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, x_cond=None):
        N = x.shape[0]
        assert_shape(x, (N, *self.x_shape))
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        x = x.long()
        x = self.x_emb(x)
        assert_shape(x, (N, *self.x_shape, self.width))
        x = x + x_cond
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x


class SimpleEmbedding(nn.Module):

    def __init__(self, bins, out_width, init_scale):
        super().__init__()
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)

    def forward(self, y):
        assert len(y.shape) == 2, f'Expected shape with 2 dims, got {y.shape}'
        assert isinstance(y, t.cuda.LongTensor), f'Expected dtype {t.cuda.LongTensor}, got {y.dtype}'
        assert (0 <= y).all() and (y < self.bins).all(), f'Bins {self.bins}, got label {y}'
        return self.emb(y)


class RangeEmbedding(nn.Module):

    def __init__(self, n_time, bins, range, out_width, init_scale, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        assert len(pos_start.shape) == 2, f'Expected shape with 2 dims, got {pos_start.shape}'
        assert (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all(), f'Range is [{self.pos_min},{self.pos_max}), got {pos_start}'
        pos_start = pos_start.float()
        if pos_end is not None:
            assert len(pos_end.shape) == 2, f'Expected shape with 2 dims, got {pos_end.shape}'
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)
            assert (self.pos_min <= pos_end).all() and (pos_end <= self.pos_max).all(), f'Range is [{self.pos_min},{self.pos_max}), got {pos_end}'
            pos_end = pos_end.float()
        n_time = self.n_time
        if n_time != 1:
            assert pos_end is not None
            interpolation = t.arange(0, n_time, dtype=t.float, device='cuda').view(1, n_time) / n_time
            position = pos_start + (pos_end - pos_start) * interpolation
        else:
            position = pos_start
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min)
        bins = (self.bins * normalised_position).floor().long().detach()
        return self.emb(bins)


class LabelConditioner(nn.Module):

    def __init__(self, y_bins, t_bins, sr, min_duration, max_duration, n_time, out_width, init_scale, max_bow_genre_size, include_time_signal):
        super().__init__()
        self.n_time = n_time
        self.out_width = out_width
        assert len(y_bins) == 2, f'Expecting (genre, artist) bins, got {y_bins}'
        bow_genre_bins, artist_bins = y_bins
        self.max_bow_genre_size = max_bow_genre_size
        self.bow_genre_emb = SimpleEmbedding(bow_genre_bins, out_width, init_scale)
        self.artist_emb = SimpleEmbedding(artist_bins, out_width, init_scale)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            t_ranges = (min_duration * sr, max_duration * sr), (0.0, max_duration * sr), (0.0, 1.0)
            assert len(t_ranges) == 3, f'Expecting (total, absolute, relative) ranges, got {t_ranges}'
            total_length_range, absolute_pos_range, relative_pos_range = t_ranges
            self.total_length_emb = RangeEmbedding(1, t_bins, total_length_range, out_width, init_scale)
            self.absolute_pos_emb = RangeEmbedding(n_time, t_bins, absolute_pos_range, out_width, init_scale)
            self.relative_pos_emb = RangeEmbedding(n_time, t_bins, relative_pos_range, out_width, init_scale, clamp=True)

    def forward(self, y):
        assert len(y.shape) == 2, f'Expected shape with 2 dims, got {y.shape}'
        assert y.shape[-1] == 4 + self.max_bow_genre_size, f'Expected shape (N,{4 + self.max_bow_genre_size}), got {y.shape}'
        assert isinstance(y, t.cuda.LongTensor), f'Expected dtype {t.cuda.LongTensor}, got {y.dtype}'
        N = y.shape[0]
        total_length, offset, length, artist, genre = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:]
        artist_emb = self.artist_emb(artist)
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb
        assert_shape(start_emb, (N, 1, self.out_width))
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length, start, end = total_length.float(), start.float(), end.float()
            pos_emb = self.total_length_emb(total_length) + self.absolute_pos_emb(start, end) + self.relative_pos_emb(start / total_length, end / total_length)
            assert_shape(pos_emb, (N, self.n_time, self.out_width))
        else:
            pos_emb = None
        return start_emb, pos_emb


def create_reverse_lookup(atoi):
    itoa = {}
    for a, i in atoi.items():
        if i not in itoa:
            itoa[i] = []
        itoa[i].append(a)
    indices = sorted(list(itoa.keys()))
    for i in indices:
        itoa[i] = '_'.join(sorted(itoa[i]))
    return itoa


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


class ArtistGenreProcessor:

    def __init__(self, v3=False):
        self.v3 = v3
        dirname = os.path.dirname(__file__)
        if self.v3:
            self.artist_id_file = f'{dirname}/ids/v3_artist_ids.txt'
            self.genre_id_file = f'{dirname}/ids/v3_genre_ids.txt'
        else:
            self.artist_id_file = f'{dirname}/ids/v2_artist_ids.txt'
            self.genre_id_file = f'{dirname}/ids/v2_genre_ids.txt'
        self.load_artists()
        self.load_genres()

    def get_artist_id(self, artist):
        input_artist = artist
        if self.v3:
            artist = artist.lower()
        else:
            artist = norm(artist)
        if artist not in self.artist_ids:
            None
        return self.artist_ids.get(artist, 0)

    def get_genre_ids(self, genre):
        if self.v3:
            genres = [genre.lower()]
        else:
            genres = norm(genre).split('_')
        for word in genres:
            if word not in self.genre_ids:
                None
        return [self.genre_ids.get(word, 0) for word in genres]

    def get_artist(self, artist_id):
        return self.artists[artist_id]

    def get_genre(self, genre_ids):
        if self.v3:
            assert len(genre_ids) == 1
            genre = self.genres[genre_ids[0]]
        else:
            genre = '_'.join([self.genres[genre_id] for genre_id in genre_ids if genre_id >= 0])
        return genre

    def load_artists(self):
        None
        self.artist_ids = {}
        with open(self.artist_id_file, 'r', encoding='utf-8') as f:
            for line in f:
                artist, artist_id = line.strip().split(';')
                self.artist_ids[artist.lower()] = int(artist_id)
        self.artists = create_reverse_lookup(self.artist_ids)

    def load_genres(self):
        None
        self.genre_ids = {}
        with open(self.genre_id_file, 'r', encoding='utf-8') as f:
            for line in f:
                genre, genre_id = line.strip().split(';')
                self.genre_ids[genre.lower()] = int(genre_id)
        self.genres = create_reverse_lookup(self.genre_ids)


class TextProcessor:

    def __init__(self, v3=False):
        if v3:
            vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-\'"()[] \t\n'
            not_vocab = re.compile('[^A-Za-z0-9.,:;!?\\-\'"()\\[\\] \t\n]+')
        else:
            vocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+\'"()[] \t\n'
            not_vocab = re.compile('[^A-Za-z0-9.,:;!?\\-+\'"()\\[\\] \t\n]+')
        self.vocab = {vocab[index]: (index + 1) for index in range(len(vocab))}
        self.vocab['<unk>'] = 0
        self.n_vocab = len(vocab) + 1
        self.tokens = {v: k for k, v in self.vocab.items()}
        self.tokens[0] = ''
        self.not_vocab = not_vocab

    def clean(self, text):
        text = unidecode(text)
        text = text.replace('\\', '\n')
        text = self.not_vocab.sub('', text)
        return text

    def tokenise(self, text):
        return [self.vocab[char] for char in text]

    def textise(self, tokens):
        return ''.join([self.tokens[token] for token in tokens])

    def characterise(self, tokens):
        return [self.tokens[token] for token in tokens]


def get_relevant_lyric_tokens(full_tokens, n_tokens, total_length, offset, duration):
    if len(full_tokens) < n_tokens:
        tokens = [0] * (n_tokens - len(full_tokens)) + full_tokens
        indices = [-1] * (n_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        assert 0 <= offset < total_length
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, n_tokens // 2), len(full_tokens) - n_tokens // 2)
        tokens = full_tokens[midpoint - n_tokens // 2:midpoint + n_tokens // 2]
        indices = list(range(midpoint - n_tokens // 2, midpoint + n_tokens // 2))
    assert len(tokens) == n_tokens, f'Expected length {n_tokens}, got {len(tokens)}'
    assert len(indices) == n_tokens, f'Expected length {n_tokens}, got {len(indices)}'
    assert tokens == [(full_tokens[index] if index != -1 else 0) for index in indices]
    return tokens, indices


class Labeller:

    def __init__(self, max_genre_words, n_tokens, sample_length, v3=False):
        self.ag_processor = ArtistGenreProcessor(v3)
        self.text_processor = TextProcessor(v3)
        self.n_tokens = n_tokens
        self.max_genre_words = max_genre_words
        self.sample_length = sample_length
        self.label_shape = 4 + self.max_genre_words + self.n_tokens,

    def get_label(self, artist, genre, lyrics, total_length, offset):
        artist_id = self.ag_processor.get_artist_id(artist)
        genre_ids = self.ag_processor.get_genre_ids(genre)
        lyrics = self.text_processor.clean(lyrics)
        full_tokens = self.text_processor.tokenise(lyrics)
        tokens, _ = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, self.sample_length)
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f'Expected {self.label_shape}, got {y.shape}'
        info = dict(artist=artist, genre=genre, lyrics=lyrics, full_tokens=full_tokens)
        return dict(y=y, info=info)

    def get_y_from_ids(self, artist_id, genre_ids, lyric_tokens, total_length, offset):
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        if self.n_tokens > 0:
            assert len(lyric_tokens) == self.n_tokens
        else:
            lyric_tokens = []
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *lyric_tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f'Expected {self.label_shape}, got {y.shape}'
        return y

    def get_batch_labels(self, metas, device='cpu'):
        ys, infos = [], []
        for meta in metas:
            label = self.get_label(**meta)
            y, info = label['y'], label['info']
            ys.append(y)
            infos.append(info)
        ys = t.stack([t.from_numpy(y) for y in ys], dim=0).long()
        assert ys.shape[0] == len(metas)
        assert len(infos) == len(metas)
        return dict(y=ys, info=infos)

    def set_y_lyric_tokens(self, ys, labels):
        info = labels['info']
        assert ys.shape[0] == len(info)
        if self.n_tokens > 0:
            tokens_list = []
            indices_list = []
            for i in range(ys.shape[0]):
                full_tokens = info[i]['full_tokens']
                total_length, offset, duration = ys[i, 0], ys[i, 1], ys[i, 2]
                tokens, indices = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, duration)
                tokens_list.append(tokens)
                indices_list.append(indices)
            ys[:, -self.n_tokens:] = t.tensor(tokens_list, dtype=t.long, device='cuda')
            return indices_list
        else:
            return None

    def describe_label(self, y):
        assert y.shape == self.label_shape, f'Expected {self.label_shape}, got {y.shape}'
        y = np.array(y).tolist()
        total_length, offset, length, artist_id, *genre_ids = y[:4 + self.max_genre_words]
        tokens = y[4 + self.max_genre_words:]
        artist = self.ag_processor.get_artist(artist_id)
        genre = self.ag_processor.get_genre(genre_ids)
        lyrics = self.text_processor.textise(tokens)
        return dict(artist=artist, genre=genre, lyrics=lyrics)


def calculate_strides(strides, downs):
    return [(stride ** down) for stride, down in zip(strides, downs)]


def print_once(msg):
    if not dist.is_available() or dist.get_rank() == 0:
        None


class SimplePrior(nn.Module):

    def __init__(self, z_shapes, l_bins, encoder, decoder, level, downs_t, strides_t, labels, prior_kwargs, x_cond_kwargs, y_cond_kwargs, prime_kwargs, copy_input, labels_v3=False, merged_decoder=False, single_enc_dec=False):
        super().__init__()
        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')
        self.copy_input = copy_input
        if self.copy_input:
            prime_kwargs['bins'] = l_bins
        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)
        self.z_shape = self.z_shapes[level]
        self.level = level
        assert level < self.levels, f'Total levels {self.levels}, got level {level}'
        self.l_bins = l_bins
        self.encoder = encoder
        self.decoder = decoder
        self.x_cond = level != self.levels - 1
        self.cond_level = level + 1
        self.y_cond = labels
        self.single_enc_dec = single_enc_dec
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            conditioner_block = lambda _level: Conditioner(input_shape=z_shapes[_level], bins=l_bins, down_t=downs_t[_level], stride_t=strides_t[_level], **x_cond_kwargs)
            if dist.get_rank() == 0:
                None
            self.conditioner_blocks.append(conditioner_block(self.cond_level))
        if self.y_cond:
            self.n_time = self.z_shape[0]
            self.y_emb = LabelConditioner(n_time=self.n_time, include_time_signal=not self.x_cond, **y_cond_kwargs)
        if single_enc_dec:
            self.prior_shapes = [(self.n_tokens,), prior_kwargs.pop('input_shape')]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(f'Creating cond. autoregress with prior bins {self.prior_bins}, ')
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')
            self.prime_loss_dims, self.gen_loss_dims = self.prior_dims[0], self.prior_dims[1]
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(input_shape=(sum(self.prior_dims),), bins=sum(self.prior_bins), x_cond=self.x_cond or self.y_cond, y_cond=True, prime_len=self.prime_loss_dims, **prior_kwargs)
        else:
            if self.n_tokens != 0 and self.use_tokens:
                prime_input_shape = self.n_tokens,
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = prime_kwargs['width'], prior_kwargs['width']
                self.prime_prior = ConditionalAutoregressive2D(input_shape=prime_input_shape, x_cond=False, y_cond=False, only_encode=True, **prime_kwargs)
                self.prime_state_proj = Conv1D(self.prime_acts_width, self.prime_state_width, init_scale=prime_kwargs['init_scale'])
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                self.prime_x_out = nn.Linear(self.prime_state_width, self.prime_bins, bias=False)
                nn.init.normal_(self.prime_x_out.weight, std=0.02 * prior_kwargs['init_scale'])
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(x_cond=self.x_cond or self.y_cond, y_cond=self.y_cond, encoder_dims=self.prime_loss_dims, merged_decoder=merged_decoder, **prior_kwargs)
        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = self.downsamples[level + 1] if level != self.levels - 1 else None
        self.raw_to_tokens = np.prod(self.downsamples[:level + 1])
        self.sample_length = self.n_ctx * self.raw_to_tokens
        if labels:
            self.labels_v3 = labels_v3
            self.labeller = Labeller(self.y_emb.max_bow_genre_size, self.n_tokens, self.sample_length, v3=self.labels_v3)
        None

    def get_y(self, labels, start, get_indices=False):
        y = labels['y'].clone()
        y[:, (2)] = int(self.sample_length)
        y[:, 1:2] = y[:, 1:2] + int(start * self.raw_to_tokens)
        indices = self.labeller.set_y_lyric_tokens(y, labels)
        if get_indices:
            return y, indices
        else:
            return y

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert start % self.cond_downsample == end % self.cond_downsample == 0
            z_cond = zs[self.level + 1][:, start // self.cond_downsample:end // self.cond_downsample]
            assert z_cond.shape[1] == self.n_ctx // self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def prior_preprocess(self, xs, conds):
        N = xs[0].shape[0]
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            assert isinstance(x, t.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            xs[i] = (xs[i] + bins_shift).view(N, -1)
        for i in range(len(conds)):
            cond, shape, dims = conds[i], self.prior_shapes[i], self.prior_dims[i]
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = t.zeros((N, dims, self.prior_width), dtype=t.float, device='cuda')
        return t.cat(xs, dim=1), t.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = self.prior_dims[0], z.shape[1] - self.prior_dims[0]
        xs = list(t.split(z, dims, dim=1))
        for i in range(len(xs)):
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = t.clamp(xs[i], min=0)
            assert (xs[i] < bins).all(), f'rank: {dist.get_rank()}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'
        return xs[-1]

    def x_emb(self, z_conds):
        z_conds = z_conds[:self.cond_level - self.level]
        assert len(z_conds) == len(self.conditioner_blocks) == self.cond_level - self.level, f'Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}'
        x_cond = None
        for z_cond, conditioner_block in reversed(list(zip(z_conds, self.conditioner_blocks))):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond

    def encode(self, x, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        with t.no_grad():
            zs = self.encoder(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return zs

    def decode(self, zs, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        with t.no_grad():
            x_out = self.decoder(zs, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return x_out

    def get_cond(self, z_conds, y):
        if y is not None:
            assert y.shape[1] == 4 + self.y_emb.max_bow_genre_size + self.n_tokens, f'Expected {4} + {self.y_emb.max_bow_genre_size} + {self.n_tokens}, got {y.shape[1]}'
            n_labels = y.shape[1] - self.n_tokens
            y, prime = y[:, :n_labels], y[:, n_labels:]
        else:
            y, prime = None, None
        y_cond, y_pos = self.y_emb(y) if self.y_cond else (None, None)
        x_cond = self.x_emb(z_conds) if self.x_cond else y_pos
        return x_cond, y_cond, prime

    def sample(self, n_samples, z=None, z_conds=None, y=None, fp16=False, temp=1.0, top_k=0, top_p=0.0, chunk_size=None, sample_tokens=None):
        N = n_samples
        if z is not None:
            assert z.shape[0] == N, f'Expected shape ({N},**), got shape {z.shape}'
        if y is not None:
            assert y.shape[0] == N, f'Expected shape ({N},**), got shape {y.shape}'
        if z_conds is not None:
            for z_cond in z_conds:
                assert z_cond.shape[0] == N, f'Expected shape ({N},**), got shape {z_cond.shape}'
        no_past_context = z is None or z.shape[1] == 0
        if dist.get_rank() == 0:
            name = {(True): 'Ancestral', (False): 'Primed'}[no_past_context]
            None
        with t.no_grad():
            x_cond, y_cond, prime = self.get_cond(z_conds, y)
            if self.single_enc_dec:
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                z = self.prior_postprocess(z)
            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                if no_past_context:
                    z = self.prior.sample(n_samples, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, sample_tokens=sample_tokens)
                else:
                    z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
            if sample_tokens is None:
                assert_shape(z, (N, *self.z_shape))
        return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):
        if self.n_tokens != 0 and self.use_tokens:
            if sample:
                self.prime_prior
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)
            assert_shape(prime_acts, (N, self.prime_loss_dims, self.prime_acts_width))
            assert prime_acts.dtype == t.float, f'Expected t.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == t.float, f'Expected t.float, got {encoder_kv.dtype}'
            if sample:
                self.prime_prior.cpu()
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)) / np.log(2.0)
        else:
            prime_loss = t.tensor(0.0, device='cuda')
        return prime_loss

    def z_forward(self, z, z_conds=[], y=None, fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:, :self.n_tokens]
        if self.single_enc_dec:
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
            (prime_loss, gen_loss), preds = self.prior(z, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
        loss = self.prime_loss_fraction * prime_loss * self.prime_loss_dims / self.total_loss_dims + gen_loss * self.gen_loss_dims / self.total_loss_dims
        metrics = dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(), gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics['preds'] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False):
        bs = x.shape[0]
        z, *z_conds = self.encode(x, bs_chunks=bs)
        loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics


class Mask(nn.Module):

    def __init__(self, n_ctx):
        super().__init__()
        self.register_buffer('b', t.tril(t.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, w):
        w = w * self.b + -1000000000.0 * (1 - self.b)
        return w


class BottleneckBlock(nn.Module):

    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        y = self._tile(x)
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]
        dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = t.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = t.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with t.no_grad():
            x_l_onehot = t.zeros(k_bins, x.shape[0], device=x.device)
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)
            _k_sum = t.matmul(x_l_onehot, x)
            _k_elem = x_l_onehot.sum(dim=-1)
            y = self._tile(x)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]
            dist.broadcast(_k_rand, 0)
            dist.all_reduce(_k_sum)
            dist.all_reduce(_k_elem)
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1.0 - mu) * _k_sum
            self.k_elem = mu * self.k_elem + (1.0 - mu) * _k_elem
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) + (1 - usage) * _k_rand
            _k_prob = _k_elem / t.sum(_k_elem)
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-08))
            used_curr = (_k_elem >= self.threshold).sum()
            usage = t.sum(usage)
            dk = t.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        if x.shape[-1] == self.emb_width:
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[(...), :self.emb_width], x[(...), self.emb_width:]
            prenorm = t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape)) + t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape))
            x = x1 + x2
        else:
            assert False, f'Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}'
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        k_w = self.k.t()
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0, keepdim=True)
        min_distance, x_l = t.min(distance, dim=-1)
        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape
        x, prenorm = self.preprocess(x)
        x_l, fit = self.quantise(x)
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width
        x_d = self.dequantise(x_l)
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape
        x, prenorm = self.preprocess(x)
        if update_k and not self.init:
            self.init_k(x)
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}
        commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)
        x_d = x + (x_d - x).detach()
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):

    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for level_block, x in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.decode(z) for level_block, z in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics


class NoBottleneckBlock(nn.Module):

    def restore_k(self):
        pass


class NoBottleneck(nn.Module):

    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for level in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        return zs

    def forward(self, xs):
        zero = t.zeros(())
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics


class EncoderConvBlock(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t), Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale))
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // stride_t ** down_t
            assert_shape(x, (N, emb, T))
            xs.append(x)
        return xs


class Decoder(nn.Module):

    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t
        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * stride_t ** down_t
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]
        x = self.out(x)
        return x


class ResConvBlock(nn.Module):

    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(nn.ReLU(), nn.Conv2d(n_in, n_state, 3, 1, 1), nn.ReLU(), nn.Conv2d(n_state, n_in, 1, 1, 0))

    def forward(self, x):
        return x + self.model(x)


class Resnet(nn.Module):

    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)


def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f'Unknown loss_fn {loss_fn}'


def audio_postprocess(x, hps):
    return x


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: (sum(vals) / len(vals)) for key, vals in metrics.items()}


class STFTValues:

    def __init__(self, hps, n_fft, hop_length, window_size):
        self.sr = hps.sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size


def stft(sig, hps):
    return t.stft(sig, hps.n_fft, hps.hop_length, win_length=hps.window_size, window=t.hann_window(hps.window_size, device=sig.device))


def spec(x, hps):
    return t.norm(stft(x, hps), p=2, dim=-1)


def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = t.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x


def multispectral_loss(x_in, x_out, hps):
    losses = []
    assert len(hps.multispec_loss_n_fft) == len(hps.multispec_loss_hop_length) == len(hps.multispec_loss_window_size)
    args = [hps.multispec_loss_n_fft, hps.multispec_loss_hop_length, hps.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        hps = STFTValues(hps, n_fft, hop_length, window_size)
        spec_in = spec(squeeze(x_in.float()), hps)
        spec_out = spec(squeeze(x_out.float()), hps)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)


class DefaultSTFTValues:

    def __init__(self, hps):
        self.sr = hps.sr
        self.n_fft = 2048
        self.hop_length = 256
        self.window_size = 6 * self.hop_length


def spectral_convergence(x_in, x_out, hps, epsilon=0.002):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return residual_norm * mask / t.clamp(gt_norm, min=epsilon)


def spectral_loss(x_in, x_out, hps):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    return norm(spec_in - spec_out)


class VQVAE(nn.Module):

    def __init__(self, input_shape, levels, downs_t, strides_t, emb_width, l_bins, mu, commit, spectral, multispectral, multipliers=None, use_bottleneck=True, **block_kwargs):
        super().__init__()
        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels
        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, 'Invalid number of multipliers'
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs['width'] *= self.multipliers[level]
            this_block_kwargs['depth'] *= self.multipliers[level]
            return this_block_kwargs
        encoder = lambda level: Encoder(x_channels, emb_width, level + 1, downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1, downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))
        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)
        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral

    def preprocess(self, x):
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, hps, loss_fn='l1'):
        metrics = {}
        N = x.shape[0]
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            sl = t.mean(sl)
            return sl
        recons_loss = t.zeros(())
        spec_loss = t.zeros(())
        multispec_loss = t.zeros(())
        x_target = audio_postprocess(x.float(), hps)
        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss
        commit_loss = sum(commit_losses)
        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn('l2', x_target, x_out, hps)
            l1_loss = _loss_fn('l1', x_target, x_out, hps)
            linf_loss = _loss_fn('linf', x_target, x_out, hps)
        quantiser_metrics = average_metrics(quantiser_metrics)
        metrics.update(dict(recons_loss=recons_loss, spectral_loss=spec_loss, multispectral_loss=multispec_loss, spectral_convergence=sc, l2_loss=l2_loss, l1_loss=l1_loss, linf_loss=linf_loss, commit_loss=commit_loss, **quantiser_metrics))
        for key, val in metrics.items():
            metrics[key] = val.detach()
        return x_out, loss, metrics


class M(nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.cn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=128, out_features=2)

    def forward(self, i):
        i = self.cn1(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = self.cn2(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = i.view(len(i), -1)
        i = self.fc1(i)
        i = F.log_softmax(i, dim=1)
        return i


class LinearInLinear(nn.Module):

    def __init__(self):
        super(LinearInLinear, self).__init__()
        self.l = nn.Linear(3, 5)

    def forward(self, x):
        return self.l(x)


class MultipleInput(nn.Module):

    def __init__(self):
        super(MultipleInput, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)

    def forward(self, x, y):
        return self.Linear_1(x + y)


class MultipleOutput(nn.Module):

    def __init__(self):
        super(MultipleOutput, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)
        self.Linear_2 = nn.Linear(3, 7)

    def forward(self, x):
        return self.Linear_1(x), self.Linear_2(x)


class MultipleOutput_shared(nn.Module):

    def __init__(self):
        super(MultipleOutput_shared, self).__init__()
        self.Linear_1 = nn.Linear(3, 5)

    def forward(self, x):
        return self.Linear_1(x), self.Linear_1(x)


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = Net1()

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


n_categories = 10


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden, input

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'l_bins': 4, 'emb_width': 4, 'mu': 4, 'levels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1D,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecoderConvBock,
     lambda: ([], {'input_emb_width': 4, 'output_emb_width': 4, 'down_t': 4, 'stride_t': 1, 'width': 4, 'depth': 1, 'm_conv': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (DummyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (EncoderConvBlock,
     lambda: ([], {'input_emb_width': 4, 'output_emb_width': 4, 'down_t': 4, 'stride_t': 1, 'width': 4, 'depth': 1, 'm_conv': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (FactoredAttention,
     lambda: ([], {'n_in': 4, 'n_ctx': 4, 'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mask,
     lambda: ([], {'n_ctx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoBottleneck,
     lambda: ([], {'levels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionEmbedding,
     lambda: ([], {'input_shape': 4, 'width': 4}),
     lambda: ([], {}),
     False),
    (ResConv1DBlock,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ResConvBlock,
     lambda: ([], {'n_in': 4, 'n_state': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet,
     lambda: ([], {'n_in': 4, 'n_depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resnet1D,
     lambda: ([], {'n_in': 4, 'n_depth': 1}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (SimpleModel,
     lambda: ([], {}),
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

class Test_openai_jukebox(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

