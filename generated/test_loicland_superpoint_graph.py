import sys
_module = sys.modules[__name__]
del sys
learning = _module
custom_dataset = _module
GraphConvInfo = _module
GraphConvModule = _module
GraphPoolInfo = _module
GraphPoolModule = _module
ecc = _module
cuda_kernels = _module
test_GraphConvModule = _module
test_GraphPoolModule = _module
utils = _module
evaluate = _module
graphnet = _module
main = _module
metrics = _module
modules = _module
pointnet = _module
s3dis_dataset = _module
sema3d_dataset = _module
spg = _module
vkitti_dataset = _module
partition = _module
graphs = _module
ply_c = _module
provider = _module
visualize = _module
write_Semantic3d = _module
supervized_partition = _module
evaluate_partition = _module
folderhierarchy = _module
generate_partition = _module
graph_processing = _module
losses = _module
supervized_partition = _module

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


import random


import numpy as np


import functools


import torch


from collections import defaultdict


import torch.nn as nn


from torch.autograd import Variable


from torch.autograd import Function


from collections import namedtuple


from torch.autograd import gradcheck


import torch.nn.init as init


import time


import math


import logging


import torch.optim as optim


from torch.optim.lr_scheduler import MultiStepLR


import torch.nn.functional as nnf


from sklearn.linear_model import RANSACRegressor


from sklearn import preprocessing


class GraphConvFunction(Function):
    """Computes operations for each edge and averages the results over respective nodes.
    The operation is either matrix-vector multiplication (for 3D weight tensors) or element-wise
    vector-vector multiplication (for 2D weight tensors). The evaluation is computed in blocks of
    size `edge_mem_limit` to reduce peak memory load. See `GraphConvInfo` for info on `idxn, idxe, degs`.
    """

    def init(self, in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=1e+20):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._idxn = idxn
        self._idxe = idxe
        self._degs = degs
        self._degs_gpu = degs_gpu
        self._shards = utils.get_edge_shards(degs, edge_mem_limit)

    def _multiply(ctx, a, b, out, f_a=None, f_b=None):
        """Performs operation on edge weights and node signal"""
        if ctx._full_weight_mat:
            torch.bmm(f_a(a) if f_a else a, f_b(b) if f_b else b, out=out)
        else:
            torch.mul(a, b.expand_as(a), out=out)

    @staticmethod
    def forward(ctx, input, weights, in_channels, out_channels, idxn, idxe, degs, degs_gpu, edge_mem_limit=1e+20):
        ctx.save_for_backward(input, weights)
        ctx._in_channels = in_channels
        ctx._out_channels = out_channels
        ctx._idxn = idxn
        ctx._idxe = idxe
        ctx._degs = degs
        ctx._degs_gpu = degs_gpu
        ctx._shards = utils.get_edge_shards(degs, edge_mem_limit)
        ctx._full_weight_mat = weights.dim() == 3
        assert ctx._full_weight_mat or in_channels == out_channels and weights.size(1) == in_channels
        output = input.new(degs.numel(), out_channels)
        startd, starte = 0, 0
        for numd, nume in ctx._shards:
            sel_input = torch.index_select(input, 0, idxn.narrow(0, starte, nume))
            if ctx._idxe is not None:
                sel_weights = torch.index_select(weights, 0, idxe.narrow(0, starte, nume))
            else:
                sel_weights = weights.narrow(0, starte, nume)
            products = input.new()
            GraphConvFunction._multiply(ctx, sel_input, sel_weights, products, lambda a: a.unsqueeze(1))
            if ctx._idxn.is_cuda:
                cuda_kernels.conv_aggregate_fw(output.narrow(0, startd, numd), products.view(-1, ctx._out_channels), ctx._degs_gpu.narrow(0, startd, numd))
            else:
                k = 0
                for i in range(startd, startd + numd):
                    if ctx._degs[i] > 0:
                        torch.mean(products.narrow(0, k, ctx._degs[i]), 0, out=output[i])
                    else:
                        output[i].fill_(0)
                    k = k + ctx._degs[i]
            startd += numd
            starte += nume
            del sel_input, sel_weights, products
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        grad_input = input.new(input.size()).fill_(0)
        grad_weights = weights.new(weights.size())
        if ctx._idxe is not None:
            grad_weights.fill_(0)
        startd, starte = 0, 0
        for numd, nume in ctx._shards:
            grad_products, tmp = input.new(nume, ctx._out_channels), input.new()
            if ctx._idxn.is_cuda:
                cuda_kernels.conv_aggregate_bw(grad_products, grad_output.narrow(0, startd, numd), ctx._degs_gpu.narrow(0, startd, numd))
            else:
                k = 0
                for i in range(startd, startd + numd):
                    if ctx._degs[i] > 0:
                        torch.div(grad_output[i], ctx._degs[i], out=grad_products[k])
                        if ctx._degs[i] > 1:
                            grad_products.narrow(0, k + 1, ctx._degs[i] - 1).copy_(grad_products[k].expand(ctx._degs[i] - 1, 1, ctx._out_channels).squeeze(1))
                        k = k + ctx._degs[i]
            sel_input = torch.index_select(input, 0, ctx._idxn.narrow(0, starte, nume))
            if ctx._idxe is not None:
                GraphConvFunction._multiply(ctx, sel_input, grad_products, tmp, lambda a: a.unsqueeze(1).transpose_(2, 1), lambda b: b.unsqueeze(1))
                grad_weights.index_add_(0, ctx._idxe.narrow(0, starte, nume), tmp)
            else:
                GraphConvFunction._multiply(ctx, sel_input, grad_products, grad_weights.narrow(0, starte, nume), lambda a: a.unsqueeze(1).transpose_(2, 1), lambda b: b.unsqueeze(1))
            if ctx._idxe is not None:
                torch.index_select(weights, 0, ctx._idxe.narrow(0, starte, nume), out=tmp)
                GraphConvFunction._multiply(ctx, grad_products, tmp, sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2, 1))
                del tmp
            else:
                GraphConvFunction._multiply(ctx, grad_products, weights.narrow(0, starte, nume), sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2, 1))
            grad_input.index_add_(0, ctx._idxn.narrow(0, starte, nume), sel_input)
            startd += numd
            starte += nume
            del grad_products, sel_input
        return grad_input, grad_weights, None, None, None, None, None, None, None


class GraphConvModule(nn.Module):
    """ Computes graph convolution using filter weights obtained from a filter generating network (`filter_net`).
        The input should be a 2D tensor of size (# nodes, `in_channels`). Multiple graphs can be concatenated in the same tensor (minibatch).
    
    Parameters:
    in_channels: number of input channels
    out_channels: number of output channels
    filter_net: filter-generating network transforming a 2D tensor (# edges, # edge features) to (# edges, in_channels*out_channels) or (# edges, in_channels)
    gc_info: GraphConvInfo object containing graph(s) structure information, can be also set with `set_info()` method.
    edge_mem_limit: block size (number of evaluated edges in parallel) for convolution evaluation, a low value reduces peak memory. 
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None, edge_mem_limit=1e+20):
        super(GraphConvModule, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, input):
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)
        weights = self._fnet(edgefeats)
        assert input.dim() == 2 and weights.dim() == 2 and (weights.size(1) == self._in_channels * self._out_channels or self._in_channels == self._out_channels and weights.size(1) == self._in_channels)
        if weights.size(1) == self._in_channels * self._out_channels:
            weights = weights.view(-1, self._in_channels, self._out_channels)
        return GraphConvFunction(self._in_channels, self._out_channels, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(input, weights)


class GraphConvModulePureAutograd(nn.Module):
    """
    Autograd-only equivalent of `GraphConvModule` + `GraphConvFunction`. Unfortunately, autograd needs to store intermediate products, which makes the module work only for very small graphs. The module is kept for didactic purposes only.
    """

    def __init__(self, in_channels, out_channels, filter_net, gc_info=None):
        super(GraphConvModulePureAutograd, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        self.set_info(gc_info)

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, input):
        idxn, idxe, degs, edgefeats = self._gci.get_buffers()
        idxn = Variable(idxn, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)
        weights = self._fnet(edgefeats)
        assert input.dim() == 2 and weights.dim() == 2 and weights.size(1) == self._in_channels * self._out_channels
        weights = weights.view(-1, self._in_channels, self._out_channels)
        if idxe is not None:
            idxe = Variable(idxe, requires_grad=False)
            weights = torch.index_select(weights, 0, idxe)
        sel_input = torch.index_select(input, 0, idxn)
        products = torch.bmm(sel_input.view(-1, 1, self._in_channels), weights)
        output = Variable(input.data.new(len(degs), self._out_channels))
        k = 0
        for i in range(len(degs)):
            if degs[i] > 0:
                output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(products.narrow(0, k, degs[i]), 0).view(1, -1))
            else:
                output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
            k = k + degs[i]
        return output


class GraphPoolFunction(Function):
    """ Computes node feature aggregation for each node of the coarsened graph. The evaluation is computed in blocks of size `edge_mem_limit` to reduce peak memory load. See `GraphPoolInfo` for info on `idxn, degs`.
    """
    AGGR_MEAN = 0
    AGGR_MAX = 1

    def __init__(self, idxn, degs, degs_gpu, aggr, edge_mem_limit=1e+20):
        super(GraphPoolFunction, self).__init__()
        self._idxn = idxn
        self._degs = degs
        self._degs_gpu = degs_gpu
        self._aggr = aggr
        self._shards = utils.get_edge_shards(degs, edge_mem_limit)

    def forward(self, input):
        output = input.new(self._degs.numel(), input.size(1))
        if self._aggr == GraphPoolFunction.AGGR_MAX:
            self._max_indices = self._idxn.new(self._degs.numel(), input.size(1)).fill_(-1)
        self._input_size = input.size()
        startd, starte = 0, 0
        for numd, nume in self._shards:
            sel_input = torch.index_select(input, 0, self._idxn.narrow(0, starte, nume))
            if self._idxn.is_cuda:
                if self._aggr == GraphPoolFunction.AGGR_MEAN:
                    cuda_kernels.avgpool_fw(output.narrow(0, startd, numd), sel_input, self._degs_gpu.narrow(0, startd, numd))
                elif self._aggr == GraphPoolFunction.AGGR_MAX:
                    cuda_kernels.maxpool_fw(output.narrow(0, startd, numd), self._max_indices.narrow(0, startd, numd), sel_input, self._degs_gpu.narrow(0, startd, numd))
            else:
                k = 0
                for i in range(startd, startd + numd):
                    if self._degs[i] > 0:
                        if self._aggr == GraphPoolFunction.AGGR_MEAN:
                            torch.mean(sel_input.narrow(0, k, self._degs[i]), 0, out=output[i])
                        elif self._aggr == GraphPoolFunction.AGGR_MAX:
                            torch.max(sel_input.narrow(0, k, self._degs[i]), 0, out=(output[i], self._max_indices[i]))
                    else:
                        output[i].fill_(0)
                    k = k + self._degs[i]
            startd += numd
            starte += nume
            del sel_input
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(self._input_size).fill_(0)
        startd, starte = 0, 0
        for numd, nume in self._shards:
            grad_sel_input = grad_output.new(nume, grad_output.size(1))
            if self._idxn.is_cuda:
                if self._aggr == GraphPoolFunction.AGGR_MEAN:
                    cuda_kernels.avgpool_bw(grad_input, self._idxn.narrow(0, starte, nume), grad_output.narrow(0, startd, numd), self._degs_gpu.narrow(0, startd, numd))
                elif self._aggr == GraphPoolFunction.AGGR_MAX:
                    cuda_kernels.maxpool_bw(grad_input, self._idxn.narrow(0, starte, nume), self._max_indices.narrow(0, startd, numd), grad_output.narrow(0, startd, numd), self._degs_gpu.narrow(0, startd, numd))
            else:
                k = 0
                for i in range(startd, startd + numd):
                    if self._degs[i] > 0:
                        if self._aggr == GraphPoolFunction.AGGR_MEAN:
                            torch.div(grad_output[i], self._degs[i], out=grad_sel_input[k])
                            if self._degs[i] > 1:
                                grad_sel_input.narrow(0, k + 1, self._degs[i] - 1).copy_(grad_sel_input[k].expand(self._degs[i] - 1, 1, grad_output.size(1)))
                        elif self._aggr == GraphPoolFunction.AGGR_MAX:
                            grad_sel_input.narrow(0, k, self._degs[i]).fill_(0).scatter_(0, self._max_indices[i].view(1, -1), grad_output[i].view(1, -1))
                        k = k + self._degs[i]
                grad_input.index_add_(0, self._idxn.narrow(0, starte, nume), grad_sel_input)
            startd += numd
            starte += nume
            del grad_sel_input
        return grad_input


class GraphPoolModule(nn.Module):
    """ Performs graph pooling.
        The input should be a 2D tensor of size (# nodes, `in_channels`). Multiple graphs can be concatenated in the same tensor (minibatch).    
    
    Parameters:
    aggr: aggregation type (GraphPoolFunction.AGGR_MEAN, GraphPoolFunction.AGGR_MAX)
    gp_info: GraphPoolInfo object containing node mapping information, can be also set with `set_info()` method.
    edge_mem_limit: block size (number of evaluated edges in parallel), a low value reduces peak memory.
    """

    def __init__(self, aggr, gp_info=None, edge_mem_limit=1e+20):
        super(GraphPoolModule, self).__init__()
        self._aggr = aggr
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gp_info)

    def set_info(self, gp_info):
        self._gpi = gp_info

    def forward(self, input):
        idxn, degs, degs_gpu = self._gpi.get_buffers()
        return GraphPoolFunction(idxn, degs, degs_gpu, self._aggr, self._edge_mem_limit)(input)


class GraphAvgPoolModule(GraphPoolModule):

    def __init__(self, gp_info=None, edge_mem_limit=1e+20):
        super(GraphAvgPoolModule, self).__init__(GraphPoolFunction.AGGR_MEAN, gp_info, edge_mem_limit)


class GraphMaxPoolModule(GraphPoolModule):

    def __init__(self, gp_info=None, edge_mem_limit=1e+20):
        super(GraphMaxPoolModule, self).__init__(GraphPoolFunction.AGGR_MAX, gp_info, edge_mem_limit)


class ECC_CRFModule(nn.Module):
    """
    Adapted "Conditional Random Fields as Recurrent Neural Networks" (https://arxiv.org/abs/1502.03240)
    `propagation` should be ECC with Filter generating network producing 2D matrix.
    """

    def __init__(self, propagation, nrepeats=1):
        super(ECC_CRFModule, self).__init__()
        self._propagation = propagation
        self._nrepeats = nrepeats

    def forward(self, input):
        Q = nnf.softmax(input)
        for i in range(self._nrepeats):
            Q = self._propagation(Q)
            Q = input - Q
            if i < self._nrepeats - 1:
                Q = nnf.softmax(Q)
        return Q


class GRUCellEx(nn.GRUCell):
    """ Usual GRU cell extended with layer normalization and input gate.
    """

    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(GRUCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-05, affine=False, track_running_stats=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-05, affine=False, track_running_stats=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm:
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = torch.sigmoid(self._modules['ig'](hidden)) * input
        if input.is_cuda and torch.__version__.split('.')[0] == '0':
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try:
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except:
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)
        gi = nnf.linear(input, self.weight_ih)
        gh = nnf.linear(hidden, self.weight_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        bih_r, bih_i, bih_n = self.bias_ih.chunk(3)
        bhh_r, bhh_i, bhh_n = self.bias_hh.chunk(3)
        resetgate = torch.sigmoid(i_r + bih_r + h_r + bhh_r)
        inputgate = torch.sigmoid(i_i + bih_i + h_i + bhh_i)
        newgate = torch.tanh(i_n + bih_n + resetgate * (h_n + bhh_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def __repr__(self):
        s = super(GRUCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class LSTMCellEx(nn.LSTMCell):
    """ Usual LSTM cell extended with layer normalization and input gate.
    """

    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(LSTMCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-05, affine=False, track_running_stats=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-05, affine=False, track_running_stats=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm:
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
        if self._ingate:
            input = torch.sigmoid(self._modules['ig'](hidden[0])) * input
        if input.is_cuda and torch.__version__.split('.')[0] == '0':
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden[0], self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.LSTMFused
            try:
                return state.apply(gi, gh, hidden[1]) if self.bias_ih is None else state.apply(gi, gh, hidden[1], self.bias_ih, self.bias_hh)
            except:
                return state()(gi, gh, hidden[1]) if self.bias_ih is None else state()(gi, gh, hidden[1], self.bias_ih, self.bias_hh)
        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden[0], self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)
        ingate, forgetgate, cellgate, outgate = (gi + gh).chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * hidden[1] + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, cy

    def __repr__(self):
        s = super(LSTMCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class RNNGraphConvModule(nn.Module):
    """
    Computes recurrent graph convolution using filter weights obtained from a Filter generating network (`filter_net`).
    Its result is passed to RNN `cell` and the process is repeated over `nrepeats` iterations.
    Weight sharing over iterations is done both in RNN cell and in Filter generating network.
    """

    def __init__(self, cell, filter_net, nfeat, vv=True, gc_info=None, nrepeats=1, cat_all=False, edge_mem_limit=1e+20, use_pyg=True, cuda=True):
        super(RNNGraphConvModule, self).__init__()
        self._cell = cell
        self._isLSTM = 'LSTM' in type(cell).__name__
        self._fnet = filter_net
        self._nrepeats = nrepeats
        self._cat_all = cat_all
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)
        self.use_pyg = use_pyg
        if use_pyg:
            self.nn = NNConv(nfeat, nfeat, vv=vv)
            if cuda:
                self.nn = self.nn

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, hx):
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edge_indexes = self._gci.get_pyg_buffers()
        weights = self._fnet(edgefeats)
        nc = hx.size(1)
        assert hx.dim() == 2 and weights.dim() == 2 and weights.size(1) in [nc, nc * nc]
        if weights.size(1) != nc:
            weights = weights.view(-1, nc, nc)
        hxs = [hx]
        if self._isLSTM:
            cx = Variable(hx.data.new(hx.size()).fill_(0))
        for r in range(self._nrepeats):
            if self.use_pyg:
                input = self.nn(hx, edge_indexes, weights)
            else:
                input = ecc.GraphConvFunction.apply(hx, weights, nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)
            if self._isLSTM:
                hx, cx = self._cell(input, (hx, cx))
            else:
                hx = self._cell(input, hx)
            hxs.append(hx)
        return torch.cat(hxs, 1) if self._cat_all else hx


def create_fnet(widths, orthoinit, llbias, bnidx=-1):
    """ Creates feature-generating network, a multi-layer perceptron.
    Parameters:
    widths: list of widths of layers (including input and output widths)
    orthoinit: whether to use orthogonal weight initialization
    llbias: whether to use bias in the last layer
    bnidx: index of batch normalization (-1 if not used)
    """
    fnet_modules = []
    for k in range(len(widths) - 2):
        fnet_modules.append(nn.Linear(widths[k], widths[k + 1]))
        if orthoinit:
            init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        if bnidx == k:
            fnet_modules.append(nn.BatchNorm1d(widths[k + 1]))
        fnet_modules.append(nn.ReLU(True))
    fnet_modules.append(nn.Linear(widths[-2], widths[-1], bias=llbias))
    if orthoinit:
        init.orthogonal_(fnet_modules[-1].weight)
    if bnidx == len(widths) - 1:
        fnet_modules.append(nn.BatchNorm1d(fnet_modules[-1].weight.size(0)))
    return nn.Sequential(*fnet_modules)


class GraphNetwork(nn.Module):
    """ It is constructed in a flexible way based on `config` string, which contains sequence of comma-delimited layer definiton tokens layer_arg1_arg2_... See README.md for examples.
    """

    def __init__(self, config, nfeat, fnet_widths, fnet_orthoinit=True, fnet_llbias=True, fnet_bnidx=-1, edge_mem_limit=1e+20, use_pyg=True, cuda=True):
        super(GraphNetwork, self).__init__()
        self.gconvs = []
        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')
            if conf[0] == 'f':
                self.add_module(str(d), nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0] == 'b':
                self.add_module(str(d), nn.BatchNorm1d(nfeat, eps=1e-05, affine=len(conf) == 1))
            elif conf[0] == 'r':
                self.add_module(str(d), nn.ReLU(True))
            elif conf[0] == 'd':
                self.add_module(str(d), nn.Dropout(p=float(conf[1]), inplace=False))
            elif conf[0] == 'crf':
                nrepeats = int(conf[1])
                fnet = create_fnet(fnet_widths + [nfeat * nfeat], fnet_orthoinit, fnet_llbias, fnet_bnidx)
                gconv = ecc.GraphConvModule(nfeat, nfeat, fnet, edge_mem_limit=edge_mem_limit)
                crf = ECC_CRFModule(gconv, nrepeats)
                self.add_module(str(d), crf)
                self.gconvs.append(gconv)
            elif conf[0] == 'gru' or conf[0] == 'lstm':
                nrepeats = int(conf[1])
                vv = bool(int(conf[2])) if len(conf) > 2 else True
                layernorm = bool(int(conf[3])) if len(conf) > 3 else True
                ingate = bool(int(conf[4])) if len(conf) > 4 else True
                cat_all = bool(int(conf[5])) if len(conf) > 5 else True
                fnet = create_fnet(fnet_widths + [nfeat ** 2 if not vv else nfeat], fnet_orthoinit, fnet_llbias, fnet_bnidx)
                if conf[0] == 'gru':
                    cell = GRUCellEx(nfeat, nfeat, bias=True, layernorm=layernorm, ingate=ingate)
                else:
                    cell = LSTMCellEx(nfeat, nfeat, bias=True, layernorm=layernorm, ingate=ingate)
                gconv = RNNGraphConvModule(cell, fnet, nfeat, vv=vv, nrepeats=nrepeats, cat_all=cat_all, edge_mem_limit=edge_mem_limit, use_pyg=use_pyg, cuda=cuda)
                self.add_module(str(d), gconv)
                self.gconvs.append(gconv)
                if cat_all:
                    nfeat *= nrepeats + 1
            elif len(conf[0]) > 0:
                raise NotImplementedError('Unknown module: ' + conf[0])

    def set_info(self, gc_infos, cuda):
        """ Provides convolution modules with graph structure information for the current batch.
        """
        gc_infos = gc_infos if isinstance(gc_infos, (list, tuple)) else [gc_infos]
        for i, gc in enumerate(self.gconvs):
            if cuda:
                gc_infos[i]
            gc.set_info(gc_infos[i])

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class STNkD(nn.Module):
    """
    Spatial Transformer Net for PointNet, producing a KxK transformation matrix.
    Parameters:
      nfeat: number of input features
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
    """

    def __init__(self, nfeat, nf_conv, nf_fc, K=2, norm='batch', affine=True, n_group=1):
        super(STNkD, self).__init__()
        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i - 1] if i > 0 else nfeat, nf_conv[i], 1))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_conv[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1, nf_conv[i]))
            elif norm == 'group':
                modules.append(nn.GroupNorm(n_group, nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)
        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i - 1] if i > 0 else nf_conv[-1], nf_fc[i]))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_fc[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1, nf_fc[i]))
            elif norm == 'group':
                modules.append(nn.GroupNorm(n_group, nf_fc[i]))
            modules.append(nn.ReLU(True))
        self.fcs = nn.Sequential(*modules)
        self.proj = nn.Linear(nf_fc[-1], K * K)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
        self.eye = torch.eye(K).unsqueeze(0)

    def forward(self, input):
        self.eye = self.eye if input.is_cuda else self.eye
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        input = self.fcs(input)
        input = self.proj(input)
        return input.view(-1, self.eye.size(1), self.eye.size(2)) + Variable(self.eye)


class PointNet(nn.Module):
    """
    PointNet with only one spatial transformer and additional "global" input concatenated after maxpool.
    Parameters:
      nf_conv: list of layer widths of point embeddings (before maxpool)
      nf_fc: list of layer widths of joint embeddings (after maxpool)
      nfeat: number of input features
      nf_conv_stn, nf_fc_stn, nfeat_stn: as above but for Spatial transformer
      nfeat_global: number of features concatenated after maxpooling
      prelast_do: dropout after the pre-last parameteric layer
      last_ac: whether to use batch norm and relu after the last parameteric layer
    """

    def __init__(self, nf_conv, nf_fc, nf_conv_stn, nf_fc_stn, nfeat, nfeat_stn=2, nfeat_global=1, prelast_do=0.5, last_ac=False, is_res=False, norm='batch', affine=True, n_group=1, last_bn=False):
        super(PointNet, self).__init__()
        torch.manual_seed(0)
        if nfeat_stn > 0:
            self.stn = STNkD(nfeat_stn, nf_conv_stn, nf_fc_stn, norm=norm, n_group=n_group)
        self.nfeat_stn = nfeat_stn
        modules = []
        for i in range(len(nf_conv)):
            modules.append(nn.Conv1d(nf_conv[i - 1] if i > 0 else nfeat, nf_conv[i], 1))
            if norm == 'batch':
                modules.append(nn.BatchNorm1d(nf_conv[i]))
            elif norm == 'layer':
                modules.append(nn.GroupNorm(1, nf_conv[i]))
            elif norm == 'group':
                modules.append(nn.GroupNorm(n_group, nf_conv[i]))
            modules.append(nn.ReLU(True))
        self.convs = nn.Sequential(*modules)
        modules = []
        for i in range(len(nf_fc)):
            modules.append(nn.Linear(nf_fc[i - 1] if i > 0 else nf_conv[-1] + nfeat_global, nf_fc[i]))
            if i < len(nf_fc) - 1 or last_ac:
                if norm == 'batch':
                    modules.append(nn.BatchNorm1d(nf_fc[i]))
                elif norm == 'layer':
                    modules.append(nn.GroupNorm(1, nf_fc[i]))
                elif norm == 'group':
                    modules.append(nn.GroupNorm(n_group, nf_fc[i]))
                modules.append(nn.ReLU(True))
            if i == len(nf_fc) - 2 and prelast_do > 0:
                modules.append(nn.Dropout(prelast_do))
        if is_res:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.normal_(modules[-1].bias, mean=0, std=0.01)
        self.fcs = nn.Sequential(*modules)

    def forward(self, input, input_global):
        if self.nfeat_stn > 0:
            T = self.stn(input[:, :self.nfeat_stn, :])
            xy_transf = torch.bmm(input[:, :2, :].transpose(1, 2), T).transpose(1, 2)
            input = torch.cat([xy_transf, input[:, 2:, :]], 1)
        input = self.convs(input)
        input = nnf.max_pool1d(input, input.size(2)).squeeze(2)
        if input_global is not None:
            if len(input_global.shape) == 1 or input_global.shape[1] == 1:
                input = torch.cat([input, input_global.view(-1, 1)], 1)
            else:
                input = torch.cat([input, input_global], 1)
        return self.fcs(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ECC_CRFModule,
     lambda: ([], {'propagation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRUCellEx,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (STNkD,
     lambda: ([], {'nfeat': 4, 'nf_conv': [4, 4], 'nf_fc': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
]

class Test_loicland_superpoint_graph(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

