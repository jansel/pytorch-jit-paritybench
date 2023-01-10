import sys
_module = sys.modules[__name__]
del sys
rev = _module
gcn_revop = _module
memgcn = _module
rev_layer = _module
modelnet_cls = _module
architecture = _module
config = _module
data = _module
main = _module
ogbg_mol = _module
args = _module
main = _module
model = _module
test = _module
ogbg_ppa = _module
main = _module
model = _module
test = _module
ogbl_collab = _module
main = _module
model = _module
test = _module
ogbn_arxiv = _module
main = _module
model = _module
test = _module
ogbn_products = _module
main = _module
model = _module
test = _module
ogbn_proteins = _module
dataset = _module
main = _module
model = _module
test = _module
ogbn_arxiv_dgl = _module
loss = _module
main = _module
model_rev = _module
dataset = _module
main = _module
model_rev = _module
test = _module
part_sem_seg = _module
architecture = _module
config = _module
data = _module
eval = _module
main = _module
visualize = _module
architecture = _module
main = _module
opt = _module
sem_seg_dense = _module
architecture = _module
config = _module
test = _module
train = _module
sem_seg_sparse = _module
architecture = _module
config = _module
test = _module
train = _module
gcn_lib = _module
dense = _module
torch_edge = _module
torch_nn = _module
torch_vertex = _module
sparse = _module
torch_edge = _module
torch_message = _module
torch_nn = _module
torch_vertex = _module
utils = _module
ckpt_util = _module
data_util = _module
logger = _module
loss = _module
metrics = _module
optim = _module
pc_viz = _module
pyg_util = _module
tf_logger = _module

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


import numpy as np


import torch


import torch.nn as nn


import copy


import torch.nn.functional as F


from torch.nn import Sequential as Seq


import random


import logging


import time


import uuid


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset


from torch import nn


from torch.utils.data import DataLoader


import sklearn.metrics as metrics


import torch.optim as optim


from functools import partial


from torch.utils.checkpoint import checkpoint


import pandas as pd


from sklearn import preprocessing


import scipy.sparse as sp


import math


from functools import reduce


from matplotlib import pyplot as plt


from matplotlib.ticker import AutoMinorLocator


from matplotlib.ticker import MultipleLocator


import logging.config


from torch.nn import Linear as Lin


from sklearn.metrics import f1_score


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn import DataParallel


from torch.nn import Conv2d


from collections import OrderedDict


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


def get_device_states(*args):
    fwd_gpu_devices = list(set(arg.get_device() for arg in args if isinstance(arg, torch.Tensor) and arg.is_cuda))
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.device(device):
            fwd_gpu_states.append(torch.get_rng_state())
    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states):
    for device, state in zip(devices, states):
        with torch.device(device):
            torch.set_rng_state(state)


class InvertibleCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, preserve_rng_state, num_inputs, *inputs_and_weights):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.preserve_rng_state = preserve_rng_state
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs)
        ctx.input_requires_grad = [element.requires_grad for element in inputs]
        with torch.no_grad():
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                else:
                    x.append(element)
            outputs = ctx.fn(*x)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        detached_outputs = tuple([element.detach_() for element in outputs])
        if not ctx.keep_input:
            if not pytorch_version_one_and_above:
                inputs[0].data.set_()
            else:
                inputs[0].storage().resize_(0)
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes
        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible')
        if len(ctx.outputs) == 0:
            raise RuntimeError('Trying to perform backward on the InvertibleCheckpointFunction for more than {} times! Try raising `num_bwd_passes` by one.'.format(ctx.num_bwd_passes))
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()
        if not ctx.keep_input:
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                with torch.no_grad():
                    inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
                    if not pytorch_version_one_and_above:
                        for element in outputs:
                            element.data.set_()
                    else:
                        for element in outputs:
                            element.storage().resize_(0)
                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = inputs_inverted,
                    if pytorch_version_one_and_above:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.storage().resize_(int(np.prod(element_original.size())))
                            element_original.set_(element_inverted)
                    else:
                        for element_original, element_inverted in zip(inputs, inputs_inverted):
                            element_original.set_(element_inverted)
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for det_input, requires_grad in zip(detached_inputs, ctx.input_requires_grad):
                det_input.requires_grad = requires_grad
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = temp_output,
        filtered_detached_inputs = tuple(filter(lambda x: x.requires_grad, detached_inputs))
        gradients = torch.autograd.grad(outputs=temp_output, inputs=filtered_detached_inputs + ctx.weights, grad_outputs=grad_outputs)
        filtered_inputs = list(filter(lambda x: x.requires_grad, inputs))
        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)
        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]
        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):

    def __init__(self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1, disable=False, preserve_rng_state=False):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`

            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.

            disable : :obj:`bool`, optional
                This will disable using the InvertibleCheckpointFunction altogether.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.

            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is False since most invertible modules should have a valid inverse and hence are
                deterministic.

        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.

        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(self._fn.forward, self._fn.inverse, self.keep_input, self.num_bwd_passes, self.preserve_rng_state, len(xin), *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            y = self._fn(*xin)
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(self._fn.inverse, self._fn.forward, self.keep_input_inverse, self.num_bwd_passes, self.preserve_rng_state, len(yin), *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))
        else:
            x = self._fn.inverse(*yin)
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x


class GroupAdditiveCoupling(torch.nn.Module):

    def __init__(self, Fms, split_dim=-1, group=2):
        super(GroupAdditiveCoupling, self).__init__()
        self.Fms = Fms
        self.split_dim = split_dim
        self.group = group

    def forward(self, x, edge_index, *args):
        xs = torch.chunk(x, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])
        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            y = xs[i] + Fmd
            y_in = y
            ys.append(y)
        out = torch.cat(ys, dim=self.split_dim)
        return out

    def inverse(self, y, edge_index, *args):
        ys = torch.chunk(y, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))
        xs = []
        for i in range(self.group - 1, -1, -1):
            if i != 0:
                y_in = ys[i - 1]
            else:
                y_in = sum(xs)
            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            x = ys[i] - Fmd
            xs.append(x)
        x = torch.cat(xs[::-1], dim=self.split_dim)
        return x


class SharedDropout(nn.Module):

    def __init__(self):
        super(SharedDropout, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.training:
            assert self.mask is not None
            out = x * self.mask
            return out
        else:
            return x


def norm_layer(norm_type, nc):
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class BasicBlock(nn.Module):

    def __init__(self, norm, in_channels):
        super(BasicBlock, self).__init__()
        self.norm = norm_layer(norm, in_channels)
        self.dropout = SharedDropout()

    def forward(self, x, edge_index, dropout_mask=None, edge_emb=None):
        out = self.norm(x)
        out = F.relu(out)
        if isinstance(self.dropout, SharedDropout):
            if dropout_mask is not None:
                self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)
        if edge_emb is not None:
            out = self.gcn(out, edge_index, edge_emb)
        else:
            out = self.gcn(out, edge_index)
        return out


allowable_features = {'possible_atomic_num_list': list(range(1, 119)) + ['misc'], 'possible_chirality_list': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'], 'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'], 'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'], 'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'], 'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'], 'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'], 'possible_is_aromatic_list': [False, True], 'possible_is_in_ring_list': [False, True], 'possible_bond_type_list': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'], 'possible_bond_stereo_list': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'], 'possible_is_conjugated_list': [False, True]}


def get_bond_feature_dims():
    return list(map(len, [allowable_features['possible_bond_type_list'], allowable_features['possible_bond_stereo_list'], allowable_features['possible_is_conjugated_list']]))


class BondEncoder(nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = nn.ModuleList()
        full_bond_feature_dims = get_bond_feature_dims()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return bond_embedding


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class MLP(Seq):

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.0, last_lin=False):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if i == len(channels) - 1 and last_lin:
                pass
            else:
                if norm is not None and norm.lower() != 'none':
                    m.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))
        self.m = m
        super(MLP, self).__init__(*self.m)


class MsgNorm(torch.nn.Module):

    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()
        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg


class GENBlock(BasicBlock):

    def __init__(self, in_channels, out_channels, aggr='max', t=1.0, learn_t=False, p=1.0, learn_p=False, y=0.0, learn_y=False, msg_norm=False, learn_msg_scale=False, encode_edge=False, edge_feat_dim=0, norm='layer', mlp_layers=1):
        super(GENBlock, self).__init__(norm, in_channels)
        self.gcn = GENConv(in_channels, out_channels, aggr=aggr, t=t, learn_t=learn_t, p=p, learn_p=learn_p, y=y, learn_y=learn_y, msg_norm=msg_norm, learn_msg_scale=learn_msg_scale, encode_edge=encode_edge, edge_feat_dim=edge_feat_dim, norm=norm, mlp_layers=mlp_layers)


class GCNBlock(BasicBlock):

    def __init__(self, in_channels, out_channels, norm='layer'):
        super(GCNBlock, self).__init__(norm, in_channels)
        self.gcn = GCNConv(in_channels, out_channels)


class SAGEBlock(BasicBlock):

    def __init__(self, in_channels, out_channels, norm='layer', dropout=0.0):
        super(SAGEBlock, self).__init__(norm, in_channels)
        self.gcn = SAGEConv(in_channels, out_channels)


class GATConv(nn.Module):
    """
    Graph Attention Convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, heads=8):
        super(GATConv, self).__init__()
        self.gconv = tg.nn.GATConv(in_channels, out_channels, heads, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class GATBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, heads=1, norm='layer', att_dropout=0.0, dropout=0.0):
        super(GATBlock, self).__init__(norm, in_channels)
        self.gcn = GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=att_dropout, add_self_loops=False)


def scatter_(name, src, index, dim=0, dim_size=None):
    """Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    assert name in ['add', 'mean', 'min', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out
    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0
    return out


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(MRConv, self).__init__()
        self.nn = MLP([in_channels * 2, out_channels], act, norm, bias)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x_j = scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
        return self.nn(torch.cat([x, x_j], dim=1))


class SemiGCNConv(nn.Module):
    """
    SemiGCN convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(GraphConv, self).__init__()
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels // heads, act, norm, bias, heads)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, False)
        elif conv.lower() == 'rsage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, True)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DenseGraphBlock(nn.Module):
    """
    Dense Static graph convolution block
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(in_channels, out_channels, conv, act, norm, bias, heads)

    def forward(self, x, edge_index):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index


class MultiSeq(Seq):

    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """

    def __init__(self, channels, conv='edge', act='relu', norm=None, bias=True, heads=8, res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv, act, norm, bias, heads)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        return self.body(x, edge_index) + x * self.res_scale, edge_index


class DeepGCN(torch.nn.Module):
    """
    static graph

    """

    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        conv = opt.conv
        heads = opt.n_heads
        c_growth = 0
        self.n_blocks = opt.n_blocks
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias, heads)
        res_scale = 1 if opt.block.lower() == 'res' else 0
        if opt.block.lower() == 'dense':
            c_growth = channels
            self.backbone = MultiSeq(*[DenseGraphBlock(channels + i * c_growth, c_growth, conv, act, norm, bias, heads) for i in range(self.n_blocks - 1)])
        else:
            self.backbone = MultiSeq(*[ResGraphBlock(channels, conv, act, norm, bias, heads, res_scale) for _ in range(self.n_blocks - 1)])
        fusion_dims = int(channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.fusion_block = MLP([fusion_dims, 1024], act, None, bias)
        self.prediction = Seq(*[MLP([1 + fusion_dims, 512], act, norm, bias), torch.nn.Dropout(p=opt.dropout), MLP([512, 256], act, norm, bias), torch.nn.Dropout(p=opt.dropout), MLP([256, opt.n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1], edge_index)[0])
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        out = self.prediction(torch.cat((feats, fusion), 1))
        return out


class DeeperGCN(torch.nn.Module):

    def __init__(self, args):
        super(DeeperGCN, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block
        self.checkpoint_grad = False
        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale
        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path
        self.use_one_hot_encoding = args.use_one_hot_encoding
        if aggr not in ['add', 'max', 'mean'] and self.num_layers > 15:
            self.checkpoint_grad = True
            self.ckp_k = 9
        None
        if self.block == 'res+':
            None
        elif self.block == 'res':
            None
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == 'plain':
            None
        else:
            raise Exception('Unknown block Type')
        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels, aggr=aggr, t=t, learn_t=self.learn_t, p=p, learn_p=self.learn_p, y=y, learn_y=self.learn_y, msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale, encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels, norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.layer_norms.append(norm_layer(norm, hidden_channels))
        self.node_features = torch.load(node_features_file_path)
        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(8, hidden_channels)
        self.edge_encoder = torch.nn.Linear(8, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, node_index, edge_index, edge_attr):
        node_features_1st = self.node_features[node_index]
        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st
        h = self.node_features_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)
        if self.block == 'res+':
            h = self.gcns[0](h, edge_index, edge_emb)
            if self.checkpoint_grad:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index, edge_emb)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index, edge_emb) + h
            else:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index, edge_emb) + h
            h = F.relu(self.layer_norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            return self.node_pred_linear(h)
        elif self.block == 'res':
            h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)
            return self.node_pred_linear(h)
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == 'plain':
            h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return self.node_pred_linear(h)
        else:
            raise Exception('Unknown block Type')

    def print_params(self, epoch=None, final=False):
        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                None
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                None
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                None
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                None
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))


class LinkPredictor(torch.nn.Module):

    def __init__(self, args):
        super(LinkPredictor, self).__init__()
        in_channels = args.hidden_channels
        hidden_channels = args.hidden_channels
        out_channels = args.num_tasks
        num_layers = args.lp_num_layers
        norm = args.lp_norm
        if norm.lower() == 'none':
            self.norms = None
        else:
            self.norms = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.norms is not None:
                self.norms.append(norm_layer(norm, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = args.dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            if self.norms is not None:
                x = self.norms(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class ElementWiseLinear(nn.Module):

    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class RevGATBlock(nn.Module):

    def __init__(self, node_feats, edge_feats, edge_emb, out_feats, n_heads=1, attn_drop=0.0, edge_drop=0.0, negative_slope=0.2, residual=True, activation=None, use_attn_dst=True, allow_zero_in_degree=True, use_symmetric_norm=False):
        super(RevGATBlock, self).__init__()
        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = GATConv(node_feats, out_feats, num_heads=n_heads, attn_drop=attn_drop, edge_drop=edge_drop, negative_slope=negative_slope, residual=residual, activation=activation, use_attn_dst=use_attn_dst, allow_zero_in_degree=allow_zero_in_degree, use_symmetric_norm=use_symmetric_norm)
        self.dropout = SharedDropout()
        if edge_emb > 0:
            self.edge_encoder = nn.Linear(edge_feats, edge_emb)
        else:
            self.edge_encoder = None

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        if isinstance(self.dropout, SharedDropout):
            self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)
        if self.edge_encoder is not None:
            if efeat is None:
                efeat = graph.edata['feat']
            efeat_emb = self.edge_encoder(efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)
        else:
            efeat_emb = None
        out = self.conv(graph, out, perm).flatten(1, -1)
        return out


class RevGAT(nn.Module):

    def __init__(self, in_feats, n_classes, n_hidden, n_layers, n_heads, activation, dropout=0.0, input_drop=0.0, attn_drop=0.0, edge_drop=0.0, use_attn_dst=True, use_symmetric_norm=False, group=2):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.group = group
        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(n_heads * n_hidden)
        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads
            if i == 0 or i == n_layers - 1:
                self.convs.append(GATConv(in_hidden, out_hidden, num_heads=num_heads, attn_drop=attn_drop, edge_drop=edge_drop, use_attn_dst=use_attn_dst, use_symmetric_norm=use_symmetric_norm, residual=True))
            else:
                Fms = nn.ModuleList()
                fm = RevGATBlock(in_hidden // group, 0, 0, out_hidden // group, n_heads=num_heads, attn_drop=attn_drop, edge_drop=edge_drop, use_attn_dst=use_attn_dst, use_symmetric_norm=use_symmetric_norm, residual=True)
                for i in range(self.group):
                    if i == 0:
                        Fms.append(fm)
                    else:
                        Fms.append(copy.deepcopy(fm))
                invertible_module = memgcn.GroupAdditiveCoupling(Fms, group=self.group)
                conv = memgcn.InvertibleModuleWrapper(fn=invertible_module, keep_input=False)
                self.convs.append(conv)
        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)
        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(), device=graph.device)
            self.perms.append(perm)
        h = self.convs[0](graph, h, self.perms[0]).flatten(1, -1)
        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        for i in range(1, self.n_layers - 1):
            graph.requires_grad = False
            perm = torch.stack([self.perms[i]] * self.group, dim=1)
            h = self.convs[i](h, graph, mask, perm)
        h = self.norm(h)
        h = self.activation(h, inplace=True)
        h = self.dp_last(h)
        h = self.convs[-1](graph, h, self.perms[-1])
        h = h.mean(1)
        h = self.bias_last(h)
        return h


class RevGCN(torch.nn.Module):

    def __init__(self, args):
        super(RevGCN, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.group = args.group
        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale
        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path
        self.use_one_hot_encoding = args.use_one_hot_encoding
        self.gcns = torch.nn.ModuleList()
        self.last_norm = norm_layer(norm, hidden_channels)
        for layer in range(self.num_layers):
            Fms = nn.ModuleList()
            fm = GENBlock(hidden_channels // self.group, hidden_channels // self.group, aggr=aggr, t=t, learn_t=self.learn_t, p=p, learn_p=self.learn_p, y=y, learn_y=self.learn_y, msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale, encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels, norm=norm, mlp_layers=mlp_layers)
            for i in range(self.group):
                if i == 0:
                    Fms.append(fm)
                else:
                    Fms.append(copy.deepcopy(fm))
            invertible_module = memgcn.GroupAdditiveCoupling(Fms, group=self.group)
            gcn = memgcn.InvertibleModuleWrapper(fn=invertible_module, keep_input=False)
            self.gcns.append(gcn)
        self.node_features = torch.load(node_features_file_path)
        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(8, hidden_channels)
        self.edge_encoder = torch.nn.Linear(8, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, node_index, edge_index, edge_attr, epoch=-1):
        node_features_1st = self.node_features[node_index]
        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st
        h = self.node_features_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)
        edge_emb = torch.cat([edge_emb] * self.group, dim=-1)
        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)
        h = self.gcns[0](h, edge_index, mask, edge_emb)
        for layer in range(1, self.num_layers):
            h = self.gcns[layer](h, edge_index, mask, edge_emb)
        h = F.relu(self.last_norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.node_pred_linear(h)

    def print_params(self, epoch=None, final=False):
        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                None
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                None
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                None
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                None
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))


class BasicConv(Seq):

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = dense_knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k * self.dilation)
        return self._dilated(edge_index)


class Dilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index, batch=None):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]
                return edge_index.view(2, -1)
            else:
                edge_index = edge_index[:, ::self.dilation]
        else:
            edge_index = edge_index[:, ::self.dilation]
        return edge_index


def knn_matrix(x, k=16, batch=None):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (num_points, num_points)
        k: int
    Returns:
        nearest neighbors: (num_points*k ,1) (num_points, k)
    """
    with torch.no_grad():
        if batch is None:
            batch_size = 1
        else:
            batch_size = batch[-1] + 1
        x = x.view(batch_size, -1, x.shape[-1])
        neg_adj = -pairwise_distance(x.detach())
        _, nn_idx = torch.topk(neg_adj, k=k)
        n_points = x.shape[1]
        start_idx = torch.arange(0, n_points * batch_size, n_points, device=x.device).view(batch_size, 1, 1)
        nn_idx += start_idx
        nn_idx = nn_idx.view(1, -1)
        center_idx = torch.arange(0, n_points * batch_size, device=x.device).expand(k, -1).transpose(1, 0).contiguous().view(1, -1)
    return nn_idx, center_idx


def knn_graph_matrix(x, k=16, batch=None):
    """Construct edge feature for each point
    Args:
        x: (num_points, num_dims)
        batch: (num_points, )
        k: int
    Returns:
        edge_index: (2, num_points*k)
    """
    nn_idx, center_idx = knn_matrix(x, k, batch)
    return torch.cat((nn_idx, center_idx), dim=0)


class DilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(k, dilation, stochastic, epsilon)
        if knn == 'matrix':
            self.knn = knn_graph_matrix
        else:
            self.knn = knn_graph

    def forward(self, x, batch):
        edge_index = self.knn(x, self.k * self.dilation, batch)
        return self._dilated(edge_index, batch)


def batched_index_select(x, idx):
    """fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times N \\times 1}`.
        idx (Tensor): edge_idx
                :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times N \\times k}`.
    """
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)


class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)


class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index) + x * self.res_scale


class DenseDeepGCN(torch.nn.Module):

    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks
        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)
        if opt.block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(channels, k, 1 + i, conv, act, norm, bias, stochastic, epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels + c_growth * i, c_growth, k, 1 + i, conv, act, norm, bias, stochastic, epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int((channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            stochastic = False
            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm, bias, stochastic, epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        self.fusion_block = BasicConv([fusion_dims, 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([fusion_dims + 1024, 512], act, norm, bias), BasicConv([512, 256], act, norm, bias), torch.nn.Dropout(p=opt.dropout), BasicConv([256, opt.n_classes], None, None, bias)])

    def forward(self, inputs):
        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)
        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return self.prediction(torch.cat((fusion, feats), dim=1)).squeeze(-1)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(in_channels, out_channels, kernel_size, dilation, conv, act, norm, bias, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        dense = self.body(x, batch, edge_index)
        return torch.cat((x, dense), 1), batch


class PlainDynBlock(nn.Module):
    """
    Plain Dynamic graph convolution block
    """

    def __init__(self, channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, res_scale=1, **kwargs):
        super(PlainDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv, act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index), batch


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """

    def __init__(self, channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv, act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index) + x * self.res_scale, batch


class SparseDeepGCN(torch.nn.Module):

    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks
        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)
        if opt.block.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1 + i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels + c_growth * i, c_growth, k, 1 + i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int((channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            stochastic = False
            self.backbone = MultiSeq(*[PlainDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon) for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([fusion_dims + 1024, 512], act, norm, bias), MLP([512, 256], act, norm, bias, drop=opt.dropout), MLP([256, opt.n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        corr, color, batch = data.pos, data.x, data.batch
        x = torch.cat((corr, color), dim=1)
        feats = [self.head(x, self.knn(x[:, 0:3], batch))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1], batch)[0])
        feats = torch.cat(feats, dim=1)
        fusion = scatter_('max', self.fusion_block(feats), batch)
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0] // fusion.shape[0], dim=0)
        return self.prediction(torch.cat((fusion, feats), dim=1))


def get_atom_feature_dims():
    return list(map(len, [allowable_features['possible_atomic_num_list'], allowable_features['possible_chirality_list'], allowable_features['possible_degree_list'], allowable_features['possible_formal_charge_list'], allowable_features['possible_numH_list'], allowable_features['possible_number_radical_e_list'], allowable_features['possible_hybridization_list'], allowable_features['possible_is_aromatic_list'], allowable_features['possible_is_in_ring_list']]))


class AtomEncoder(nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class SmoothCrossEntropy(torch.nn.Module):

    def __init__(self, smoothing=True, eps=0.2):
        super(SmoothCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.eps = eps

    def forward(self, pred, gt):
        gt = gt.contiguous().view(-1)
        if self.smoothing:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, reduction='mean')
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseDilated,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dilated,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ElementWiseLinear,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MsgNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiSeq,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (SharedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lightaime_deep_gcns_torch(_paritybench_base):
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

