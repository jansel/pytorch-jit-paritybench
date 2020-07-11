import sys
_module = sys.modules[__name__]
del sys
modelnet_cls = _module
architecture = _module
config = _module
data = _module
main = _module
part_sem_seg = _module
architecture = _module
config = _module
eval = _module
main = _module
visualize = _module
architecture = _module
main = _module
opt = _module
architecture = _module
opt = _module
test = _module
train = _module
architecture = _module
opt = _module
test = _module
train = _module
gcn_lib = _module
dense = _module
torch_edge = _module
torch_nn = _module
torch_vertex = _module
sparse = _module
torch_edge = _module
torch_nn = _module
torch_vertex = _module
utils = _module
ckpt_util = _module
data_util = _module
loss = _module
metrics = _module
optim = _module
pc_viz = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Sequential as Seq


import random


import numpy as np


import logging


import time


import uuid


from torch.utils.data import Dataset


from torch import nn


from torch.utils.data import DataLoader


import sklearn.metrics as metrics


import logging.config


from torch.nn import Linear as Lin


from sklearn.metrics import f1_score


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn import DataParallel


from torch.nn import Conv2d


from collections import OrderedDict


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """
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


def norm_layer(norm_type, nc):
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):

    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act:
                m.append(act_layer(act))
            if norm:
                m.append(norm_layer(norm, channels[-1]))
        self.m = m
        super(MLP, self).__init__(*self.m)


class GATConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
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
        x_j = tg.utils.scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
        return self.nn(torch.cat([x, x_j], dim=1))


class SemiGCNConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
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
    Residual Static graph convolution block
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


class BasicConv(Seq):

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.0):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
            if act:
                m.append(act_layer(act))
            if norm:
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
                edge_index = edge_index[:, :, :, (randnum)]
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
        center_idx = torch.arange(0, n_points).repeat(batch_size, k, 1).transpose(2, 1)
        center_idx = center_idx
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
                edge_index = edge_index[:, :, (randnum)]
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
        del neg_adj
        n_points = x.shape[1]
        start_idx = torch.arange(0, n_points * batch_size, n_points).long().view(batch_size, 1, 1)
        if x.is_cuda:
            start_idx = start_idx
        nn_idx += start_idx
        del start_idx
        if x.is_cuda:
            torch.cuda.empty_cache()
        nn_idx = nn_idx.view(1, -1)
        center_idx = torch.arange(0, n_points * batch_size).repeat(k, 1).transpose(1, 0).contiguous().view(1, -1)
        if x.is_cuda:
            center_idx = center_idx
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


def batched_index_select(inputs, index):
    """

    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """
    batch_size, num_dims, num_vertices, _ = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.view(batch_size, -1)
    inputs = inputs.transpose(2, 1).contiguous().view(-1, num_dims)
    index = index.view(batch_size, -1) + idx.type(index.dtype)
    index = index.view(-1)
    return torch.index_select(inputs, 0, index).view(batch_size, -1, num_dims).transpose(2, 1).view(batch_size, num_dims, -1, k)


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

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)


class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x) + x * self.res_scale


class DenseDeepGCN(torch.nn.Module):

    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
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
        elif opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels + c_growth * i, c_growth, k, 1 + i, conv, act, norm, bias, stochastic, epsilon) for i in range(self.n_blocks - 1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = BasicConv([channels + c_growth * (self.n_blocks - 1), 1024], act, norm, bias)
        self.prediction = Seq(*[BasicConv([channels + c_growth * (self.n_blocks - 1) + 1024, 512], act, norm, bias), BasicConv([512, 256], act, norm, bias), torch.nn.Dropout(p=opt.dropout), BasicConv([256, opt.n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

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

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(channels * 2, channels, kernel_size, dilation, conv, act, norm, bias, **kwargs)

    def forward(self, x, batch=None):
        dense = self.body(x, batch)
        return torch.cat((x, dense), 1), batch


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """

    def __init__(self, channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv, act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        return self.body(x, batch) + x * self.res_scale, batch


class SparseDeepGCN(torch.nn.Module):

    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
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
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels, k, 1 + i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon) for i in range(self.n_blocks - 1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = MLP([channels + c_growth * (self.n_blocks - 1), 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([channels + c_growth * (self.n_blocks - 1) + 1024, 512], act, norm, bias), MLP([512, 256], act, norm, bias), torch.nn.Dropout(p=opt.dropout), MLP([256, opt.n_classes], None, None, bias)])
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
        fusion = tg.utils.scatter_('max', self.fusion_block(feats), batch)
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0] // fusion.shape[0], dim=0)
        return self.prediction(torch.cat((fusion, feats), dim=1))


class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv, act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x):
        return self.body(x)


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
     False),
    (MLP,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiSeq,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
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

