import sys
_module = sys.modules[__name__]
del sys
cascade_rcnn_r50_fpn_1x = _module
feeder = _module
feeder = _module
feeder_kinetics = _module
tools = _module
main = _module
net = _module
st_gcn = _module
st_gcn_twostream = _module
utils = _module
graph = _module
tgcn = _module
processor = _module
demo_offline = _module
demo_old = _module
demo_realtime = _module
io = _module
processor = _module
recognition = _module
kinetics_gendata = _module
ntu_gendata = _module
ntu_read_skeleton = _module
openpose = _module
video = _module
visualization = _module
setup = _module
torchlight = _module
gpu = _module
io = _module
mmskeleton = _module
apis = _module
estimation = _module
datasets = _module
coco = _module
data_pipeline = _module
skeleton = _module
loader = _module
skeleton_process = _module
coco_transform = _module
video_demo = _module
zipreader = _module
kinetics_feeder = _module
skeleton_feeder = _module
pseudo = _module
models = _module
backbones = _module
hrnet = _module
st_gcn_aaai18 = _module
estimator = _module
base = _module
hrnet_pose = _module
twodim_pose = _module
JointsMSELoss = _module
JointsOHKMMSELoss = _module
loss = _module
skeleton_head = _module
simplehead = _module
ops = _module
nms = _module
setup_linux = _module
gconv = _module
gconv_origin = _module
apis = _module
image2skeleton = _module
pose_demo = _module
recognition = _module
recognition_demo = _module
skeleton_dataset = _module
twodimestimation = _module
infernce_utils = _module
checkpoint = _module
config = _module
importer = _module
third_party = _module
mmskl = _module
publish_model = _module

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


import numpy as np


import random


import torch


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


import time


from torch.autograd import Variable


import warnings


from collections import OrderedDict


import logging


from torch.nn.modules.batchnorm import _BatchNorm


from abc import ABCMeta


from abc import abstractmethod


import math


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    AD = np.dot(A, Dn)
    return AD


class Graph:
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, layout='openpose', strategy='uniform', max_hop=1,
        dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=
            max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 
                1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6,
                5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (
                13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (
                19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for i, j in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6
                ), (8, 7), (9, 2), (10, 9), (11, 10), (12, 11), (13, 1), (
                14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (
                20, 19), (21, 22), (22, 8), (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for i, j in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        else:
            raise ValueError('Do Not Exist This Layout.')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.
                    hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError('Do Not Exist This Strategy')


class Model(nn.Module):
    """Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
        edge_importance_weighting, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False
            )
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = temporal_kernel_size, spatial_kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((st_gcn(in_channels, 64,
            kernel_size, 1, residual=False, **kwargs0), st_gcn(64, 64,
            kernel_size, 1, **kwargs), st_gcn(64, 64, kernel_size, 1, **
            kwargs), st_gcn(64, 64, kernel_size, 1, **kwargs), st_gcn(64, 
            128, kernel_size, 2, **kwargs), st_gcn(128, 128, kernel_size, 1,
            **kwargs), st_gcn(128, 128, kernel_size, 1, **kwargs), st_gcn(
            128, 256, kernel_size, 2, **kwargs), st_gcn(256, 256,
            kernel_size, 1, **kwargs), st_gcn(256, 256, kernel_size, 1, **
            kwargs)))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.
                ones(self.A.size())) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (kernel_size[0] - 1) // 2, 0
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
            kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, (
            kernel_size[0], 1), (stride, 1), padding), nn.BatchNorm2d(
            out_channels), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=(stride, 1)), nn.
                BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x):
        N, C, T, V, M = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(), x[:,
            :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2], torch.cuda.
            FloatTensor(N, C, 1, V, M).zero_()), 2)
        res = self.origin_stream(x) + self.motion_stream(m)
        return res


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, kernel_size,
        t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=
            (t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM
            )
        self.relu = nn.ReLU(inplace=True)
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


class HRModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels,
            num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels,
        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                num_channels[branch_index] * block.expansion, momentum=
                BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels
            [branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, 1, 0, bias
                        =False), nn.BatchNorm2d(num_inchannels[i]), nn.
                        Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), nn.BatchNorm2d(
                                num_outchannels_conv3x3), nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


mmskeleton_model_urls = {'st_gcn/kinetics-skeleton':
    'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.kinetics-6fa43f73.pth'
    , 'st_gcn/ntu-xsub':
    'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xsub-300b57d4.pth'
    , 'st_gcn/ntu-xview':
    'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/st-gcn/st_gcn.ntu-xview-9ba67746.pth'
    , 'mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e':
    'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
    , 'pose_estimation/pose_hrnet_w32_256x192':
    'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmskeleton/models/pose_estimation/pose_hrnet_w32_256x192-76ea353b.pth'
    , 'mmdet/cascade_rcnn_r50_fpn_20e':
    'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth'
    }


def get_mmskeleton_url(filename):
    if filename.startswith('mmskeleton://'):
        model_name = filename[13:]
        model_url = mmskeleton_model_urls[model_name]
        return model_url
    return filename


url_error_message = """

==================================================
MMSkeleton fail to load checkpoint from url: 
    {}
Please check your network connection. Or manually download checkpoints according to the instructor:
    https://github.com/open-mmlab/mmskeleton/blob/master/doc/MODEL_ZOO.md
"""


def load_checkpoint(model, filename, *args, **kwargs):
    try:
        filename = get_mmskeleton_url(filename)
        return mmcv_load_checkpoint(model, filename, *args, **kwargs)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise Exception(url_error_message.format(filename)) from e


class HRNet(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, extra, **kwargs):
        self.inplanes = 64
        self.extra = extra
        super(HRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, num_channels, num_blocks)
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels
            ]
        self.transition1 = self._make_transition_layer([stage1_out_channels
            ], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels)
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels
            ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels)
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels
            ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels)
        self.init_weights()

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, 1, 1, bias=False), nn.BatchNorm2d(
                        num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i
                        ] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels,
                        outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(
                        outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True
        ):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        fuse_method = layer_config['fuse_method']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HRModule(num_branches, block, num_blocks,
                num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)


class ST_GCN_18(nn.Module):
    """Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_cfg,
        edge_importance_weighting=True, data_bn=True, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False
            )
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = temporal_kernel_size, spatial_kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)
            ) if data_bn else lambda x: x
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((st_gcn_block(in_channels, 64,
            kernel_size, 1, residual=False, **kwargs0), st_gcn_block(64, 64,
            kernel_size, 1, **kwargs), st_gcn_block(64, 64, kernel_size, 1,
            **kwargs), st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs), st_gcn_block(
            128, 128, kernel_size, 1, **kwargs), st_gcn_block(128, 128,
            kernel_size, 1, **kwargs), st_gcn_block(128, 256, kernel_size, 
            2, **kwargs), st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs)))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.
                ones(self.A.size())) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


class st_gcn_block(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (kernel_size[0] - 1) // 2, 0
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
            kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(
            inplace=True), nn.Conv2d(out_channels, out_channels, (
            kernel_size[0], 1), (stride, 1), padding), nn.BatchNorm2d(
            out_channels), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=(stride, 1)), nn.
                BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class BaseEstimator(nn.Module):
    """Base class for pose estimation"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseEstimator, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, input, meta, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, input, meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, input, meta, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, input, **kwargs):
        pass

    def forward(self, image, meta=None, targets=None, target_weights=None,
        return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(image, meta, targets, target_weights,
                **kwargs)
        else:
            return self.forward_test(image, **kwargs)


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1
            )
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight
                    [:, (idx)]), heatmap_gt.mul(target_weight[:, (idx)]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):

    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.0
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0,
                sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, outs, targets, target_weights):
        batch_size = outs.size(0)
        num_joints = outs.size(1)
        heatmaps_pred = outs.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_targets = targets.reshape((batch_size, num_joints, -1)).split(
            1, 1)
        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmaps_targets = heatmaps_targets[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(heatmap_pred.mul(
                    target_weights[:, (idx)]), heatmaps_targets.mul(
                    target_weights[:, (idx)])))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred,
                    heatmaps_targets))
        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)
        return self.ohkm(loss)


def import_obj(type):
    if not isinstance(type, str):
        raise ImportError('Object type should be string.')
    mod_str, _sep, class_str = type.rpartition('.')
    try:
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except ModuleNotFoundError:
        if type[0:11] != 'mmskeleton.':
            return import_obj('mmskeleton.' + type)
        raise ModuleNotFoundError('Object {} cannot be found in {}.'.format
            (class_str, mod_str))


def call_obj(type, **kwargs):
    if isinstance(type, str):
        return import_obj(type)(**kwargs)
    elif callable(type):
        return type(**kwargs)
    else:
        raise ValueError('type should be string all callable.')


class SimpleSkeletonHead(nn.Module):

    def __init__(self, num_convs, in_channels, embed_channels=None,
        kernel_size=None, num_joints=None, reg_loss=dict(name=
        'JointsMSELoss', use_target_weight=False)):
        super(SimpleSkeletonHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.kernel_size = kernel_size
        self.num_joints = num_joints
        self.skeleton_reg = self.make_layers()
        self.reg_loss = call_obj(**reg_loss)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def make_layers(self):
        assert isinstance(self.embed_channels, list) or isinstance(self.
            embed_channels, int) or self.embed_channels is None
        assert isinstance(self.kernel_size, list) or isinstance(self.
            kernel_size, int)
        if isinstance(self.embed_channels, list):
            assert len(self.embed_channels) == self.num_convs - 1
        if isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == self.num_convs
        module_list = []
        for i in range(self.num_convs):
            if i == 0:
                in_channels = self.in_channels
            if i < self.num_convs - 1:
                if isinstance(self.embed_channels, list):
                    out_channels = self.embed_channels[i]
                elif isinstance(self.embed_channels, int):
                    out_channels = self.embed_channels
            elif i == self.num_convs - 1 or isinstance(self.embed_channels,
                None):
                out_channels = self.num_joints
            if isinstance(self.kernel_size, list):
                kernel_size = self.kernel_size[i]
            else:
                kernel_size = self.kernel_size
            padding = kernel_size // 2
            module_list.append(nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, padding
                =padding, stride=1))
            in_channels = out_channels
        return nn.Sequential(*module_list)

    def forward(self, x):
        reg_pred = self.skeleton_reg(x[0])
        return reg_pred

    def loss(self, outs, targets, target_weights):
        losses = dict()
        losses['reg_loss'] = self.reg_loss(outs, targets, target_weights)
        return losses


class GraphConvND(nn.Module):

    def __init__(self, N, in_channels, out_channels, kernel_size, stride,
        padding, dilation, groups, bias, padding_mode):
        graph_kernel_size = kernel_size[0]
        graph_stride = stride[0]
        graph_padding = padding[0]
        graph_dilation = dilation[0]
        if graph_stride != 1 or graph_padding != 0 or graph_dilation != 1:
            raise NotImplementedError
        if N == 1:
            conv_type = nn.Conv1d
            self.einsum_func = 'nkcv,kvw->ncw'
        elif N == 2:
            conv_type = nn.Conv2d
            self.einsum_func = 'nkcvx,kvw->ncwx'
        elif N == 3:
            conv_type = nn.Conv3d
            self.einsum_func = 'nkcvxy,kvw->ncwxy'
        self.out_channels = out_channels
        self.graph_kernel_size = graph_kernel_size
        self.conv = conv_type(in_channels, out_channels * graph_kernel_size,
            kernel_size=[1] + kernel_size[1:], stride=[1] + stride[1:],
            padding=[0] + padding[1:], dilation=[1] + dilation[1:], groups=
            groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x, graph):
        if graph.dim() == 2:
            A, out_graph = self.normalize_adjacency_matrix(graph)
        elif graph.dim() == 3:
            A, out_graph = graph, None
        else:
            raise ValueError('input[1].dim() should be 2 or 3.')
        x = self.conv(x)
        x = x.view((x.size(0), self.graph_kernel_size, self.out_channels) +
            x.size()[2:])
        x = torch.einsum(self.einsum_func, (x, A))
        return x.contiguous(), out_graph

    def normalize_adjacency_matrix(self, graph):
        raise NotImplementedError
        return None, graph


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, kernel_size,
        t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=
            (t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class Gconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        if isinstance(kernel_size, int):
            gcn_kernel_size = kernel_size
            feature_dim = 0
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            gcn_kernel_size = kernel_size[0]
            cnn_kernel_size = [1] + kernel_size[1:]
            feature_dim = len(kernel_size) - 1
        else:
            raise ValueError(
                'The type of kernel_size should be int, list or tuple.')
        if feature_dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels *
                gcn_kernel_size, kernel_size=cnn_kernel_size)
        elif feature_dim == 2:
            pass
        elif feature_dim == 3:
            pass
        elif feature_dim == 0:
            pass
        else:
            raise ValueError(
                'The length of kernel_size should be 1, 2, 3, or 4')

    def forward(self, X, A):
        pass


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_open_mmlab_mmskeleton(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(JointsMSELoss(*[], **{'use_target_weight': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

