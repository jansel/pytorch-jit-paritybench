import sys
_module = sys.modules[__name__]
del sys
ntu60_xsub_hcn = _module
ntu60_xsub_rot = _module
ntu60_xsub_tscnn = _module
ntu60_xview_hcn = _module
ntu60_xview_rot = _module
ntu60_xview_tscnn = _module
default_runtime = _module
dgcn_60 = _module
hcn_60 = _module
hcnb_60 = _module
tacnn_60 = _module
tscnn_60 = _module
adam_500e = _module
adam_800e = _module
sgd_65e = _module
dgcn_65e_ntu60_xsub_joint = _module
dgcn_65e_ntu60_xview_joint = _module
hcn_ntu60_xsub_joint = _module
hcn_ntu60_xview_joint = _module
hcnb_ntu60_xsub_joint = _module
tacnn_ntu60_xsub_joint = _module
tacnn_ntu60_xview_joint = _module
tscnn_ntu60_xsub_joint = _module
tscnn_ntu60_xview_joint = _module
skelact = _module
datasets = _module
transforms = _module
models = _module
backbones = _module
dgcn = _module
hcn = _module
tacnn = _module
tscnn = _module
heads = _module
cnn_head = _module
version = _module
get_flops = _module
gen_ntu_rgbd_raw = _module
test = _module
train = _module

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


import torch.nn.functional as F


import math


import torch.nn as nn


from torchvision.ops import SqueezeExcitation


import warnings


import copy


import time


import torch.distributed as dist


class CeN(nn.Module):

    def __init__(self, in_channels, num_joints=25, clip_len=64):
        super().__init__()
        self.num_joints = num_joints
        self.conv_c = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.conv_t = nn.Conv2d(in_channels=clip_len, out_channels=1, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels=num_joints, out_channels=num_joints * num_joints, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_joints)

    def forward(self, x):
        x = self.conv_c(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.conv_t(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.bn(x)
        x = self.conv_v(x)
        n = x.size(0)
        A = x.view(n, self.num_joints, self.num_joints)
        d = torch.sum(torch.pow(A, 2), dim=1, keepdim=True)
        A = torch.div(A, torch.sqrt(d))
        return A


class STCAttention(nn.Module):

    def __init__(self, out_channels, num_joints):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        y = x
        se = y.mean(-2)
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class JointProject(nn.Module):

    def __init__(self, in_channels, in_joints=25, out_joints=15):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints
        self.proj_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        self.bn = nn.BatchNorm2d(in_channels)
        nn.init.kaiming_normal_(self.proj_mat)
        constant_init(self.bn, 1)

    def forward(self, x):
        n, c, t, v = x.size()
        x = x.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat)
        y = y.view(n, c, t, -1)
        y = self.bn(y)
        return y

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints, self.out_joints)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        lamb (float):  The lambda parameter for fusion of static and dynamic
            branches in Eq. (4)
        A (torch.Tensor | None): The adjacency matrix
        adj_len (int, optional): The length of the adjacency matrix
            Default: 17
        clip_len (int): Input clip length

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, lamb=1.0, A=None, adj_len=25, clip_len=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.lamb = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.lamb, lamb)
        if A.size(1) == adj_len:
            assert A is not None
            self.PA = nn.Parameter(A.clone())
        else:
            self.PA = nn.Parameter(torch.empty(3, adj_len, adj_len))
            nn.init.constant_(self.PA, 1e-06)
        self.cen = CeN(in_channels, num_joints=adj_len, clip_len=clip_len)
        self.conv_cen = nn.Conv2d(in_channels, out_channels, 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        self.num_subset = 3
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            conv = nn.Conv2d(in_channels, out_channels, 1)
            conv_branch_init(conv, self.num_subset)
            self.conv_d.append(conv)
        self.bn = nn.BatchNorm2d(out_channels)
        constant_init(self.bn, 1e-06)
        self.relu = nn.ReLU()
        self.attention = STCAttention(out_channels, adj_len)

    def forward(self, x):
        """Defines the computation performed at every call."""
        n, c, t, v = x.size()
        x1 = x.view(n, c * t, v)
        y = None
        for i in range(self.num_subset):
            A1 = self.PA[i]
            z = self.conv_d[i](torch.matmul(x1, A1).view(n, c, t, v))
            y = z + y if y is not None else z
        A2 = self.cen(x)
        z2 = torch.matmul(x1, A2).view(n, c, t, v)
        z2 = self.conv_cen(z2)
        y += self.lamb * z2
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        y = self.attention(y)
        return y


def identity(x):
    """return input itself."""
    return x


def zero(x):
    """return zero."""
    return 0


class DGCNBlock(nn.Module):
    """Applies spatial graph convolution and  temporal convolution over an
    input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        adj_len (int, optional): The length of the adjacency matrix.
            Default: 17
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lamb=1.0, A=None, adj_len=25, clip_len=64, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (kernel_size[0] - 1) // 2, 0
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1], lamb, A, adj_len, clip_len)
        self.tcn = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding), nn.BatchNorm2d(out_channels))
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        if not residual:
            self.residual = zero
        elif in_channels == out_channels and stride == 1:
            self.residual = identity
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x)


class DGCN(nn.Module):
    """Backbone of Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        alpha (float): The alpha parameter for joint reduction.
        lamb (float):  The lambda parameter for fusion of static and dynamic
            branches in Eq. (4).
        clip_len (int): Input clip length.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, graph_cfg, data_bn=True, alpha=0.6, lamb=1.0, clip_len=64, pretrained=None, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = temporal_kernel_size, spatial_kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else identity
        v1 = A.size(1)
        v2 = int(v1 * alpha)
        v3 = int(v2 * alpha)
        self.agcn_networks = nn.ModuleList((DGCNBlock(in_channels, 64, kernel_size, 1, lamb, A, v1, clip_len, residual=False), DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs), DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs), DGCNBlock(64, 64, kernel_size, 1, lamb, A, v1, clip_len, **kwargs), DGCNBlock(64, 128, kernel_size, 2, lamb, A, v1, clip_len, **kwargs), JointProject(128, v1, v2), DGCNBlock(128, 128, kernel_size, 1, lamb, A, v2, clip_len // 2, **kwargs), DGCNBlock(128, 128, kernel_size, 1, lamb, A, v2, clip_len // 2, **kwargs), DGCNBlock(128, 256, kernel_size, 2, lamb, A, v2, clip_len // 2, **kwargs), JointProject(256, v2, v3), DGCNBlock(256, 256, kernel_size, 1, lamb, A, v3, clip_len // 4, **kwargs), DGCNBlock(256, 256, kernel_size, 1, lamb, A, v3, clip_len // 4, **kwargs)))
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = x.float()
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        for gcn in self.agcn_networks:
            x = gcn(x)
        return x


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input):
        return input.permute(self.dims).contiguous()


class ConvBN(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_channels))


class HCNBlock(nn.Sequential):
    """Extracts hierarchical co-occurrence feature from an input skeleton
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        num_joints (int): Number of joints in each skeleton.
        with_bn (bool): Whether to append a BN layer after conv1.

    Shape:
        - Input: Input skeleton sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Output: Output feature map in :math:`(N, out_channels, T_{out},
            C_{out})` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of joints,
            :math:`C_{out}` is the output size of the coordinate dimension.
    """

    def __init__(self, in_channels=3, out_channels=64, num_joints=25, with_bn=False):
        inter_channels = out_channels // 2
        conv1 = ConvBN if with_bn else nn.Conv2d
        super().__init__(conv1(in_channels, out_channels, 1), nn.ReLU(), nn.Conv2d(out_channels, inter_channels, (3, 1), padding=(1, 0)), Permute((0, 3, 2, 1)), nn.Conv2d(num_joints, inter_channels, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.Conv2d(inter_channels, out_channels, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.Dropout(p=0.5))


class HCN(nn.Module):
    """Backbone of Co-occurrence Feature Learning from Skeleton Data for Action
    Recognition and Detection with Hierarchical Aggregation.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        clip_len (int): Skeleton sequence length.
        with_bn (bool): Whether to append a BN layer after conv1.
        reduce (str): Reduction mode along the temporal dimension,'flatten' or
            'mean'.
        pretrained (str | None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T, V, M)`
        - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
    """

    def __init__(self, in_channels=3, num_joints=25, clip_len=64, with_bn=False, reduce='flatten', pretrained=None):
        super().__init__()
        assert reduce in ('flatten', 'mean')
        self.reduce = reduce
        self.net_l = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.net_m = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.ReLU(), nn.Dropout(p=0.5))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.ReLU())
        self.drop6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Sequential(nn.Linear(256 * 2 * clip_len // 16, 256), nn.ReLU(), nn.Dropout(p=0.5)) if self.reduce == 'flatten' else None
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(-1, c, t, v)
        vel1 = x[:, :, :1] * 0
        vel2 = x[:, :, 1:] - x[:, :, :-1]
        vel = torch.cat((vel1, vel2), dim=2)
        out_l = self.net_l(x)
        out_m = self.net_m(vel)
        out = torch.cat((out_l, out_m), dim=1)
        out = self.conv5(out)
        out = self.conv6(out)
        if self.reduce == 'mean':
            out = out.mean(dim=2)
        out = out.view(n, m, -1)
        out = out.max(dim=1)[0]
        out = self.drop6(out)
        if self.fc7 is not None:
            out = self.fc7(out)
        return out


class DualGroupConv(nn.Module):
    """Dual grouped convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias):
        super().__init__()
        assert out_channels % groups == 0
        assert groups % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, groups=groups // 2, bias=bias)

    def forward(self, input):
        out1 = F.relu(self.conv1(input), inplace=True)
        out2 = F.relu(self.conv2(input), inplace=True)
        out = out1 + out2
        return out


class SELayer(SqueezeExcitation):

    def __init__(self, input_channels, squeeze_factor=1, bias=True):
        squeeze_channels = input_channels // squeeze_factor
        super().__init__(input_channels, squeeze_channels)
        if not bias:
            self.fc1.register_parameter('bias', None)
            self.fc2.register_parameter('bias', None)


class CrossChannelFeatureAugment(nn.Module):
    """Cross-channel feature augmentation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=10, squeeze_factor=1):
        super().__init__()
        inter_channels = out_channels // groups * groups
        self.map1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.attend = SELayer(inter_channels, squeeze_factor, bias=False)
        self.group = DualGroupConv(inter_channels, inter_channels, kernel_size, stride, padding, groups, bias=False)
        self.map2 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = F.relu(self.map1(x), inplace=True)
        out = self.attend(out)
        out = self.group(out)
        out = F.relu(self.map2(out), inplace=True)
        return out


class TaCNNBlock(nn.Sequential):
    """Building block for Ta-CNN.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        num_joints (int): Number of joints in each skeleton.
        groups (tuple): Number of groups for conv2 (CAG) and conv3 (VAG).
        squeeze_factor (int): Squeeze factor in the SE layer.

    Shape:
        - Input: Input skeleton sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Output: Output feature map in :math:`(N, out_channels, T_{out},
            C_{out})` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of joints,
            :math:`C_{out}` is the output size of the coordinate dimension.
    """

    def __init__(self, in_channels=3, out_channels=64, num_joints=25, groups=(10, 6), squeeze_factor=1):
        inter_channels = out_channels // 2
        super().__init__(ConvBN(in_channels, out_channels, 1), nn.ReLU(), CrossChannelFeatureAugment(out_channels, inter_channels, (3, 1), stride=1, padding=(1, 0), groups=groups[0], squeeze_factor=squeeze_factor), Permute((0, 3, 2, 1)), CrossChannelFeatureAugment(num_joints, inter_channels, 3, stride=1, padding=1, groups=groups[1], squeeze_factor=squeeze_factor), nn.MaxPool2d(2, stride=2), nn.Conv2d(inter_channels, out_channels, 3, padding=1), nn.MaxPool2d(2, stride=2), nn.Dropout(p=0.5))


class TaCNN(HCN):
    """Backbone of Topology-aware Convolutional Neural Network for Efficient
    Skeleton-based Action Recognition.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        groups (tuple): Number of groups for conv2 (CAG) and conv3 (VAG).
        squeeze_factor (int): Squeeze factor in the SE layer.
        pretrained (str | None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T, V, M)`
        - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
    """

    def __init__(self, in_channels=3, num_joints=25, groups=(10, 6), squeeze_factor=1, pretrained=None):
        clip_len = 64
        with_bn = True
        reduce = 'mean'
        super().__init__(in_channels, num_joints, clip_len, with_bn, reduce, pretrained)
        self.net_l = TaCNNBlock(in_channels, 64, num_joints, groups, squeeze_factor)
        self.net_m = TaCNNBlock(in_channels, 64, num_joints, groups, squeeze_factor)


class SkeletonTransformer(nn.Module):

    def __init__(self, in_joints=25, out_joints=30):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints
        self.trans_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        nn.init.orthogonal_(self.trans_mat)

    def forward(self, x):
        n, c, t, v = x.size()
        x = x.view(-1, v)
        y = torch.matmul(x, self.trans_mat)
        y = y.view(n, c, t, -1)
        return y

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints, self.out_joints)


class TSBlock(nn.Sequential):
    """Extracts two-stream feature from an input skeleton sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        in_joints (int): Number of input joints.
        out_joints (int): Number of output joints.

    Shape:
        - Input: Input skeleton sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Output: Output feature map in :math:`(N, out_channels, T_{out},
            C_{out})` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of joints,
            :math:`C_{out}` is the output size of the coordinate dimension.
    """

    def __init__(self, in_channels=3, out_channels=64, in_joints=25, out_joints=30):
        super().__init__(SkeletonTransformer(in_joints, out_joints), nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.Dropout(p=0.5))


class TSCNN(nn.Module):
    """Backbone of Skeleton-based Action Recognition with Convolutional Neural
    Networks.

    Args:
        in_channels (int): Number of channels in the input data.
        num_joints (int): Number of joints in each skeleton.
        clip_len (int): Skeleton sequence length.
        pretrained (str | None): Name of pretrained model.

    Shape:
        - Input: :math:`(N, in_channels, T, V, M)`
        - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
    """

    def __init__(self, in_channels=3, num_joints=25, clip_len=32, pretrained=None):
        super().__init__()
        self.net_l = TSBlock(in_channels, 64, num_joints, 30)
        self.net_m = TSBlock(in_channels, 64, num_joints, 30)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.PReLU(init=0.1), nn.Dropout(p=0.5))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=(1, 0)), nn.MaxPool2d(2, stride=2), nn.PReLU(init=0.1))
        self.drop4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Sequential(nn.Linear(256 * 2 * clip_len // 8, 256), nn.PReLU(init=0.1), nn.Dropout(p=0.5))
        self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU(init=0.1))
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(-1, c, t, v)
        vel1 = x[:, :, :1] * 0
        vel2 = x[:, :, 1:] - x[:, :, :-1]
        vel = torch.cat((vel1, vel2), dim=2)
        out_l = self.net_l(x)
        out_m = self.net_m(vel)
        out = torch.cat((out_l, out_m), dim=1)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(n, m, -1)
        out = out.max(dim=1)[0]
        out = self.drop4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HCNBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 25, 25])], {}),
     True),
    (SELayer,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (STCAttention,
     lambda: ([], {'out_channels': 4, 'num_joints': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TaCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 25, 25, 4])], {}),
     True),
    (TaCNNBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 25, 25])], {}),
     True),
]

class Test_hikvision_research_skelact(_paritybench_base):
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

