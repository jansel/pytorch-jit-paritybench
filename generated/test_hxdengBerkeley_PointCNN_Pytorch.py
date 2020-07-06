import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
pointnet_cls = _module
pointnet_cls_basic = _module
pointnet_seg = _module
transform_nets = _module
pointnet_part_seg = _module
test = _module
train = _module
provider = _module
batch_inference = _module
collect_indoor3d_data = _module
eval_iou_accuracy = _module
gen_indoor3d_h5 = _module
indoor3d_util = _module
model = _module
train_pytorch = _module
data_prep_util = _module
data_utils = _module
eulerangles = _module
model = _module
pc_util = _module
plyfile = _module
tf_util = _module
util_funcs = _module
util_layers = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import numpy as np


import random


import time


import torch


from torch import nn


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from torch import FloatTensor


from typing import Tuple


from typing import Callable


from typing import Optional


from torch import cuda


from torch import LongTensor


from sklearn.neighbors import NearestNeighbors


from typing import Union


UFloatTensor = Union[FloatTensor, cuda.FloatTensor]


class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features: int, out_features: int, drop_rate: int=0, with_bn: bool=True, activation: Callable[[UFloatTensor], UFloatTensor]=nn.ReLU()) ->None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x: UFloatTensor) ->UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)
        return x


ULongTensor = Union[LongTensor, cuda.LongTensor]


class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], with_bn: bool=True, activation: Callable[[UFloatTensor], UFloatTensor]=nn.ReLU()) ->None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) ->UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


def EndChannels(f, make_contiguous=False):
    """ Class decorator to apply 2D convolution along end channels. """


    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x
    return WrappedLayer()


class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], depth_multiplier: int=1, with_bn: bool=True, activation: Callable[[UFloatTensor], UFloatTensor]=nn.ReLU()) ->None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups=in_channels), nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias=not with_bn))
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x: UFloatTensor) ->UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, P: int, C_mid: int, depth_multiplier: int) ->None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()
        if __debug__:
            self.C_in = C_in
            self.C_mid = C_mid
            self.dims = dims
            self.K = K
        self.P = P
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)
        self.x_trans = nn.Sequential(EndChannels(Conv(in_channels=dims, out_channels=K * K, kernel_size=(1, K), with_bn=False)), Dense(K * K, K * K, with_bn=False), Dense(K * K, K * K, with_bn=False, activation=None))
        self.end_conv = EndChannels(SepConv(in_channels=C_mid + C_in, out_channels=C_out, kernel_size=(1, K), depth_multiplier=depth_multiplier))

    def forward(self, x: Tuple[UFloatTensor, UFloatTensor, Optional[UFloatTensor]]) ->UFloatTensor:
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x
        if fts is not None:
            assert rep_pt.size()[0] == pts.size()[0] == fts.size()[0]
            assert rep_pt.size()[1] == pts.size()[1] == fts.size()[1]
            assert pts.size()[2] == fts.size()[2] == self.K
            assert fts.size()[3] == self.C_in
        else:
            assert rep_pt.size()[0] == pts.size()[0]
            assert rep_pt.size()[1] == pts.size()[1]
            assert pts.size()[2] == self.K
        assert rep_pt.size()[2] == pts.size()[3] == self.dims
        N = len(pts)
        P = rep_pt.size()[1]
        p_center = torch.unsqueeze(rep_pt, dim=2)
        pts_local = pts - p_center
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted = self.dense2(fts_lifted0)
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)
        X_shape = N, P, self.K, self.K
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)
        return fts_p


class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int, r_indices_func: Callable[[UFloatTensor, UFloatTensor, int, int], ULongTensor]) ->None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()
        C_mid = C_out // 2 if C_in == 0 else C_out // 4
        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)
        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts: UFloatTensor, pts_idx: ULongTensor) ->UFloatTensor:
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([pts[n][(idx), :] for n, idx in enumerate(torch.unbind(pts_idx, dim=0))], dim=0)
        return regions

    def forward(self, x: Tuple[FloatTensor, FloatTensor, FloatTensor]) ->FloatTensor:
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu())
        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))
        return fts_p


class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int, r_indices_func: Callable[[UFloatTensor, UFloatTensor, int, int], ULongTensor]) ->None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x: Tuple[UFloatTensor, UFloatTensor]) ->Tuple[UFloatTensor, UFloatTensor]:
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            idx = np.random.choice(pts.size()[1], self.P, replace=False).tolist()
            rep_pts = pts[:, (idx), :]
        else:
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts


def knn_indices_func_gpu(rep_pts: cuda.FloatTensor, pts: cuda.FloatTensor, k: int, d: int) ->cuda.LongTensor:
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []
    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref) ** 2, 2).squeeze()
        _, inds = torch.topk(dist2, k * d + 1, dim=1, largest=False)
        region_idx.append(inds[:, 1::d])
    region_idx = torch.stack(region_idx, dim=0)
    return region_idx


AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


NUM_CLASS = 40


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(AbbPointCNN(32, 64, 8, 2, -1), AbbPointCNN(64, 96, 8, 4, -1), AbbPointCNN(96, 128, 12, 4, 120), AbbPointCNN(128, 160, 12, 6, 120))
        self.fcn = nn.Sequential(Dense(160, 128), Dense(128, 64, drop_rate=0.5), Dense(64, NUM_CLASS, with_bn=False, activation=None))

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            None
            k = make_dot(x[1])
            None
            k.view()
            None
            assert False
        x = self.pcnn2(x)[1]
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    """

    def __init__(self, N: int, dim: int, *args, **kwargs) ->None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError('Dimensionality %i not supported' % dim)
        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dense,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SepConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_hxdengBerkeley_PointCNN_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

