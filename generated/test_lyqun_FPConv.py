import sys
_module = sys.modules[__name__]
del sys
datasets = _module
s3dis_dataset = _module
s3dis_dataset_test = _module
scannet_dataset_rgb = _module
scannet_dataset_rgb_test = _module
base = _module
fpconv = _module
pointnet2 = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
fpcnn_s3dis = _module
fpcnn_scannet = _module
test_s3dis = _module
test_scannet = _module
train_s3dis = _module
train_scannet = _module
vis_scannet = _module
utils = _module
collect_indoor3d_data = _module
collect_scannet_pickle = _module
indoor3d_util = _module
saver = _module
switchnorm = _module

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


from torch.utils.data import Dataset


import torch.utils.data as torch_data


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from typing import List


from torch.autograd import Variable


from torch.autograd import Function


from typing import Tuple


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.data import DataLoader


import time


import torch.distributed as dist


relu_alpha = 0.2


class PointNet(nn.Module):

    def __init__(self, mlp, pool='max', bn=True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, pcd):
        """
        :param pcd: B, C, npoint, nsample
        :return:
            new_pcd: B, C_new, npoint, 1
        """
        new_pcd = self.mlp(pcd)
        new_pcd = F.max_pool2d(new_pcd, kernel_size=[1, new_pcd.size(3)])
        return new_pcd


class ProjWeightModule(nn.Module):

    def __init__(self, mlp_pn, mlp_wts, map_size, bn=True):
        super().__init__()
        map_len = map_size ** 2
        mlp_pn = [3] + mlp_pn
        mlp_wts = [mlp_pn[-1] + 3] + mlp_wts + [map_len]
        self.pn_layer = PointNet(mlp_pn, bn=bn)
        self.wts_layer = pt_utils.SharedMLP(mlp_wts, bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, xyz):
        """
        :param xyz: B, 3, npoint, nsample <local>
        :return:
            proj_wts: B, map_len, npoint, nsample
        """
        nsample = xyz.size(3)
        dist_feat = self.pn_layer(xyz)
        dist_feat = dist_feat.expand(-1, -1, -1, nsample)
        dist_feat = torch.cat([xyz, dist_feat], dim=1)
        proj_wts = self.wts_layer(dist_feat)
        return proj_wts


class PN_Block(nn.Module):

    def __init__(self, in_channel, out_channel, bn=True, activation=True):
        super().__init__()
        self.conv = pt_utils.Conv2d(in_size=in_channel, out_size=out_channel, kernel_size=(1, 1), bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True) if activation else None)

    def forward(self, pcd):
        """
        :param pcd: B, C_in, npoint
        :return:
            new_pcd: B, C_out, npoint
        """
        pcd = pcd.unsqueeze(-1)
        return self.conv(pcd).squeeze(-1)


class Pooling_Block(nn.Module):

    def __init__(self, radius, nsample, in_channel, out_channel, npoint=None, bn=True, activation=True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.npoint = npoint
        self.conv = PN_Block(in_channel, out_channel, bn=bn, activation=activation)

    def forward(self, xyz, feats, new_xyz=None):
        """
        :param pcd: B, C_in, N
        :return:
            new_pcd: B, C_out, np
        """
        if new_xyz is None:
            assert self.npoint is not None
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx)
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous()
        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
        gped_feats = pointnet2_utils.grouping_operation(feats, idx)
        gped_feats = F.max_pool2d(gped_feats, kernel_size=[1, self.nsample])
        gped_feats = gped_feats.squeeze(-1)
        return self.conv(gped_feats)


class Resnet_BaseBlock(nn.Module):

    def __init__(self, FPCONV, npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):
        """
        pcd => 1x1 conv <relu+bn> => tconv <relu+bn> => 1x1 conv <bn>
        shortcut: pcd => (max_pooling) => 1x1 conv <bn> [apply projection shortcut]
        :param npoint: set to None to ignore 'max_pooling'
        :param nsample, radius: params related to grouper
        """
        super().__init__()
        self.keep_pcd = npoint is None
        self.is_im = in_channel == out_channel
        self.mid_channel = out_channel // 2
        self.conv1 = PN_Block(in_channel=in_channel, out_channel=self.mid_channel, bn=bn)
        self.conv2 = FPCONV(npoint=npoint, nsample=nsample, radius=radius, in_channel=self.mid_channel, out_channel=self.mid_channel, bn=bn, use_xyz=use_xyz)
        self.conv3 = PN_Block(in_channel=self.mid_channel, out_channel=out_channel, bn=bn, activation=False)
        if self.keep_pcd and not self.is_im:
            self.sonv0 = PN_Block(in_channel=in_channel, out_channel=out_channel, bn=bn, activation=False)
        elif not self.keep_pcd:
            self.sonv0 = Pooling_Block(radius=radius, nsample=nsample, in_channel=in_channel, out_channel=out_channel, bn=bn, activation=False)

    def forward(self, xyz, feats, new_xyz=None):
        assert self.keep_pcd and new_xyz is None or not self.keep_pcd, 'invalid new_xyz.'
        new_feats = self.conv1(feats)
        new_xyz, new_feats = self.conv2(xyz, new_feats, new_xyz)
        new_feats = self.conv3(new_feats)
        shc_feats = feats
        if self.keep_pcd and not self.is_im:
            shc_feats = self.sonv0(shc_feats)
        if not self.keep_pcd:
            shc_feats = self.sonv0(xyz, feats, new_xyz)
        new_feats = F.leaky_relu(shc_feats + new_feats, negative_slope=relu_alpha, inplace=True)
        return new_xyz, new_feats


class AssemRes_BaseBlock(nn.Module):

    def __init__(self, CONV_BASE, npoint, nsample, radius, channel_list, nsample_ds=None, radius_ds=None, bn=True, use_xyz=False):
        """
        Apply downsample and conv on input pcd
        :param npoint: the number of points to sample
        :param nsample: the number of neighbors to group when conv
        :param radius: radius of ball query to group neighbors
        :param channel_list: List<a, c, c, ...>, the elements from <1> to the last must be the same
        """
        super().__init__()
        if nsample_ds is None:
            nsample_ds = nsample
        if radius_ds is None:
            radius_ds = radius
        self.conv_blocks = nn.ModuleList()
        for i in range(len(channel_list) - 1):
            in_channel = channel_list[i]
            out_channel = channel_list[i + 1]
            self.conv_blocks.append(Resnet_BaseBlock(FPCONV=CONV_BASE, npoint=npoint if i == 0 else None, nsample=nsample if i == 0 else nsample_ds, radius=radius if i == 0 else radius_ds, in_channel=in_channel, out_channel=out_channel, bn=bn, use_xyz=use_xyz))

    def forward(self, xyz, feats, new_xyz=None):
        for i, block in enumerate(self.conv_blocks):
            xyz, feats = block(xyz, feats, new_xyz)
        return xyz, feats


class FPConv4x4_BaseBlock(nn.Module):

    def __init__(self, npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):
        super().__init__()
        None
        self.npoint = npoint
        self.nsample = nsample
        self.keep_pcd = npoint is None
        self.use_xyz = use_xyz
        self.grouper = pointnet2_utils.QueryAndGroupLocal(radius, nsample)
        self.wts_layer = base.ProjWeightModule(mlp_pn=[8, 16], mlp_wts=[16], map_size=4, bn=bn)
        if use_xyz:
            in_channel += 3
        self.proj_conv = pt_utils.Conv2d(in_size=in_channel, out_size=out_channel, kernel_size=(16, 1), bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True))

    def forward(self, xyz, features, new_xyz=None):
        """
        :param xyz: B,N,3
        :param features: B,C,N
        :returns:
            new_xyz: B,np,3
            new_feats: B,C,np
        """
        if not self.keep_pcd and new_xyz is None:
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx)
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous()
        elif new_xyz is not None:
            self.npoint = new_xyz.size(1)
        else:
            new_xyz = xyz
            self.npoint = new_xyz.size(1)
        grouped_xyz, grouped_feats = self.grouper(xyz, new_xyz, features)
        proj_wts = self.wts_layer(grouped_xyz)
        if self.use_xyz:
            grouped_feats = torch.cat([grouped_xyz, grouped_feats], dim=1)
        proj_wts2_ = proj_wts ** 2
        proj_wts_sum = torch.sum(proj_wts2_, dim=1, keepdim=True)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-08))
        proj_wts_sum = torch.sqrt(proj_wts_sum)
        proj_wts = proj_wts / proj_wts_sum
        proj_wts_sum = torch.sum(proj_wts2_, dim=3, keepdim=True)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-08))
        proj_wts_sum = torch.sqrt(proj_wts_sum)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1.0))
        proj_wts = proj_wts / proj_wts_sum
        proj_wts = proj_wts.transpose(1, 2)
        grouped_feats = grouped_feats.permute(0, 2, 3, 1)
        multi = proj_wts.matmul(grouped_feats)
        proj_feats = F.leaky_relu(proj_wts.matmul(grouped_feats), negative_slope=relu_alpha, inplace=True)
        proj_feats = proj_feats.transpose(1, 3)
        proj_feats = self.proj_conv(proj_feats)
        proj_feats = proj_feats.squeeze(2)
        return new_xyz, proj_feats


class FPConv6x6_BaseBlock(nn.Module):

    def __init__(self, npoint, nsample, radius, in_channel, out_channel, bn=True, use_xyz=False):
        super().__init__()
        None
        self.npoint = npoint
        self.map_size = 6
        self.map_len = self.map_size ** 2
        self.nsample = nsample
        self.keep_pcd = npoint is None
        self.use_xyz = use_xyz
        self.grouper = pointnet2_utils.QueryAndGroupLocal(radius, nsample)
        self.wts_layer = base.ProjWeightModule(mlp_pn=[8, 16, 16], mlp_wts=[16, 32], map_size=6, bn=bn)
        if use_xyz:
            in_channel += 3
        self.bias = Parameter(torch.Tensor(in_channel))
        mid_channel = in_channel
        self.proj_conv = nn.Sequential(pt_utils.Conv3d(in_size=in_channel, out_size=mid_channel, kernel_size=(3, 3, 1), bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)), pt_utils.Conv3d(in_size=in_channel, out_size=mid_channel, kernel_size=(3, 3, 1), bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)), pt_utils.Conv3d(in_size=mid_channel, out_size=out_channel, kernel_size=(2, 2, 1), bn=bn, activation=nn.LeakyReLU(negative_slope=relu_alpha, inplace=True)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, -0.05)

    def forward(self, xyz, features, new_xyz=None):
        """
        :param xyz: B,N,3
        :param features: B,C,N
        :returns:
            new_xyz: B,np,3
            new_feats: B,C,np
        """
        if not self.keep_pcd and new_xyz is None:
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz_flipped = pointnet2_utils.gather_operation(xyz_flipped, idx)
            new_xyz = new_xyz_flipped.transpose(1, 2).contiguous()
        elif new_xyz is not None:
            idx = None
            self.npoint = new_xyz.size(1)
        else:
            idx = None
            new_xyz = xyz
            self.npoint = new_xyz.size(1)
        grouped_xyz, grouped_feats = self.grouper(xyz, new_xyz, features)
        proj_wts = self.wts_layer(grouped_xyz)
        if self.use_xyz:
            grouped_feats = torch.cat([grouped_xyz, grouped_feats], dim=1)
        proj_wts2_ = proj_wts ** 2
        proj_wts_sum = torch.sum(proj_wts2_, dim=1, keepdim=True)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-08))
        proj_wts_sum = torch.sqrt(proj_wts_sum)
        proj_wts = proj_wts / proj_wts_sum
        proj_wts_sum = torch.sum(proj_wts2_, dim=3, keepdim=True)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1e-08))
        proj_wts_sum = torch.sqrt(proj_wts_sum)
        proj_wts_sum = torch.max(proj_wts_sum, torch.tensor(1.0))
        proj_wts = proj_wts / proj_wts_sum
        proj_wts = proj_wts.transpose(1, 2)
        grouped_feats = grouped_feats.permute(0, 2, 3, 1)
        proj_feats = F.leaky_relu(proj_wts.matmul(grouped_feats) + self.bias, negative_slope=relu_alpha, inplace=True)
        bs = proj_feats.size(0)
        proj_feats = proj_feats.transpose(1, 3)
        proj_feats = proj_feats.view(bs, -1, self.map_size, self.map_size, self.npoint).contiguous()
        proj_feats = self.proj_conv(proj_feats)
        proj_feats = proj_feats.squeeze(3).squeeze(2)
        return new_xyz, proj_feats


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, new_xyz=None) ->(torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn, instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pool_method='max_pool', instance_norm=False):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method, instance_norm=instance_norm)


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) ->torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) ->torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.IntTensor(B, npoint, nsample).zero_()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) ->torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.FloatTensor(B, C, nfeatures, nsample)
        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)
        ctx.for_backwards = idx, N
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards
        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):

    def __init__(self, radius: float, nsample: int, use_xyz: bool=True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) ->Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features


class QueryAndGroupLocal(nn.Module):

    def __init__(self, radius: float, nsample: int):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) ->Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            grouped_xyz: B, 3, npoint, nsample <local coordinates>
            new_features: (B, C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        grouped_features = grouping_operation(features, idx)
        return grouped_xyz, grouped_features


class QueryAndGroupXYZ(nn.Module):

    def __init__(self, radius: float, nsample: int):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None) ->Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :return:
            grouped_xyz: B, 3, npoint, nsample <local coordinates>
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        return grouped_xyz


class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name='', instance_norm=False, instance_norm_func=None):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv2d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)


class SharedMLP(nn.Sequential):

    def __init__(self, args: List[int], *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str='', instance_norm: bool=False):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact, instance_norm=instance_norm))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


class Conv3d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int, int]=(1, 1, 1), stride: Tuple[int, int, int]=(1, 1, 1), padding: Tuple[int, int, int]=(0, 0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv3d, batch_norm=BatchNorm3d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm3d)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


CLS_FC = [64]


FP_MLPS = [[64, 64], [128, 64], [256, 128], [512, 256]]


MLPS = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]


NPOINT = 8192


NPOINTS = [NPOINT // 2, NPOINT // 8, NPOINT // 32, NPOINT // 128]


NSAMPLE = [32, 32, 32, 32, 16]


RADIUS = [0.1, 0.2, 0.4, 0.8, 1.6]


class Pointnet2SSG(nn.Module):

    def __init__(self, num_class, input_channels=3, use_xyz=False):
        super().__init__()
        None
        self.SA_modules = nn.ModuleList()
        self.conv0 = AssemRes_BaseBlock(CONV_BASE=FPConv6x6_BaseBlock, npoint=None, radius=RADIUS[0], nsample=NSAMPLE[0], channel_list=[input_channels] + MLPS[0], use_xyz=use_xyz)
        channel_in = MLPS[0][-1]
        skip_channel_list = [channel_in]
        for k in range(NPOINTS.__len__()):
            mlps = [MLPS[k + 1].copy()]
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            None
            if k < 2:
                self.SA_modules.append(AssemRes_BaseBlock(CONV_BASE=FPConv6x6_BaseBlock, npoint=NPOINTS[k], nsample=NSAMPLE[k], radius=RADIUS[k], channel_list=mlps[0], nsample_ds=NSAMPLE[k + 1], radius_ds=RADIUS[k + 1], use_xyz=use_xyz))
            else:
                self.SA_modules.append(AssemRes_BaseBlock(CONV_BASE=FPConv4x4_BaseBlock, npoint=NPOINTS[k], nsample=NSAMPLE[k], radius=RADIUS[k], channel_list=mlps[0], nsample_ds=NSAMPLE[k + 1], radius_ds=RADIUS[k + 1], use_xyz=use_xyz))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            mlp = [pre_channel + skip_channel_list[k]] + FP_MLPS[k]
            None
            self.FP_modules.append(PointnetFPModule(mlp=mlp))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv2d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv2d(pre_channel, num_class, activation=None, bn=False))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        _, features = self.conv0(xyz, features)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        fn_feats = l_features[0].unsqueeze(-1)
        pred_cls = self.cls_layer(fn_feats).squeeze(-1).transpose(1, 2).contiguous()
        return pred_cls


class FPCNN_ScanNet(nn.Module):

    def __init__(self, num_pts, num_class, input_channels, use_xyz=False):
        super().__init__()
        NPOINT = num_pts
        NPOINTS = [NPOINT // 2, NPOINT // 8, NPOINT // 32, NPOINT // 128]
        None
        self.SA_modules = nn.ModuleList()
        self.conv0 = AssemRes_BaseBlock(CONV_BASE=FPConv6x6_BaseBlock, npoint=None, radius=RADIUS[0], nsample=NSAMPLE[0], channel_list=[input_channels] + MLPS[0], use_xyz=use_xyz)
        channel_in = MLPS[0][-1]
        skip_channel_list = [channel_in]
        for k in range(NPOINTS.__len__()):
            mlps = [MLPS[k + 1].copy()]
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            None
            if k < 2:
                self.SA_modules.append(AssemRes_BaseBlock(CONV_BASE=FPConv6x6_BaseBlock, npoint=NPOINTS[k], nsample=NSAMPLE[k], radius=RADIUS[k], channel_list=mlps[0], nsample_ds=NSAMPLE[k + 1], radius_ds=RADIUS[k + 1], use_xyz=use_xyz))
            else:
                self.SA_modules.append(AssemRes_BaseBlock(CONV_BASE=FPConv4x4_BaseBlock, npoint=NPOINTS[k], nsample=NSAMPLE[k], radius=RADIUS[k], channel_list=mlps[0], nsample_ds=NSAMPLE[k + 1], radius_ds=RADIUS[k + 1], use_xyz=use_xyz))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            mlp = [pre_channel + skip_channel_list[k]] + FP_MLPS[k]
            None
            self.FP_modules.append(PointnetFPModule(mlp=mlp))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv2d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv2d(pre_channel, num_class, activation=None, bn=False))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        _, features = self.conv0(xyz, features)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        fn_feats = l_features[0].unsqueeze(-1)
        pred_cls = self.cls_layer(fn_feats).squeeze(-1).transpose(1, 2).contiguous()
        return pred_cls


NUM_CLASSES = 21


class CrossEntropyLossWithWeights(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, predict, target, weights):
        """
        :param predict: (B,N,C)
        :param target: (B,N)
        :param weights: (B,N)
        :return:
        """
        predict = predict.view(-1, NUM_CLASSES).contiguous()
        target = target.view(-1).contiguous().long()
        weights = weights.view(-1).contiguous().float()
        loss = self.cross_entropy_loss(predict, target)
        loss *= weights
        loss = torch.mean(loss)
        return loss


class SwitchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.95, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        x = x.unsqueeze(-1)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        x = x * self.weight + self.bias
        return x.squeeze(-1)


class SwitchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.95, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997, using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm3d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwitchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SwitchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SwitchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_lyqun_FPConv(_paritybench_base):
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

