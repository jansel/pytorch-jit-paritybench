import sys
_module = sys.modules[__name__]
del sys
decoder_2d = _module
bev_decoder = _module
encoder_2d = _module
bev_encoder = _module
backbones_3d = _module
cfe = _module
pillar_dsa = _module
pillar_fsa = _module
point_dsa = _module
point_fsa = _module
voxel_dsa = _module
voxel_fsa = _module
pfe = _module
def_voxel_set_abstraction = _module
sa_voxel_set_abstraction = _module
pointnet2_backbone = _module
sa_block = _module
spconv_backbone = _module
detector3d_template = _module
pointnet2_modules = _module
pointnet2_modules = _module
eval_utils = _module
noise_robustness = _module
test = _module

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


import torch.nn.functional as F


import math


from torch import nn


from functools import partial


from typing import List


import time


from collections import defaultdict


import torch.utils.data as torch_data


import re


class BaseBEVDecoder(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('NUM_FILTERS', None) is not None:
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            num_filters = []
        self.num_levels = len(num_filters)
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(self.num_levels):
            stride = int(spatial_features.shape[2] / x.shape[2])
            x = data_dict['spatial_features_%dx' % stride]
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class ConcatBEVDecoder(BaseBEVDecoder):

    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        if self.model_cfg.get('NUM_FILTERS', None) is not None:
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            num_filters = []
        self.num_levels = len(num_filters)
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(self.model_cfg.IN_DIM + num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(self.model_cfg.IN_DIM + num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(self.num_levels):
            x = data_dict['spatial_features_%dx' % i]
            x = torch.cat([x, data_dict['pillar_context'][i]], dim=1)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class ConcatVoxelDecoder(BaseBEVDecoder):

    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        if self.model_cfg.get('NUM_FILTERS', None) is not None:
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            num_filters = []
        self.num_levels = len(num_filters)
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(self.model_cfg.NUM_FILTERS[idx] + self.model_cfg.ATTN_DIM, num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(self.model_cfg.NUM_FILTERS[idx] + self.model_cfg.ATTN_DIM, num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(self.num_levels):
            x = data_dict['spatial_features_%dx' % i]
            x = torch.cat([x, data_dict['voxel_context'][i]], dim=1)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class BaseBEVEncoder(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [nn.ZeroPad2d(1), nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()]
            for k in range(layer_nums[idx]):
                cur_layers.extend([nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()])
            self.blocks.append(nn.Sequential(*cur_layers))

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict['spatial_features_%dx' % i] = x
            data_dict['spatial_features_stride_%dx' % i] = stride
        return data_dict


class SA_block(nn.Module):
    """Self-Attention block with dot product for point/voxel/pillar context.
    A part of the code is from MLCVNet (CVPR 2020).
    """

    def __init__(self, inplanes, planes, groups=4):
        super().__init__()
        self.groups = groups
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.view(b, c, h).permute(0, 2, 1)
        proj_key = g
        energy = torch.bmm(proj_query, proj_key)
        total_energy = energy
        attention = self.softmax(total_energy)
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h)
        return out

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class PillarContext3D_dsa(nn.Module):

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.3):
        super().__init__()
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        mlps = self.model_cfg.LOCAL_CONTEXT.MLPS
        for k in range(len(mlps)):
            mlps[k] = [self.model_cfg.NUM_BEV_FEATURES] + mlps[k]
        self.adapt_context = pointnet2_stack_modules.StackSAModuleMSGAdapt(radii=self.model_cfg.LOCAL_CONTEXT.POOL_RADIUS, deform_radii=self.model_cfg.LOCAL_CONTEXT.DEFORM_RADIUS, nsamples=self.model_cfg.LOCAL_CONTEXT.NSAMPLE, mlps=mlps, use_xyz=True, pool_method=self.model_cfg.LOCAL_CONTEXT.POOL_METHOD, pc_range=self.point_cloud_range)
        mlps_decode = self.model_cfg.DECODE.MLPS
        for k in range(len(mlps_decode)):
            mlps_decode[k] = [self.model_cfg.IN_DIM] + mlps_decode[k]
        self.decode = pointnet2_stack_modules.StackSAModuleMSGDecode(radii=self.model_cfg.DECODE.POOL_RADIUS, nsamples=self.model_cfg.DECODE.NSAMPLE, mlps=mlps_decode, use_xyz=True, pool_method=self.model_cfg.DECODE.POOL_METHOD)
        self.self_full_fast_attn = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.reduce_dim = nn.Sequential(nn.Conv1d(2 * self.model_cfg.IN_DIM, self.model_cfg.IN_DIM, kernel_size=1), nn.BatchNorm1d(self.model_cfg.IN_DIM), nn.ReLU(inplace=True), nn.Conv1d(self.model_cfg.IN_DIM, self.model_cfg.IN_DIM, kernel_size=1), nn.BatchNorm1d(self.model_cfg.IN_DIM), nn.ReLU(inplace=True))
        self.self_attn_ms1 = SA_block(inplanes=2 * self.model_cfg.IN_DIM, planes=2 * self.model_cfg.IN_DIM)
        self.self_attn_ms2 = SA_block(inplanes=2 * self.model_cfg.IN_DIM, planes=2 * self.model_cfg.IN_DIM)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Select keypoints, i.e. a subset of pillar coords to deform, aggregate local features and then attend to.
        :param batch_size:
        :param coords:
        :param src_points:
        :return: B x num_keypoints x 3
        """
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = coords[:, 0] == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3], self.model_cfg.NUM_KEYPOINTS).long()
            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    def get_local_keypoint_features(self, keypoints, pillar_center, pillar_features, coords):
        """
        Get local features of deformed pillar-subset/keypoints.
        :param keypoints:
        :param pillar_center:
        :param pillar_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        xyz_batch_cnt = torch.zeros([batch_size]).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()
        def_xyz, local_features = self.adapt_context(xyz=pillar_center, xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=pillar_features)
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, local_features):
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            global_feat = self.self_full_fast_attn(local_feat)
            ms_feat1 = torch.cat([local_feat, global_feat], dim=1)
            attn_feat1 = self.self_attn_ms1(ms_feat1)
            attn_feat1 = self.reduce_dim(attn_feat1)
            ms_feat2 = torch.cat([local_feat, attn_feat1], dim=1)
            attn_feat2 = self.self_attn_ms2(ms_feat2)
            attn_feat2 = self.reduce_dim(attn_feat2)
            context_feat = attn_feat2.permute(0, 2, 1).contiguous().squeeze(0)
            batch_global_features.append(context_feat)
        batch_global_features = torch.cat(batch_global_features, 0)
        return batch_global_features

    def get_context_image(self, batch_size, keypoints, pillar_center, global_features, coords):
        new_xyz = pillar_center
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        batch_idx = coords[:, 0]
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (batch_idx == k).sum()
        xyz = keypoints.view(-1, 3)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(keypoints.shape[1])
        pillar_features = self.decode(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt, features=global_features)
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            pillars = pillar_features[batch_mask, :]
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(self.model_cfg.NUM_BEV_FEATURES, self.nz * self.nx * self.ny, dtype=pillars.dtype, device=pillars.device)
            spatial_feature[:, indices] = pillars.t()
            batch_context_features.append(spatial_feature)
        context_pillar_features = torch.cat(batch_context_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * self.nz, self.ny, self.nx)
        return context_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
        Returns:
            context_pillar_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        pillar_center = torch.zeros_like(coords[:, :3])
        pillar_center[:, 0] = coords[:, 3] * self.voxel_x + self.x_offset
        pillar_center[:, 1] = coords[:, 2] * self.voxel_y + self.y_offset
        pillar_center[:, 2] = coords[:, 1] * self.voxel_z + self.z_offset
        keypoints = self.get_keypoints(batch_size, coords, pillar_center)
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, pillar_center, pillars, coords)
        local_keypoint_feats = local_keypoint_feats.view(batch_size * self.model_cfg.NUM_KEYPOINTS, -1).contiguous()
        context_features = self.get_context_features(batch_size, local_keypoint_feats)
        context_pillar_features = self.get_context_image(batch_size, def_xyz, pillar_center, context_features, coords)
        pillar_context = [F.interpolate(context_pillar_features, scale_factor=0.5, mode='bilinear'), F.interpolate(context_pillar_features, scale_factor=0.25, mode='bilinear'), F.interpolate(context_pillar_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict


class SA_block_def(nn.Module):
    """Self-Attention block with dot product for point/voxel/pillar context.
    """

    def __init__(self, inplanes, planes, groups=4):
        super().__init__()
        self.groups = groups
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.permute(0, 2, 1)
        proj_key = g
        energy = torch.bmm(proj_query, proj_key)
        total_energy = energy
        attention = self.softmax(total_energy)
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return out

    def forward(self, x, y):
        residual = x
        t = self.t(y)
        p = self.p(x)
        g = self.g(y)
        b, c, h = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class PillarContext3D_def(nn.Module):
    """Up-sampling method based on Set-transformer (ICML 2019)"""

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.3):
        super().__init__()
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        mlps = self.model_cfg.LOCAL_CONTEXT.MLPS
        for k in range(len(mlps)):
            mlps[k] = [self.model_cfg.NUM_BEV_FEATURES] + mlps[k]
        self.adapt_context = pointnet2_stack_modules.StackSAModuleMSGAdapt(radii=self.model_cfg.LOCAL_CONTEXT.POOL_RADIUS, deform_radii=self.model_cfg.LOCAL_CONTEXT.DEFORM_RADIUS, nsamples=self.model_cfg.LOCAL_CONTEXT.NSAMPLE, mlps=mlps, use_xyz=True, pool_method=self.model_cfg.LOCAL_CONTEXT.POOL_METHOD, pc_range=self.point_cloud_range)
        self.self_full_fast_attn = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn1 = SA_block_def(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block_def(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Select keypoints, i.e. a subset of pillar coords to deform, aggregate local features and then attend to.
        :param batch_size:
        :param coords:
        :param src_points:
        :return: B x num_keypoints x 3
        """
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = coords[:, 0] == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3], self.model_cfg.NUM_KEYPOINTS).long()
            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    def get_local_keypoint_features(self, keypoints, pillar_center, pillar_features, coords):
        """
        Get local features of deformed pillar-subset/keypoints.
        :param keypoints:
        :param pillar_center:
        :param pillar_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        xyz_batch_cnt = torch.zeros([batch_size]).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()
        def_xyz, local_features = self.adapt_context(xyz=pillar_center, xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=pillar_features)
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, pillars, local_features, coords):
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            local_sa_feat = self.self_full_fast_attn(local_feat)
            batch_mask = coords[:, 0] == batch_idx
            pillar_feat = pillars[batch_mask, :].unsqueeze(0).permute(0, 2, 1).contiguous()
            attn_feat1 = self.self_attn1(pillar_feat, local_sa_feat)
            attn_feat2 = self.self_attn2(attn_feat1, local_sa_feat)
            context_pillar = attn_feat2.permute(0, 2, 1).contiguous().squeeze(0)
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(self.model_cfg.NUM_BEV_FEATURES, self.nz * self.nx * self.ny, dtype=context_pillar.dtype, device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_global_features.append(spatial_feature)
        context_pillar_features = torch.cat(batch_global_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * self.nz, self.ny, self.nx)
        return context_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
        Returns:
            context_pillar_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        pillar_center = torch.zeros_like(coords[:, :3])
        pillar_center[:, 0] = coords[:, 3] * self.voxel_x + self.x_offset
        pillar_center[:, 1] = coords[:, 2] * self.voxel_y + self.y_offset
        pillar_center[:, 2] = coords[:, 1] * self.voxel_z + self.z_offset
        keypoints = self.get_keypoints(batch_size, coords, pillar_center)
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, pillar_center, pillars, coords)
        local_keypoint_feats = local_keypoint_feats.view(batch_size * self.model_cfg.NUM_KEYPOINTS, -1).contiguous()
        context_pillar_features = self.get_context_features(batch_size, pillars, local_keypoint_feats, coords)
        pillar_context = [F.interpolate(context_pillar_features, scale_factor=0.5, mode='bilinear'), F.interpolate(context_pillar_features, scale_factor=0.25, mode='bilinear'), F.interpolate(context_pillar_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict


class PositionalEncoding(nn.Module):
    """
    Positional encoding from https://github.com/tatp22/multidim-positional-encoding
    """

    def __init__(self, d_model, height, width, depth=2):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.depth = depth
        self.register_buffer('pos_table', self._positionalencoding3d())

    def _positionalencoding3d(self):
        """
        :return: d_model*height*width position matrix
        """
        if self.d_model % 4 != 0:
            raise ValueError('Cannot use sin/cos positional encoding with odd dimension (got dim={:d})'.format(self.d_model))
        pe = torch.zeros(self.d_model, self.height, self.width, self.depth)
        d_model = int(math.ceil(self.d_model / 3))
        if d_model % 2:
            d_model += 1
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        div_term_depth = torch.exp(torch.arange(0.0, d_model - 2, 2) * -(math.log(10000.0) / d_model - 2))
        pos_w = torch.arange(0.0, self.width).unsqueeze(1)
        pos_h = torch.arange(0.0, self.height).unsqueeze(1)
        pos_d = torch.arange(0.0, self.depth).unsqueeze(1)
        pe[0:d_model:2, :, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, self.height, 1, self.depth)
        pe[1:d_model:2, :, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, self.height, 1, self.depth)
        pe[d_model:2 * d_model:2, :, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.width, self.depth)
        pe[d_model + 1:2 * d_model:2, :, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.width, self.depth)
        pe[2 * d_model::2, :, :, :] = torch.sin(pos_d * div_term_depth).transpose(0, 1).unsqueeze(1).unsqueeze(2).repeat(1, self.height, self.width, 1)
        pe[2 * d_model + 1::2, :, :, :] = torch.cos(pos_d * div_term_depth).transpose(0, 1).unsqueeze(1).unsqueeze(2).repeat(1, self.height, self.width, 1)
        return pe

    def forward(self, x, coords):
        pos_encode = self.pos_table[:, coords[:, 2].type(torch.LongTensor), coords[:, 3].type(torch.LongTensor), coords[:, 1].type(torch.LongTensor)]
        return x + pos_encode.permute(1, 0).contiguous().clone().detach()


class PillarContext3D_fsa(nn.Module):
    """
    Full pair-wise self-attention module for Pillars.
    """

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = grid_size
        self.position_enc = PositionalEncoding(self.model_cfg.IN_DIM, height=grid_size[1], width=grid_size[0])
        self.layer_norm = nn.LayerNorm(self.model_cfg.IN_DIM, eps=1e-06)
        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def add_context_to_pillars(self, pillar_features, coords, nx, ny, nz):
        batch_size = coords[:, 0].max().int().item() + 1
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            pillars = pillar_features[batch_mask, :].unsqueeze(0)
            context_pillar = self.self_attn1(pillars.permute(0, 2, 1).contiguous())
            context_pillar = self.self_attn2(context_pillar)
            context_pillar = context_pillar.permute(0, 2, 1).contiguous().squeeze(0)
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(self.model_cfg.NUM_BEV_FEATURES, nz * nx * ny, dtype=context_pillar.dtype, device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_context_features.append(spatial_feature)
        context_pillar_features = torch.cat(batch_context_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * nz, ny, nx)
        return context_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
s
        Returns:
            context_pillar_features: (N, C)
        """
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        pillar_pos_enc = self.position_enc(pillars, coords)
        pillar_pos_enc = self.layer_norm(pillar_pos_enc)
        context_features = self.add_context_to_pillars(pillar_pos_enc, coords, self.nx, self.ny, self.nz)
        pillar_context = [F.interpolate(context_features, scale_factor=0.5, mode='bilinear'), F.interpolate(context_features, scale_factor=0.25, mode='bilinear'), F.interpolate(context_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict


class PointContext3D(nn.Module):

    def __init__(self, model_cfg, IN_DIM, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.IN_DIM = IN_DIM
        self.self_attn1 = SA_block(inplanes=self.model_cfg.ATTN_DIM, planes=self.model_cfg.ATTN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.ATTN_DIM, planes=self.model_cfg.ATTN_DIM)
        self.reduce_dim = nn.Sequential(nn.Conv1d(IN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1), nn.BatchNorm1d(self.model_cfg.ATTN_DIM), nn.ReLU(inplace=True), nn.Conv1d(self.model_cfg.ATTN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1), nn.BatchNorm1d(self.model_cfg.ATTN_DIM), nn.ReLU(inplace=True))

    def add_context_to_points(self, point_feats):
        """Full pairwise self-attention for all point features"""
        context_points = self.self_attn1(point_feats)
        context_points = self.self_attn2(context_points)
        return context_points

    def forward(self, batch_size, l_features, l_xyz):
        """
        Args:
            :param batch_size:
            :param l_xyz:
            :param l_features:
        """
        l_features_red = self.reduce_dim(l_features)
        point_context_features = self.add_context_to_points(l_features_red)
        return point_context_features


class VoxelContext3D_dsa(nn.Module):

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        mlps = self.model_cfg.LOCAL_CONTEXT.MLPS
        for k in range(len(mlps)):
            mlps[k] = [self.model_cfg.NUM_BEV_FEATURES] + mlps[k]
        self.adapt_context = pointnet2_stack_modules.StackSAModuleMSGAdapt(radii=self.model_cfg.LOCAL_CONTEXT.POOL_RADIUS, deform_radii=self.model_cfg.LOCAL_CONTEXT.DEFORM_RADIUS, nsamples=self.model_cfg.LOCAL_CONTEXT.NSAMPLE, mlps=mlps, use_xyz=True, pool_method=self.model_cfg.LOCAL_CONTEXT.POOL_METHOD, pc_range=self.point_cloud_range)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn3 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        mlps_decode = self.model_cfg.DECODE.MLPS
        for k in range(len(mlps_decode)):
            mlps_decode[k] = [self.model_cfg.IN_DIM] + mlps_decode[k]
        self.decode = pointnet2_stack_modules.StackSAModuleMSGDecode(radii=self.model_cfg.DECODE.POOL_RADIUS, nsamples=self.model_cfg.DECODE.NSAMPLE, mlps=mlps_decode, use_xyz=True, pool_method=self.model_cfg.DECODE.POOL_METHOD)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Get subset of voxels for deformation and context calculation.
        :param batch_size:
        :param coords:
        :param src_points:
        :return: B x num_keypoints x 3
        """
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = coords[:, 0] == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3], self.model_cfg.NUM_KEYPOINTS).long()
            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    def get_local_keypoint_features(self, keypoints, voxel_center, voxel_features, coords):
        """
        :param keypoints:
        :param voxel_center:
        :param voxel_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        xyz_batch_cnt = torch.zeros([batch_size]).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()
        def_xyz, local_features = self.adapt_context(xyz=voxel_center, xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=voxel_features)
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, local_features):
        """
        Self-attention on subset of voxels deformed
        :param batch_size:
        :param local_features:
        :return:
        """
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            global_loc_feat = self.self_attn1(local_feat)
            attn_feat_1 = self.self_attn2(global_loc_feat)
            attn_feat_2 = self.self_attn3(attn_feat_1)
            context_feat = attn_feat_2.permute(0, 2, 1).contiguous().squeeze(0)
            batch_global_features.append(context_feat)
        batch_global_features = torch.cat(batch_global_features, 0)
        return batch_global_features

    def get_context_image(self, batch_size, keypoints, voxel_center, global_features, coords, nx, ny, nz):
        new_xyz = voxel_center
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        batch_idx = coords[:, 0]
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (batch_idx == k).sum()
        xyz = keypoints.view(-1, 3)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(keypoints.shape[1])
        voxel_features = self.decode(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt, features=global_features)
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            voxel_pillars = voxel_features[batch_mask, :]
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(self.model_cfg.NUM_BEV_FEATURES, nz * nx * ny, dtype=voxel_pillars.dtype, device=voxel_pillars.device)
            spatial_feature[:, indices] = voxel_pillars.t()
            batch_context_features.append(spatial_feature)
        voxel_pillar_features = torch.cat(batch_context_features, 0)
        voxel_pillar_features = voxel_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * nz, ny, nx)
        return voxel_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
        Returns:
            context_pillar_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        nz, ny, nx = encoded_spconv_tensor.spatial_shape
        cur_coords = encoded_spconv_tensor.indices
        voxel_feats = encoded_spconv_tensor.features
        xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=8, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
        keypoints = self.get_keypoints(batch_size, cur_coords, xyz)
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, xyz, voxel_feats, cur_coords)
        local_keypoint_feats = local_keypoint_feats.view(batch_size * self.model_cfg.NUM_KEYPOINTS, -1).contiguous()
        context_features = self.get_context_features(batch_size, local_keypoint_feats)
        voxel_context_features = self.get_context_image(batch_size, def_xyz, xyz, context_features, cur_coords, nx, ny, nz)
        voxel_context = [voxel_context_features, F.interpolate(voxel_context_features, scale_factor=0.5, mode='bilinear')]
        batch_dict['voxel_context'] = voxel_context
        return batch_dict


class VoxelContext3D_fsa(nn.Module):

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.position_enc = PositionalEncoding(self.model_cfg.IN_DIM, height=grid_size[1] // self.model_cfg.downsampled, width=grid_size[0] // self.model_cfg.downsampled, depth=2)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.model_cfg.IN_DIM, eps=1e-06)
        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def add_context_to_voxels(self, voxel_features, coords, nx, ny, nz):
        batch_size = coords[:, 0].max().int().item() + 1
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            voxel_pillars = voxel_features[batch_mask, :].unsqueeze(0)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous()
            voxel_pillars = self.self_attn1(voxel_pillars)
            voxel_pillars = self.self_attn2(voxel_pillars)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous().squeeze(0)
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(self.model_cfg.NUM_BEV_FEATURES, nz * nx * ny, dtype=voxel_pillars.dtype, device=voxel_pillars.device)
            spatial_feature[:, indices] = voxel_pillars.t()
            batch_context_features.append(spatial_feature)
        voxel_pillar_features = torch.cat(batch_context_features, 0)
        voxel_pillar_features = voxel_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * nz, ny, nx)
        return voxel_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
        Returns:
            context_pillar_features: (N, C)
        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        nz, ny, nx = encoded_spconv_tensor.spatial_shape
        cur_coords = encoded_spconv_tensor.indices
        voxel_feats = encoded_spconv_tensor.features
        voxel_pos_enc = self.dropout(self.position_enc(voxel_feats, cur_coords))
        voxel_pos_enc = self.layer_norm(voxel_pos_enc)
        voxel_context_features = self.add_context_to_voxels(voxel_pos_enc, cur_coords, nx, ny, nz)
        voxel_context = [voxel_context_features, F.interpolate(voxel_context_features, scale_factor=0.5, mode='bilinear')]
        batch_dict['voxel_context'] = voxel_context
        return batch_dict


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t(torch.t(Ia) * wa) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class DefVoxelSetAbstraction(nn.Module):

    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None, num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            if src_name in self.model_cfg.FEATURES_SOURCE:
                cur_layer = pointnet2_stack_modules.StackSAModuleMSGAdapt(radii=SA_cfg[src_name].POOL_RADIUS, deform_radii=SA_cfg[src_name].POOL_RADIUS, nsamples=SA_cfg[src_name].NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool')
            else:
                cur_layer = pointnet2_stack_modules.StackSAModuleMSG(radii=SA_cfg[src_name].POOL_RADIUS, nsamples=SA_cfg[src_name].NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool')
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += sum([x[-1] for x in mlps])
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]
            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSGGated(radii=SA_cfg['raw_points'].POOL_RADIUS, nsamples=SA_cfg['raw_points'].NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool')
            c_in += sum([x[-1] for x in mlps])
        self.vsa_point_feature_fusion = nn.Sequential(nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False), nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES), nn.ReLU())
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in
        self.pred_bev_offset = nn.Sequential(nn.Conv1d(num_bev_features, 2, kernel_size=1, bias=False), nn.Tanh())
        self.mod_bev_offset = nn.Conv1d(num_bev_features, 1, kernel_size=1, bias=False)
        in_dim = self.model_cfg.NUM_OUTPUT_FEATURES
        self.self_attn1 = SA_block(inplanes=in_dim, planes=in_dim)
        self.self_attn2 = SA_block(inplanes=in_dim, planes=in_dim)
        self.sa_point_feature_fusion = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False), nn.BatchNorm1d(in_dim), nn.ReLU())
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.layer_norm2 = nn.LayerNorm(in_dim)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            offsets = self.pred_bev_offset(point_bev_features.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).contiguous().squeeze(0)
            mod = self.mod_bev_offset(point_bev_features.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).contiguous().squeeze(0)
            offsets = torch.mul(offsets, mod)
            cur_x_idxs = cur_x_idxs + offsets[:, 0]
            cur_y_idxs = cur_y_idxs + offsets[:, 1]
            cur_x_idxs = torch.clamp(cur_x_idxs, 0, bev_features.shape[3])
            cur_y_idxs = torch.clamp(cur_y_idxs, 0, bev_features.shape[2])
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
        point_bev_features = torch.cat(point_bev_features_list, dim=0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(batch_dict['voxel_coords'][:, 1:4], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = batch_indices == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS).long()
                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(keypoints, batch_dict['spatial_features'], batch_dict['batch_size'], bev_stride=batch_dict['spatial_features_stride'])
            point_features_list.append(point_bev_features)
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            pooled_points, pooled_features = self.SA_rawpoints(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=raw_points[:, 1:5])
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name], voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            pooled_points, pooled_features = self.SA_layers[k](xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous())
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        point_features = torch.cat(point_features_list, dim=2)
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)
        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        mid_feat = self.layer_norm1(point_features)
        mid_feat = mid_feat.view(batch_size, -1, point_features.shape[-1]).permute(0, 2, 1).contiguous()
        c_point_features = self.self_attn1(mid_feat)
        c_point_features = self.self_attn2(c_point_features)
        point_features = point_features + self.sa_point_feature_fusion(self.layer_norm2(c_point_features.permute(0, 2, 1).contiguous().view(-1, point_features.shape[-1])))
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords
        return batch_dict


class SAVoxelSetAbstraction(nn.Module):

    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None, num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(radii=SA_cfg[src_name].POOL_RADIUS, nsamples=SA_cfg[src_name].NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool')
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += sum([x[-1] for x in mlps])
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]
            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(radii=SA_cfg['raw_points'].POOL_RADIUS, nsamples=SA_cfg['raw_points'].NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool')
            c_in += sum([x[-1] for x in mlps])
        self.vsa_point_feature_fusion = nn.Sequential(nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False), nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES), nn.ReLU())
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in
        self.self_attn1 = SA_block(inplanes=self.model_cfg.NUM_OUTPUT_FEATURES, planes=self.model_cfg.NUM_OUTPUT_FEATURES // 2)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.NUM_OUTPUT_FEATURES, planes=self.model_cfg.NUM_OUTPUT_FEATURES // 2)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
        point_bev_features = torch.cat(point_bev_features_list, dim=0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(batch_dict['voxel_coords'][:, 1:4], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = batch_indices == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS).long()
                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)
        point_features_list = []
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(keypoints, batch_dict['spatial_features'], batch_dict['batch_size'], bev_stride=batch_dict['spatial_features_stride'])
            point_features_list.append(point_bev_features)
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
            pooled_points, pooled_features = self.SA_rawpoints(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=point_features)
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name], voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            pooled_points, pooled_features = self.SA_layers[k](xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous())
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
        point_features = torch.cat(point_features_list, dim=2)
        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)
        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        c_point_features = self.self_attn1(point_features.view(batch_size, -1, point_features.shape[-1]).permute(0, 2, 1).contiguous())
        c_point_features = self.self_attn2(c_point_features)
        point_features = c_point_features.permute(0, 2, 1).contiguous().view(-1, point_features.shape[-1])
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = point_coords
        return batch_dict


class PointNet2MSG(nn.Module):

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules.PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        point_features = l_features[0].permute(0, 2, 1).contiguous()
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2MSG_fsa(nn.Module):
    """
    Use point_fsa from cfe module here
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            channel_out += self.model_cfg.SA_CONFIG.ATTN[k]
            self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.context_conv3 = point_fsa.PointContext3D(self.model_cfg, IN_DIM=self.model_cfg.SA_CONFIG.MLPS[2][0][-1] + self.model_cfg.SA_CONFIG.MLPS[2][1][-1])
        self.context_conv4 = point_fsa.PointContext3D(self.model_cfg, IN_DIM=self.model_cfg.SA_CONFIG.MLPS[3][0][-1] + self.model_cfg.SA_CONFIG.MLPS[3][1][-1])
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules.PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            if i == 2:
                l_context_3 = self.context_conv3(batch_size, li_features, li_xyz)
                li_features = torch.cat([li_features, l_context_3], dim=1)
            if i == 3:
                l_context_4 = self.context_conv4(batch_size, li_features, li_xyz)
                li_features = torch.cat([li_features, l_context_4], dim=1)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        point_features = l_features[0].permute(0, 2, 1).contiguous()
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2MSG_dsa(nn.Module):
    """
    Use point_dsa from cfe module here
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            channel_out += self.model_cfg.SA_CONFIG.ATTN[k]
            if k > 1:
                self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSGAdapt(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], deform_radii=self.model_cfg.SA_CONFIG.DEFORM_RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            else:
                self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.context_conv3 = point_dsa.PointContext3D(self.model_cfg, IN_DIM=self.model_cfg.SA_CONFIG.MLPS[2][0][-1] + self.model_cfg.SA_CONFIG.MLPS[2][1][-1])
        self.context_conv4 = point_dsa.PointContext3D(self.model_cfg, IN_DIM=self.model_cfg.SA_CONFIG.MLPS[3][0][-1] + self.model_cfg.SA_CONFIG.MLPS[3][1][-1])
        mlps_conv1 = self.model_cfg.MS_CONFIG.MLPS[0].copy()
        for idx in range(mlps_conv1.__len__()):
            mlps_conv1[idx] = [skip_channel_list[1]] + mlps_conv1[idx]
        self.decode_1 = pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.MS_CONFIG.NPOINTS[0], radii=self.model_cfg.MS_CONFIG.RADIUS[0], nsamples=self.model_cfg.MS_CONFIG.NSAMPLE[0], mlps=mlps_conv1, use_xyz=self.model_cfg.MS_CONFIG.get('USE_XYZ', True))
        mlps_conv2 = self.model_cfg.MS_CONFIG.MLPS[1].copy()
        for idx in range(mlps_conv2.__len__()):
            mlps_conv2[idx] = [skip_channel_list[2]] + mlps_conv2[idx]
        self.decode_2 = pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.MS_CONFIG.NPOINTS[1], radii=self.model_cfg.MS_CONFIG.RADIUS[1], nsamples=self.model_cfg.MS_CONFIG.NSAMPLE[1], mlps=mlps_conv2, use_xyz=self.model_cfg.MS_CONFIG.get('USE_XYZ', True))
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules.PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            if i == 2:
                _, l_conv1 = self.decode_1(l_xyz[1], l_features[1], li_xyz)
                l_context_3 = self.context_conv3(batch_size, li_features, li_xyz, l_conv1)
                li_features = torch.cat([li_features, l_context_3], dim=1)
            if i == 3:
                _, l_conv1 = self.decode_1(l_xyz[1], l_features[1], li_xyz)
                _, l_conv2 = self.decode_2(l_xyz[2], l_features[2], li_xyz)
                l_context_4 = self.context_conv4(batch_size, li_features, li_xyz, l_conv1, l_conv2)
                li_features = torch.cat([li_features, l_context_4], dim=1)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        point_features = l_features[0].permute(0, 2, 1).contiguous()
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules_stack.StackSAModuleMSG(radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules_stack.StackPointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points:(k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)
            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)
        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1], known=l_xyz[i], known_batch_cnt=l_batch_cnt[i], unknown_feats=l_features[i - 1], known_feats=l_features[i])
        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    m = spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())
    return m


class VoxelBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        return batch_dict


class VoxelResBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        return batch_dict


class SlimVoxelBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(64), nn.ReLU())
        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        return batch_dict


class Detector3DTemplate(nn.Module):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.module_topology = ['vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'encoder_2d_module', 'cfe', 'decoder_2d_module', 'backbone_2d', 'dense_head', 'point_head', 'roi_head']

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {'module_list': [], 'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, 'num_point_features': self.dataset.point_feature_encoder.num_point_features, 'grid_size': self.dataset.grid_size, 'point_cloud_range': self.dataset.point_cloud_range, 'voxel_size': self.dataset.voxel_size}
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(model_info_dict=model_info_dict)
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](model_cfg=self.model_cfg.VFE, num_point_features=model_info_dict['num_rawpoint_features'], point_cloud_range=model_info_dict['point_cloud_range'], voxel_size=model_info_dict['voxel_size'])
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](model_cfg=self.model_cfg.BACKBONE_3D, input_channels=model_info_dict['num_point_features'], grid_size=model_info_dict['grid_size'], voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'])
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict
        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](model_cfg=self.model_cfg.MAP_TO_BEV, grid_size=model_info_dict['grid_size'])
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_encoder_2d_module(self, model_info_dict):
        if self.model_cfg.get('ENCODER_2D', None) is None:
            return None, model_info_dict
        encoder_2d_module = encoder_2d.__all__[self.model_cfg.ENCODER_2D.NAME](model_cfg=self.model_cfg.ENCODER_2D, input_channels=model_info_dict['num_bev_features'])
        model_info_dict['module_list'].append(encoder_2d_module)
        return encoder_2d_module, model_info_dict

    def build_cfe(self, model_info_dict):
        if self.model_cfg.get('CFE', None) is None:
            return None, model_info_dict
        cfe_module = cfe.__all__[self.model_cfg.CFE.NAME](model_cfg=self.model_cfg.CFE, grid_size=model_info_dict['grid_size'], voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'])
        model_info_dict['module_list'].append(cfe_module)
        return cfe_module, model_info_dict

    def build_decoder_2d_module(self, model_info_dict):
        if self.model_cfg.get('DECODER_2D', None) is None:
            return None, model_info_dict
        decoder_2d_module = decoder_2d.__all__[self.model_cfg.DECODER_2D.NAME](model_cfg=self.model_cfg.DECODER_2D, input_channels=model_info_dict['num_bev_features'])
        model_info_dict['module_list'].append(decoder_2d_module)
        model_info_dict['num_bev_features'] = decoder_2d_module.num_bev_features
        return decoder_2d_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](model_cfg=self.model_cfg.BACKBONE_2D, input_channels=model_info_dict['num_bev_features'])
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict
        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](model_cfg=self.model_cfg.PFE, voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'], num_bev_features=model_info_dict['num_bev_features'], num_rawpoint_features=model_info_dict['num_rawpoint_features'])
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](model_cfg=self.model_cfg.DENSE_HEAD, input_channels=model_info_dict['num_bev_features'], num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, class_names=self.class_names, grid_size=model_info_dict['grid_size'], point_cloud_range=model_info_dict['point_cloud_range'], predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False))
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict
        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']
        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](model_cfg=self.model_cfg.POINT_HEAD, input_channels=num_point_features, num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1, predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False))
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](model_cfg=self.model_cfg.ROI_HEAD, input_channels=model_info_dict['num_point_features'], num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1)
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=cls_preds, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % str(cur_thresh)] = 0
                recall_dict['rcnn_%s' % str(cur_thresh)] = 0
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        if 'version' in checkpoint:
            None
        logger.info('==> Done')
        return it, epoch


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

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
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
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, pool_method=pool_method)


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool=True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

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


class _PointnetAdaptSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.pc_range = []
        self.npoint = None
        self.groupers = None
        self.adapt_groupers = nn.ModuleList()
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
        mod_xyz = new_xyz.clone()
        grouped_features = self.adapt_groupers[0](xyz, new_xyz, features)
        grouped_xyz = grouped_features[:, :3, :, :]
        semantic_trans = self.pred_offset(grouped_features)
        node_offset = (semantic_trans * grouped_xyz).mean(dim=-1)
        mod_xyz += node_offset.squeeze(-1).transpose(1, 2).contiguous()
        mod_xyz_c = mod_xyz.clone()
        mod_xyz_c[:, :, 0] = torch.clamp(mod_xyz[:, :, 0], self.pc_range[0], self.pc_range[3])
        mod_xyz_c[:, :, 1] = torch.clamp(mod_xyz[:, :, 1], self.pc_range[1], self.pc_range[4])
        mod_xyz_c[:, :, 2] = torch.clamp(mod_xyz[:, :, 2], self.pc_range[2], self.pc_range[5])
        for i in range(len(self.groupers)):
            with torch.no_grad():
                new_features = self.groupers[i](xyz, mod_xyz, features)
            new_features = self.mlps[i](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return mod_xyz_c, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSGAdapt(_PointnetAdaptSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], deform_radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, pool_method='max_pool', pc_range=None):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()
        if pc_range is None:
            pc_range = [0, -40, -3, 70.4, 40, 3]
        assert len(radii) == len(nsamples) == len(mlps)
        self.pc_range = pc_range
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.adapt_groupers = nn.ModuleList()
        c_out = mlps[0][0] + 3 if use_xyz else mlps[0][0]
        offset_dim = 3
        self.pred_offset = nn.Sequential(nn.Conv2d(c_out, offset_dim, kernel_size=1, bias=False), nn.Tanh())
        self.adapt_groupers.append(pointnet2_utils.QueryAndGroup(deform_radii[0], nsamples[0], use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):

    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-08)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]
        new_features = self.mlp(new_features)
        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
        return new_features


class StackSAModuleMSGGated(StackSAModuleMSG):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Gating module for processing raw-point cloud features
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__(radii=radii, nsamples=nsamples, mlps=mlps, use_xyz=use_xyz, pool_method=pool_method)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.mlps_gate = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            shared_mlps_gate = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.ReLU()])
                shared_mlps_gate.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False)])
            self.mlps.append(nn.Sequential(*shared_mlps))
            self.mlps_gate.append(nn.Sequential(*shared_mlps_gate))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyzF.sigmoid()
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            features = features.contiguous()
            new_features, ball_idxs = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features_conv = self.mlps[k](new_features)
            gate = self.mlps_gate[k](new_features)
            new_features = new_features_conv * torch.sigmoid(gate)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features


class StackSAModuleMSGAdapt(nn.Module):

    def __init__(self, *, radii: List[float], deform_radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool', pc_range=None):
        """
        Deformation module for adaptively processing multi-scale 3D convolutional features
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.pc_range = [0, -40, -3, 70.4, 40, 3]
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.adapt_groupers = nn.ModuleList()
        c_out = mlps[0][0] + 3 if use_xyz else mlps[0][0]
        offset_dim = 3
        self.pred_offset = nn.Sequential(nn.Conv2d(c_out, offset_dim, kernel_size=1, bias=False), nn.Tanh())
        for i in range(len(radii)):
            radius = radii[i]
            deform_radius = deform_radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            self.adapt_groupers.append(pointnet2_utils.QueryAndGroup(deform_radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        new_xyz_list = []
        for k in range(len(self.groupers)):
            mod_xyz = new_xyz.clone()
            grouped_features, _ = self.adapt_groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            grouped_features = grouped_features.unsqueeze(0).permute(0, 2, 1, 3)
            grouped_xyz = grouped_features[:, :3, :]
            semantic_trans = self.pred_offset(grouped_features)
            node_offset = (semantic_trans * grouped_xyz).mean(dim=-1)
            mod_xyz += node_offset.squeeze(-1).transpose(1, 2).squeeze(0).contiguous()
            mod_xyz_c = mod_xyz.clone()
            mod_xyz_c[:, 0] = torch.clamp(mod_xyz[:, 0], self.pc_range[0], self.pc_range[3])
            mod_xyz_c[:, 1] = torch.clamp(mod_xyz[:, 1], self.pc_range[1], self.pc_range[4])
            mod_xyz_c[:, 2] = torch.clamp(mod_xyz[:, 2], self.pc_range[2], self.pc_range[5])
            new_xyz_list.append(mod_xyz_c)
            with torch.no_grad():
                new_features, _ = self.groupers[k](xyz, xyz_batch_cnt, mod_xyz_c, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz_list[-1], new_features


class StackSAModuleMSGDecode(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Help Unpool step
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            with torch.no_grad():
                new_features, _ = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PointContext3D,
     lambda: ([], {'model_cfg': _mock_config(ATTN_DIM=4), 'IN_DIM': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (SA_block,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SA_block_def,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_AutoVision_cloud_SA_Det3D(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

