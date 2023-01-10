import sys
_module = sys.modules[__name__]
del sys
annotation = _module
calibration = _module
kitti_dataset = _module
kitti_utils = _module
object3d = _module
roipool3d_utils = _module
generate_box_dataset = _module
config = _module
config_backup = _module
kitti_boxgen_dataset = _module
kitti_boxplace_dataset = _module
kitti_dataset = _module
kitti_rcnn_dataset = _module
point_rcnn = _module
pointnet2_msg = _module
rcnn_net = _module
rpn = _module
train_functions = _module
transformer = _module
center_proposal_layer = _module
proposal_layer = _module
proposal_target_layer = _module
bbox_transform = _module
distance = _module
giou_utils = _module
giou_utilsbackup = _module
gious = _module
greedFurthestPoint = _module
iou3d_utils = _module
setup = _module
kitti_utils = _module
loss_utils = _module
roipool3d_utils = _module
setup = _module
weighted_sample = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
_init_path = _module
dataset = _module
pointnet2_msg = _module
train_and_eval = _module
eval_active = _module
eval_auto = _module
eval = _module
evaluate = _module
kitti_common = _module
rotate_iou = _module
train_cascade1 = _module
train_cascade_later = _module
fastai_optim = _module
learning_schedules_fastai = _module
train_utils = _module

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


import matplotlib.pyplot as plt


import numpy as np


from torch.utils.data import DataLoader


import time


from scipy.stats import multivariate_normal


import matplotlib


import matplotlib.patches as patches


import torch


import torch.utils.data as torch_data


from scipy.spatial import Delaunay


import scipy


import logging


from matplotlib.lines import Line2D


import warnings


from copy import deepcopy


import copy


import random


import math


import torch.nn as nn


import torch.nn.functional as F


import torch.autograd as grad


from collections import namedtuple


from torch.autograd import Function


from scipy.spatial import ConvexHull


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from typing import List


from torch.autograd import Variable


from typing import Tuple


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


from torch.nn.utils import clip_grad_norm_


import re


from sklearn.cluster import MeanShift


from sklearn.cluster import estimate_bandwidth


from scipy.optimize import linear_sum_assignment


from sklearn.covariance import MinCovDet


from functools import partial


from collections import Iterable


from torch import nn


from torch.nn.utils import parameters_to_vector


from torch._utils import _unflatten_dense_tensors


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


class Transformer(nn.Module):

    def __init__(self, num_points=2000, K=3):
        super(Transformer, self).__init__()
        self.K = K
        self.N = num_points
        self.identity = grad.Variable(torch.eye(self.K).float().view(-1))
        self.block1 = nn.Sequential(nn.Conv1d(K, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, K * K))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.max_pool1d(x, self.N).squeeze(2)
        x = self.mlp(x)
        x += self.identity
        x = x.view(-1, self.K, self.K)
        return x


def box2center_box(pred_boxes3d):
    pred_boxes3d_ce = pred_boxes3d.clone()
    pred_boxes3d_ce[:, :, 1] -= pred_boxes3d_ce[:, :, 3] / 2
    pred_boxes3d[:, :, 6] = pred_boxes3d[:, :, 6] % (np.pi * 2)
    return pred_boxes3d_ce


def center_box2box(pred_boxes3d_ce):
    pred_boxes3d = pred_boxes3d_ce.clone()
    pred_boxes3d[:, :, 1] += pred_boxes3d[:, :, 3] / 2
    pred_boxes3d[:, :, 6] = pred_boxes3d[:, :, 6] % (np.pi * 2)
    return pred_boxes3d


def decode_bbox_target_stage_2(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size, get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    anchor_size = anchor_size
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
    z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
    start_offset = z_res_r
    x_bin = torch.argmax(pred_reg[:, x_bin_l:x_bin_r], dim=1)
    z_bin = torch.argmax(pred_reg[:, z_bin_l:z_bin_r], dim=1)
    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        x_res_norm = torch.gather(pred_reg[:, x_res_l:x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l:z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size
        pos_x += x_res
        pos_z += z_res
    else:
        pos_x = pred_reg[:, x_res_l] * loc_scope
        pos_z = pred_reg[:, z_res_l] * loc_scope
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r
        y_bin = torch.argmax(pred_reg[:, y_bin_l:y_bin_r], dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l:y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r
        pos_y = pred_reg[:, y_offset_l]
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    ry_bin = torch.argmax(pred_reg[:, ry_bin_l:ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l:ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_ry_fine:
        angle_per_class = np.pi / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = ry_bin.float() * angle_per_class + angle_per_class / 2 + ry_res
    else:
        angle_per_class = 2 * np.pi / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r - pred_reg.shape[1] < 3
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
    return ret_box3d


def refine_box(pred_boxes3d, rcnn_ref):
    refine_boxes3d = pred_boxes3d.clone()
    refine_boxes3d[:, :3] = pred_boxes3d[:, :3] + pred_boxes3d[:, 3:6] * rcnn_ref[:, :3]
    refine_boxes3d[:, 3:6] = pred_boxes3d[:, 3:6] * (1 + rcnn_ref[:, 3:6])
    refine_boxes3d[:, 6] = pred_boxes3d[:, 6] + rcnn_ref[:, 6]
    return refine_boxes3d


class RCNNNet(nn.Module):

    def __init__(self, num_classes, num_point=512, input_channels=0, use_xyz=True):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.ATT_modules = nn.ModuleList()
        channel_in = input_channels
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0])
        self.rcnn_input_channel = 5
        self.input_tansformer = Transformer(num_point, 3)
        self.xyz_up_layer = pt_utils.SharedMLP([3] + cfg.RCNN.XYZ_UP_LAYER, bn=cfg.RCNN.USE_BN)
        self.feature_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel - 3] + cfg.RCNN.XYZ_UP_LAYER, bn=cfg.RCNN.USE_BN)
        c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
        self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)
        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            if cfg.ATTENTION:
                self.ATT_modules.append(pt_utils.SharedMLP([channel_in], bn=cfg.RCNN.USE_BN, activation=nn.ReLU(inplace=True)))
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]
            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(PointnetSAModule(npoint=npoint, radius=cfg.RCNN.SA_CONFIG.RADIUS[k], nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k], mlp=mlps, use_xyz=use_xyz, bn=cfg.RCNN.USE_BN))
            channel_in = mlps[-1]
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)
        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0], gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)
        if cfg.IOUN.ENABLED:
            self.cascade = cfg.CASCADE
            self.can_xyz_up_layer = nn.ModuleList()
            self.can_feature_up_layer = nn.ModuleList()
            self.can_merge_down_layer = nn.ModuleList()
            self.SA_score_modules = nn.ModuleList()
            self.ATT_score_modules = nn.ModuleList()
            self.IOU_layer = nn.ModuleList()
            self.ICL_layer = nn.ModuleList()
            self.ref_layer = nn.ModuleList()
            for i in range(self.cascade):
                for p in self.parameters():
                    p.requires_grad = False
                self.can_xyz_up_layer.append(pt_utils.SharedMLP([3] + cfg.RCNN.XYZ_UP_LAYER, bn=cfg.RCNN.USE_BN))
                self.can_feature_up_layer.append(pt_utils.SharedMLP([2] + cfg.RCNN.XYZ_UP_LAYER, bn=cfg.RCNN.USE_BN))
                c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
                self.can_merge_down_layer.append(pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN))
                iou_channel_in = input_channels
                for k in range(cfg.IOUN.SA_CONFIG.NPOINTS.__len__()):
                    mlps = [iou_channel_in] + cfg.IOUN.SA_CONFIG.MLPS[k]
                    if cfg.ATTENTION:
                        self.ATT_score_modules.append(pt_utils.SharedMLP([iou_channel_in], bn=cfg.RCNN.USE_BN, activation=nn.ELU(inplace=True)))
                    npoint = cfg.IOUN.SA_CONFIG.NPOINTS[k] if cfg.IOUN.SA_CONFIG.NPOINTS[k] != -1 else None
                    self.SA_score_modules.append(PointnetSAModule(npoint=npoint, radius=cfg.IOUN.SA_CONFIG.RADIUS[k], nsample=cfg.IOUN.SA_CONFIG.NSAMPLE[k], mlp=mlps, use_xyz=use_xyz, bn=cfg.IOUN.USE_BN))
                    iou_channel_in = mlps[-1]
                IOU_channel = 1
                IOU_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.CLS_FC.__len__()):
                    IOU_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.CLS_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.CLS_FC[k]
                IOU_layers.append(pt_utils.Conv1d(pre_channel, IOU_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    IOU_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.IOU_layer.append(nn.Sequential(*IOU_layers))
                ICL_channel = 1
                ICL_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.CLS_FC.__len__()):
                    ICL_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.CLS_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.CLS_FC[k]
                ICL_layers.append(pt_utils.Conv1d(pre_channel, ICL_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    ICL_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.ICL_layer.append(nn.Sequential(*ICL_layers))
                per_loc_bin_num = int(cfg.IOUN.LOC_SCOPE / cfg.IOUN.LOC_BIN_SIZE) * 2
                loc_y_bin_num = int(cfg.IOUN.LOC_Y_SCOPE / cfg.IOUN.LOC_Y_BIN_SIZE) * 2
                ref_channel = 7
                ref_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.REG_FC.__len__()):
                    ref_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.REG_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.REG_FC[k]
                ref_layers.append(pt_utils.Conv1d(pre_channel, ref_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    ref_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.ref_layer.append(nn.Sequential(*ref_layers))
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def get_rotation_matrix(self, batch_size, ry):
        Rot_y = torch.zeros((batch_size, 3, 3))
        Rot_y[:, 0, 0] = torch.cos(ry[:, 0])
        Rot_y[:, 0, 2] = torch.sin(ry[:, 0])
        Rot_y[:, 1, 1] = 1
        Rot_y[:, 2, 0] = -torch.sin(ry[:, 0])
        Rot_y[:, 2, 2] = torch.cos(ry[:, 0])
        return Rot_y

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            xyz, raw_features = input_data['cur_box_point'], torch.cat((input_data['cur_box_reflect'], input_data['train_mask']), dim=-1)
            features = torch.cat((xyz, raw_features), dim=-1)
        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = xyz.transpose(1, 2).unsqueeze(dim=3)
            raw_features_input = raw_features.transpose(1, 2).unsqueeze(dim=3)
            uper_xyz = self.xyz_up_layer(xyz_input)
            uper_feature = self.feature_up_layer(raw_features_input)
            if 'cur_pts_feature' in input_data.keys():
                uper_feature = input_data['cur_pts_feature'].transpose(1, 2).unsqueeze(dim=3)
            merged_feature = torch.cat((uper_xyz, uper_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if cfg.ATTENTION:
                mean_channels = np.sqrt(float(l_features[i].shape[1]))
                context = self.ATT_modules[i](l_features[i].unsqueeze(dim=3)).squeeze(dim=3)
                attention = F.softmax(torch.bmm(l_features[i].transpose(1, 2), l_features[i]) / mean_channels, dim=1)
                l_features[i] = torch.bmm(context, attention) + l_features[i]
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
        roi_boxes3d = torch.zeros((rcnn_reg.shape[0], 3))
        pred_boxes3d_ce = decode_bbox_target_stage_2(roi_boxes3d.view(-1, 3), rcnn_reg.view(-1, rcnn_reg.shape[-1]), anchor_size=self.MEAN_SIZE, loc_scope=cfg.RCNN.LOC_SCOPE, loc_bin_size=cfg.RCNN.LOC_BIN_SIZE, num_head_bin=cfg.RCNN.NUM_HEAD_BIN, get_xz_fine=False, loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE, get_ry_fine=False).view(-1, 1, 7).detach()
        pred_boxes3d = pred_boxes3d_ce.clone()
        pred_boxes3d_ce = box2center_box(pred_boxes3d_ce)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg, 'pred_boxes3d': pred_boxes3d}
        if cfg.IOUN.ENABLED:
            SA_SCORE_layer = 0
            for c in range(self.cascade):
                xyz = input_data['cur_box_point'].clone()
                if c != 0:
                    rcnn_ref = rcnn_ref.view(rcnn_ref.shape[0], 1, rcnn_ref.shape[-1])
                    pred_boxes3d_ce = refine_box(pred_boxes3d_ce.view(-1, 7), rcnn_ref.view(rcnn_ref.shape[0], rcnn_ref.shape[-1]))
                    pred_boxes3d_ce = pred_boxes3d_ce.view(-1, 1, 7)
                if 'iou_trans' in input_data.keys():
                    iou_trans_noise = input_data['iou_trans'][..., c]
                    iou_scale_noise = input_data['iou_scale'][..., c]
                    iou_ry_noise = input_data['iou_ry'][..., c]
                    pred_boxes3d_ce[:, :, 0:3] += iou_trans_noise
                    pred_boxes3d_ce[:, :, 3:6] *= iou_scale_noise
                    pred_boxes3d_ce[:, :, 6] += iou_ry_noise[:, :, 0]
                xyz[:, :, 0] = xyz[:, :, 0] - pred_boxes3d_ce[:, :, 0]
                xyz[:, :, 1] = xyz[:, :, 1] - pred_boxes3d_ce[:, :, 1]
                xyz[:, :, 2] = xyz[:, :, 2] - pred_boxes3d_ce[:, :, 2]
                Rot_y = self.get_rotation_matrix(xyz.shape[0], -pred_boxes3d_ce[:, :, 6])
                canional_xyz = torch.einsum('ijk,ikl->ijl', xyz, Rot_y.permute(0, 2, 1))
                extend_factor = 1.2
                canional_xyz[:, :, 0] = canional_xyz[:, :, 0] / (pred_boxes3d_ce[:, :, 5] / 2)
                canional_xyz[:, :, 1] = canional_xyz[:, :, 1] / (pred_boxes3d_ce[:, :, 3] / 2)
                canional_xyz[:, :, 2] = canional_xyz[:, :, 2] / (pred_boxes3d_ce[:, :, 4] / 2)
                canional_mask = torch.max(torch.abs(canional_xyz), dim=-1)[0] > extend_factor
                canional_xyz[canional_mask] = canional_xyz[canional_mask] = 0.0
                can_xyz_input = canional_xyz.transpose(1, 2).unsqueeze(dim=3)
                uper_can_xyz = self.can_xyz_up_layer[c](can_xyz_input)
                uper_can_feature = self.can_feature_up_layer[c](raw_features_input)
                can_merged_feature = torch.cat((uper_can_xyz, uper_can_feature), dim=1)
                can_merged_feature = self.can_merge_down_layer[c](can_merged_feature)
                l_xyz, l_features = [canional_xyz], [can_merged_feature.squeeze(dim=3)]
                for i in range(int(len(self.SA_score_modules) / cfg.CASCADE)):
                    if cfg.ATTENTION:
                        mean_channels = np.sqrt(float(l_features[i].shape[1]))
                        context = self.ATT_score_modules[SA_SCORE_layer + i](l_features[i].unsqueeze(dim=3)).squeeze(dim=3)
                        attention = F.softmax(torch.bmm(l_features[i].transpose(1, 2), l_features[i]) / mean_channels, dim=1)
                        l_features[i] = torch.bmm(context, attention) + l_features[i]
                    li_xyz, li_features = self.SA_score_modules[SA_SCORE_layer + i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
                SA_SCORE_layer += i + 1
                rcnn_iou = self.IOU_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
                rcnn_ref = self.ref_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
                ioun_cls = self.ICL_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)
                pred_boxes3d = center_box2box(pred_boxes3d_ce).view(-1, 1, 7)
                refined_box = refine_box(pred_boxes3d.view(-1, 7), rcnn_ref.view(rcnn_ref.shape[0], rcnn_ref.shape[-1])).view(-1, 1, 7)
                ret_dict.update({'rcnn_iou': rcnn_iou, 'rcnn_ref': rcnn_ref, 'ioun_cls': ioun_cls, 'pred_boxes3d': pred_boxes3d, 'refined_box': refined_box})
        ret_dict.update(input_data)
        return ret_dict


class RPN(nn.Module):

    def __init__(self, use_xyz=True, mode='TRAIN', old_model=False):
        super().__init__()
        self.training_mode = mode == 'TRAIN'
        MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)
        cls_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_cls_layer = nn.Sequential(*cls_layers)
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4
        if old_model:
            reg_channel = per_loc_bin_num * 4 + 12 * 2 + 3
            reg_channel += 1
        reg_layers = []
        pre_channel = cfg.RPN.FP_MLPS[0][-1]
        for k in range(0, cfg.RPN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
            pre_channel = cfg.RPN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RPN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        self.rpn_reg_layer = nn.Sequential(*reg_layers)
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0], gamma=cfg.RPN.FOCAL_GAMMA)
        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError
        self.init_weights()

    def init_weights(self):
        if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
            pi = 0.01
            nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input']
        backbone_xyz, backbone_features = self.backbone_net(pts_input)
        rpn_cls = self.rpn_cls_layer(backbone_features).transpose(1, 2).contiguous()
        rpn_reg = self.rpn_reg_layer(backbone_features).transpose(1, 2).contiguous()
        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg, 'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}
        return ret_dict


class PointRCNN(nn.Module):

    def __init__(self, num_classes, num_point=512, use_xyz=True, mode='TRAIN', old_model=False):
        super().__init__()
        self.mode = mode
        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED or cfg.IOUN.ENABLED
        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode, old_model=old_model)
        if cfg.RCNN.ENABLED or cfg.IOUN.ENABLED:
            rcnn_input_channels = 128
            self.rcnn_net = RCNNNet(num_point=num_point, num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            with torch.set_grad_enabled(not cfg.RPN.FIXED and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
        elif cfg.RCNN.ENABLED or cfg.IOUN.ENABLED:
            output = {}
            rcnn_output = self.rcnn_net(input_data)
            output.update(rcnn_output)
        else:
            raise NotImplementedError
        return output

    def rpn_forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            with torch.set_grad_enabled(not cfg.RPN.FIXED and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
        return output

    def rcnn_forward(self, rcnn_input_info):
        output = {}
        rcnn_output = self.rcnn_net(rcnn_input_info)
        output.update(rcnn_output)
        return output


CLS_FC = [128]


FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]


MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]


NPOINTS = [4096, 1024, 256, 64]


NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]


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


RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]


class Pointnet2MSG(nn.Module):

    def __init__(self, input_channels=6):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=NSAMPLE[k], mlps=mlps, use_xyz=True, bn=True))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k]))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: torch.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()
        return pred_cls


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 512, 3 + C)
    :param rot_angle: (N)
    :return:
    TODO: merge with rotate_pc_along_y_torch in bbox_transform.py
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)
    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)
    pc_temp = pc[:, :, [0, 2]]
    pc[:, :, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1))
    return pc


def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size, get_xz_fine=False, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):
    """
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    anchor_size = anchor_size
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r
    x_bin = torch.argmax(pred_reg[:, x_bin_l:x_bin_r], dim=1)
    z_bin = torch.argmax(pred_reg[:, z_bin_l:z_bin_r], dim=1)
    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r
        x_res_norm = torch.gather(pred_reg[:, x_res_l:x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
        z_res_norm = torch.gather(pred_reg[:, z_res_l:z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size
        pos_x += x_res
        pos_z += z_res
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r
        y_bin = torch.argmax(pred_reg[:, y_bin_l:y_bin_r], dim=1)
        y_res_norm = torch.gather(pred_reg[:, y_res_l:y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = y_bin.float() * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r
        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
    ry_bin = torch.argmax(pred_reg[:, ry_bin_l:ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l:ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
    if get_ry_fine:
        angle_per_class = np.pi / 2 / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = ry_bin.float() * angle_per_class + angle_per_class / 2 + ry_res - np.pi / 4
    else:
        angle_per_class = 2 * np.pi / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
        ry[ry > np.pi] -= 2 * np.pi
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, -roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]
    return ret_box3d


class CenterProposalLayer(nn.Module):

    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0])

    def forward(self, rpn_reg, xyz, gt_center=None):
        """
        :param rpn_scores: (B, N)
        :param rpn_reg: (B, N, 8)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 7)
        """
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]), anchor_size=self.MEAN_SIZE, loc_scope=cfg.RPN.LOC_SCOPE, loc_bin_size=cfg.RPN.LOC_BIN_SIZE, num_head_bin=cfg.RPN.NUM_HEAD_BIN, get_xz_fine=cfg.RPN.LOC_XZ_FINE, get_y_by_bin=False, get_ry_fine=False)
        proposals = proposals.view(batch_size, -1, 3)
        return proposals[..., :3]

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]
        scores_single_list, proposals_single_list = [], []
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            dist_mask = (dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i])
            if dist_mask.sum() != 0:
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':
                keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError
            keep_idx = keep_idx[:post_top_n_list[i]]
            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])
        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_scores = scores_ordered[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        return cur_scores, cur_proposals
        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        return cur_scores[keep_idx], cur_proposals[keep_idx]


class ProposalLayer(nn.Module):

    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0])

    def forward(self, rpn_scores, rpn_reg, xyz):
        """
        :param rpn_scores: (B, N)
        :param rpn_reg: (B, N, 8)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 7)
        """
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]), anchor_size=self.MEAN_SIZE, loc_scope=cfg.RPN.LOC_SCOPE, loc_bin_size=cfg.RPN.LOC_BIN_SIZE, num_head_bin=cfg.RPN.NUM_HEAD_BIN, get_xz_fine=cfg.RPN.LOC_XZ_FINE, get_y_by_bin=False, get_ry_fine=False)
        proposals = proposals.view(batch_size, -1, 3)
        return proposals[..., :3], rpn_scores
        scores = rpn_scores
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True)
        batch_size = scores.size(0)
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 3).zero_()
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        for k in range(batch_size):
            scores_single = scores[k]
            proposals_single = proposals[k]
            order_single = sorted_idxs[k]
            if cfg.TEST.RPN_DISTANCE_BASED_PROPOSE:
                scores_single, proposals_single = self.distance_based_proposal(scores_single, proposals_single, order_single)
            else:
                scores_single, proposals_single = self.score_based_proposal(scores_single, proposals_single, order_single)
            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single
        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]
        scores_single_list, proposals_single_list = [], []
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            dist_mask = (dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i])
            if dist_mask.sum() != 0:
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':
                keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError
            keep_idx = keep_idx[:post_top_n_list[i]]
            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])
        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_scores = scores_ordered[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        return cur_scores, cur_proposals
        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]
        return cur_scores[keep_idx], cur_proposals[keep_idx]


class ProposalTargetLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_dict):
        roi_boxes3d, gt_boxes3d = input_dict['roi_boxes3d'], input_dict['gt_boxes3d']
        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)
        rpn_xyz, rpn_features = input_dict['rpn_xyz'], input_dict['rpn_features']
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [input_dict['rpn_intensity'].unsqueeze(dim=2), input_dict['seg_mask'].unsqueeze(dim=2)]
        else:
            pts_extra_input_list = [input_dict['seg_mask'].unsqueeze(dim=2)]
        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_dict['pts_depth'] / 70.0 - 0.5
            pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=2)
        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
        pooled_features, pooled_empty_flag = roipool3d_utils.roipool3dball_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH, sampled_pt_num=cfg.RCNN.NUM_POINTS)
        sampled_pts, sampled_features = pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:]
        if cfg.AUG_DATA:
            sampled_pts, batch_rois, batch_gt_of_rois = self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - roi_center.unsqueeze(dim=2)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry
        for k in range(batch_size):
            sampled_pts[k] = kitti_utils.rotate_pc_along_y_torch(sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1), roi_ry[k]).squeeze(dim=1)
        valid_mask = pooled_empty_flag == 0
        reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long()
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1
        output_dict = {'sampled_pts': sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3), 'pts_feature': sampled_features.view(-1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]), 'cls_label': batch_cls_label.view(-1), 'reg_valid_mask': reg_valid_mask.view(-1), 'gt_of_rois': batch_gt_of_rois.view(-1, 7), 'gt_iou': batch_roi_iou.view(-1), 'roi_boxes3d': batch_rois.view(-1, 7)}
        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        roi_boxes3d = gt_boxes3d + 0.001
        batch_size = roi_boxes3d.size(0)
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))
        batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()
        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
            k = cur_gt.__len__() - 1
            while cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            fg_inds = torch.nonzero(max_overlaps >= fg_thresh).view(-1)
            easy_bg_inds = torch.nonzero(max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO).view(-1)
            hard_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) & (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)
            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
            if fg_num_rois > 0 and bg_num_rois > 0:
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)
                fg_rois_per_this_image = 0
            else:
                pdb.set_trace()
                raise NotImplementedError
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois, iou3d_src, aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)
            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois, iou3d_src, aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)
            rois = torch.cat(roi_list, dim=0)
            iou_of_rois = torch.cat(roi_iou_list, dim=0)
            gt_of_rois = torch.cat(roi_gt_list, dim=0)
            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois
        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]
            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError
        return bg_inds

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        iou_of_rois = torch.zeros(roi_boxes3d.shape[0]).type_as(gt_boxes3d)
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k].view(1, 7)
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = aug_box3d.view((1, 7))
                iou3d = iou3d_utils.boxes_iou3d_gpu(aug_box3d, gt_box3d)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = torch.rand(3, device=box3d.device) - 0.5
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.15) + 1.0
            angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / (0.5 / (np.pi / 12))
            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            range_config = [[0.2, 0.1, np.pi / 12, 0.7], [0.3, 0.15, np.pi / 12, 0.6], [0.5, 0.15, np.pi / 9, 0.5], [0.8, 0.15, np.pi / 6, 0.3], [1.0, 0.15, np.pi / 3, 0.2]]
            idx = torch.randint(low=0, high=len(range_config), size=(1,))[0].long()
            pos_shift = (torch.rand(3, device=box3d.device) - 0.5) / 0.5 * range_config[idx][0]
            hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / 0.5 * range_config[idx][1] + 1.0
            angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / 0.5 * range_config[idx][2]
            aug_box3d = torch.cat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], dim=0)
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = (torch.rand() - 0.5) / 0.5 * np.pi / 12
            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift, box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift], dtype=np.float32)
            aug_box3d = torch.from_numpy(aug_box3d).type_as(box3d)
            return aug_box3d
        else:
            raise NotImplementedError

    def data_augmentation(self, pts, rois, gt_of_rois):
        """
        :param pts: (B, M, 512, 3)
        :param rois: (B, M. 7)
        :param gt_of_rois: (B, M, 7)
        :return:
        """
        batch_size, boxes_num = pts.shape[0], pts.shape[1]
        angles = (torch.rand((batch_size, boxes_num), device=pts.device) - 0.5 / 0.5) * (np.pi / cfg.AUG_ROT_RANGE)
        temp_x, temp_z, temp_ry = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2], gt_of_rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        gt_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry
        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = torch.atan2(temp_z, temp_x)
        roi_alpha = -torch.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry
        for k in range(batch_size):
            pts[k] = kitti_utils.rotate_pc_along_y_torch(pts[k], angles[k])
            gt_of_rois[k] = kitti_utils.rotate_pc_along_y_torch(gt_of_rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)
            rois[k] = kitti_utils.rotate_pc_along_y_torch(rois[k].unsqueeze(dim=1), angles[k]).squeeze(dim=1)
            temp_x, temp_z = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2]
            temp_beta = torch.atan2(temp_z, temp_x)
            gt_of_rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + gt_alpha - temp_beta
            temp_x, temp_z = rois[:, :, 0], rois[:, :, 2]
            temp_beta = torch.atan2(temp_z, temp_x)
            rois[:, :, 6] = torch.sign(temp_beta) * np.pi / 2 + roi_alpha - temp_beta
        scales = 1 + (torch.rand((batch_size, boxes_num), device=pts.device) - 0.5) / 0.5 * 0.05
        pts = pts * scales.unsqueeze(dim=2).unsqueeze(dim=3)
        gt_of_rois[:, :, 0:6] = gt_of_rois[:, :, 0:6] * scales.unsqueeze(dim=2)
        rois[:, :, 0:6] = rois[:, :, 0:6] * scales.unsqueeze(dim=2)
        flip_flag = torch.sign(torch.rand((batch_size, boxes_num), device=pts.device) - 0.5)
        pts[:, :, :, 0] = pts[:, :, :, 0] * flip_flag.unsqueeze(dim=2)
        gt_of_rois[:, :, 0] = gt_of_rois[:, :, 0] * flip_flag
        src_ry = gt_of_rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        gt_of_rois[:, :, 6] = ry
        rois[:, :, 0] = rois[:, :, 0] * flip_flag
        src_ry = rois[:, :, 6]
        ry = (flip_flag == 1).float() * src_ry + (flip_flag == -1).float() * (torch.sign(src_ry) * np.pi - src_ry)
        rois[:, :, 6] = ry
        return pts, rois, gt_of_rois


class rbbox_corners_aligned(nn.Module):

    def _init_(self, gboxes):
        super(rbbox_corners_aligned, self)._init_()
        self.corners_gboxes = gboxes
        return

    def forward(ctx, gboxes):
        eps = 0.0
        N = gboxes.shape[0]
        center_x = gboxes[:, 0]
        center_y = gboxes[:, 1]
        x_d = gboxes[:, 2]
        y_d = gboxes[:, 3]
        corners = torch.zeros([N, 2, 4], device=gboxes.device, dtype=torch.float32)
        corners[:, 0, 0] = x_d.mul(-0.5)
        corners[:, 1, 0] = y_d.mul(-0.5)
        corners[:, 0, 1] = x_d.mul(-0.5)
        corners[:, 1, 1] = y_d.mul(0.5)
        corners[:, 0, 2] = x_d.mul(0.5)
        corners[:, 1, 2] = y_d.mul(0.5)
        corners[:, 0, 3] = x_d.mul(0.5)
        corners[:, 1, 3] = y_d.mul(-0.5)
        b = center_x.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        c = center_y.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        return corners + torch.cat((b, c), 1)


class rbbox_to_corners(nn.Module):

    def _init_(self, rbbox):
        super(rbbox_to_corners, self)._init_()
        self.rbbox = rbbox
        return

    def forward(ctx, rbbox):
        assert rbbox.shape[1] == 5
        device = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device=device)
        dxcos = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        dxsin = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        dycos = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        dysin = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        corners[:, 0] = -dxcos - dysin + rbbox[:, 0]
        corners[:, 1] = dxsin - dycos + rbbox[:, 1]
        corners[:, 2] = -dxcos + dysin + rbbox[:, 0]
        corners[:, 3] = dxsin + dycos + rbbox[:, 1]
        corners[:, 4] = dxcos + dysin + rbbox[:, 0]
        corners[:, 5] = -dxsin + dycos + rbbox[:, 1]
        corners[:, 6] = dxcos - dysin + rbbox[:, 0]
        corners[:, 7] = -dxsin - dycos + rbbox[:, 1]
        return corners


class align_inter_aligned(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(align_inter_aligned, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        N = gboxes.shape[0]
        M = qboxes.shape[0]
        eps = 1e-16
        assert N == M
        inter_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rbbox_corners_aligned_object = rbbox_corners_aligned()
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 2, 3, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 2, 3, 5, 6]])
        for i in range(N):
            iw = min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) - max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps
            if iw > 0:
                ih = min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
                if ih > 0:
                    inter_area_xoz[i] = iw * ih
            iwmbr = max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) - min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps
            ihmbr = max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
            mbr_area_xoz[i] = iwmbr * ihmbr
        inter_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 1, 3, 4, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 1, 3, 4, 6]])
        for i in range(N):
            iw = min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) - max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps
            if iw > 0:
                ih = min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
                if ih > 0:
                    inter_area_xoy[i] = iw * ih
            iwmbr = max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) - min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps
            ihmbr = max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
            mbr_area_xoy[i] = iwmbr * ihmbr
        inter_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [1, 2, 4, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [1, 2, 4, 5, 6]])
        for i in range(N):
            iw = min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) - max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps
            if iw > 0:
                ih = min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
                if ih > 0:
                    inter_area_yoz[i] = iw * ih
            iwmbr = max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) - min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps
            ihmbr = max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) - min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps
            mbr_area_yoz[i] = iwmbr * ihmbr
        return inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz


class mbr_convex_hull(nn.Module):

    def _init_(self, hull_points_2d):
        super(mbr_convex_hull, self)._init_()
        self.hull_points_2d = hull_points_2d
        return

    def forward(ctx, hull_points_2d):
        device = hull_points_2d.device
        N = hull_points_2d.shape[0]
        edges = hull_points_2d[1:N, :].add(-hull_points_2d[0:N - 1, :])
        edge_angles = torch.atan2(edges[:, 1], edges[:, 0])
        edge_angles = torch.fmod(edge_angles, 3.1415926 / 2.0)
        edge_angles = torch.abs(edge_angles)
        a = torch.stack((torch.cos(edge_angles), torch.cos(edge_angles - 3.1415926 / 2.0)), 1)
        a = torch.unsqueeze(a, 1)
        b = torch.stack((torch.cos(edge_angles + 3.1415926 / 2.0), torch.cos(edge_angles)), 1)
        b = torch.unsqueeze(b, 1)
        R_tensor = torch.cat((a, b), 1)
        hull_points_2d_ = torch.unsqueeze(torch.transpose(hull_points_2d, 0, 1), 0)
        rot_points = R_tensor.matmul(hull_points_2d_)
        min_x = torch.min(rot_points, 2)[0]
        max_x = torch.max(rot_points, 2)[0]
        areas = (max_x[:, 0] - min_x[:, 0]).mul(max_x[:, 1] - min_x[:, 1])
        return torch.min(areas)


class mbr_area_compute(nn.Module):

    def _init_(self, corners):
        super(mbr_area_compute, self)._init_()
        self.corners = corners
        return

    def forward(ctx, corners):
        N = corners.shape[0]
        mbr_rect_area = []
        for i in range(N):
            mbr_rect_area.append(torch.zeros((1,), dtype=torch.float32, device=corners.device))
        for iter in range(N):
            convex_hull_pts = find_convex_hull(corners[iter, :, :].squeeze())
            mbr_convex_hull_object = mbr_convex_hull()
            mbr_rect_area[iter] = mbr_convex_hull_object(convex_hull_pts)
        mbr_rect_areas = torch.stack(mbr_rect_area)
        return mbr_rect_areas


class rinter_area_compute(nn.Module):

    def _init_(self, corners_gboxes, corners_qboxes):
        super(rinter_area_compute, self)._init_()
        self.corners_gboxes = corners_gboxes
        self.corners_qboxes = corners_qboxes
        return

    def forward(ctx, corners_gboxes, corners_qboxes):
        intersections, num_of_intersections = compute_vertex(corners_gboxes, corners_qboxes)
        sorted_int_pts = sort_vertex(intersections, num_of_intersections)
        inter_area = area_polygon(sorted_int_pts, num_of_intersections)
        return inter_area


class gious_3D(nn.Module):

    def _init_(self, gboxes, qboxes, aligned=False):
        super(gious_3D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, aligned):
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        gious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
        if aligned:
            align_inter_aligned_object = align_inter_aligned()
            inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(gboxes, qboxes)
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
            iou_xoz = torch.div(inter_area_xoz, union_xoz)
            iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz, mbr_area_xoz)
            gious_xoz = iou_xoz - iou_bis_xoz
            union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
            iou_xoy = torch.div(inter_area_xoy, union_xoy)
            iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
            gious_xoy = iou_xoy - iou_bis_xoy
            union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
            iou_yoz = torch.div(inter_area_yoz, union_yoz)
            iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
            gious_xoy = iou_yoz - iou_bis_yoz
            gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]]) / 3.0
        else:
            rbbox_to_corners_object = rbbox_to_corners()
            corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 2, 3, 5, 6]])
            corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 2, 3, 5, 6]])
            rinter_area_compute_object = rinter_area_compute()
            inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)
            corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
            corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
            corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)
            mbr_area_compute_object = mbr_area_compute()
            mbr_area = mbr_area_compute_object(corners_pts)
            inter_h = torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4])
            oniou_h = torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4])
            inter_h[inter_h < 0] = 0
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            inter_area_cuda = inter_area
            mbr_area_cuda = mbr_area
            volume_inc = inter_h.mul(inter_area_cuda)
            volume_con = oniou_h.mul(mbr_area_cuda)
            volume_union = volume_gboxes + volume_qboxes - volume_inc
            volume_ca = volume_con - volume_union
            ious = torch.div(volume_inc, volume_union)
            gious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
            gious[index_loc[:, 0]] = ious[index_loc[:, 0]] - torch.div(volume_ca[index_loc[:, 0]], volume_con[index_loc[:, 0]])
            for i in range(inter_area.shape[0]):
                if gious[i] < -1:
                    None
        return torch.unsqueeze(gious, 1)


class gious_3d_loss(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(gious_3d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        gious_3D_object = gious_3D()
        gious = gious_3D_object(gboxes, qboxes)
        loss = torch.sum(gious)
        return loss


class ious_3D(nn.Module):

    def _init_(self, gboxes, qboxes, aligned=False):
        super(ious_3D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, aligned=False):
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        gious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
        if aligned:
            align_inter_aligned_object = align_inter_aligned()
            inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(gboxes, qboxes)
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
            iou_xoz = torch.div(inter_area_xoz, union_xoz)
            iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz, mbr_area_xoz)
            gious_xoz = iou_xoz - iou_bis_xoz
            union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
            iou_xoy = torch.div(inter_area_xoy, union_xoy)
            iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
            gious_xoy = iou_xoy - iou_bis_xoy
            union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
            iou_yoz = torch.div(inter_area_yoz, union_yoz)
            iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
            gious_xoy = iou_yoz - iou_bis_yoz
            gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]]) / 3.0
        else:
            rbbox_to_corners_object = rbbox_to_corners()
            corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 2, 3, 5, 6]])
            corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 2, 3, 5, 6]])
            rinter_area_compute_object = rinter_area_compute()
            inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)
            inter_h = torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4])
            inter_h[inter_h < 0] = 0
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            inter_area_cuda = inter_area
            volume_inc = inter_h.mul(inter_area_cuda)
            volume_union = volume_gboxes + volume_qboxes - volume_inc
            ious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
            ious[index_loc[:, 0]] = torch.div(volume_inc[index_loc[:, 0]], volume_union[index_loc[:, 0]])
        return torch.unsqueeze(ious, 1)


    class gious_3D(nn.Module):

        def _init_(self, gboxes, qboxes, aligned=False):
            super(gious_3D, self)._init_()
            self.gboxes = gboxes
            self.qboxes = qboxes
            self.aligned = aligned
            return

        def forward(ctx, gboxes, qboxes, aligned):
            assert gboxes.shape[0] == qboxes.shape[0]
            indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
            index_loc = torch.nonzero(indicator)
            gious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
            if aligned:
                align_inter_aligned_object = align_inter_aligned()
                inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(gboxes, qboxes)
                volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
                volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
                union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
                iou_xoz = torch.div(inter_area_xoz, union_xoz)
                iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz, mbr_area_xoz)
                gious_xoz = iou_xoz - iou_bis_xoz
                union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
                iou_xoy = torch.div(inter_area_xoy, union_xoy)
                iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
                gious_xoy = iou_xoy - iou_bis_xoy
                union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
                iou_yoz = torch.div(inter_area_yoz, union_yoz)
                iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
                gious_xoy = iou_yoz - iou_bis_yoz
                gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]]) / 3.0
            else:
                corners_gboxes = rbbox_to_corners(gboxes[:, [0, 2, 3, 5, 6]])
                corners_qboxes = rbbox_to_corners(qboxes[:, [0, 2, 3, 5, 6]])
                rinter_area_compute_object = rinter_area_compute()
                inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)
                corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
                corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
                corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)
                mbr_area_compute_object = mbr_area_compute()
                mbr_area = mbr_area_compute_object(corners_pts)
                inter_h = torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4])
                oniou_h = torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4])
                inter_h[inter_h < 0] = 0
                volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
                volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
                inter_area_cuda = inter_area
                mbr_area_cuda = mbr_area
                volume_inc = inter_h.mul(inter_area_cuda)
                volume_con = oniou_h.mul(mbr_area_cuda)
                volume_union = volume_gboxes + volume_qboxes - volume_inc
                volume_ca = volume_con - volume_union
                ious = torch.div(volume_inc, volume_union)
                gious = torch.zeros([gboxes.shape[0]], device=gboxes.device, dtype=torch.float32)
                gious[index_loc[:, 0]] = ious[index_loc[:, 0]] - torch.div(volume_ca[index_loc[:, 0]], volume_con[index_loc[:, 0]])
                for i in range(inter_area.shape[0]):
                    if gious[i] < -1:
                        None
            return torch.unsqueeze(gious, 1)


class ious_3d_loss(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(ious_3d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        ious_3D_object = ious_3D()
        ious = ious_3D_object(gboxes, qboxes)
        loss = torch.mean(1 - ious)
        return loss


class _second_box_decode_operation(nn.Module):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """

    def _init_(self, box_encodings, anchors, encode_angle_to_vector, smooth_dim):
        super(_second_box_decode_operation, self)._init_()
        self.box_encodings = box_encodings
        self.anchors = anchors
        self.encode_angle_to_vector = False
        self.smooth_dim = False
        return

    def forward(ctx, box_encodings, anchors, encode_angle_to_vector, smooth_dim):
        """box decode for VoxelNet in lidar
        Args:
            boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
            anchors ([N, 7] Tensor): anchors
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za
        if smooth_dim:
            lg = (lt + 1) * la
            wg = (wt + 1) * wa
            hg = (ht + 1) * ha
        else:
            lg = torch.exp(lt) * la
            wg = torch.exp(wt) * wa
            hg = torch.exp(ht) * ha
        if encode_angle_to_vector:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


class gious_2D(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(gious_2D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        corners_gboxes = rbbox_to_corners(gboxes[:, [0, 1, 3, 4, 6]])
        corners_qboxes = rbbox_to_corners(qboxes[:, [0, 1, 3, 4, 6]])
        rinter_area_compute_object = rinter_area_compute()
        inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)
        mbr_area_compute_object = mbr_area_compute()
        mbr_area = mbr_area_compute_object(corners_gboxes, corners_qboxes)
        union_area = gboxes[:, 2].mul(gboxes[:, 3]) + qboxes[:, 2].mul(qboxes[:, 3]) - inter_area
        ious = torch.div(inter_area, union_area)
        gious = ious - torch.div(mbr_area - union_area, mbr_area)
        return gious


class gious_2d_loss(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(gious_2d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        gious_2D_object = gious_2D()
        gious = gious_2D_object(gboxes, qboxes)
        loss = torch.sum(gious)
        return loss


class corners_loss(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(corners_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(self, gboxes, qboxes):
        corners_gboxes = rbbox_to_corners(gboxes[:, [0, 2, 3, 5, 6]])
        corners_qboxes = rbbox_to_corners(qboxes[:, [0, 2, 3, 5, 6]])
        loss = torch.sum(torch.abs(corners_gboxes - corners_qboxes))
        return loss


class DiceLoss(nn.Module):

    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)


def _sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = target_tensor * prediction_probabilities + (1 - target_tensor) * (1 - prediction_probabilities)
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha)
        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
        return focal_cross_entropy_loss * weights


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


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


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
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SigmoidFocalClassificationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (mbr_convex_hull,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 4])], {}),
     True),
    (rbbox_corners_aligned,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_hlesmqh_WS3D(_paritybench_base):
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

