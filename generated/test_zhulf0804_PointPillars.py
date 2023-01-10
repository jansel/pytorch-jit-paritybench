import sys
_module = sys.modules[__name__]
del sys
main = _module
dataset = _module
data_aug = _module
dataloader = _module
kitti = _module
evaluate = _module
loss = _module
loss = _module
vis_data_gt = _module
model = _module
anchors = _module
pointpillars = _module
ops = _module
iou3d_module = _module
setup = _module
voxel_module = _module
pre_process_kitti = _module
test = _module
train = _module
utils = _module
io = _module
process = _module
vis_o3d = _module

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


import random


import numpy as np


import torch


from torch.utils.data import DataLoader


from functools import partial


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.tensorboard import SummaryWriter


import copy


class Loss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, beta=1 / 9, cls_w=1.0, reg_w=2.0, dir_w=0.2):
        super().__init__()
        self.alpha = 0.25
        self.gamma = 2.0
        self.cls_w = cls_w
        self.reg_w = reg_w
        self.dir_w = dir_w
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none', beta=beta)
        self.dir_cls = nn.CrossEntropyLoss()

    def forward(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_labels, num_cls_pos, batched_bbox_reg, batched_dir_labels):
        """
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        """
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float()
        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        loss_dict = {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'dir_cls_loss': dir_cls_loss, 'total_loss': total_loss}
        return loss_dict


class _Voxelization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
        voxel_num = hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3, deterministic)
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num].flip(-1)
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


class Voxelization(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        input: shape=(N, c)
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        return _Voxelization.apply(input, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr


class PillarLayer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        """
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        """
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)
        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=0.001, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        """
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        """
        device = pillars.device
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1)
        features[:, :, 0:1] = x_offset_pi_center
        features[:, :, 1:2] = y_offset_pi_center
        voxel_ids = torch.arange(0, pillars.size(1))
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]
        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas


class Backbone(nn.Module):

    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=0.001, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))
            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=0.001, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))
            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):

    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], out_channels[i], upsample_strides[i], stride=upsample_strides[i], bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=0.001, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))
            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):

    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channel, n_anchors * n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors * 2, 1)
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        """
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class Anchors:

    def __init__(self, ranges, sizes, rotations):
        assert len(ranges) == len(sizes)
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        """
        feature_map_size: (y_l, x_l)
        anchor_range: [x1, y1, z1, x2, y2, z2]
        anchor_size: [w, l, h]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 2, 7)
        """
        device = feature_map_size.device
        x_centers = torch.linspace(anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=device)
        y_centers = torch.linspace(anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=device)
        z_centers = torch.linspace(anchor_range[2], anchor_range[5], 1 + 1, device=device)
        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2
        x_centers = x_centers[:feature_map_size[1]] + x_shift
        y_centers = y_centers[:feature_map_size[0]] + y_shift
        z_centers = z_centers[:1] + z_shift
        meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        meshgrids = list(meshgrids)
        for i in range(len(meshgrids)):
            meshgrids[i] = meshgrids[i][..., None]
        anchor_size = anchor_size[None, None, None, None, :]
        repeat_shape = [feature_map_size[1], feature_map_size[0], 1, len(rotations), 1]
        anchor_size = anchor_size.repeat(repeat_shape)
        meshgrids.insert(3, anchor_size)
        anchors = torch.cat(meshgrids, dim=-1).permute(2, 1, 0, 3, 4).contiguous()
        return anchors.squeeze(0)

    def get_multi_anchors(self, feature_map_size):
        """
        feature_map_size: (y_l, x_l)
        ranges: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
        sizes: [[w, l, h], [w, l, h], [w, l, h]]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 3, 2, 7)
        """
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)
        sizes = torch.tensor(self.sizes, device=device)
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for i in range(len(ranges)):
            anchors = self.get_anchors(feature_map_size=feature_map_size, anchor_range=ranges[i], anchor_size=sizes[i], rotations=rotations)
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)
        return multi_anchors


def bboxes2deltas(bboxes, anchors):
    """
    bboxes: (M, 7), (x, y, z, w, l, h, theta)
    anchors: (M, 7)
    return: (M, 7)
    """
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    dx = (bboxes[:, 0] - anchors[:, 0]) / da
    dy = (bboxes[:, 1] - anchors[:, 1]) / da
    zb = bboxes[:, 2] + bboxes[:, 5] / 2
    za = anchors[:, 2] + anchors[:, 5] / 2
    dz = (zb - za) / anchors[:, 5]
    dw = torch.log(bboxes[:, 3] / anchors[:, 3])
    dl = torch.log(bboxes[:, 4] / anchors[:, 4])
    dh = torch.log(bboxes[:, 5] / anchors[:, 5])
    dtheta = bboxes[:, 6] - anchors[:, 6]
    deltas = torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=1)
    return deltas


def iou2d(bboxes1, bboxes2, metric=0):
    """
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    """
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])
    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)
    iou_area = bboxes_w * bboxes_h
    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-08)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-08)
    return iou


def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def nearest_bev(bboxes):
    """
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    """
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi)
    bboxes_bev = torch.where(torch.abs(bboxes_angle[:, None]) > np.pi / 4, bboxes_bev[:, [0, 1, 3, 2]], bboxes_bev)
    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat([bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1)
    return bboxes_bev_x1y1x2y2


def iou2d_nearest(bboxes1, bboxes2):
    """
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    """
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def anchor_target(batched_anchors, batched_gt_bboxes, batched_gt_labels, assigners, nclasses):
    """
    batched_anchors: [(y_l, x_l, 3, 2, 7), (y_l, x_l, 3, 2, 7), ... ]
    batched_gt_bboxes: [(n1, 7), (n2, 7), ...]
    batched_gt_labels: [(n1, ), (n2, ), ...]
    return: 
           dict = {batched_anchors_labels: (bs, n_anchors),
                   batched_labels_weights: (bs, n_anchors),
                   batched_anchors_reg: (bs, n_anchors, 7),
                   batched_reg_weights: (bs, n_anchors),
                   batched_anchors_dir: (bs, n_anchors),
                   batched_dir_weights: (bs, n_anchors)}
    """
    assert len(batched_anchors) == len(batched_gt_bboxes) == len(batched_gt_labels)
    batch_size = len(batched_anchors)
    n_assigners = len(assigners)
    batched_labels, batched_label_weights = [], []
    batched_bbox_reg, batched_bbox_reg_weights = [], []
    batched_dir_labels, batched_dir_labels_weights = [], []
    for i in range(batch_size):
        anchors = batched_anchors[i]
        gt_bboxes, gt_labels = batched_gt_bboxes[i], batched_gt_labels[i]
        multi_labels, multi_label_weights = [], []
        multi_bbox_reg, multi_bbox_reg_weights = [], []
        multi_dir_labels, multi_dir_labels_weights = [], []
        d1, d2, d3, d4, d5 = anchors.size()
        for j in range(n_assigners):
            assigner = assigners[j]
            pos_iou_thr, neg_iou_thr, min_iou_thr = assigner['pos_iou_thr'], assigner['neg_iou_thr'], assigner['min_iou_thr']
            cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)
            overlaps = iou2d_nearest(gt_bboxes, cur_anchors)
            max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)
            gt_max_overlaps, _ = torch.max(overlaps, dim=1)
            assigned_gt_inds = -torch.ones_like(cur_anchors[:, 0], dtype=torch.long)
            assigned_gt_inds[max_overlaps < neg_iou_thr] = 0
            assigned_gt_inds[max_overlaps >= pos_iou_thr] = max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1
            for i in range(len(gt_bboxes)):
                if gt_max_overlaps[i] >= min_iou_thr:
                    assigned_gt_inds[overlaps[i] == gt_max_overlaps[i]] = i + 1
            pos_flag = assigned_gt_inds > 0
            neg_flag = assigned_gt_inds == 0
            assigned_gt_labels = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long) + nclasses
            assigned_gt_labels[pos_flag] = gt_labels[assigned_gt_inds[pos_flag] - 1].long()
            assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_labels_weights[pos_flag] = 1
            assigned_gt_labels_weights[neg_flag] = 1
            assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_reg_weights[pos_flag] = 1
            assigned_gt_reg = torch.zeros_like(cur_anchors)
            positive_anchors = cur_anchors[pos_flag]
            corr_gt_bboxes = gt_bboxes[assigned_gt_inds[pos_flag] - 1]
            assigned_gt_reg[pos_flag] = bboxes2deltas(corr_gt_bboxes, positive_anchors)
            assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_dir_weights[pos_flag] = 1
            assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)
            dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi)
            dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
            assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)
            multi_labels.append(assigned_gt_labels.reshape(d1, d2, 1, d4))
            multi_label_weights.append(assigned_gt_labels_weights.reshape(d1, d2, 1, d4))
            multi_bbox_reg.append(assigned_gt_reg.reshape(d1, d2, 1, d4, -1))
            multi_bbox_reg_weights.append(assigned_gt_reg_weights.reshape(d1, d2, 1, d4))
            multi_dir_labels.append(assigned_gt_dir.reshape(d1, d2, 1, d4))
            multi_dir_labels_weights.append(assigned_gt_dir_weights.reshape(d1, d2, 1, d4))
        multi_labels = torch.cat(multi_labels, dim=-2).reshape(-1)
        multi_label_weights = torch.cat(multi_label_weights, dim=-2).reshape(-1)
        multi_bbox_reg = torch.cat(multi_bbox_reg, dim=-3).reshape(-1, d5)
        multi_bbox_reg_weights = torch.cat(multi_bbox_reg_weights, dim=-2).reshape(-1)
        multi_dir_labels = torch.cat(multi_dir_labels, dim=-2).reshape(-1)
        multi_dir_labels_weights = torch.cat(multi_dir_labels_weights, dim=-2).reshape(-1)
        batched_labels.append(multi_labels)
        batched_label_weights.append(multi_label_weights)
        batched_bbox_reg.append(multi_bbox_reg)
        batched_bbox_reg_weights.append(multi_bbox_reg_weights)
        batched_dir_labels.append(multi_dir_labels)
        batched_dir_labels_weights.append(multi_dir_labels_weights)
    rt_dict = dict(batched_labels=torch.stack(batched_labels, 0), batched_label_weights=torch.stack(batched_label_weights, 0), batched_bbox_reg=torch.stack(batched_bbox_reg, 0), batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0), batched_dir_labels=torch.stack(batched_dir_labels, 0), batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0))
    return rt_dict


def anchors2bboxes(anchors, deltas):
    """
    anchors: (M, 7),  (x, y, z, w, l, h, theta)
    deltas: (M, 7)
    return: (M, 7)
    """
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = deltas[:, 0] * da + anchors[:, 0]
    y = deltas[:, 1] * da + anchors[:, 1]
    z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2
    w = anchors[:, 3] * torch.exp(deltas[:, 3])
    l = anchors[:, 4] * torch.exp(deltas[:, 4])
    h = anchors[:, 5] * torch.exp(deltas[:, 5])
    z = z - h / 2
    theta = anchors[:, 6] + deltas[:, 6]
    bboxes = torch.stack([x, y, z, w, l, h, theta], dim=1)
    return bboxes


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out]].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


class PointPillars(nn.Module):

    def __init__(self, nclasses=3, voxel_size=[0.16, 0.16, 4], point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1], max_num_points=32, max_voxels=(16000, 40000)):
        super().__init__()
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, point_cloud_range=point_cloud_range, max_num_points=max_num_points, max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range, in_channel=9, out_channel=64)
        self.backbone = Backbone(in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[64, 128, 256], upsample_strides=[1, 2, 4], out_channels=[128, 128, 128])
        self.head = Head(in_channel=384, n_anchors=2 * nclasses, n_classes=nclasses)
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)
        self.assigners = [{'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35}, {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45}]
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        """
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        """
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]
        bbox_pred = anchors2bboxes(anchors, bbox_pred)
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2, bbox_pred2d_xy + bbox_pred2d_lw / 2, bbox_pred[:, 6:]], dim=-1)
        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue
            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, scores=cur_bbox_cls_pred, thresh=self.nms_thr, pre_maxsize=None, post_max_size=None)
            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi)
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi
            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {'lidar_bboxes': ret_bboxes.detach().cpu().numpy(), 'labels': ret_labels.detach().cpu().numpy(), 'scores': ret_scores.detach().cpu().numpy()}
        return result

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        """
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        """
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i], bbox_pred=bbox_pred[i], bbox_dir_cls_pred=bbox_dir_cls_pred[i], anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        xs = self.backbone(pillar_features)
        x = self.neck(xs)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, batched_gt_bboxes=batched_gt_bboxes, batched_gt_labels=batched_gt_labels, assigners=self.assigners, nclasses=self.nclasses)
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, bbox_pred=bbox_pred, bbox_dir_cls_pred=bbox_dir_cls_pred, batched_anchors=batched_anchors)
            return results
        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, bbox_pred=bbox_pred, bbox_dir_cls_pred=bbox_dir_cls_pred, batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Head,
     lambda: ([], {'in_channel': 4, 'n_anchors': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zhulf0804_PointPillars(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

