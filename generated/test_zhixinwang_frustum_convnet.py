import sys
_module = sys.modules[__name__]
del sys
configs = _module
collections = _module
config = _module
datasets = _module
check_utils = _module
data_utils = _module
dataset_info = _module
provider_sample = _module
provider_sample_refine = _module
kitti = _module
draw_util = _module
kitti_object = _module
kitti_util = _module
prepare_data = _module
prepare_data_refine = _module
models = _module
box_transform = _module
common = _module
det_base = _module
model_util = _module
pybind11 = _module
rbbox_iou = _module
rbbox_iou_torch = _module
setup = _module
query_depth_point = _module
query_depth_point = _module
test = _module
train = _module
test_net_det = _module
train_net_det = _module
utils = _module
box_util = _module
logger = _module
training_states = _module

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


import math


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from torch.autograd import Function


import random as pyrandom


import logging


import torch.optim as optim


import torch.backends.cudnn as cudnn


def separable_conv2d(in_channels, out_channels, k, s=(1, 1), depth_multiplier=1
    ):
    conv = [nn.Conv2d(in_channels, in_channels * depth_multiplier, k,
        groups=in_channels)]
    if out_channels is not None:
        conv += [nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, 1)]
    conv = nn.Sequential(*conv)
    return conv


class XConv(nn.Module):

    def __init__(self, K, C, depth_multiplier=1, with_X_transformation=True):
        super(XConv, self).__init__()
        self.conv_t0 = nn.Sequential(nn.Conv2d(3, K * K, (1, K)), nn.ELU(
            inplace=True), nn.BatchNorm2d(K * K))
        self.conv_t1 = nn.Sequential(separable_conv2d(K, None, (1, K), (1, 
            1), K), nn.ELU(inplace=True), nn.BatchNorm2d(K * K))
        self.conv_t2 = nn.Sequential(separable_conv2d(K, None, (1, K), (1, 
            1), K), nn.BatchNorm2d(K * K))
        self.separable_conv2d = nn.Sequential(separable_conv2d(C, None, (1,
            K), (1, 1), depth_multiplier), nn.ELU(inplace=True), nn.
            BatchNorm2d(C))
        self.with_X_transformation = with_X_transformation
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, nn_pts_local, nn_fts_input):
        """
        pts: nn_pts_local (N, 3, P, K)
        nn_fts_input : (N, C, P, K) 

        """
        N, C0, P, K = nn_pts_local.shape
        assert C0 == 3
        if self.with_X_transformation:
            X_0 = self.conv_t0(nn_pts_local)
            X_0 = X_0.view(N, K, K, P).transpose(2, 3).contiguous()
            X_1 = self.conv_t1(X_0)
            X_1 = X_1.view(N, K, K, P).transpose(2, 3).contiguous()
            X_2 = self.conv_t2(X_1)
            X_2 = X_1.view(N, K, K, P).permute(0, 3, 1, 2).contiguous()
            X = X_2.view(N * P, K, K)
            nn_fts_input = nn_fts_input.permute(0, 2, 3, 1).contiguous().view(
                N * P, K, -1)
            fts_X = torch.bmm(X, nn_fts_input)
            fts_X = fts_X.view(N, P, K, -1).permute(0, 3, 1, 2).contiguous()
        else:
            fts_X = nn_fts_input
        fts = self.separable_conv2d(fts_X)
        return fts


def init_params(m, method='constant'):
    """
    method: xavier_uniform, kaiming_normal, constant
    """
    if isinstance(m, list):
        for im in m:
            init_params(im, method)
    else:
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        elif isinstance(method, (int, float)):
            m.weight.data.fill_(method)
        else:
            raise ValueError('unknown method.')
        if m.bias is not None:
            m.bias.data.zero_()


def Conv2d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p, bias=False), nn.
            BatchNorm2d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p), nn.ReLU(True))


class PointNetModule(nn.Module):

    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature
        =True):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz
        if Infea > 0:
            use_feature = True
        else:
            use_feature = False
        self.use_feature = use_feature
        self.query_depth_point = QueryDepthPoint(dist, nsample)
        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)
        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)
        init_params([self.conv1[0], self.conv2[0], self.conv3[0]],
            'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self, pc, feat, new_pc=None):
        batch_size = pc.size(0)
        npoint = new_pc.shape[2]
        k = self.nsample
        indices, num = self.query_depth_point(pc, new_pc)
        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0
        grouped_pc = None
        grouped_feature = None
        if self.use_xyz:
            grouped_pc = torch.gather(pc, 2, indices.view(batch_size, 1, 
                npoint * k).expand(-1, 3, -1)).view(batch_size, 3, npoint, k)
            grouped_pc = grouped_pc - new_pc.unsqueeze(3)
        if self.use_feature:
            grouped_feature = torch.gather(feat, 2, indices.view(batch_size,
                1, npoint * k).expand(-1, feat.size(1), -1)).view(batch_size,
                feat.size(1), npoint, k)
        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()
        grouped_feature = self.conv1(grouped_feature)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)
        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = grouped_feature * valid.float()
        return grouped_feature


_global_config['DATA'] = 4


class PointNetFeat(nn.Module):

    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()
        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        assert len(u) == 4
        self.pointnet1 = PointNetModule(input_channel - 3, [64, 64, 128], u
            [0], 32, use_xyz=True, use_feature=True)
        self.pointnet2 = PointNetModule(input_channel - 3, [64, 64, 128], u
            [1], 64, use_xyz=True, use_feature=True)
        self.pointnet3 = PointNetModule(input_channel - 3, [128, 128, 256],
            u[2], 64, use_xyz=True, use_feature=True)
        self.pointnet4 = PointNetModule(input_channel - 3, [256, 256, 512],
            u[3], 128, use_xyz=True, use_feature=True)

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        pc = point_cloud
        pc1 = sample_pc[0]
        pc2 = sample_pc[1]
        pc3 = sample_pc[2]
        pc4 = sample_pc[3]
        feat1 = self.pointnet1(pc, feat, pc1)
        feat1, _ = torch.max(feat1, -1)
        feat2 = self.pointnet2(pc, feat, pc2)
        feat2, _ = torch.max(feat2, -1)
        feat3 = self.pointnet3(pc, feat, pc3)
        feat3, _ = torch.max(feat3, -1)
        feat4 = self.pointnet4(pc, feat, pc4)
        feat4, _ = torch.max(feat4, -1)
        if one_hot_vec is not None:
            assert self.num_vec == one_hot_vec.shape[1]
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat1.shape[-1])
            feat1 = torch.cat([feat1, one_hot], 1)
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat2.shape[-1])
            feat2 = torch.cat([feat2, one_hot], 1)
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat3.shape[-1])
            feat3 = torch.cat([feat3, one_hot], 1)
            one_hot = one_hot_vec.unsqueeze(-1).expand(-1, -1, feat4.shape[-1])
            feat4 = torch.cat([feat4, one_hot], 1)
        return feat1, feat2, feat3, feat4


def Conv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p, bias=False), nn.
            BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p), nn.ReLU(True))


def DeConv1d(i_c, o_c, k, s=1, p=0, bn=True):
    if bn:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p, bias=
            False), nn.BatchNorm1d(o_c), nn.ReLU(True))
    else:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p), nn.ReLU
            (True))


class ConvFeatNet(nn.Module):

    def __init__(self, i_c=128, num_vec=3):
        super(ConvFeatNet, self).__init__()
        self.block1_conv1 = Conv1d(i_c + num_vec, 128, 3, 1, 1)
        self.block2_conv1 = Conv1d(128, 128, 3, 2, 1)
        self.block2_conv2 = Conv1d(128, 128, 3, 1, 1)
        self.block2_merge = Conv1d(128 + 128 + num_vec, 128, 1, 1)
        self.block3_conv1 = Conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = Conv1d(256, 256, 3, 1, 1)
        self.block3_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)
        self.block4_conv1 = Conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = Conv1d(512, 512, 3, 1, 1)
        self.block4_merge = Conv1d(512 + 512 + num_vec, 512, 1, 1)
        self.block2_deconv = DeConv1d(128, 256, 1, 1, 0)
        self.block3_deconv = DeConv1d(256, 256, 2, 2, 0)
        self.block4_deconv = DeConv1d(512, 256, 4, 4, 0)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        x = self.block1_conv1(x1)
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = torch.cat([x, x2], 1)
        x = self.block2_merge(x)
        xx1 = x
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = torch.cat([x, x3], 1)
        x = self.block3_merge(x)
        xx2 = x
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = torch.cat([x, x4], 1)
        x = self.block4_merge(x)
        xx3 = x
        xx1 = self.block2_deconv(xx1)
        xx2 = self.block3_deconv(xx2)
        xx3 = self.block4_deconv(xx3)
        x = torch.cat([xx1, xx2[:, :, :xx1.shape[-1]], xx3[:, :, :xx1.shape
            [-1]]], 1)
        return x


def angle_encode(gt_angle, num_bins=12):
    gt_angle = gt_angle % (2 * np.pi)
    assert ((gt_angle >= 0) & (gt_angle <= 2 * np.pi)).all()
    angle_per_class = 2 * np.pi / float(num_bins)
    shifted_angle = (gt_angle + angle_per_class / 2) % (2 * np.pi)
    gt_class_id = torch.floor(shifted_angle / angle_per_class).long()
    gt_res = shifted_angle - (gt_class_id.float() * angle_per_class + 
        angle_per_class / 2)
    gt_res /= angle_per_class / 2
    return gt_class_id, gt_res


def huber_loss(error, delta, weight=None):
    delta = torch.ones_like(error) * delta
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    losses = 0.5 * quadratic ** 2 + delta * linear
    if weight is not None:
        losses *= weight
    return losses.mean()


def size_encode(gt, class_mean_size, size_class_label):
    ex = class_mean_size[size_class_label]
    return (gt - ex) / ex


def center_encode(gt, ex):
    return gt - ex


def softmax_focal_loss_ignore(prob, target, alpha=0.25, gamma=2, ignore_idx=-1
    ):
    keep = (target != ignore_idx).nonzero().view(-1)
    num_fg = (target > 0).data.sum()
    target = target[keep]
    prob = prob[(keep), :]
    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1
        ).float()
    prob_t = prob[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)
    loss = loss.sum() / (num_fg + 1e-14)
    return loss


def get_accuracy(output, target, ignore=None):
    assert output.shape[0] == target.shape[0]
    if ignore is not None:
        assert isinstance(ignore, int)
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]
    pred = torch.argmax(output, -1)
    correct = (pred.view(-1) == target.view(-1)).float().sum()
    acc = correct * (1.0 / target.view(-1).shape[0])
    return acc


def angle_decode(ex_res, ex_class_id, num_bins=12, to_label_format=True):
    ex_res_select = torch.gather(ex_res, 1, ex_class_id.unsqueeze(1))
    ex_res_select = ex_res_select.squeeze(1)
    angle_per_class = 2 * np.pi / float(num_bins)
    angle = ex_class_id.float() * angle_per_class + ex_res_select * (
        angle_per_class / 2)
    if to_label_format:
        flag = angle > np.pi
        angle[flag] = angle[flag] - 2 * np.pi
    return angle


class KITTICategory(object):
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    CLASS_MEAN_SIZE = {'Car': np.array([3.88311640418, 1.62856739989, 
        1.52563191462]), 'Pedestrian': np.array([0.84422524, 0.66068622, 
        1.76255119]), 'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127])
        }
    NUM_SIZE_CLUSTER = len(CLASSES)
    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[(i), :] = CLASS_MEAN_SIZE[CLASSES[i]]


DATASET_INFO = {'KITTI': KITTICategory}


def center_decode(ex, offset):
    return ex + offset


def size_decode(offset, class_mean_size, size_class_label):
    offset_select = torch.gather(offset, 1, size_class_label.view(-1, 1, 1)
        .expand(-1, -1, 3))
    offset_select = offset_select.squeeze(1)
    ex = class_mean_size[size_class_label]
    return offset_select * ex + ex


def get_box3d_corners_helper(centers, headings, sizes):
    N = centers.shape[0]
    l = sizes[:, (0)]
    w = sizes[:, (1)]
    h = sizes[:, (2)]
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l /
        2, -l / 2], 1)
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h /
        2, -h / 2], 1)
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -
        w / 2, w / 2], 1)
    corners = torch.stack([x_corners, y_corners, z_corners], 1)
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = headings.new_ones(N)
    zeros = headings.new_zeros(N)
    row1 = torch.stack([c, zeros, s], 1)
    row2 = torch.stack([zeros, ones, zeros], 1)
    row3 = torch.stack([-s, zeros, c], 1)
    R = torch.stack([row1, row2, row3], 1)
    corners_3d = torch.bmm(R, corners)
    corners_3d = corners_3d + centers.unsqueeze(2)
    corners_3d = torch.transpose(corners_3d, 1, 2).contiguous()
    return corners_3d


def boxes3d2corners(boxes_3d):
    """ b, 7 (cx, cy, cz, l, w, h, r)"""
    N = boxes_3d.shape[0]
    centers = boxes_3d[:, :3]
    l = boxes_3d[:, (3)]
    w = boxes_3d[:, (4)]
    h = boxes_3d[:, (5)]
    headings = boxes_3d[:, (6)]
    x_corners = np.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 
        2, -l / 2], 1)
    y_corners = np.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 
        2, -h / 2], 1)
    z_corners = np.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w /
        2, w / 2], 1)
    corners = np.stack([x_corners, y_corners, z_corners], 1)
    c = np.cos(headings)
    s = np.sin(headings)
    ones = np.ones(N, dtype=boxes_3d.dtype)
    zeros = np.zeros(N, dtype=boxes_3d.dtype)
    row1 = np.stack([c, zeros, s], 1)
    row2 = np.stack([zeros, ones, zeros], 1)
    row3 = np.stack([-s, zeros, c], 1)
    R = np.stack([row1, row2, row3], 1)
    corners_3d = np.einsum('bij,bjk->bik', R, corners)
    corners_3d = corners_3d + np.expand_dims(centers, 2)
    corners_3d = np.transpose(corners_3d, (0, 2, 1))
    return corners_3d


def rbbox_iou_3d_pair(boxes_3d, qboxes_3d):
    """
    boxes_3d, qboxes_3d: (cx, cy, cz, l, w, h, r) n, 7

    """
    assert boxes_3d.shape == qboxes_3d.shape
    bbox_corner_3d = boxes3d2corners(boxes_3d)
    qbbox_corner_3d = boxes3d2corners(qboxes_3d)
    o = box_ops_cc.rbbox_iou_3d_pair(bbox_corner_3d, qbbox_corner_3d)
    return o


_global_config['IOU_THRESH'] = 4


_global_config['LOSS'] = 4


class PointNetDet(nn.Module):

    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()
        dataset_name = cfg.DATA.DATASET_NAME
        assert dataset_name in DATASET_INFO
        self.category_info = DATASET_INFO[dataset_name]
        self.num_size_cluster = len(self.category_info.CLASSES)
        self.mean_size_array = self.category_info.MEAN_SIZE_ARRAY
        self.feat_net = PointNetFeat(input_channel, num_vec)
        self.conv_net = ConvFeatNet(128, num_vec)
        self.num_classes = num_classes
        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins
        output_size = 3 + num_bins * 2 + self.num_size_cluster * 4
        self.reg_out = nn.Conv1d(768, output_size, 1)
        self.cls_out = nn.Conv1d(768, 2, 1)
        self.relu = nn.ReLU(True)
        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')
        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def _slice_output(self, output):
        batch_size = output.shape[0]
        num_bins = self.num_bins
        num_sizes = self.num_size_cluster
        center = output[:, 0:3].contiguous()
        heading_scores = output[:, 3:3 + num_bins].contiguous()
        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous(
            )
        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + num_sizes
            ].contiguous()
        size_res_norm = output[:, 3 + num_bins * 2 + num_sizes:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, num_sizes, 3)
        return (center, heading_scores, heading_res_norm, size_scores,
            size_res_norm)

    def get_center_loss(self, pred_offsets, gt_offsets):
        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)
        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm,
        heading_class_label, heading_res_norm_label):
        heading_class_loss = F.cross_entropy(heading_scores,
            heading_class_label)
        heading_res_norm_select = torch.gather(heading_res_norm, 1,
            heading_class_label.view(-1, 1))
        heading_res_norm_loss = huber_loss(heading_res_norm_select.squeeze(
            1) - heading_res_norm_label, delta=1.0)
        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label,
        size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)
        size_res_norm_select = torch.gather(size_res_norm, 1,
            size_class_label.view(batch_size, 1, 1).expand(batch_size, 1, 3))
        size_norm_dist = torch.norm(size_res_label_norm -
            size_res_norm_select.squeeze(1), 2, dim=-1)
        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)
        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):
        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds
        corners_3d_gt = get_box3d_corners_helper(center_label,
            heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, 
            heading_label + np.pi, size_label)
        corners_3d_pred = get_box3d_corners_helper(center_preds,
            heading_preds, size_preds)
        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt,
            2, dim=-1).mean(-1), torch.norm(corners_3d_pred -
            corners_3d_gt_flip, 2, dim=-1).mean(-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)
        return corners_loss, corners_3d_gt

    def forward(self, data_dicts):
        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        cls_label = data_dicts.get('label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')
        center_ref1 = data_dicts.get('center_ref1')
        center_ref2 = data_dicts.get('center_ref2')
        center_ref3 = data_dicts.get('center_ref3')
        center_ref4 = data_dicts.get('center_ref4')
        batch_size = point_cloud.shape[0]
        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:, ([3]), :].contiguous()
        else:
            object_point_cloud_i = None
        mean_size_array = torch.from_numpy(self.mean_size_array).type_as(
            point_cloud)
        feat1, feat2, feat3, feat4 = self.feat_net(object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            object_point_cloud_i, one_hot_vec)
        x = self.conv_net(feat1, feat2, feat3, feat4)
        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)
        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)
        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)
        cls_probs = F.softmax(cls_scores, -1)
        if center_label is None:
            assert not self.training, 'Please provide labels for training.'
            det_outputs = self._slice_output(outputs)
            (center_boxnet, heading_scores, heading_res_norm, size_scores,
                size_res_norm) = det_outputs
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)
            center_preds = center_boxnet + center_ref2
            heading_preds = angle_decode(heading_res_norm,
                heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array,
                size_pred_label)
            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)
            size_preds = size_preds.view(batch_size, -1, 3)
            heading_preds = heading_preds.view(batch_size, -1)
            outputs = cls_probs, center_preds, heading_preds, size_preds
            return outputs
        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)
        assert fg_idx.numel() != 0
        outputs = outputs[(fg_idx), :]
        center_ref2 = center_ref2[fg_idx]
        det_outputs = self._slice_output(outputs)
        (center_boxnet, heading_scores, heading_res_norm, size_scores,
            size_res_norm) = det_outputs
        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1),
            ignore_idx=-1)
        center_label = center_label.unsqueeze(1).expand(-1, num_out, -1
            ).contiguous().view(-1, 3)[fg_idx]
        heading_label = heading_label.expand(-1, num_out).contiguous().view(-1
            )[fg_idx]
        size_label = size_label.unsqueeze(1).expand(-1, num_out, -1
            ).contiguous().view(-1, 3)[fg_idx]
        size_class_label = size_class_label.expand(-1, num_out).contiguous(
            ).view(-1)[fg_idx]
        center_gt_offsets = center_encode(center_label, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(
            heading_label, num_bins=self.num_bins)
        size_res_label_norm = size_encode(size_label, mean_size_array,
            size_class_label)
        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(
            heading_scores, heading_res_norm, heading_class_label,
            heading_res_norm_label)
        size_class_loss, size_res_norm_loss = self.get_size_loss(size_scores,
            size_res_norm, size_class_label, size_res_label_norm)
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label,
            num_bins=self.num_bins)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)
        corners_loss, corner_gts = self.get_corner_loss((center_preds,
            heading, size), (center_label, heading_label, size_label))
        BOX_LOSS_WEIGHT = cfg.LOSS.BOX_LOSS_WEIGHT
        CORNER_LOSS_WEIGHT = cfg.LOSS.CORNER_LOSS_WEIGHT
        HEAD_REG_WEIGHT = cfg.LOSS.HEAD_REG_WEIGHT
        SIZE_REG_WEIGHT = cfg.LOSS.SIZE_REG_WEIGHT
        loss = cls_loss + BOX_LOSS_WEIGHT * (center_loss +
            heading_class_loss + size_class_loss + HEAD_REG_WEIGHT *
            heading_res_norm_loss + SIZE_REG_WEIGHT * size_res_norm_loss + 
            CORNER_LOSS_WEIGHT * corners_loss)
        with torch.no_grad():
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1), ignore=-1)
            heading_prec = get_accuracy(heading_probs, heading_class_label.
                view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)
            heading_preds = angle_decode(heading_res_norm,
                heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array,
                size_pred_label)
            corner_preds = get_box3d_corners_helper(center_preds,
                heading_preds, size_preds)
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(),
                corner_gts.detach().cpu().numpy())
            iou2ds, iou3ds = overlap[:, (0)], overlap[:, (1)]
            iou2d_mean = iou2ds.mean()
            iou3d_mean = iou3ds.mean()
            iou3d_gt_mean = (iou3ds >= cfg.IOU_THRESH).mean()
            iou2d_mean = torch.tensor(iou2d_mean).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3d_mean).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor(iou3d_gt_mean).type_as(cls_prec)
        losses = {'total_loss': loss, 'cls_loss': cls_loss, 'center_loss':
            center_loss, 'head_cls_loss': heading_class_loss,
            'head_res_loss': heading_res_norm_loss, 'size_cls_loss':
            size_class_loss, 'size_res_loss': size_res_norm_loss,
            'corners_loss': corners_loss}
        metrics = {'cls_acc': cls_prec, 'head_acc': heading_prec,
            'size_acc': size_prec, 'IoU_2D': iou2d_mean, 'IoU_3D':
            iou3d_mean, ('IoU_' + str(cfg.IOU_THRESH)): iou3d_gt_mean}
        return losses, metrics


class _query_depth_point(Function):

    @staticmethod
    def forward(ctx, dis_z, nsample, xyz1, xyz2):
        """
        Input:
            dis_z: float32, depth distance search distance
            nsample: int32, number of points selected in each ball region
            xyz1: (batch_size, 3, ndataset) float32 array, input points
            xyz2: (batch_size, 3, npoint) float32 array, query points
        Output:
            idx: (batch_size, npoint, nsample) int32 array, indices to input points
            pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        """
        assert xyz1.is_cuda and xyz1.size(1) == 3
        assert xyz2.is_cuda and xyz2.size(1) == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        b = xyz1.size(0)
        n = xyz1.size(1)
        m = xyz2.size(1)
        idx = xyz1.new(b, m, nsample).long().zero_()
        pts_cnt = xyz1.new(b, m).int().zero_()
        query_depth_point_cuda.forward(b, n, m, dis_z, nsample, xyz1, xyz2,
            idx, pts_cnt)
        return idx, pts_cnt

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


class QueryDepthPoint(nn.Module):

    def __init__(self, dis_z, nsample):
        super(QueryDepthPoint, self).__init__()
        self.dis_z = dis_z
        self.nsample = nsample

    def forward(self, xyz1, xyz2):
        return _query_depth_point.apply(self.dis_z, self.nsample, xyz1, xyz2)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhixinwang_frustum_convnet(_paritybench_base):
    pass
