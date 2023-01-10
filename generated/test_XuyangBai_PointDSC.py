import sys
_module = sys.modules[__name__]
del sys
baseline_3DMatch = _module
baseline_KITTI = _module
config = _module
KITTI = _module
Redwood = _module
ThreeDMatch = _module
datasets = _module
dataloader = _module
stats = _module
demo_registration = _module
benchmark_utils = _module
benchmark_utils_predator = _module
test_3DLoMatch = _module
test_3DMatch = _module
test_KITTI = _module
loss = _module
trainer = _module
cal_fcgf = _module
cal_fpfh = _module
eigen = _module
fcgf = _module
svd_speed = _module
OANet = _module
PointDSC = _module
common = _module
fileio = _module
initialize_config = _module
make_fragments = _module
optimize_posegraph = _module
test_multi = _module
test_multi_ate = _module
trajectory = _module
train_3DMatch = _module
train_KITTI = _module
SE3 = _module
utils = _module
max_clique = _module
pointcloud = _module
timer = _module

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


import math


import logging


import numpy as np


import torch.utils.data as data


import random


import copy


from collections import defaultdict


import torch.nn as nn


import torch.nn.functional as F


from sklearn.metrics import recall_score


from sklearn.metrics import precision_score


from sklearn.metrics import f1_score


import warnings


import time


from torch import optim


def decompose_trans(trans):
    """
    Decompose SE3 transformations into R and t, support torch.Tensor and np.ndarry.
    Input
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    """
    if len(trans.shape) == 3:
        return trans[:, :3, :3], trans[:, :3, 3:4]
    else:
        return trans[:3, :3], trans[:3, 3:4]


def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T


class TransformationLoss(nn.Module):

    def __init__(self, re_thre=15, te_thre=30):
        super(TransformationLoss, self).__init__()
        self.re_thre = re_thre
        self.te_thre = te_thre

    def forward(self, trans, gt_trans, src_keypts, tgt_keypts, probs):
        """
        Transformation Loss
        Inputs:
            - trans:      [bs, 4, 4] SE3 transformation matrices
            - gt_trans:   [bs, 4, 4] ground truth SE3 transformation matrices
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - probs:     [bs, num_corr] predicted inlier probability
        Outputs:
            - loss     transformation loss 
            - recall   registration recall (re < re_thre & te < te_thre)
            - RE       rotation error 
            - TE       translation error
            - RMSE     RMSE under the predicted transformation
        """
        bs = trans.shape[0]
        R, t = decompose_trans(trans)
        gt_R, gt_t = decompose_trans(gt_trans)
        recall = 0
        RE = torch.tensor(0.0)
        TE = torch.tensor(0.0)
        RMSE = torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for i in range(bs):
            re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
            te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
            warp_src_keypts = transform(src_keypts[i], trans[i])
            rmse = torch.norm(warp_src_keypts - tgt_keypts, dim=-1).mean()
            re = re * 180 / np.pi
            te = te * 100
            if te < self.te_thre and re < self.re_thre:
                recall += 1
            RE += re
            TE += te
            RMSE += rmse
            pred_inliers = torch.where(probs[i] > 0)[0]
            if len(pred_inliers) < 1:
                loss += torch.tensor(0.0)
            else:
                warp_src_keypts = transform(src_keypts[i], trans[i])
                loss += ((warp_src_keypts - tgt_keypts) ** 2).sum(-1).mean()
        return loss / bs, recall * 100.0 / bs, RE / bs, TE / bs, RMSE / bs


class ClassificationLoss(nn.Module):

    def __init__(self, balanced=True):
        super(ClassificationLoss, self).__init__()
        self.balanced = balanced

    def forward(self, pred, gt, weight=None):
        """ 
        Classification Loss for the inlier confidence
        Inputs:
            - pred: [bs, num_corr] predicted logits/labels for the putative correspondences
            - gt:   [bs, num_corr] ground truth labels
        Outputs:(dict)
            - loss          (weighted) BCE loss for inlier confidence 
            - precision:    inlier precision (# kept inliers / # kepts matches)
            - recall:       inlier recall (# kept inliers / # all inliers)
            - f1:           (precision * recall * 2) / (precision + recall)
            - logits_true:  average logits for inliers
            - logits_false: average logits for outliers
        """
        num_pos = torch.relu(torch.sum(gt) - 1) + 1
        num_neg = torch.relu(torch.sum(1 - gt) - 1) + 1
        if weight is not None:
            loss = nn.BCEWithLogitsLoss(reduction='none')(pred, gt.float())
            loss = torch.mean(loss * weight)
        elif self.balanced is False:
            loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt.float())
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=num_neg * 1.0 / num_pos, reduction='mean')(pred, gt.float())
        pred_labels = pred > 0
        gt, pred_labels, pred = gt.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), pred.detach().cpu().numpy()
        precision = precision_score(gt[0], pred_labels[0])
        recall = recall_score(gt[0], pred_labels[0])
        f1 = f1_score(gt[0], pred_labels[0])
        mean_logit_true = np.sum(pred * gt) / max(1, np.sum(gt))
        mean_logit_false = np.sum(pred * (1 - gt)) / max(1, np.sum(1 - gt))
        eval_stats = {'loss': loss, 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'logit_true': float(mean_logit_true), 'logit_false': float(mean_logit_false)}
        return eval_stats


class SpectralMatchingLoss(nn.Module):

    def __init__(self, balanced=True):
        super(SpectralMatchingLoss, self).__init__()
        self.balanced = balanced

    def forward(self, M, gt_labels):
        """ 
        Spectral Matching Loss
        Inputs:
            - M:    [bs, num_corr, num_corr] feature similarity matrix
            - gt:   [bs, num_corr] ground truth inlier/outlier labels
        Output:
            - loss  
        """
        gt_M = (gt_labels[:, None, :] + gt_labels[:, :, None] == 2).float()
        for i in range(gt_M.shape[0]):
            gt_M[i].fill_diagonal_(0)
        if self.balanced:
            sm_loss_p = ((M - 1) ** 2 * gt_M).sum(-1).sum(-1) / (torch.relu(gt_M.sum(-1).sum(-1) - 1.0) + 1.0)
            sm_loss_n = ((M - 0) ** 2 * (1 - gt_M)).sum(-1).sum(-1) / (torch.relu((1 - gt_M).sum(-1).sum(-1) - 1.0) + 1.0)
            loss = torch.mean(sm_loss_p * 0.5 + sm_loss_n * 0.5)
        else:
            loss = torch.nn.MSELoss(reduction='mean')(M, gt_M)
        return loss


def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, region_type=0, dimension=3):
    if not isinstance(region_type, ME.RegionType):
        if region_type == 0:
            region_type = ME.RegionType.HYPER_CUBE
        elif region_type == 1:
            region_type = ME.RegionType.HYPER_CROSS
        else:
            raise ValueError('Unsupported region type')
    kernel_generator = ME.KernelGenerator(kernel_size=kernel_size, stride=stride, dilation=dilation, region_type=region_type, dimension=dimension)
    return ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, stride=stride, kernel_generator=kernel_generator, dimension=dimension)


def get_norm(norm_type, num_feats, bn_momentum=0.05, dimension=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats)
    elif norm_type == 'INBN':
        return nn.Sequential(ME.MinkowskiInstanceNorm(num_feats), ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum))
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, region_type=0, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, region_type=region_type, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, dilation=dilation, region_type=region_type, dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = MEF.relu(out)
        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = 'INBN'


class diff_pool(nn.Module):

    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(nn.InstanceNorm1d(in_channel, eps=0.001), nn.BatchNorm1d(in_channel), nn.ReLU(inplace=True), nn.Conv1d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2)
        out = torch.matmul(x, S.transpose(1, 2))
        return out


class diff_unpool(nn.Module):

    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(nn.InstanceNorm1d(in_channel, eps=0.001), nn.BatchNorm1d(in_channel), nn.ReLU(inplace=True), nn.Conv1d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1)
        out = torch.matmul(x_down, S)
        return out


class Transpose(nn.Module):

    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):

    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(nn.InstanceNorm1d(channels, eps=0.001), nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Conv1d(channels, out_channels, kernel_size=1), Transpose(1, 2))
        self.conv2 = nn.Sequential(nn.BatchNorm1d(points), nn.ReLU(inplace=True), nn.Conv1d(points, points, kernel_size=1))
        self.conv3 = nn.Sequential(Transpose(1, 2), nn.InstanceNorm1d(out_channels, eps=0.001), nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class ContextNormalization(nn.Module):

    def __init__(self):
        super(ContextNormalization, self).__init__()

    def forward(self, x):
        var_eps = 0.001
        mean = torch.mean(x, 2, keepdim=True)
        variance = torch.var(x, 2, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + var_eps)
        return x


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-06)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-06)
    Am = A - centroid_A
    Bm = B - centroid_B
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U, S, Vt
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    return integrate_trans(R, t)


class OANet(nn.Module):

    def __init__(self, in_dim=6, num_layers=6, num_channels=128, num_clusters=10, act_pos='post'):
        super(OANet, self).__init__()
        assert act_pos == 'pre' or act_pos == 'post'
        self.num_channels = num_channels
        self.num_layers = num_channels
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        modules = [nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)]
        for i in range(num_layers // 2):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.l1_1 = nn.Sequential(*modules)
        modules = []
        for i in range(num_layers // 2):
            modules.append(OAFilter(num_channels, num_clusters))
        self.l2 = nn.Sequential(*modules)
        self.down1 = diff_pool(num_channels, num_clusters)
        self.up1 = diff_unpool(num_channels, num_clusters)
        modules = [nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=True)]
        for i in range(num_layers // 2 - 1):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.l1_2 = nn.Sequential(*modules)
        self.output = nn.Conv1d(num_channels, 1, kernel_size=1)

    def forward(self, data):
        corr_pos = data['corr_pos'].permute(0, 2, 1)
        x1_1 = self.l1_1(corr_pos)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2(torch.cat([x1_1, x_up], dim=1))
        return out
        logits = self.output(out).squeeze(1)
        if len(torch.where(logits > 0)[1]) >= 3:
            R, t = rigid_transform_3d(A=src_keypts[:, torch.where(logits > 0)[1], :], B=tgt_keypts[:, torch.where(logits > 0)[1], :], scores=torch.relu(torch.tanh(logits[:, torch.where(logits > 0)[1]])))
        else:
            R = torch.eye(3)[None, :, :]
            t = torch.ones(1, 3)[None, :, :]
        R = torch.cat([R, torch.zeros_like(R[:, 0:1, :])], dim=1)
        t = t.permute(0, 2, 1)
        t = torch.cat([t, torch.ones_like(t[:, 0:1, :])], dim=1)
        trans = torch.cat([R, t], dim=-1)
        res = {'final_trans': trans, 'final_labels': logits, 'M': None}
        return res


class NonLocalBlock(nn.Module):

    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(nn.Conv1d(num_channels, num_channels // 2, kernel_size=1), nn.BatchNorm1d(num_channels // 2), nn.ReLU(inplace=True), nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1), nn.BatchNorm1d(num_channels // 2), nn.ReLU(inplace=True), nn.Conv1d(num_channels // 2, num_channels, kernel_size=1))
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        weight = torch.softmax(attention[:, None, :, :] * feat_attention, dim=-1)
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res


class NonLocalNet(nn.Module):

    def __init__(self, in_dim=6, num_layers=6, num_channels=128):
        super(NonLocalNet, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True), nn.BatchNorm1d(num_channels), nn.ReLU(inplace=True))
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)

    def forward(self, corr_feat, corr_compatibility):
        """
        Input: 
            - corr_feat:          [bs, in_dim, num_corr]   input feature map
            - corr_compatibility: [bs, num_corr, num_corr] spatial consistency matrix 
        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat, corr_compatibility)
        return feat


def knn(x, k, ignore_self=False, normalized=True):
    """ find feature space knn neighbor of x 
    Input:
        - x:       [bs, num_corr, num_channels],  input features
        - k:       
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_corr, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)
    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


class PointDSC(nn.Module):

    def __init__(self, in_dim=6, num_layers=6, num_channels=128, num_iterations=10, ratio=0.1, inlier_threshold=0.1, sigma_d=0.1, k=40, nms_radius=0.1):
        super(PointDSC, self).__init__()
        self.num_iterations = num_iterations
        self.ratio = ratio
        self.num_channels = num_channels
        self.inlier_threshold = inlier_threshold
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_spat = nn.Parameter(torch.Tensor([sigma_d]).float(), requires_grad=False)
        self.k = k
        self.nms_radius = nms_radius
        self.encoder = NonLocalNet(in_dim=in_dim, num_layers=num_layers, num_channels=num_channels)
        self.classification = nn.Sequential(nn.Conv1d(num_channels, 32, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv1d(32, 32, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv1d(32, 1, kernel_size=1, bias=True))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        """
        Input:
            - corr_pos:   [bs, num_corr, 6]
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
            - testing:    flag for test phase, if False will not calculate M and post-refinement.
        Output: (dict)
            - final_trans:   [bs, 4, 4], the predicted transformation matrix. 
            - final_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
            - M:             [bs, num_corr, num_corr], feature similarity matrix, for SM loss calculation.
            - seed_trans:    [bs, num_seeds, 4, 4],  the predicted transformation matrix associated with each seeding point, deprecated.
            - corr_features: [bs, num_corr, num_channels], the feature for each correspondence, for circle loss calculation, deprecated.
            - confidence:    [bs], confidence of returned results, for safe guard, deprecated.
        """
        corr_pos, src_keypts, tgt_keypts = data['corr_pos'], data['src_keypts'], data['tgt_keypts']
        bs, num_corr = corr_pos.shape[0], corr_pos.shape[1]
        testing = 'testing' in data.keys()
        with torch.no_grad():
            src_dist = torch.norm(src_keypts[:, :, None, :] - src_keypts[:, None, :, :], dim=-1)
            corr_compatibility = src_dist - torch.norm(tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :], dim=-1)
            corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)
        corr_features = self.encoder(corr_pos.permute(0, 2, 1), corr_compatibility).permute(0, 2, 1)
        normed_corr_features = F.normalize(corr_features, p=2, dim=-1)
        if not testing:
            M = torch.matmul(normed_corr_features, normed_corr_features.permute(0, 2, 1))
            M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
        else:
            M = None
        confidence = self.classification(corr_features.permute(0, 2, 1)).squeeze(1)
        if testing:
            seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]
        seed_trans, seed_fitness, final_trans, final_labels = self.cal_seed_trans(seeds, normed_corr_features, src_keypts, tgt_keypts)
        if testing:
            final_trans = self.post_refinement(final_trans, src_keypts, tgt_keypts)
        if not testing:
            final_labels = confidence
        res = {'final_trans': final_trans, 'final_labels': final_labels, 'M': M}
        return res

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds      
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1
        score_relation = scores.T >= scores
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()
        return torch.argsort(scores * is_local_max, dim=1, descending=True)[:, 0:max_num].detach()

    def cal_seed_trans(self, seeds, corr_features, src_keypts, tgt_keypts):
        """
        Calculate the transformation for each seeding correspondences.
        Input: 
            - seeds:         [bs, num_seeds]              the index to the seeding correspondence
            - corr_features: [bs, num_corr, num_channels]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
        Output: leading eigenvector
            - pairwise_trans:    [bs, num_seeds, 4, 4]  transformation matrix for each seeding point.
            - pairwise_fitness:  [bs, num_seeds]        fitness (inlier ratio) for each seeding point
            - final_trans:       [bs, 4, 4]             best transformation matrix (after post refinement) for each batch.
            - final_labels:      [bs, num_corr]         inlier/outlier label given by best transformation matrix.
        """
        bs, num_corr, num_channels = corr_features.shape[0], corr_features.shape[1], corr_features.shape[2]
        num_seeds = seeds.shape[-1]
        k = min(self.k, num_corr - 1)
        knn_idx = knn(corr_features, k=k, ignore_self=True, normalized=True)
        knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k))
        knn_features = corr_features.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view([bs, -1, k, num_channels])
        knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
        knn_M = torch.clamp(1 - (1 - knn_M) / self.sigma ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        feature_knn_M = knn_M
        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        knn_M = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - ((tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
        knn_M = torch.clamp(1 - knn_M ** 2 / self.sigma_spat ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        spatial_knn_M = knn_M
        total_knn_M = feature_knn_M * spatial_knn_M
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        total_weight = total_weight.view([bs, -1, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-06)
        total_weight = total_weight.view([-1, k])
        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view([bs, -1, k, 3])
        src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
        seed_as_center = False
        if seed_as_center:
            src_center = src_keypts.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, 3))
            tgt_center = tgt_keypts.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, 3))
            src_center, tgt_center = src_center.view([-1, 3]), tgt_center.view([-1, 3])
            src_pts = src_knn[:, :, :, None] - src_center[:, None, :, None]
            tgt_pts = tgt_knn[:, :, :, None] - tgt_center[:, None, :, None]
            cov = torch.einsum('nkmo,nkop->nkmp', src_pts, tgt_pts.permute(0, 1, 3, 2))
            Covariances = torch.einsum('nkmp,nk->nmp', cov, total_weight)
            U, S, Vt = torch.svd(Covariances.cpu())
            U, S, Vt = U, S, Vt
            delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
            eye = torch.eye(3)[None, :, :].repeat(U.shape[0], 1, 1)
            eye[:, -1, -1] = delta_UV
            R = Vt @ eye @ U.permute(0, 2, 1)
            t = tgt_center[:, None, :] - src_center[:, None, :] @ R.permute(0, 2, 1)
            seedwise_trans = torch.eye(4)[None, :, :].repeat(R.shape[0], 1, 1)
            seedwise_trans[:, 0:3, 0:3] = R.permute(0, 2, 1)
            seedwise_trans[:, 0:3, 3:4] = t.permute(0, 2, 1)
            seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
        else:
            seedwise_trans = rigid_transform_3d(src_knn, tgt_knn, total_weight)
            seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
        pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3], src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3, 3:4]
        pred_position = pred_position.permute(0, 1, 3, 2)
        L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)
        seedwise_fitness = torch.mean((L2_dis < self.inlier_threshold).float(), dim=-1)
        batch_best_guess = seedwise_fitness.argmax(dim=1)
        final_trans = seedwise_trans.gather(dim=1, index=batch_best_guess[:, None, None, None].expand(-1, -1, 4, 4)).squeeze(1)
        final_labels = L2_dis.gather(dim=1, index=batch_best_guess[:, None, None].expand(-1, -1, L2_dis.shape[2])).squeeze(1)
        final_labels = (final_labels < self.inlier_threshold).float()
        return seedwise_trans, seedwise_fitness, final_trans, final_labels

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input: 
            - M:      [bs, num_corr, num_corr] the compatibility matrix 
            - method: select different method for calculating the learding eigenvector.
        Output: 
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-06)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def cal_confidence(self, M, leading_eig, method='eig_value'):
        """
        Calculate the confidence of the spectral matching solution based on spectral analysis.
        Input: 
            - M:          [bs, num_corr, num_corr] the compatibility matrix 
            - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
        Output: 
            - confidence  
        """
        if method == 'eig_value':
            max_eig_value = leading_eig[:, None, :] @ M @ leading_eig[:, :, None] / (leading_eig[:, None, :] @ leading_eig[:, :, None])
            confidence = max_eig_value.squeeze(-1)
            return confidence
        elif method == 'eig_value_ratio':
            max_eig_value = leading_eig[:, None, :] @ M @ leading_eig[:, :, None] / (leading_eig[:, None, :] @ leading_eig[:, :, None])
            B = M - max_eig_value * leading_eig[:, :, None] @ leading_eig[:, None, :]
            solution = torch.ones_like(B[:, :, 0:1])
            for i in range(self.num_iterations):
                solution = torch.bmm(B, solution)
                solution = solution / (torch.norm(solution, dim=1, keepdim=True) + 1e-06)
            solution = solution.squeeze(-1)
            second_eig = solution
            second_eig_value = second_eig[:, None, :] @ B @ second_eig[:, :, None] / (second_eig[:, None, :] @ second_eig[:, :, None])
            confidence = max_eig_value / second_eig_value
            return confidence
        elif method == 'xMx':
            confidence = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
            confidence = confidence.squeeze(-1) / M.shape[1]
            return confidence

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4] 
            - src_keypts:    [bs, num_corr, 3]    
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:    
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 0.1:
            inlier_threshold_list = [0.1] * 20
        else:
            inlier_threshold_list = [1.2] * 20
        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(A=src_keypts[:, pred_inlier, :], B=tgt_keypts[:, pred_inlier, :], weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier])
        return initial_trans


class EdgeConv(nn.Module):

    def __init__(self, in_dim, out_dim, k, idx=None):
        super(EdgeConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.idx = idx
        self.conv = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        bs = x.shape[0]
        num_corr = x.shape[2]
        device = x.device
        self.idx = knn(x.permute(0, 2, 1), self.k, normalized=False)
        idx_base = torch.arange(0, bs, device=device).view(-1, 1, 1) * num_corr
        idx = self.idx + idx_base
        idx = idx.view(-1)
        x = x.transpose(2, 1).contiguous()
        features = x.view(bs * num_corr, -1)[idx, :]
        features = features.view(bs, num_corr, self.k, self.in_dim)
        x = x.view(bs, num_corr, 1, self.in_dim).repeat(1, 1, self.k, 1)
        features = torch.cat([features - x, x], dim=3).permute(0, 3, 1, 2).contiguous()
        output = self.conv(features)
        output = output.max(dim=-1, keepdim=False)[0]
        return output


class PointCN(nn.Module):

    def __init__(self, in_dim=6, num_layers=6, num_channels=128, act_pos='post'):
        super(PointCN, self).__init__()
        assert act_pos == 'pre' or act_pos == 'post'
        modules = [nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)]
        for i in range(num_layers):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        features = self.encoder(x)
        return features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContextNormalization,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeConv,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (OAFilter,
     lambda: ([], {'channels': 4, 'points': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PointCN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64])], {}),
     True),
    (SpectralMatchingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Transpose,
     lambda: ([], {'dim1': 4, 'dim2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (diff_pool,
     lambda: ([], {'in_channel': 4, 'output_points': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (diff_unpool,
     lambda: ([], {'in_channel': 4, 'output_points': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_XuyangBai_PointDSC(_paritybench_base):
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

