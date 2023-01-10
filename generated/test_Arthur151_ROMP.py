import sys
_module = sys.modules[__name__]
del sys
romp = _module
_init_paths_ = _module
base = _module
eval = _module
blender_mocap = _module
config = _module
constants = _module
AICH = _module
MuCo = _module
MuPoTS = _module
dataset = _module
agora = _module
camera_parameters = _module
cmu_panoptic_eval = _module
coco14 = _module
crowdhuman = _module
crowdpose = _module
h36m = _module
image_base = _module
image_base_relative = _module
internet = _module
lsp = _module
mixed_dataset = _module
mpi_inf_3dhp = _module
mpi_inf_3dhp_test = _module
mpi_inf_3dhp_validation = _module
mpii = _module
posetrack = _module
posetrack21 = _module
h36m_extract_frames = _module
pw3d = _module
relative_human = _module
up = _module
evaluation = _module
collect_3DPW_results = _module
collect_CRMH_3DPW_results = _module
collect_VIBE_3DPW_results = _module
crowdposetools = _module
coco = _module
cocoeval = _module
mask = _module
demo = _module
setup = _module
eval_CRMH_results = _module
eval_ds_utils = _module
eval_pckh = _module
evaluation_matrix = _module
SMPL = _module
pw3d_eval = _module
evaluate = _module
utils = _module
loss_funcs = _module
calc_loss = _module
keypoints_loss = _module
learnable_loss = _module
maps_loss = _module
params_loss = _module
prior_loss = _module
relative_loss = _module
maps_utils = _module
centermap = _module
debug_utils = _module
kp_group = _module
result_parser = _module
target_generators = _module
CoordConv = _module
models = _module
balanced_dataparallel = _module
base = _module
basic_modules = _module
bev_model = _module
build = _module
hrnet_32 = _module
resnet_50 = _module
romp_model = _module
smpl_family = _module
smpl = _module
smpl_regressor = _module
smpl_wrapper = _module
smpl_wrapper_relative = _module
smpla = _module
tracking = _module
basetrack = _module
matching = _module
tracker = _module
io = _module
kalman_filter = _module
log = _module
nms = _module
parse_config = _module
timer = _module
utils = _module
visualization = _module
augments = _module
cam_utils = _module
center_utils = _module
demo_utils = _module
projection = _module
rot_6D = _module
temporal_optimization = _module
train_utils = _module
util = _module
create_meshes = _module
open3d_visualizer = _module
renderer_pt3d = _module
renderer_pyrd = _module
socket_utils = _module
vedo_visualizer = _module
vis_client = _module
vis_server = _module
vis_server_o3d13 = _module
vis_server_py36_o3d9 = _module
vis_utils_o3d13 = _module
vis_utils_py36_o3d9 = _module
visualization = _module
web_vis = _module
predict = _module
base_predictor = _module
image = _module
video = _module
webcam = _module
pretrain = _module
test = _module
train = _module
simple_romp = _module
bev = _module
main = _module
model = _module
post_parser = _module
split2process = _module
RH_evaluation = _module
evaluation = _module
eval_AGORA = _module
eval_Relative_Human = _module
eval_cmu_panoptic = _module
main = _module
model = _module
post_parser = _module
smpl = _module
utils = _module
setup = _module
convert2fbx = _module
convert_checkpoints = _module
byte_tracker_3dcenter = _module
kalman_filter_3dcenter = _module
vis_human = _module
main = _module
pyrenderer = _module
sim3drender = _module
renderer = _module
vedo_vis = _module
vis_utils = _module

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


import time


import logging


import copy


import random


import itertools


import torch


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


import math


import matplotlib.pyplot as plt


import matplotlib as mpl


import scipy.io as scio


import torchvision


from collections import OrderedDict


from scipy.sparse import csr_matrix


import torch.nn.functional as F


from torch import nn


from torch.autograd import Variable


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


from torch.nn.modules import Module


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


import torchvision.models.resnet as resnet


import torchvision.transforms.functional as F


from collections import deque


from torchvision.ops import nms


import functools


from torch.nn import functional as F


from scipy.spatial.transform import Rotation as R


import matplotlib


import pandas


from torch.cuda.amp import autocast


from itertools import product


from time import time


import re


import numpy


from sklearn.model_selection import PredefinedSplit


DEFAULT_DTYPE = torch.float32


def _calc_matched_PCKh_(real, pred, kp2d_mask, error_thresh=0.143):
    PCKs = torch.ones(len(kp2d_mask)).float() * -1.0
    if kp2d_mask.sum() > 0:
        vis = (real > -1.0).sum(-1) == real.shape[-1]
        error = torch.norm(real - pred, p=2, dim=-1)
        for ind, (e, v) in enumerate(zip(error, vis)):
            if v.sum() < 2:
                continue
            real_valid = real[ind, v]
            person_scales = torch.sqrt((real_valid[:, 0].max(-1).values - real_valid[:, 0].min(-1).values) ** 2 + (real_valid[:, 1].max(-1).values - real_valid[:, 1].min(-1).values) ** 2)
            error_valid = e[v]
            correct_kp_mask = (error_valid / person_scales < error_thresh).float()
            PCKs[ind] = correct_kp_mask.sum() / len(correct_kp_mask)
    return PCKs


def _check_params_(params):
    assert params.shape[0] > 0, logging.error('meta_data[params] dim 0 is empty, params: {}'.format(params))
    assert params.shape[1] > 0, logging.error('meta_data[params] dim 1 is empty, params shape: {}, params: {}'.format(params.shape, params))


def batch_kp_2d_l2_loss(real, pred):
    """ 
    Directly supervise the 2D coordinates of global joints, like torso
    While supervise the relative 2D coordinates of part joints, like joints on face, feets
    """
    vis_mask = ((real > -1.99).sum(-1) == real.shape[-1]).float()
    for parent_joint, leaf_joints in constants.joint2D_tree.items():
        parent_id = constants.SMPL_ALL_54[parent_joint]
        leaf_ids = np.array([constants.SMPL_ALL_54[leaf_joint] for leaf_joint in leaf_joints])
        vis_mask[:, leaf_ids] = vis_mask[:, [parent_id]] * vis_mask[:, leaf_ids]
        real[:, leaf_ids] -= real[:, [parent_id]]
        pred[:, leaf_ids] -= pred[:, [parent_id]]
    bv_mask = torch.logical_and(vis_mask.sum(-1) > 0, (real - pred).sum(-1).sum(-1) != 0)
    vis_mask = vis_mask[bv_mask]
    loss = 0
    if vis_mask.sum() > 0:
        diff = torch.norm(real[bv_mask] - pred[bv_mask], p=2, dim=-1)
        loss = (diff * vis_mask).sum(-1) / (vis_mask.sum(-1) + 0.0001)
        if torch.isnan(loss).sum() > 0 or (loss > 1000).sum() > 0:
            return 0
            None
            non_position = torch.isnan(loss)
            None
            return 0
    return loss


def batch_l2_loss(real, predict):
    loss_batch = torch.norm(real - predict, p=2, dim=1)
    return loss_batch.mean()


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    batch_size = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def batch_smpl_pose_l2_error(real, predict):
    batch_size = real.shape[0]
    real = batch_rodrigues(real.reshape(-1, 3)).contiguous()
    predict = batch_rodrigues(predict.reshape(-1, 3)).contiguous()
    loss = torch.norm((real - predict).view(-1, 9), p=2, dim=-1)
    loss = loss.reshape(batch_size, -1).mean(-1)
    return loss


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def calc_mpjpe(real, pred, align_inds=None, sample_wise=True, trans=None, return_org=False):
    vis_mask = real[:, :, 0] != -2.0
    if align_inds is not None:
        pred_aligned = align_by_parts(pred, align_inds=align_inds)
        if trans is not None:
            pred_aligned += trans
        real_aligned = align_by_parts(real, align_inds=align_inds)
    else:
        pred_aligned, real_aligned = pred, real
    mpjpe_each = compute_mpjpe(pred_aligned, real_aligned, vis_mask, sample_wise=sample_wise)
    if return_org:
        return mpjpe_each, (real_aligned, pred_aligned, vis_mask)
    return mpjpe_each


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)
    K = X1.bmm(X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    t = mu2 - scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(mu1)
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)
    return S1_hat, (scale, R, t)


def calc_pampjpe(real, pred, sample_wise=True, return_transform_mat=False):
    real, pred = real.float(), pred.float()
    vis_mask = (real[:, :, 0] != -2.0).sum(0) == len(real)
    pred_tranformed, PA_transform = batch_compute_similarity_transform_torch(pred[:, vis_mask], real[:, vis_mask])
    pa_mpjpe_each = compute_mpjpe(pred_tranformed, real[:, vis_mask], sample_wise=sample_wise)
    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each


def focal_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = torch.zeros(gt.size(0))
    pred_log = torch.clamp(pred.clone(), min=0.001, max=1 - 0.001)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1)
    neg_loss = neg_loss.sum(-1).sum(-1)
    mask = num_pos > 0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask] + 0.0001)
    return loss.mean(-1)


def focal_loss_3D(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x z x h x w)
      gt_regr (batch x z x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = torch.zeros(gt.size(0))
    pred_log = torch.clamp(pred.clone(), min=0.001, max=1 - 0.001)
    pos_loss = torch.log(pred_log) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred_log) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum(-1).sum(-1).sum(-1)
    pos_loss = pos_loss.sum(-1).sum(-1).mean(-1)
    neg_loss = neg_loss.sum(-1).sum(-1).mean(-1)
    mask = num_pos > 0
    loss[~mask] = loss[~mask] - neg_loss[~mask]
    loss[mask] = loss[mask] - (pos_loss[mask] + neg_loss[mask]) / (num_pos[mask] + 0.0001)
    return loss.mean(-1)


def kid_offset_loss(kid_offset_preds, kid_offset_gts, matched_mask=None):
    device = kid_offset_preds.device
    kid_offset_gts = kid_offset_gts
    age_vmask = kid_offset_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum() == 0:
        return 0
    return ((kid_offset_preds[age_vmask] - kid_offset_gts[age_vmask]) ** 2).mean()


def relative_age_loss(kid_offset_preds, age_gts, matched_mask=None):
    device = kid_offset_preds.device
    age_gts = age_gts
    age_vmask = age_gts != -1
    if matched_mask is not None:
        age_vmask = age_vmask * matched_mask
    if age_vmask.sum() == 0:
        return 0
    adult_loss = (kid_offset_preds * (age_gts == 0)) ** 2
    teen_thresh = constants.age_threshold['teen']
    teen_loss = ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds > teen_thresh[2]).float() * (age_gts == 1).float()) ** 2 + ((kid_offset_preds - teen_thresh[1]) * (kid_offset_preds <= teen_thresh[0]).float() * (age_gts == 1).float()) ** 2
    kid_thresh = constants.age_threshold['kid']
    kid_loss = ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds > kid_thresh[2]).float() * (age_gts == 2).float()) ** 2 + ((kid_offset_preds - kid_thresh[1]) * (kid_offset_preds <= kid_thresh[0]).float() * (age_gts == 2).float()) ** 2
    baby_thresh = constants.age_threshold['baby']
    baby_loss = ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds > baby_thresh[2]).float() * (age_gts == 3).float()) ** 2 + ((kid_offset_preds - baby_thresh[1]) * (kid_offset_preds <= baby_thresh[0]).float() * (age_gts == 3).float()) ** 2
    age_loss = adult_loss.mean() + teen_loss.mean() + kid_loss.mean() + baby_loss.mean()
    return age_loss


def relative_depth_loss(pred_depths, depth_ids, reorganize_idx, dist_thresh=0.3, uncertainty=None, matched_mask=None):
    depth_ordering_loss = []
    depth_ids = depth_ids
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    if uncertainty is not None:
        uncertainty_valid = uncertainty[depth_ids_vmask]
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds = sample_inds * matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1, did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1, did_num))[triu_mask]
            sample_loss = []
            if args().depth_loss_type == 'Piecewise':
                eq_mask = did_mat == 0
                cd_mask = did_mat < 0
                cd_mask[did_mat < 0] = cd_mask[did_mat < 0] * (dist_mat[did_mat < 0] - did_mat[did_mat < 0] * dist_thresh) > 0
                fd_mask = did_mat > 0
                fd_mask[did_mat > 0] = fd_mask[did_mat > 0] * (dist_mat[did_mat > 0] - did_mat[did_mat > 0] * dist_thresh) < 0
                if eq_mask.sum() > 0:
                    sample_loss.append(dist_mat[eq_mask] ** 2)
                if cd_mask.sum() > 0:
                    cd_loss = torch.log(1 + torch.exp(dist_mat[cd_mask]))
                    sample_loss.append(cd_loss)
                if fd_mask.sum() > 0:
                    fd_loss = torch.log(1 + torch.exp(-dist_mat[fd_mask]))
                    sample_loss.append(fd_loss)
            elif args().depth_loss_type == 'Log':
                eq_loss = dist_mat[did_mat == 0] ** 2
                cd_loss = torch.log(1 + torch.exp(dist_mat[did_mat < 0]))
                fd_loss = torch.log(1 + torch.exp(-dist_mat[did_mat > 0]))
                sample_loss = [eq_loss, cd_loss, fd_loss]
            else:
                raise NotImplementedError
            if len(sample_loss) > 0:
                this_sample_loss = torch.cat(sample_loss).mean()
                depth_ordering_loss.append(this_sample_loss)
    if len(depth_ordering_loss) == 0:
        depth_ordering_loss = 0
    else:
        depth_ordering_loss = sum(depth_ordering_loss) / len(depth_ordering_loss)
    return depth_ordering_loss


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.gmm_prior = MaxMixturePrior(smpl_prior_path=args().smpl_prior_path, num_gaussians=8, dtype=torch.float32)
        if args().HMloss_type == 'focal':
            args().heatmap_weight /= 1000
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.joint_lossweights = torch.from_numpy(constants.SMPL54_weights).float()
        self.align_inds_MPJPE = np.array([constants.SMPL_ALL_54['L_Hip'], constants.SMPL_ALL_54['R_Hip']])
        self.shape_pca_weight = torch.Tensor([1, 0.64, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16]).unsqueeze(0).float()

    def forward(self, outputs, **kwargs):
        meta_data = outputs['meta_data']
        detect_loss_dict = self._calc_detection_loss(outputs, meta_data)
        detection_flag = outputs['detection_flag'].sum()
        loss_dict = detect_loss_dict
        kp_error = None
        if (detection_flag or args().model_return_loss) and args().calc_mesh_loss:
            mPCKh = _calc_matched_PCKh_(outputs['meta_data']['full_kp2d'].float(), outputs['pj2d'].float(), outputs['meta_data']['valid_masks'][:, 0])
            matched_mask = mPCKh > args().matching_pckh_thresh
            kp_loss_dict, kp_error = self._calc_keypoints_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **kp_loss_dict)
            params_loss_dict = self._calc_param_loss(outputs, meta_data, matched_mask)
            loss_dict = dict(loss_dict, **params_loss_dict)
        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name], tuple):
                loss_dict[name] = loss_dict[name][0]
            elif isinstance(loss_dict[name], int):
                loss_dict[name] = torch.zeros(1, device=outputs[list(outputs.keys())[0]].device)
            loss_dict[name] = loss_dict[name].mean() * eval('args().{}_weight'.format(name))
        return {'loss_dict': loss_dict, 'kp_error': kp_error}

    def _calc_detection_loss(self, outputs, meta_data):
        detect_loss_dict = {'CenterMap': 0}
        if args().calc_mesh_loss and 'center_map' in outputs:
            all_person_mask = meta_data['all_person_detected_mask']
            if all_person_mask.sum() > 0:
                detect_loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask])
        reorganize_idx_on_each_gpu = outputs['reorganize_idx'] - outputs['meta_data']['batch_ids'][0]
        if 'center_map_3d' in outputs:
            detect_loss_dict['CenterMap_3D'] = 0
            valid_mask_c3d = meta_data['valid_centermap3d_mask'].squeeze()
            valid_mask_c3d = valid_mask_c3d.reshape(-1)
            if meta_data['valid_centermap3d_mask'].sum() > 0:
                detect_loss_dict['CenterMap_3D'] = focal_loss_3D(outputs['center_map_3d'][valid_mask_c3d], meta_data['centermap_3d'][valid_mask_c3d])
        return detect_loss_dict

    def _calc_keypoints_loss(self, outputs, meta_data, matched_mask):
        kp_loss_dict, error = {'P_KP2D': 0, 'MPJPE': 0, 'PAMPJPE': 0}, {'3d': {'error': [], 'idx': []}, '2d': {'error': [], 'idx': []}}
        if 'pj2d' in outputs:
            real_2d = meta_data['full_kp2d']
            if args().model_version == 3:
                kp_loss_dict['joint_sampler'] = self.joint_sampler_loss(real_2d, outputs['joint_sampler_pred'])
            kp_loss_dict['P_KP2D'] = batch_kp_2d_l2_loss(real_2d.float().clone(), outputs['pj2d'].float().clone())
            kp3d_mask = meta_data['valid_masks'][:, 1]
        if kp3d_mask.sum() > 1 and 'j3d' in outputs:
            kp3d_gt = meta_data['kp_3d'].contiguous()
            preds_kp3d = outputs['j3d'][:, :kp3d_gt.shape[1]].contiguous()
            if not args().model_return_loss and args().PAMPJPE_weight > 0:
                try:
                    pampjpe_each = calc_pampjpe(kp3d_gt[kp3d_mask].contiguous(), preds_kp3d[kp3d_mask].contiguous())
                    kp_loss_dict['PAMPJPE'] = pampjpe_each
                except Exception as exp_error:
                    None
            if args().MPJPE_weight > 0:
                fit_mask = kp3d_mask.bool()
                if fit_mask.sum() > 0:
                    mpjpe_each = calc_mpjpe(kp3d_gt[fit_mask].contiguous(), preds_kp3d[fit_mask].contiguous(), align_inds=self.align_inds_MPJPE)
                    kp_loss_dict['MPJPE'] = mpjpe_each
                    error['3d']['error'].append(mpjpe_each.detach() * 1000)
                    error['3d']['idx'].append(torch.where(fit_mask)[0])
        return kp_loss_dict, error

    def _calc_param_loss(self, outputs, meta_data, matched_mask):
        params_loss_dict = {'Pose': 0, 'Shape': 0, 'Cam': 0, 'Prior': 0}
        if args().learn_relative:
            params_loss_dict.update({'R_Age': 0, 'R_Depth': 0})
        if 'params' in outputs:
            _check_params_(meta_data['params'])
            device = outputs['params']['body_pose'].device
            grot_masks, smpl_pose_masks, smpl_shape_masks = meta_data['valid_masks'][:, 3], meta_data['valid_masks'][:, 4], meta_data['valid_masks'][:, 5]
            if grot_masks.sum() > 0:
                params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][grot_masks, :3].contiguous(), outputs['params']['global_orient'][grot_masks].contiguous()).mean()
            if smpl_pose_masks.sum() > 0:
                params_loss_dict['Pose'] += batch_smpl_pose_l2_error(meta_data['params'][smpl_pose_masks, 3:22 * 3].contiguous(), outputs['params']['body_pose'][smpl_pose_masks, :21 * 3].contiguous()).mean()
            if smpl_shape_masks.sum() > 0:
                smpl_shape_diff = meta_data['params'][smpl_shape_masks, -10:].contiguous() - outputs['params']['betas'][smpl_shape_masks, :10].contiguous()
                params_loss_dict['Shape'] += torch.norm(smpl_shape_diff * self.shape_pca_weight, p=2, dim=-1).mean() / 20.0
            if (~smpl_shape_masks).sum() > 0:
                params_loss_dict['Shape'] += (outputs['params']['betas'][~smpl_shape_masks, :10] ** 2).mean() / 20.0
            if args().supervise_cam_params:
                cam_mask, pred_cam_params = meta_data['cam_mask'], outputs['params']['cam']
                if cam_mask.sum() > 0:
                    params_loss_dict['Cam'] += batch_l2_loss(meta_data['cams'][cam_mask], pred_cam_params[cam_mask])
            if args().learn_relative:
                if args().learn_relative_age:
                    params_loss_dict['R_Age'] = relative_age_loss(outputs['kid_offsets_pred'], meta_data['depth_info'][:, 0], matched_mask=matched_mask) + kid_offset_loss(outputs['kid_offsets_pred'], meta_data['kid_shape_offsets'], matched_mask=matched_mask) * 2
                if args().learn_relative_depth:
                    params_loss_dict['R_Depth'] = relative_depth_loss(outputs['cam_trans'][:, 2], meta_data['depth_info'][:, 3], outputs['reorganize_idx'], matched_mask=matched_mask)
            gmm_prior_loss = self.gmm_prior(outputs['params']['body_pose']).mean() / 100.0
            valuable_prior_loss_thresh = 5.0
            gmm_prior_loss[gmm_prior_loss < valuable_prior_loss_thresh] = 0
            params_loss_dict['Prior'] = gmm_prior_loss
        return params_loss_dict

    def joint_sampler_loss(self, real_2d, joint_sampler):
        batch_size = joint_sampler.shape[0]
        joint_sampler = joint_sampler.view(batch_size, -1, 2)
        joint_gt = real_2d[:, constants.joint_sampler_mapper]
        loss = batch_kp_2d_l2_loss(joint_gt, joint_sampler)
        return loss


class Learnable_Loss(nn.Module):
    """docstring for  Learnable_Loss"""

    def __init__(self, ID_num=0):
        super(Learnable_Loss, self).__init__()
        self.loss_class = {'det': ['CenterMap', 'CenterMap_3D'], 'reg': ['MPJPE', 'PAMPJPE', 'P_KP2D', 'Pose', 'Shape', 'Cam', 'Prior']}
        self.all_loss_names = np.concatenate([loss_list for task_name, loss_list in self.loss_class.items()]).tolist()
        if args().learn_2dpose:
            self.loss_class['reg'].append('heatmap')
        if args().learn_AE:
            self.loss_class['reg'].append('AE')
        if args().learn_relative:
            self.loss_class['rel'] = ['R_Age', 'R_Gender', 'R_Weight', 'R_Depth', 'R_Depth_scale']

    def forward(self, outputs, new_training=False):
        loss_dict = outputs['loss_dict']
        if args().model_return_loss and args().calc_mesh_loss and not new_training:
            if args().PAMPJPE_weight > 0 and outputs['detection_flag'].sum() > 0:
                try:
                    kp3d_mask = outputs['meta_data']['valid_masks'][:, 1]
                    kp3d_gt = outputs['meta_data']['kp_3d'][kp3d_mask].contiguous()
                    preds_kp3d = outputs['j3d'][kp3d_mask, :kp3d_gt.shape[1]].contiguous()
                    if len(preds_kp3d) > 0:
                        loss_dict['PAMPJPE'] = calc_pampjpe(kp3d_gt.contiguous().float(), preds_kp3d.contiguous().float()).mean() * args().PAMPJPE_weight
                except Exception as exp_error:
                    None
        loss_dict = {key: value.mean() for key, value in loss_dict.items() if not isinstance(value, int)}
        if new_training and args().model_version == 6:
            loss_dict['CenterMap_3D'] = loss_dict['CenterMap_3D'] / 1000.0
            loss_dict = {key: loss_dict[key] for key in self.loss_class['det']}
        loss_list = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if not torch.isnan(value):
                    if value.item() < args().loss_thresh:
                        loss_list.append(value)
                    else:
                        loss_list.append(value / (value.item() / args().loss_thresh))
        loss = sum(loss_list)
        loss_tasks = {}
        for loss_class in self.loss_class:
            loss_tasks[loss_class] = sum([loss_dict[item] for item in self.loss_class[loss_class] if item in loss_dict])
        left_loss = sum([loss_dict[loss_item] for loss_item in loss_dict if loss_item not in self.all_loss_names])
        if left_loss != 0:
            loss_tasks.update({'Others': left_loss})
        outputs['loss_dict'] = dict(loss_tasks, **loss_dict)
        return loss, outputs


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp
    return inp


class AELoss(nn.Module):

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp)) ** 2)
        num_tags = len(tags)
        if num_tags == 0:
            return make_input(torch.zeros(1).float()), make_input(torch.zeros(1).float())
        elif num_tags == 1:
            return make_input(torch.zeros(1).float()), pull / num_tags
        tags = torch.stack(tags)
        size = num_tags, num_tags
        A = tags.expand(*size)
        B = A.permute(1, 0)
        diff = A - B
        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')
        return push / ((num_tags - 1) * num_tags) * 0.5, pull / num_tags

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class Heatmap_AE_loss(nn.Module):

    def __init__(self, num_joints, loss_type_HM='MSE', loss_type_AE='exp'):
        super().__init__()
        self.num_joints = num_joints
        self.heatmaps_loss = HeatmapLoss(loss_type_HM)
        self.heatmaps_loss_factor = 1.0
        self.ae_loss = AELoss(loss_type_AE)
        self.push_loss_factor = 1.0
        self.pull_loss_factor = 1.0

    def forward(self, outputs, heatmaps, joints):
        heatmaps_pred = outputs[:, :self.num_joints]
        tags_pred = outputs[:, self.num_joints:]
        heatmaps_loss = None
        push_loss = None
        pull_loss = None
        if self.heatmaps_loss is not None:
            heatmaps_loss = self.heatmaps_loss(heatmaps_pred, heatmaps)
            heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor
        if self.ae_loss is not None:
            batch_size = tags_pred.size()[0]
            tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
            push_loss, pull_loss = self.ae_loss(tags_pred, joints)
            push_loss = push_loss * self.push_loss_factor
            pull_loss = pull_loss * self.pull_loss_factor
        return heatmaps_loss, push_loss, pull_loss


class Interperlation_penalty(nn.Module):

    def __init__(self, faces_tensor, df_cone_height=0.5, point2plane=False, penalize_outside=True, max_collisions=8, part_segm_fn=None):
        super(Interperlation_penalty, self).__init__()
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=df_cone_height, point2plane=point2plane, vectorized=True, penalize_outside=penalize_outside)
        self.coll_loss_weight = 1.0
        self.search_tree = BVH(max_collisions=max_collisions)
        self.body_model_faces = faces_tensor
        if part_segm_fn:
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file, encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            self.tri_filtering_module = FilterFaces(faces_segm=faces_segm, faces_parents=faces_parents)

    def forward(self, vertices):
        pen_loss = 0.0
        batch_size = vertices.shape[0]
        triangles = torch.index_select(vertices, 1, self.body_model_faces).view(batch_size, -1, 3, 3)
        with torch.no_grad():
            collision_idxs = self.search_tree(triangles)
        if self.tri_filtering_module is not None:
            collision_idxs = self.tri_filtering_module(collision_idxs)
        if collision_idxs.ge(0).sum().item() > 0:
            pen_loss = torch.sum(self.coll_loss_weight * self.pen_distance(triangles, collision_idxs))
        return pen_loss


class SMPLifyAnglePrior(nn.Module):

    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32 if dtype == torch.float32 else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        """ Returns the angle prior loss for the given pose
        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        """
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):

    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MultiLossFactory(nn.Module):

    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.num_stages = 1
        self.heatmaps_loss = nn.ModuleList([(HeatmapLoss() if with_heatmaps_loss else None) for with_heatmaps_loss in [True]])
        self.heatmaps_loss_factor = [1.0]
        self.ae_loss = nn.ModuleList([(AELoss('exp') if with_ae_loss else None) for with_ae_loss in [True]])
        self.push_loss_factor = [0.001]
        self.pull_loss_factor = [0.001]

    def forward(self, outputs, heatmaps, masks, joints):
        self._forward_check(outputs, heatmaps, masks, joints)
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred, heatmaps[idx], masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)
            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)
                push_loss, pull_loss = self.ae_loss[idx](tags_pred, joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]
                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)
        return heatmaps_losses, push_losses, pull_losses

    def _forward_check(self, outputs, heatmaps, masks, joints):
        assert isinstance(outputs, list), 'outputs should be a list, got {} instead.'.format(type(outputs))
        assert isinstance(heatmaps, list), 'heatmaps should be a list, got {} instead.'.format(type(heatmaps))
        assert isinstance(masks, list), 'masks should be a list, got {} instead.'.format(type(masks))
        assert isinstance(joints, list), 'joints should be a list, got {} instead.'.format(type(joints))
        assert len(outputs) == self.num_stages, 'len(outputs) and num_stages should been same, got {} vs {}.'.format(len(outputs), self.num_stages)
        assert len(outputs) == len(heatmaps), 'outputs and heatmaps should have same length, got {} vs {}.'.format(len(outputs), len(heatmaps))
        assert len(outputs) == len(masks), 'outputs and masks should have same length, got {} vs {}.'.format(len(outputs), len(masks))
        assert len(outputs) == len(joints), 'outputs and joints should have same length, got {} vs {}.'.format(len(outputs), len(joints))
        assert len(outputs) == len(self.heatmaps_loss), 'outputs and heatmaps_loss should have same length, got {} vs {}.'.format(len(outputs), len(self.heatmaps_loss))
        assert len(outputs) == len(self.ae_loss), 'outputs and ae_loss should have same length, got {} vs {}.'.format(len(outputs), len(self.ae_loss))


def gather_feature(fmap, index, mask=None):
    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CenterMap(object):

    def __init__(self, conf_thresh):
        self.size = 64
        self.max_person = 64
        self.sigma = 1
        self.conf_thresh = conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels([5])

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size - 1) // 2, (kernel_size - 1) // 2
            gaussian_distribution = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size - 1) // 2)
        return gk_group, pool_group

    def parse_centermap(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[5])
        b, c, h, w = center_map_nms.shape
        K = self.max_person
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds.long() // w).float()
        topk_xs = (topk_inds % w).int().float()
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_clses = index.long() // K
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)
        mask = topk_score > self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1, 0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]


class Params(object):

    def __init__(self):
        self.num_joints = 17
        self.max_num_people = 5
        self.detection_threshold = 0.1
        self.tag_threshold = 1.0
        self.use_detection_val = True
        self.ignore_too_much = True


def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp


def match_by_tag(inp, params):
    assert isinstance(params, Params), 'params should be class Params()'
    tag_k, loc_k, val_k = inp
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]))
    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = i
        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]
        if joints.shape[0] == 0:
            continue
        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]
            if params.ignore_too_much and len(grouped_keys) == params.max_num_people:
                continue
            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)
            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]
            num_added = diff.shape[0]
            num_grouped = diff.shape[1]
            if num_added > num_grouped:
                diff_normed = np.concatenate((diff_normed, np.zeros((num_added, num_added - num_grouped)) + 10000000000.0), axis=1)
            pairs = py_max_match(diff_normed)
            for row, col in pairs:
                if row < num_added and col < num_grouped and diff_saved[row][col] < params.tag_threshold:
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = joints[row]
                    tag_dict[key] = [tags[row]]
    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans


class HeatmapParser(object):

    def __init__(self):
        self.params = Params()
        self.tag_per_joint = True
        NMS_KERNEL, NMS_PADDING = 5, 2
        self.map_size = 128
        self.pool = torch.nn.MaxPool2d(NMS_KERNEL, 1, NMS_PADDING)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: match_by_tag(x, self.params)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def top_k(self, det, tag):
        det = self.nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.params.max_num_people, dim=2)
        tag = tag.view(tag.size(0), tag.size(1), w * h, -1)
        if not self.tag_per_joint:
            tag = tag.expand(-1, self.params.num_joints, -1, -1)
        tag_k = torch.stack([torch.gather(tag[:, :, :, i], 2, ind) for i in range(tag.size(3))], dim=3)
        x = ind % w
        y = (ind / float(w)).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'tag_k': tag_k.cpu().numpy(), 'loc_k': ind_k.cpu().numpy(), 'val_k': val_k.cpu().numpy()}
        return ans

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25
                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = y + 0.5, x + 0.5
        return ans

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return: 
        """
        if len(tag.shape) == 3:
            tag = tag[:, :, :, None]
        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])
        prev_tag = np.mean(tags, axis=0)
        ans = []
        for i in range(keypoints.shape[0]):
            tmp = det[i, :, :]
            tt = ((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5
            tmp2 = tmp - np.round(tt)
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            val = tmp[y, x]
            x += 0.5
            y += 0.5
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25
            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25
            ans.append((x, y, val))
        ans = np.array(ans)
        if ans is not None:
            for i in range(det.shape[0]):
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]
        return keypoints

    def parse(self, det, tag, adjust=True, refine=True, get_best=False):
        ans = self.match(**self.top_k(det, tag))
        if adjust:
            ans = self.adjust(ans, det)
        scores = [i[:, 2].mean() for i in ans[0]]
        if refine:
            ans = ans[0]
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy()
                tag_numpy = tag[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy, (self.params.num_joints, 1, 1, 1))
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
            ans = [ans]
        if len(scores) > 0:
            kp2ds = np.array(ans[0][:, :, :2])
            kp2ds = 2 * kp2ds / float(self.map_size) - 1
            return kp2ds, scores
        else:
            return np.zeros((1, self.params.num_joints, 2)), [0]

    def batch_parse(self, dets_tags, **kwargs):
        dets, tags = dets_tags[:, :self.params.num_joints], dets_tags[:, self.params.num_joints:]
        results, scores = [], []
        for det, tag in zip(dets, tags):
            kp2ds, each_scores = self.parse(det.unsqueeze(0), tag.unsqueeze(0), **kwargs)
            results.append(kp2ds)
            scores.append(each_scores)
        return results, scores


class VertexJointSelector(nn.Module):

    def __init__(self, extra_joints_idxs, J_regressor_extra9, J_regressor_h36m17, dtype=torch.float32):
        super(VertexJointSelector, self).__init__()
        self.register_buffer('extra_joints_idxs', extra_joints_idxs)
        self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)

    def forward(self, vertices, joints):
        extra_joints21 = torch.index_select(vertices, 1, self.extra_joints_idxs)
        extra_joints9 = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_extra9])
        joints_h36m17 = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor_h36m17])
        joints54_17 = torch.cat([joints, extra_joints21, extra_joints9, joints_h36m17], dim=1)
        return joints54_17


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])
    return posed_joints, rel_transforms


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, dtype=torch.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """
    batch_size = betas.shape[0]
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    J = torch.einsum('bik,ji->bjk', [v_shaped, J_regressor])
    dtype = pose.dtype
    posedirs = posedirs.type(dtype)
    ident = torch.eye(3, dtype=dtype, device=J_regressor.device)
    rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3]).type(dtype)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).type(dtype)
    pose_offsets = torch.matmul(pose_feature, posedirs.type(dtype)).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=J_regressor.device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


class SMPL(nn.Module):

    def __init__(self, model_path, model_type='smpl', dtype=torch.float32):
        super(SMPL, self).__init__()
        self.dtype = dtype
        model_info = torch.load(model_path)
        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'], model_info['J_regressor_extra9'], model_info['J_regressor_h36m17'], dtype=self.dtype)
        self.register_buffer('faces_tensor', model_info['f'])
        self.register_buffer('v_template', model_info['v_template'])
        if model_type == 'smpl':
            self.register_buffer('shapedirs', model_info['shapedirs'])
        elif model_type == 'smpla':
            self.register_buffer('shapedirs', model_info['smpla_shapedirs'])
        self.register_buffer('J_regressor', model_info['J_regressor'])
        self.register_buffer('posedirs', model_info['posedirs'])
        self.register_buffer('parents', model_info['kintree_table'])
        self.register_buffer('lbs_weights', model_info['weights'])

    def forward(self, betas=None, poses=None, root_align=False):
        """ Forward pass for the SMPL model
            Parameters
            ----------
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 54 joints of body meshes, (B x 54 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        """
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas).type(self.dtype)
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).type(self.dtype)
        default_device = self.shapedirs.device
        betas, poses = betas, poses
        vertices, joints = lbs(betas, poses, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        joints54 = self.vertex_joint_selector(vertices, joints)
        if root_align:
            root_trans = joints54[:, [45, 46]].mean(1).unsqueeze(1)
            joints54 = joints54 - root_trans
            vertices = vertices - root_trans
        return vertices, joints54, self.faces_tensor


class SMPLA_parser(nn.Module):

    def __init__(self, smpla_path, smil_path):
        super(SMPLA_parser, self).__init__()
        self.smil_model = SMPL(smil_path, model_type='smpl')
        self.smpl_model = SMPL(smpla_path, model_type='smpla')
        self.baby_thresh = 0.8

    def forward(self, betas=None, thetas=None, root_align=True):
        baby_mask = betas[:, 10] > self.baby_thresh
        if baby_mask.sum() > 0:
            adult_mask = ~baby_mask
            person_num = len(thetas)
            verts, joints = torch.zeros(person_num, 6890, 3, device=thetas.device).float(), torch.zeros(person_num, 54 + 17, 3, device=thetas.device).float()
            verts[baby_mask], joints[baby_mask], face = self.smil_model(betas[baby_mask, :10], thetas[baby_mask])
            if adult_mask.sum() > 0:
                verts[adult_mask], joints[adult_mask], face = self.smpl_model(betas[adult_mask], thetas[adult_mask])
        else:
            verts, joints, face = self.smpl_model(betas, thetas)
        if root_align:
            root_trans = joints[:, [45, 46]].mean(1).unsqueeze(1)
            joints = joints - root_trans
            verts = verts - root_trans
        return verts, joints, face


def parse_age_cls_results(age_probs):
    age_preds = torch.ones_like(age_probs).long() * -1
    age_preds[(age_probs <= constants.age_threshold['adult'][2]) & (age_probs > constants.age_threshold['adult'][0])] = 0
    age_preds[(age_probs <= constants.age_threshold['teen'][2]) & (age_probs > constants.age_threshold['teen'][0])] = 1
    age_preds[(age_probs <= constants.age_threshold['kid'][2]) & (age_probs > constants.age_threshold['kid'][0])] = 2
    age_preds[(age_probs <= constants.age_threshold['baby'][2]) & (age_probs > constants.age_threshold['baby'][0])] = 3
    return age_preds


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-06)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-06)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)
    return rot_mats


def quaternion_to_angle_axis(quaternion: torch.Tensor) ->torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(quaternion)))
    if not quaternion.shape[-1] == 4:
        raise ValueError('Input must be a tensor of shape Nx4 or 4. Got {}'.format(quaternion.shape))
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))
    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(rotation_matrix)))
    if len(rotation_matrix.shape) > 3:
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = torch.transpose(rotation_matrix, 1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)
    q *= 0.5
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 3) 
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float()
    img_pad_size, crop_trbl, pad_trbl = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10]
    leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
    kp2ds_on_orgimg = (kp2ds + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    return kp2ds_on_orgimg


INVALID_TRANS = np.ones(3) * -1


def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.0, 512.0]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0, 0], camK[1, 1] = focal_length, focal_length
        camK[:2, 2] = img_size // 2
    else:
        camK = proj_mat
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist, flags=cv2.SOLVEPNP_EPNP, reprojectionError=20, iterationsCount=100)
    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:, 0]
        return tra_pred


def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512.0, 512.0]), proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        f = np.array([focal_length, focal_length])
        center = img_size / 2.0
    else:
        f = np.array([proj_mat[0, 0], proj_mat[1, 1]])
        center = proj_mat[:2, 2]
    Z = np.reshape(np.tile(joints_3d[:, 2], (2, 1)).T, -1)
    XY = np.reshape(joints_3d[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints), O - np.reshape(joints_2d, -1)]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)
    trans = np.linalg.solve(A, b)
    return trans


def estimate_translation(joints_3d, joints_2d, pts_mnum=4, focal_length=600, proj_mats=None, cam_dists=None, img_size=np.array([512.0, 512.0])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()
    if joints_2d.shape[-1] == 2:
        joints_conf = joints_2d[:, :, -1] > -2.0
    elif joints_2d.shape[-1] == 3:
        joints_conf = joints_2d[:, :, -1] > 0
    joints3d_conf = joints_3d[:, :, -1] != -2.0
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i, :, :2]
        valid_mask = joints_conf[i] * joints3d_conf[i]
        if valid_mask.sum() < pts_mnum:
            trans[i] = INVALID_TRANS
            continue
        if len(img_size.shape) == 1:
            imgsize = img_size
        elif len(img_size.shape) == 2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask], focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i], cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask], valid_mask[valid_mask].astype(np.float32), focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])
    return torch.from_numpy(trans).float()


def vertices_kp3d_projection(outputs, meta_data=None, presp=False):
    vertices, j3ds = outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, outputs['cam'], mode='3d', keep_dim=True)
    pj3d = batch_orth_proj(j3ds, outputs['cam'], mode='2d')
    predicts_j3ds = j3ds[:, :24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:, :, :2][:, :24].detach().cpu().numpy() + 1) * 256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, focal_length=443.4, img_size=np.array([512, 512]))
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:, :, :2], 'cam_trans': cam_trans}
    if meta_data is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'], meta_data['offsets'])
    return projected_outputs


class SMPLWrapper(nn.Module):

    def __init__(self):
        super(SMPLWrapper, self).__init__()
        logging.info('Building SMPL family for relative learning!!')
        self.smpl_model = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args().cam_dim, args().rot_dim, (args().smpl_joint_num - 1) * args().rot_dim, 11]
        self.params_num = np.array(self.part_idx).sum()
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)
        self.joint_mapper_op25 = torch.from_numpy(constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)).long()

    def forward(self, outputs, meta_data):
        params_dict = self.pack_params_dict(outputs['params_pred'])
        params_dict['betas'], cls_dict = self.process_betas(params_dict['betas'])
        vertices, joints54_17 = self.smpl_model(betas=params_dict['betas'], poses=params_dict['poses'])
        outputs.update({'params': params_dict, 'verts': vertices, 'j3d': joints54_17[:, :54], 'joints_h36m17': joints54_17[:, 54:], **cls_dict})
        outputs.update(vertices_kp3d_projection(outputs['j3d'], outputs['params']['cam'], joints_h36m17_preds=outputs['joints_h36m17'], input2orgimg_offsets=meta_data['offsets'], presp=args().perspective_proj, vertices=outputs['verts']))
        return outputs

    def add_template_mesh_pose(self, params):
        template_mesh = self.template_mesh.repeat(len(params['poses']), 1, 1)
        template_joint = self.template_joint.repeat(len(params['poses']), 1, 1)
        return {'verts': template_mesh, 'j3d': template_joint, 'joints_smpl24': template_joint}

    def pack_params_dict(self, params_pred):
        idx_list, params_dict = [0], {}
        for i, (idx, name) in enumerate(zip(self.part_idx, self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = params_pred[:, idx_list[i]:idx_list[i + 1]].contiguous()
        if args().Rot_type == '6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N, 6)], 1)
        params_dict['poses'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)
        return params_dict

    def process_betas(self, betas_pred):
        smpl_betas = betas_pred[:, :10]
        kid_offsets = betas_pred[:, 10]
        Age_preds = parse_age_cls_results(kid_offsets)
        cls_dict = {'Age_preds': Age_preds, 'kid_offsets_pred': kid_offsets}
        return betas_pred, cls_dict


def flatten_inds(coords):
    coords = torch.clamp(coords, 0, args().centermap_size - 1)
    return coords[:, 0].long() * args().centermap_size + coords[:, 1].long()


def convert_cam_params_to_centermap_coords(cam_params, cam3dmap_anchor):
    center_coords = torch.ones_like(cam_params)
    center_coords[:, 1:] = cam_params[:, 1:].clone()
    cam3dmap_anchors = cam3dmap_anchor[None]
    scale_num = len(cam3dmap_anchor)
    if len(cam_params) != 0:
        center_coords[:, 0] = torch.argmin(torch.abs(cam_params[:, [0]].repeat(1, scale_num) - cam3dmap_anchors), dim=1).float() / 128 * 2.0 - 1.0
    return center_coords


def denormalize_center(center, size=128):
    center = (center + 1) / 2 * size
    center = torch.clamp(center, 1, size - 1).long()
    return center


def process_gt_center(center_normed):
    valid_mask = center_normed[:, :, 0] > -1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    center_gt = ((center_normed + 1) / 2 * args().centermap_size).long()
    center_gt_valid = center_gt[valid_mask]
    return valid_batch_inds, valid_person_ids, center_gt_valid


def reorganize_gts(meta_data, key_list, batch_ids):
    for key in key_list:
        if key in meta_data:
            if isinstance(meta_data[key], torch.Tensor):
                meta_data[key] = meta_data[key][batch_ids]
            elif isinstance(meta_data[key], list):
                meta_data[key] = [meta_data[key][ind] for ind in batch_ids]
    return meta_data


def reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
    exclude_keys += gt_keys
    outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
    info_vis = []
    for key, item in meta_data.items():
        if key not in exclude_keys:
            info_vis.append(key)
    meta_data = reorganize_gts(meta_data, info_vis, batch_ids)
    for gt_key in gt_keys:
        if gt_key in meta_data:
            try:
                meta_data[gt_key] = meta_data[gt_key][batch_ids, person_ids]
            except Exception as error:
                None
    return outputs, meta_data


class ResultParser(nn.Module):

    def __init__(self, with_smpl_parser=True):
        super(ResultParser, self).__init__()
        self.map_size = args().centermap_size
        self.with_smpl_parser = with_smpl_parser
        if args().calc_smpl_mesh and with_smpl_parser:
            self.params_map_parser = SMPLWrapper()
        self.heatmap_parser = HeatmapParser()
        self.centermap_parser = CenterMap()
        self.match_preds_to_gts_for_supervision = args().match_preds_to_gts_for_supervision

    def matching_forward(self, outputs, meta_data, cfg):
        if args().model_version in [6, 8, 9]:
            outputs, meta_data = self.match_params_new(outputs, meta_data, cfg)
        else:
            outputs, meta_data = self.match_params(outputs, meta_data, cfg)
        if 'params_pred' in outputs and self.with_smpl_parser and args().calc_smpl_mesh:
            outputs = self.params_map_parser(outputs, meta_data)
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        return outputs, meta_data

    @torch.no_grad()
    def parsing_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        if 'params_pred' in outputs and self.with_smpl_parser:
            outputs = self.params_map_parser(outputs, meta_data)
        if 'detection_flag' not in outputs:
            outputs['detection_flag'] = self.determine_detection_flag(outputs, meta_data)
        return outputs, meta_data

    def determine_detection_flag(self, outputs, meta_data):
        detected_ids = torch.unique(outputs['reorganize_idx'])
        detection_flag = torch.Tensor([(batch_id in detected_ids) for batch_id in meta_data['batch_ids']])
        return detection_flag

    def process_reorganize_idx_data_parallel(self, outputs):
        gpu_num = torch.cuda.device_count()
        current_device_id = outputs['params_maps'].device.index
        data_size = outputs['params_maps'].shape[0]
        outputs['reorganize_idx'] += data_size * current_device_id
        return outputs

    def suppressing_silimar_mesh_and_2D_center(self, params_preds, pred_batch_ids, pred_czyxs, top_score, center2D_thresh=5, pose_thresh=2.5):
        pose_params_preds = params_preds[:, args().cam_dim:args().cam_dim + 22 * args().rot_dim]
        N = len(pred_czyxs)
        center2D_similarity = torch.norm((pred_czyxs[:, 1:].unsqueeze(1).repeat(1, N, 1) - pred_czyxs[:, 1:].unsqueeze(0).repeat(N, 1, 1)).float(), p=2, dim=-1)
        same_batch_id_mask = pred_batch_ids.unsqueeze(1).repeat(1, N) == pred_batch_ids.unsqueeze(0).repeat(N, 1)
        center2D_similarity[~same_batch_id_mask] = center2D_thresh + 1
        similarity = center2D_similarity <= center2D_thresh
        center_similar_inds = torch.where(similarity.sum(-1) > 1)[0]
        for s_inds in center_similar_inds:
            pose_angulars = rot6D_to_angular(pose_params_preds[similarity[s_inds]])
            pose_angular_base = rot6D_to_angular(pose_params_preds[s_inds].unsqueeze(0)).repeat(len(pose_angulars), 1)
            pose_similarity = batch_smpl_pose_l2_error(pose_angulars, pose_angular_base)
            sim_past = similarity[s_inds].clone()
            similarity[s_inds, sim_past] = pose_similarity < pose_thresh
        score_map = similarity * top_score.unsqueeze(0).repeat(N, 1)
        nms_inds = torch.argmax(score_map, 1) == torch.arange(N)
        return [item[nms_inds] for item in [pred_batch_ids, pred_czyxs, top_score]], nms_inds

    def suppressing_duplicate_mesh(self, outputs):
        (pred_batch_ids, pred_czyxs, top_score), nms_inds = self.suppressing_silimar_mesh_and_2D_center(outputs['params_pred'], outputs['pred_batch_ids'], outputs['pred_czyxs'], outputs['top_score'])
        outputs['params_pred'], outputs['cam_czyx'] = outputs['params_pred'][nms_inds], outputs['cam_czyx'][nms_inds]
        if 'motion_offsets' in outputs:
            outputs['motion_offsets'] = outputs['motion_offsets'][nms_inds]
        outputs.update({'pred_batch_ids': pred_batch_ids, 'pred_czyxs': pred_czyxs, 'top_score': top_score})
        return outputs

    def match_params_new(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'valid_masks', 'subject_ids', 'verts', 'cam_mask', 'kid_shape_offsets', 'root_trans', 'cams']
        if args().learn_relative:
            gt_keys += ['depth_info']
        exclude_keys = ['heatmap', 'centermap', 'AE_joints', 'person_centers', 'params_pred', 'all_person_detected_mask', 'person_scales']
        if cfg['with_nms']:
            outputs = self.suppressing_duplicate_mesh(outputs)
        cam_mask = meta_data['cam_mask']
        center_gts_info_3d = parse_gt_center3d(cam_mask, meta_data['cams'])
        person_centers = meta_data['person_centers'].clone()
        person_centers[cam_mask] = -2.0
        center_gts_info_2d = process_gt_center(person_centers)
        mc = self.match_gt_pred_3d_2d(center_gts_info_2d, center_gts_info_3d, outputs['pred_batch_ids'], outputs['pred_czyxs'], outputs['top_score'], outputs['cam_czyx'], outputs['center_map_3d'].device, cfg['is_training'], batch_size=len(cam_mask), with_2d_matching=cfg['with_2d_matching'])
        batch_ids, person_ids, matched_pred_ids, center_confs = mc['batch_ids'], mc['person_ids'], mc['matched_ids'], mc['conf']
        outputs['params_pred'] = outputs['params_pred'][matched_pred_ids]
        for center_key in ['pred_batch_ids', 'pred_czyxs', 'top_score']:
            outputs[center_key] = outputs[center_key][matched_pred_ids]
        outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        exclude_keys += ['centermap_3d', 'valid_centermap3d_mask']
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
        outputs['center_confs'] = center_confs
        return outputs, meta_data

    def match_gt_pred_3d_2d(self, center_gts_info_2d, center_gts_info_3d, pred_batch_ids, pred_czyxs, top_score, cam_czyx, device, is_training, batch_size=1, with_2d_matching=True):
        vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info_2d
        vgt_batch_ids_3d, vgt_person_ids_3d, vgt_czyxs = center_gts_info_3d
        mc = {key: [] for key in ['batch_ids', 'matched_ids', 'person_ids', 'conf']}
        for match_ind in torch.arange(len(vgt_batch_ids_3d)):
            batch_id, person_id, center_gt = vgt_batch_ids_3d[match_ind], vgt_person_ids_3d[match_ind], vgt_czyxs[match_ind]
            pids = torch.where(pred_batch_ids == batch_id)[0]
            if len(pids) == 0:
                continue
            center_dist_3d = torch.norm(pred_czyxs[pids].float() - center_gt[None].float(), dim=-1)
            matched_pred_id = pids[torch.argmin(center_dist_3d)]
            mc['batch_ids'].append(batch_id)
            mc['matched_ids'].append(matched_pred_id)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[matched_pred_id])
        for match_ind in torch.arange(len(vgt_batch_ids)):
            batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
            pids = torch.where(pred_batch_ids == batch_id)[0]
            if len(pids) == 0:
                continue
            matched_pred_id = pids[torch.argmin(torch.norm(pred_czyxs[pids, 1:].float() - center_gt[None].float(), dim=-1))]
            center_matched = pred_czyxs[matched_pred_id].long()
            mc['batch_ids'].append(batch_id)
            mc['matched_ids'].append(matched_pred_id)
            mc['person_ids'].append(person_id)
            mc['conf'].append(top_score[matched_pred_id])
        if args().eval_2dpose:
            for inds, (batch_id, person_id, center_gt) in enumerate(zip(vgt_batch_ids, vgt_person_ids, vgt_centers)):
                if batch_id in pred_batch_ids:
                    center_pred = pred_czyxs[pred_batch_ids == batch_id]
                    matched_id = torch.argmin(torch.norm(center_pred[:, 1:].float() - center_gt[None].float(), dim=-1))
                    matched_pred_id = np.where((pred_batch_ids == batch_id).cpu())[0][matched_id]
                    mc['matched_ids'].append(matched_pred_id)
                    mc['batch_ids'].append(batch_id)
                    mc['person_ids'].append(person_id)
        if len(mc['matched_ids']) == 0:
            mc.update({'batch_ids': [0], 'matched_ids': [0], 'person_ids': [0], 'conf': [0]})
        keys_list = list(mc.keys())
        for key in keys_list:
            if key == 'conf':
                mc[key] = torch.Tensor(mc[key])
            else:
                mc[key] = torch.Tensor(mc[key]).long()
            if args().max_supervise_num != -1 and is_training:
                mc[key] = mc[key][:args().max_supervise_num]
        return mc

    def match_params(self, outputs, meta_data, cfg):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'subject_ids', 'valid_masks']
        exclude_keys = ['heatmap', 'centermap', 'AE_joints', 'person_centers', 'all_person_detected_mask']
        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = self.match_gt_pred(center_gts_info, center_preds_info, outputs['center_map'].device, cfg['is_training'])
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        if len(batch_ids) == 0:
            if 'new_training' in cfg:
                if cfg['new_training']:
                    outputs['detection_flag'] = torch.Tensor([(False) for _ in range(len(meta_data['batch_ids']))])
                    outputs['reorganize_idx'] = meta_data['batch_ids']
                    return outputs, meta_data
            batch_ids, flat_inds = torch.zeros(1).long(), (torch.ones(1) * self.map_size ** 2 / 2.0).long()
            person_ids = batch_ids.clone()
        outputs['detection_flag'] = torch.Tensor([(True) for _ in range(len(batch_ids))])
        if 'params_maps' in outputs and 'params_pred' not in outputs:
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs, meta_data = reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['centers_pred'] = torch.stack([flat_inds % args().centermap_size, flat_inds // args().centermap_size], 1)
        return outputs, meta_data

    def match_gt_pred(self, center_gts_info, center_preds_info, device, is_training):
        vgt_batch_ids, vgt_person_ids, vgt_centers = center_gts_info
        vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
        mc = {key: [] for key in ['batch_ids', 'flat_inds', 'person_ids', 'conf']}
        if self.match_preds_to_gts_for_supervision:
            for match_ind in torch.arange(len(vgt_batch_ids)):
                batch_id, person_id, center_gt = vgt_batch_ids[match_ind], vgt_person_ids[match_ind], vgt_centers[match_ind]
                pids = torch.where(vpred_batch_ids == batch_id)[0]
                if len(pids) == 0:
                    continue
                closet_center_ind = pids[torch.argmin(torch.norm(cyxs[pids].float() - center_gt[None].float(), dim=-1))]
                center_matched = cyxs[closet_center_ind].long()
                cy, cx = torch.clamp(center_matched, 0, self.map_size - 1)
                flat_ind = cy * args().centermap_size + cx
                mc['batch_ids'].append(batch_id)
                mc['flat_inds'].append(flat_ind)
                mc['person_ids'].append(person_id)
                mc['conf'].append(top_score[closet_center_ind])
            keys_list = list(mc.keys())
            for key in keys_list:
                if key != 'conf':
                    mc[key] = torch.Tensor(mc[key]).long()
                if args().max_supervise_num != -1 and is_training:
                    mc[key] = mc[key][:args().max_supervise_num]
        else:
            mc['batch_ids'] = vgt_batch_ids.long()
            mc['flat_inds'] = flatten_inds(vgt_centers.long())
            mc['person_ids'] = vgt_person_ids.long()
            mc['conf'] = torch.zeros(len(vgt_person_ids))
        return mc

    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids, flat_inds].contiguous()
        return results

    @torch.no_grad()
    def parse_maps(self, outputs, meta_data, cfg):
        if args().model_version in [6]:
            if cfg['with_nms']:
                outputs = self.suppressing_duplicate_mesh(outputs)
            batch_ids = outputs['pred_batch_ids'].long()
            outputs['center_preds'] = outputs['pred_czyxs'] * args().input_size / args().centermap_size
            outputs['center_confs'] = outputs['top_score']
        else:
            batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'])
            if len(batch_ids) == 0:
                batch_ids, flat_inds, cyxs, top_score = self.centermap_parser.parse_centermap_heatmap_adaptive_scale_batch(outputs['center_map'], top_n_people=1)
                outputs['detection_flag'] = torch.Tensor([(False) for _ in range(len(batch_ids))])
            outputs['centers_pred'] = torch.stack([flat_inds % args().centermap_size, torch.div(flat_inds, args().centermap_size, rounding_mode='floor')], 1)
            outputs['centers_conf'] = self.parameter_sampling(outputs['center_map'], batch_ids, flat_inds, use_transform=True)
            outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image', 'offsets', 'imgpath', 'camMats']
        meta_data = reorganize_gts(meta_data, info_vis, batch_ids)
        if 'pred_batch_ids' in outputs:
            outputs['pred_batch_ids'] += meta_data['batch_ids'][0]
        return outputs, meta_data

    def parse_kps(self, heatmap_AEs, kp2d_thresh=0.1):
        kps = []
        heatmap_AE_results = self.heatmap_parser.batch_parse(heatmap_AEs.detach())
        for batch_id in range(len(heatmap_AE_results)):
            kp2d, kp2d_conf = heatmap_AE_results[batch_id]
            kps.append(kp2d[np.array(kp2d_conf) > kp2d_thresh])
        return kps


class AddCoords(nn.Module):

    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]
        xx_ones = torch.ones([1, in_tensor.shape[2]], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)
        xx_range = torch.arange(in_tensor.shape[2], dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)
        yy_ones = torch.ones([1, in_tensor.shape[3]], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)
        yy_range = torch.arange(in_tensor.shape[3], dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)
        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)
        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)
        out = torch.cat([in_tensor, xx_channel, yy_channel], dim=1)
        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, radius_calc], dim=1)
        return out


class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """

    def __init__(self, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""

    def __init__(self, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=False)
        self.convT = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class _DataParallel(Module):
    """Implements data parallelism at the module level.
    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.
    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).
    See also: :ref:`cuda-nn-dataparallel-instead`
    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.
    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
    Example::
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


class Base(nn.Module):

    def forward(self, meta_data, **cfg):
        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous())
        outputs = self.head_forward(x)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        if args().model_precision == 'fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
        else:
            outputs = self.feed_forward(meta_data)
        return outputs

    def head_forward(self, x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class IBN_a(nn.Module):

    def __init__(self, planes, momentum=BN_MOMENTUM):
        super(IBN_a, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2, momentum=momentum)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock_IBN_a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_IBN_a, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = IBN_a(planes, momentum=BN_MOMENTUM)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
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


def conv3x3_1D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


def conv3x3_3D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
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
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i]), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3), nn.ReLU(True)))
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


def get_3Dcoord_maps(size=128, z_base=None):
    range_arr = torch.arange(size, dtype=torch.float32)
    if z_base is None:
        Z_map = range_arr.reshape(1, size, 1, 1, 1).repeat(1, 1, size, size, 1) / size * 2 - 1
    else:
        Z_map = z_base.reshape(1, size, 1, 1, 1).repeat(1, 1, size, size, 1)
    Y_map = range_arr.reshape(1, 1, size, 1, 1).repeat(1, size, 1, size, 1) / size * 2 - 1
    X_map = range_arr.reshape(1, 1, 1, size, 1).repeat(1, size, size, 1, 1) / size * 2 - 1
    out = torch.cat([Z_map, Y_map, X_map], dim=-1)
    return out


class CenterMap3D(object):

    def __init__(self, conf_thresh):
        None
        self.size = 128
        self.max_person = 64
        self.sigma = 1
        self.conf_thresh = conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels([5])
        self.prepare_parsing()

    def prepare_parsing(self):
        self.coordmap_3d = get_3Dcoord_maps(size=self.size)
        self.maxpool3d = torch.nn.MaxPool3d(5, 1, (5 - 1) // 2)

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size - 1) // 2, (kernel_size - 1) // 2
            gaussian_distribution = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size - 1) // 2)
        return gk_group, pool_group

    def parse_3dcentermap(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.maxpool3d).squeeze(1)
        b, c, h, w = center_map_nms.shape
        K = self.max_person
        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds.long() // w).float()
        topk_xs = (topk_inds % w).int().float()
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_zs = index.long() // K
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)
        mask = topk_score > self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_zyxs = torch.stack([topk_zs[mask].long(), topk_ys[mask].long(), topk_xs[mask].long()]).permute((1, 0)).long()
        return [batch_ids, center_zyxs, topk_score[mask]]


def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HigherResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(HigherResolutionNet, self).__init__()
        self.make_baseline()
        self.backbone_channels = 32

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1, BN=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, BN=BN))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BN=BN))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def make_baseline(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4, BN=nn.BatchNorm2d)
        self.stage2_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [32, 64, 128], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [32, 64, 128, 256], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)

    @torch.no_grad()
    def forward(self, x):
        x = (BHWC_to_BCHW(x) / 255.0 * 2.0 - 1.0).contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = y_list[0]
        return x


def get_3Dcoord_maps_halfz(size, z_base):
    range_arr = torch.arange(size, dtype=torch.float32)
    z_len = len(z_base)
    Z_map = z_base.reshape(1, z_len, 1, 1, 1).repeat(1, 1, size, size, 1)
    Y_map = range_arr.reshape(1, 1, size, 1, 1).repeat(1, z_len, 1, size, 1) / size * 2 - 1
    X_map = range_arr.reshape(1, 1, 1, size, 1).repeat(1, z_len, size, 1, 1) / size * 2 - 1
    out = torch.cat([Z_map, Y_map, X_map], dim=-1)
    return out


def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    map_coord_range_each_level = (np.array([2 / 64.0, 25 / 64.0, 3 / 64.0, 2 / 64.0], dtype=np.float32) * centermap_size).astype(np.int)
    scale_level = 1 / np.tan(np.radians(FOV / 2.0)) / depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache - np.arange(1, coord_range + 1) / coord_range * (scale_cache - scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor


class BEVv1(nn.Module):

    def __init__(self, **kwargs):
        super(BEVv1, self).__init__()
        None
        self.backbone = HigherResolutionNet()
        self._build_head()
        self._build_parser(conf_thresh=kwargs.get('center_thresh', 0.1))

    def _build_parser(self, conf_thresh=0.12):
        self.centermap_parser = CenterMap3D(conf_thresh=conf_thresh)

    def _build_head(self):
        params_num, cam_dim = 3 + 22 * 6 + 11, 3
        self.outmap_size = 128
        self.output_cfg = {'NUM_PARAMS_MAP': params_num - cam_dim, 'NUM_CENTER_MAP': 1, 'NUM_CAM_MAP': cam_dim}
        self.head_cfg = {'NUM_BASIC_BLOCKS': 1, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size // 2, 'NUM_BLOCK': 2}
        self.backbone_channels = self.backbone.backbone_channels
        self.transformer_cfg = {'INPUT_C': self.head_cfg['NUM_CHANNELS'], 'NUM_CHANNELS': 512}
        self._make_transformer()
        self.cam3dmap_anchor = torch.from_numpy(get_cam3dmap_anchor(60, self.outmap_size)).float()
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_halfz(self.outmap_size, z_base=self.cam3dmap_anchor))
        self._make_final_layers(self.backbone_channels)

    def _make_transformer(self, drop_ratio=0.2):
        self.position_embeddings = nn.Embedding(self.outmap_size, self.transformer_cfg['INPUT_C'], padding_idx=0)
        self.transformer = nn.Sequential(nn.Linear(self.transformer_cfg['INPUT_C'], self.transformer_cfg['NUM_CHANNELS']), nn.ReLU(inplace=True), nn.Dropout(drop_ratio), nn.Linear(self.transformer_cfg['NUM_CHANNELS'], self.transformer_cfg['NUM_CHANNELS']), nn.ReLU(inplace=True), nn.Dropout(drop_ratio), nn.Linear(self.transformer_cfg['NUM_CHANNELS'], self.output_cfg['NUM_PARAMS_MAP']))

    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP'])
        self.param_head = self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'], with_outlayer=False)
        self._make_bv_center_layers(input_channels, self.bv_center_cfg['NUM_DEPTH_LEVEL'] * 2)
        self._make_3D_map_refiner()

    def _make_head_layers(self, input_channels, output_channels, num_channels=None, with_outlayer=True):
        head_layers = []
        if num_channels is None:
            num_channels = self.head_cfg['NUM_CHANNELS']
        for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
            head_layers.append(nn.Sequential(BasicBlock(input_channels, num_channels, downsample=nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0))))
            input_channels = num_channels
        if with_outlayer:
            head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*head_layers)

    def _make_bv_center_layers(self, input_channels, output_channels):
        num_channels = self.outmap_size // 8
        self.bv_pre_layers = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP']) * self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(BasicBlock_1D(input_channels, inter_channels), BasicBlock_1D(inter_channels, inter_channels), BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))

    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, img_feats], 1).view(img_feats.size(0), -1, self.outmap_size)
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1, self.bv_center_cfg['NUM_DEPTH_LEVEL'], 1, 1) * center_maps_bv.unsqueeze(2).repeat(1, 1, self.outmap_size, 1)
        return center_map_3d, cam_maps_offset_bv

    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:, :self.output_cfg['NUM_CENTER_MAP']]
        cam_maps_offset = maps_fv[:, self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP'] + self.output_cfg['NUM_CAM_MAP']]
        center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv, cam_maps_offset)
        center_maps_3d = self.center_map_refiner(center_maps_3d.unsqueeze(1)).squeeze(1)
        cam_maps_3d = self.coordmap_3d + cam_maps_offset.unsqueeze(-1).transpose(4, 1).contiguous()
        cam_maps_3d[:, :, :, :, 2] = cam_maps_3d[:, :, :, :, 2] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5, 1).squeeze(-1))
        return center_maps_3d, cam_maps_3d, center_maps_fv

    def differentiable_person_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:, 0], pred_czyxs[:, 1], pred_czyxs[:, 2]
        position_encoding = self.position_embeddings(cz)
        feature_sampled = feature[pred_batch_ids, :, cy, cx]
        input_features = feature_sampled + position_encoding
        return input_features

    def mesh_parameter_regression(self, fv_f, cams_preds, pred_batch_ids):
        cam_czyx = denormalize_center(convert_cam_params_to_centermap_coords(cams_preds.clone(), self.cam3dmap_anchor), size=self.outmap_size)
        feature_sampled = self.differentiable_person_feature_sampling(fv_f, cam_czyx, pred_batch_ids)
        params_preds = self.transformer(feature_sampled)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds, cam_czyx

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        center_preds_info_3d = self.centermap_parser.parse_3dcentermap(center_maps_3d)
        if len(center_preds_info_3d[0]) == 0:
            None
            return None
        pred_batch_ids, pred_czyxs, center_confs = center_preds_info_3d
        cams_preds = cam_maps_3d[pred_batch_ids, :, pred_czyxs[:, 0], pred_czyxs[:, 1], pred_czyxs[:, 2]]
        front_view_features = self.param_head(x)
        params_preds, cam_czyx = self.mesh_parameter_regression(front_view_features, cams_preds, pred_batch_ids)
        output = {'params_pred': params_preds.float(), 'cam_czyx': cam_czyx.float(), 'center_map': center_maps_fv.float(), 'center_map_3d': center_maps_3d.float().squeeze(), 'pred_batch_ids': pred_batch_ids, 'pred_czyxs': pred_czyxs, 'center_confs': center_confs}
        return output


class ResultSaver:

    def __init__(self, mode='image', save_path=None, save_npz=True):
        self.is_dir = len(osp.splitext(save_path)[1]) == 0
        self.mode = mode
        self.save_path = save_path
        self.save_npz = save_npz
        self.save_dir = save_path if self.is_dir else osp.dirname(save_path)
        if self.mode in ['image', 'video']:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.mode == 'video':
            self.frame_save_paths = []

    def __call__(self, outputs, input_path, prefix=None, img_ext='.png'):
        if self.mode == 'video' or self.is_dir:
            save_name = osp.basename(input_path)
            save_path = osp.join(self.save_dir, osp.splitext(save_name)[0]) + img_ext
        elif self.mode == 'image':
            save_path = self.save_path
        if prefix is not None:
            save_path = osp.splitext(save_path)[0] + f'_{prefix}' + osp.splitext(save_path)[1]
        rendered_image = None
        if outputs is not None:
            if 'rendered_image' in outputs:
                rendered_image = outputs.pop('rendered_image')
            if self.save_npz:
                np.savez(osp.splitext(save_path)[0] + '.npz', results=outputs)
        if rendered_image is None:
            rendered_image = cv2.imread(input_path)
        cv2.imwrite(save_path, rendered_image)
        if self.mode == 'video':
            self.frame_save_paths.append(save_path)

    def save_video(self, save_path, frame_rate=24):
        if len(self.frame_save_paths) == 0:
            return
        height, width = cv2.imread(self.frame_save_paths[0]).shape[:2]
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        for frame_path in self.frame_save_paths:
            writer.write(cv2.imread(frame_path))
        writer.release()


def convert_cam_to_3d_trans2(j3ds, pj3d):
    predicts_j3ds = j3ds[:, :24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:, :, :2][:, :24].detach().cpu().numpy() + 1) * 256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, focal_length=443.4, img_size=np.array([512, 512]))
    return cam_trans


def convert_proejection_from_input_to_orgimg(kps, offsets):
    top, bottom, left, right, h, w = offsets
    img_pad_size = max(h, w)
    kps[:, :, 0] = (kps[:, :, 0] + 1) * img_pad_size / 2 - left
    kps[:, :, 1] = (kps[:, :, 1] + 1) * img_pad_size / 2 - top
    if kps.shape[-1] == 3:
        kps[:, :, 2] = (kps[:, :, 2] + 1) * img_pad_size / 2
    return kps


def body_mesh_projection2image(j3d_preds, cam_preds, vertices=None, input2org_offsets=None):
    pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
    pred_cam_t = convert_cam_to_3d_trans2(j3d_preds, pj3d)
    projected_outputs = {'pj2d': pj3d[:, :, :2], 'cam_trans': pred_cam_t}
    if vertices is not None:
        projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d', keep_dim=True)
    if input2org_offsets is not None:
        projected_outputs['pj2d_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['pj2d'], input2org_offsets)
        projected_outputs['verts_camed_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['verts_camed'], input2org_offsets)
    return projected_outputs


class LowPassFilter:

    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x, print_inter=False):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        if isinstance(edx, float):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, np.ndarray):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, torch.Tensor):
            cutoff = self.mincutoff + self.beta * torch.abs(edx)
        if print_inter:
            None
        return self.x_filter.process(x, self.compute_alpha(cutoff))


def create_OneEuroFilter(smooth_coeff):
    return {'smpl_thetas': OneEuroFilter(smooth_coeff, 0.7), 'cam': OneEuroFilter(1.6, 0.7), 'smpl_betas': OneEuroFilter(0.6, 0.7), 'global_rot': OneEuroFilter(smooth_coeff, 0.7)}


def check_filter_state(OE_filters, signal_ID, show_largest=False, smooth_coeff=3.0):
    if len(OE_filters) > 100:
        del OE_filters
    if signal_ID not in OE_filters:
        if show_largest:
            OE_filters[signal_ID] = create_OneEuroFilter(smooth_coeff)
        else:
            OE_filters[signal_ID] = {}
    if len(OE_filters[signal_ID]) > 1000:
        del OE_filters[signal_ID]


def convert_tensor2numpy(outputs, del_keys=['verts_camed', 'smpl_face', 'pj2d', 'verts_camed_org']):
    for key in del_keys:
        if key in outputs:
            del outputs[key]
    result_keys = list(outputs.keys())
    for key in result_keys:
        if isinstance(outputs[key], torch.Tensor):
            outputs[key] = outputs[key].cpu().numpy()
    return outputs


tan_fov = np.tan(np.radians(60 / 2.0))


def convert_scale_to_depth(scale):
    return 1 / (scale * tan_fov + 0.001)


def denormalize_cam_params_to_trans(normed_cams, positive_constrain=False):
    scale = normed_cams[:, 0]
    if positive_constrain:
        positive_mask = (normed_cams[:, 0] > 0).float()
        scale = scale * positive_mask
    trans_XY_normed = torch.flip(normed_cams[:, 1:], [1])
    depth = convert_scale_to_depth(scale).unsqueeze(1)
    trans_XY = trans_XY_normed * depth * tan_fov
    trans = torch.cat([trans_XY, depth], 1)
    return trans


def determine_device(gpu_id):
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    return device


def padding_image(image):
    h, w = image.shape[:2]
    side_length = max(h, w)
    pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top + h), int(left + w)
    pad_image[top:bottom, left:right] = image
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info


def img_preprocess(image, input_size=512):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, image_pad_info = padding_image(image)
    input_image = torch.from_numpy(cv2.resize(pad_image, (input_size, input_size), interpolation=cv2.INTER_CUBIC))[None].float()
    return input_image, image_pad_info


def pack_params_dict(params_pred):
    idx_list, params_dict = [0], {}
    part_name = ['cam', 'global_orient', 'body_pose', 'smpl_betas']
    part_idx = [3, 6, 21 * 6, 10]
    for i, (idx, name) in enumerate(zip(part_idx, part_name)):
        idx_list.append(idx_list[i] + idx)
        params_dict[name] = params_pred[:, idx_list[i]:idx_list[i + 1]].contiguous()
    params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
    params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
    N = params_dict['body_pose'].shape[0]
    params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N, 6)], 1)
    params_dict['smpl_thetas'] = torch.cat([params_dict['global_orient'], params_dict['body_pose']], 1)
    return params_dict


def remove_subjects(outputs, removed_subj_inds):
    N = len(outputs['params_pred'])
    remove_mask = torch.ones(N).bool()
    remove_mask[removed_subj_inds] = False
    left_subj_inds = torch.where(remove_mask)[0].tolist()
    keys = list(outputs.keys())
    for key in keys:
        if key in ['smpl_face', 'center_map', 'center_map_3d']:
            continue
        outputs[key] = outputs[key][left_subj_inds]
    return outputs


def remove_outlier(outputs, relative_scale_thresh=3, scale_thresh=0.25):
    cam_trans = outputs['cam_trans']
    N = len(cam_trans)
    if N < 3:
        return outputs
    trans_diff = cam_trans.unsqueeze(1).repeat(1, N, 1) - cam_trans.unsqueeze(0).repeat(N, 1, 1)
    trans_dist_mat = torch.norm(trans_diff, p=2, dim=-1)
    trans_dist_mat = torch.sort(trans_dist_mat).values[:, 1:-1]
    mean_dist = trans_dist_mat.mean(1)
    relative_scale = mean_dist / ((mean_dist.sum() - mean_dist) / (N - 1))
    outlier_mask = relative_scale > relative_scale_thresh
    outlier_mask *= outputs['cam'][:, 0] < scale_thresh
    removed_subj_inds = torch.where(outlier_mask)[0]
    if len(removed_subj_inds) > 0:
        outputs = remove_subjects(outputs, removed_subj_inds)
    return outputs


smpl24_connMat = np.array([0, 1, 0, 2, 0, 3, 1, 4, 4, 7, 7, 10, 2, 5, 5, 8, 8, 11, 3, 6, 6, 9, 9, 12, 12, 15, 12, 13, 13, 16, 16, 18, 18, 20, 20, 22, 12, 14, 14, 17, 17, 19, 19, 21, 21, 23]).reshape(-1, 2)


class Plotter3dPoses:

    def __init__(self, canvas_size=(512, 512), origin=(0.5, 0.5), scale=200):
        self.canvas_size = canvas_size
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)
        self.scale = np.float32(scale)
        self.theta, self.phi = 0, np.pi / 2
        axis_length = 200
        axes = [np.array([[-axis_length / 2, -axis_length / 2, 0], [axis_length / 2, -axis_length / 2, 0]], dtype=np.float32), np.array([[-axis_length / 2, -axis_length / 2, 0], [-axis_length / 2, axis_length / 2, 0]], dtype=np.float32), np.array([[-axis_length / 2, -axis_length / 2, 0], [-axis_length / 2, -axis_length / 2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0], [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0], [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, pose_3ds, bones=smpl24_connMat, colors=[(255, 0, 0)], img=None):
        img = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 0 if img is None else img
        R = self._get_rotation(self.theta, self.phi)
        for vertices, color in zip(pose_3ds, colors):
            self._plot_edges(img, vertices, bones, R, color)
        return img

    def encircle_plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255 if img is None else img
        encircle_theta, encircle_phi = [0, 0, 0, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 2], [np.pi / 2, 5 * np.pi / 7, -2 * np.pi / 7, np.pi / 2, 5 * np.pi / 7, -2 * np.pi / 7, np.pi / 2, 5 * np.pi / 7, -2 * np.pi / 7]
        encircle_origin = np.array([[0.165, 0.165], [0.165, 0.495], [0.165, 0.825], [0.495, 0.165], [0.495, 0.495], [0.495, 0.825], [0.825, 0.165], [0.825, 0.495], [0.825, 0.825]], dtype=np.float32) * np.array(self.canvas_size)[None]
        for self.theta, self.phi, self.origin in zip(encircle_theta, encircle_phi, encircle_origin):
            R = self._get_rotation(self.theta, self.phi)
            for vertices, color in zip(pose_3ds, colors):
                self._plot_edges(img, vertices * 0.6, bones, R, color)
        return img

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R, color):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        org_verts = vertices.reshape((-1, 3))[edges]
        for inds, edge_vertices in enumerate(edges_vertices):
            if 0 in org_verts[inds]:
                continue
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), color, 10, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([[cos(theta), sin(theta) * sin(phi)], [-sin(theta), cos(theta) * sin(phi)], [0, -cos(phi)]], dtype=np.float32)


def draw_skeleton(image, pts, bones=smpl24_connMat, cm=None, label_kp_order=False, r=8):
    for i, pt in enumerate(pts):
        if len(pt) > 1:
            if pt[0] > 0 and pt[1] > 0:
                image = cv2.circle(image, (int(pt[0]), int(pt[1])), r, cm[i % len(cm)], -1)
                if label_kp_order and i in bones:
                    img = cv2.putText(image, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 215, 0), 1)
    if bones is not None:
        set_colors = np.array([cm for i in range(len(bones))]).astype(np.int)
        bones = np.concatenate([bones, set_colors], 1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa > 0).all() and (pb > 0).all():
                xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
                image = cv2.line(image, (xa, ya), (xb, yb), (int(line[2]), int(line[3]), int(line[4])), r)
    return image


def draw_skeleton_multiperson(image, pts_group, colors):
    for ind, pts in enumerate(pts_group):
        image = draw_skeleton(image, pts, cm=colors[ind])
    return image


color_table_default = np.array([[0.4, 0.6, 1], [0.8, 0.7, 1], [0.1, 0.9, 1], [0.8, 0.9, 1], [1, 0.6, 0.4], [1, 0.7, 0.8], [1, 0.9, 0.1], [1, 0.9, 0.8], [0.9, 1, 1], [0.9, 0.7, 0.4], [0.8, 0.7, 1], [0.8, 0.9, 1], [0.9, 0.3, 0.1], [0.7, 1, 0.6], [0.7, 0.4, 0.6], [0.3, 0.5, 1]])[:, ::-1]


def mesh_color_left2right(trans, color_table=None):
    left2right_order = torch.sort(trans[:, 0].cpu()).indices.numpy()
    color_inds = np.arange(len(trans))
    color_inds[left2right_order] = np.arange(len(trans))
    if color_table is None:
        color_table = color_table_default
    return np.array([color_table[ind % len(color_table)] for ind in color_inds])


tracking_color_list = np.array([0.0, 0.447, 0.741, 0.85, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.749, 0.749, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.667, 0.0, 1.0, 0.333, 0.333, 0.0, 0.333, 0.667, 0.0, 0.333, 1.0, 0.0, 0.667, 0.333, 0.0, 0.667, 0.667, 0.0, 0.667, 1.0, 0.0, 1.0, 0.333, 0.0, 1.0, 0.667, 0.0, 1.0, 1.0, 0.0, 0.0, 0.333, 0.5, 0.0, 0.667, 0.5, 0.0, 1.0, 0.5, 0.333, 0.0, 0.5, 0.333, 0.333, 0.5, 0.333, 0.667, 0.5, 0.333, 1.0, 0.5, 0.667, 0.0, 0.5, 0.667, 0.333, 0.5, 0.667, 0.667, 0.5, 0.667, 1.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.333, 0.5, 1.0, 0.667, 0.5, 1.0, 1.0, 0.5, 0.0, 0.333, 1.0, 0.0, 0.667, 1.0, 0.0, 1.0, 1.0, 0.333, 0.0, 1.0, 0.333, 0.333, 1.0, 0.333, 0.667, 1.0, 0.333, 1.0, 1.0, 0.667, 0.0, 1.0, 0.667, 0.333, 1.0, 0.667, 0.667, 1.0, 0.667, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.333, 1.0, 1.0, 0.667, 1.0, 0.167, 0.0, 0.0, 0.333, 0.0, 0.0, 0.5, 0.0, 0.0, 0.667, 0.0, 0.0, 0.833, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.167, 0.0, 0.0, 0.333, 0.0, 0.0, 0.5, 0.0, 0.0, 0.667, 0.0, 0.0, 0.833, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.167, 0.0, 0.0, 0.333, 0.0, 0.0, 0.5, 0.0, 0.0, 0.667, 0.0, 0.0, 0.833, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.0, 1.0, 1.0]).astype(np.float32).reshape((-1, 3))


def mesh_color_trackID(track_ids, color_table=None):
    if color_table is None:
        color_table = tracking_color_list
    return np.array([color_table[tid % len(color_table)] for tid in track_ids])


def get_rotate_x_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return rot_mat


def get_rotate_y_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return rot_mat


def rotate_view_perspective(verts, rx=30, ry=0, FOV=60, bbox3D_center=None, depth=None):
    device, dtype = verts.device, verts.dtype
    Rx_mat = get_rotate_x_mat(rx).type(dtype)
    Ry_mat = get_rotate_y_mat(ry).type(dtype)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    if depth is None:
        dist_min = torch.abs(verts_aligned.view(-1, 3).min(0).values)
        dist_max = torch.abs(verts_aligned.view(-1, 3).max(0).values)
        z = dist_max[:2].max() / np.tan(np.radians(FOV / 2)) + dist_min[2]
        depth = torch.tensor([[[0, 0, z]]], device=device)
    verts_aligned = verts_aligned + depth
    return verts_aligned, bbox3D_center, depth


def rendering_mesh_rotating_view(vert_trans, renderer, triangles, image, background, internal=5):
    result_imgs = []
    pause_num = 24
    pause = np.zeros(pause_num).astype(np.int32)
    change_time = 90 // internal
    roates = np.ones(change_time) * internal
    go_up = np.sin(np.arange(change_time).astype(np.float32) / change_time) * 1
    go_down = np.sin(np.arange(change_time).astype(np.float32) / change_time - 1) * 1
    azimuth_angles = np.concatenate([pause, roates, roates, roates, roates])
    elevation_angles = np.concatenate([pause, go_up, go_down, go_up, go_down])
    camera_pose = np.eye(4)
    elevation_start = 20
    camera_pose[:3, :3] = get_rotate_x_mat(-elevation_start)
    cam_height = 1.4 * vert_trans[:, :, 2].mean().item() * np.tan(np.radians(elevation_start))
    camera_pose[:3, 3] = np.array([0, cam_height, 0])
    verts_rotated = vert_trans.clone()
    bbox3D_center, move_depth = None, None
    for azimuth_angle, elevation_angle in zip(azimuth_angles, elevation_angles):
        verts_rotated, bbox3D_center, move_depth = rotate_view_perspective(verts_rotated, rx=0, ry=azimuth_angle, depth=move_depth)
        rendered_image, rend_depth = renderer(verts_rotated.cpu().numpy(), triangles, background, mesh_colors=np.array([[0.9, 0.9, 0.8]]), camera_pose=camera_pose)
        result_imgs.append(rendered_image)
    return result_imgs


def rotate_view_weak_perspective(verts, rx=30, ry=0, img_shape=[512, 512], expand_ratio=1.2, bbox3D_center=None, scale=None):
    device, dtype = verts.device, verts.dtype
    h, w = img_shape
    Rx_mat = get_rotate_x_mat(rx).type(dtype)
    Ry_mat = get_rotate_y_mat(ry).type(dtype)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    rendered_image_center = torch.Tensor([[[w / 2, h / 2]]]).type(verts_aligned.dtype)
    if scale is None:
        scale = 1 / (expand_ratio * torch.abs(torch.div(verts_aligned[:, :, :2], rendered_image_center)).max())
    verts_aligned *= scale
    verts_aligned[:, :, :2] += rendered_image_center
    return verts_aligned, bbox3D_center, scale


def rendering_romp_bev_results(renderer, outputs, image, rendering_cfgs, alpha=1):
    triangles = outputs['smpl_face'].cpu().numpy().astype(np.int32)
    h, w = image.shape[:2]
    background = np.ones([h, h, 3], dtype=np.uint8) * 255
    result_image = [image]
    cam_trans = outputs['cam_trans']
    if rendering_cfgs['mesh_color'] == 'identity':
        if 'track_ids' in outputs:
            mesh_colors = mesh_color_trackID(outputs['track_ids'])
        else:
            mesh_colors = mesh_color_left2right(cam_trans)
    elif rendering_cfgs['mesh_color'] == 'same':
        mesh_colors = np.array([[0.9, 0.9, 0.8] for _ in range(len(cam_trans))])
    if rendering_cfgs['renderer'] == 'sim3dr':
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=True).indices.numpy()
        vertices = outputs['verts_camed_org'][depth_order].cpu().numpy()
        mesh_colors = mesh_colors[depth_order]
        verts_tran = (outputs['verts'] + cam_trans.unsqueeze(1))[depth_order]
        vertices[:, :, 2] = vertices[:, :, 2] * -1
        verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
        if 'mesh' in rendering_cfgs['items']:
            rendered_image = renderer(vertices, triangles, image, mesh_colors=mesh_colors)
            result_image.append(rendered_image)
        if 'mesh_bird_view' in rendering_cfgs['items']:
            verts_bird_view, bbox3D_center, scale = rotate_view_weak_perspective(verts_tran, rx=-90, ry=0, img_shape=background.shape[:2], expand_ratio=1.2)
            rendered_bv_image = renderer(verts_bird_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(rendered_bv_image)
        if 'mesh_side_view' in rendering_cfgs['items']:
            verts_side_view, bbox3D_center, scale = rotate_view_weak_perspective(verts_tran, rx=0, ry=-90, img_shape=image.shape[:2], expand_ratio=1.2)
            rendered_sv_image = renderer(verts_side_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(rendered_sv_image)
    if rendering_cfgs['renderer'] == 'pyrender':
        verts_tran = outputs['verts'] + cam_trans.unsqueeze(1)
        if 'mesh' in rendering_cfgs['items']:
            rendered_image, rend_depth = renderer(verts_tran.cpu().numpy(), triangles, image, mesh_colors=mesh_colors)
            result_image.append(rendered_image)
        if 'mesh_bird_view' in rendering_cfgs['items']:
            verts_bird_view, bbox3D_center, move_depth = rotate_view_perspective(verts_tran, rx=90, ry=0)
            rendered_bv_image, rend_depth = renderer(verts_bird_view.cpu().numpy(), triangles, background, persp=False, mesh_colors=mesh_colors)
            result_image.append(cv2.resize(rendered_bv_image, (h, h)))
        if 'mesh_side_view' in rendering_cfgs['items']:
            verts_side_view, bbox3D_center, move_depth = rotate_view_perspective(verts_tran, rx=0, ry=90)
            rendered_sv_image, rend_depth = renderer(verts_side_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(cv2.resize(rendered_sv_image, (h, h)))
        if 'rotate_mesh' in rendering_cfgs['items']:
            rot_trans = cam_trans.unsqueeze(1)
            rot_trans[:, :, 2] /= 1.5
            verts_tran_rot = outputs['verts'] + rot_trans
            rotate_renderings = rendering_mesh_rotating_view(verts_tran_rot, renderer, triangles, image, background)
            time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
            save_path = os.path.join(os.path.expanduser('~'), 'rotate-{}.mp4'.format(time_stamp))
            frame_rate = 24
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, rotate_renderings[0].shape[:2])
            for frame in rotate_renderings:
                writer.write(frame)
            writer.release()
            None
    if 'pj2d' in rendering_cfgs['items']:
        img_skeleton2d = draw_skeleton_multiperson(copy.deepcopy(image), outputs['pj2d_org'].cpu().numpy()[:, :24], mesh_colors * 255)
        result_image.append(img_skeleton2d)
    if 'j3d' in rendering_cfgs['items']:
        plot_3dpose = Plotter3dPoses(canvas_size=(h, h))
        joint_trans = (outputs['joints'] + cam_trans.unsqueeze(1)).cpu().numpy()[:, :24] * 3
        img_skeleton3d = plot_3dpose.plot(joint_trans, colors=mesh_colors * 255)
        result_image.append(img_skeleton3d)
    if 'center_conf' in rendering_cfgs['items']:
        for ind, kp in enumerate(outputs['pj2d_org'].cpu().numpy()[:, 0]):
            cv2.putText(result_image[1], '{:.3f}'.format(outputs['center_confs'][ind]), tuple(kp.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
    if 'tracking' in rendering_cfgs['items'] and 'track_ids' in outputs:
        for ind, kp in enumerate(outputs['pj2d_org'].cpu().numpy()[:, 0]):
            cv2.putText(result_image[1], '{:d}'.format(outputs['track_ids'][ind]), tuple(kp.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
    outputs['rendered_image'] = np.concatenate(result_image, 1)
    return outputs


def setup_renderer(name='sim3dr', **kwargs):
    if name == 'sim3dr':
        renderer = Sim3DR(**kwargs)
    elif name == 'pyrender':
        renderer = Py3DR(**kwargs)
    elif name == 'open3d':
        renderer = O3DDR(multi_mode=True, **kwargs)
    return renderer


def transform_rot_representation(rot, input_type='mat', out_type='quat', input_is_degrees=True):
    """
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees in x,y,z (3)
    """
    if input_type == 'mat':
        r = R.from_matrix(rot)
    elif input_type == 'quat':
        r = R.from_quat(rot)
    elif input_type == 'vec':
        r = R.from_rotvec(rot)
    elif input_type == 'euler':
        r = R.from_euler('xyz', rot, degrees=input_is_degrees)
    if out_type == 'mat':
        out = r.as_matrix()
    elif out_type == 'quat':
        out = r.as_quat()
    elif out_type == 'vec':
        out = r.as_rotvec()
    elif out_type == 'euler':
        out = r.as_euler('xyz', degrees=False)
    return out


def smooth_global_rot_matrix(pred_rots, OE_filter):
    rot_mat = batch_rodrigues(pred_rots[None]).squeeze(0)
    smoothed_rot_mat = OE_filter.process(rot_mat)
    smoothed_rot = rotation_matrix_to_angle_axis(smoothed_rot_mat.reshape(1, 3, 3)).reshape(-1)
    return smoothed_rot
    device = pred_rots.device
    rot_euler = transform_rot_representation(pred_rots.cpu().numpy(), input_type='vec', out_type='mat')
    smoothed_rot = OE_filter.process(rot_euler)
    smoothed_rot = transform_rot_representation(smoothed_rot, input_type='mat', out_type='vec')
    smoothed_rot = torch.from_numpy(smoothed_rot).float()
    return smoothed_rot


def smooth_results(filters, body_pose=None, body_shape=None, cam=None):
    if body_pose is not None:
        global_rot = smooth_global_rot_matrix(body_pose[:3], filters['global_rot'])
        body_pose = torch.cat([global_rot, filters['smpl_thetas'].process(body_pose[3:])], 0)
    if body_shape is not None:
        body_shape = filters['smpl_betas'].process(body_shape)
    if cam is not None:
        cam = filters['cam'].process(cam)
    return body_pose, body_shape, cam


def suppressing_redundant_prediction_via_projection(outputs, img_shape, thresh=16, conf_based=False):
    pj2ds = outputs['pj2d']
    N = len(pj2ds)
    if N == 1:
        return outputs
    pj2d_diff = pj2ds.unsqueeze(1).repeat(1, N, 1, 1) - pj2ds.unsqueeze(0).repeat(N, 1, 1, 1)
    pj2d_dist_mat = torch.norm(pj2d_diff, p=2, dim=-1).mean(-1)
    person_scales = outputs['cam'][:, 0] * 2
    ps1, ps2 = person_scales.unsqueeze(1).repeat(1, N), person_scales.unsqueeze(0).repeat(N, 1)
    max_scale_mat = torch.where(ps1 > ps2, ps1, ps2)
    pj2d_dist_mat_normalized = pj2d_dist_mat / max_scale_mat
    triu_mask = torch.triu(torch.ones_like(pj2d_dist_mat), diagonal=1) < 0.5
    pj2d_dist_mat_normalized[triu_mask] = 10000.0
    max_length = max(img_shape)
    thresh = thresh * max_length / 640
    repeat_subj_inds = torch.where(pj2d_dist_mat_normalized < thresh)
    if len(repeat_subj_inds) > 0:
        if conf_based:
            center_confs = outputs['center_confs']
            removed_subj_inds = torch.where(center_confs[repeat_subj_inds[0]] < center_confs[repeat_subj_inds[1]], repeat_subj_inds[0], repeat_subj_inds[1])
        else:
            removed_subj_inds = torch.where(person_scales[repeat_subj_inds[0]] < person_scales[repeat_subj_inds[1]], repeat_subj_inds[0], repeat_subj_inds[1])
        outputs = remove_subjects(outputs, removed_subj_inds)
    return outputs


def time_cost(name='ROMP'):

    def time_counter(func):

        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            cost_time = t2 - t1
            fps = 1.0 / cost_time
            None
            return result
        return wrap_func
    return time_counter


def wait_func(mode):
    if mode == 'image':
        None
        while 1:
            if cv2.waitKey() == 27:
                break
    elif mode == 'webcam' or mode == 'video':
        cv2.waitKey(1)


class BEV(nn.Module):

    def __init__(self, romp_settings):
        super(BEV, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()

    def _build_model_(self):
        model = BEVv1(center_thresh=self.settings.center_thresh).eval()
        model.load_state_dict(torch.load(self.settings.model_path, map_location=self.tdevice), strict=False)
        model = model
        self.model = nn.DataParallel(model)

    def _initilization_(self):
        if self.settings.calc_smpl:
            self.smpl_parser = SMPLA_parser(self.settings.smpl_path, self.settings.smil_path)
        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_(self.settings.smooth_coeff)
        if self.settings.render_mesh or self.settings.mode == 'webcam':
            self.renderer = setup_renderer(name=self.settings.renderer)
        self.visualize_items = self.settings.show_items.split(',')
        self.result_keys = ['smpl_thetas', 'smpl_betas', 'cam', 'cam_trans', 'params_pred', 'center_confs', 'pred_batch_ids']

    def _initialize_optimization_tools_(self, smooth_coeff):
        self.OE_filters = {}
        if not self.settings.show_largest:
            self.tracker = Tracker(det_thresh=0.12, low_conf_det_thresh=0.05, track_buffer=60, match_thresh=300, frame_rate=30)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        parsed_results = self.model(input_image)
        if parsed_results is None:
            return None, image_pad_info
        parsed_results.update(pack_params_dict(parsed_results['params_pred']))
        parsed_results.update({'cam_trans': denormalize_cam_params_to_trans(parsed_results['cam'])})
        all_result_keys = list(parsed_results.keys())
        for key in all_result_keys:
            if key not in self.result_keys:
                del parsed_results[key]
        return parsed_results, image_pad_info

    @time_cost('BEV')
    @torch.no_grad()
    def forward(self, image, signal_ID=0, **kwargs):
        if image.shape[1] / image.shape[0] >= 2 and self.settings.crowd:
            outputs = self.process_long_image(image, show_patch_results=self.settings.show_patch_results)
        else:
            outputs = self.process_normal_image(image, signal_ID)
        if outputs is None:
            return None
        if self.settings.render_mesh:
            mesh_color_type = 'identity' if self.settings.mode != 'webcam' and not self.settings.save_video else 'same'
            rendering_cfgs = {'mesh_color': mesh_color_type, 'items': self.visualize_items, 'renderer': self.settings.renderer}
            outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
        if self.settings.show:
            h, w = outputs['rendered_image'].shape[:2]
            show_image = outputs['rendered_image'] if h <= 1080 else cv2.resize(outputs['rendered_image'], (int(w * (1080 / h)), 1080))
            cv2.imshow('rendered', show_image)
            wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs)

    def process_normal_image(self, image, signal_ID):
        outputs, image_pad_info = self.single_image_forward(image)
        meta_data = {'input2org_offsets': image_pad_info}
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
            if outputs is None:
                return None
            outputs.update({'cam_trans': denormalize_cam_params_to_trans(outputs['cam'])})
        if self.settings.calc_smpl:
            verts, joints, face = self.smpl_parser(outputs['smpl_betas'], outputs['smpl_thetas'])
            outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
            if self.settings.render_mesh:
                meta_data['vertices'] = outputs['verts']
            projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
            outputs.update(projection)
            outputs = suppressing_redundant_prediction_via_projection(outputs, image.shape, thresh=self.settings.nms_thresh)
            outputs = remove_outlier(outputs, relative_scale_thresh=self.settings.relative_scale_thresh)
        return outputs

    def process_long_image(self, full_image, show_patch_results=False):
        None
        full_image_pad, image_pad_info, pad_length = padding_image_overlap(full_image, overlap_ratio=self.settings.overlap_ratio)
        meta_data = {'input2org_offsets': image_pad_info}
        fh, fw = full_image_pad.shape[:2]
        crop_boxes = get_image_split_plan(full_image_pad, overlap_ratio=self.settings.overlap_ratio)
        croped_images, outputs_list = [], []
        for cid, crop_box in enumerate(crop_boxes):
            l, r, t, b = crop_box
            croped_image = full_image_pad[t:b, l:r]
            crop_outputs, image_pad_info = self.single_image_forward(croped_image)
            if crop_outputs is None:
                outputs_list.append(crop_outputs)
                continue
            verts, joints, face = self.smpl_parser(crop_outputs['smpl_betas'], crop_outputs['smpl_thetas'])
            crop_outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
            outputs_list.append(crop_outputs)
            croped_images.append(croped_image)
        for cid in range(len(crop_boxes)):
            this_outs = outputs_list[cid]
            if this_outs is not None:
                if cid != len(crop_boxes) - 1:
                    this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid + 1, 0]
                    drop_boundary_ratio = (this_right - next_left) / fh / 2
                    exclude_boudary_subjects(this_outs, drop_boundary_ratio, ptype='left', torlerance=0)
                ch, cw = croped_images[cid].shape[:2]
                projection = body_mesh_projection2image(this_outs['joints'], this_outs['cam'], vertices=this_outs['verts'], input2org_offsets=torch.Tensor([0, ch, 0, cw, ch, cw]))
                this_outs.update(projection)
        for cid in range(1, len(crop_boxes) - 1):
            this_outs, next_outs = outputs_list[cid], outputs_list[cid + 1]
            this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid + 1, 0]
            drop_boundary_ratio = (this_right - next_left) / fh / 2
            if next_outs is not None:
                exclude_boudary_subjects(next_outs, drop_boundary_ratio, ptype='right', torlerance=0)
        for cid, crop_image in enumerate(croped_images):
            this_outs = outputs_list[cid]
            ch, cw = croped_images[cid].shape[:2]
            this_outs = suppressing_redundant_prediction_via_projection(this_outs, [ch, cw], thresh=self.settings.nms_thresh, conf_based=True)
            this_outs = remove_outlier(this_outs, scale_thresh=1, relative_scale_thresh=self.settings.relative_scale_thresh)
        if show_patch_results:
            rendering_cfgs = {'mesh_color': 'identity', 'items': ['mesh', 'center_conf', 'pj2d'], 'renderer': self.settings.renderer}
            for cid, crop_image in enumerate(croped_images):
                this_outs = outputs_list[cid]
                this_outs = rendering_romp_bev_results(self.renderer, this_outs, crop_image, rendering_cfgs)
                saver = ResultSaver(self.settings.mode, self.settings.save_path)
                saver(this_outs, 'crop.jpg', prefix=f'{self.settings.center_thresh}_{cid}')
        outputs = {}
        for cid, crop_box in enumerate(crop_boxes):
            crop_outputs = outputs_list[cid]
            if crop_outputs is None:
                continue
            crop_box[:2] -= pad_length
            crop_outputs['cam'] = convert_crop_cam_params2full_image(crop_outputs['cam'], crop_box, full_image.shape[:2])
            collect_outputs(crop_outputs, outputs)
        if self.settings.render_mesh:
            meta_data['vertices'] = outputs['verts']
        projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
        outputs.update(projection)
        outputs = suppressing_redundant_prediction_via_projection(outputs, full_image.shape, thresh=self.settings.nms_thresh, conf_based=True)
        outputs = remove_outlier(outputs, scale_thresh=0.5, relative_scale_thresh=self.settings.relative_scale_thresh)
        return outputs

    def temporal_optimization(self, outputs, signal_ID, image_scale=128, depth_scale=30):
        check_filter_state(self.OE_filters, signal_ID, self.settings.show_largest, self.settings.smooth_coeff)
        if self.settings.show_largest:
            max_id = torch.argmax(outputs['cam'][:, 0])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = smooth_results(self.OE_filters[signal_ID], outputs['smpl_thetas'][max_id], outputs['smpl_betas'][max_id], outputs['cam'][max_id])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = outputs['smpl_thetas'].unsqueeze(0), outputs['smpl_betas'].unsqueeze(0), outputs['cam'].unsqueeze(0)
        else:
            cam_trans = outputs['cam_trans'].cpu().numpy()
            cams = outputs['cam'].cpu().numpy()
            det_confs = outputs['center_confs'].cpu().numpy()
            tracking_points = np.concatenate([(cams[:, [2, 1]] + 1) * image_scale, cam_trans[:, [2]] * depth_scale, cams[:, [0]] * image_scale / 2], 1)
            tracked_ids, results_inds = self.tracker.update(tracking_points, det_confs)
            if len(tracked_ids) == 0:
                return None
            for key in self.result_keys:
                outputs[key] = outputs[key][results_inds]
            for ind, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(self.settings.smooth_coeff)
                outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind] = smooth_results(self.OE_filters[signal_ID][tid], outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind])
            outputs['track_ids'] = np.array(tracked_ids).astype(np.int32)
        return outputs


def copy_state_dict(cur_state_dict, pre_state_dict, prefix='module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []

    def _get_params(key):
        key = key.replace(drop_prefix, '')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix != '':
                k = k.split(prefix)[1]
            success_layers.append(k)
        except:
            None
            continue
    None
    if fix_loaded and len(failed_layers) > 0:
        logging.info('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad = False
            except:
                logging.info('fixing the layer {} failed'.format(k))
    return success_layers


class ResNet_50(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(ResNet_50, self).__init__()
        self.make_resnet()
        self.backbone_channels = 64

    def load_pretrain_params(self):
        if os.path.exists(args().resnet_pretrain):
            success_layer = copy_state_dict(self.state_dict(), torch.load(args().resnet_pretrain), prefix='', fix_loaded=True)

    def image_preprocess(self, x):
        x = BHWC_to_BCHW(x) / 255.0
        x = torch.stack(list(map(lambda x: F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False), x)))
        return x

    def make_resnet(self):
        block, layers = Bottleneck, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_resnet_layer(block, 64, layers[0])
        self.layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_resnet_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(3, (256, 128, 64), (4, 4, 4))

    def forward(self, x):
        x = self.image_preprocess(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        return x

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            if i == 0:
                self.inplanes = 2048
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


def get_coord_maps(size=128):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)
    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)
    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)
    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)
    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)
    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)
    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)
    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out


class ROMPv1(nn.Module):

    def __init__(self, **kwargs):
        super(ROMPv1, self).__init__()
        None
        self.backbone = HigherResolutionNet()
        self._build_head()

    def _build_head(self):
        self.outmap_size = 64
        params_num, cam_dim = 3 + 22 * 6 + 10, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': 2}
        self.output_cfg = {'NUM_PARAMS_MAP': params_num - cam_dim, 'NUM_CENTER_MAP': 1, 'NUM_CAM_MAP': cam_dim}
        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = [None]
        input_channels += 2
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
        return nn.ModuleList(final_layers)

    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']
        head_layers.append(nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))
        head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*head_layers)

    @torch.no_grad()
    def forward(self, image):
        x = self.backbone(image)
        x = torch.cat((x, self.coordmaps.repeat(x.shape[0], 1, 1, 1)), 1)
        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        cam_maps = self.final_layers[3](x)
        params_maps = torch.cat([cam_maps, params_maps], 1)
        return center_maps, params_maps


class SMPL_parser(nn.Module):

    def __init__(self, model_path):
        super(SMPL_parser, self).__init__()
        self.smpl_model = SMPL(model_path)

    def forward(self, outputs, root_align=False):
        verts, joints, face = self.smpl_model(outputs['smpl_betas'], outputs['smpl_thetas'], root_align=root_align)
        outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
        return outputs


def convert_cam_to_3d_trans(cams, weight=2.0):
    s, tx, ty = cams[:, 0], cams[:, 1], cams[:, 2]
    depth, dx, dy = 1.0 / s, tx / s, ty / s
    trans3d = torch.stack([dx, dy, depth], 1) * weight
    return trans3d


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points[0] for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points - point[None], axis=1))] for point in org_points]
    return tracked_ids


def parameter_sampling(maps, batch_ids, flat_inds, use_transform=True):
    if use_transform:
        batch, channel = maps.shape[:2]
        maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
    results = maps[batch_ids, flat_inds].contiguous()
    return results


def parsing_outputs(center_maps, params_maps, centermap_parser):
    center_preds_info = centermap_parser.parse_centermap(center_maps)
    batch_ids, flat_inds, cyxs, center_confs = center_preds_info
    if len(batch_ids) == 0:
        None
        return None
    params_pred = parameter_sampling(params_maps, batch_ids, flat_inds, use_transform=True)
    parsed_results = pack_params_dict(params_pred)
    parsed_results['center_preds'] = torch.stack([flat_inds % 64, flat_inds // 64], 1) * 512 // 64
    parsed_results['center_confs'] = parameter_sampling(center_maps, batch_ids, flat_inds, use_transform=True)
    return parsed_results


class SMPLR(nn.Module):

    def __init__(self, use_gender=False):
        super(SMPLR, self).__init__()
        model_path = os.path.join(config.model_dir, 'parameters', 'smpl')
        self.smpls = {}
        self.smpls['n'] = SMPL(args().smpl_model_path, model_type='smpl')
        if use_gender:
            self.smpls['f'] = SMPL(os.path.join(config.smpl_model_dir, 'SMPL_FEMALE.pth'))
            self.smpls['m'] = SMPL(os.path.join(config.smpl_model_dir, 'SMPL_MALE.pth'))

    def forward(self, pose, betas, gender='n'):
        if isinstance(pose, np.ndarray):
            pose, betas = torch.from_numpy(pose).float(), torch.from_numpy(betas).float()
        if len(pose.shape) == 1:
            pose, betas = pose.unsqueeze(0), betas.unsqueeze(0)
        verts, joints54_17 = self.smpls[gender](poses=pose, betas=betas)
        return verts.numpy(), joints54_17[:, :54].numpy()


class MeshRendererWithDepth(nn.Module):

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) ->torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class Renderer(object):

    def __init__(self, focal_length=600, height=512, width=512, **kwargs):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.camera_center = np.array([width / 2.0, height / 2.0])
        self.focal_length = focal_length
        self.colors = [(0.7, 0.7, 0.6, 1.0), (0.7, 0.5, 0.5, 1.0), (0.5, 0.5, 0.7, 1.0), (0.5, 0.55, 0.3, 1.0), (0.3, 0.5, 0.55, 1.0)]

    def __call__(self, verts, faces, colors=None, focal_length=None, camera_pose=None, **kwargs):
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        num_people = verts.shape[0]
        verts = verts.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        if camera_pose is None:
            camera_pose = np.eye(4)
        if focal_length is None:
            fx, fy = self.focal_length, self.focal_length
        else:
            fx, fy = focal_length, focal_length
        camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], faces[n])
            mesh.apply_transform(rot)
            trans = np.array([0, 0, 0])
            if colors is None:
                mesh_color = self.colors[0]
            else:
                mesh_color = colors[n % len(colors)]
            material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')
            light_pose = np.eye(4)
            light_pose[:3, 3] = np.array([0, -1, 1]) + trans
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([0, 1, 1]) + trans
            scene.add(light, pose=light_pose)
            light_pose[:3, 3] = np.array([1, 1, 2]) + trans
            scene.add(light, pose=light_pose)
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color

    def delete(self):
        self.renderer.delete()


def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets


def justify_detection_state(detection_flag, reorganize_idx):
    if detection_flag.sum() == 0:
        detection_flag = False
    else:
        reorganize_idx = reorganize_idx[detection_flag.bool()].long()
        detection_flag = True
    return detection_flag, reorganize_idx


def reorganize_items(items, reorganize_idx):
    items_new = [[] for _ in range(len(items))]
    for idx, item in enumerate(items):
        for ridx in reorganize_idx:
            items_new[idx].append(item[ridx])
    return items_new


class Predictor(Base):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self._build_model_()
        self._prepare_modules_()
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}
        if self.character == 'nvxia':
            assert os.path.exists(os.path.join('model_data', 'characters', 'nvxia')), 'Current released version does not support other characters, like Nvxia.'
            self.character_model = create_nvxia_model(self.nvxia_model_path)

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg)
        else:
            outputs = self.model(meta_data, **cfg)
        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')

    def __initialize__(self):
        if self.save_mesh:
            self.smpl_faces = torch.load(args().smpl_model_path)['f'].numpy()
        None

    def single_image_forward(self, image):
        meta_data = img_preprocess(image, '0', input_size=args().input_size, single_img_input=True)
        if '-1' not in self.gpu:
            meta_data['image'] = meta_data['image']
        outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
        return outputs

    def reorganize_results(self, outputs, img_paths, reorganize_idx):
        results = {}
        cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
        trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
        smpl_pose_results = outputs['params']['poses'].detach().cpu().numpy().astype(np.float16)
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy().astype(np.float16)
        joints_54 = outputs['j3d'].detach().cpu().numpy().astype(np.float16)
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy().astype(np.float16)
        kp3d_spin24_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:, constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)
        pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)
        pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
        center_confs = outputs['centers_conf'].detach().cpu().numpy().astype(np.float16)
        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx == vid)[0]
            img_path = img_paths[verts_vids[0]]
            results[img_path] = [{} for idx in range(len(verts_vids))]
            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
                results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
                results[img_path][subject_idx]['poses'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
                results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
        return results


class Time_counter:

    def __init__(self, thresh=0.1):
        self.thresh = thresh
        self.runtime = 0
        self.frame_num = 0

    def start(self):
        self.start_time = time.time()

    def count(self, frame_num=1):
        time_cost = time.time() - self.start_time
        if time_cost < self.thresh:
            self.runtime += time_cost
            self.frame_num += frame_num
        self.start()

    def fps(self):
        None
        None

    def reset(self):
        self.runtime = 0
        self.frame_num = 0


def collect_image_list(image_folder=None, collect_subdirs=False, img_exts=None):

    def collect_image_from_subfolders(image_folder, file_list, collect_subdirs, img_exts):
        for path in glob.glob(os.path.join(image_folder, '*')):
            if os.path.isdir(path) and collect_subdirs:
                collect_image_from_subfolders(path, file_list, collect_subdirs, img_exts)
            elif os.path.splitext(path)[1] in img_exts:
                file_list.append(path)
        return file_list
    file_list = collect_image_from_subfolders(image_folder, [], collect_subdirs, img_exts)
    return file_list


def save_obj(verts, faces, obj_mesh_name='mesh.obj'):
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx == vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4', '').replace('.jpg', '').replace('.png', '') + '.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), smpl_faces, obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))


def save_result_dict_tonpz(results, test_save_dir):
    for img_path, result_dict in results.items():
        if platform.system() == 'Windows':
            path_list = img_path.split('\\')
        else:
            path_list = img_path.split('/')
        file_name = '_'.join(path_list)
        file_name = '_'.join(os.path.splitext(file_name)).replace('.', '') + '.npz'
        save_path = os.path.join(test_save_dir, file_name)
        np.savez(save_path, results=result_dict)


class Image_processor(Predictor):

    def __init__(self, **kwargs):
        super(Image_processor, self).__init__(**kwargs)
        self.__initialize__()

    @torch.no_grad()
    def run(self, image_folder, tracker=None):
        None
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = self.output_dir
        counter = Time_counter(thresh=1)
        if self.show_mesh_stand_on_image:
            visualizer = Vedo_visualizer()
            stand_on_imgs_frames = []
        file_list = collect_image_list(image_folder=image_folder, collect_subdirs=self.collect_subdirs, img_exts=constants.img_exts)
        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, file_list=file_list, shuffle=False)
        counter.start()
        results_all = {}
        for test_iter, meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)
            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=show_items_list, vis_cfg={'settings': ['put_org']}, save2html=False)
                for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['mesh_rendering_orgimgs']['figs']):
                    save_name = os.path.join(self.output_dir, os.path.basename(img_name))
                    cv2.imwrite(save_name, cv2.cvtColor(mesh_rendering_orgimg, cv2.COLOR_RGB2BGR))
            if self.show_mesh_stand_on_image:
                stand_on_imgs = visualizer.plot_multi_meshes_batch(outputs['verts'], outputs['params']['cam'], outputs['meta_data'], outputs['reorganize_idx'].cpu().numpy(), interactive_show=self.interactive_vis)
                stand_on_imgs_frames += stand_on_imgs
            if self.save_mesh:
                save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
            if test_iter % 8 == 0:
                None
            counter.start()
            results_all.update(results)
        return results_all


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self.ndim = ndim
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [(2 * self._std_weight_position * measurement[self.ndim - 1]) for _ in range(self.ndim)] + [(10 * self._std_weight_velocity * measurement[self.ndim - 1]) for _ in range(self.ndim)]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [(self._std_weight_position * mean[self.ndim - 1]) for _ in range(self.ndim)]
        std_vel = [(self._std_weight_velocity * mean[self.ndim - 1]) for _ in range(self.ndim)]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [(self._std_weight_position * mean[self.ndim - 1]) for _ in range(self.ndim)]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [(self._std_weight_position * mean[:, self.ndim - 1]) for _ in range(self.ndim)]
        std_vel = [(self._std_weight_velocity * mean[:, self.ndim - 1]) for _ in range(self.ndim)]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:self.ndim - 1], covariance[:self.ndim - 1, :self.ndim - 1]
            measurements = measurements[:, :self.ndim - 1]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = np.inf, np.inf

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, trans_uv, body_pose, conf, buffer_size=30):
        self._centers = np.asarray(trans_uv, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = conf
        self.tracklet_len = 0
        self.smooth_feat = None
        self.update_poses(body_pose)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_poses(self, body_pose):
        self.body_pose = body_pose
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        """

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][5] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.center)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.center)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_center = new_track.center
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_center)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def center(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return np.concatenate([self._centers.copy(), self.score])
        ret = self.mean[:4].copy()
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


class Tracker(object):

    def __init__(self, seq_name='test', frame_rate=30):
        self.seq_name = seq_name
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = 15
        self.kalman_filter = KalmanFilter()
        self.deal_with_unconfirm = False

    def post_process(self, outputs):
        cams = [result['cam'] for result in outputs]
        kp3ds = [result['j3d_all54'] for result in outputs]
        center_conf = [result['center_conf'] for result in outputs]
        return cams, kp3ds, center_conf

    def update(self, outputs):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        cams, kp3ds, center_conf = self.post_process(outputs)
        if len(cams) > 0:
            """Detections"""
            detections = [STrack(trans_suv, body_pose, conf, 30) for trans_suv, body_pose, conf in zip(cams, kp3ds, center_conf)]
            tracked_ids = np.zeros(len(detections))
        else:
            detections = []
            tracked_ids = np.array([0])
        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        """ Step 2: First association, with embedding"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.gate_cost_matrix(self.kalman_filter, strack_pool, detections, only_position=True)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            tracked_ids[idet] = track.track_id
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in r_tracked_stracks:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        if self.deal_with_unconfirm:
            """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
            detections = [detections[i] for i in u_detection]
            dists = matching.gate_cost_matrix(self.kalman_filter, unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            tracked_ids[inew] = track.track_id
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        online_centers = []
        online_ids = []
        for t in output_stracks:
            online_centers.append(t.center)
            online_ids.append(t.track_id)
        return online_ids


def frames2video(images_path, video_name, fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)
    for path in images_path:
        image = imageio.imread(path)
        writer.append_data(image)
    writer.close()


def temporal_optimize_result(result, filter_dict):
    result['cam'] = filter_dict['cam'].process(result['cam'])
    result['betas'] = filter_dict['betas'].process(result['betas'])
    pose_euler = np.array([transform_rot_representation(vec, input_type='vec', out_type='euler') for vec in result['poses'].reshape((-1, 3))])
    body_pose_euler = filter_dict['poses'].process(pose_euler[1:].reshape(-1))
    result['poses'][3:] = np.array([transform_rot_representation(bodypose, input_type='euler', out_type='vec') for bodypose in body_pose_euler.reshape(-1, 3)]).reshape(-1)
    return result


def video2frame(video_path, frame_save_dir=None):
    cap = cv2.VideoCapture(video_path)
    for frame_id in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        success_flag, frame = cap.read()
        if success_flag:
            save_path = os.path.join(frame_save_dir, '{:08d}.jpg'.format(frame_id))
            cv2.imwrite(save_path, frame)


class Webcam_processor(Predictor):

    def __init__(self, **kwargs):
        super(Webcam_processor, self).__init__(**kwargs)
        if self.character == 'nvxia':
            assert os.path.exists(os.path.join('model_data', 'characters', 'nvxia')), 'Current released version does not support other characters, like Nvxia.'
            self.character_model = create_nvxia_model(self.nvxia_model_path)

    def webcam_run_local(self, video_file_path=None):
        """
        24.4 FPS of forward prop. on 1070Ti
        """
        capture = OpenCVCapture(video_file_path, show=False)
        None
        frame_id = 0
        if self.visulize_platform == 'integrated':
            visualizer = Open3d_visualizer(multi_mode=not args().show_largest_person_only)
        elif self.visulize_platform == 'blender':
            sender = SocketClient_blender()
        elif self.visulize_platform == 'vis_server':
            RS = Results_sender()
        if self.make_tracking:
            if args().tracker == 'norfair':
                if args().tracking_target == 'centers':
                    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=80)
                elif args().tracking_target == 'keypoints':
                    tracker = Tracker(distance_function=keypoints_distance, distance_threshold=60)
            else:
                tracker = Tracker()
        if self.temporal_optimization:
            filter_dict = {}
            subjects_motion_sequences = {}
        for i in range(10):
            self.single_image_forward(np.zeros((512, 512, 3)).astype(np.uint8))
        counter = Time_counter(thresh=1)
        while True:
            start_time_perframe = time.time()
            frame = capture.read()
            if frame is None:
                continue
            frame_id += 1
            counter.start()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            counter.count()
            if outputs is not None and outputs['detection_flag']:
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                results = self.reorganize_results(outputs, [frame_id for _ in range(len(reorganize_idx))], reorganize_idx)
                if args().show_largest_person_only or self.visulize_platform == 'blender':
                    max_id = np.argmax(np.array([result['cam'][0] for result in results[frame_id]]))
                    results[frame_id] = [results[frame_id][max_id]]
                    tracked_ids = np.array([0])
                elif args().make_tracking:
                    if args().tracker == 'norfair':
                        if args().tracking_target == 'centers':
                            detections = [Detection(points=result['cam'][[2, 1]] * args().input_size) for result in results[frame_id]]
                        elif args().tracking_target == 'keypoints':
                            detections = [Detection(points=result['pj2d_org']) for result in results[frame_id]]
                        if frame_id == 1:
                            for _ in range(8):
                                tracked_objects = tracker.update(detections=detections)
                        tracked_objects = tracker.update(detections=detections)
                        if len(tracked_objects) == 0:
                            continue
                        tracked_ids = get_tracked_ids(detections, tracked_objects)
                    else:
                        tracked_ids = tracker.update(results[frame_id])
                    if len(tracked_ids) == 0 or len(tracked_ids) > len(results[frame_id]):
                        continue
                else:
                    tracked_ids = np.arange(len(results[frame_id]))
                cv2.imshow('Input', frame[:, :, ::-1])
                cv2.waitKey(1)
                if self.temporal_optimization:
                    for sid, tid in enumerate(tracked_ids):
                        if tid not in filter_dict:
                            filter_dict[tid] = create_OneEuroFilter(args().smooth_coeff)
                            subjects_motion_sequences[tid] = {}
                        results[frame_id][sid] = temporal_optimize_result(results[frame_id][sid], filter_dict[tid])
                        subjects_motion_sequences[tid][frame_id] = results[frame_id][sid]
                cams = np.array([result['cam'] for result in results[frame_id]])
                cams[:, 2] -= 0.26
                trans = np.array([convert_cam_to_3d_trans(cam) for cam in cams])
                poses = np.array([result['poses'] for result in results[frame_id]])
                betas = np.array([result['betas'] for result in results[frame_id]])
                kp3ds = np.array([result['j3d_smpl24'] for result in results[frame_id]])
                verts = np.array([result['verts'] for result in results[frame_id]])
                if self.visulize_platform == 'vis_server':
                    RS.send_results(poses=poses, betas=betas, trans=trans, ids=tracked_ids)
                elif self.visulize_platform == 'blender':
                    sender.send([0, poses[0].tolist(), trans[0].tolist(), frame_id])
                elif self.visulize_platform == 'integrated':
                    if self.character == 'nvxia':
                        verts = self.character_model(poses)['verts'].numpy()
                    if args().show_largest_person_only:
                        trans_largest = trans[0] if self.add_trans else None
                        visualizer.run(verts[0], trans=trans_largest)
                    else:
                        visualizer.run_multiperson(verts, trans=trans, tracked_ids=tracked_ids)

    def webcam_run_remote(self):
        None
        capture = Server_port_receiver()
        while True:
            frame = capture.receive()
            if isinstance(frame, list):
                continue
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            if outputs is not None:
                verts = outputs['verts'][0].cpu().numpy()
                verts = verts * 50 + np.array([0, 0, 100])
                capture.send(verts)
            else:
                capture.send(['failed'])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter_Dict(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dict_store = {}
        self.count = 0

    def update(self, val, n=1):
        for key, value in val.items():
            if key not in self.dict_store:
                self.dict_store[key] = []
            if torch.is_tensor(value):
                value = value.item()
            self.dict_store[key].append(value)
        self.count += n

    def sum(self):
        dict_sum = {}
        for k, v in self.dict_store.items():
            dict_sum[k] = round(float(sum(v)), 2)
        return dict_sum

    def avg(self):
        dict_sum = self.sum()
        dict_avg = {}
        for k, v in dict_sum.items():
            dict_avg[k] = round(v / self.count, 2)
        return dict_avg


def determ_worst_best(VIS_IDX, top_n=2):
    sellected_ids, sellected_errors = [], []
    if VIS_IDX is not None:
        for ds_type in VIS_IDX:
            for error, idx in zip(VIS_IDX[ds_type]['error'], VIS_IDX[ds_type]['idx']):
                if torch.is_tensor(error):
                    error, idx = error.cpu().numpy(), idx.cpu().numpy()
                worst_id = np.argsort(error)[-top_n:]
                sellected_ids.append(idx[worst_id])
                sellected_errors.append(error[worst_id])
                best_id = np.argsort(error)[:top_n]
                sellected_ids.append(idx[best_id])
                sellected_errors.append(error[best_id])
    if len(sellected_ids) > 0 and len(sellected_errors) > 0:
        sellected_ids = np.concatenate(sellected_ids).tolist()
        sellected_errors = np.concatenate(sellected_errors).tolist()
    else:
        sellected_ids, sellected_errors = [0], [0]
    return sellected_ids, sellected_errors


def fix_backbone(params, exclude_key=['backbone.']):
    for exclude_name in exclude_key:
        for index, (name, param) in enumerate(params.named_parameters()):
            if exclude_name in name:
                param.requires_grad = False
    logging.info('Fix params that include in {}'.format(exclude_key))
    return params


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def save_single_model(model, path):
    logging.info('saving {}'.format(path))
    torch.save(model.module.state_dict(), path)


def save_model(model, title, parent_folder=None):
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    if parent_folder is not None:
        title = os.path.join(parent_folder, title)
    save_single_model(model, title)


def train_entire_model(net):
    exclude_layer = []
    for index, (name, param) in enumerate(net.named_parameters()):
        if 'smpl' not in name:
            param.requires_grad = True
        else:
            if param.requires_grad:
                exclude_layer.append(name)
            param.requires_grad = False
    if len(exclude_layer) == 0:
        logging.info('Training all layers.')
    else:
        logging.info('Train all layers, except: {}'.format(exclude_layer))
    return net


def _init_error_dict():
    ED = {}
    ED['MPJPE'], ED['PA_MPJPE'], ED['PCK3D'], ED['imgpaths'] = [{ds: [] for ds in constants.dataset_involved} for _ in range(4)]
    ED['MPJAE'] = {ds: [] for ds in constants.MPJAE_ds}
    ED['PVE_new'] = {ds: [] for ds in constants.PVE_ds}
    ED['PVE'] = {ds: {'target_theta': [], 'pred_theta': []} for ds in constants.PVE_ds}
    ED['ds_bias'] = {ds: {'scale': [], 'trans': []} for ds in constants.dataset_involved}
    ED['root_depth'] = {ds: [] for ds in constants.dataset_depth}
    ED['mPCKh'] = {ds: [] for ds in constants.dataset_kp2ds}
    ED['depth_relative'] = {ds: {'eq': [], 'cd': [], 'fd': [], 'eq_age': [], 'cd_age': [], 'fd_age': []} for ds in constants.dataset_relative_depth + constants.dataset_depth}
    ED['age_relative'] = {ds: {age_name: [] for age_name in constants.relative_age_types} for ds in constants.dataset_relative_age}
    return ED


def _calc_joint_angle_error(pred_mat, gt_mat, return_axis_angle=False):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 9g, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 9, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """
    r1 = pred_mat.reshape(-1, 3, 3)
    r2 = gt_mat.reshape(-1, 3, 3)
    r2t = r2.permute(0, 2, 1)
    r = torch.matmul(r1, r2t)
    axis_angles = rotation_matrix_to_angle_axis(r)
    angles = torch.norm(axis_angles, dim=-1) * (180.0 / np.pi)
    if return_axis_angle:
        return angles, axis_angles
    return angles


def trans_relative_rot_to_global_rotmat(params, with_global_rot=False):
    """
    calculate absolute rotation matrix in the global coordinate frame of K body parts. 
    The rotation is the map from the local bone coordinate frame to the global one.
    K= 9 parts in the following order: 
    root (JOINT 0) , left hip  (JOINT 1), right hip (JOINT 2), left knee (JOINT 4), right knee (JOINT 5), 
    left shoulder (JOINT 16), right shoulder (JOINT 17), left elbow (JOINT 18), right elbow (JOINT 19).
    parent kinetic tree [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    """
    batch_size, param_num = params.shape[0], params.shape[1] // 3
    pose_rotmat = batch_rodrigues(params.reshape(-1, 3)).view(batch_size, param_num, 3, 3).contiguous()
    if with_global_rot:
        sellect_joints = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19], dtype=np.int)
        results = [pose_rotmat[:, 0]]
        for idx in range(param_num - 1):
            i_val = int(idx + 1)
            joint_rot = pose_rotmat[:, i_val]
            parent = constants.kintree_parents[i_val]
            glob_transf_mat = torch.matmul(results[parent], joint_rot)
            results.append(glob_transf_mat)
    else:
        sellect_joints = np.array([1, 2, 4, 5, 16, 17, 18, 19], dtype=np.int) - 1
        results = [torch.eye(3, 3)[None].repeat(batch_size, 1, 1)]
        for i_val in range(param_num - 1):
            joint_rot = pose_rotmat[:, i_val]
            parent = constants.kintree_parents[i_val + 1]
            glob_transf_mat = torch.matmul(results[parent], joint_rot)
            results.append(glob_transf_mat)
    global_rotmat = torch.stack(results, axis=1)[:, sellect_joints].contiguous()
    return global_rotmat


def _calc_MPJAE(rel_pose_pred, rel_pose_real):
    global_pose_rotmat_pred = trans_relative_rot_to_global_rotmat(rel_pose_pred, with_global_rot=True)
    global_pose_rotmat_real = trans_relative_rot_to_global_rotmat(rel_pose_real, with_global_rot=True)
    MPJAE_error = _calc_joint_angle_error(global_pose_rotmat_pred, global_pose_rotmat_real).cpu().numpy()
    return MPJAE_error


def _calc_relative_age_error_weak_(age_preds, age_gts, matched_mask=None):
    valid_mask = age_gts != -1
    if matched_mask is not None:
        valid_mask *= matched_mask
    error_dict = {age_name: [] for age_name in constants.relative_age_types}
    if valid_mask.sum() > 0:
        for age_id, age_name in enumerate(constants.relative_age_types):
            age_gt = age_gts[valid_mask].long() == age_id
            age_pred = age_preds[valid_mask][age_gt].long()
            error_dict.update({age_name: [age_pred]})
    return error_dict


relative_age_types = ['adult', 'teen', 'kid', 'baby']


def _calc_relative_depth_error_weak_(pred_depths, depth_ids, reorganize_idx, age_gts=None, matched_mask=None):
    depth_ids = depth_ids
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    age_gts = age_gts[depth_ids_vmask]
    error_dict = {'eq': [], 'cd': [], 'fd': [], 'eq_age': [], 'cd_age': [], 'fd_age': []}
    error_each_age = {age_type: [] for age_type in relative_age_types}
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds *= matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1, did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1, did_num))[triu_mask]
            error_dict['eq'].append(dist_mat[did_mat == 0])
            error_dict['cd'].append(dist_mat[did_mat < 0])
            error_dict['fd'].append(dist_mat[did_mat > 0])
            if age_gts is not None:
                age_sample = age_gts[sample_inds]
                age_mat = torch.cat([age_sample.unsqueeze(0).repeat(did_num, 1).unsqueeze(-1), age_sample.unsqueeze(1).repeat(1, did_num).unsqueeze(-1)], -1)[triu_mask]
                error_dict['eq_age'].append(age_mat[did_mat == 0])
                error_dict['cd_age'].append(age_mat[did_mat < 0])
                error_dict['fd_age'].append(age_mat[did_mat > 0])
    return error_dict


def _calc_relative_depth_error_withgts_(pred_depths, depth_gts, reorganize_idx, age_gts=None, thresh=0.3, matched_mask=None):
    depth_gts = depth_gts
    error_dict = {'eq': [], 'cd': [], 'fd': [], 'eq_age': [], 'cd_age': [], 'fd_age': []}
    for b_ind in torch.unique(reorganize_idx):
        sample_inds = reorganize_idx == b_ind
        if matched_mask is not None:
            sample_inds *= matched_mask
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1, did_num))[triu_mask]
            dist_mat_gt = (depth_gts[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_gts[sample_inds].unsqueeze(1).repeat(1, did_num))[triu_mask]
            error_dict['eq'].append(dist_mat[torch.abs(dist_mat_gt) < thresh])
            error_dict['cd'].append(dist_mat[dist_mat_gt < -thresh])
            error_dict['fd'].append(dist_mat[dist_mat_gt > thresh])
            if age_gts is not None:
                age_sample = age_gts[sample_inds]
                age_mat = torch.cat([age_sample.unsqueeze(0).repeat(did_num, 1).unsqueeze(-1), age_sample.unsqueeze(1).repeat(1, did_num).unsqueeze(-1)], -1)[triu_mask]
                error_dict['eq_age'].append(age_mat[torch.abs(dist_mat_gt) < thresh])
                error_dict['cd_age'].append(age_mat[dist_mat_gt < -thresh])
                error_dict['fd_age'].append(age_mat[dist_mat_gt > thresh])
    return error_dict


def calc_pck(real, pred, align_inds=None, pck_joints=None):
    vis_mask = real[:, :, 0] != -2.0
    pred_aligned = align_by_parts(pred, align_inds=align_inds)
    real_aligned = align_by_parts(real, align_inds=align_inds)
    mpjpe_pck_batch = compute_mpjpe(pred_aligned, real_aligned, vis_mask, pck_joints=pck_joints)
    return mpjpe_pck_batch


def calc_outputs_evaluation_matrix(self, outputs, ED):
    for ds in set(outputs['meta_data']['data_set']):
        val_idx = np.where(np.array(outputs['meta_data']['data_set']) == ds)[0]
        real_3d = outputs['meta_data']['kp_3d'][val_idx].contiguous()
        if ds in constants.dataset_smpl2lsp:
            real_3d = real_3d[:, self.All54_to_LSP14_mapper].contiguous()
            if (self.All54_to_LSP14_mapper == -1).sum() > 0:
                real_3d[:, self.All54_to_LSP14_mapper == -1] = -2.0
            predicts = outputs['joints_h36m17'][:, :14].contiguous()
            align_inds = [constants.LSP_14['R_Hip'], constants.LSP_14['L_Hip']]
            bones, colors, kp_colors = constants.lsp14_connMat, constants.cm_body14, constants.lsp14_kpcm
        else:
            predicts = outputs['j3d'][val_idx, :24].contiguous()
            real_3d = real_3d[:, :24].contiguous()
            align_inds = [constants.SMPL_24['Pelvis_SMPL']]
            bones, colors, kp_colors = constants.smpl24_connMat, constants.cm_smpl24, constants.smpl24_kpcm
        mPCKh = _calc_matched_PCKh_(outputs['meta_data']['full_kp2d'].float(), outputs['pj2d'].float(), outputs['meta_data']['valid_masks'][:, 0])
        ED['mPCKh'][ds].append(mPCKh)
        matched_mask = mPCKh > args().matching_pckh_thresh
        if ds in constants.dataset_depth:
            predicts_j3ds = outputs['j3d'][val_idx].contiguous().detach().cpu().numpy()
            predicts_pj2ds = outputs['pj2d_org'].detach().cpu().numpy()
            if ds in ['agora', 'mini']:
                predicts_j3ds = predicts_j3ds[:, :24] - predicts_j3ds[:, [0]]
                predicts_pj2ds = predicts_pj2ds[:, :24]
            trans_preds = outputs['cam_trans'].detach().cpu()
            trans_gts = outputs['meta_data']['root_trans']
            ED['root_depth'][ds].append(np.concatenate([trans_preds.numpy()[None], trans_gts.cpu().numpy()[None]]))
            age_gts = outputs['meta_data']['depth_info'][:, 0] if 'depth_info' in outputs['meta_data'] else None
            relative_depth_errors = _calc_relative_depth_error_withgts_(trans_preds[:, 2], trans_gts[:, 2], outputs['reorganize_idx'], age_gts=age_gts, matched_mask=matched_mask)
            for dr_type in constants.relative_depth_types:
                ED['depth_relative'][ds][dr_type] += relative_depth_errors[dr_type]
                ED['depth_relative'][ds][dr_type + '_age'] += relative_depth_errors[dr_type + '_age']
        if ds in ED['depth_relative']:
            age_gts = outputs['meta_data']['depth_info'][:, 0] if 'depth_info' in outputs['meta_data'] else None
            relative_depth_errors = _calc_relative_depth_error_weak_(outputs['cam_trans'][:, 2], outputs['meta_data']['depth_info'][:, 3], outputs['reorganize_idx'], age_gts=age_gts, matched_mask=matched_mask)
            for dr_type in constants.relative_depth_types:
                ED['depth_relative'][ds][dr_type] += relative_depth_errors[dr_type]
                ED['depth_relative'][ds][dr_type + '_age'] += relative_depth_errors[dr_type + '_age']
        if ds in ED['age_relative'] and args().learn_relative:
            relative_age_errors = _calc_relative_age_error_weak_(outputs['Age_preds'], outputs['meta_data']['depth_info'][:, 0], matched_mask=matched_mask)
            for age_type in constants.relative_age_types:
                ED['age_relative'][ds][age_type] += relative_age_errors[age_type]
        if ds not in constants.dataset_nokp3ds:
            if args().calc_PVE_error and ds in constants.PVE_ds:
                batch_PVE = torch.norm(outputs['meta_data']['verts'][val_idx] - outputs['verts'][val_idx], p=2, dim=-1).mean(-1)
                ED['PVE_new'][ds].append(batch_PVE)
            abs_error, aligned_poses = calc_mpjpe(real_3d, predicts, align_inds=align_inds, return_org=True)
            abs_error = abs_error.float().cpu().numpy() * 1000
            rt_error = calc_pampjpe(real_3d, predicts).float().cpu().numpy() * 1000
            kp3d_vis = *aligned_poses, bones
            if self.calc_pck:
                pck_joints_sampled = constants.SMPL_MAJOR_JOINTS if real_3d.shape[1] == 24 else np.arange(12)
                mpjpe_pck_batch = calc_pck(real_3d, predicts, lrhip=lrhip, pck_joints=pck_joints_sampled).cpu().numpy() * 1000
                ED['PCK3D'][ds].append((mpjpe_pck_batch.reshape(-1) < self.PCK_thresh).astype(np.float32) * 100)
                if ds in constants.MPJAE_ds:
                    rel_pose_pred = torch.cat([outputs['params']['global_orient'][val_idx], outputs['params']['body_pose'][val_idx]], 1)[:, :22 * 3].contiguous()
                    rel_pose_real = outputs['meta_data']['params'][val_idx, :22 * 3]
                    MPJAE_error = _calc_MPJAE(rel_pose_pred, rel_pose_real)
                    ED['MPJAE'][ds].append(MPJAE_error)
            ED['MPJPE'][ds].append(abs_error.astype(np.float32))
            ED['PA_MPJPE'][ds].append(rt_error.astype(np.float32))
            ED['imgpaths'][ds].append(np.array(outputs['meta_data']['imgpath'])[val_idx])
        else:
            kp3d_vis = None
    return ED, kp3d_vis


def h36m_evaluation_act_wise(results, imgpaths, action_names):
    actions = []
    action_results = []
    for imgpath in imgpaths:
        actions.append(os.path.basename(imgpath).split('.jpg')[0].split('_')[1].split(' ')[0])
    for action_name in action_names:
        action_idx = np.where(np.array(actions) == action_name)[0]
        action_results.append('{:.2f}'.format(results[action_idx].mean()))
    return action_results


def print_table(eval_results):
    matrix_dict = {}
    em_col_id = 0
    matrix_list = []
    for name in eval_results:
        ds, em = name.split('-')
        if em not in matrix_dict:
            matrix_dict[em] = em_col_id
            matrix_list.append(em)
            em_col_id += 1
    raw_dict = {}
    for name, result in eval_results.items():
        ds, em = name.split('-')
        if ds not in raw_dict:
            raw_dict[ds] = np.zeros(em_col_id).tolist()
        raw_dict[ds][matrix_dict[em]] = '{:.3f}'.format(result)
    table = PrettyTable(['DS/EM'] + matrix_list)
    for idx, (ds, mat_list) in enumerate(raw_dict.items()):
        table.add_row([ds] + mat_list)
    None
    None


def process_matrix(matrix, name, times=1.0):
    eval_results = {}
    for ds, error_list in matrix.items():
        if len(error_list) > 0:
            result = np.concatenate(error_list, axis=0)
            result = result[~np.isnan(result)].mean()
            eval_results['{}-{}'.format(ds, name)] = result * times
    return eval_results


def print_results(ED):
    eval_results = {}
    for key, results in ED['root_depth'].items():
        if len(results) > 0:
            results_all = np.concatenate(results, axis=1)
            axis_error = np.abs(results_all[0] - results_all[1]).mean(0)
            root_error = np.sqrt(np.sum((results_all[0] - results_all[1]) ** 2, axis=1)).mean()
            None
    for ds, results in ED['depth_relative'].items():
        result_length = sum([len(ED['depth_relative'][ds][dr_type]) for dr_type in constants.relative_depth_types])
        if result_length > 0:
            eq_dists = torch.cat(ED['depth_relative'][ds]['eq'], 0)
            cd_dists = torch.cat(ED['depth_relative'][ds]['cd'], 0)
            fd_dists = torch.cat(ED['depth_relative'][ds]['fd'], 0)
            age_flag = len(ED['depth_relative'][ds]['eq_age']) > 0
            if age_flag:
                eq_age_ids = torch.cat(ED['depth_relative'][ds]['eq_age'], 0)
                cd_age_ids = torch.cat(ED['depth_relative'][ds]['cd_age'], 0)
                fd_age_ids = torch.cat(ED['depth_relative'][ds]['fd_age'], 0)
                dr_age_ids = torch.cat([eq_age_ids, cd_age_ids, fd_age_ids], 0)
            dr_all = np.array([len(eq_dists), len(cd_dists), len(fd_dists)])
            for dr_thresh in [0.2]:
                dr_corrects = [torch.abs(eq_dists) < dr_thresh, cd_dists < -dr_thresh, fd_dists > dr_thresh]
                None
                dr_corrects = torch.cat(dr_corrects, 0)
                eval_results['{}-PCRD_{}'.format(ds, dr_thresh)] = dr_corrects.sum() / dr_all.sum()
                if age_flag:
                    for age_ind, age_name in enumerate(constants.relative_age_types):
                        age_mask = (dr_age_ids == age_ind).sum(-1).bool()
                        if age_mask.sum() > 0:
                            eval_results['{}-PCRD_{}_{}'.format(ds, dr_thresh, age_name)] = dr_corrects[age_mask].sum() / age_mask.sum()
    for ds, results in ED['age_relative'].items():
        result_length = sum([len(ED['age_relative'][ds][age_type]) for age_type in constants.relative_age_types])
        if result_length > 0:
            None
            age_error_results = {}
            for age_id, age_type in enumerate(constants.relative_age_types):
                age_pred_ids = torch.cat(ED['age_relative'][ds][age_type], 0)
                age_error_results[age_type] = (age_pred_ids == age_id).float()
                if age_id == 0:
                    near_error_results = (age_pred_ids == 1).float()
                elif age_id == 1:
                    near_error_results = (age_pred_ids == 0).float() + (age_pred_ids == 2).float()
                elif age_id == 2:
                    near_error_results = (age_pred_ids == 1).float() + (age_pred_ids == 3).float()
                elif age_id == 3:
                    near_error_results = (age_pred_ids == 2).float()
                age_error_results[age_type] += near_error_results.float() * 0.667
                eval_results['{}-acc_{}'.format(ds, age_type)] = age_error_results[age_type].sum() / len(age_error_results[age_type])
            age_all_results = torch.cat(list(age_error_results.values()), 0)
            eval_results['{}-age_acc'.format(ds)] = age_all_results.sum() / len(age_all_results)
    for ds, results in ED['mPCKh'].items():
        if len(ED['mPCKh'][ds]) > 0:
            mPCKh = torch.cat(ED['mPCKh'][ds], 0)
            mPCKh = mPCKh[mPCKh != -1]
            for thresh in range(6, 7):
                thresh = thresh / 10.0
                eval_results['{}-mPCKh_{}'.format(ds, thresh)] = (mPCKh >= thresh).sum() / len(mPCKh)
    eval_results.update(process_matrix(ED['MPJPE'], 'MPJPE'))
    eval_results.update(process_matrix(ED['PA_MPJPE'], 'PA_MPJPE'))
    if args().calc_pck:
        eval_results.update(process_matrix(ED['PCK3D'], 'PCK3D'))
    if args().calc_PVE_error:
        for ds_name in constants.PVE_ds:
            if len(ED['PVE_new'][ds_name]) > 0:
                eval_results['{}-PVE'.format(ds_name)] = torch.cat(ED['PVE_new'][ds_name], 0).mean() * 1000
    for ds_name in constants.MPJAE_ds:
        if ds_name in ED['MPJAE']:
            if len(ED['MPJAE'][ds_name]) > 0:
                eval_results['{}-MPJAE'.format(ds_name)] = np.concatenate(ED['MPJAE'][ds_name], axis=0).mean()
    print_table(eval_results)
    if len(ED['MPJPE']['h36m']) > 0:
        None
        PA_MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['PA_MPJPE']['h36m'], axis=0), np.concatenate(np.array(ED['imgpaths']['h36m']), axis=0), constants.h36m_action_names)
        MPJPE_acts = h36m_evaluation_act_wise(np.concatenate(ED['MPJPE']['h36m'], axis=0), np.concatenate(np.array(ED['imgpaths']['h36m']), axis=0), constants.h36m_action_names)
        table = PrettyTable(['Protocol'] + constants.h36m_action_names)
        table.add_row(['1'] + MPJPE_acts)
        table.add_row(['2'] + PA_MPJPE_acts)
        None
    return eval_results


@torch.no_grad()
def val_result(self, loader_val, evaluation=False, vis_results=False):
    eval_model = nn.DataParallel(self.model.module).eval()
    ED = _init_error_dict()
    for iter_num, meta_data in enumerate(loader_val):
        if meta_data is None:
            continue
        meta_data_org = meta_data.copy()
        try:
            outputs = self.network_forward(eval_model, meta_data, self.eval_cfg)
        except:
            continue
        if outputs['detection_flag'].sum() == 0:
            None
            continue
        ED, kp3d_vis = calc_outputs_evaluation_matrix(self, outputs, ED)
        if iter_num % (self.val_batch_size * 2) == 0:
            None
            if not evaluation:
                outputs = self.network_forward(eval_model, meta_data_org, self.val_cfg)
            vis_ids = np.arange(max(min(self.val_batch_size, len(outputs['reorganize_idx'])), 8) // 4), None
            save_name = '{}_{}'.format(self.global_count, iter_num)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            show_items = ['mesh', 'joint_sampler', 'pj2d', 'classify']
            if kp3d_vis is not None:
                show_items.append('j3d')
            self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=show_items, vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir': self.result_img_dir, 'save_name': save_name}, kp3ds=kp3d_vis)
    None
    eval_results = print_results(ED)
    return eval_results


def write2log(log_file, massage):
    with open(log_file, 'a') as f:
        f.write(massage)


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__()
        self._build_model_()
        self._build_optimizer()
        self.set_up_val_loader()
        self._calc_loss = Loss()
        self.loader = self._create_data_loader(train_flag=True)
        self.merge_losses = Learnable_Loss(self.loader.dataset._get_ID_num_())
        self.train_cfg = {'mode': 'matching_gts', 'is_training': True, 'update_data': True, 'calc_loss': True if self.model_return_loss else False, 'with_nms': False, 'with_2d_matching': True, 'new_training': args().new_training}
        logging.info('Initialization of Trainer finished!')

    def train(self):
        init_seeds(self.local_rank, cuda_deterministic=False)
        logging.info('start training')
        self.model.train()
        if self.fix_backbone_training_scratch:
            fix_backbone(self.model, exclude_key=['backbone.'])
        else:
            train_entire_model(self.model)
        for epoch in range(self.epoch):
            if epoch == 1:
                train_entire_model(self.model)
            self.train_epoch(epoch)
        self.summary_writer.close()

    def train_step(self, meta_data):
        self.optimizer.zero_grad()
        outputs = self.network_forward(self.model, meta_data, self.train_cfg)
        if not self.model_return_loss:
            outputs.update(self._calc_loss(outputs))
        loss, outputs = self.merge_losses(outputs, self.train_cfg['new_training'])
        if torch.isnan(loss):
            return outputs, torch.zeros(1)
        if self.model_precision == 'fp16':
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return outputs, loss

    def train_log_visualization(self, outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index):
        losses.update(loss.item())
        losses_dict.update(outputs['loss_dict'])
        if self.global_count % self.print_freq == 0:
            message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(epoch, iter_index + 1, len(self.loader), losses_dict.avg(), data_time=data_time, run_time=run_time, loss=losses, lr=self.optimizer.param_groups[0]['lr'])
            logging.info(message)
            write2log(self.log_file, '%s\n' % message)
            self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
            self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)
            losses.reset()
            losses_dict.reset()
            data_time.reset()
            self.summary_writer.flush()
        if self.global_count % (4 * self.print_freq) == 0 or self.global_count == 50:
            vis_ids, vis_errors = determ_worst_best(outputs['kp_error'], top_n=3)
            save_name = '{}'.format(self.global_count)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            train_vis_dict = self.visualizer.visulize_result(outputs, outputs['meta_data'], show_items=['org_img', 'mesh', 'joint_sampler', 'pj2d', 'centermap'], vis_cfg={'settings': ['save_img'], 'vids': vis_ids, 'save_dir': self.train_img_dir, 'save_name': save_name, 'verrors': [vis_errors], 'error_names': ['E']})

    def train_epoch(self, epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict = AverageMeter_Dict()
        batch_start_time = time.time()
        for iter_index, meta_data in enumerate(self.loader):
            if self.fast_eval_iter == 0:
                self.validation(epoch)
                break
            self.global_count += 1
            if args().new_training:
                if self.global_count == args().new_training_iters:
                    self.train_cfg['new_training'], self.val_cfg['new_training'], self.eval_cfg['new_training'] = False, False, False
            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()
            outputs, loss = self.train_step(meta_data)
            if self.local_rank in [-1, 0]:
                run_time.update(time.time() - run_start_time)
                self.train_log_visualization(outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index)
            if self.global_count % self.test_interval == 0 or self.global_count == self.fast_eval_iter:
                save_model(self.model, '{}_val_cache.pkl'.format(self.tab), parent_folder=self.model_save_dir)
                self.validation(epoch)
            if self.distributed_training:
                torch.distributed.barrier()
            batch_start_time = time.time()
        title = '{}_epoch_{}.pkl'.format(self.tab, epoch)
        save_model(self.model, title, parent_folder=self.model_save_dir)
        self.e_sche.step()

    def validation(self, epoch):
        logging.info('evaluation result on {} iters: '.format(epoch))
        for ds_name, val_loader in self.dataset_val_list.items():
            logging.info('Evaluation on {}'.format(ds_name))
            eval_results = val_result(self, loader_val=val_loader, evaluation=False)
            if ds_name == 'relative':
                if 'relativity-PCRD_0.2' not in eval_results:
                    continue
                PCRD = eval_results['relativity-PCRD_0.2']
                age_baby_acc = eval_results['relativity-acc_baby']
                if PCRD > max(self.evaluation_results_dict['relative']['PCRD']) or age_baby_acc > max(self.evaluation_results_dict['relative']['AGE_baby']):
                    eval_results = val_result(self, loader_val=self.dataset_test_list['relative'], evaluation=True)
                self.evaluation_results_dict['relative']['PCRD'].append(PCRD)
                self.evaluation_results_dict['relative']['AGE_baby'].append(age_baby_acc)
            else:
                MPJPE, PA_MPJPE = eval_results['{}-{}'.format(ds_name, 'MPJPE')], eval_results['{}-{}'.format(ds_name, 'PA_MPJPE')]
                test_flag = False
                if ds_name in self.dataset_test_list:
                    test_flag = True
                    if ds_name in self.val_best_PAMPJPE:
                        if PA_MPJPE < self.val_best_PAMPJPE[ds_name]:
                            self.val_best_PAMPJPE[ds_name] = PA_MPJPE
                        else:
                            test_flag = False
                if test_flag or self.test_interval < 100:
                    eval_results = val_result(self, loader_val=self.dataset_test_list[ds_name], evaluation=True)
                    self.summary_writer.add_scalars('{}-test'.format(ds_name), eval_results, self.global_count)
        title = '{}_{:.4f}_{:.4f}_{}.pkl'.format(epoch, MPJPE, PA_MPJPE, self.tab)
        logging.info('Model saved as {}'.format(title))
        save_model(self.model, title, parent_folder=self.model_save_dir)
        self.model.train()
        self.summary_writer.flush()

    def get_running_results(self, ds):
        mpjpe = np.array(self.evaluation_results_dict[ds]['MPJPE'])
        pampjpe = np.array(self.evaluation_results_dict[ds]['PAMPJPE'])
        mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var = np.mean(mpjpe), np.var(mpjpe), np.mean(pampjpe), np.var(pampjpe)
        return mpjpe_mean, mpjpe_var, pampjpe_mean, pampjpe_var


class Demo(Base):

    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self.test_cfg = {'mode': 'parsing', 'calc_loss': False, 'with_nms': True, 'new_training': args().new_training}
        self.eval_dataset = args().eval_dataset
        self.save_mesh = False
        None

    def test_eval(self):
        if self.eval_dataset == 'pw3d_test':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag=False, mode='vibe', split='test')
        elif self.eval_dataset == 'pw3d_oc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag=False, split='all', mode='OC')
        elif self.eval_dataset == 'pw3d_pc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag=False, split='all', mode='PC')
        elif self.eval_dataset == 'pw3d_nc':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag=False, split='all', mode='NC')
        MPJPE, PA_MPJPE, eval_results = val_result(self, loader_val=data_loader, evaluation=True)

    def net_forward(self, meta_data, mode='val'):
        if mode == 'val':
            cfg_dict = self.test_cfg
        elif mode == 'eval':
            cfg_dict = self.eval_cfg
        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg_dict)
        else:
            outputs = self.model(meta_data, **cfg_dict)
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def test_cmu_panoptic(self):
        action_name = ['haggling', 'mafia', 'ultimatum', 'pizza']
        mpjpe_cacher = {aname: AverageMeter() for aname in action_name}
        J_regressor_h36m = torch.from_numpy(np.load(args().smpl_J_reg_h37m_path)).float()
        data_loader = self._create_single_data_loader(dataset='cmup', train_flag=False, split='test')
        bias = []
        self.model.eval()
        with torch.no_grad():
            for test_iter, meta_data in enumerate(data_loader):
                outputs = self.net_forward(meta_data, mode='eval')
                meta_data = outputs['meta_data']
                pred_vertices = outputs['verts'].float()
                J_regressor_batch = J_regressor_h36m[None, :].expand(pred_vertices.shape[0], -1, -1)
                pred_kp3ds = torch.matmul(J_regressor_batch, pred_vertices)
                gt_kp3ds = meta_data['kp_3d']
                visible_kpts = (gt_kp3ds[:, :, 0] > -2.0).float()
                pred_kp3ds -= pred_kp3ds[:, [0]]
                gt_kp3ds -= gt_kp3ds[:, [0]]
                mpjpes = torch.sqrt(((pred_kp3ds - gt_kp3ds) ** 2).sum(dim=-1)) * visible_kpts * 1000
                mpjpes = mpjpes.mean(-1)
                pampjpes, transform_mat = calc_pampjpe(gt_kp3ds, pred_kp3ds, return_transform_mat=True)
                pampjpes = pampjpes * 1000
                bias.append(transform_mat[2].reshape(-1, 3).mean(0).cpu().numpy())
                for img_path, mpjpe in zip(meta_data['imgpath'], mpjpes):
                    for aname in action_name:
                        if aname in os.path.basename(img_path):
                            mpjpe_cacher[aname].update(float(mpjpe.item()))
                if test_iter % 50 == 0:
                    None
                    None
                    for key, value in mpjpe_cacher.items():
                        None
        None
        None
        None
        avg_all = []
        for key, value in mpjpe_cacher.items():
            None
            avg_all.append(value.avg)
        None

    def test_crowdpose(self, set_name='val'):
        predicted_results = []
        test_save_dir = os.path.join(config.project_dir, 'results_out/results_crowdpose')
        os.makedirs(test_save_dir, exist_ok=True)
        results_json_name = os.path.join(config.project_dir, 'results_out/V{}_crowdpose_{}_{}.json'.format(self.model_version, set_name, self.backbone))
        self.model.eval()
        kp2d_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.Crowdpose_14)
        data_loader = self._create_single_data_loader(dataset='crowdpose', train_flag=False, split=set_name)
        vis_dict = {}
        with torch.no_grad():
            for test_iter, meta_data in enumerate(data_loader):
                outputs = self.net_forward(meta_data, mode='val')
                meta_data = outputs['meta_data']
                pj2ds_onorg = outputs['pj2d_org'][:, kp2d_mapper].detach().contiguous().cpu().numpy()
                for batch_idx, (pj2d_onorg, imgpath) in enumerate(zip(pj2ds_onorg, meta_data['imgpath'])):
                    image_id = int(os.path.basename(imgpath).split('.')[0])
                    keypoints = np.concatenate([pj2d_onorg, np.ones((pj2d_onorg.shape[0], 1))], 1).reshape(-1).tolist()
                    predicted_results.append({'image_id': image_id, 'category_id': 1, 'keypoints': keypoints, 'score': 1})
                    if imgpath not in vis_dict:
                        vis_dict[imgpath] = []
                    vis_dict[imgpath].append(pj2d_onorg)
                if test_iter % 50 == 0:
                    None
        with open(results_json_name, 'w') as f:
            json.dump(predicted_results, f)
        gt_file = os.path.join(args().dataset_rootdir, 'crowdpose/json/crowdpose_{}.json'.format(set_name))
        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(results_json_name)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock_1D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (BasicBlock_3D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicBlock_IBN_a,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 2, 4, 4])], {}),
     True),
    (IBN_a,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (JointsMSELoss,
     lambda: ([], {'use_target_weight': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (L2Prior,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Arthur151_ROMP(_paritybench_base):
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

