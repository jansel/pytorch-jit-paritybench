import sys
_module = sys.modules[__name__]
del sys
base = _module
logger = _module
layer = _module
loss = _module
module = _module
resnet = _module
timer = _module
utils = _module
dir = _module
mano = _module
manopth_demo = _module
manopth_mindemo = _module
webuser = _module
lbs = _module
posemapper = _module
serialization = _module
smpl_handpca_wrapper_HAND_only = _module
verts = _module
manopth = _module
argutils = _module
demo = _module
manolayer = _module
rodrigues_layer = _module
rot6d = _module
rotproj = _module
tensutils = _module
setup = _module
test_demo = _module
occluder = _module
posefix = _module
preprocessing = _module
renderer = _module
smpl = _module
demo = _module
display_utils = _module
setup = _module
smplpytorch = _module
native = _module
pytorch = _module
rodrigues_layer = _module
smpl_layer = _module
tensutils = _module
transforms = _module
vis = _module
CrowdPose = _module
Human36M = _module
MPII = _module
MSCOCO = _module
MuCo = _module
MuPoTs = _module
PW3D = _module
dataset = _module
demo = _module
config = _module
model = _module
test = _module
train = _module
check_crowdidx = _module
convert_simple_to_i2l = _module
match_3dpw_2dpose = _module
match_mupots_2dpose = _module

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


import math


import time


import abc


from torch.utils.data import DataLoader


import torch.optim


import torchvision.transforms as transforms


from torch.nn.parallel.data_parallel import DataParallel


import torch


import torch.nn as nn


from torch.nn import functional as F


import numpy as np


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import model_urls


from matplotlib import pyplot as plt


from torch.nn import Module


from torch.autograd import gradcheck


from torch.autograd import Variable


import warnings


import copy


import scipy.io as sio


import random


from torch.utils.data.dataset import Dataset


import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt


import torch.cuda.amp as amp


import matplotlib as mpl


class GraphConvBlock(nn.Module):

    def __init__(self, adj, dim_in, dim_out):
        super(GraphConvBlock, self).__init__()
        self.adj = adj
        self.vertex_num = adj.shape[0]
        self.fcbn_list = nn.ModuleList([nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

    def forward(self, feat):
        batch_size = feat.shape[0]
        feat = torch.stack([fcbn(feat[:, i, :]) for i, fcbn in enumerate(self.fcbn_list)], 1)
        adj = self.adj[None, :, :].repeat(batch_size, 1, 1)
        feat = torch.bmm(adj, feat)
        out = F.relu(feat)
        return out


class GraphResBlock(nn.Module):

    def __init__(self, adj, dim):
        super(GraphResBlock, self).__init__()
        self.adj = adj
        self.graph_block1 = GraphConvBlock(adj, dim, dim)
        self.graph_block2 = GraphConvBlock(adj, dim, dim)

    def forward(self, feat):
        feat_out = self.graph_block1(feat)
        feat_out = self.graph_block2(feat_out)
        out = feat_out + feat
        return out


class CoordLoss(nn.Module):

    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:, :, 2:] * is_3D[:, None, None].float()
            loss = torch.cat((loss[:, :, :2], loss_z), 2)
        return loss


class ParamLoss(nn.Module):

    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss


class NormalVectorLoss(nn.Module):

    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face)
        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)
        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)
        valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]
        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) * valid_mask
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss


class EdgeLengthLoss(nn.Module):

    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face)
        d1_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
        valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
        valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]
        diff1 = torch.abs(d1_out - d1_gt) * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt) * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt) * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Conv2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and bnrelu_final:
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Pose2Feat(nn.Module):

    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64 + joint_num, 64])

    def forward(self, img_feat, joint_heatmap):
        feat = torch.cat((img_feat, joint_heatmap), 1)
        feat = self.conv(feat)
        return feat


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))


def ready_arguments(fname_or_dict):
    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        dd = fname_or_dict
    backwards_compatibility_replacements(dd)
    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3
    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])
    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if s in dd and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    return dd


def subtract_flat_id(rot_mats):
    id_flat = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], 23)
    results = rot_mats - id_flat
    return results


def th_posemap_axisang(pose_vectors):
    """
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    """
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = rodrigues_layer.batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)
    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False
    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


class ManoLayer(Module):
    __constants__ = ['use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check', 'side', 'center_idx', 'joint_rot_mode']

    def __init__(self, center_idx=None, flat_hand_mean=True, ncomps=6, side='right', mano_root='mano/models', use_pca=True, root_rot_mode='axisang', joint_rot_mode='axisang', robust_rot=False):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()
        self.center_idx = center_idx
        self.robust_rot = robust_rot
        if root_rot_mode == 'axisang':
            self.rot = 3
        else:
            self.rot = 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45
        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')
        smpl_data = ready_arguments(self.mano_path)
        hands_components = smpl_data['hands_components']
        self.smpl_data = smpl_data
        self.register_buffer('th_betas', torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs', torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs', torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer('th_v_template', torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer('th_J_regressor', torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights', torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces', torch.Tensor(smpl_data['f'].astype(np.int32)).long())
        hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == 'axisang':
            self.register_buffer('th_hands_mean', th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer('th_selected_comps', torch.Tensor(selected_components))
        else:
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer('th_hands_mean_rotmat', th_hands_mean_rotmat)
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self, th_pose_coeffs, th_betas=torch.zeros(1), th_trans=torch.zeros(1), root_palm=torch.Tensor([0]), share_betas=torch.Tensor([0])):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """
        batch_size = th_pose_coeffs.shape[0]
        if self.use_pca or self.joint_rot_mode == 'axisang':
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot + self.ncomps]
            if self.use_pca:
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs
            th_full_pose = torch.cat([th_pose_coeffs[:, :self.rot], self.th_hands_mean + th_full_hand_pose], 1)
            if self.root_rot_mode == 'axisang':
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, 'When not self.use_pca, th_pose_coeffs should have 4 dims, got {}'.format(th_pose_coeffs.dim())
            assert th_pose_coeffs.shape[2:4] == (3, 3), 'When not self.use_pca, th_pose_coeffs have 3x3 matrix for twolast dims, got {}'.format(th_pose_coeffs.shape[2:4])
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1)
        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
        th_v_posed = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))
        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [(idx - 1) for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [(idx - 1) for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [(idx - 1) for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))
        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results
        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)
        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))
        th_rest_shape_h = torch.cat([th_v_posed.transpose(2, 1), torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device)], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)
        th_verts = th_verts * 1000
        th_jtr = th_jtr * 1000
        return th_verts, th_jtr


class MANO(object):

    def __init__(self):
        self.layer = self.get_layer()
        self.vertex_num = 778
        self.face = self.layer.th_faces.numpy()
        self.joint_regressor = self.layer.th_J_regressor.numpy()
        self.joint_num = 21
        self.joints_name = 'Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinly_4'
        self.skeleton = (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
        self.root_joint_idx = self.joints_name.index('Wrist')
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673]
        thumbtip_onehot = np.array([(1 if i == 745 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        indextip_onehot = np.array([(1 if i == 317 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        middletip_onehot = np.array([(1 if i == 445 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        ringtip_onehot = np.array([(1 if i == 556 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        pinkytip_onehot = np.array([(1 if i == 673 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :]

    def get_layer(self):
        return ManoLayer(mano_root=osp.join(cfg.mano_path, 'mano', 'models'), flat_hand_mean=False, use_pca=False)


def make_list(tensor):
    return tensor


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


class SMPL_Layer(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self, center_idx=None, gender='neutral', model_root='smpl/native/models'):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super().__init__()
        self.center_idx = center_idx
        self.gender = gender
        if gender == 'neutral':
            self.model_path = os.path.join(model_root, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'female':
            self.model_path = os.path.join(model_root, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        elif gender == 'male':
            self.model_path = os.path.join(model_root, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')
        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data
        self.register_buffer('th_betas', torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs', torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs', torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer('th_v_template', torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer('th_J_regressor', torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights', torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces', torch.Tensor(smpl_data['f'].astype(np.int32)).long())
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)

    def forward(self, th_pose_axisang, th_betas=torch.zeros(1), th_trans=torch.zeros(1)):
        """
        Args:
        th_pose_axisang (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        """
        batch_size = th_pose_axisang.shape[0]
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)
        if th_betas is None or bool(torch.norm(th_betas) == 0):
            th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(batch_size, 1, 1)
        else:
            th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
        th_v_posed = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        th_results = []
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val * 9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results
        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints), dtype=root_j.dtype, device=root_j.device)
        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat([th_j[:, i], padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)
        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))
        th_rest_shape_h = torch.cat([th_v_posed.transpose(2, 1), torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device)], 1)
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]
        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)
        return th_verts, th_jtr


def build_adj(vertex_num, skeleton, flip_pairs):
    adj_matrix = np.zeros((vertex_num, vertex_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for pair in flip_pairs:
        adj_matrix[pair] = 1
        adj_matrix[pair[1], pair[0]] = 1
    return adj_matrix


def normalize_adj(adj):
    vertex_num = adj.shape[0]
    adj_self = adj + np.eye(vertex_num)
    D = np.diag(adj_self.sum(0)) + np.spacing(np.array(0))
    _D = 1 / np.sqrt(D)
    _D = _D * np.eye(vertex_num)
    normalized_adj = np.dot(np.dot(_D, adj_self), _D)
    return normalized_adj


class SMPL(object):

    def __init__(self):
        self.layer = {'neutral': self.get_layer(), 'male': self.get_layer('male'), 'female': self.get_layer('female')}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].th_faces.numpy()
        self.joint_regressor = self.layer['neutral'].th_J_regressor.numpy()
        self.shape_param_dim = 10
        self.vposer_code_dim = 32
        self.face_kps_vertex = 331, 2802, 6262, 3489, 3990
        nose_onehot = np.array([(1 if i == 331 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        left_eye_onehot = np.array([(1 if i == 2802 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        right_eye_onehot = np.array([(1 if i == 6262 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        left_ear_onehot = np.array([(1 if i == 3489 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        right_ear_onehot = np.array([(1 if i == 3990 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.joint_regressor = np.concatenate((self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))
        self.joint_regressor_extra = np.load(osp.join('..', 'data', 'J_regressor_extra.npy'))
        self.joint_regressor = np.concatenate((self.joint_regressor, self.joint_regressor_extra[3:4, :])).astype(np.float32)
        self.orig_joint_num = 24
        self.joint_num = 30
        self.joints_name = 'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_top'
        self.flip_pairs = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.skeleton = (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26), (25, 27), (26, 28), (24, 29)
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.graph_joint_num = 15
        self.graph_joints_name = 'Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'Head_top', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
        self.graph_flip_pairs = (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14)
        self.graph_skeleton = (0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (4, 6), (0, 7), (7, 8), (7, 9), (9, 11), (11, 13), (7, 10), (10, 12), (12, 14)
        self.graph_adj = self.get_graph_adj()

    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.graph_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:, idx, :])
        new_joint = torch.stack(new_joint, 1)
        return new_joint

    def get_graph_adj(self):
        adj_mat = build_adj(self.graph_joint_num, self.graph_skeleton, self.graph_flip_pairs)
        normalized_adj = normalize_adj(adj_mat)
        return normalized_adj

    def get_layer(self, gender='neutral'):
        return SMPL_Layer(gender=gender, model_root=cfg.smpl_path + '/smplpytorch/native/models')


class PositionNet(nn.Module):

    def __init__(self):
        super(PositionNet, self).__init__()
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num
        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]
        self.conv = make_conv_layers([2048, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_3d(self, heatmap3d):
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2]))
        accu_x = heatmap3d.sum(dim=(2, 3))
        accu_y = heatmap3d.sum(dim=(2, 4))
        accu_z = heatmap3d.sum(dim=(3, 4))
        accu_x = accu_x * torch.arange(self.hm_shape[2]).float()[None, None, :]
        accu_y = accu_y * torch.arange(self.hm_shape[1]).float()[None, None, :]
        accu_z = accu_z * torch.arange(self.hm_shape[0]).float()[None, None, :]
        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)
        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def forward(self, img_feat):
        joint_heatmap = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = self.soft_argmax_3d(joint_heatmap)
        scores = []
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2])
        joint_heatmap = F.softmax(joint_heatmap, 2)
        joint_heatmap = joint_heatmap.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        for j in range(self.joint_num):
            x = joint_coord[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            z = joint_coord[:, j, 2] / (self.hm_shape[0] - 1) * 2 - 1
            grid = torch.stack((x, y, z), 1)[:, None, None, None, :]
            score_j = F.grid_sample(joint_heatmap[:, j, None, :, :, :], grid, align_corners=True)[:, 0, 0, 0, 0]
            scores.append(score_j)
        scores = torch.stack(scores)
        joint_score = scores.permute(1, 0)[:, :, None]
        return joint_coord, joint_score


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class RotationNet(nn.Module):

    def __init__(self):
        super(RotationNet, self).__init__()
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.joint_num = self.human_model.graph_joint_num
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()
        else:
            self.human_model = SMPL()
            self.joint_num = self.human_model.graph_joint_num
            self.graph_adj = torch.from_numpy(self.human_model.graph_adj).float()
        self.graph_block = nn.Sequential(*[GraphConvBlock(self.graph_adj, 2048 + 4, 128), GraphResBlock(self.graph_adj, 128), GraphResBlock(self.graph_adj, 128), GraphResBlock(self.graph_adj, 128), GraphResBlock(self.graph_adj, 128)])
        self.hm_shape = [cfg.output_hm_shape[0] // 8, cfg.output_hm_shape[1] // 8, cfg.output_hm_shape[2] // 8]
        self.root_pose_out = make_linear_layers([self.joint_num * 128, 6], relu_final=False)
        self.pose_out = make_linear_layers([self.joint_num * 128, self.human_model.vposer_code_dim], relu_final=False)
        self.shape_out = make_linear_layers([self.joint_num * 128, self.human_model.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([self.joint_num * 128, 3], relu_final=False)

    def sample_image_feature(self, img_feat, joint_coord_img):
        img_feat_joints = []
        for j in range(self.joint_num):
            x = joint_coord_img[:, j, 0] / (self.hm_shape[2] - 1) * 2 - 1
            y = joint_coord_img[:, j, 1] / (self.hm_shape[1] - 1) * 2 - 1
            grid = torch.stack((x, y), 1)[:, None, None, :]
            img_feat = img_feat.float()
            img_feat_j = F.grid_sample(img_feat, grid, align_corners=True)[:, :, 0, 0]
            img_feat_joints.append(img_feat_j)
        img_feat_joints = torch.stack(img_feat_joints)
        img_feat_joints = img_feat_joints.permute(1, 0, 2)
        return img_feat_joints

    def forward(self, img_feat, joint_coord_img, joint_score):
        img_feat_joints = self.sample_image_feature(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img, joint_score), 2)
        feat = self.graph_block(feat)
        root_pose = self.root_pose_out(feat.view(-1, self.joint_num * 128))
        pose_param = self.pose_out(feat.view(-1, self.joint_num * 128))
        shape_param = self.shape_out(feat.view(-1, self.joint_num * 128))
        cam_param = self.cam_out(feat.view(-1, self.joint_num * 128))
        return root_pose, pose_param, shape_param, cam_param


class Vposer(nn.Module):

    def __init__(self):
        super(Vposer, self).__init__()
        self.vposer, _ = load_vposer(osp.join(cfg.human_model_path, 'smpl', 'VPOSER_CKPT'), vp_model='snapshot')
        self.vposer.eval()

    def forward(self, z):
        batch_size = z.shape[0]
        body_pose = self.vposer.decode(z, output_type='aa').view(batch_size, -1).view(-1, 24 - 3, 3)
        zero_pose = torch.zeros((batch_size, 1, 3)).float()
        body_pose = torch.cat((body_pose, zero_pose, zero_pose), 1)
        body_pose = body_pose.view(batch_size, -1)
        return body_pose


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
        resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'), (34): (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'), (50): (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'), (101): (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'), (152): (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, skip_early=False):
        if not skip_early:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)
    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).float()], 2)
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


class Model(nn.Module):

    def __init__(self, backbone, pose2feat, position_net, rotation_net, vposer):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral']
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param, meta_info, is_render):
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (cfg.input_img_shape[0] * cfg.input_img_shape[1]))]).view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(cfg.input_img_shape[0] * cfg.input_img_shape[1]) / (bbox[:, 2] * bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].float()
        yy = yy[None, None, :, :].float()
        x = joint_coord_img[:, :, 0, None, None]
        y = joint_coord_img[:, :, 1, None, None]
        heatmap = torch.exp(-((xx - x) / cfg.sigma) ** 2 / 2 - ((yy - y) / cfg.sigma) ** 2 / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor)[None, :, :].repeat(batch_size, 1, 1), mesh_cam)
        root_joint_idx = self.human_model.root_joint_idx
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 0.0001) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 0.0001) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)
        mesh_cam_render = mesh_cam.clone()
        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def forward(self, inputs, targets, meta_info, mode):
        early_img_feat = self.backbone(inputs['img'])
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:, :, :, None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)
        joint_img, joint_score = self.position_net(pose_guided_img_feat)
        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
        root_pose = rot6d_to_axis_angle(root_pose_6d)
        pose_param = self.vposer(z)
        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=cfg.render and mode == 'test')
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)
        if mode == 'train':
            loss = {}
            loss['body_joint_img'] = 1 / 8 * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']), meta_info['is_3D'])
            loss['smpl_joint_img'] = 1 / 8 * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(targets['fit_joint_img']), self.human_model.reduce_joint_set(meta_info['fit_joint_trunc']) * meta_info['is_valid_fit'][:, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'], meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = 1 / 8 * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2], meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])
            return loss
        else:
            out = {'cam_param': cam_param}
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param
            out['smpl_shape'] = shape_param
            out['mesh_cam_render'] = mesh_cam_render
            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CoordLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GraphConvBlock,
     lambda: ([], {'adj': torch.rand([4, 4]), 'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GraphResBlock,
     lambda: ([], {'adj': torch.rand([4, 4]), 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ParamLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hongsukchoi_3DCrowdNet_RELEASE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

