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
human_models = _module
preprocessing = _module
demo = _module
demo_layers = _module
vis_flame_vertices = _module
vis_mano_vertices = _module
setup = _module
smplx = _module
body_models = _module
joint_names = _module
lbs = _module
utils = _module
vertex_ids = _module
vertex_joint_selector = _module
tools = _module
clean_ch = _module
merge_smplh_mano = _module
transforms = _module
vis = _module
vis_from_different_view = _module
AGORA = _module
EHF = _module
Human36M = _module
MPII = _module
MPI_INF_3DHP = _module
MSCOCO = _module
PW3D = _module
dataset = _module
demo = _module
config = _module
model = _module
test = _module
train = _module
affine_transform = _module
agora2coco = _module
tensor_to_numpy_smpl_parameter = _module
merge_hand_to_all = _module
reset_epoch = _module

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


import torchvision


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import model_urls


import numpy as np


import random


from typing import Optional


from typing import Dict


from typing import Union


from typing import Tuple


from typing import List


import torch.nn.functional as F


from typing import NewType


import scipy


import copy


from torch.utils.data.dataset import Dataset


import torch.backends.cudnn as cudnn


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


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, out, gt_index):
        loss = self.ce_loss(out, gt_index)
        return loss


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Conv2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and bnrelu_final:
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


Array = NewType('Array', np.ndarray)


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array: Union[Array, Tensor], dtype=torch.float32) ->Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None, use_hands=True, use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()
        extra_joints_idxs = []
        face_keyp_idxs = np.array([vertex_ids['nose'], vertex_ids['reye'], vertex_ids['leye'], vertex_ids['rear'], vertex_ids['lear']], dtype=np.int64)
        extra_joints_idxs = np.concatenate([extra_joints_idxs, face_keyp_idxs])
        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'], vertex_ids['LSmallToe'], vertex_ids['LHeel'], vertex_ids['RBigToe'], vertex_ids['RSmallToe'], vertex_ids['RHeel']], dtype=np.int32)
            extra_joints_idxs = np.concatenate([extra_joints_idxs, feet_keyp_idxs])
        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])
            extra_joints_idxs = np.concatenate([extra_joints_idxs, tips_idxs])
        self.register_buffer('extra_joints_idxs', to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)
        return joints


def transform_mat(R: Tensor, t: Tensor) ->Tensor:
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats: Tensor, joints: Tensor, parents: Tensor, dtype=torch.float32) ->Tensor:
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


def batch_rodrigues(rot_vecs: Tensor, epsilon: float=1e-08) ->Tensor:
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-08, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def blend_shapes(betas: Tensor, shape_disps: Tensor) ->Tensor:
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor: Tensor, vertices: Tensor) ->Tensor:
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(betas: Tensor, pose: Tensor, v_template: Tensor, shapedirs: Tensor, posedirs: Tensor, J_regressor: Tensor, parents: Tensor, lbs_weights: Tensor, pose2rot: bool=True) ->Tuple[Tensor, Tensor]:
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
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(vertices: Tensor, pose: Tensor, dynamic_lmk_faces_idx: Tensor, dynamic_lmk_b_coords: Tensor, neck_kin_chain: List[int], pose2rot: bool=True) ->Tuple[Tensor, Tensor]:
    """ Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    """
    dtype = vertices.dtype
    batch_size = vertices.shape[0]
    if pose2rot:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)
    rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
    y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39))
    neg_mask = y_rot_angle.lt(0)
    mask = y_rot_angle.lt(-39)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def vertices2landmarks(vertices: Tensor, faces: Tensor, lmk_faces_idx: Tensor, lmk_bary_coords: Tensor) ->Tensor:
    """ Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)
    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))
    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))
    accu_x = accu_x * torch.arange(width).float()[None, None, :]
    accu_y = accu_y * torch.arange(height).float()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float()[None, None, :]
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


class PositionNet(nn.Module):

    def __init__(self, part, resnet_type):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
            self.hm_shape = cfg.output_hand_hm_shape
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048
        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]), 2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        return joint_hm, joint_coord


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]
    img_feat = img_feat.permute(0, 2, 1).contiguous()
    return img_feat


class RotationNet(nn.Module):

    def __init__(self, part, resnet_type):
        super(RotationNet, self).__init__()
        self.part = part
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body']) + 4 + 4
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
        if resnet_type == 18:
            feat_dim = 512
        elif resnet_type == 50:
            feat_dim = 2048
        if part == 'body':
            self.body_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.lhand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.rhand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num * 515, 6], relu_final=False)
            self.body_pose_out = make_linear_layers([self.joint_num * 515, (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)
            self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        elif part == 'hand':
            self.hand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
            self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6], relu_final=False)

    def forward(self, img_feat, joint_coord_img, lhand_img_feat=None, lhand_joint_coord_img=None, rhand_img_feat=None, rhand_joint_coord_img=None):
        batch_size = img_feat.shape[0]
        if self.part == 'body':
            shape_param = self.shape_out(img_feat.mean((2, 3)))
            cam_param = self.cam_out(img_feat.mean((2, 3)))
            body_img_feat = self.body_conv(img_feat)
            body_img_feat = sample_joint_features(body_img_feat, joint_coord_img[:, :, :2])
            body_feat = torch.cat((body_img_feat, joint_coord_img), 2)
            lhand_img_feat = self.lhand_conv(lhand_img_feat)
            lhand_img_feat = sample_joint_features(lhand_img_feat, lhand_joint_coord_img[:, :, :2])
            lhand_feat = torch.cat((lhand_img_feat, lhand_joint_coord_img), 2)
            rhand_img_feat = self.rhand_conv(rhand_img_feat)
            rhand_img_feat = sample_joint_features(rhand_img_feat, rhand_joint_coord_img[:, :, :2])
            rhand_feat = torch.cat((rhand_img_feat, rhand_joint_coord_img), 2)
            feat = torch.cat((body_feat, lhand_feat, rhand_feat), 1)
            root_pose = self.root_pose_out(feat.view(batch_size, -1))
            body_pose = self.body_pose_out(feat.view(batch_size, -1))
            return root_pose, body_pose, shape_param, cam_param
        elif self.part == 'hand':
            img_feat = self.hand_conv(img_feat)
            img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:, :, :2])
            feat = torch.cat((img_feat_joints, joint_coord_img), 2)
            hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
            return hand_pose


class FaceRegressor(nn.Module):

    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([512, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([512, 6], relu_final=False)

    def forward(self, img_feat):
        expr_param = self.expr_out(img_feat.mean((2, 3)))
        jaw_pose = self.jaw_pose_out(img_feat.mean((2, 3)))
        return expr_param, jaw_pose


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.ConvTranspose2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and bnrelu_final:
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def soft_argmax_2d(heatmap2d):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height * width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))
    accu_x = heatmap2d.sum(dim=2)
    accu_y = heatmap2d.sum(dim=3)
    accu_x = accu_x * torch.arange(width).float()[None, None, :]
    accu_y = accu_y * torch.arange(height).float()[None, None, :]
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out


class BoxNet(nn.Module):

    def __init__(self):
        super(BoxNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([2048 + self.joint_num * cfg.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm, joint_img):
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * cfg.output_hm_shape[0], cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        img_feat = torch.cat((img_feat, joint_hm), 1)
        img_feat = self.deconv(img_feat)
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center, face_center = bbox_center[:, 0, :], bbox_center[:, 1, :], bbox_center[:, 2, :]
        lhand_feat = sample_joint_features(img_feat, lhand_center[:, None, :].detach())[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:, None, :].detach())[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        face_feat = sample_joint_features(img_feat, face_center[:, None, :].detach())[:, 0, :]
        face_size = self.face_size(face_feat)
        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size


class HandRoI(nn.Module):

    def __init__(self, backbone):
        super(HandRoI, self).__init__()
        self.backbone = backbone

    def forward(self, img, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float()[:, None], lhand_bbox), 1)
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float()[:, None], rhand_bbox), 1)
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_img = torchvision.ops.roi_align(img, lhand_bbox_roi, cfg.input_hand_shape, aligned=False)
        lhand_img = torch.flip(lhand_img, [3])
        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_img = torchvision.ops.roi_align(img, rhand_bbox_roi, cfg.input_hand_shape, aligned=False)
        hand_img = torch.cat((lhand_img, rhand_img))
        hand_feat = self.backbone(hand_img)
        return hand_feat


class FaceRoI(nn.Module):

    def __init__(self, backbone):
        super(FaceRoI, self).__init__()
        self.backbone = backbone

    def forward(self, img, face_bbox):
        face_bbox = torch.cat((torch.arange(face_bbox.shape[0]).float()[:, None], face_bbox), 1)
        face_bbox_roi = face_bbox.clone()
        face_bbox_roi[:, 1] = face_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:, 2] = face_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        face_bbox_roi[:, 3] = face_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:, 4] = face_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        face_img = torchvision.ops.roi_align(img, face_bbox_roi, cfg.input_face_shape, aligned=False)
        face_feat = self.backbone(face_img)
        return face_feat


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    bbox = bbox_center.view(-1, 1, 2) + torch.cat((-bbox_size.view(-1, 1, 2) / 2.0, bbox_size.view(-1, 1, 2) / 2.0), 1)
    bbox[:, :, 0] = bbox[:, :, 0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1]
    bbox[:, :, 1] = bbox[:, :, 1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0]
    bbox = bbox.view(-1, 4)
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2]
    h = bbox[:, 3]
    c_x = bbox[:, 0] + w / 2.0
    c_y = bbox[:, 1] + h / 2.0
    mask1 = w > aspect_ratio * h
    mask2 = w < aspect_ratio * h
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio
    bbox[:, 2] = w * extension_ratio
    bbox[:, 3] = h * extension_ratio
    bbox[:, 0] = c_x - bbox[:, 2] / 2.0
    bbox[:, 1] = c_y - bbox[:, 3] / 2.0
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox


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

    def __init__(self, backbone, body_position_net, body_rotation_net, box_net, hand_roi_net, hand_position_net, hand_rotation_net, face_roi_net, face_regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.body_position_net = body_position_net
        self.body_rotation_net = body_rotation_net
        self.box_net = box_net
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net
        self.face_roi_net = face_roi_net
        self.face_regressor = face_regressor
        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral'])
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.ce_loss = CELoss()
        self.trainable_modules = [self.backbone, self.body_position_net, self.body_rotation_net, self.box_net, self.hand_roi_net, self.hand_position_net, self.hand_rotation_net, self.face_roi_net, self.face_regressor]

    def get_camera_trans(self, cam_param):
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().repeat(batch_size, 1)
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose, left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(cfg.trainset_2d) == 0:
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 0.0001) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 0.0001) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 0.0001) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 0.0001) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)
        return joint_proj, joint_cam, mesh_cam

    def forward(self, inputs, targets, meta_info, mode):
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)
        img_feat = self.backbone(body_img)
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach(), body_joint_img.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0], 2.0).detach()
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1] / cfg.input_face_shape[0], 1.5).detach()
        hand_feat = self.hand_roi_net(inputs['img'], lhand_bbox, rhand_bbox)
        face_feat = self.face_roi_net(inputs['img'], face_bbox)
        _, hand_joint_img = self.hand_position_net(hand_feat)
        hand_pose = self.hand_rotation_net(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat((cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]
        batch_size = hand_feat.shape[0] // 2
        lhand_feat = torch.flip(hand_feat[:batch_size, :], [3])
        rhand_feat = hand_feat[batch_size:, :]
        root_pose, body_pose, shape, cam_param = self.body_rotation_net(img_feat, body_joint_img.detach(), lhand_feat, lhand_joint_img[:, smpl_x.pos_joint_part['L_MCP'], :].detach(), rhand_feat, rhand_joint_img[:, smpl_x.pos_joint_part['R_MCP'], :].detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)
        cam_trans = self.get_camera_trans(cam_param)
        expr, jaw_pose = self.face_regressor(face_feat)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)
        joint_proj, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode)
        pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose), 1)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)
        if mode == 'train':
            loss = {}
            loss['smplx_pose'] = self.param_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])
            loss['smplx_shape'] = self.param_loss(shape, targets['smplx_shape'], meta_info['smplx_shape_valid'][:, None])
            loss['smplx_expr'] = self.param_loss(expr, targets['smplx_expr'], meta_info['smplx_expr_valid'][:, None])
            loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smplx_joint_cam'] = self.coord_loss(joint_cam, targets['smplx_joint_cam'], meta_info['smplx_joint_valid'])
            loss['lhand_bbox'] = self.coord_loss(lhand_bbox_center, targets['lhand_bbox_center'], meta_info['lhand_bbox_valid'][:, None]) + self.coord_loss(lhand_bbox_size, targets['lhand_bbox_size'], meta_info['lhand_bbox_valid'][:, None])
            loss['rhand_bbox'] = self.coord_loss(rhand_bbox_center, targets['rhand_bbox_center'], meta_info['rhand_bbox_valid'][:, None]) + self.coord_loss(rhand_bbox_size, targets['rhand_bbox_size'], meta_info['rhand_bbox_valid'][:, None])
            loss['face_bbox'] = self.coord_loss(face_bbox_center, targets['face_bbox_center'], meta_info['face_bbox_valid'][:, None]) + self.coord_loss(face_bbox_size, targets['face_bbox_size'], meta_info['face_bbox_valid'][:, None])
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                for coord_name, trunc_name in (('joint_img', 'joint_trunc'), ('smplx_joint_img', 'smplx_joint_trunc')):
                    x = targets[coord_name][:, smpl_x.joint_part[part_name], 0]
                    y = targets[coord_name][:, smpl_x.joint_part[part_name], 1]
                    z = targets[coord_name][:, smpl_x.joint_part[part_name], 2]
                    trunc = meta_info[trunc_name][:, smpl_x.joint_part[part_name], 0]
                    x -= bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
                    x *= cfg.output_hand_hm_shape[2] / ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                    y -= bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
                    y *= cfg.output_hand_hm_shape[1] / ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                    z *= cfg.output_hand_hm_shape[0] / cfg.output_hm_shape[0]
                    trunc *= (x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (y < cfg.output_hand_hm_shape[1])
                    coord = torch.stack((x, y, z), 2)
                    trunc = trunc[:, :, None]
                    targets[coord_name] = torch.cat((targets[coord_name][:, :smpl_x.joint_part[part_name][0], :], coord, targets[coord_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)
                    meta_info[trunc_name] = torch.cat((meta_info[trunc_name][:, :smpl_x.joint_part[part_name][0], :], trunc, meta_info[trunc_name][:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                x = joint_proj[:, smpl_x.joint_part[part_name], 0]
                y = joint_proj[:, smpl_x.joint_part[part_name], 1]
                x -= bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
                x *= cfg.output_hand_hm_shape[2] / ((bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2])
                y -= bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
                y *= cfg.output_hand_hm_shape[1] / ((bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1])
                coord = torch.stack((x, y), 2)
                trans = []
                for bid in range(coord.shape[0]):
                    mask = meta_info['joint_trunc'][bid, smpl_x.joint_part[part_name], 0] == 1
                    if torch.sum(mask) == 0:
                        trans.append(torch.zeros(2).float())
                    else:
                        trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part[part_name], :][bid, mask, :2]).mean(0))
                trans = torch.stack(trans)[:, None, :]
                coord = coord + trans
                joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part[part_name][0], :], coord, joint_proj[:, smpl_x.joint_part[part_name][-1] + 1:, :]), 1)
            coord = joint_proj[:, smpl_x.joint_part['face'], :]
            trans = []
            for bid in range(coord.shape[0]):
                mask = meta_info['joint_trunc'][bid, smpl_x.joint_part['face'], 0] == 1
                if torch.sum(mask) == 0:
                    trans.append(torch.zeros(2).float())
                else:
                    trans.append((-coord[bid, mask, :2] + targets['joint_img'][:, smpl_x.joint_part['face'], :][bid, mask, :2]).mean(0))
            trans = torch.stack(trans)[:, None, :]
            coord = coord + trans
            joint_proj = torch.cat((joint_proj[:, :smpl_x.joint_part['face'][0], :], coord, joint_proj[:, smpl_x.joint_part['face'][-1] + 1:, :]), 1)
            loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:, :, :2], meta_info['joint_trunc'])
            loss['joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['joint_img']), smpl_x.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D'])
            loss['smplx_joint_img'] = self.coord_loss(joint_img, smpl_x.reduce_joint_set(targets['smplx_joint_img']), smpl_x.reduce_joint_set(meta_info['smplx_joint_trunc']))
            return loss
        else:
            for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (bbox[:, None, 2] - bbox[:, None, 0]) / cfg.input_body_shape[1] * cfg.output_hm_shape[2] / cfg.output_hand_hm_shape[2]
                joint_img[:, smpl_x.pos_joint_part[part_name], 0] += bbox[:, None, 0] / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (bbox[:, None, 3] - bbox[:, None, 1]) / cfg.input_body_shape[0] * cfg.output_hm_shape[1] / cfg.output_hand_hm_shape[1]
                joint_img[:, smpl_x.pos_joint_part[part_name], 1] += bbox[:, None, 1] / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
            for bbox in (lhand_bbox, rhand_bbox, face_bbox):
                bbox[:, 0] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 1] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
                bbox[:, 2] *= cfg.input_img_shape[1] / cfg.input_body_shape[1]
                bbox[:, 3] *= cfg.input_img_shape[0] / cfg.input_body_shape[0]
            out = {}
            out['img'] = inputs['img']
            out['joint_img'] = joint_img
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            out['cam_trans'] = cam_trans
            out['lhand_bbox'] = lhand_bbox
            out['rhand_bbox'] = rhand_bbox
            out['face_bbox'] = face_bbox
            if 'smplx_mesh_cam' in targets:
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ParamLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RotationNet,
     lambda: ([], {'part': 4, 'resnet_type': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_mks0601_Hand4Whole_RELEASE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

