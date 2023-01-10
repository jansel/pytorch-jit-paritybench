import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cocoeft = _module
h36m_smpl = _module
hp3d = _module
mix_dataset = _module
mix_dataset2_cam = _module
mix_dataset_cam = _module
mscoco = _module
pw3d = _module
HRNetWithCam = _module
HRNetWithCamReg = _module
models = _module
builder = _module
criterion = _module
Resnet = _module
hrnet = _module
SMPL = _module
lbs = _module
simple3dposeBaseSMPL = _module
simple3dposeBaseSMPL24 = _module
simple3dposeSMPLWithCam = _module
simple3dposeSMPLWithCamReg = _module
opt = _module
utils = _module
bbox = _module
config = _module
env = _module
logger = _module
metrics = _module
pose_utils = _module
presets = _module
simple_transform = _module
simple_transform_3d_cam_eft = _module
simple_transform_3d_smpl = _module
simple_transform_3d_smpl_cam = _module
simple_transform_cam = _module
registry = _module
render = _module
render_pytorch3d = _module
transforms = _module
vis = _module
version = _module
demo_image = _module
demo_video = _module
train_smpl = _module
train_smpl_cam = _module
validate_smpl = _module
validate_smpl_cam = _module
setup = _module

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


import torch.utils.data as data


import random


import torch.nn as nn


from torch.nn import functional as F


from torch import nn


import math


import torch.nn.functional as F


import logging


from collections import namedtuple


from types import MethodType


import re


import torch.distributed as dist


import inspect


from scipy.spatial.transform import Rotation


from torchvision import transforms as T


from torchvision.models.detection import fasterrcnn_resnet50_fpn


import torch.multiprocessing as mp


import torch.utils.data


from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils import clip_grad


import time


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


ModelOutput = namedtuple(typename='ModelOutput', field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf', 'pred_uvd_jts', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct', 'pred_xyz_jts_17', 'pred_vertices', 'maxvals'])


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


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
        Locations of joints. (Template Pose)
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
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()
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


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=torch.float32):
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
    device = rot_vecs.device
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


def blend_shapes(betas, shape_disps):
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


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def vertices2joints(J_regressor, vertices):
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


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, lbs_weights, pose2rot=True, dtype=torch.float32):
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
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        if pose.numel() == batch_size * 24 * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
        else:
            rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents[:24], dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]
    J_from_verts = vertices2joints(J_regressor_h36m, verts)
    return verts, J_transformed, rot_mats, J_from_verts


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'jaw', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_thumb', 'right_thumb', 'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe']
    LEAF_NAMES = ['head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe']
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, model_path, h36m_jregressor, gender='neutral', dtype=torch.float32, num_joints=29):
        """ SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        """
        super(SMPL_layer, self).__init__()
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9
        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))
        self.gender = gender
        self.dtype = dtype
        self.faces = self.smpl_data.f
        """ Register Buffer """
        self.register_buffer('faces_tensor', to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))
        self.register_buffer('shapedirs', to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        self.register_buffer('J_regressor_h36m', to_tensor(to_np(h36m_jregressor), dtype=dtype))
        self.num_joints = num_joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:self.NUM_JOINTS + 1] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map', self._parents_to_children(parents))
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1
        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')
        return children

    def forward(self, pose_axis_angle, betas, global_orient, transl=None, return_verts=True):
        """ Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        """
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle
        pose2rot = True
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(betas, full_pose, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.J_regressor_h36m, self.parents, self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)
        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
        output = ModelOutput(vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output

    def hybrik(self, pose_skeleton, betas, phis, global_orient, transl=None, return_verts=True, leaf_thetas=None):
        """ Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        """
        batch_size = pose_skeleton.shape[0]
        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)
        vertices, new_joints, rot_mats, joints_from_verts = hybrik(betas, global_orient, pose_skeleton, phis, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map, self.lbs_weights, dtype=self.dtype, train=self.training, leaf_thetas=leaf_thetas)
        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
        output = ModelOutput(vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


SPPE = Registry('sppe')


def flip(x):
    assert x.dim() == 3 or x.dim() == 4
    dim = x.dim() - 1
    return x.flip(dims=(dim,))


BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


logger = logging.getLogger('')


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

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


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.generate_feat = kwargs['generate_feat']
        self.generate_hm = kwargs.get('generate_hm', True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=self.generate_feat)
        if self.generate_feat:
            self.incre_modules, self.downsamp_modules, self.final_feat_layer = self._make_cls_head(pre_stage_channels)
        if self.generate_hm:
            self.final_layer = nn.Conv2d(in_channels=pre_stage_channels[0], out_channels=cfg['MODEL']['NUM_JOINTS'] * cfg['MODEL']['DEPTH_DIM'], kernel_size=extra['FINAL_CONV_KERNEL'], stride=1, padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_cls_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_cls_layer(head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=head_channels[3] * head_block.expansion, out_channels=2048, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(2048, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

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

    def _make_cls_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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

    def forward(self, x):
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
        if self.generate_hm:
            out_heatmap = self.final_layer(y_list[0])
            if self.generate_feat:
                y = self.incre_modules[0](y_list[0])
                for i in range(len(self.downsamp_modules)):
                    y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
                y = self.final_feat_layer(y)
                if torch._C._get_tracing_state():
                    feat = y.flatten(start_dim=2).mean(dim=2)
                else:
                    feat = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
                return out_heatmap, feat
            else:
                return out_heatmap
        else:
            assert self.generate_feat
            y = self.incre_modules[0](y_list[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
            y = self.final_feat_layer(y)
            if torch._C._get_tracing_state():
                feat = y.flatten(start_dim=2).mean(dim=2)
            else:
                feat = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
            return feat

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
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
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, map_location='cpu')
            logger.info('=> loading pretrained model {}'.format(pretrained))
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def load_hrnet_cfg(file_name):
    with open(file_name) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def get_hrnet(type_name, num_joints, depth_dim, **kwargs):
    cfg = load_hrnet_cfg(f'./hybrik/models/layers/hrnet/w{type_name}.yaml')
    cfg['MODEL']['NUM_JOINTS'] = num_joints
    cfg['MODEL']['DEPTH_DIM'] = depth_dim
    model = PoseHighResolutionNet(cfg, **kwargs)
    return model


def norm_heatmap(norm_name, heatmap):
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    assert isinstance(heatmap, torch.Tensor), 'Heatmap to be normalized must be torch.Tensor!'
    shape = heatmap.shape
    if norm_name == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_name == 'sigmoid':
        return heatmap.sigmoid()
    elif norm_name == 'divide_sum':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


class HRNetSMPLCam(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCam, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.pretrain_hrnet = kwargs['HR_PRETRAINED']
        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=self.num_joints, depth_dim=self.depth_dim, is_train=True, generate_feat=True, generate_hm=True)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.root_idx_smpl = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        init_cam = torch.tensor([0.9])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        self.decshape = nn.Linear(2048, 10)
        self.decphi = nn.Linear(2048, 23 * 2)
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, 29)
        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 0.001
        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_xyz_coord(self, pred_jts, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]
        pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]
        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)
        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def flip_sigma(self, pred_sigma):
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_sigma[:, idx] = pred_sigma[:, inv_idx]
        return pred_sigma

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]
        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]
        return heatmaps

    def forward(self, x, flip_test=False, **kwargs):
        batch_size = x.shape[0]
        out, x0 = self.preact(x)
        out = out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
        if flip_test:
            flip_x = flip(x)
            flip_out, flip_x0 = self.preact(flip_x)
            flip_out = flip_out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            flip_out = self.flip_heatmap(flip_out)
            out = out.reshape((out.shape[0], self.num_joints, -1))
            flip_out = flip_out.reshape((flip_out.shape[0], self.num_joints, -1))
            heatmaps = norm_heatmap(self.norm_type, out)
            flip_heatmaps = norm_heatmap(self.norm_type, flip_out)
            heatmaps = (heatmaps + flip_heatmaps) / 2
        else:
            out = out.reshape((out.shape[0], self.num_joints, -1))
            heatmaps = norm_heatmap(self.norm_type, out)
        assert heatmaps.dim() == 3, heatmaps.shape
        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x0 = heatmaps.sum((2, 3))
        hm_y0 = heatmaps.sum((2, 4))
        hm_z0 = heatmaps.sum((3, 4))
        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device).unsqueeze(-1)
        coord_x = hm_x0.matmul(range_tensor)
        coord_y = hm_y0.matmul(range_tensor)
        coord_z = hm_z0.matmul(range_tensor)
        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        xc = x0
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        sigma = self.decsigma(xc).reshape(batch_size, 29, 1).sigmoid()
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        if flip_test:
            flip_delta_shape = self.decshape(flip_x0)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam
            flip_sigma = self.decsigma(flip_x0).reshape(batch_size, 29, 1).sigmoid()
            pred_shape = (pred_shape + flip_pred_shape) / 2
            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2
            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)
            flip_sigma = self.flip_sigma(flip_sigma)
            sigma = (sigma + flip_sigma) / 2
        camScale = pred_camera[:, :1].unsqueeze(1)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-09)
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]
        output = edict(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1), pred_xyz_jts_29=pred_xyz_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17_flat, pred_vertices=pred_vertices, maxvals=maxvals, cam_scale=camScale[:, 0], cam_root=camera_root, transl=transl, pred_camera=pred_camera, pred_sigma=sigma, scores=1 - sigma, img_feat=x0)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


def flip_coord(preds, joint_pairs, width_dim, shift=False, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores = preds
    if flatten:
        assert pred_jts.dim() == 2 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]
    if shift:
        pred_jts[:, :, 0] = -pred_jts[:, :, 0]
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]
    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]
    pred_jts = pred_jts.reshape(num_batches, num_joints * 3)
    return pred_jts, pred_scores


class HRNetSMPLCamReg(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCamReg, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=self.num_joints, depth_dim=self.depth_dim, is_train=True, generate_feat=True, generate_hm=False, pretrain=kwargs['HR_PRETRAINED'])
        self.pretrain_hrnet = kwargs['HR_PRETRAINED']
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.root_idx_smpl = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        init_cam = torch.tensor([0.9])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        self.decshape = nn.Linear(2048, 10)
        self.decphi = nn.Linear(2048, 23 * 2)
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, 29)
        self.fc_coord = nn.Linear(2048, 29 * 3)
        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 0.001
        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def flip_sigma(self, pred_sigma):
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_sigma[:, idx] = pred_sigma[:, inv_idx]
        return pred_sigma

    def update_scale(self, pred_uvd, weight, init_scale, pred_shape, pred_phi, **kwargs):
        cam_depth = self.focal_length / (self.input_size * init_scale + 1e-09)
        pred_phi = pred_phi.reshape(-1, 23, 2)
        pred_xyz = torch.zeros_like(pred_uvd)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
            pred_xyz[:, :, 2:] = pred_uvd[:, :, 2:].clone()
            pred_xy = (pred_uvd[:, :, :2] + bbox_center) * self.input_size / self.focal_length * (pred_xyz[:, :, 2:] * self.depth_factor + cam_depth)
            pred_xyz[:, :, :2] = pred_xy / self.depth_factor
            camera_root = pred_xyz[:, 0, :] * self.depth_factor
        else:
            pred_xyz[:, :, 2:] = pred_uvd[:, :, 2:].clone()
            pred_xy = pred_uvd[:, :, :2] * self.input_size / self.focal_length * (pred_xyz[:, :, 2:] * self.depth_factor + cam_depth)
            pred_xyz[:, :, :2] = pred_xy / self.depth_factor
            camera_root = pred_xyz[:, 0, :] * self.depth_factor
        pred_xyz = pred_xyz - pred_xyz[:, [0]]
        output = self.smpl.hybrik(pose_skeleton=pred_xyz.type(self.smpl_dtype) * self.depth_factor, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_xyz24 = output.joints.float()
        pred_xyz24 = pred_xyz24 - pred_xyz24.reshape(-1, 24, 3)[:, [0], :]
        pred_xyz24 = pred_xyz24 + camera_root.unsqueeze(dim=1)
        pred_uvd24 = pred_uvd[:, :24, :].clone()
        if 'bboxes' in kwargs.keys():
            pred_uvd24[:, :, :2] = pred_uvd24[:, :, :2] + bbox_center
        bs = pred_uvd.shape[0]
        weight_uv24 = weight[:, :24, :].reshape(bs, 24, 1)
        Ax = torch.zeros((bs, 24, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)
        Ay = torch.zeros((bs, 24, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)
        Ax[:, :, 0] = pred_uvd24[:, :, 0]
        Ay[:, :, 0] = pred_uvd24[:, :, 1]
        Ax = Ax * weight_uv24
        Ay = Ay * weight_uv24
        A = torch.cat((Ax, Ay), dim=1)
        bx = (pred_xyz24[:, :, 0] - self.input_size * pred_uvd24[:, :, 0] / self.focal_length * pred_xyz24[:, :, 2]) * weight_uv24[:, :, 0]
        by = (pred_xyz24[:, :, 1] - self.input_size * pred_uvd24[:, :, 1] / self.focal_length * pred_xyz24[:, :, 2]) * weight_uv24[:, :, 0]
        b = torch.cat((bx, by), dim=1)[:, :, None]
        res = torch.inverse(A.transpose(1, 2).bmm(A)).bmm(A.transpose(1, 2)).bmm(b)
        scale = 1.0 / res
        assert scale.shape == init_scale.shape
        return scale

    def forward(self, x, flip_test=False, **kwargs):
        batch_size, _, _, width_dim = x.shape
        x0 = self.preact(x)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        delta_shape = self.decshape(x0)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(x0)
        pred_camera = self.deccam(x0).reshape(batch_size, -1) + init_cam
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        out_coord = self.fc_coord(x0).reshape(batch_size, self.num_joints, 3)
        out_sigma = self.decsigma(x0).sigmoid().reshape(batch_size, self.num_joints, 1)
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_out_coord = self.fc_coord(flip_x0).reshape(batch_size, self.num_joints, 3)
            flip_out_sigma = self.decsigma(flip_x0).sigmoid().reshape(batch_size, self.num_joints, 1)
            flip_out_coord, flip_out_sigma = flip_coord((flip_out_coord, flip_out_sigma), self.joint_pairs_29, width_dim, shift=True, flatten=False)
            flip_out_coord = flip_out_coord.reshape(batch_size, self.num_joints, 3)
            flip_out_sigma = flip_out_sigma.reshape(batch_size, self.num_joints, 1)
            out_coord = (out_coord + flip_out_coord) / 2
            out_sigma = (out_sigma + flip_out_sigma) / 2
            flip_delta_shape = self.decshape(flip_x0)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam
            pred_shape = (pred_shape + flip_pred_shape) / 2
            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2
            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)
        maxvals = 1 - out_sigma
        camScale = pred_camera[:, :1].unsqueeze(1)
        pred_uvd_jts_29 = out_coord.reshape(batch_size, self.num_joints, 3)
        if not self.training:
            camScale = self.update_scale(pred_uvd=pred_uvd_jts_29, weight=1 - out_sigma * 5, init_scale=camScale, pred_shape=pred_shape, pred_phi=pred_phi, **kwargs)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-09)
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]
        output = edict(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1), pred_xyz_jts_29=pred_xyz_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17_flat, pred_vertices=pred_vertices, maxvals=maxvals, cam_scale=camScale[:, 0], cam_root=camera_root, transl=transl, pred_camera=pred_camera, pred_sigma=out_sigma, scores=1 - out_sigma, img_feat=x0)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


LOSS = Registry('loss')


def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


class L1LossDimSMPL(nn.Module):

    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPL, self).__init__()
        self.elements = ELEMENTS
        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']
        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']
        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels):
        smpl_weight = labels['target_smpl_weight']
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)
        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight
        loss += loss_uvd * self.uvd24_weight
        return loss


class L1LossDimSMPLCam(nn.Module):

    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS
        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']
        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']
        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce
        self.pretrain_epoch = 40

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])
        pred_xyz = output.pred_xyz_jts_29[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)
        batch_size = pred_xyz.shape[0]
        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]
        loss_uvd = weighted_l1_loss(pred_uvd.reshape(batch_size, -1), target_uvd.reshape(batch_size, -1), target_uvd_weight.reshape(batch_size, -1), self.size_average)
        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight
        loss += loss_uvd * self.uvd24_weight
        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        if 'cam_trans' in output.keys():
            pred_trans = output.cam_trans * smpl_weight
            target_trans = labels['camera_trans'] * smpl_weight
            trans_loss = self.criterion_smpl(pred_trans, target_trans)
            loss += 1 * trans_loss
        pred_scale = output.cam_scale * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        scale_loss = self.criterion_smpl(pred_scale, target_scale)
        loss += 1 * scale_loss
        return loss


amp = 1 / math.sqrt(2 * math.pi)


def weighted_laplace_loss(input, sigma, target, weights, size_average):
    input = input
    target = target
    out = torch.log(sigma / amp) + torch.abs(input - target) / (math.sqrt(2) * sigma + 1e-05)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


class LaplaceLossDimSMPLCam(nn.Module):

    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(LaplaceLossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS
        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']
        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']
        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce
        self.pretrain_epoch = 40

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])
        pred_xyz = output.pred_xyz_jts_29[:, :72]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        batch_size = pred_xyz.shape[0]
        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        pred_sigma = output.pred_sigma
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]
        loss_uvd = weighted_laplace_loss(pred_uvd.reshape(batch_size, 29, -1), pred_sigma.reshape(batch_size, 29, -1), target_uvd.reshape(batch_size, 29, -1), target_uvd_weight.reshape(batch_size, 29, -1), self.size_average)
        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight
        loss += loss_uvd * self.uvd24_weight
        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        if 'cam_trans' in output.keys():
            pred_trans = output.cam_trans * smpl_weight
            target_trans = labels['camera_trans'] * smpl_weight
            trans_loss = self.criterion_smpl(pred_trans, target_trans)
            loss += 1 * trans_loss
        pred_scale = output.cam_scale * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        scale_loss = self.criterion_smpl(pred_scale, target_scale)
        loss += 1 * scale_loss
        return loss


class ResNet(nn.Module):
    """ ResNet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.inplanes = 64
        if architecture == 'resnet18' or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_dcn = [(dcn if with_dcn else None) for with_dcn in stage_with_dcn]
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self._norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, dcn=dcn))
        return nn.Sequential(*layers)


class Simple3DPoseBaseSMPL(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPL, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        backbone = ResNet
        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.leaf_pairs = (0, 1), (3, 4)
        self.root_idx_smpl = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)
        dz = uvd_jts_new[:, :, 2]
        uv_homo_jts = torch.cat((uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]), dim=2)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        cam_2d_homo = torch.cat((uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]), dim=2)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)
        if return_relative:
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)
        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)
        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]
        if shift:
            pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]
        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)
        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False):
        batch_size = x.shape[0]
        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)
        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape
        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out.shape[:2], 1), dtype=torch.float, device=out.device)
        heatmaps = out / out.sum(dim=2, keepdim=True)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))
        hm_x = hm_x * torch.cuda.comm.broadcast(torch.arange(hm_x.shape[-1]).type(torch.FloatTensor), devices=[hm_x.device.index])[0]
        hm_y = hm_y * torch.cuda.comm.broadcast(torch.arange(hm_y.shape[-1]).type(torch.FloatTensor), devices=[hm_y.device.index])[0]
        hm_z = hm_z * torch.cuda.comm.broadcast(torch.arange(hm_z.shape[-1]).type(torch.FloatTensor), devices=[hm_z.device.index])[0]
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)
        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape((batch_size, self.num_joints * 3))
        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        xc = x0
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_29_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item
        if flip_output:
            pred_uvd_jts_29 = self.flip_uvd_coord(pred_uvd_jts_29, flatten=False, shift=True)
        if flip_output and flip_item is not None:
            pred_uvd_jts_29 = (pred_uvd_jts_29 + pred_uvd_jts_29_orig.reshape(batch_size, 29, 3)) / 2
        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape((batch_size, self.num_joints * 3))
        pred_xyz_jts_29 = self.uvd_to_cam(pred_uvd_jts_29, trans_inv, intrinsic_param, joint_root, depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts_29)) == 0, ('pred_xyz_jts_29', pred_xyz_jts_29)
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, self.root_idx_smpl, :].unsqueeze(1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        if flip_output:
            pred_phi = self.flip_phi(pred_phi)
        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * 2, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / 2
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        output = ModelOutput(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17, pred_vertices=pred_vertices, maxvals=maxvals)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


class Simple3DPoseBaseSMPL24(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPL24, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = 24
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        backbone = ResNet
        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype, num_joints=self.num_joints)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.leaf_pairs = (0, 1), (3, 4)
        self.root_idx_24 = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)
        self.decleaf = nn.Linear(1024, 5 * 4)

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * self.width_dim * 4
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * self.height_dim * 4
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)
        dz = uvd_jts_new[:, :, 2]
        uv_homo_jts = torch.cat((uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]), dim=2)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        cam_2d_homo = torch.cat((uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]), dim=2)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)
        if return_relative:
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)
        xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)
        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True):
        num_joints = 24
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]
        if shift:
            pred_jts[:, :, 0] = -pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]
        if flatten:
            pred_jts = pred_jts.reshape(num_batches, num_joints * 3)
        return pred_jts

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def flip_leaf(self, pred_leaf):
        pred_leaf[:, :, 2] = -1 * pred_leaf[:, :, 2]
        pred_leaf[:, :, 3] = -1 * pred_leaf[:, :, 3]
        for pair in self.leaf_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_leaf[:, idx] = pred_leaf[:, inv_idx]
        return pred_leaf

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False):
        batch_size = x.shape[0]
        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)
        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape
        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out.shape[:2], 1), dtype=torch.float, device=out.device)
        heatmaps = out / out.sum(dim=2, keepdim=True)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))
        hm_x = hm_x * torch.cuda.comm.broadcast(torch.arange(hm_x.shape[-1]).type(torch.FloatTensor), devices=[hm_x.device.index])[0]
        hm_y = hm_y * torch.cuda.comm.broadcast(torch.arange(hm_y.shape[-1]).type(torch.FloatTensor), devices=[hm_y.device.index])[0]
        hm_z = hm_z * torch.cuda.comm.broadcast(torch.arange(hm_z.shape[-1]).type(torch.FloatTensor), devices=[hm_z.device.index])[0]
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)
        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5
        pred_uvd_jts_24 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        pred_uvd_jts_24_flat = pred_uvd_jts_24.reshape((batch_size, self.num_joints * 3))
        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        xc = x0
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_leaf = self.decleaf(xc)
        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_24_orig, pred_phi_orig, pred_leaf_orig, pred_shape_orig = flip_item
        if flip_output:
            pred_uvd_jts_24 = self.flip_uvd_coord(pred_uvd_jts_24, flatten=False, shift=True)
        if flip_output and flip_item is not None:
            pred_uvd_jts_24 = (pred_uvd_jts_24 + pred_uvd_jts_24_orig.reshape(batch_size, 24, 3)) / 2
        pred_uvd_jts_24_flat = pred_uvd_jts_24.reshape((batch_size, self.num_joints * 3))
        pred_xyz_jts_24 = self.uvd_to_cam(pred_uvd_jts_24[:, :24, :], trans_inv, intrinsic_param, joint_root, depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts_24)) == 0, ('pred_xyz_jts_24', pred_xyz_jts_24)
        pred_xyz_jts_24 = pred_xyz_jts_24 - pred_xyz_jts_24[:, self.root_idx_24, :].unsqueeze(1)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        pred_leaf = pred_leaf.reshape(batch_size, 5, 4)
        if flip_output:
            pred_phi = self.flip_phi(pred_phi)
            pred_leaf = self.flip_leaf(pred_leaf)
        if flip_output and flip_item is not None:
            pred_phi = (pred_phi + pred_phi_orig) / 2
            pred_leaf = (pred_leaf + pred_leaf_orig) / 2
            pred_shape = (pred_shape + pred_shape_orig) / 2
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_24.type(self.smpl_dtype) * 2, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), leaf_thetas=pred_leaf.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / 2
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 4)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        output = ModelOutput(pred_phi=pred_phi, pred_leaf=pred_leaf, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_24_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17, pred_vertices=pred_vertices, maxvals=maxvals)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


class Simple3DPoseBaseSMPLCam(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLCam, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        backbone = ResNet
        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.leaf_pairs = (0, 1), (3, 4)
        self.root_idx_smpl = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        init_cam = torch.tensor([0.9, 0, 0])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)
        self.deccam = nn.Linear(1024, 3)
        self.decsigma = nn.Linear(1024, 29)
        self.focal_length = kwargs['FOCAL_LENGTH']
        self.bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.depth_factor = float(self.bbox_3d_shape[2]) * 0.001
        self.input_size = 256.0

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]
        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]
        return heatmaps

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def forward(self, x, flip_test=False, **kwargs):
        batch_size = x.shape[0]
        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_out = self.deconv_layers(flip_x0)
            flip_out = self.final_layer(flip_out)
            flip_out = flip_out.reshape(batch_size, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
            flip_out = self.flip_heatmap(flip_out)
            out = out.reshape((out.shape[0], self.num_joints, -1))
            flip_out = flip_out.reshape((flip_out.shape[0], self.num_joints, -1))
            heatmaps = norm_heatmap(self.norm_type, out)
            flip_heatmaps = norm_heatmap(self.norm_type, flip_out)
            heatmaps = (heatmaps + flip_heatmaps) / 2
        else:
            out = out.reshape((out.shape[0], self.num_joints, -1))
            out = norm_heatmap(self.norm_type, out)
            assert out.dim() == 3, out.shape
            heatmaps = out / out.sum(dim=2, keepdim=True)
        maxvals, _ = torch.max(heatmaps, dim=2, keepdim=True)
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        hm_x0 = heatmaps.sum((2, 3))
        hm_y0 = heatmaps.sum((2, 4))
        hm_z0 = heatmaps.sum((3, 4))
        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
        hm_x = hm_x0 * range_tensor
        hm_y = hm_y0 * range_tensor
        hm_z = hm_z0 * range_tensor
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)
        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        xc = x0
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam
        sigma = self.decsigma(xc).reshape(batch_size, 29, 1).sigmoid()
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        if flip_test:
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)
            flip_xc = self.fc1(flip_x0)
            flip_xc = self.drop1(flip_xc)
            flip_xc = self.fc2(flip_xc)
            flip_xc = self.drop2(flip_xc)
            flip_delta_shape = self.decshape(flip_xc)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_xc)
            flip_pred_camera = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam
            flip_sigma = self.decsigma(flip_x0).reshape(batch_size, 29, 1).sigmoid()
            pred_shape = (pred_shape + flip_pred_shape) / 2
            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2
            flip_pred_camera[:, 1] = -flip_pred_camera[:, 1]
            pred_camera = (pred_camera + flip_pred_camera) / 2
            flip_sigma = self.flip_sigma(flip_sigma)
            sigma = (sigma + flip_sigma) / 2
        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-09)
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xyz_jts_29_meter = pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth) - camTrans
            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]
        output = edict(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1), pred_xyz_jts_29=pred_xyz_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17_flat, pred_vertices=pred_vertices, maxvals=maxvals, cam_scale=camScale[:, 0], cam_trans=camTrans[:, 0], cam_root=camera_root, transl=transl, pred_sigma=sigma, scores=1 - sigma)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


class Simple3DPoseBaseSMPLCamReg(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Simple3DPoseBaseSMPLCamReg, self).__init__()
        self.deconv_dim = kwargs['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        backbone = ResNet
        self.preact = backbone(f"resnet{kwargs['NUM_LAYERS']}")
        import torchvision.models as tm
        if kwargs['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
            self.feature_channel = 2048
        elif kwargs['NUM_LAYERS'] == 34:
            x = tm.resnet34(pretrained=True)
            self.feature_channel = 512
        elif kwargs['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
            self.feature_channel = 512
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer('./model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', h36m_jregressor=h36m_jregressor, dtype=self.smpl_dtype)
        self.joint_pairs_24 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)
        self.joint_pairs_29 = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.leaf_pairs = (0, 1), (3, 4)
        self.root_idx_smpl = 0
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        init_cam = torch.tensor([0.9, 0, 0])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.decshape = nn.Linear(self.feature_channel, 10)
        self.decphi = nn.Linear(self.feature_channel, 23 * 2)
        self.deccam = nn.Linear(self.feature_channel, 3)
        self.decsigma = nn.Linear(self.feature_channel, 29)
        self.fc_coord = nn.Linear(self.feature_channel, 29 * 3)
        self.focal_length = kwargs['FOCAL_LENGTH']
        self.bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.depth_factor = float(self.bbox_3d_shape[2]) * 0.001
        self.input_size = 256.0

    def _initialize(self):
        pass

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))
        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]
        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]
        return heatmaps

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]
        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]
        return pred_phi

    def forward(self, x, flip_test=False, **kwargs):
        batch_size, _, _, width_dim = x.shape
        x0 = self.preact(x)
        x0 = self.avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)
        delta_shape = self.decshape(x0)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(x0)
        pred_camera = self.deccam(x0).reshape(batch_size, -1) + init_cam
        pred_phi = pred_phi.reshape(batch_size, 23, 2)
        out_coord = self.fc_coord(x0)
        out_sigma = self.decsigma(x0).sigmoid()
        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)
            flip_x0 = self.avg_pool(flip_x0)
            flip_x0 = flip_x0.view(flip_x0.size(0), -1)
            flip_out_coord = self.fc_coord(flip_x0)
            flip_out_sigma = self.decsigma(flip_x0).sigmoid()
            flip_out_coord, flip_out_sigma = flip_coord((flip_out_coord, flip_out_sigma), self.joint_pairs_29, width_dim, shift=True, flatten=False)
            out_coord = (out_coord + flip_out_coord) / 2
            out_sigma = (out_sigma + flip_out_sigma) / 2
            flip_delta_shape = self.decshape(flip_x0)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam
            pred_shape = (pred_shape + flip_pred_shape) / 2
            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2
            flip_pred_camera[:, 1] = -flip_pred_camera[:, 1]
            pred_camera = (pred_camera + flip_pred_camera) / 2
        maxvals = 1 - out_sigma
        pred_uvd_jts_29 = out_coord.reshape(batch_size, self.num_joints, 3)
        camScale = pred_camera[:, :1].unsqueeze(1)
        camTrans = pred_camera[:, 1:].unsqueeze(1)
        camDepth = self.focal_length / (self.input_size * camScale + 1e-09)
        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)
            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()
            pred_xyz_jts_29_meter = pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth) - camTrans
            pred_xyz_jts_29[:, :, :2] = pred_xyz_jts_29_meter / self.depth_factor
            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]
        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)
        output = self.smpl.hybrik(pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor, betas=pred_shape.type(self.smpl_dtype), phis=pred_phi.type(self.smpl_dtype), global_orient=None, return_verts=True)
        pred_vertices = output.vertices.float()
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)
        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]
        output = edict(pred_phi=pred_phi, pred_delta_shape=delta_shape, pred_shape=pred_shape, pred_theta_mats=pred_theta_mats, pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1), pred_sigma=out_sigma, pred_xyz_jts_29=pred_xyz_jts_29_flat, pred_xyz_jts_24=pred_xyz_jts_24, pred_xyz_jts_24_struct=pred_xyz_jts_24_struct, pred_xyz_jts_17=pred_xyz_jts_17_flat, pred_vertices=pred_vertices, maxvals=maxvals, cam_scale=camScale[:, 0], cam_trans=camTrans[:, 0], cam_root=camera_root, transl=transl)
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):
        output = self.smpl(pose_axis_angle=gt_theta, betas=gt_beta, global_orient=None, return_verts=True)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Jeff_sjtu_HybrIK(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

