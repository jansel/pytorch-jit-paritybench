import sys
_module = sys.modules[__name__]
del sys
animate = _module
animation_demo = _module
augmentation = _module
extract_imgs = _module
face_swap_demo = _module
frames_dataset = _module
logger = _module
FLAME = _module
dense_motion = _module
discriminator = _module
generator = _module
keypoint_detector = _module
lbs = _module
model = _module
renderer_util = _module
tdmm_estimator = _module
util = _module
optic_flow_utils = _module
reconstruction = _module
run_ddp = _module
tdmm_inference = _module
train_ddp = _module
video_ldmk_meta = _module

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


from torch.utils.data import DataLoader


from scipy.spatial import ConvexHull


import numpy as np


import matplotlib


import copy


import torch.nn.functional as F


import time


from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset


import pandas as pd


import matplotlib.pyplot as plt


import collections


import torch.nn as nn


from torch import nn


from torchvision import models


from torch.autograd import grad


import math


import torchvision.models as models


from time import gmtime


from time import strftime


import torch.distributed as dist


from torch.optim.lr_scheduler import MultiStepLR


from torch.nn.parallel import DistributedDataParallel as DDP


class Struct(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


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
    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
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


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, pose2rot=True, dtype=torch.float32):
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
    device = betas.device
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
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


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
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


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        None
        with open(config['flame_model_path'], 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config['n_shape']], shapedirs[:, :, 300:300 + config['n_exp']]], 2)
        self.register_buffer('shapedirs', shapedirs)
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))
        lmk_embeddings = np.load(config['flame_lmk_embedding_path'], allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']))
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'])
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer('full_lmk_bary_coords', torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']))
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """
        batch_size = pose.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
        rel_rot_mat = torch.eye(3, device=pose.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39))
        neg_mask = y_rot_angle.lt(0)
        mask = y_rot_angle.lt(-39)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)
        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1) * num_verts
        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor, self.full_lmk_faces_idx.repeat(vertices.shape[0], 1), self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords, self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor, self.full_lmk_faces_idx.repeat(bz, 1), self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return vertices, landmarks2d, landmarks3d


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * 2 ** (i + 1))
            out_filters = min(max_features, block_expansion * 2 ** i)
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * 2 ** i), min(max_features, block_expansion * 2 ** (i + 1)), kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)
    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from all sparse motions including 3D face motion and 2D affine motions.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False, scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 2) * (num_channels + 1), max_features=max_features, num_blocks=num_blocks)
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 2, kernel_size=(7, 7), padding=(3, 3))
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
            self.occlusion1 = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
            self.down_motion = AntiAliasInterpolation2d(2, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        heatmaps for the 2D affine motions
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        deformed source images
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving=None, kp_source=None, render_ops=None):
        if type(render_ops) != type(None):
            reenact = render_ops['reenact']
            motion_field = render_ops['motion_field']
            source_normal_map = render_ops['source_normal_images']
            driving_normal_map = render_ops['driving_normal_images']
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            if type(render_ops) != type(None):
                reenact = self.down(reenact)
                motion_field = self.down_motion(motion_field)
                source_normal_map = self.down(source_normal_map)
                driving_normal_map = self.down(driving_normal_map)
        bs, _, h, w = source_image.shape
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source
        identity_grid = make_coordinate_grid((h, w), type=motion_field.type())
        identity_grid = identity_grid.unsqueeze(0).repeat(bs, 1, 1, 1).permute(0, 3, 1, 2)
        motion_field = motion_field + identity_grid
        if type(render_ops) != type(None):
            reenact = reenact.unsqueeze(1)
            source_normal_map = source_normal_map.unsqueeze(1)
            driving_normal_map = driving_normal_map.unsqueeze(1)
            heatmap_representation = torch.cat([heatmap_representation, -(driving_normal_map[:, :, 2:, :, :] - source_normal_map[:, :, 2:, :, :])], dim=1)
            deformed_source = torch.cat([deformed_source, reenact], dim=1)
            out_dict['motion_field'] = motion_field
        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        prediction = self.hourglass(input)
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        sparse_motion = torch.cat([sparse_motion, motion_field.unsqueeze(1)], dim=1)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict['deformation'] = deformation
        if self.occlusion:
            occlusion_map1 = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map1'] = occlusion_map1
            occlusion_map2 = torch.sigmoid(self.occlusion1(prediction))
            out_dict['occlusion_map2'] = occlusion_map2
        return out_dict


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512, sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * 2 ** i), min(max_features, block_expansion * 2 ** (i + 1)), norm=i != 0, kernel_size=4, pool=i != num_blocks - 1, sn=sn))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = torch.cat([out, heatmap], dim=1)
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)
        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict


def same_padding(images, ksizes, strides, rates):
    """
    Borrowed from https://github.com/daa233/generative-inpainting-pytorch/blob/master/utils/tools.py
    """
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = padding_left, padding_right, padding_top, padding_bottom
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Borrowed from https://github.com/daa233/generative-inpainting-pytorch/blob/master/utils/tools.py
    
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


def reduce_mean(x, axis=None, keepdim=False):
    """
    Borrowed from https://github.com/daa233/generative-inpainting-pytorch/blob/master/utils/tools.py
    """
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    """
    Borrowed from https://github.com/daa233/generative-inpainting-pytorch/blob/master/utils/tools.py
    """
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class ContextualAttention(nn.Module):
    """
    Borrowed from https://github.com/daa233/generative-inpainting-pytorch/blob/master/model/networks.py
    """

    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10, fuse=True):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        if self.fuse:
            fuse_weight = torch.eye(fuse_k).view(1, 1, fuse_k, fuse_k)
            self.register_buffer('fuse_weight', fuse_weight)

    def forward(self, f, b, mask):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel], strides=[self.rate * self.stride, self.rate * self.stride], rates=[1, 1], padding='same')
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        f = F.interpolate(f, scale_factor=1.0 / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1.0 / self.rate, mode='nearest')
        int_fs = list(f.size())
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)
        mask = F.interpolate(mask, scale_factor=1.0 / self.rate, mode='nearest')
        int_ms = list(mask.size())
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)
        mm = reduce_mean(m, axis=[3, 4]).unsqueeze(-1)
        y = []
        for i, (xi, wi, raw_wi) in enumerate(zip(f_groups, w_groups, raw_w_groups)):
            """
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front;
            wi : separated patch tensor along batch dimension of back;
            raw_wi : separated tensor along batch dimension of back;
            """
            wi = wi[0]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + 0.0001, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1)
            if self.fuse:
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [self.fuse_k, self.fuse_k], [1, 1], [1, 1])
                yi = F.conv2d(yi, self.fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [self.fuse_k, self.fuse_k], [1, 1], [1, 1])
                yi = F.conv2d(yi, self.fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            yi = yi * mm[i:i + 1]
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mm[i:i + 1]
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.0
            y.append(yi)
        y = torch.cat(y, dim=0)
        y.contiguous().view(raw_int_fs)
        return y


class GADEUpBlock2d(nn.Module):
    """
    Geometrically-Adaptive Denormalization Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, z_size=1280):
        super(GADEUpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.fc_1 = nn.Linear(z_size, out_features)
        self.fc_2 = nn.Linear(z_size, out_features)
        self.conv_f = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        m = self.sigmoid(self.conv_f(out))
        r = self.fc_1(z).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        beta = self.fc_2(z).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        addin_z = r * out + beta
        out = m * addin_z + (1 - m) * out
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class OcclusionAwareGenerator(nn.Module):
    """
    Given source image, dense motion field, occlusion map, and driving 3DMM code, 
    the U-net shaped Generator outputs the final reconstructed image.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False, blend_scale=1, num_dilation_group=4, dilation_rates=[1, 2, 4, 8]):
        super(OcclusionAwareGenerator, self).__init__()
        None
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels, estimate_occlusion_map=estimate_occlusion_map, **dense_motion_params)
        else:
            self.dense_motion_network = None
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * 2 ** i)
            out_features = min(max_features, block_expansion * 2 ** (i + 1))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * 2 ** (num_down_blocks - i))
            out_features = min(max_features, block_expansion * 2 ** (num_down_blocks - i - 1))
            up_blocks.append(GADEUpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * 2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)
        self.fc_1 = nn.Linear(1280, in_features)
        self.fc_2 = nn.Linear(1280, in_features)
        self.context_atten_layer = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True)
        self.num_dilation_group = num_dilation_group
        for i in range(num_dilation_group):
            self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(nn.Conv2d(in_features, in_features // num_dilation_group, kernel_size=3, dilation=dilation_rates[i], padding=dilation_rates[i]), nn.BatchNorm2d(in_features // num_dilation_group, affine=True), nn.ReLU()))
        self.concat_conv = nn.Sequential(nn.Conv2d(in_features * 2, in_features, kernel_size=3, padding=1), nn.BatchNorm2d(in_features, affine=True), nn.ReLU())

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving=None, kp_source=None, render_ops=None, blend_mask=None, driving_image=None, driving_features=None):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_source=kp_source, kp_driving=kp_driving, render_ops=render_ops)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            output_dict['motion_field'] = dense_motion['motion_field']
            output_dict['reenact'] = render_ops['reenact']
            if 'occlusion_map1' in dense_motion:
                occlusion_map1 = dense_motion['occlusion_map1']
                output_dict['occlusion_map1'] = occlusion_map1
                occlusion_map2 = dense_motion['occlusion_map2']
                output_dict['occlusion_map2'] = occlusion_map2
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)
            if occlusion_map1 is not None:
                if out.shape[2] != occlusion_map1.shape[2] or out.shape[3] != occlusion_map1.shape[3]:
                    occlusion_map1 = F.interpolate(occlusion_map1, size=out.shape[2:], mode='bilinear')
                    occlusion_map2 = F.interpolate(occlusion_map2, size=out.shape[2:], mode='bilinear')
                output_dict['occlusion_map1'] = occlusion_map1
                output_dict['occlusion_map2'] = occlusion_map2
            output_dict['deformation'] = deformation
            output_dict['deformed'] = self.deform_input(source_image, deformation)
        code = driving_features['feature_vec']
        r = self.fc_1(code).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        beta = self.fc_2(code).unsqueeze(-1).unsqueeze(-1).expand_as(out)
        addin_driving = r * out + beta
        out = (1.0 - occlusion_map1) * addin_driving + occlusion_map1 * out
        if blend_mask is not None:
            blend_mask = self.blend_downsample(blend_mask)
            enc_driving = self.first(driving_image)
            for i in range(len(self.down_blocks)):
                enc_driving = self.down_blocks[i](enc_driving)
            out = enc_driving * (1 - blend_mask) + out * blend_mask
        out = self.bottleneck(out)
        ca_occ = occlusion_map2
        ca_out = self.context_atten_layer(out, out, 1.0 - ca_occ)
        tmp = []
        for i in range(self.num_dilation_group):
            tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(out))
        dilated_out = torch.cat(tmp, dim=1)
        out = torch.cat([ca_out, dilated_out], dim=1)
        out = self.concat_conv(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out, code)
        out = self.final(out)
        out = F.sigmoid(out)
        output_dict['prediction'] = out
        return output_dict


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features, num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels, max_features=max_features, num_blocks=num_blocks)
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7), padding=pad)
        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None
        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}
        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)
        out = self.gaussian2kp(heatmap)
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian
        return out


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))), requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))), requires_grad=False)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs
        if 'sigma_tps' in kwargs and 'points_tps' in kwargs:
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode='reflection')

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)
        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)
            result = distances ** 2
            result = result * torch.log(distances + 1e-06)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, tdmm, train_params, with_eye=True):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.tdmm = tdmm
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.loss_weights = train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
        self.l2 = nn.MSELoss()
        self.with_eye = with_eye

    def forward(self, x):
        image_size = x['source'].shape[2]
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        source_ldmk_2d_gt = x['source_ldmk_2d']
        driving_ldmk_2d_gt = x['driving_ldmk_2d']
        source_codedict = self.tdmm.encode(x['source'])
        driving_codedict = self.tdmm.encode(x['driving'])
        source_verts, source_transformed_verts, source_ldmk_2d = self.tdmm.decode_flame(source_codedict)
        driving_verts, driving_transformed_verts, driving_ldmk_2d = self.tdmm.decode_flame(driving_codedict)
        source_albedo = self.tdmm.extract_texture(x['source'], source_transformed_verts, with_eye=self.with_eye)
        render_ops = self.tdmm.render(source_transformed_verts, driving_transformed_verts, source_albedo)
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, render_ops=render_ops, driving_features=driving_codedict)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        if self.loss_weights['landmark'] != 0:
            source_ldmk_loss = F.l1_loss(source_ldmk_2d, source_ldmk_2d_gt)
            driving_ldmk_loss = F.l1_loss(driving_ldmk_2d, driving_ldmk_2d_gt)
            ldmk_loss = source_ldmk_loss + driving_ldmk_loss
            loss_values['landmark'] = self.loss_weights['landmark'] * ldmk_loss
        if self.loss_weights['photometric'] != 0:
            source_photometric_loss = F.l1_loss(render_ops['source'], x['source'] * render_ops['source_face_mask'])
            driving_photometric_loss = F.l1_loss(render_ops['reenact'], x['driving'] * render_ops['driving_face_mask'])
            photometric_loss = source_photometric_loss + driving_photometric_loss
            loss_values['photometric'] = self.loss_weights['photometric'] * photometric_loss
        if self.loss_weights['tdmm_param'] != 0:
            source_param_loss = torch.mean(source_codedict['shape'] ** 2) + 0.8 * torch.mean(source_codedict['exp'] ** 2)
            driving_param_loss = torch.mean(driving_codedict['shape'] ** 2) + 0.8 * torch.mean(driving_codedict['exp'] ** 2)
            param_loss = source_param_loss + driving_param_loss
            loss_values['tdmm_param'] = self.loss_weights['tdmm_param'] * param_loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                gen_driving_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                driving_real_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(gen_driving_vgg[i] - driving_real_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total
            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total
        if self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian'] != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian'])
                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())
                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total
        return loss_values


class TdmmFullModel(torch.nn.Module):

    def __init__(self, tdmm):
        super(TdmmFullModel, self).__init__()
        self.tdmm = tdmm

    def forward(self, x):
        codedict = self.tdmm.encode(x['image'])
        verts, transformed_verts, landmark_2d = self.tdmm.decode_flame(codedict)
        ldmk_loss = F.l1_loss(x['ldmk'], landmark_2d)
        param_loss = 0.001 * (torch.mean(codedict['shape'] ** 2) + 0.8 * torch.mean(codedict['exp'] ** 2))
        losses = {}
        losses['ldmk_loss'] = ldmk_loss
        losses['param_loss'] = param_loss
        return losses


def dict2obj(d):
    if not isinstance(d, dict):
        return d


    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


class Pytorch3dRasterizer(nn.Module):

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {'image_size': image_size, 'blur_radius': 0.0, 'faces_per_pixel': 1, 'bin_size': None, 'max_faces_per_bin': None, 'perspective_correct': False}
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, _, bary_coords, _ = rasterize_meshes(meshes_screen, image_size=raster_settings.image_size, blur_radius=raster_settings.blur_radius, faces_per_pixel=raster_settings.faces_per_pixel, bin_size=raster_settings.bin_size, max_faces_per_bin=raster_settings.max_faces_per_bin, perspective_correct=raster_settings.perspective_correct)
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, D)
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


def batch_orth_proj(X, camera):
    """ orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    """
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    shape = X_trans.shape
    Xn = camera[:, :, 0:1] * X_trans
    return Xn


def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[2]))
    return vertices[faces.long()]


flame_model_dir = './modules'


class mymobilenetv2(nn.Module):

    def __init__(self, num_classes=1000, image_size=256):
        super(mymobilenetv2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.n_layers = len(self.model.features)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.model.last_channel, num_classes)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        self.model.classifier = nn.Sequential(self.dropout, self.fc)
        self.scale_factor = 224.0 / image_size
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, self.scale_factor)
        self.feature_maps = {}

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        feature = self.model.features[0](x)
        for i in range(18):
            feature = self.model.features[i + 1](feature)
        feature = nn.functional.adaptive_avg_pool2d(feature, 1).reshape(feature.shape[0], -1)
        code = self.model.classifier(feature)
        return code, feature


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3)
    faces = faces + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]
    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)
    normals.index_add_(0, faces[:, 1].long(), torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(), torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))
    normals = F.normalize(normals, eps=1e-06, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals


class TDMMEstimator(nn.Module):

    def __init__(self):
        super(TDMMEstimator, self).__init__()
        code_dim = flame_config['model']['n_shape'] + flame_config['model']['n_exp'] + flame_config['model']['n_pose'] + flame_config['model']['n_cam']
        self.encoder = mymobilenetv2(num_classes=code_dim, image_size=flame_config['dataset']['image_size'])
        self.flame = FLAME(flame_config['model'])
        self.rasterizer = Pytorch3dRasterizer(flame_config['dataset']['image_size'])
        self.uv_rasterizer = Pytorch3dRasterizer(flame_config['model']['uv_size'])
        verts, faces, aux = load_obj(flame_config['model']['topology_path'])
        uvcoords = aux.verts_uvs[None, ...]
        uvfaces = faces.textures_idx[None, ...]
        faces = faces.verts_idx[None, ...]
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1)
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        self.register_buffer('faces', faces)
        mask = imread(flame_config['model']['face_eye_mask_path']).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [flame_config['model']['uv_size'], flame_config['model']['uv_size']])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)
        mask = imread(flame_config['model']['face_mask_path']).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_mask = F.interpolate(mask, [flame_config['model']['uv_size'], flame_config['model']['uv_size']])
        self.register_buffer('uv_face_mask', uv_face_mask)
        self.image_size = flame_config['dataset']['image_size']
        self.sigmoid = nn.Sigmoid()

    def encode(self, images):
        code, feature = self.encoder(images)
        code_dict = {}
        code_dict['code'] = code.clone()
        code_dict['feature_vec'] = feature
        code[:, 156] += 10.0
        code_dict['shape'] = code[:, 0:100]
        code_dict['exp'] = code[:, 100:150]
        code_dict['pose'] = code[:, 150:156]
        code_dict['cam'] = code[:, 156:159]
        code_dict['cam'][:, 1:] = self.sigmoid(code_dict['cam'][:, 1:]) - 0.5
        return code_dict

    def decode_flame(self, code_dict):
        verts, landmarks_2d, _ = self.flame(shape_params=code_dict['shape'], expression_params=code_dict['exp'], pose_params=code_dict['pose'])
        transformed_verts = batch_orth_proj(verts, code_dict['cam'])
        transformed_verts[:, :, 1:] = -transformed_verts[:, :, 1:]
        landmarks_2d = batch_orth_proj(landmarks_2d, code_dict['cam'])[:, :, :2]
        landmarks_2d[:, :, 1:] = -landmarks_2d[:, :, 1:]
        landmarks_2d = landmarks_2d * self.image_size / 2 + self.image_size / 2
        return verts, transformed_verts, landmarks_2d

    def extract_texture(self, images, transformed_verts, with_eye=True):
        uv_pverts = self.world2uv(transformed_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear')
        if with_eye:
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask
        else:
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_mask
        albedo = uv_texture_gt
        return albedo

    def render(self, source_transformed_verts, driving_transformed_verts, source_albedo):
        batch_size = source_transformed_verts.shape[0]
        vert_motions = source_transformed_verts[:, :, 0:2] - driving_transformed_verts[:, :, 0:2]
        face_motions = face_vertices(vert_motions, self.faces.expand(batch_size, -1, -1))
        source_transformed_normals = vertex_normals(source_transformed_verts, self.faces.expand(batch_size, -1, -1))
        source_transformed_face_normals = face_vertices(source_transformed_normals, self.faces.expand(batch_size, -1, -1))
        driving_transformed_normals = vertex_normals(driving_transformed_verts, self.faces.expand(batch_size, -1, -1))
        driving_transformed_face_normals = face_vertices(driving_transformed_normals, self.faces.expand(batch_size, -1, -1))
        source_attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), source_transformed_face_normals], -1)
        source_transformed_verts[:, :, 2] = source_transformed_verts[:, :, 2] + 10
        source_rendering = self.rasterizer(source_transformed_verts, self.faces.expand(batch_size, -1, -1), source_attributes)
        driving_attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), driving_transformed_face_normals, face_motions], -1)
        driving_transformed_verts[:, :, 2] = driving_transformed_verts[:, :, 2] + 10
        driving_rendering = self.rasterizer(driving_transformed_verts, self.faces.expand(batch_size, -1, -1), driving_attributes)
        source_alpha_images = source_rendering[:, -1, :, :][:, None, :, :]
        driving_alpha_images = driving_rendering[:, -1, :, :][:, None, :, :]
        source_uvcoords_images = source_rendering[:, :3, :, :]
        source_grid = source_uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        source = F.grid_sample(source_albedo, source_grid, align_corners=False)
        source = source * source_alpha_images
        driving_uvcoords_images = driving_rendering[:, :3, :, :]
        driving_grid = driving_uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        reenact = F.grid_sample(source_albedo, driving_grid, align_corners=False)
        reenact = reenact * driving_alpha_images
        source_face_region_mask = (torch.sum(source.detach(), dim=1, keepdim=True) != 0).float()
        driving_face_region_mask = (torch.sum(reenact.detach(), dim=1, keepdim=True) != 0).float()
        source_transformed_normal_map = source_rendering[:, 3:6, :, :]
        driving_transformed_normal_map = driving_rendering[:, 3:6, :, :]
        motion_field = driving_rendering[:, 6:8, :, :]
        outputs = {'source': source * source_face_region_mask, 'reenact': reenact * driving_face_region_mask, 'source_face_mask': source_face_region_mask, 'driving_face_mask': driving_face_region_mask, 'source_normal_images': source_transformed_normal_map * source_face_region_mask, 'driving_normal_images': driving_transformed_normal_map * driving_face_region_mask, 'source_uv_images': source_uvcoords_images[:, :2, ...] * source_face_region_mask, 'driving_uv_images': driving_uvcoords_images[:, :2, ...] * driving_face_region_mask, 'motion_field': motion_field * driving_face_region_mask}
        return outputs

    def world2uv(self, vertices):
        """
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices_ = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), self.uvfaces.expand(batch_size, -1, -1), face_vertices_)[:, :3]
        return uv_vertices


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AntiAliasInterpolation2d,
     lambda: ([], {'channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock2d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'block_expansion': 4, 'in_features': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Hourglass,
     lambda: ([], {'block_expansion': 4, 'in_features': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (KPDetector,
     lambda: ([], {'block_expansion': 4, 'num_kp': 4, 'num_channels': 4, 'max_features': 4, 'num_blocks': 4, 'temperature': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SameBlock2d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpBlock2d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (mymobilenetv2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_Qiulin_W_SAFA(_paritybench_base):
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

