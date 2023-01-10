import sys
_module = sys.modules[__name__]
del sys
src = _module
dataset = _module
base_data_loader = _module
base_dataset = _module
voxceleb_dataset = _module
decalib = _module
aflw2000 = _module
build_datasets = _module
datasets = _module
detectors = _module
ethnicity = _module
now = _module
train_datasets = _module
vggface = _module
vox = _module
deca = _module
FLAME = _module
decoders = _module
encoders = _module
frnet = _module
lbs = _module
resnet = _module
trainer = _module
config = _module
rotation_converter = _module
util = _module
fitting = _module
VGG19_LOSS = _module
models = _module
base_model = _module
discriminator = _module
face2face_rho_model = _module
image_pyramid = _module
motion_network = _module
networks = _module
rendering_network = _module
options = _module
parse_config = _module
reenact = _module
train = _module
html = _module
landmark_image_generation = _module
util = _module
visualizer = _module

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


import torch.utils.data


import torch.utils.data as data


import torchvision.transforms as transforms


import torch


import numpy as np


import scipy


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


import scipy.io


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


from torch.nn.parameter import Parameter


import torch.optim as optim


import torchvision


from time import time


from collections import OrderedDict


from scipy.ndimage import morphology


import copy


from torch.nn import functional as F


from torch import nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


from abc import ABC


class ResnetEncoder(nn.Module):

    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model()
        self.layers = nn.Sequential(nn.Linear(feature_size, 1024), nn.ReLU(), nn.Linear(1024, outsize))
        self.last_op = last_op

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        if self.last_op:
            parameters = self.last_op(parameters)
        return parameters


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1, 2))[..., :4]
    quaternion[..., 0] += cx * cy * cz - sx * sy * sz
    quaternion[..., 1] += cx * sy * sz + cy * cz * sx
    quaternion[..., 2] += cx * cz * sy - sx * cy * sz
    quaternion[..., 3] += cx * cy * sz + sx * cz * sy
    return quaternion


def quaternion_to_angle_axis(quaternion: torch.Tensor):
    """Convert quaternion vector to angle axis of rotation. TODO: CORRECT

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


def batch_euler2axis(r):
    return quaternion_to_angle_axis(euler_to_quaternion(r))


class DECA(nn.Module):

    def __init__(self, config=None, device='cuda'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self._create_model(self.cfg.model)

    def _create_model(self, model_cfg):
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}
        self.E_flame = ResnetEncoder(outsize=self.n_param)
        self.E_detail = ResnetEncoder(outsize=self.n_detail)
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            None
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
        else:
            None
        self.E_flame.eval()
        self.E_detail.eval()

    def decompose_code(self, code, num_dict):
        """ Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        """
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def encode(self, images, use_detail=False):
        if use_detail:
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict

    def ensemble_3DMM_params(self, codedict, image_size, original_image_size):
        i = 0
        cam = codedict['cam']
        tform = codedict['tform']
        scale, tx, ty, sz = util.calculate_scale_tx_ty(cam, tform, image_size, original_image_size)
        crop_scale, crop_tx, crop_ty, crop_sz = util.calculate_crop_scale_tx_ty(cam)
        scale = float(scale[i].cpu())
        tx = float(tx[i].cpu())
        ty = float(ty[i].cpu())
        sz = float(sz[i].cpu())
        crop_scale = float(crop_scale[i].cpu())
        crop_tx = float(crop_tx[i].cpu())
        crop_ty = float(crop_ty[i].cpu())
        crop_sz = float(crop_sz[i].cpu())
        shape_params = codedict['shape'][i].cpu().numpy()
        expression_params = codedict['exp'][i].cpu().numpy()
        pose_params = codedict['pose'][i].cpu().numpy()
        face_model_paras = dict()
        face_model_paras['shape'] = shape_params.tolist()
        face_model_paras['exp'] = expression_params.tolist()
        face_model_paras['pose'] = pose_params.tolist()
        face_model_paras['cam'] = cam[i].cpu().numpy().tolist()
        face_model_paras['scale'] = scale
        face_model_paras['tx'] = tx
        face_model_paras['ty'] = ty
        face_model_paras['sz'] = sz
        face_model_paras['crop_scale'] = crop_scale
        face_model_paras['crop_tx'] = crop_tx
        face_model_paras['crop_ty'] = crop_ty
        face_model_paras['crop_sz'] = crop_sz
        return face_model_paras


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


def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=torch.float32):
    """  same as batch_matrix2axis
    Calculates the rotation matrices for a batch of rotation vectors
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
        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.n_shape], shapedirs[:, :, 300:300 + config.n_exp]], 2)
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
        with open(config.flame_lmk_embedding_path, 'r') as f:
            lmk_embeddings = json.load(f)
        self.lmk_faces_idx = torch.tensor(lmk_embeddings['lmk_faces_idx']).long().unsqueeze(0)
        self.lmk_bary_coords = torch.tensor(lmk_embeddings['lmk_bary_coords']).unsqueeze(0)

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
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(betas, full_pose, template_vertices, self.shapedirs, self.posedirs, self.J_regressor, self.parents, self.lbs_weights, dtype=self.dtype)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor, self.lmk_faces_idx.repeat(bz, 1), self.lmk_bary_coords.repeat(bz, 1, 1))
        return vertices, landmarks3d


class FLAMETex(nn.Module):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)
        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            None
            raise NotImplementedError
        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


class Generator(nn.Module):

    def __init__(self, latent_dim=100, out_channels=1, out_scale=0.01, sample_mode='bilinear'):
        super(Generator, self).__init__()
        self.out_scale = out_scale
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode=sample_mode), nn.Conv2d(32, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16, 0.8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, out_channels, 3, stride=1, padding=1), nn.Tanh())

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img * self.out_scale


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x1 = self.layer4(x)
        x2 = self.avgpool(x1)
        x2 = x2.view(x2.size(0), -1)
        return x2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VGG19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
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
        X = (X + 1) / 2
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG19LOSS(torch.nn.Module):

    def __init__(self):
        super(VGG19LOSS, self).__init__()
        self.model = VGG19()

    def forward(self, fake, target, weight_mask=None, loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        vgg_fake = self.model(fake)
        vgg_target = self.model(target)
        value_total = 0
        for i, weight in enumerate(loss_weights):
            value = torch.abs(vgg_fake[i] - vgg_target[i].detach())
            if weight_mask is not None:
                bs, c, H1, W1 = value.shape
                _, _, H2, W2 = weight_mask.shape
                if H1 != H2 or W1 != W2:
                    cur_weight_mask = F.interpolate(weight_mask, size=(H1, W1))
                    value = value * cur_weight_mask
                else:
                    value = value * weight_mask
            value = torch.mean(value, dim=[x for x in range(1, len(value.size()))])
            value_total += loss_weights[i] * value
        return value_total


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512, sn=False, use_kp=False):
        super(Discriminator, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(num_channels + 3 * use_kp if i == 0 else min(max_features, block_expansion * 2 ** i), min(max_features, block_expansion * 2 ** (i + 1)), norm=i != 0, kernel_size=4, pool=i != num_blocks - 1, sn=sn))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            bs, _, h1, w1 = kp.shape
            bs, C, h2, w2 = out.shape
            if h1 != h2 or w1 != w2:
                kp = F.interpolate(kp, size=(h2, w2), mode='bilinear')
            out = torch.cat([out, kp], dim=1)
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

    def forward(self, input):
        if self.scale == 1.0:
            return input
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))
        return out


class ImagePyramide(torch.nn.Module):

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


class DownBlock(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups, stride=2)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class UpBlock(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True, sample_mode='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu
        self.sample_mode = sample_mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode=self.sample_mode)
        out = self.conv(out)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class MotionNet(nn.Module):

    def __init__(self, opt):
        super(MotionNet, self).__init__()
        ngf = opt.mn_ngf
        n_local_enhancers = opt.n_local_enhancers
        n_downsampling = opt.mn_n_downsampling
        n_blocks_local = opt.mn_n_blocks_local
        in_features = [9, 9, 9]
        f1_model_ngf = ngf * 2 ** n_local_enhancers
        f1_model = [nn.Conv2d(in_channels=in_features[0], out_channels=f1_model_ngf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(f1_model_ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            f1_model += [DownBlock(f1_model_ngf * mult, f1_model_ngf * mult * 2, kernel_size=4, padding=1, use_relu=True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            f1_model += [UpBlock(f1_model_ngf * mult, int(f1_model_ngf * mult / 2), kernel_size=3, padding=1)]
        self.f1_model = nn.Sequential(*f1_model)
        self.f1_motion = nn.Conv2d(f1_model_ngf, 2, kernel_size=(3, 3), padding=(1, 1))
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_first_downsample = [DownBlock(in_features[n], ngf_global * 2, kernel_size=4, padding=1, use_relu=True)]
            model_other = []
            model_other += [DownBlock(ngf_global * 2, ngf_global * 4, kernel_size=4, padding=1, use_relu=True), DownBlock(ngf_global * 4, ngf_global * 8, kernel_size=4, padding=1, use_relu=True)]
            for i in range(n_blocks_local):
                model_other += [ResBlock(ngf_global * 8, 3, 1)]
            model_other += [UpBlock(ngf_global * 8, ngf_global * 4, kernel_size=3, padding=1), UpBlock(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1), UpBlock(ngf_global * 2, ngf_global, kernel_size=3, padding=1)]
            model_motion = nn.Conv2d(ngf_global, out_channels=2, kernel_size=3, padding=1, groups=1)
            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_first_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_other))
            setattr(self, 'model' + str(n) + '_3', model_motion)

    def forward(self, input1, input2, input3):
        output_prev = self.f1_model(input1)
        low_motion = self.f1_motion(output_prev)
        output_prev = self.model1_2(self.model1_1(input2) + output_prev)
        middle_motion = self.model1_3(output_prev)
        middle_motion = middle_motion + nn.Upsample(scale_factor=2, mode='nearest')(low_motion)
        output_prev = self.model2_2(self.model2_1(input3) + output_prev)
        high_motion = self.model2_3(output_prev)
        high_motion = high_motion + nn.Upsample(scale_factor=2, mode='nearest')(middle_motion)
        low_motion = low_motion.permute(0, 2, 3, 1)
        middle_motion = middle_motion.permute(0, 2, 3, 1)
        high_motion = high_motion.permute(0, 2, 3, 1)
        return [low_motion, middle_motion, high_motion]


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class PoseEncoder(nn.Module):

    def __init__(self, ngf, headpose_dims):
        super(PoseEncoder, self).__init__()
        self.ngf = ngf
        self.embedding_module1 = nn.Sequential(nn.ConvTranspose2d(headpose_dims, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.embedding_module2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 4, 0, bias=False))

    def get_embedding_feature_map_channel(self):
        return self.ngf * 4

    def forward(self, headpose):
        bs, dim = headpose.size()
        cur_embedding = self.embedding_module1(headpose.view(bs, dim, 1, 1))
        cur_embedding = self.embedding_module2(cur_embedding)
        return cur_embedding


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, final_use_norm=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        norm = nn.BatchNorm2d
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.use_res_connect:
            final_use_norm = True
        if expand_ratio == 1:
            conv = [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), norm(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)]
            if final_use_norm:
                conv += [norm(oup)]
        else:
            conv = [nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), norm(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), norm(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)]
            if final_use_norm:
                conv += [norm(oup)]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SPADE(nn.Module):

    def __init__(self, input_channel, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(input_channel, affine=False)
        nhidden = label_nc * 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, input_channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, input_channel, kernel_size=3, padding=1)

    def forward(self, x, condition_map):
        normalized = self.param_free_norm(x)
        _, c1, h1, w1 = x.size()
        _, c2, h2, w2 = condition_map.size()
        if h1 != h2 or w1 != w2:
            raise ValueError('x and condition_map have different sizes.')
        actv = self.mlp_shared(condition_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class RenderingNet(nn.Module):

    def __init__(self, opt):
        super(RenderingNet, self).__init__()
        self.src_headpose_encoder = PoseEncoder(headpose_dims=opt.headpose_dims, ngf=opt.headpose_embedding_ngf)
        self.headpose_feature_cn = self.src_headpose_encoder.get_embedding_feature_map_channel()
        self.drv_headpose_encoder = PoseEncoder(headpose_dims=opt.headpose_dims, ngf=opt.headpose_embedding_ngf)
        norm = nn.BatchNorm2d

        def encoder_block(inc, ouc, t, n, s, final_use_norm=True):
            model = []
            input_channel = int(inc)
            output_channel = int(ouc)
            for i in range(n):
                if i == 0:
                    if n > 1:
                        model.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        model.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t, final_use_norm=final_use_norm))
                else:
                    model.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, final_use_norm=final_use_norm))
                input_channel = output_channel
            return nn.Sequential(nn.Sequential(*model))

        def decoder_block(inc, ouc, t, n, s, final_use_norm=True):
            model = []
            input_channel = int(inc)
            output_channel = int(ouc)
            for i in range(n):
                model.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, final_use_norm=final_use_norm))
                input_channel = output_channel
            if s == 2:
                model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            return nn.Sequential(nn.Sequential(*model))
        en_channels = opt.mobilev2_encoder_channels
        de_channels = opt.mobilev2_decoder_channels
        en_layers = opt.mobilev2_encoder_layers
        de_layers = opt.mobilev2_decoder_layers
        en_expansion_factor = opt.mobilev2_encoder_expansion_factor
        de_expansion_factor = opt.mobilev2_decoder_expansion_factor
        self.en_conv_block = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=en_channels[0], kernel_size=3, stride=1, padding=1, bias=False), norm(en_channels[0]), nn.ReLU(True))
        self.en_down_block1 = nn.Sequential(encoder_block(t=en_expansion_factor[0], ouc=en_channels[1], n=en_layers[0], s=1, inc=en_channels[0]), encoder_block(t=en_expansion_factor[1], ouc=en_channels[2], n=en_layers[1], s=2, inc=en_channels[1]), encoder_block(t=en_expansion_factor[2], ouc=en_channels[3], n=en_layers[2], s=2, inc=en_channels[2], final_use_norm=False))
        self.en_SPADE1 = SPADE(en_channels[3], self.headpose_feature_cn)
        self.en_SPADE1_act = nn.ReLU(True)
        self.en_down_block2 = nn.Sequential(encoder_block(t=en_expansion_factor[3], ouc=en_channels[4], n=en_layers[3], s=2, inc=en_channels[3]), encoder_block(t=en_expansion_factor[4], ouc=en_channels[5], n=en_layers[4], s=1, inc=en_channels[4], final_use_norm=False))
        self.en_SPADE_2 = SPADE(en_channels[5], self.headpose_feature_cn)
        self.en_SPADE_2_act = nn.ReLU(True)
        self.en_down_block3 = nn.Sequential(encoder_block(t=en_expansion_factor[5], ouc=en_channels[6], n=en_layers[5], s=2, inc=en_channels[5], final_use_norm=False))
        self.en_SPADE_3 = SPADE(en_channels[6], self.headpose_feature_cn)
        self.en_SPADE_3_act = nn.ReLU(True)
        self.en_res_block = nn.Sequential(encoder_block(t=en_expansion_factor[6], ouc=en_channels[7], n=en_layers[6], s=1, inc=en_channels[6], final_use_norm=False))
        self.en_SPADE_4 = SPADE(en_channels[7], self.headpose_feature_cn)
        self.en_SPADE_4_act = nn.ReLU(True)
        self.de_SPADE_1 = SPADE(de_channels[7], self.headpose_feature_cn)
        self.de_SPADE_1_act = nn.ReLU(True)
        self.de_res_block = nn.Sequential(decoder_block(t=de_expansion_factor[6], ouc=de_channels[6], n=de_layers[6], s=1, inc=en_channels[7], final_use_norm=False))
        self.de_SPADE_2 = SPADE(de_channels[6] + en_channels[6], self.headpose_feature_cn)
        self.de_SPADE_2_act = nn.ReLU(True)
        self.de_up_block1 = nn.Sequential(decoder_block(t=de_expansion_factor[5], ouc=de_channels[5], n=de_layers[5], s=2, inc=de_channels[6] + en_channels[6]))
        self.de_SPADE_3 = SPADE(de_channels[5] + en_channels[5], self.headpose_feature_cn)
        self.de_SPADE_3_act = nn.ReLU(True)
        self.de_up_block2 = nn.Sequential(decoder_block(t=de_expansion_factor[4], ouc=de_channels[4], n=de_layers[4], s=1, inc=de_channels[5] + en_channels[5]), decoder_block(t=de_expansion_factor[3], ouc=de_channels[3], n=de_layers[3], s=2, inc=de_channels[4]))
        self.de_SPADE_4 = SPADE(de_channels[3] + en_channels[3], self.headpose_feature_cn)
        self.de_SPADE_4_act = nn.ReLU(True)
        self.de_up_block3 = nn.Sequential(decoder_block(t=de_expansion_factor[2], ouc=de_channels[2], n=de_layers[2], s=2, inc=de_channels[3] + en_channels[3]), decoder_block(t=de_expansion_factor[1], ouc=de_channels[1], n=de_layers[1], s=2, inc=de_channels[2]), decoder_block(t=de_expansion_factor[0], ouc=de_channels[0], n=de_layers[0], s=1, inc=de_channels[1]))
        self.de_conv_block = nn.Sequential(nn.Conv2d(in_channels=de_channels[0], out_channels=3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())

    @staticmethod
    def deform_input(inp, deformation):
        bs, h1, w1, _ = deformation.shape
        bs, c, h2, w2 = inp.shape
        if h1 != h2 or w1 != w2:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h2, w2), mode='nearest')
            deformation = deformation.permute(0, 2, 3, 1)
        trans_feature = F.grid_sample(inp, deformation)
        return trans_feature

    @staticmethod
    def resize_headpose_embedding(inp, embedding):
        bs, c, h1, w1 = inp.shape
        _, _, h2, w2 = embedding.shape
        if h1 != h2 or w1 != w2:
            embedding = F.interpolate(embedding, size=(h1, w1), mode='nearest')
        return embedding

    def forward(self, src_img, motion_field, src_headpose, drv_headpose):
        src_headpose_embedding = self.src_headpose_encoder(src_headpose)
        x = self.en_conv_block(src_img)
        x1 = self.en_down_block1(x)
        x2_in = self.en_SPADE1(x1, self.resize_headpose_embedding(x1, src_headpose_embedding))
        x2_in = nn.ReLU(True)(x2_in)
        x2 = self.en_down_block2(x2_in)
        x3_in = self.en_SPADE_2(x2, self.resize_headpose_embedding(x2, src_headpose_embedding))
        x3_in = nn.ReLU(True)(x3_in)
        x3 = self.en_down_block3(x3_in)
        x4_in = self.en_SPADE_3(x3, self.resize_headpose_embedding(x3, src_headpose_embedding))
        x4_in = nn.ReLU(True)(x4_in)
        x4 = self.en_res_block(x4_in)
        de_x4_in = self.en_SPADE_4(x4, self.resize_headpose_embedding(x4, src_headpose_embedding))
        de_x4_in = nn.ReLU(True)(de_x4_in)
        trans_features = []
        trans_features.append(RenderingNet.deform_input(x2_in, motion_field))
        trans_features.append(RenderingNet.deform_input(x3_in, motion_field))
        trans_features.append(RenderingNet.deform_input(x4_in, motion_field))
        trans_features.append(RenderingNet.deform_input(de_x4_in, motion_field))
        drv_headpose_embedding = self.drv_headpose_encoder(drv_headpose)
        x4_in = self.de_SPADE_1(trans_features[-1], self.resize_headpose_embedding(trans_features[-1], drv_headpose_embedding))
        x4_in = nn.ReLU(True)(x4_in)
        x4_out = self.de_res_block(x4_in)
        x3_in = torch.cat([x4_out, trans_features[-2]], dim=1)
        x3_in = self.de_SPADE_2(x3_in, self.resize_headpose_embedding(x3_in, drv_headpose_embedding))
        x3_in = nn.ReLU(True)(x3_in)
        x3_out = self.de_up_block1(x3_in)
        x2_in = torch.cat([x3_out, trans_features[-3]], dim=1)
        x2_in = self.de_SPADE_3(x2_in, self.resize_headpose_embedding(x2_in, drv_headpose_embedding))
        x2_in = nn.ReLU(True)(x2_in)
        x2_out = self.de_up_block2(x2_in)
        x1_in = torch.cat([x2_out, trans_features[-4]], dim=1)
        x1_in = self.de_SPADE_4(x1_in, self.resize_headpose_embedding(x1_in, drv_headpose_embedding))
        x1_in = nn.ReLU(True)(x1_in)
        x1_out = self.de_up_block3(x1_in)
        x_out = self.de_conv_block(x1_out)
        return x_out

    def register_source_face(self, src_img, src_headpose):
        src_headpose_embeddings = self.src_headpose_encoder(src_headpose)
        x = self.en_conv_block(src_img)
        x1 = self.en_down_block1(x)
        x2_in = self.en_SPADE1(x1, self.resize_headpose_embedding(x1, src_headpose_embeddings))
        self.x2_in = nn.ReLU(True)(x2_in)
        x2 = self.en_down_block2(x2_in)
        x3_in = self.en_SPADE_2(x2, self.resize_headpose_embedding(x2, src_headpose_embeddings))
        self.x3_in = nn.ReLU(True)(x3_in)
        x3 = self.en_down_block3(x3_in)
        x4_in = self.en_SPADE_3(x3, self.resize_headpose_embedding(x3, src_headpose_embeddings))
        self.x4_in = nn.ReLU(True)(x4_in)
        x4 = self.en_res_block(x4_in)
        de_x4_in = self.en_SPADE_4(x4, self.resize_headpose_embedding(x4, src_headpose_embeddings))
        self.de_x4_in = nn.ReLU(True)(de_x4_in)

    def reenactment(self, motion_field, drv_headpose):
        trans_features = []
        trans_features.append(RenderingNet.deform_input(self.x2_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.x3_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.x4_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.de_x4_in, motion_field))
        drv_headpose_embeddings = self.drv_headpose_encoder(drv_headpose)
        x4_in = self.de_SPADE_1(trans_features[-1], self.resize_headpose_embedding(trans_features[-1], drv_headpose_embeddings))
        x4_in = nn.ReLU(True)(x4_in)
        x4_out = self.de_res_block(x4_in)
        x3_in = torch.cat([x4_out, trans_features[-2]], dim=1)
        x3_in = self.de_SPADE_2(x3_in, self.resize_headpose_embedding(x3_in, drv_headpose_embeddings))
        x3_in = nn.ReLU(True)(x3_in)
        x3_out = self.de_up_block1(x3_in)
        x2_in = torch.cat([x3_out, trans_features[-3]], dim=1)
        x2_in = self.de_SPADE_3(x2_in, self.resize_headpose_embedding(x2_in, drv_headpose_embeddings))
        x2_in = nn.ReLU(True)(x2_in)
        x2_out = self.de_up_block2(x2_in)
        x1_in = torch.cat([x2_out, trans_features[-4]], dim=1)
        x1_in = self.de_SPADE_4(x1_in, self.resize_headpose_embedding(x1_in, drv_headpose_embeddings))
        x1_in = nn.ReLU(True)(x1_in)
        x1_out = self.de_up_block3(x1_in)
        x_out = self.de_conv_block(x1_out)
        return x_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AntiAliasInterpolation2d,
     lambda: ([], {'channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock2d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoseEncoder,
     lambda: ([], {'ngf': 4, 'headpose_dims': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ResnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (SPADE,
     lambda: ([], {'input_channel': 4, 'label_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Up,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (UpBlock,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGG19LOSS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_NetEase_GameAI_Face2FaceRHO(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

