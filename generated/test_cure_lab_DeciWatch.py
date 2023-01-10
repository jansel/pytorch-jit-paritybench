import sys
_module = sys.modules[__name__]
del sys
demo = _module
eval = _module
config = _module
evaluate = _module
loss = _module
trainer = _module
dataset = _module
aist_dataset = _module
h36m_dataset = _module
jhmdb_dataset = _module
pw3d_dataset = _module
deciwatch = _module
smpl = _module
cam_utils = _module
eval_metrics = _module
geometry_utils = _module
render = _module
sampling_utils = _module
utils = _module
visualize = _module
visualize_2d = _module
visualize_3d = _module
visualize_smpl = _module
train = _module

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


import torch.nn as nn


import numpy as np


import math


import torch.nn.functional as F


import logging


import time


from abc import ABC


from abc import abstractmethod


import torch.utils.data as data


from torch import index_add


import random


import copy


from typing import Optional


from torch import nn


from torch import Tensor


import scipy.interpolate


from torch.nn import functional as F


import functools


import torch.nn.init as init


from torch.nn.parameter import Parameter


import torch.distributions as distributions


import matplotlib


import matplotlib.pyplot as plt


from matplotlib.backends.backend_agg import FigureCanvasAgg


from scipy.interpolate import CubicSpline


from scipy.interpolate import interp1d


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


import torch.optim as optim


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


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
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
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
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6D_to_axis(rot6D):
    rotmat = rot6d_to_rotmat(rot6D)
    axis = rotation_matrix_to_angle_axis(rotmat)
    return axis


class DeciWatchLoss(nn.Module):

    def __init__(self, w_denoise, lamada, smpl_model_dir, smpl):
        super().__init__()
        self.w_denoise = w_denoise
        self.lamada = lamada
        self.smpl_model_dir = smpl_model_dir
        self.smpl = smpl

    def mask_lr1_loss(self, inputs, mask, targets):
        Bs, C, L = inputs.shape
        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()
        N = not_mask.sum(dtype=torch.float32)
        loss = F.l1_loss(inputs * not_mask, targets * not_mask, reduction='sum') / N
        return loss

    def forward(self, recover, denoise, gt, mask_src, mask_pad, use_smpl_loss=False):
        if use_smpl_loss == True and self.smpl == True:
            return self.forward_smpl(recover, denoise, gt, mask_src, mask_pad)
        else:
            return self.forward_lr1(recover, denoise, gt, mask_src, mask_pad)

    def forward_lr1(self, recover, denoise, gt, mask_src, mask_pad):
        B, L, C = recover.shape
        recover = recover.permute(0, 2, 1)
        denoise = denoise.permute(0, 2, 1)
        gt = gt.permute(0, 2, 1)
        loss_denoise = self.mask_lr1_loss(denoise, mask_src, gt)
        loss_pose = self.mask_lr1_loss(recover, mask_pad, gt)
        weighted_loss = self.w_denoise * loss_denoise + self.lamada * loss_pose
        return weighted_loss

    def forward_smpl(self, recover, denoise, gt, mask_src, mask_pad):
        SMPL_TO_J14 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 38]
        B, L, C = recover.shape
        recover = rot6D_to_axis(recover.reshape(-1, 6)).reshape(-1, 24 * 3)
        denoise = rot6D_to_axis(denoise.reshape(-1, 6)).reshape(-1, 24 * 3)
        gt = rot6D_to_axis(gt.reshape(-1, 6)).reshape(-1, 24 * 3)
        device = recover.device
        smpl = SMPL(model_path=self.smpl_model_dir, gender='neutral', batch_size=1)
        gt_smpl_joints = smpl.forward(global_orient=gt[:, 0:3], body_pose=gt[:, 3:]).joints[:, SMPL_TO_J14]
        denoise_smpl_joints = smpl.forward(global_orient=denoise[:, 0:3], body_pose=denoise[:, 3:]).joints[:, SMPL_TO_J14]
        recover_smpl_joints = smpl.forward(global_orient=recover[:, 0:3], body_pose=recover[:, 3:]).joints[:, SMPL_TO_J14]
        gt_smpl_joints = gt_smpl_joints.reshape(B, L, -1).permute(0, 2, 1)
        denoise_smpl_joints = denoise_smpl_joints.reshape(B, L, -1).permute(0, 2, 1)
        recover_smpl_joints = recover_smpl_joints.reshape(B, L, -1).permute(0, 2, 1)
        loss_denoise = self.mask_lr1_loss(denoise_smpl_joints, mask_src, gt_smpl_joints)
        loss_pose = self.mask_lr1_loss(recover_smpl_joints, mask_pad, gt_smpl_joints)
        weighted_loss = self.w_denoise * loss_denoise + self.lamada * loss_pose
        return weighted_loss


class PositionEmbeddingSine_1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, B, L):
        position = torch.arange(0, L, dtype=torch.float32).unsqueeze(0)
        position = position.repeat(B, 1)
        if self.normalize:
            eps = 1e-06
            position = position / (position[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 1, rounding_mode='trunc') / self.num_pos_feats)
        pe = torch.zeros(B, L, self.num_pos_feats * 2)
        pe[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        pe[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)
        pe = pe.permute(1, 0, 2)
        return pe


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'leaky_relu':
        return F.leaky_relu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, decoder_hidden_dim, nhead, dim_feedforward=256, dropout=0.1, activation='leaky_relu', pre_norm=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(decoder_hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(decoder_hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(decoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, decoder_hidden_dim)
        self.norm1 = nn.LayerNorm(decoder_hidden_dim)
        self.norm2 = nn.LayerNorm(decoder_hidden_dim)
        self.norm3 = nn.LayerNorm(decoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None, query_pos: Optional[Tensor]=None):
        if self.pre_norm:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, encoder_hidden_dim, nhead, dim_feedforward=256, dropout=0.1, activation='leaky_relu', pre_norm=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(encoder_hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(encoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, encoder_hidden_dim)
        self.norm1 = nn.LayerNorm(encoder_hidden_dim)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor]=None, src_key_padding_mask: Optional[Tensor]=None, pos: Optional[Tensor]=None):
        if self.pre_norm:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class Transformer(nn.Module):

    def __init__(self, input_nc, encoder_hidden_dim=64, decoder_hidden_dim=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, dropout=0.1, activation='leaky_relu', pre_norm=True, recovernet_interp_method='linear', recovernet_mode='transformer'):
        super(Transformer, self).__init__()
        self.joints_dim = input_nc
        self.decoder_embed = nn.Conv1d(self.joints_dim, decoder_hidden_dim, kernel_size=5, stride=1, padding=2)
        self.encoder_embed = nn.Linear(self.joints_dim, encoder_hidden_dim)
        encoder_layer = TransformerEncoderLayer(encoder_hidden_dim, nhead, dim_feedforward, dropout, activation, pre_norm)
        encoder_norm = nn.LayerNorm(encoder_hidden_dim) if pre_norm else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(decoder_hidden_dim, nhead, dim_feedforward, dropout, activation, pre_norm)
        decoder_norm = nn.LayerNorm(decoder_hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.decoder_joints_embed = nn.Linear(decoder_hidden_dim, self.joints_dim)
        self.encoder_joints_embed = nn.Linear(encoder_hidden_dim, self.joints_dim)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.nhead = nhead
        self.recovernet_interp_method = recovernet_interp_method
        self.recovernet_mode = recovernet_mode

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_seq, encoder_mask, encoder_pos_embed, input_seq_interp, decoder_mask, decoder_pos_embed, sample_interval, device):
        self.device = device
        bs, c, l = input_seq.shape
        input_seq = input_seq.permute(2, 0, 1)
        input_seq_interp = input_seq_interp.permute(2, 0, 1)
        input = input_seq.clone()
        trans_src = self.encoder_embed(input_seq)
        mem = self.encode(trans_src, encoder_mask, encoder_pos_embed)
        reco = self.encoder_joints_embed(mem) + input
        interp = torch.nn.functional.interpolate(input=reco[::sample_interval, :, :].permute(1, 2, 0), size=reco.shape[0], mode=self.recovernet_interp_method, align_corners=True).permute(2, 0, 1)
        center = interp.clone()
        trans_tgt = self.decoder_embed(interp.permute(1, 2, 0)).permute(2, 0, 1)
        output = self.decode(mem, encoder_mask, encoder_pos_embed, trans_tgt, decoder_mask, decoder_pos_embed)
        joints = self.decoder_joints_embed(output) + center
        if self.recovernet_mode == 'transformer':
            return joints, reco
        elif self.recovernet_mode == 'tradition_interp':
            return interp, reco

    def encode(self, src, src_mask, pos_embed):
        mask = torch.eye(src.shape[0]).bool()
        memory = self.encoder(src, mask=mask, src_key_padding_mask=src_mask, pos=pos_embed)
        return memory

    def decode(self, memory, memory_mask, memory_pos, tgt, tgt_mask, tgt_pos):
        hs = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=memory_mask, pos=memory_pos, query_pos=tgt_pos)
        return hs


def build_model(args):
    return Transformer(input_nc=args['input_dim'], decoder_hidden_dim=args['decoder_hidden_dim'], encoder_hidden_dim=args['encoder_hidden_dim'], dropout=args['dropout'], nhead=args['nheads'], dim_feedforward=args['dim_feedforward'], num_encoder_layers=args['enc_layers'], num_decoder_layers=args['dec_layers'], activation=args['activation'], pre_norm=args['pre_norm'], recovernet_interp_method=args['recovernet_interp_method'], recovernet_mode=args['recovernet_mode'])


class DeciWatch(nn.Module):

    def __init__(self, input_dim, sample_interval, encoder_hidden_dim, decoder_hidden_dim, dropout=0.1, nheads=4, dim_feedforward=256, enc_layers=3, dec_layers=3, activation='leaky_relu', pre_norm=True, recovernet_interp_method='linear', recovernet_mode='transformer'):
        """
        pos_embed_dim: position embedding dim
        sample_interval: uniform sampling interval N
        """
        super(DeciWatch, self).__init__()
        self.pos_embed_dim = encoder_hidden_dim
        self.pos_embed = self.build_position_encoding(self.pos_embed_dim)
        self.sample_interval = sample_interval
        self.deciwatch_par = {'input_dim': input_dim, 'encoder_hidden_dim': encoder_hidden_dim, 'decoder_hidden_dim': decoder_hidden_dim, 'dropout': dropout, 'nheads': nheads, 'dim_feedforward': dim_feedforward, 'enc_layers': enc_layers, 'dec_layers': dec_layers, 'activation': activation, 'pre_norm': pre_norm, 'recovernet_interp_method': recovernet_interp_method, 'recovernet_mode': recovernet_mode}
        self.transformer = build_model(self.deciwatch_par)

    def build_position_encoding(self, pos_embed_dim):
        N_steps = pos_embed_dim // 2
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True)
        return position_embedding

    def generate_unifrom_mask(self, L, sample_interval=10):
        seq_len = L
        if (seq_len - 1) % sample_interval != 0:
            raise Exception('The following equation should be satisfied: [Window size] = [sample interval] * Q + 1, where Q is an integer.')
        sample_mask = np.ones(seq_len, dtype=np.int32)
        sample_mask[::sample_interval] = 0
        encoder_mask = sample_mask
        decoder_mask = np.array([0] * L, dtype=np.int32)
        return torch.tensor(encoder_mask), torch.tensor(decoder_mask)

    def seqence_interpolation(self, motion, rate):
        seq_len = motion.shape[-1]
        indice = torch.arange(seq_len, dtype=int)
        chunk = torch.div(indice, rate, rounding_mode='trunc')
        remain = indice % rate
        prev = motion[:, :, chunk * rate]
        next = torch.cat([motion[:, :, (chunk[:-1] + 1) * rate], motion[:, :, -1, np.newaxis]], -1)
        remain = remain
        interpolate = prev / rate * (rate - remain) + next / rate * remain
        return interpolate

    def forward(self, sequence, device):
        B, L, C = sequence.shape
        seq = sequence.permute(0, 2, 1)
        encoder_mask, decoder_mask = self.generate_unifrom_mask(L, sample_interval=self.sample_interval)
        self.input_seq = seq * (1 - encoder_mask.int())
        self.input_seq_interp = self.seqence_interpolation(self.input_seq, self.sample_interval)
        self.encoder_mask = encoder_mask.unsqueeze(0).repeat(B, 1)
        self.decoder_mask = decoder_mask.unsqueeze(0).repeat(B, 1)
        self.encoder_pos_embed = self.pos_embed(B, L)
        self.decoder_pos_embed = self.encoder_pos_embed.clone()
        self.recover, self.denoise = self.transformer.forward(input_seq=self.input_seq, encoder_mask=self.encoder_mask, encoder_pos_embed=self.encoder_pos_embed, input_seq_interp=self.input_seq_interp, decoder_mask=self.decoder_mask, decoder_pos_embed=self.decoder_pos_embed, sample_interval=self.sample_interval, device=device)
        self.recover = self.recover.permute(1, 0, 2).reshape(B, L, C)
        self.denoise = self.denoise.permute(1, 0, 2).reshape(B, L, C)
        return self.recover, self.denoise


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TransformerDecoderLayer,
     lambda: ([], {'decoder_hidden_dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (TransformerEncoderLayer,
     lambda: ([], {'encoder_hidden_dim': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_cure_lab_DeciWatch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

