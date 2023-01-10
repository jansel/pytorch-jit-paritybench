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
preprocessing = _module
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
FreiHAND = _module
Human36M = _module
MSCOCO = _module
MuCo = _module
PW3D = _module
dataset = _module
demo = _module
config = _module
model = _module
test = _module
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


import random


import copy


import scipy.io as sio


from torch.utils.data.dataset import Dataset


import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt


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


def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Conv1d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and bnrelu_final:
            layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.ConvTranspose2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and bnrelu_final:
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PoseNet(nn.Module):

    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256 * cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)
        img_feat_x = img_feat_xy.mean(2)
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        img_feat_y = img_feat_xy.mean(3)
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, 256, cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)
        joint_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return joint_coord


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
        self.conv = make_conv_layers([64 + joint_num * cfg.output_hm_shape[0], 64])

    def forward(self, img_feat, joint_heatmap_3d):
        joint_heatmap_3d = joint_heatmap_3d.view(-1, self.joint_num * cfg.output_hm_shape[0], cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        feat = torch.cat((img_feat, joint_heatmap_3d), 1)
        feat = self.conv(feat)
        return feat


class MeshNet(nn.Module):

    def __init__(self, vertex_num):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256 * cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat)
        img_feat_x = img_feat_xy.mean(2)
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        img_feat_y = img_feat_xy.mean(3)
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)
        img_feat_z = img_feat.mean((2, 3))[:, :, None]
        img_feat_z = self.conv_z_1(img_feat_z)
        img_feat_z = img_feat_z.view(-1, 256, cfg.output_hm_shape[0])
        heatmap_z = self.conv_z_2(img_feat_z)
        coord_z = self.soft_argmax_1d(heatmap_z)
        mesh_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return mesh_coord


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or i == len(feat_dims) - 2 and relu_final:
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ParamRegressor(nn.Module):

    def __init__(self, joint_num):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        self.fc = make_linear_layers([self.joint_num * 3, 1024, 512], use_bn=True)
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.fc_pose = make_linear_layers([512, 16 * 6], relu_final=False)
        else:
            self.fc_pose = make_linear_layers([512, 24 * 6], relu_final=False)
        self.fc_shape = make_linear_layers([512, 10], relu_final=False)

    def rot6d_to_rotmat(self, x):
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose_3d):
        batch_size = pose_3d.shape[0]
        pose_3d = pose_3d.view(-1, self.joint_num * 3)
        feat = self.fc(pose_3d)
        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).float()], 2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch_size, -1)
        shape = self.fc_shape(feat)
        return pose, shape


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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x, x4

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


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


class SMPL(object):

    def __init__(self):
        self.layer = {'neutral': self.get_layer(), 'male': self.get_layer('male'), 'female': self.get_layer('female')}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].th_faces.numpy()
        self.joint_regressor = self.layer['neutral'].th_J_regressor.numpy()
        self.face_kps_vertex = 331, 2802, 6262, 3489, 3990
        nose_onehot = np.array([(1 if i == 331 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        left_eye_onehot = np.array([(1 if i == 2802 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        right_eye_onehot = np.array([(1 if i == 6262 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        left_ear_onehot = np.array([(1 if i == 3489 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        right_ear_onehot = np.array([(1 if i == 3990 else 0) for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1, -1)
        self.joint_regressor = np.concatenate((self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))
        self.joint_num = 29
        self.joints_name = 'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
        self.flip_pairs = (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28)
        self.skeleton = (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26), (25, 27), (26, 28)
        self.root_joint_idx = self.joints_name.index('Pelvis')

    def get_layer(self, gender='neutral'):
        return SMPL_Layer(gender=gender, model_root=cfg.smpl_path + '/smplpytorch/native/models')


class Model(nn.Module):

    def __init__(self, pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor):
        super(Model, self).__init__()
        self.pose_backbone = pose_backbone
        self.pose_net = pose_net
        self.pose2feat = pose2feat
        self.mesh_backbone = mesh_backbone
        self.mesh_net = mesh_net
        self.param_regressor = param_regressor
        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral']
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor
        if cfg.stage == 'lixel':
            self.trainable_modules = [self.pose_backbone, self.pose_net, self.pose2feat, self.mesh_backbone, self.mesh_net]
        else:
            self.trainable_modules = [self.param_regressor]
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.normal_loss = NormalVectorLoss(self.mesh_face)
        self.edge_loss = EdgeLengthLoss(self.mesh_face)

    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].float()
        yy = yy[None, None, :, :, :].float()
        zz = zz[None, None, :, :, :].float()
        x = joint_coord_img[:, :, 0, None, None, None]
        y = joint_coord_img[:, :, 1, None, None, None]
        z = joint_coord_img[:, :, 2, None, None, None]
        heatmap = torch.exp(-((xx - x) / cfg.sigma) ** 2 / 2 - ((yy - y) / cfg.sigma) ** 2 / 2 - ((zz - z) / cfg.sigma) ** 2 / 2)
        return heatmap

    def forward(self, inputs, targets, meta_info, mode):
        if cfg.stage == 'lixel':
            cm = nullcontext()
        else:
            cm = torch.no_grad()
        with cm:
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
            joint_coord_img = self.pose_net(pose_img_feat)
            with torch.no_grad():
                joint_heatmap = self.make_gaussian_heatmap(joint_coord_img.detach())
            shared_img_feat = self.pose2feat(shared_img_feat, joint_heatmap)
            _, mesh_img_feat = self.mesh_backbone(shared_img_feat, skip_early=True)
            mesh_coord_img = self.mesh_net(mesh_img_feat)
            joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor)[None, :, :].repeat(mesh_coord_img.shape[0], 1, 1), mesh_coord_img)
            mesh_coord_cam = None
        if cfg.stage == 'param':
            pose_param, shape_param = self.param_regressor(joint_img_from_mesh.detach())
            mesh_coord_cam, _ = self.human_model_layer(pose_param, shape_param)
            joint_coord_cam = torch.bmm(torch.from_numpy(self.joint_regressor)[None, :, :].repeat(mesh_coord_cam.shape[0], 1, 1), mesh_coord_cam)
            root_joint_cam = joint_coord_cam[:, self.root_joint_idx, None, :]
            mesh_coord_cam = mesh_coord_cam - root_joint_cam
            joint_coord_cam = joint_coord_cam - root_joint_cam
        if mode == 'train':
            loss = {}
            if cfg.stage == 'lixel':
                loss['joint_fit'] = self.coord_loss(joint_coord_img, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:, None, None])
                loss['joint_orig'] = self.coord_loss(joint_coord_img, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
                loss['mesh_fit'] = self.coord_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:, None, None])
                loss['mesh_joint_orig'] = self.coord_loss(joint_img_from_mesh, targets['orig_joint_img'], meta_info['orig_joint_trunc'], meta_info['is_3D'])
                loss['mesh_joint_fit'] = self.coord_loss(joint_img_from_mesh, targets['fit_joint_img'], meta_info['fit_joint_trunc'] * meta_info['is_valid_fit'][:, None, None])
                loss['mesh_normal'] = self.normal_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:, None, None]) * cfg.normal_loss_weight
                loss['mesh_edge'] = self.edge_loss(mesh_coord_img, targets['fit_mesh_img'], meta_info['fit_mesh_trunc'] * meta_info['is_valid_fit'][:, None, None])
            else:
                loss['pose_param'] = self.param_loss(pose_param, targets['pose_param'], meta_info['is_valid_fit'][:, None])
                loss['shape_param'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
                loss['joint_orig_cam'] = self.coord_loss(joint_coord_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
                loss['joint_fit_cam'] = self.coord_loss(joint_coord_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])
            return loss
        else:
            out = {}
            out['joint_coord_img'] = joint_coord_img
            out['mesh_coord_img'] = mesh_coord_img
            out['bb2img_trans'] = meta_info['bb2img_trans']
            if cfg.stage == 'param':
                out['mesh_coord_cam'] = mesh_coord_cam
            if 'fit_mesh_coord_cam' in targets:
                out['mesh_coord_cam_target'] = targets['fit_mesh_coord_cam']
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
    (ParamLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mks0601_I2L_MeshNet_RELEASE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

