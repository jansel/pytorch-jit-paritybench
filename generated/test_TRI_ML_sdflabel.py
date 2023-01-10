import sys
_module = sys.modules[__name__]
del sys
crops = _module
kitti = _module
main = _module
resnet_css = _module
unet_parts = _module
constants = _module
detection_3d = _module
evaluate_dump = _module
optimizer = _module
refine_css = _module
refine_css_demo = _module
rotate_iou = _module
train_css = _module
deepsdf = _module
networks = _module
deep_sdf_decoder_scale = _module
workspace = _module
grid = _module
main = _module
renderer = _module
primitives = _module
projection = _module
rasterer = _module
utils_rasterer = _module
utils = _module
data = _module
pose = _module
refinement = _module
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


import numpy as np


import torch


from torch.utils.data.dataset import Dataset


from scipy.spatial.transform import Rotation as R


import torchvision.transforms as transforms


import random


from torch.utils.data import Dataset


from collections import OrderedDict


import numpy


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


from torch.autograd import Variable


from sklearn.neighbors import KDTree


import math


from collections import defaultdict


import torchvision.utils as vis


from torch.utils.data import ConcatDataset


from torch import nn


from scipy.spatial import ConvexHull


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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch, sigmoid=False):
        super(outconv, self).__init__()
        if sigmoid:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.Sigmoid())
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def project_vecs_onto_sphere(vectors, radius, surface_only=True):
    for i in range(len(vectors)):
        v = vectors[i]
        length = torch.norm(v).detach()
        if surface_only or length.cpu().data.numpy() > radius:
            vectors[i] = vectors[i].mul(radius / (length + 1e-08))
    return vectors


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True, add_shortcut=True):
        super(up, self).__init__()
        self.add_shortcut = add_shortcut
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        if self.add_shortcut:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
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
        self.up1_u = up(384, 128)
        self.up2_u = up(192, 64)
        self.up3_u = up(128, 64)
        self.up4_u = up(64, 64, add_shortcut=False)
        self.up1_v = up(384, 128)
        self.up2_v = up(192, 64)
        self.up3_v = up(128, 64)
        self.up4_v = up(64, 64, add_shortcut=False)
        self.up1_w = up(384, 128)
        self.up2_w = up(192, 64)
        self.up3_w = up(128, 64)
        self.up4_w = up(64, 64, add_shortcut=False)
        self.up1_mask = up(384, 128)
        self.up2_mask = up(192, 64)
        self.up3_mask = up(128, 64)
        self.up4_mask = up(64, 64, add_shortcut=False)
        self.out_u = outconv(64, 256)
        self.out_v = outconv(64, 256)
        self.out_w = outconv(64, 256)
        self.out_lat = outconv(256, 3)
        self.out_mask = outconv(64, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        _freeze_module(self.conv1)
        _freeze_module(self.bn1)
        _freeze_module(self.layer1)

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
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x3 = self.layer2(x3)
        x4 = self.layer3(x3)
        x_lat = self.out_lat(x4)
        x_lat = torch.mean(x_lat.view(x_lat.size(0), x_lat.size(1), -1), dim=2)
        lat = project_vecs_onto_sphere(x_lat, 1, True)
        x_u = self.up1_u(x4, x3)
        x_u = self.up2_u(x_u, x2)
        x_u = self.up3_u(x_u, x1)
        x_u = self.up4_u(x_u, x)
        u = self.out_u(x_u)
        u = F.log_softmax(u, dim=1)
        x_v = self.up1_v(x4, x3)
        x_v = self.up2_v(x_v, x2)
        x_v = self.up3_v(x_v, x1)
        x_v = self.up4_v(x_v, x)
        v = self.out_v(x_v)
        v = F.log_softmax(v, dim=1)
        x_w = self.up1_w(x4, x3)
        x_w = self.up2_w(x_w, x2)
        x_w = self.up3_w(x_w, x1)
        x_w = self.up4_w(x_w, x)
        w = self.out_w(x_w)
        w = F.log_softmax(w, dim=1)
        x_mask = self.up1_mask(x4, x3)
        x_mask = self.up2_mask(x_mask, x2)
        x_mask = self.up3_mask(x_mask, x1)
        x_mask = self.up4_mask(x_mask, x)
        mask = self.out_mask(x_mask)
        sm_hardness = 100
        prob_u = torch.softmax(u * sm_hardness, dim=1)
        prob_v = torch.softmax(v * sm_hardness, dim=1)
        prob_w = torch.softmax(w * sm_hardness, dim=1)
        colors = torch.arange(256).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        colors_u = torch.sum(colors * prob_u, dim=1, keepdim=True)
        colors_v = torch.sum(colors * prob_v, dim=1, keepdim=True)
        colors_w = torch.sum(colors * prob_w, dim=1, keepdim=True)
        uvw_sm = torch.cat([colors_u, colors_v, colors_w], dim=1)
        prob_mask = torch.softmax(mask * sm_hardness, dim=1)
        values_mask = torch.arange(2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mask_sm = torch.sum(values_mask * prob_mask, dim=1, keepdim=True)
        uvw_sm_masked = uvw_sm * mask.argmax(dim=1, keepdim=True).expand_as(uvw_sm).float()
        output = {}
        output['u'] = u
        output['v'] = v
        output['w'] = w
        output['uvw_sm'] = uvw_sm
        output['uvw_sm_masked'] = uvw_sm_masked
        output['mask'] = mask
        output['mask_sm'] = mask_sm
        output['latent'] = lat
        return output


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_size, dims, dropout=None, dropout_prob=0.0, norm_layers=(), latent_in=(), weight_norm=False, xyz_in_all=None, use_tanh=False, latent_dropout=False, samples_per_scene=None):
        super(Decoder, self).__init__()

        def make_sequence():
            return []
        dims = [latent_size + 3] + dims + [1]
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.samples_per_scene = samples_per_scene
        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3
            if weight_norm and l in self.norm_layers:
                setattr(self, 'lin' + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, 'lin' + str(l), nn.Linear(dims[l], out_dim))
            if not weight_norm and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, 'bn' + str(l), nn.LayerNorm(out_dim))
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.scale_net = nn.Sequential(nn.Linear(latent_size, 3), nn.ReLU(True), nn.Linear(3, 3), nn.ReLU(True), nn.Linear(3, 1))

    def forward(self, input):
        xyz = input[:, -3:]
        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            if l == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, 'bn' + str(l))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        if hasattr(self, 'th'):
            x = self.th(x)
        if self.samples_per_scene:
            scale = self.scale_net(input[:, :-3].view(-1, self.samples_per_scene, input[:, :-3].size(1))[:, 0, :])
        else:
            scale = self.scale_net(input[:, :-3][0])
        return x, scale


def calibration_matrix(resolution_px, diagonal_mm, focal_len_mm, skew=0.0):
    """
    Return calibration matrix K given camera information
    Diagonal in mm of the camera sensor (ratio will match px_ratio)
    Source: https://github.com/ndrplz/differentiable-renderer/blob/pytorch/rastering/utils.py
    """
    resolution_x_px, resolution_y_px = resolution_px
    diagonal_px = np.sqrt(resolution_x_px ** 2 + resolution_y_px ** 2)
    resolution_x_mm = resolution_x_px / diagonal_px * diagonal_mm
    resolution_y_mm = resolution_y_px / diagonal_px * diagonal_mm
    skew = skew
    m_x = resolution_x_px / resolution_x_mm
    m_y = resolution_y_px / resolution_y_mm
    alpha_x = focal_len_mm * m_x
    alpha_y = focal_len_mm * m_y
    x_0 = resolution_x_px / 2
    y_0 = resolution_y_px / 2
    return np.array([[alpha_x, skew, x_0], [0, alpha_y, y_0], [0, 0, 1]])


def inside_circle(K, grid_2d, vertex_2d, vertex_3d, normals, diam=0.07, depth_constant=100, softclamp=True, softclamp_constant=3, add_bg=False):
    """
    Compute output color probabilies per pixel using 2d circles as primitives.
    Compute distances between screen points and the projected 3d vertex points.
    Clamp distances over diam value to form circles.
    Use a rendering function to compute final color probabilities.

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    diff = vertex_2d[:, :2].view([-1, 1, 2]) - grid_2d
    if softclamp:
        dist_to_point = torch.sigmoid((abs(K[0, 0] * diam / (vertex_3d[:, 2] + eps)).unsqueeze(-1) - diff.pow(2).sum(-1).sqrt()) * softclamp_constant)
    else:
        dist_to_point = torch.clamp(abs(K[0, 0] * diam / (vertex_3d[:, 2] + eps)).unsqueeze(-1) - diff.pow(2).sum(-1).sqrt(), min=0.0)
    dist_to_point = (dist_to_point > 0).detach()
    z = -vertex_3d[:, 2:]
    z_norm = torch.norm(z, p=2, dim=0).detach()
    z = torch.clamp(z.div(z_norm.unsqueeze(0) + eps) + 1, min=0) * depth_constant
    if add_bg:
        z_bg = (z.min() - 1).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([dist_to_point, torch.ones_like(dist_to_point[:1, :])])
    prob_z = torch.softmax(z * dist_to_point, dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, dist_to_point.size(-1))
    return prob_color


def inside_circle_opt(K, grid_2d, vertex_2d, vertex_3d, normals, diam=0.06, depth_constant=10000, softclamp=True, softclamp_constant=5, add_bg=True):
    """
    Compute output color probabilies per pixel using 2d circles as primitives.
    Use sparse matrices to store primitives and save memory.
    Compute distances between screen points and the projected 3d vertex points.
    Form circles using a sigmoid function, i.e. probabilistic distance.
    Use a rendering function to compute final color probabilities.

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    x_px = K[0, 2].int().item() * 2
    y_px = K[1, 2].int().item() * 2
    dist_primitive = grid_2d.pow(2).sum(-1).sqrt()
    diam_primitives = abs(K[0, 0] * diam / (vertex_3d[:, 2] + eps))
    if softclamp:
        primitives = torch.sigmoid((diam_primitives.unsqueeze(-1) - dist_primitive) * softclamp_constant)
    else:
        primitives = torch.clamp(diam_primitives.unsqueeze(-1) - dist_primitive, min=0)
    ids_sparse_size = [vertex_2d.size(0), *grid_2d[0].size()]
    ids_sparse = (grid_2d.expand(ids_sparse_size) + vertex_2d.unsqueeze(-2).expand(ids_sparse_size)).contiguous().long()
    ids_sparse_l = torch.Tensor([[0, 0]])
    ids_sparse_u = torch.Tensor([[x_px - 1, y_px - 1]])
    ids_sparse = torch.max(torch.min(ids_sparse, ids_sparse_u.long()), ids_sparse_l.long())
    third_dim = torch.arange(diam_primitives.size()[0]).unsqueeze(-1).unsqueeze(-1).expand(ids_sparse[:, :, :1].size())
    ids_sparse = torch.cat([third_dim, ids_sparse], dim=2)
    sparse_prim_dist = torch.sparse.FloatTensor(ids_sparse[:, :, [0, 2, 1]].contiguous().view(-1, 3).t(), primitives.view(-1).float(), torch.Size([primitives.size(0), y_px, x_px]))
    z = -vertex_3d[:, 2:]
    z_norm = torch.norm(z, p=2, dim=0).detach()
    z = torch.clamp(z.div(z_norm.unsqueeze(0) + eps) + 1, min=0) * depth_constant
    if add_bg:
        z_bg = (z.min() - 1).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([sparse_prim_dist.to_dense().view(-1, y_px * x_px), torch.ones(1, y_px * x_px)])
    else:
        dist_to_point = sparse_prim_dist.to_dense().view(primitives.size(0), -1)
    dist_to_point = (dist_to_point > 0).detach()
    prob_z = torch.softmax(z.masked_fill((1 - dist_to_point).bool(), torch.finfo(dtype).min), dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, dist_to_point.size(-1))
    return prob_color


def inside_surfel(K, grid_2d, vertex_2d, vertex_3d, normals, diam=0.03, depth_constant=150, softclamp=True, softclamp_constant=5, add_bg=True):
    """
    Compute output color probabilies per pixel using 3d tangent disks as primitives.
    Use normals and 3d vertex points to compute planes's [x, y, z] coordinates.
    Compute distances between plane points and the actual 3d vertex points.
    Clamp distances over diam value to form tangent disks.
    Use a rendering function to compute final color probabilities

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    n_v3d = torch.bmm(normals.unsqueeze(-2), vertex_3d.unsqueeze(-1))
    Kinv_grid2d = (K.float().inverse() @ torch.cat([grid_2d[0], torch.ones(grid_2d.shape[1], 1)], dim=-1).t()).t().unsqueeze(0)
    n_Kinv_grid2d = torch.bmm(Kinv_grid2d.expand(normals.size(0), Kinv_grid2d.size(1), -1), normals.unsqueeze(-1))
    n_Kinv_grid2d[n_Kinv_grid2d.abs() < 0.01] = torch.Tensor([eps])
    z = n_v3d.expand_as(n_Kinv_grid2d) / n_Kinv_grid2d
    grid_3d = Kinv_grid2d * z
    vectors_to_point = vertex_3d.view([-1, 1, 3]) - grid_3d
    if softclamp:
        dist_to_point = torch.sigmoid((diam - vectors_to_point.pow(2).sum(-1).sqrt()) * softclamp_constant)
    else:
        dist_to_point = torch.clamp(diam - vectors_to_point.pow(2).sum(-1).sqrt(), min=0)
    del vectors_to_point, grid_3d
    dist_to_point = (dist_to_point > 0).detach()
    z = -z[:, :, 0] * dist_to_point
    z_norm = torch.norm(z, p=2, dim=0).detach()
    z = torch.clamp(z.div(z_norm.unsqueeze(0) + eps) + 1, min=0) * depth_constant
    if add_bg:
        z2d = -vertex_3d[:, 2:] * depth_constant
        z_bg = (z2d.min() - 1).unsqueeze(-1).unsqueeze(-1).expand_as(dist_to_point[:1, :])
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([dist_to_point, torch.ones_like(dist_to_point[:1, :])])
    prob_z = torch.softmax(z.masked_fill((1 - dist_to_point).bool(), torch.finfo(dtype).min), dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, prob_z.size(-1))
    return prob_color


def convexHull(points):
    """
    Function used to Obtain the Convex hull
    Source: https://github.com/williamsea/Hidden_Points_Removal_HPR.git
    """
    points = np.append(points, [[0, 0, 0]], axis=0)
    hull = ConvexHull(points)
    return hull


def sphericalFlip(points, center, param):
    """
    Function used to Perform Spherical Flip on the Original Point Cloud
    Source: https://github.com/williamsea/Hidden_Points_Removal_HPR.git
    """
    n = len(points)
    points[:, 1] *= -1
    points[:, 2] *= -1
    points = points - np.repeat(center, n, axis=0)
    normPoints = np.linalg.norm(points, axis=1)
    R = np.repeat(max(normPoints) * np.power(30, param), n, axis=0)
    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]), axis=1))
    flippedPoints += points
    return flippedPoints


def project_in_2D(K, camera_pose, points, normals, colors, resolution_px, filter_normals=True, filter_hpr=False, output_nocs=True):
    """
    Project all 3D points onto the 2D image of given resolution using DCM rotation matrix

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        camera_pose (torch.Tensor): Camera pose as DCM (4,4)
        points (torch.Tensor): Object points to project (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        colors (torch.Tensor): Point colors (N,3)
        resolution_px (tuple): Screen resolution
        filter_normals (bool): Filter points based on the normals
        filter_hpr (bool): Filter points based on the hpr filter
        output_nocs (bool): Output NOCS
    """
    resolution_x_px, resolution_y_px = resolution_px
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    output = {}
    RT = camera_pose[:-1, :]
    correction_factor = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    RT = correction_factor @ RT
    ones = torch.ones(points[:, :1].shape)
    coords_3d_h = torch.cat([points, ones], dim=-1)
    coords_3d_h = coords_3d_h.t()
    normals_projected = (RT[:, :3] @ normals.t()).t()
    if output_nocs:
        colors = points.clone()
        colors[:, 0] *= -1
    coords_projected_3d = (RT @ coords_3d_h).t()
    if filter_normals:
        dot_prod = torch.bmm(normals_projected.unsqueeze(-2), coords_projected_3d.unsqueeze(-1)).squeeze(-1)
        coords_projected_3d_filt = coords_projected_3d.masked_select(dot_prod < 0).view(-1, 3)
        colors_filt = colors.masked_select(dot_prod < 0).view(-1, 3)
        normals_projected_filt = normals_projected.masked_select(dot_prod < 0).view(-1, 3)
        output['points_3d_filt'] = coords_projected_3d_filt
        output['normals_3d_filt'] = normals_projected_filt
        output['colors_3d_filt'] = colors_filt
    if filter_hpr:
        C = np.array([[0, 0, 0]])
        coords_projected_3d_numpy = coords_projected_3d.detach().cpu().numpy()
        coords_projected_3d_numpy /= coords_projected_3d_numpy.max()
        flippedPoints = sphericalFlip(coords_projected_3d_numpy, C, math.pi)
        mask_ids = convexHull(flippedPoints).vertices[:-1]
        mask = np.zeros_like(coords_projected_3d_numpy[:, 2:])
        mask[mask_ids] = 1
        mask = torch.BoolTensor(mask)
        coords_projected_3d = coords_projected_3d.masked_select(mask).view(-1, 3)
        colors = colors.masked_select(mask).view(-1, 3)
        normals_projected = normals_projected.masked_select(mask).view(-1, 3)
    coords_projected_2d_h = (K @ coords_projected_3d.t()).t()
    coords_projected_2d = coords_projected_2d_h[:, :2] / (coords_projected_2d_h[:, 2:] + eps)
    coords_projected_2d_x_clip = torch.clamp(coords_projected_2d[:, 0:1], -1, resolution_x_px)
    coords_projected_2d_y_clip = torch.clamp(coords_projected_2d[:, 1:2], -1, resolution_y_px)
    output['points_3d'] = coords_projected_3d
    output['normals_3d'] = normals_projected
    output['colors_3d'] = colors
    output['points_2d'] = torch.cat([coords_projected_2d_x_clip, coords_projected_2d_y_clip], dim=-1)
    return output


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = v.shape
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def project_in_2D_quat(K, camera_pose, points, normals, colors, resolution_px, filter_normals=False, filter_hpr=False, output_nocs=True):
    """
    Project all 3D points onto the 2D image of given resolution using quaternions

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        camera_pose (torch.Tensor): Camera pose as quaternion [:4] and translation vector [4:]
        points (torch.Tensor): Object points to project (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        colors (torch.Tensor): Point colors (N,3)
        resolution_px (tuple): Screen resolution
        filter_normals (bool): Filter points based on the normals
        filter_hpr (bool): Filter points based on the hpr filter
        output_nocs (bool): Output NOCS
    """
    resolution_x_px, resolution_y_px = resolution_px
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    output = {}
    q = camera_pose[:4]
    t = camera_pose[4:]
    correction_factor = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    correction_factor.requires_grad = True
    normals_projected = qrot(q.unsqueeze(0).expand([normals.size(0), 4]), normals)
    normals_projected = (correction_factor @ normals_projected.t()).t()
    if output_nocs:
        colors = points.clone()
    coords_rotated_3d_quat = qrot(q.unsqueeze(0).expand([points.size(0), 4]), points)
    corrT = correction_factor @ torch.cat([torch.eye(3), t.unsqueeze(-1)], dim=-1)
    coords_projected_3d = (corrT @ torch.cat([coords_rotated_3d_quat, torch.ones(points[:, :1].shape)], dim=-1).t()).t()
    if filter_normals:
        dot_prod = torch.bmm(normals_projected.unsqueeze(-2), coords_projected_3d.unsqueeze(-1)).squeeze(-1)
        coords_projected_3d_filt = coords_projected_3d.masked_select(dot_prod < 0).view(-1, 3)
        colors_filt = colors.masked_select(dot_prod < 0).view(-1, 3)
        normals_projected_filt = normals_projected.masked_select(dot_prod < 0).view(-1, 3)
        output['points_3d_filt'] = coords_projected_3d_filt
        output['normals_3d_filt'] = normals_projected_filt
        output['colors_3d_filt'] = colors_filt
    elif filter_hpr:
        C = np.array([[0, 0, 0]])
        coords_projected_3d_numpy = coords_projected_3d.detach().cpu().numpy()
        flippedPoints = sphericalFlip(coords_projected_3d_numpy, C, math.pi)
        mask_ids = convexHull(flippedPoints).vertices[:-1]
        mask = np.zeros_like(coords_projected_3d_numpy[:, 2:])
        mask[mask_ids] = 1
        mask = torch.BoolTensor(mask)
        coords_projected_3d = coords_projected_3d.masked_select(mask).view(-1, 3)
        colors = colors.masked_select(mask).view(-1, 3)
        normals_projected = normals_projected.masked_select(mask).view(-1, 3)
    coords_projected_2d_h = (K @ coords_projected_3d.t()).t()
    coords_projected_2d = coords_projected_2d_h[:, :2] / (coords_projected_2d_h[:, 2:] + eps)
    coords_projected_2d_x_clip = torch.clamp(coords_projected_2d[:, 0:1], -1, resolution_x_px)
    coords_projected_2d_y_clip = torch.clamp(coords_projected_2d[:, 1:2], -1, resolution_y_px)
    output['points_3d'] = coords_projected_3d
    output['normals_3d'] = normals_projected
    output['colors_3d'] = colors
    output['points_2d'] = torch.cat([coords_projected_2d_x_clip, coords_projected_2d_y_clip], dim=-1)
    return output


class Rasterer(torch.nn.Module):

    def __init__(self, K, resolution_px, diagonal_mm=20, focal_len_mm=70, precision=torch.float32):
        """
        Rasterizer constructor

        Args:
            K (torch.Tensor): Intrinsic camera parameters (3,3)
            resolution_px (tuple): Camera resolution in pixels
            diagonal_mm: Camera focal length in millimeters
            focal_len_mm: Limit the maximum amount of point, to avoid memory run-outs
            precision: Precision of the used variables
        """
        super(Rasterer, self).__init__()
        self.res_x_px, self.res_y_px = resolution_px
        yy, xx = np.mgrid[0:self.res_y_px, 0:self.res_x_px]
        grid = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        self.register_buffer('grid', torch.from_numpy(grid.reshape((1, -1, 2))))
        yy, xx = np.mgrid[-7:8, -7:8]
        grid_prim = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        self.register_buffer('grid_prim', torch.from_numpy(grid_prim.reshape((1, -1, 2))))
        if K is None:
            K = calibration_matrix(resolution_px=(self.res_x_px, self.res_y_px), diagonal_mm=diagonal_mm, focal_len_mm=focal_len_mm, skew=0)
            self.register_buffer('K', torch.from_numpy(K))
        else:
            self.register_buffer('K', K)

    def __call__(self, *args, **kwargs):
        return super(Rasterer, self).__call__(*args, **kwargs)

    def forward(self, coords, normals, colors, camera_matrix, rot='quat', primitives='disc', bg=None, output_mask=False, output_depth=False, output_normals=False, output_nocs=False, output_points=True):
        if rot == 'dcm':
            points_proj = project_in_2D(self.K, camera_matrix, coords, normals, colors, resolution_px=(self.res_x_px, self.res_y_px), output_nocs=output_nocs)
        elif rot == 'quat':
            points_proj = project_in_2D_quat(self.K, camera_matrix, coords, normals, colors, resolution_px=(self.res_x_px, self.res_y_px), output_nocs=output_nocs)
        vertices_3d = points_proj['points_3d']
        vertices_2d = points_proj['points_2d']
        normals = points_proj['normals_3d']
        colors = points_proj['colors_3d']
        if primitives == 'circle':
            prob_color = inside_circle(self.K, self.grid, vertices_2d, vertices_3d, normals, diam=0.02, add_bg=bg is not None)
        elif primitives == 'circle_opt':
            prob_color = inside_circle_opt(self.K, self.grid_prim, vertices_2d, vertices_3d, normals, diam=0.025, add_bg=bg is not None)
        elif primitives == 'disc':
            prob_color = inside_surfel(self.K, self.grid, vertices_2d, vertices_3d, normals, diam=0.04, softclamp=False, add_bg=bg is not None)
        if bg is not None:
            normals_ext = ((normals + 1) / 2).unsqueeze(-1).expand_as(prob_color[:-1, :, :])
            colors_ext = ((colors + 1) / 2).unsqueeze(-1).expand_as(prob_color[:-1, :, :])
            colors_ext = torch.cat([colors_ext, bg.view(1, colors_ext.size(1), colors_ext.size(2))])
        else:
            if output_nocs:
                colors_ext = ((colors + 1) / 2).unsqueeze(-1).expand_as(prob_color)
            else:
                colors_ext = colors.unsqueeze(-1).expand_as(prob_color)
            normals_ext = ((normals + 1) / 2).unsqueeze(-1).expand_as(prob_color)
        rendering = {}
        rendering_color = prob_color * colors_ext
        rendering['color'] = torch.clamp(torch.sum(rendering_color, dim=0).view(3, self.res_y_px, self.res_x_px), max=1)
        if output_mask:
            prob_mask_depth = prob_color[:, :1, :]
            rendering['mask'] = torch.clamp(torch.sum(prob_mask_depth, dim=0).view(1, self.res_y_px, self.res_x_px), max=1)
        if output_depth:
            prob_mask_depth = prob_color[:, :1, :]
            rendering_depth = prob_mask_depth * vertices_3d[:, 2:].unsqueeze(-1).expand_as(prob_mask_depth)
            rendering['depth'] = torch.sum(rendering_depth, dim=0).view(1, self.res_y_px, self.res_x_px)
        if output_normals:
            rendering_normals = prob_color * normals_ext
            rendering['normals'] = torch.clamp(torch.sum(rendering_normals, dim=0).view(3, self.res_y_px, self.res_x_px), max=1)
        if output_points:
            points = {}
            points['xyz'] = vertices_3d
            points['rgb'] = (colors + 1) / 2
            points['xyzf'] = points_proj['points_3d_filt']
            points['rgbf'] = (points_proj['colors_3d_filt'] + 1) / 2
            return rendering, points
        return rendering


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (double_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
]

class Test_TRI_ML_sdflabel(_paritybench_base):
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

