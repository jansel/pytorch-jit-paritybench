import sys
_module = sys.modules[__name__]
del sys
lib = _module
config = _module
lm = _module
model = _module
CDPN = _module
models = _module
monte_carlo_pose_loss = _module
resnet_backbone = _module
resnet_rot_head = _module
resnet_trans_head = _module
ops = _module
pnp = _module
camera = _module
common = _module
cost_fun = _module
distributions = _module
epropnp = _module
levenberg_marquardt = _module
rotation_conversions = _module
ref = _module
test = _module
train = _module
utils = _module
draw_orient_density = _module
eval = _module
fancy_logger = _module
fs = _module
img = _module
io = _module
tictoc = _module
transform3d = _module
_init_paths = _module
main = _module
epropnp_det_basic = _module
epropnp_det_coord_regr = _module
epropnp_det_coord_regr_trainval = _module
epropnp_det_no_reproj = _module
epropnp_det_v1b_220312 = _module
epropnp_det_v1b_220411 = _module
infer_imgs = _module
infer_nuscenes_sequence = _module
epropnp_det = _module
apis = _module
inference = _module
test = _module
core = _module
bbox_3d = _module
builder = _module
center_target = _module
dim_coder = _module
multiclass_log_dim_coder = _module
iou_calculators = _module
bbox3d_iou_calculator = _module
rotate_iou_calculator = _module
rotate_iou_kernel = _module
misc = _module
proj_error_coder = _module
dist_dim_proj_error_coder = _module
evaluation = _module
kitti_utils = _module
rotate_iou = _module
visualizer = _module
deformable_point_vis = _module
image_bev_vis = _module
datasets = _module
dataset_wrappers = _module
kitti3d_dataset = _module
kitti3dcar_dataset = _module
nuscenes3d_dataset = _module
pipelines = _module
formating = _module
loading = _module
transforms = _module
dense_heads = _module
deform_pnp_head = _module
fcos_emb_head = _module
detectors = _module
losses = _module
cosine_angle_loss = _module
monte_carlo_pose_loss = _module
mvd_gaussian_mixture_nll_loss = _module
smooth_l1_loss = _module
positional_encoding = _module
deformable_attention_sampler = _module
group_linear = _module
inter_roi_ops = _module
iou3d = _module
iou3d_utils = _module
camera = _module
common = _module
cost_fun = _module
distributions = _module
epropnp = _module
levenberg_marquardt = _module
runner = _module
hooks = _module
model_updater = _module
optimizer = _module
timer = _module
setup = _module
test = _module
checkpoint_cleaner = _module
nuscenes_converter = _module
test = _module
train = _module
train = _module
camera = _module
common = _module
cost_fun = _module
distributions = _module
epropnp = _module
levenberg_marquardt = _module

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


import torch.utils.data as data


import numpy as np


import random


import torch


from torch.utils import model_zoo


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


import torchvision.models as models


import torch.nn as nn


import math


from torch.distributions import VonMises


from torch.distributions.multivariate_normal import _batch_mahalanobis


from torch.distributions.multivariate_normal import _standard_normal


from torch.distributions.multivariate_normal import _batch_mv


from abc import ABCMeta


from abc import abstractmethod


from functools import partial


import torch.nn.functional as F


from typing import Optional


from scipy.linalg import logm


import numpy.linalg as LA


import time


import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R


import torch.utils.data


import matplotlib


import warnings


from torch.nn.modules.batchnorm import _BatchNorm


from torch._six import inf


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import copy


class MonteCarloPoseLoss(nn.Module):

    def __init__(self, loss_weight=1.0, init_norm_factor=1.0, momentum=0.01, reduction='mean'):
        super(MonteCarloPoseLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor, weight=None, avg_factor=None, reduction_override=None):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                norm_factor = reduce_mean(norm_factor)
                self.norm_factor.mul_(1 - self.momentum).add_(self.momentum * norm_factor)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = monte_carlo_pose_loss(pose_sample_logweights, cost_target, weight=weight, reduction=reduction, avg_factor=avg_factor) * (self.loss_weight / self.norm_factor)
        return loss


class CDPN(nn.Module):

    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()

    def forward(self, x):
        features = self.backbone(x)
        cc_maps = self.rot_head_net(features)
        trans = self.trans_head_net(features)
        return cc_maps, trans


class ResNetBackboneNet(nn.Module):

    def __init__(self, block, layers, in_channel=3, freeze=False):
        self.freeze = freeze
        self.inplanes = 64
        super(ResNetBackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x_low_feature = self.maxpool(x)
                x = self.layer1(x_low_feature)
                x = self.layer2(x)
                x = self.layer3(x)
                x_high_feature = self.layer4(x)
                return x_high_feature.detach()
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x_low_feature = self.maxpool(x)
            x = self.layer1(x_low_feature)
            x = self.layer2(x)
            x = self.layer3(x)
            x_high_feature = self.layer4(x)
            return x_high_feature


class RotHeadNet(nn.Module):

    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1, output_dim=5, freeze=False):
        super(RotHeadNet, self).__init__()
        self.freeze = freeze
        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0
        assert output_kernel_size == 1 or output_kernel_size == 3, 'Only support kenerl 1 and 3'
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
        self.out_layer = nn.Conv2d(num_filters, output_dim, kernel_size=output_kernel_size, padding=pad, bias=True)
        self.scale_branch = nn.Linear(256, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x3d, w2d = self.out_layer(x).split([3, 2], dim=1)
                scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x3d, w2d = self.out_layer(x).split([3, 2], dim=1)
            scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
        return x3d, w2d, scale


class TransHeadNet(nn.Module):

    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_dim=3, freeze=False, with_bias_end=True):
        super(TransHeadNet, self).__init__()
        self.freeze = freeze
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0
        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(256 * 8 * 8, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, output_dim))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.view(-1, 256 * 8 * 8)
                for i, l in enumerate(self.linears):
                    x = l(x)
                return x.detach()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.view(-1, 256 * 8 * 8)
            for i, l in enumerate(self.linears):
                x = l(x)
            return x


def evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_jacobian=False, out_residual=False, out_cost=False, **kwargs):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        x2d (torch.Tensor): Shape (*, n, 2)
        w2d (torch.Tensor): Shape (*, n, 2)
        pose (torch.Tensor): Shape (*, 4 or 7)
        camera: Camera object of batch size (*, )
        cost_fun: PnPCost object of batch size (*, )
        out_jacobian (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the Jacobian; when False, skip the computation and returns None
        out_residual (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the residual; when False, skip the computation and returns None
        out_cost (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the cost; when False, skip the computation and returns None

    Returns:
        Tuple:
            residual (torch.Tensor | None): Shape (*, n*2)
            cost (torch.Tensor | None): Shape (*, )
            jacobian (torch.Tensor | None): Shape (*, n*2, 4 or 6)
    """
    x2d_proj, jac_cam = camera.project(x3d, pose, out_jac=out_jacobian.view(x2d.shape[:-1] + (2, out_jacobian.size(-1))) if isinstance(out_jacobian, torch.Tensor) else out_jacobian, **kwargs)
    residual, cost, jacobian = cost_fun.compute(x2d_proj, x2d, w2d, jac_cam=jac_cam, out_residual=out_residual, out_cost=out_cost, out_jacobian=out_jacobian)
    return residual, cost, jacobian


def skew(x):
    """
    Args:
        x (torch.Tensor): shape (*, 3)

    Returns:
        torch.Tensor: (*, 3, 3), skew symmetric matrices
    """
    mat = x.new_zeros(x.shape[:-1] + (3, 3))
    mat[..., [2, 0, 1], [1, 2, 0]] = x
    mat[..., [1, 2, 0], [2, 0, 1]] = -x
    return mat


def quaternion_to_rot_mat(quaternions):
    """
    Args:
        quaternions (torch.Tensor): (*, 4)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    if quaternions.requires_grad:
        w, i, j, k = torch.unbind(quaternions, -1)
        rot_mats = torch.stack((1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w), 2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w), 2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)), dim=-1).reshape(quaternions.shape[:-1] + (3, 3))
    else:
        w, v = quaternions.split([1, 3], dim=-1)
        rot_mats = 2 * (w.unsqueeze(-1) * skew(v) + v.unsqueeze(-1) * v.unsqueeze(-2))
        diag = torch.diagonal(rot_mats, dim1=-2, dim2=-1)
        diag += w * w - (v.unsqueeze(-2) @ v.unsqueeze(-1)).squeeze(-1)
    return rot_mats


def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw (torch.Tensor): (*)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    rot_mats = yaw.new_zeros(yaw.shape + (3, 3))
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats


def pnp_denormalize(offset, pose_norm):
    pose = torch.empty_like(pose_norm)
    pose[..., 3:] = pose_norm[..., 3:]
    pose[..., :3] = pose_norm[..., :3] - ((yaw_to_rot_mat(pose_norm[..., 3]) if pose_norm.size(-1) == 4 else quaternion_to_rot_mat(pose_norm[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    return pose


def pnp_normalize(x3d, pose=None, detach_transformation=True):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        pose (torch.Tensor | None): Shape (*, 4)
        detach_transformation (bool)

    Returns:
        Tuple[torch.Tensor]:
            offset: Shape (*, 1, 3)
            x3d_norm: Shape (*, n, 3), normalized x3d
            pose_norm: Shape (*, ), transformed pose
    """
    offset = torch.mean(x3d.detach() if detach_transformation else x3d, dim=-2)
    x3d_norm = x3d - offset.unsqueeze(-2)
    if pose is not None:
        pose_norm = torch.empty_like(pose)
        pose_norm[..., 3:] = pose[..., 3:]
        pose_norm[..., :3] = pose[..., :3] + ((yaw_to_rot_mat(pose[..., 3]) if pose.size(-1) == 4 else quaternion_to_rot_mat(pose[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    else:
        pose_norm = None
    return offset, x3d_norm, pose_norm


class EProPnPBase(torch.nn.Module, metaclass=ABCMeta):
    """
    End-to-End Probabilistic Perspective-n-Points.

    Args:
        mc_samples (int): Number of total Monte Carlo samples
        num_iter (int): Number of AMIS iterations
        normalize (bool)
        eps (float)
        solver (dict): PnP solver
    """

    def __init__(self, mc_samples=512, num_iter=4, normalize=False, eps=1e-05, solver=None):
        super(EProPnPBase, self).__init__()
        assert num_iter > 0
        assert mc_samples % num_iter == 0
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.iter_samples = self.mc_samples // self.num_iter
        self.eps = eps
        self.normalize = normalize
        self.solver = solver

    @abstractmethod
    def allocate_buffer(self, *args, **kwargs):
        pass

    @abstractmethod
    def initial_fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_new_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_old_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def estimate_params(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    def monte_carlo_forward(self, x3d, x2d, w2d, camera, cost_fun, pose_init=None, force_init_solve=True, **kwargs):
        """
        Monte Carlo PnP forward. Returns weighted pose samples drawn from the probability
        distribution of pose defined by the correspondences {x_{3D}, x_{2D}, w_{2D}}.

        Args:
            x3d (Tensor): Shape (num_obj, num_points, 3)
            x2d (Tensor): Shape (num_obj, num_points, 2)
            w2d (Tensor): Shape (num_obj, num_points, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (Tensor | None): Shape (num_obj, 4 or 7), optional. The target pose
                (y_{gt}) can be passed for training with Monte Carlo pose loss
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None

        Returns:
            Tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7), PnP solution y*
                cost (Tensor | None): Shape (num_obj, ), is not None when with_cost=True
                pose_opt_plus (Tensor | None): Shape (num_obj, 4 or 7), y* + Δy, used in derivative
                    regularization loss, is not None when with_pose_opt_plus=True, can be backpropagated
                pose_samples (Tensor): Shape (mc_samples, num_obj, 4 or 7)
                pose_sample_logweights (Tensor): Shape (mc_samples, num_obj), can be backpropagated
                cost_init (Tensor | None): Shape (num_obj, ), is None when pose_init is None, can be
                    backpropagated
        """
        if self.normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)
        assert x3d.dim() == x2d.dim() == w2d.dim() == 3
        num_obj = x3d.size(0)
        evaluate_fun = partial(evaluate_pnp, x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
        cost_init = evaluate_fun(pose=pose_init)[1] if pose_init is not None else None
        pose_opt, pose_cov, cost, pose_opt_plus = self.solver(x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, cost_init=cost_init, with_pose_cov=True, force_init_solve=force_init_solve, normalize_override=False, **kwargs)
        if num_obj > 0:
            pose_samples = x3d.new_empty((self.num_iter, self.iter_samples) + pose_opt.size())
            logprobs = x3d.new_empty((self.num_iter, self.num_iter, self.iter_samples, num_obj))
            cost_pred = x3d.new_empty((self.num_iter, self.iter_samples, num_obj))
            distr_params = self.allocate_buffer(num_obj, dtype=x3d.dtype, device=x3d.device)
            with torch.no_grad():
                self.initial_fit(pose_opt, pose_cov, camera, *distr_params)
            for i in range(self.num_iter):
                new_trans_distr, new_rot_distr = self.gen_new_distr(i, *distr_params)
                pose_samples[i, :, :, :3] = new_trans_distr.sample((self.iter_samples,))
                pose_samples[i, :, :, 3:] = new_rot_distr.sample((self.iter_samples,))
                cost_pred[i] = evaluate_fun(pose=pose_samples[i])[1]
                logprobs[i, :i + 1] = new_trans_distr.log_prob(pose_samples[:i + 1, :, :, :3]) + new_rot_distr.log_prob(pose_samples[:i + 1, :, :, 3:]).flatten(2)
                if i > 0:
                    old_trans_distr, old_rot_distr = self.gen_old_distr(i, *distr_params)
                    logprobs[:i, i] = old_trans_distr.log_prob(pose_samples[i, :, :, :3]) + old_rot_distr.log_prob(pose_samples[i, :, :, 3:]).flatten(2)
                mix_logprobs = torch.logsumexp(logprobs[:i + 1, :i + 1], dim=0) - math.log(i + 1)
                pose_sample_logweights = -cost_pred[:i + 1] - mix_logprobs
                if i == self.num_iter - 1:
                    break
                with torch.no_grad():
                    self.estimate_params(i, pose_samples[:i + 1].reshape(((i + 1) * self.iter_samples,) + pose_opt.size()), pose_sample_logweights.reshape((i + 1) * self.iter_samples, num_obj), *distr_params)
            pose_samples = pose_samples.reshape((self.mc_samples,) + pose_opt.size())
            pose_sample_logweights = pose_sample_logweights.reshape(self.mc_samples, num_obj)
        else:
            pose_samples = x2d.new_zeros((self.mc_samples,) + pose_opt.size())
            pose_sample_logweights = x3d.reshape(self.mc_samples, 0) + x2d.reshape(self.mc_samples, 0) + w2d.reshape(self.mc_samples, 0)
        if self.normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            pose_samples = pnp_denormalize(transform, pose_samples)
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_init


class VonMisesUniformMix(VonMises):

    def __init__(self, loc, concentration, uniform_mix=0.25, **kwargs):
        super(VonMisesUniformMix, self).__init__(loc, concentration, **kwargs)
        self.uniform_mix = uniform_mix

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        assert len(sample_shape) == 1
        x = np.empty(tuple(self._extended_shape(sample_shape)), dtype=np.float32)
        uniform_samples = round(sample_shape[0] * self.uniform_mix)
        von_mises_samples = sample_shape[0] - uniform_samples
        x[:uniform_samples] = np.random.uniform(-math.pi, math.pi, size=tuple(self._extended_shape((uniform_samples,))))
        x[uniform_samples:] = np.random.vonmises(self.loc.cpu().numpy(), self.concentration.cpu().numpy(), size=tuple(self._extended_shape((von_mises_samples,))))
        return torch.from_numpy(x)

    def log_prob(self, value):
        von_mises_log_prob = super(VonMisesUniformMix, self).log_prob(value) + np.log(1 - self.uniform_mix)
        log_prob = torch.logaddexp(von_mises_log_prob, torch.full_like(von_mises_log_prob, math.log(self.uniform_mix / (2 * math.pi))))
        return log_prob


def cholesky_wrapper(mat, default_diag=None, force_cpu=True):
    device = mat.device
    if force_cpu:
        mat = mat.cpu()
    try:
        tril = torch.cholesky(mat, upper=False)
    except RuntimeError:
        n_dims = mat.size(-1)
        tril = []
        default_tril_single = torch.diag(mat.new_tensor(default_diag)) if default_diag is not None else torch.eye(n_dims, dtype=mat.dtype, device=mat.device)
        for cov in mat.reshape(-1, n_dims, n_dims):
            try:
                tril.append(torch.cholesky(cov, upper=False))
            except RuntimeError:
                tril.append(default_tril_single)
        tril = torch.stack(tril, dim=0).reshape(mat.shape)
    return tril


class EProPnP4DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 4DoF pose estimation.
    The pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: 0.75 von Mises distribution + 0.25 uniform distribution
    """

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_mode = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        rot_kappa = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_mode, rot_kappa

    def initial_fit(self, pose_opt, pose_cov, camera, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        trans_mode[0], rot_mode[0] = pose_opt.split([3, 1], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3], [1.0, 1.0, 4.0])
        rot_kappa[0] = 0.33 / pose_cov[:, 3, 3, None].clamp(min=self.eps)

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = VonMisesUniformMix(rot_mode[iter_id], rot_kappa[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        mix_trans_distr = MultivariateStudentT(3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = VonMisesUniformMix(rot_mode[:iter_id, None], rot_kappa[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1) * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov, [1.0, 1.0, 4.0])
        mean_vector = pose_samples.new_empty((pose_samples.size(1), 2))
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].sin(), dim=0, out=mean_vector[:, :1])
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].cos(), dim=0, out=mean_vector[:, 1:])
        rot_mode[iter_id + 1] = torch.atan2(mean_vector[:, :1], mean_vector[:, 1:])
        r_sq = torch.square(mean_vector).sum(dim=-1, keepdim=True)
        rot_kappa[iter_id + 1] = 0.33 * r_sq.sqrt().clamp(min=self.eps) * (2 - r_sq) / (1 - r_sq).clamp(min=self.eps)


class EProPnP6DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 6DoF pose estimation.
    The pose is parameterized as [x, y, z, w, i, j, k], where [w, i, j, k]
    is the unit quaternion.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: angular central Gaussian distribution
    """

    def __init__(self, *args, acg_mle_iter=3, acg_dispersion=0.001, **kwargs):
        super(EProPnP6DoF, self).__init__(*args, **kwargs)
        self.acg_mle_iter = acg_mle_iter
        self.acg_dispersion = acg_dispersion

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_cov_tril = torch.empty((self.num_iter, num_obj, 4, 4), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_cov_tril

    def initial_fit(self, pose_opt, pose_cov, camera, trans_mode, trans_cov_tril, rot_cov_tril):
        trans_mode[0], rot_mode = pose_opt.split([3, 4], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3])
        eye_4 = torch.eye(4, dtype=pose_opt.dtype, device=pose_opt.device)
        transform_mat = camera.get_quaternion_transfrom_mat(rot_mode)
        rot_cov = (transform_mat @ pose_cov[:, 3:, 3:].inverse() @ transform_mat.transpose(-1, -2) + eye_4).inverse()
        rot_cov.div_(rot_cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)[..., None, None])
        rot_cov_tril[0] = cholesky_wrapper(rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = AngularCentralGaussian(rot_cov_tril[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        mix_trans_distr = MultivariateStudentT(3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = AngularCentralGaussian(rot_cov_tril[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights, trans_mode, trans_cov_tril, rot_cov_tril):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1) * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov)
        eye_4 = torch.eye(4, dtype=pose_samples.dtype, device=pose_samples.device)
        rot = pose_samples[..., 3:]
        r_r_t = rot[:, :, :, None] * rot[:, :, None, :]
        rot_cov = eye_4.expand(pose_samples.size(1), 4, 4).clone()
        for _ in range(self.acg_mle_iter):
            M = rot[:, :, None, :] @ rot_cov.inverse() @ rot[:, :, :, None]
            invM_weighted = sample_weights_norm[..., None, None] / M.clamp(min=self.eps)
            invM_weighted_norm = invM_weighted / invM_weighted.sum(dim=0)
            rot_cov = (invM_weighted_norm * r_r_t).sum(dim=0) + eye_4 * self.eps
        rot_cov_tril[iter_id + 1] = cholesky_wrapper(rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))


def solve_wrapper(b, A):
    if A.numel() > 0:
        return torch.linalg.solve(A, b)
    else:
        return b + A.reshape_as(b)


class LMSolver(nn.Module):
    """
    Levenberg-Marquardt solver, with fixed number of iterations.

    - For 4DoF case, the pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    - For 6DoF case, the pose is parameterized as [x, y, z, w, i, j, k], where
    [w, i, j, k] is the unit quaternion.
    """

    def __init__(self, dof=4, num_iter=10, min_lm_diagonal=1e-06, max_lm_diagonal=1e+32, min_relative_decrease=0.001, initial_trust_region_radius=30.0, max_trust_region_radius=1e+16, eps=1e-05, normalize=False, init_solver=None):
        super(LMSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.min_lm_diagonal = min_lm_diagonal
        self.max_lm_diagonal = max_lm_diagonal
        self.min_relative_decrease = min_relative_decrease
        self.initial_trust_region_radius = initial_trust_region_radius
        self.max_trust_region_radius = max_trust_region_radius
        self.eps = eps
        self.normalize = normalize
        self.init_solver = init_solver

    def forward(self, x3d, x2d, w2d, camera, cost_fun, with_pose_opt_plus=False, pose_init=None, normalize_override=None, **kwargs):
        if isinstance(normalize_override, bool):
            normalize = normalize_override
        else:
            normalize = self.normalize
        if normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)
        pose_opt, pose_cov, cost = self.solve(x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, **kwargs)
        if with_pose_opt_plus:
            step = self.gn_step(x3d, x2d, w2d, pose_opt, camera, cost_fun)
            pose_opt_plus = self.pose_add(pose_opt, step, camera)
        else:
            pose_opt_plus = None
        if normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            if pose_cov is not None:
                raise NotImplementedError('Normalized covariance unsupported')
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, pose_cov, cost, pose_opt_plus

    def solve(self, x3d, x2d, w2d, camera, cost_fun, pose_init=None, cost_init=None, with_pose_cov=False, with_cost=False, force_init_solve=False, fast_mode=False):
        """
        Args:
            x3d (Tensor): Shape (num_obj, num_pts, 3)
            x2d (Tensor): Shape (num_obj, num_pts, 2)
            w2d (Tensor): Shape (num_obj, num_pts, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (None | Tensor): Shape (num_obj, 4 or 7) in [x, y, z, yaw], optional
            cost_init (None | Tensor): Shape (num_obj, ), PnP cost of pose_init, optional
            with_pose_cov (bool): Whether to compute the covariance of pose_opt
            with_cost (bool): Whether to compute the cost of pose_opt
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None
            fast_mode (bool): Fall back to Gauss-Newton for fast inference

        Returns:
            tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7)
                pose_cov (Tensor | None): Shape (num_obj, 4, 4) or (num_obj, 6, 6), covariance
                    of local pose parameterization
                cost (Tensor | None): Shape (num_obj, )
        """
        with torch.no_grad():
            num_obj, num_pts, _ = x2d.size()
            tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)
            if num_obj > 0:
                evaluate_fun = partial(evaluate_pnp, x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, clip_jac=not fast_mode)
                if pose_init is None or force_init_solve:
                    assert self.init_solver is not None
                    if pose_init is None:
                        pose_init_solve, _, _ = self.init_solver.solve(x3d, x2d, w2d, camera, cost_fun, fast_mode=fast_mode)
                        pose_opt = pose_init_solve
                    else:
                        if cost_init is None:
                            cost_init = evaluate_fun(pose=pose_init, out_cost=True)[1]
                        pose_init_solve, _, cost_init_solve = self.init_solver.solve(x3d, x2d, w2d, camera, cost_fun, with_cost=True, fast_mode=fast_mode)
                        use_init = cost_init < cost_init_solve
                        pose_init_solve[use_init] = pose_init[use_init]
                        pose_opt = pose_init_solve
                else:
                    pose_opt = pose_init.clone()
                jac = torch.empty((num_obj, num_pts * 2, self.dof), **tensor_kwargs)
                residual = torch.empty((num_obj, num_pts * 2), **tensor_kwargs)
                cost = torch.empty((num_obj,), **tensor_kwargs)
                if fast_mode:
                    for i in range(self.num_iter):
                        evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                        jac_t = jac.transpose(-1, -2)
                        jtj = jac_t @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)
                        diagonal += self.eps
                        gradient = jac_t @ residual.unsqueeze(-1)
                        if self.dof == 4:
                            pose_opt -= solve_wrapper(gradient, jtj).squeeze(-1)
                        else:
                            step = -solve_wrapper(gradient, jtj).squeeze(-1)
                            pose_opt[..., :3] += step[..., :3]
                            pose_opt[..., 3:] = F.normalize(pose_opt[..., 3:] + (camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]).squeeze(-1), dim=-1)
                else:
                    evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                    jac_new = torch.empty_like(jac)
                    residual_new = torch.empty_like(residual)
                    cost_new = torch.empty_like(cost)
                    radius = x2d.new_full((num_obj,), self.initial_trust_region_radius)
                    decrease_factor = x2d.new_full((num_obj,), 2.0)
                    step_is_successful = x2d.new_zeros((num_obj,), dtype=torch.bool)
                    i = 0
                    while i < self.num_iter:
                        self._lm_iter(pose_opt, jac, residual, cost, jac_new, residual_new, cost_new, step_is_successful, radius, decrease_factor, evaluate_fun, camera)
                        i += 1
                    if with_pose_cov:
                        jac[step_is_successful] = jac_new[step_is_successful]
                        jtj = jac.transpose(-1, -2) @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)
                        diagonal += self.eps
                    if with_cost:
                        cost[step_is_successful] = cost_new[step_is_successful]
                if with_pose_cov:
                    pose_cov = torch.inverse(jtj)
                else:
                    pose_cov = None
                if not with_cost:
                    cost = None
            else:
                pose_opt = torch.empty((0, 4 if self.dof == 4 else 7), **tensor_kwargs)
                pose_cov = torch.empty((0, self.dof, self.dof), **tensor_kwargs) if with_pose_cov else None
                cost = torch.empty((0,), **tensor_kwargs) if with_cost else None
            return pose_opt, pose_cov, cost

    def _lm_iter(self, pose_opt, jac, residual, cost, jac_new, residual_new, cost_new, step_is_successful, radius, decrease_factor, evaluate_fun, camera):
        jac[step_is_successful] = jac_new[step_is_successful]
        residual[step_is_successful] = residual_new[step_is_successful]
        cost[step_is_successful] = cost_new[step_is_successful]
        residual_ = residual.unsqueeze(-1)
        jac_t = jac.transpose(-1, -2)
        jtj = jac_t @ jac
        jtj_lm = jtj.clone()
        diagonal = torch.diagonal(jtj_lm, dim1=-2, dim2=-1)
        diagonal += diagonal.clamp(min=self.min_lm_diagonal, max=self.max_lm_diagonal) / radius[:, None] + self.eps
        gradient = jac_t @ residual_
        step_ = -solve_wrapper(gradient, jtj_lm)
        pose_new = self.pose_add(pose_opt, step_.squeeze(-1), camera)
        evaluate_fun(pose=pose_new, out_jacobian=jac_new, out_residual=residual_new, out_cost=cost_new)
        model_cost_change = -(step_.transpose(-1, -2) @ (jtj @ step_ / 2 + gradient)).flatten()
        relative_decrease = (cost - cost_new) / model_cost_change
        torch.bitwise_and(relative_decrease >= self.min_relative_decrease, model_cost_change > 0.0, out=step_is_successful)
        pose_opt[step_is_successful] = pose_new[step_is_successful]
        radius[step_is_successful] /= (1.0 - (2.0 * relative_decrease[step_is_successful] - 1.0) ** 3).clamp(min=1.0 / 3.0)
        radius.clamp_(max=self.max_trust_region_radius, min=self.eps)
        decrease_factor.masked_fill_(step_is_successful, 2.0)
        radius[~step_is_successful] /= decrease_factor[~step_is_successful]
        decrease_factor[~step_is_successful] *= 2.0
        return

    def gn_step(self, x3d, x2d, w2d, pose, camera, cost_fun):
        residual, _, jac = evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_jacobian=True, out_residual=True)
        jac_t = jac.transpose(-1, -2)
        jtj = jac_t @ jac
        jtj = jtj + torch.eye(self.dof, device=jtj.device, dtype=jtj.dtype) * self.eps
        gradient = jac_t @ residual.unsqueeze(-1)
        step = -solve_wrapper(gradient, jtj).squeeze(-1)
        return step

    def pose_add(self, pose_opt, step, camera):
        if self.dof == 4:
            pose_new = pose_opt + step
        else:
            pose_new = torch.cat((pose_opt[..., :3] + step[..., :3], F.normalize(pose_opt[..., 3:] + (camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]).squeeze(-1), dim=-1)), dim=-1)
        return pose_new


class RSLMSolver(LMSolver):
    """
    Random Sample Levenberg-Marquardt solver, a generalization of RANSAC.
    Used for initialization in ambiguous problems.
    """

    def __init__(self, num_points=16, num_proposals=64, num_iter=3, **kwargs):
        super(RSLMSolver, self).__init__(num_iter=num_iter, **kwargs)
        self.num_points = num_points
        self.num_proposals = num_proposals

    def center_based_init(self, x2d, x3d, camera, eps=1e-06):
        x2dc = solve_wrapper(F.pad(x2d, [0, 1], mode='constant', value=1.0).transpose(-1, -2), camera.cam_mats).transpose(-1, -2)
        x2dc = x2dc[..., :2] / x2dc[..., 2:].clamp(min=eps)
        x2dc_std, x2dc_mean = torch.std_mean(x2dc, dim=-2)
        x3d_std = torch.std(x3d, dim=-2)
        if self.dof == 4:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.0) * (x3d_std[..., 1] / x2dc_std[..., 1].clamp(min=eps)).unsqueeze(-1)
        else:
            t_vec = F.pad(x2dc_mean, [0, 1], mode='constant', value=1.0) * (math.sqrt(2 / 3) * x3d_std.norm(dim=-1) / x2dc_std.norm(dim=-1).clamp(min=eps)).unsqueeze(-1)
        return t_vec

    def solve(self, x3d, x2d, w2d, camera, cost_fun, **kwargs):
        with torch.no_grad():
            bs, pn, _ = x2d.size()
            if bs > 0:
                mean_weight = w2d.mean(dim=-1).reshape(1, bs, pn).expand(self.num_proposals, -1, -1)
                inds = torch.multinomial(mean_weight.reshape(-1, pn), self.num_points).reshape(self.num_proposals, bs, self.num_points)
                bs_inds = torch.arange(bs, device=inds.device)
                inds += (bs_inds * pn)[:, None]
                x2d_samples = x2d.reshape(-1, 2)[inds]
                x3d_samples = x3d.reshape(-1, 3)[inds]
                w2d_samples = w2d.reshape(-1, 2)[inds]
                pose_init = x2d.new_empty((self.num_proposals, bs, 4 if self.dof == 4 else 7))
                pose_init[..., :3] = self.center_based_init(x2d, x3d, camera)
                if self.dof == 4:
                    pose_init[..., 3] = torch.rand((self.num_proposals, bs), dtype=x2d.dtype, device=x2d.device) * (2 * math.pi)
                else:
                    pose_init[..., 3:] = torch.randn((self.num_proposals, bs, 4), dtype=x2d.dtype, device=x2d.device)
                    q_norm = pose_init[..., 3:].norm(dim=-1)
                    pose_init[..., 3:] /= q_norm.unsqueeze(-1)
                    pose_init.view(-1, 7)[(q_norm < self.eps).flatten(), 3:] = x2d.new_tensor([1, 0, 0, 0])
                camera_expand = camera.shallow_copy()
                camera_expand.repeat_(self.num_proposals)
                cost_fun_expand = cost_fun.shallow_copy()
                cost_fun_expand.repeat_(self.num_proposals)
                pose, _, _ = super(RSLMSolver, self).solve(x3d_samples.reshape(self.num_proposals * bs, self.num_points, 3), x2d_samples.reshape(self.num_proposals * bs, self.num_points, 2), w2d_samples.reshape(self.num_proposals * bs, self.num_points, 2), camera_expand, cost_fun_expand, pose_init=pose_init.reshape(self.num_proposals * bs, pose_init.size(-1)), **kwargs)
                pose = pose.reshape(self.num_proposals, bs, pose.size(-1))
                cost = evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_cost=True)[1]
                min_cost, min_cost_ind = cost.min(dim=0)
                pose = pose[min_cost_ind, torch.arange(bs, device=pose.device)]
            else:
                pose = x2d.new_empty((0, 4 if self.dof == 4 else 7))
                min_cost = x2d.new_empty((0,))
            return pose, None, min_cost


class CosineAngleLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * cosine_angle_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


def get_overlap_boxes(bboxes1, bboxes2):
    overlap_boxes = bboxes1.new_empty((bboxes1.size(0), bboxes2.size(0), 4))
    bboxes1 = bboxes1.unsqueeze(1)
    overlap_boxes[..., :2] = torch.max(bboxes1[..., :2], bboxes2[:, :2])
    overlap_boxes[..., 2:] = torch.min(bboxes1[..., 2:], bboxes2[:, 2:])
    pos_mask = overlap_boxes[..., 2:] - overlap_boxes[..., :2] > 0
    pos_mask = torch.all(pos_mask, dim=2)
    return pos_mask, overlap_boxes


def logsumexp_across_rois(roi_inputs, rois):
    """
    Args:
        roi_inputs (torch.Tensor): shape (bn, chn, rh, rw)
        rois (torch.Tensor): shape (bn, 5)
    Returns:
        Tensor, shape (bn, chn, rh, rw)
    """
    bn, kn, rh, rw = roi_inputs.size()
    rois_logsumexp = roi_inputs.clone()
    if bn > 0:
        roi_ids = rois[:, 0].round().int()
        roi_bboxes = rois[:, 1:]
        for roi_id in range(max(roi_ids) + 1):
            ids = (roi_id == roi_ids).nonzero(as_tuple=False).squeeze(-1)
            if len(ids) <= 1:
                continue
            boxes = roi_bboxes[ids]
            pos_mask, overlap_boxes = get_overlap_boxes(boxes, boxes)
            pos_mask.fill_diagonal_(False)
            for id_self, pos_mask_single, overlap_boxes_single in zip(ids, pos_mask, overlap_boxes):
                if not any(pos_mask_single):
                    continue
                ids_overlap = ids[pos_mask_single]
                num_overlap = ids_overlap.size(0)
                roi_input_self = roi_inputs[id_self]
                roi_inputs_overlap = roi_inputs[ids_overlap]
                bbox_self = roi_bboxes[id_self]
                bbox_overlap = roi_bboxes[ids_overlap]
                overlap_boxes_ = overlap_boxes_single[pos_mask_single]
                wh_bbox_self = bbox_self[2:] - bbox_self[:2]
                wh_bbox_overlap = bbox_overlap[:, 2:] - bbox_overlap[:, :2]
                scale = wh_bbox_self / wh_bbox_overlap
                xy_tl_in_bbox_overlap = 2 * (overlap_boxes_[:, :2] - bbox_overlap[:, :2]) / (bbox_overlap[:, 2:] - bbox_overlap[:, :2]) - 1
                xy_tl_in_bbox_self = 2 * (overlap_boxes_[:, :2] - bbox_self[:2]) / (bbox_self[2:] - bbox_self[:2]) - 1
                affine_mat = wh_bbox_self.new_zeros((num_overlap, 2, 3))
                affine_mat[:, 0, 0] = scale[:, 0]
                affine_mat[:, 1, 1] = scale[:, 1]
                affine_mat[:, :, 2] = xy_tl_in_bbox_overlap - scale * xy_tl_in_bbox_self
                grid = F.affine_grid(affine_mat, (num_overlap, 1, rh, rw), align_corners=False)
                roi_inputs_resample = F.grid_sample(roi_inputs_overlap, grid, padding_mode='border', align_corners=False)
                valid_grid = torch.all((grid > -1) & (grid < 1), dim=3).unsqueeze(1)
                roi_inputs_resampled_cat_self = roi_inputs_resample.new_empty((num_overlap + 1, kn, rh, rw))
                roi_inputs_resampled_cat_self[:-1] = roi_inputs_resample.masked_fill(~valid_grid, float('-inf'))
                roi_inputs_resampled_cat_self[-1] = roi_input_self
                rois_logsumexp[id_self] = roi_inputs_resampled_cat_self.logsumexp(dim=0)
    return rois_logsumexp


class MVDGaussianMixtureNLLLoss(nn.Module):

    def __init__(self, dim=1, reduction='mean', loss_weight=1.0, adaptive_weight=True, momentum=0.1, eps=0.0001):
        super(MVDGaussianMixtureNLLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.adaptive_weight = adaptive_weight
        self.momentum = momentum
        self.register_buffer('mean_inv_std', torch.tensor(1, dtype=torch.float))
        self.dim = dim
        self.eps = eps

    def forward(self, pred, target, logstd=None, logmixweight=None, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mvd_gaussian_mixture_nll_loss(pred, target, weight, logstd=logstd, logmixweight=logmixweight, adaptive_weight=self.adaptive_weight, momentum=self.momentum, mean_inv_std=self.mean_inv_std, dim=self.dim, eps=self.eps, training=self.training, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class DeformableAttentionSampler(nn.Module):

    def __init__(self, embed_dims=256, num_heads=8, num_points=32, stride=4, ffn_cfg=dict(type='FFN', embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.1, act_cfg=dict(type='ReLU', inplace=True)), norm_cfg=dict(type='LN'), init_cfg=None):
        super(DeformableAttentionSampler, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.stride = stride
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_points * 2)
        self.out_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.layer_norms = nn.ModuleList([build_norm_layer(norm_cfg, self.embed_dims)[1] for _ in range(2)])
        self.ffn = build_feedforward_network(self.ffn_cfg, dict(type='FFN'))
        self.init_weights()

    def init_weights(self):
        xavier_init(self.sampling_offsets, gain=2.5, distribution='uniform')
        for m in [self.layer_norms, self.ffn]:
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self._is_init = True

    def forward(self, query, obj_emb, key, value, img_dense_x2d, img_dense_x2d_mask, obj_xy_point, strides, obj_img_ind):
        """
        Args:
            query: shape (num_obj, num_head, 1, head_emb_dim)
            obj_emb: shape (num_obj, embed_dim)
            key: shape (num_img, embed_dim, h, w)
            value: shape (num_img, embed_dim, h, w)
            img_dense_x2d: shape (num_img, 2, h, w)
            img_dense_x2d_mask: shape (num_img, 1, h, w)
            obj_xy_point: shape (num_obj, 2)
            strides: shape (num_obj, )
            obj_img_ind: shape (num_obj, )

        Returns:
            tuple[tensor]:
                output (num_obj_sample, embed_dim)
                v_samples (num_obj_sample, num_head, head_emb_dim, num_point)
                a_samples (num_obj_sample, num_head, 1, num_point)
                mask_samples (num_obj_sample, num_head, 1, num_point)
                x2d_samples (num_obj_sample, num_head, 2, num_point)
        """
        num_obj_samples = query.size(0)
        num_img, _, h_out, w_out = key.size()
        head_emb_dim = self.embed_dims // self.num_heads
        offsets = self.sampling_offsets(obj_emb).reshape(num_obj_samples, self.num_heads, self.num_points, 2)
        sampling_location = obj_xy_point[:, None, None] + offsets * strides[:, None, None, None]
        hw_img = key.new_tensor(key.shape[-2:]) * self.stride
        sampling_grid = sampling_location * (2 / hw_img[[1, 0]]) - 1
        sampling_grid = sampling_grid.transpose(1, 0).reshape(self.num_heads, num_obj_samples, self.num_points, 1, 2)
        img_ind_grid = (obj_img_ind + 0.5) * (2 / num_img) - 1.0
        sampling_grid = torch.cat((sampling_grid, img_ind_grid[None, :, None, None, None].expand(self.num_heads, -1, self.num_points, 1, 1)), dim=-1)
        k_samples = F.grid_sample(key.reshape(num_img, self.num_heads, head_emb_dim, h_out, w_out).permute(1, 2, 0, 3, 4), sampling_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).permute(2, 0, 1, 3)
        v_samples = F.grid_sample(value.reshape(num_img, self.num_heads, head_emb_dim, h_out, w_out).permute(1, 2, 0, 3, 4), sampling_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).permute(2, 0, 1, 3)
        x2d_samples = F.grid_sample(img_dense_x2d.transpose(1, 0)[None].expand(self.num_heads, -1, -1, -1, -1), sampling_grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1).permute(2, 0, 1, 3)
        mask_samples = F.grid_sample(img_dense_x2d_mask.transpose(1, 0)[None].expand(self.num_heads, -1, -1, -1, -1), sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(-1).permute(2, 0, 1, 3)
        a_samples = query @ k_samples / np.sqrt(head_emb_dim)
        a_samples_softmax = a_samples.softmax(dim=-1) * mask_samples
        output = v_samples @ a_samples_softmax.reshape(num_obj_samples, self.num_heads, self.num_points, 1)
        output = output.reshape(num_obj_samples, self.embed_dims)
        output = self.out_proj(output) + obj_emb
        output = self.layer_norms[0](output)
        output = self.ffn(output, output)
        output = self.layer_norms[1](output)
        return output, v_samples, a_samples, mask_samples, x2d_samples


class GroupLinear(nn.Module):

    def __init__(self, in_features, out_features, groups, bias=True):
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(groups, out_features // groups, in_features // groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(groups, out_features // groups))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        """
        Args:
            input (Tensor): shape (*, in_features)
        """
        batch_size = input.shape[:-1]
        if self.bias is not None:
            output = self.weight @ input.reshape(*batch_size, self.groups, self.in_features // self.groups, 1) + self.bias[..., None]
        else:
            output = self.weight @ input.reshape(*batch_size, self.groups, self.in_features // self.groups, 1)
        return output.reshape(*batch_size, self.out_features)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CDPN,
     lambda: ([], {'backbone': _mock_layer(), 'rot_head_net': _mock_layer(), 'trans_head_net': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GroupLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RotHeadNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransHeadNet,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_tjiiv_cprg_EPro_PnP(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

