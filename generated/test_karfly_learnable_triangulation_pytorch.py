import sys
_module = sys.modules[__name__]
del sys
mvn = _module
datasets = _module
human36m = _module
action_to_bbox_filename = _module
action_to_una_dinosauria = _module
utils = _module
models = _module
loss = _module
pose_resnet = _module
triangulation = _module
v2v = _module
cfg = _module
img = _module
misc = _module
multiview = _module
op = _module
vis = _module
volumetric = _module
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


from collections import defaultdict


import numpy as np


import torch


from torch.utils.data import Dataset


from torch import nn


import logging


import torch.nn as nn


from collections import OrderedDict


from copy import deepcopy


import random


from scipy.optimize import least_squares


import torch.nn.functional as F


import re


import scipy.ndimage


import matplotlib


from matplotlib import pylab as plt


import time


from itertools import islice


import copy


from torch import autograd


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.nn.parallel import DistributedDataParallel


class KeypointsMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMSESmoothLoss(nn.Module):

    def __init__(self, threshold=400):
        super().__init__()
        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * self.threshold ** 0.9
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMAELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsL2Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class VolumetricCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0
        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]
            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)
            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)
            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * -torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-06)
                n_losses += 1
        return loss / n_losses


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


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class GlobalAveragePoolingHead(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512, momentum=BN_MOMENTUM), nn.MaxPool2d(2), nn.ReLU(inplace=True), nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256, momentum=BN_MOMENTUM), nn.MaxPool2d(2), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, n_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)
        out = self.head(x)
        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, num_joints, num_input_channels=3, deconv_with_bias=False, num_deconv_layers=3, num_deconv_filters=(256, 256, 256), num_deconv_kernels=(4, 4, 4), final_conv_kernel=1, alg_confidences=False, vol_confidences=False):
        super().__init__()
        self.num_joints = num_joints
        self.num_input_channels = num_input_channels
        self.inplanes = 64
        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels = num_deconv_layers, num_deconv_filters, num_deconv_kernels
        self.final_conv_kernel = final_conv_kernel
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if alg_confidences:
            self.alg_confidences = GlobalAveragePoolingHead(512 * block.expansion, num_joints)
        if vol_confidences:
            self.vol_confidences = GlobalAveragePoolingHead(512 * block.expansion, 32)
        self.deconv_layers = self._make_deconv_layer(self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels)
        self.final_layer = nn.Conv2d(in_channels=self.num_deconv_filters[-1], out_channels=self.num_joints, kernel_size=self.final_conv_kernel, stride=1, padding=1 if self.final_conv_kernel == 3 else 0)

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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
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
        alg_confidences = None
        if hasattr(self, 'alg_confidences'):
            alg_confidences = self.alg_confidences(x)
        vol_confidences = None
        if hasattr(self, 'vol_confidences'):
            vol_confidences = self.vol_confidences(x)
        x = self.deconv_layers(x)
        features = x
        x = self.final_layer(x)
        heatmaps = x
        return heatmaps, features, alg_confidences, vol_confidences


class RANSACTriangulationNet(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        self.direct_optimization = config.model.direct_optimization

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]
        images = images.view(-1, *images.shape[2:])
        heatmaps, _, _, _ = self.backbone(images)
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
        _, max_indicies = torch.max(heatmaps.view(batch_size, n_views, n_joints, -1), dim=-1)
        keypoints_2d = torch.stack([max_indicies % heatmap_shape[1], max_indicies // heatmap_shape[1]], dim=-1)
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matricies.detach().cpu().numpy()
        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = self.triangulate_ransac(current_proj_matricies, points, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d
        keypoints_3d = torch.from_numpy(keypoints_3d).type(torch.float)
        confidences = torch.from_numpy(confidences).type(torch.float)
        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matricies, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2
        proj_matricies = np.array(proj_matricies)
        points = np.array(points)
        n_views = len(points)
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2))
            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]
            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)
            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()
        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]
        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)
        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean
        if direct_optimization:

            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals
            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')
            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)
        return keypoint_3d_in_base_camera, inlier_list


class AlgebraicTriangulationNet(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()
        self.use_confidences = config.model.use_confidences
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.use_confidences:
            config.model.backbone.alg_confidences = True
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]
        images = images.view(-1, *images.shape[2:])
        if self.use_confidences:
            heatmaps, _, alg_confidences, _ = self.backbone(images)
        else:
            heatmaps, _, _, _ = self.backbone(images)
            alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float)
        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-05
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed
        try:
            keypoints_3d = multiview.triangulate_batch_of_points(proj_matricies, keypoints_2d, confidences_batch=alg_confidences)
        except RuntimeError as e:
            None
            None
            None
            None
            exit()
        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences


class Basic3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Pool3DBlock(nn.Module):

    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Res3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes), nn.ReLU(True), nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(out_planes))
        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(out_planes))

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Upsample3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert kernel_size == 2
        assert stride == 2
        self.block = nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0), nn.BatchNorm3d(out_planes), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128, 128)
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(128, 128)
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(128, 128)
        self.mid_res = Res3DBlock(128, 128)
        self.decoder_res5 = Res3DBlock(128, 128)
        self.decoder_upsample5 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res4 = Res3DBlock(128, 128)
        self.decoder_upsample4 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res3 = Res3DBlock(128, 128)
        self.decoder_upsample3 = Upsample3DBlock(128, 128, 2, 2)
        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)
        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)
        self.skip_res3 = Res3DBlock(128, 128)
        self.skip_res4 = Res3DBlock(128, 128)
        self.skip_res5 = Res3DBlock(128, 128)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        skip_x4 = self.skip_res4(x)
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        skip_x5 = self.skip_res5(x)
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)
        x = self.mid_res(x)
        x = self.decoder_res5(x)
        x = self.decoder_upsample5(x)
        x = x + skip_x5
        x = self.decoder_res4(x)
        x = self.decoder_upsample4(x)
        x = x + skip_x4
        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        return x


class V2VModel(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.front_layers = nn.Sequential(Basic3DBlock(input_channels, 16, 7), Res3DBlock(16, 32), Res3DBlock(32, 32), Res3DBlock(32, 32))
        self.encoder_decoder = EncoderDecorder()
        self.back_layers = nn.Sequential(Res3DBlock(32, 32), Basic3DBlock(32, 32, 1), Basic3DBlock(32, 32, 1))
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class VolumetricTriangulationNet(nn.Module):

    def __init__(self, config, device='cuda:0'):
        super().__init__()
        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size
        self.cuboid_side = config.model.cuboid_side
        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, 'transfer_cmu_to_human36m') else False
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False
        self.process_features = nn.Sequential(nn.Conv2d(256, 32, 1))
        self.volume_net = V2VModel(32, self.num_joints)

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]
        images = images.view(-1, *images.shape[2:])
        heatmaps, features, _, vol_confidences = self.backbone(images)
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])
        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)
        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)
        proj_matricies = proj_matricies.float()
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]
            if self.kind == 'coco':
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == 'mpii':
                base_point = keypoints_3d[6, :3]
            base_points[batch_i] = torch.from_numpy(base_point)
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)
            cuboids.append(cuboid)
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))
            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + sides[0] / (self.volume_size - 1) * grid[:, 0]
            grid_coord[:, 1] = position[1] + sides[1] / (self.volume_size - 1) * grid[:, 1]
            grid_coord[:, 2] = position[2] + sides[2] / (self.volume_size - 1) * grid[:, 2]
            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0
            if self.kind == 'coco':
                axis = [0, 1, 0]
            elif self.kind == 'mpii':
                axis = [0, 0, 1]
            center = torch.from_numpy(base_point).type(torch.float)
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center
            if self.transfer_cmu_to_human36m:
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long()
                coord_volume = coord_volume.index_select(1, inv_idx)
            coord_volumes[batch_i] = coord_volume
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)
        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Basic3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAveragePoolingHead,
     lambda: ([], {'in_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeypointsL2Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeypointsMAELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeypointsMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeypointsMSESmoothLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pool3DBlock,
     lambda: ([], {'pool_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Res3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Upsample3DBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 2, 'stride': 2}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_karfly_learnable_triangulation_pytorch(_paritybench_base):
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

