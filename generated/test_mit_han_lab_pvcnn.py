import sys
_module = sys.modules[__name__]
del sys
configs = _module
kitti = _module
frustum = _module
pointnet = _module
pointnet2 = _module
pvcnne = _module
s3dis = _module
pointnet = _module
area5 = _module
pvcnn = _module
c0p125 = _module
c0p25 = _module
c1 = _module
pvcnn2 = _module
c0p5 = _module
shapenet = _module
pointnet = _module
pointnet2msg = _module
pointnet2ssg = _module
pvcnn = _module
prepare_data = _module
datasets = _module
attributes = _module
frustum = _module
s3dis = _module
shapenet = _module
evaluate = _module
eval = _module
utils = _module
common = _module
iou = _module
eval = _module
eval = _module
meters = _module
frustum = _module
s3dis = _module
shapenet = _module
models = _module
box_estimation = _module
pointnet = _module
pointnetpp = _module
center_regression_net = _module
frustum_net = _module
segmentation = _module
pointnet = _module
pointnetpp = _module
pointnet = _module
pvcnn = _module
pvcnnpp = _module
pointnet = _module
pointnetpp = _module
pvcnn = _module
utils = _module
modules = _module
ball_query = _module
frustum = _module
functional = _module
backend = _module
ball_query = _module
devoxelization = _module
grouping = _module
interpolatation = _module
loss = _module
sampling = _module
voxelization = _module
loss = _module
pointnet = _module
pvconv = _module
se = _module
shared_mlp = _module
voxelization = _module
train = _module
train_dml = _module
config = _module
container = _module
device = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.optim as optim


import torch.nn as nn


from torch.utils.data import Dataset


import random


import functools


import torch.nn.functional as F


from torch.utils.cpp_extension import load


from torch.autograd import Function


class SharedMLP(nn.Module):

    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([conv(in_channels, oc, 1), bn(oc), nn.ReLU(True)])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return self.layers(inputs[0]), *inputs[1:]
        else:
            return self.layers(inputs)


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier
    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or len(out_channels) == 1 and out_channels[0] is None:
        return nn.Sequential(), in_channels, in_channels
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    elif classifier:
        layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
    else:
        layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


class SE3d(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)


class Voxelization(nn.Module):

    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class PVConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm3d(out_channels, eps=0.0001), nn.LeakyReLU(0.1, True), nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2), nn.BatchNorm3d(out_channels, eps=0.0001), nn.LeakyReLU(0.1, True)]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords


def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=0, width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), with_se=with_se, normalize=normalize, eps=eps)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels


class BoxEstimationNet(nn.Module):

    def __init__(self, num_classes, blocks, num_heading_angle_bins, num_size_templates, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes
        layers, channels_point, _ = create_pointnet_components(blocks=blocks, in_channels=self.in_channels, with_se=False, normalize=True, eps=1e-15, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.features = nn.Sequential(*layers)
        layers, _ = create_mlp_components(in_channels=channels_point + num_classes, out_channels=[512, 256, 3 + num_heading_angle_bins * 2 + num_size_templates * 4], classifier=True, dim=1, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2
        features, _ = self.features((coords, coords))
        features = features.max(dim=-1, keepdim=False).values
        return self.classifier(torch.cat([features, one_hot_vectors], dim=1))


class BoxEstimationPointNet(BoxEstimationNet):
    blocks = (128, 2, None), (256, 1, None), (512, 1, None)

    def __init__(self, num_classes=3, num_heading_angle_bins=12, num_size_templates=8, width_multiplier=1):
        super().__init__(num_classes=num_classes, blocks=self.blocks, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, width_multiplier=width_multiplier)


class PointNetAModule(nn.Module):

    def __init__(self, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]
        mlps = []
        total_out_channels = 0
        for _out_channels in out_channels:
            mlps.append(SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0), out_channels=_out_channels, dim=1))
            total_out_channels += _out_channels[-1]
        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords = inputs
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(mlp(features).max(dim=-1, keepdim=True).values)
            return torch.cat(features_list, dim=1), coords
        else:
            return self.mlps[0](features).max(dim=-1, keepdim=True).values, coords

    def extra_repr(self):
        return f'out_channels={self.out_channels}, include_coordinates={self.include_coordinates}'


class BallQuery(nn.Module):

    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, points_features=None):
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)
        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')


class PointNetSAModule(nn.Module):

    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)
        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(BallQuery(radius=_radius, num_neighbors=_num_neighbors, include_coordinates=include_coordinates))
            mlps.append(SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0), out_channels=_out_channels, dim=2))
            total_out_channels += _out_channels[-1]
        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords = inputs
        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        features_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            features_list.append(mlp(grouper(coords, centers_coords, features)).max(dim=-1).values)
        if len(features_list) > 1:
            return torch.cat(features_list, dim=1), centers_coords
        else:
            return features_list[0], centers_coords

    def extra_repr(self):
        return f'num_centers={self.num_centers}, out_channels={self.out_channels}'


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, with_se=False, normalize=True, eps=0, width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3
    sa_layers, sa_in_channels = [], []
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius, num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels, out_channels=out_channels, include_coordinates=True))
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


class BoxEstimationNet2(nn.Module):

    def __init__(self, num_classes, sa_blocks, num_heading_angle_bins, num_size_templates, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes
        sa_layers, _, channels_sa_features, num_centers = create_pointnet2_sa_components(sa_blocks=sa_blocks, extra_feature_channels=0, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.features = nn.Sequential(*sa_layers)
        layers, _ = create_mlp_components(in_channels=channels_sa_features * num_centers + num_classes, out_channels=[512, 256, 3 + num_heading_angle_bins * 2 + num_size_templates * 4], classifier=True, dim=1, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2
        features, _ = self.features((None, coords))
        features = features.view(features.size(0), -1)
        return self.classifier(torch.cat([features, one_hot_vectors], dim=1))


class BoxEstimationPointNet2(BoxEstimationNet2):
    sa_blocks = [(None, (128, 0.2, 64, (64, 64, 128))), (None, (32, 0.4, 64, (128, 128, 256))), (None, (None, None, None, (256, 256, 512)))]

    def __init__(self, num_classes=3, num_heading_angle_bins=12, num_size_templates=8, width_multiplier=1):
        super().__init__(num_classes=num_classes, sa_blocks=self.sa_blocks, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, width_multiplier=width_multiplier)


class CenterRegressionNet(nn.Module):
    blocks = 128, 128, 256

    def __init__(self, num_classes=3, width_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes
        layers, channels = create_mlp_components(in_channels=self.in_channels, out_channels=self.blocks, classifier=False, dim=2, width_multiplier=width_multiplier)
        self.features = nn.Sequential(*layers)
        layers, _ = create_mlp_components(in_channels=channels + num_classes, out_channels=[256, 128, 3], classifier=True, dim=1, width_multiplier=width_multiplier)
        self.regression = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2
        features = self.features(coords)
        features = features.max(dim=-1, keepdim=False).values
        return self.regression(torch.cat([features, one_hot_vectors], dim=1))


class FrustumNet(nn.Module):

    def __init__(self, num_classes, instance_segmentation_net, box_estimation_net, num_heading_angle_bins, num_size_templates, num_points_per_object, size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__()
        if not isinstance(width_multiplier, (list, tuple)):
            width_multiplier = [width_multiplier] * 3
        self.in_channels = 3 + extra_feature_channels
        self.num_classes = num_classes
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.num_points_per_object = num_points_per_object
        self.inst_seg_net = instance_segmentation_net(num_classes=num_classes, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier[0])
        self.center_reg_net = CenterRegressionNet(num_classes=num_classes, width_multiplier=width_multiplier[1])
        self.box_est_net = box_estimation_net(num_classes=num_classes, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, width_multiplier=width_multiplier[2])
        self.register_buffer('size_templates', size_templates.view(1, self.num_size_templates, 3))

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2
        mask_logits = self.inst_seg_net({'features': features, 'one_hot_vectors': one_hot_vectors})
        foreground_coords, foreground_coords_mean, _ = F.logits_mask(coords=features[:, :3, :], logits=mask_logits, num_points_per_object=self.num_points_per_object)
        delta_coords = self.center_reg_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
        estimation = self.box_est_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        estimations = estimation.split([3, self.num_heading_angle_bins, self.num_heading_angle_bins, self.num_size_templates, self.num_size_templates * 3], dim=-1)
        outputs = dict()
        outputs['mask_logits'] = mask_logits
        outputs['center_reg'] = foreground_coords_mean + delta_coords
        outputs['center'] = estimations[0] + outputs['center_reg']
        outputs['heading_scores'] = estimations[1]
        outputs['heading_residuals_normalized'] = estimations[2]
        outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
        outputs['size_scores'] = estimations[3]
        size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
        outputs['size_residuals_normalized'] = size_residuals_normalized
        outputs['size_residuals'] = size_residuals_normalized * self.size_templates
        return outputs


class InstanceSegmentationNet(nn.Module):

    def __init__(self, num_classes, point_blocks, cloud_blocks, extra_feature_channels, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_classes = num_classes
        layers, channels_point, _ = create_pointnet_components(blocks=point_blocks, in_channels=self.in_channels, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.point_features = nn.Sequential(*layers)
        layers, channels_cloud, _ = create_pointnet_components(blocks=cloud_blocks, in_channels=channels_point, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.cloud_features = nn.Sequential(*layers)
        layers, _ = create_mlp_components(in_channels=channels_point + channels_cloud + num_classes, out_channels=[512, 256, 128, 128, 0.5, 2], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs['features']
        num_points = features.size(-1)
        one_hot_vectors = inputs['one_hot_vectors'].unsqueeze(-1).repeat([1, 1, num_points])
        assert one_hot_vectors.dim() == 3
        point_features, point_coords = self.point_features((features, features[:, :3, :]))
        cloud_features, _ = self.cloud_features((point_features, point_coords))
        cloud_features = cloud_features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points])
        return self.classifier(torch.cat([one_hot_vectors, point_features, cloud_features], dim=1))


class InstanceSegmentationPointNet(InstanceSegmentationNet):
    point_blocks = (64, 3, None),
    cloud_blocks = (128, 1, None), (1024, 1, None)

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, point_blocks=self.point_blocks, cloud_blocks=self.cloud_blocks, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)


class FrustumPointNet(FrustumNet):

    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object, size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet, box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, num_points_per_object=num_points_per_object, size_templates=size_templates, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)


class PointNetFPModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

    def forward(self, inputs):
        if len(inputs) == 3:
            points_coords, centers_coords, centers_features = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features = inputs
        interpolated_features = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        if points_features is not None:
            interpolated_features = torch.cat([interpolated_features, points_features], dim=1)
        return self.mlp(interpolated_features), points_coords


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, with_se=False, normalize=True, eps=0, width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    fp_layers = []
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx], out_channels=out_channels))
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))
    return fp_layers, in_channels


class InstanceSegmentationNet2(nn.Module):

    def __init__(self, num_classes, sa_blocks, fp_blocks, extra_feature_channels, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        self.num_classes = num_classes
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(sa_blocks=sa_blocks, extra_feature_channels=extra_feature_channels, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.sa_layers = nn.ModuleList(sa_layers)
        sa_in_channels[-1] += num_classes
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.fp_layers = nn.ModuleList(fp_layers)
        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.3, 2], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2
        coords, extra_features = features[:, :3, :].contiguous(), features[:, 3:, :].contiguous()
        coords_list, in_features_list = [], []
        for sa_module in self.sa_layers:
            in_features_list.append(extra_features)
            coords_list.append(coords)
            extra_features, coords = sa_module((extra_features, coords))
        in_features_list[0] = features.contiguous()
        features = torch.cat([extra_features, one_hot_vectors.unsqueeze(-1).repeat([1, 1, extra_features.size(-1)])], dim=1)
        for fp_idx, fp_module in enumerate(self.fp_layers):
            features, coords = fp_module((coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx]))
        return self.classifier(features)


class InstanceSegmentationPointNet2(InstanceSegmentationNet2):
    sa_blocks = [(None, (128, [0.2, 0.4, 0.8], [32, 64, 128], [(32, 32, 64), (64, 64, 128), (64, 96, 128)])), (None, (32, [0.4, 0.8, 1.6], [64, 64, 128], [(64, 64, 128), (128, 128, 256), (128, 128, 256)])), (None, (None, None, None, (128, 256, 1024)))]
    fp_blocks = [((128, 128), None), ((128, 128), None), ((128, 128), None)]

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)


class FrustumPointNet2(FrustumNet):

    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object, size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet2, box_estimation_net=BoxEstimationPointNet2, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, num_points_per_object=num_points_per_object, size_templates=size_templates, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)


class InstanceSegmentationPVCNN(InstanceSegmentationNet):
    point_blocks = (64, 2, 16), (64, 1, 12), (128, 1, 12), (1024, 1, None)
    cloud_blocks = ()

    def __init__(self, num_classes=3, extra_feature_channels=1, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, point_blocks=self.point_blocks, cloud_blocks=self.cloud_blocks, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class FrustumPVCNNE(FrustumNet):

    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object, size_templates, extra_feature_channels=1, width_multiplier=1, voxel_resolution_multiplier=1):
        instance_segmentation_net = functools.partial(InstanceSegmentationPVCNN, voxel_resolution_multiplier=voxel_resolution_multiplier)
        super().__init__(num_classes=num_classes, instance_segmentation_net=instance_segmentation_net, box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins, num_size_templates=num_size_templates, num_points_per_object=num_points_per_object, size_templates=size_templates, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)


class Transformer(nn.Module):

    def __init__(self, channels):
        super(Transformer, self).__init__()
        self.channels = channels
        self.features = nn.Sequential(SharedMLP(self.channels, 64), SharedMLP(64, 128), SharedMLP(128, 1024))
        self.tranformer = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Linear(256, self.channels * self.channels))

    def forward(self, inputs):
        transform_weight = self.tranformer(torch.max(self.features(inputs), dim=-1, keepdim=False).values)
        transform_weight = transform_weight.view(-1, self.channels, self.channels)
        transform_weight = transform_weight + torch.eye(self.channels, device=transform_weight.device)
        outputs = torch.bmm(transform_weight, inputs)
        return outputs


class PointNet(nn.Module):
    blocks = (True, 64, 1), (False, 128, 2), (True, 512, 1), (False, 2048, 1)

    def __init__(self, num_classes, num_shapes, with_transformer=False, extra_feature_channels=0, width_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        r = width_multiplier
        self.in_channels = in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.with_transformer = with_transformer
        layers, concat_channels = [], 0
        for with_transformer_before, out_channels, num_blocks in self.blocks:
            with_transformer_before = with_transformer_before and with_transformer
            out_channels = int(r * out_channels)
            for block_index in range(num_blocks):
                if with_transformer_before and block_index == 0:
                    layers.append(nn.Sequential(Transformer(in_channels), SharedMLP(in_channels, out_channels)))
                else:
                    layers.append(SharedMLP(in_channels, out_channels))
                in_channels = out_channels
                concat_channels += out_channels
        self.point_features = nn.ModuleList(layers)
        self.classifier = nn.Sequential(SharedMLP(in_channels=in_channels + concat_channels + num_shapes, out_channels=int(r * 256)), nn.Dropout(0.2), SharedMLP(in_channels=int(r * 256), out_channels=int(r * 256)), nn.Dropout(0.2), SharedMLP(in_channels=int(r * 256), out_channels=int(r * 128)), nn.Conv1d(int(r * 128), num_classes, 1))

    def forward(self, inputs):
        assert inputs.size(1) == self.in_channels + self.num_shapes
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features = self.point_features[i](features)
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return self.classifier(torch.cat(out_features_list, dim=1))


class PVCNN(nn.Module):
    blocks = (64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None)

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        layers, channels_point, concat_channels_point = create_pointnet_components(blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.point_features = nn.ModuleList(layers)
        layers, _ = create_mlp_components(in_channels=num_shapes + channels_point + concat_channels_point, out_channels=[256, 0.2, 256, 0.2, 128, num_classes], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs[:, :self.in_channels, :]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        num_points = features.size(-1)
        coords = features[:, :3, :]
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        return self.classifier(torch.cat(out_features_list, dim=1))


class PVCNN2(nn.Module):
    sa_blocks = [((32, 2, 32), (1024, 0.1, 32, (32, 64))), ((64, 3, 16), (256, 0.2, 32, (64, 128))), ((128, 3, 8), (64, 0.4, 32, (128, 256))), (None, (16, 0.8, 32, (256, 256, 512)))]
    fp_blocks = [((256, 256), (256, 1, 8)), ((256, 256), (256, 1, 8)), ((256, 128), (128, 2, 16)), ((128, 128, 64), (64, 1, 32))]

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = extra_feature_channels + 3
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.sa_layers = nn.ModuleList(sa_layers)
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.fp_layers = nn.ModuleList(fp_layers)
        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['features']
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks((coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx]))
        return self.classifier(features)


class PointNet2(nn.Module):

    def __init__(self, num_classes, num_shapes, sa_blocks, fp_blocks, with_one_hot_shape_id=True, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.with_one_hot_shape_id = with_one_hot_shape_id
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(sa_blocks=sa_blocks, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier)
        self.sa_layers = nn.ModuleList(sa_layers)
        sa_in_channels[0] += num_shapes if with_one_hot_shape_id else 0
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.fp_layers = nn.ModuleList(fp_layers)
        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes], classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        features = inputs[:, :self.in_channels, :]
        if self.with_one_hot_shape_id:
            assert inputs.size(1) == self.in_channels + self.num_shapes
            features_with_one_hot_vectors = inputs
        else:
            features_with_one_hot_vectors = features
        coords, features = features[:, :3, :].contiguous(), features[:, 3:, :].contiguous()
        coords_list, in_features_list = [], []
        for sa_module in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_module((features, coords))
        in_features_list[0] = features_with_one_hot_vectors.contiguous()
        for fp_idx, fp_module in enumerate(self.fp_layers):
            features, coords = fp_module((coords_list[-1 - fp_idx], coords, features, in_features_list[-1 - fp_idx]))
        return self.classifier(features)


class PointNet2SSG(PointNet2):
    sa_blocks = [(None, (512, 0.2, 64, (64, 64, 128))), (None, (128, 0.4, 64, (128, 128, 256))), (None, (None, None, None, (256, 512, 1024)))]
    fp_blocks = [((256, 256), None), ((256, 128), None), ((128, 128, 128), None)]

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, num_shapes=num_shapes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks, with_one_hot_shape_id=False, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class PointNet2MSG(PointNet2):
    sa_blocks = [(None, (512, [0.1, 0.2, 0.4], [32, 64, 128], [(32, 32, 64), (64, 64, 128), (64, 96, 128)])), (None, (128, [0.4, 0.8], [64, 128], [(128, 128, 256), (128, 196, 256)])), (None, (None, None, None, (256, 512, 1024)))]
    fp_blocks = [((256, 256), None), ((256, 128), None), ((128, 128, 128), None)]

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(num_classes=num_classes, num_shapes=num_shapes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks, with_one_hot_shape_id=True, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


def get_box_corners_3d(centers, headings, sizes, with_flip=False):
    """
    :param centers: coords of box centers, FloatTensor[N, 3]
    :param headings: heading angles, FloatTensor[N, ]
    :param sizes: box sizes, FloatTensor[N, 3]
    :param with_flip: bool, whether to return flipped box (headings + np.pi)
    :return:
        coords of box corners, FloatTensor[N, 3, 8]
        NOTE: corner points are in counter clockwise order, e.g.,
          2--1
        3--0 5
        7--4
    """
    l = sizes[:, (0)]
    w = sizes[:, (1)]
    h = sizes[:, (2)]
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)
    c = torch.cos(headings)
    s = torch.sin(headings)
    o = torch.ones_like(headings)
    z = torch.zeros_like(headings)
    centers = centers.unsqueeze(-1)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)
    R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)
    if with_flip:
        R_flip = torch.stack([-c, z, -s, z, o, z, s, z, -c], dim=1).view(-1, 3, 3)
        return torch.matmul(R, corners) + centers, torch.matmul(R_flip, corners) + centers
    else:
        return torch.matmul(R, corners) + centers


class FrustumPointNetLoss(nn.Module):

    def __init__(self, num_heading_angle_bins, num_size_templates, size_templates, box_loss_weight=1.0, corners_loss_weight=10.0, heading_residual_loss_weight=20.0, size_residual_loss_weight=20.0):
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.corners_loss_weight = corners_loss_weight
        self.heading_residual_loss_weight = heading_residual_loss_weight
        self.size_residual_loss_weight = size_residual_loss_weight
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.register_buffer('size_templates', size_templates.view(self.num_size_templates, 3))
        self.register_buffer('heading_angle_bin_centers', torch.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_angle_bins))

    def forward(self, inputs, targets):
        mask_logits = inputs['mask_logits']
        center_reg = inputs['center_reg']
        center = inputs['center']
        heading_scores = inputs['heading_scores']
        heading_residuals_normalized = inputs['heading_residuals_normalized']
        heading_residuals = inputs['heading_residuals']
        size_scores = inputs['size_scores']
        size_residuals_normalized = inputs['size_residuals_normalized']
        size_residuals = inputs['size_residuals']
        mask_logits_target = targets['mask_logits']
        center_target = targets['center']
        heading_bin_id_target = targets['heading_bin_id']
        heading_residual_target = targets['heading_residual']
        size_template_id_target = targets['size_template_id']
        size_residual_target = targets['size_residual']
        batch_size = center.size(0)
        batch_id = torch.arange(batch_size, device=center.device)
        mask_loss = F.cross_entropy(mask_logits, mask_logits_target)
        heading_loss = F.cross_entropy(heading_scores, heading_bin_id_target)
        size_loss = F.cross_entropy(size_scores, size_template_id_target)
        center_loss = PF.huber_loss(torch.norm(center_target - center, dim=-1), delta=2.0)
        center_reg_loss = PF.huber_loss(torch.norm(center_target - center_reg, dim=-1), delta=1.0)
        heading_residuals_normalized = heading_residuals_normalized[batch_id, heading_bin_id_target]
        heading_residual_normalized_target = heading_residual_target / (np.pi / self.num_heading_angle_bins)
        heading_residual_normalized_loss = PF.huber_loss(heading_residuals_normalized - heading_residual_normalized_target, delta=1.0)
        size_residuals_normalized = size_residuals_normalized[batch_id, size_template_id_target]
        size_residual_normalized_target = size_residual_target / self.size_templates[size_template_id_target]
        size_residual_normalized_loss = PF.huber_loss(torch.norm(size_residual_normalized_target - size_residuals_normalized, dim=-1), delta=1.0)
        heading = heading_residuals[batch_id, heading_bin_id_target] + self.heading_angle_bin_centers[heading_bin_id_target]
        size = size_residuals[batch_id, size_template_id_target] + self.size_templates[size_template_id_target]
        corners = get_box_corners_3d(centers=center, headings=heading, sizes=size, with_flip=False)
        heading_target = self.heading_angle_bin_centers[heading_bin_id_target] + heading_residual_target
        size_target = self.size_templates[size_template_id_target] + size_residual_target
        corners_target, corners_target_flip = get_box_corners_3d(centers=center_target, headings=heading_target, sizes=size_target, with_flip=True)
        corners_loss = PF.huber_loss(torch.min(torch.norm(corners - corners_target, dim=1), torch.norm(corners - corners_target_flip, dim=1)), delta=1.0)
        loss = mask_loss + self.box_loss_weight * (center_loss + center_reg_loss + heading_loss + size_loss + self.heading_residual_loss_weight * heading_residual_normalized_loss + self.size_residual_loss_weight * size_residual_normalized_loss + self.corners_loss_weight * corners_loss)
        return loss


class KLLoss(nn.Module):

    def forward(self, x, y):
        return F.kl_loss(x, y)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SharedMLP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (Transformer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
]

class Test_mit_han_lab_pvcnn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

