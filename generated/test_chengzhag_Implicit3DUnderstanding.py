import sys
_module = sys.modules[__name__]
del sys
config_utils = _module
data_config = _module
pix3d_config = _module
demo = _module
mesh_util = _module
sample_util = _module
sdf = _module
external = _module
extract_mesh = _module
metrics = _module
get_cuda_version = _module
quadrics = _module
structured_implicit_function = _module
util = _module
base_util = _module
camera_util = _module
file_util = _module
gaps_util = _module
np_util = _module
common = _module
fusion = _module
libfusioncpu = _module
setup = _module
libfusiongpu = _module
libmcubes = _module
exporter = _module
librender = _module
test = _module
scale = _module
simplify = _module
pyTorchChamferDistance = _module
chamfer_distance = _module
chamfer_distance = _module
libs = _module
tools = _module
main = _module
models = _module
datasets = _module
eval_metrics = _module
ldif = _module
config = _module
dataloader = _module
modules = _module
network = _module
testing = _module
training = _module
loss = _module
mgnet = _module
dataloader = _module
network = _module
testing = _module
resnet = _module
network = _module
optimizers = _module
registers = _module
total3d = _module
dataloader = _module
gcnn = _module
layout_estimation = _module
mesh_reconstruction = _module
network = _module
object_detection = _module
relation_net = _module
testing = _module
training = _module
net_utils = _module
libs = _module
misc = _module
registry = _module
utils = _module
project = _module
test_epoch = _module
train = _module
train_epoch = _module
class_mapping_for_Meta = _module
generate_data = _module
generate_demo = _module
preprocess_pix3d = _module
preprocess_pix3d4ldif = _module
sunrgbd_config = _module
sunrgbd_utils = _module
vis_tools = _module
vis_tools_sunrgbd = _module
visualize = _module

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


from torchvision import transforms


from time import time


import math


import torch.nn.functional as F


from torch.utils.cpp_extension import load


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.utils.data


import collections


from random import random


from torch import nn


from collections import defaultdict


import torch.nn as nn


from scipy.spatial import cKDTree


import torch.utils.model_zoo as model_zoo


from scipy.io import savemat


import copy


from torch.nn import functional as F


from copy import deepcopy


import inspect


import random


from torch.utils import model_zoo


from torch.optim import lr_scheduler


class ChamferDistanceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)
        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1
            dist2 = dist2
            idx1 = idx1
            idx2 = idx2
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, *args):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1
            gradxyz2 = gradxyz2
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):

    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


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

    def __init__(self, block, layers, num_classes=1000, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNet_Full(ResNet):

    def __init__(self, block, layers, num_classes=1000, input_channels=3):
        super(ResNet_Full, self).__init__(block, layers, input_channels=input_channels)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

    def get(self, key, alter_key=None):
        if key in self._module_dict:
            return self._module_dict.get(key)
        else:
            return self._module_dict.get(alter_key, None)

    def _register_module(self, module_class):
        """
        register a module.
        :param module_class (`nn.Module`): Module to be registered.
        :return:
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


LOSSES = Registry('loss')


MODULES = Registry('module')


class BaseNetwork(nn.Module):
    """
    Base Network Module for other networks
    """

    def __init__(self, cfg):
        """
        load submodules for the network.
        :param config: customized configurations.
        """
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        """load network blocks"""
        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg.config, optim_spec)
            self.add_module(phase_name, subnet)
            """load corresponding loss functions"""
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(self.cfg.config['model'][phase_name].get('weight', 1)))
        """freeze submodules or not"""
        self.freeze_modules(cfg)

    def freeze_modules(self, cfg):
        """
        Freeze modules in training
        """
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                cfg.log_string('The module: %s is fixed.' % layer)

    def set_mode(self):
        """
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        """
        freeze_layers = self.cfg.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)
        if self.cfg.config[self.cfg.config['mode']]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()

    def load_weight(self, pretrained_model):
        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in pretrained_model.items():
            if k not in model_dict:
                None
            elif model_dict[k].shape != v.shape:
                None
            else:
                pretrained_dict[k] = v
        self.cfg.log_string(str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self, config, net_spec):
        if config['mode'] == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = net_spec['optimizer']
            else:
                optim_spec = config['optimizer']
        else:
            optim_spec = None
        return optim_spec

    def forward(self, *args, **kwargs):
        """ Performs a forward step.
        """
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """ calculate losses.
        """
        raise NotImplementedError


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-07)
        return collect_avg


class _Update_Unit(nn.Module):

    def __init__(self, dim):
        super(_Update_Unit, self).__init__()

    def forward(self, target, source):
        assert target.size() == source.size(), 'source dimension must be equal to target dimension'
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """

    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj))
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel))
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj))

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj))
        self.update_units.append(_Update_Unit(dim_rel))

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


NYU40CLASSES = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


def basis_from_ori(ori):
    """
    :param ori: torch tensor
            the orientation angle
    :return: basis: 3x3 tensor
            the basis in 3D coordinates
    """
    n = ori.size(0)
    basis = torch.zeros((n, 3, 3))
    basis[:, 0, 0] = torch.cos(ori)
    basis[:, 0, 2] = -torch.sin(ori)
    basis[:, 1, 1] = 1
    basis[:, 2, 0] = torch.sin(ori)
    basis[:, 2, 2] = torch.cos(ori)
    return basis


def get_corners_of_bb3d(basis, coeffs, centroid):
    """
    :param basis: n x 3 x 3 tensor
    :param coeffs: n x 3 tensor
    :param centroid:  n x 3 tensor
    :return: corners n x 8 x 3 tensor
    """
    n = basis.size(0)
    corners = torch.zeros((n, 8, 3))
    coeffs = coeffs.view(n, 3, 1).expand(-1, -1, 3)
    centroid = centroid.view(n, 1, 3).expand(-1, 8, -1)
    corners[:, 0, :] = -basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 1, :] = -basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 2, :] = basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 3, :] = basis[:, 0, :] * coeffs[:, 0, :] + basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 4, :] = -basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 5, :] = -basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 6, :] = basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] + basis[:, 2, :] * coeffs[:, 2, :]
    corners[:, 7, :] = basis[:, 0, :] * coeffs[:, 0, :] - basis[:, 1, :] * coeffs[:, 1, :] - basis[:, 2, :] * coeffs[:, 2, :]
    corners = corners + centroid
    return corners


def num_from_bins(bins, cls, reg):
    """
    :param bins: b x 2 tensors
    :param cls: b long tensors
    :param reg: b tensors
    :return: bin_center: b tensors
    """
    bin_width = bins[0][1] - bins[0][0]
    bin_center = (bins[cls, 0] + bins[cls, 1]) / 2
    return bin_center + reg * bin_width


def rgb_to_world(p, depth, K, cam_R, split):
    """
    Given pixel location and depth, camera parameters, to recover world coordinates.
    :param p: n x 2 tensor
    :param depth: b tensor
    :param k: b x 3 x 3 tensor
    :param cam_R: b x 3 x 3 tensor
    :param split: b x 2 split tensor.
    :return: p_world_right: n x 3 tensor in right hand coordinate
    """
    n = p.size(0)
    K_ex = torch.cat([K[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    cam_R_ex = torch.cat([cam_R[index].expand(interval[1] - interval[0], -1, -1) for index, interval in enumerate(split)], 0)
    x_temp = (p[:, 0] - K_ex[:, 0, 2]) / K_ex[:, 0, 0]
    y_temp = (p[:, 1] - K_ex[:, 1, 2]) / K_ex[:, 1, 1]
    z_temp = 1
    ratio = depth / torch.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    x3 = z_cam
    y3 = -y_cam
    z3 = x_cam
    p_cam = torch.stack((x3, y3, z3), 1).view(n, 3, 1)
    p_world = torch.bmm(cam_R_ex, p_cam)
    return p_world


def get_bdb_3d_result(bins_tensor, ori_cls_gt, ori_reg_result, centroid_cls_gt, centroid_reg_result, size_cls_gt, size_reg_result, P, K, cam_R, split):
    size_cls_gt = torch.argmax(size_cls_gt, 1)
    coeffs = (size_reg_result + 1) * bins_tensor['avg_size'][size_cls_gt, :]
    centroid_reg = torch.gather(centroid_reg_result, 1, centroid_cls_gt.view(centroid_cls_gt.size(0), 1).expand(centroid_cls_gt.size(0), 1)).squeeze(1)
    centroid_depth = num_from_bins(bins_tensor['centroid_bin'], centroid_cls_gt, centroid_reg)
    centroid = rgb_to_world(P, centroid_depth, K, cam_R, split)
    ori_reg = torch.gather(ori_reg_result, 1, ori_cls_gt.view(ori_cls_gt.size(0), 1).expand(ori_cls_gt.size(0), 1)).squeeze(1)
    ori = num_from_bins(bins_tensor['ori_bin'], ori_cls_gt, ori_reg)
    basis = basis_from_ori(ori)
    bdb = get_corners_of_bb3d(basis, coeffs, centroid)
    bdb_form = {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}
    return bdb, bdb_form


def get_bdb_form_from_corners(corners, mask_status=None):
    if mask_status is not None:
        corners = corners[mask_status.nonzero()]
    vec_0 = (corners[:, 2, :] - corners[:, 1, :]) / 2.0
    vec_1 = (corners[:, 0, :] - corners[:, 4, :]) / 2.0
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.0
    coeffs_0 = torch.norm(vec_0, dim=1)
    coeffs_1 = torch.norm(vec_1, dim=1)
    coeffs_2 = torch.norm(vec_2, dim=1)
    coeffs = torch.cat([coeffs_0.unsqueeze(-1), coeffs_1.unsqueeze(-1), coeffs_2.unsqueeze(-1)], -1)
    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.0
    basis_0 = torch.mm(torch.diag(1 / coeffs_0), vec_0)
    basis_1 = torch.mm(torch.diag(1 / coeffs_1), vec_1)
    basis_2 = torch.mm(torch.diag(1 / coeffs_2), vec_2)
    basis = torch.cat([basis_0.unsqueeze(1), basis_1.unsqueeze(1), basis_2.unsqueeze(1)], dim=1)
    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}


def R_from_yaw_pitch_roll(yaw, pitch, roll):
    """
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    """
    n = yaw.size(0)
    R = torch.zeros((n, 3, 3))
    R[:, 0, 0] = torch.cos(yaw) * torch.cos(pitch)
    R[:, 0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    R[:, 0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    R[:, 1, 0] = torch.sin(pitch)
    R[:, 1, 1] = torch.cos(pitch) * torch.cos(roll)
    R[:, 1, 2] = -torch.cos(pitch) * torch.sin(roll)
    R[:, 2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    R[:, 2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    R[:, 2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
    return R


def get_rotation_matix_result(bins_tensor, pitch_cls_gt, pitch_reg_result, roll_cls_gt, roll_reg_result, return_degrees=False):
    """
    get rotation matrix from predicted camera pitch, roll angles.
    """
    pitch_result = torch.gather(pitch_reg_result, 1, pitch_cls_gt.view(pitch_cls_gt.size(0), 1).expand(pitch_cls_gt.size(0), 1)).squeeze(1)
    roll_result = torch.gather(roll_reg_result, 1, roll_cls_gt.view(roll_cls_gt.size(0), 1).expand(roll_cls_gt.size(0), 1)).squeeze(1)
    pitch = num_from_bins(bins_tensor['pitch_bin'], pitch_cls_gt, pitch_result)
    roll = num_from_bins(bins_tensor['roll_bin'], roll_cls_gt, roll_result)
    cam_R = R_from_yaw_pitch_roll(torch.zeros_like(pitch), pitch, roll)
    if return_degrees:
        return cam_R, pitch, roll
    else:
        return cam_R


pix3d_n_classes = 9


def recover_points_to_obj_sys(bdb3D, obj_sample, ldif_center, ldif_coef):
    """
    Get 3D point cloud from mesh with estimated position and orientation.
    :param bdb3D: 3D object bounding boxes with keys ['coeffs', 'basis', 'centroid'].
    :param obj_sample: Number_of_objects x Number_of_points x 3.
    :return: points on world system
    """
    obj_sample_in_obj_sys = obj_sample - bdb3D['centroid'].unsqueeze(1)
    obj_sample_in_obj_sys = torch.matmul(obj_sample_in_obj_sys, torch.inverse(bdb3D['basis']))
    obj_sample_in_obj_sys = torch.matmul(obj_sample_in_obj_sys, torch.diag_embed(1.0 / bdb3D['coeffs']))
    sample = torch.matmul(obj_sample_in_obj_sys, torch.diag_embed(ldif_coef))
    sample = sample + ldif_center.unsqueeze(1)
    return sample


def recover_points_to_world_sys(bdb3D, mesh_coordinates, ldif_center=None, ldif_coef=None):
    """
    Get 3D point cloud from mesh with estimated position and orientation.
    :param bdb3D: 3D object bounding boxes with keys ['coeffs', 'basis', 'centroid'].
    :param mesh_coordinates: Number_of_objects x Number_of_points x 3.
    :return: points on world system
    """
    if ldif_center is None or ldif_coef is None:
        mesh_coordinates_in_world_sys = []
        for obj_idx, mesh_coordinate in enumerate(mesh_coordinates):
            mesh_coordinate = mesh_coordinate.transpose(-1, -2)
            mesh_center = (mesh_coordinate.max(dim=0)[0] + mesh_coordinate.min(dim=0)[0]) / 2.0
            mesh_center = mesh_center.detach()
            mesh_coordinate = mesh_coordinate - mesh_center
            mesh_coef = (mesh_coordinate.max(dim=0)[0] - mesh_coordinate.min(dim=0)[0]) / 2.0
            mesh_coef = mesh_coef.detach()
            mesh_coordinate = torch.mm(torch.mm(mesh_coordinate, torch.diag(1.0 / mesh_coef)), torch.diag(bdb3D['coeffs'][obj_idx]))
            mesh_coordinate = torch.mm(mesh_coordinate, bdb3D['basis'][obj_idx])
            mesh_coordinate = mesh_coordinate + bdb3D['centroid'][obj_idx].view(1, 3)
            mesh_coordinates_in_world_sys.append(mesh_coordinate)
    else:
        mesh_coordinates = mesh_coordinates - ldif_center.unsqueeze(1)
        mesh_coordinates = torch.matmul(mesh_coordinates, torch.diag_embed(1.0 / ldif_coef))
        mesh_coordinates = torch.matmul(mesh_coordinates, torch.diag_embed(bdb3D['coeffs']))
        mesh_coordinates = torch.matmul(mesh_coordinates, bdb3D['basis'])
        mesh_coordinates_in_world_sys = mesh_coordinates + bdb3D['centroid'].unsqueeze(1)
    return mesh_coordinates_in_world_sys


class GCNN(nn.Module):

    def __init__(self, cfg, optim_spec=None):
        super(GCNN, self).__init__()
        """Optimizer parameters used in training"""
        self.optim_spec = optim_spec
        """configs and params"""
        self.cfg = cfg
        self.lo_features = cfg.config['model']['output_adjust']['lo_features']
        self.obj_features = cfg.config['model']['output_adjust']['obj_features']
        self.rel_features = cfg.config['model']['output_adjust']['rel_features']
        feature_dim = cfg.config['model']['output_adjust']['feature_dim']
        self.feat_update_step = cfg.config['model']['output_adjust']['feat_update_step']
        self.res_output = cfg.config['model']['output_adjust'].get('res_output', False)
        self.feat_update_group = cfg.config['model']['output_adjust'].get('feat_update_group', 1)
        self.res_group = cfg.config['model']['output_adjust'].get('res_group', False)
        self.feature_length = {'size_cls': len(NYU40CLASSES), 'cls_codes': pix3d_n_classes, 'bdb2D_pos': 4, 'g_features': 32, 'mgn_afeature': 1024, 'K': 3, 'pitch_reg_result': 2, 'roll_reg_result': 2, 'pitch_cls_result': 2, 'roll_cls_result': 2, 'lo_ori_reg_result': 2, 'lo_ori_cls_result': 2, 'lo_centroid_result': 3, 'lo_coeffs_result': 3, 'lo_afeatures': 2048, 'size_reg_result': 3, 'ori_reg_result': 6, 'ori_cls_result': 6, 'centroid_reg_result': 6, 'centroid_cls_result': 6, 'offset_2D_result': 2, 'odn_afeature': 2048, 'odn_rfeatures': 2048, 'odn_arfeatures': 2048, 'ldif_afeature': cfg.config['model']['mesh_reconstruction'].get('bottleneck_size', None), 'analytic_code': cfg.config['model']['mesh_reconstruction'].get('analytic_code_len', None), 'blob_center': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0) + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) * 3, 'ldif_phy': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0) + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) // 2, 'structured_implicit_vector': cfg.config['model']['mesh_reconstruction'].get('structured_implicit_vector_len', None)}
        obj_features_len = sum([self.feature_length[k] for k in self.obj_features])
        rel_features_len = sum([self.feature_length[k] for k in self.rel_features]) * 2
        lo_features_len = sum([self.feature_length[k] for k in self.lo_features])
        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])
        """modules"""
        self.obj_embedding = nn.Sequential(nn.Linear(obj_features_len, feature_dim), nn.ReLU(True), nn.Linear(feature_dim, feature_dim))
        self.rel_embedding = nn.Sequential(nn.Linear(rel_features_len, feature_dim), nn.ReLU(True), nn.Linear(feature_dim, feature_dim))
        self.lo_embedding = nn.Sequential(nn.Linear(lo_features_len, feature_dim), nn.ReLU(True), nn.Linear(feature_dim, feature_dim))
        if self.feat_update_step > 0:
            self.gcn_collect_feat = nn.ModuleList([_GraphConvolutionLayer_Collect(feature_dim, feature_dim) for i in range(self.feat_update_group)])
            self.gcn_update_feat = nn.ModuleList([_GraphConvolutionLayer_Update(feature_dim, feature_dim) for i in range(self.feat_update_group)])
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, 3)
        self.fc3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc4 = nn.Linear(feature_dim // 2, self.OBJ_ORI_BIN * 2)
        self.fc5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_centroid = nn.Linear(feature_dim // 2, self.OBJ_CENTER_BIN * 2)
        self.fc_off_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_off_2 = nn.Linear(feature_dim // 2, 2)
        self.fc_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_2 = nn.Linear(feature_dim // 2, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        self.fc_layout = nn.Linear(feature_dim, feature_dim)
        self.fc_3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_4 = nn.Linear(feature_dim // 2, self.LO_ORI_BIN * 2)
        self.fc_5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_6 = nn.Linear(feature_dim // 2, 6)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _K2feature(self, K):
        camKs = K.reshape(K.shape[0], -1)
        camKs = camKs.index_select(1, torch.tensor([0, 2, 4, 5], device=camKs.device))
        camKs = camKs[:, :3] / camKs[:, 3:]
        return camKs

    def _get_bdb3D_form(self, data):
        cam_R_out = get_rotation_matix_result(self.cfg.bins_tensor, torch.argmax(data['pitch_cls_result'], 1), data['pitch_reg_result'], torch.argmax(data['roll_cls_result'], 1), data['roll_reg_result'])
        P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 - (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * data['offset_2D_result'][:, 0], (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 - (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * data['offset_2D_result'][:, 1]), 1)
        bdb3D_result, _ = get_bdb_3d_result(self.cfg.bins_tensor, torch.argmax(data['ori_cls_result'], 1), data['ori_reg_result'], torch.argmax(data['centroid_cls_result'], 1), data['centroid_reg_result'], data['size_cls'], data['size_reg_result'], P_result, data['K'], cam_R_out, data['split'])
        bdb3D_form = get_bdb_form_from_corners(bdb3D_result)
        return bdb3D_form

    def _get_object_features(self, data, type):
        features = []
        keys = self.obj_features if type == 'obj' else self.rel_features
        for k in keys:
            if k in ['size_cls', 'cls_codes', 'size_reg_result', 'ori_reg_result', 'ori_cls_result', 'centroid_reg_result', 'centroid_cls_result', 'offset_2D_result', 'ldif_afeature', 'mgn_afeature', 'odn_afeature', 'odn_rfeatures', 'odn_arfeatures']:
                v = data[k]
            elif k == 'g_features':
                assert type == 'rel'
                v = data[k]
            elif k == 'bdb2D_pos':
                v = data[k].clone()
                center_inds = data['K'][:, :2, 2]
                for center_ind, (start, end) in zip(center_inds, data['split']):
                    for i in range(start.item(), end.item()):
                        v[i][0] = (v[i][0] - center_ind[0]) / center_ind[0]
                        v[i][2] = (v[i][2] - center_ind[0]) / center_ind[0]
                        v[i][1] = (v[i][1] - center_ind[1]) / center_ind[1]
                        v[i][3] = (v[i][3] - center_ind[1]) / center_ind[1]
            elif k == 'K':
                camKs = self._K2feature(data[k])
                v = []
                for i, (start, end) in enumerate(data['split']):
                    v.append(camKs[i:i + 1, :].expand(end - start, -1))
                v = torch.cat(v, 0)
            elif k in ['analytic_code', 'structured_implicit_vector', 'blob_center']:
                if k == 'analytic_code':
                    v = data['structured_implicit'].analytic_code
                elif k == 'structured_implicit_vector':
                    v = data['structured_implicit'].vector
                elif k == 'blob_center':
                    bdb3D_form = self._get_bdb3D_form(data)
                    centers = data['structured_implicit'].all_centers.clone()
                    centers[:, :, 2] *= -1
                    v = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                v = v.reshape([v.shape[0], -1])
            elif k == 'ldif_phy':
                assert type == 'rel'
                bdb3D_form = self._get_bdb3D_form(data)
                structured_implicit = data['structured_implicit']
                ldif_center, ldif_coef = data['obj_center'], data['obj_coef']
                centers = data['structured_implicit'].all_centers.clone()
                centers[:, :, 2] *= -1
                obj_samples = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                element_count = centers.shape[1]
                max_sample_points = (data['split'][:, 1] - data['split'][:, 0]).max() * element_count
                other_obj_samples = torch.zeros([len(obj_samples), max_sample_points, 3], device=centers.device)
                for start, end in data['split']:
                    other_obj_sample = obj_samples[start:end].reshape(1, -1, 3).expand(end - start, -1, -1)
                    other_obj_samples[start:end, :other_obj_sample.shape[1]] = other_obj_sample
                other_obj_samples = recover_points_to_obj_sys(bdb3D_form, other_obj_samples, ldif_center, ldif_coef)
                other_obj_samples[:, :, 2] *= -1
                est_sdf = data['mgn'](samples=other_obj_samples, structured_implicit=structured_implicit.dict(), apply_class_transfer=False)['global_decisions'] + 0.07
                v = [est_sdf[start:end, :(end - start) * element_count].reshape(-1, element_count) for start, end in data['split']]
                v = torch.cat(v)
            else:
                raise NotImplementedError
            if type == 'obj' or k in ('g_features', 'ldif_phy'):
                features.append(v)
            else:
                features_rel = []
                for start, end in data['split']:
                    features_rel.append(torch.stack([torch.cat([loc1, loc2], -1) for loc1 in v[start:end] for loc2 in v[start:end]]))
                features.append(torch.cat(features_rel, 0))
        return torch.cat(features, -1)

    def _get_layout_features(self, data):
        features = []
        keys = self.lo_features
        for k in keys:
            if k in ['pitch_reg_result', 'roll_reg_result', 'pitch_cls_result', 'roll_cls_result', 'lo_ori_reg_result', 'lo_ori_cls_result', 'lo_centroid_result', 'lo_coeffs_result', 'lo_afeatures']:
                v = data[k]
            elif k == 'K':
                v = self._K2feature(data[k])
            else:
                raise NotImplementedError
            features.append(v)
        return torch.cat(features, -1)

    def _get_map(self, data):
        device = data['g_features'].device
        split = data['split']
        obj_num = split[-1][-1] + split.shape[0]
        obj_obj_map = torch.zeros([obj_num, obj_num])
        rel_inds = []
        rel_masks = []
        obj_masks = torch.zeros(obj_num, dtype=torch.bool)
        lo_masks = torch.zeros(obj_num, dtype=torch.bool)
        for lo_index, (start, end) in enumerate(split):
            start = start + lo_index
            end = end + lo_index + 1
            obj_obj_map[start:end, start:end] = 1
            obj_ind = torch.arange(start, end, dtype=torch.long)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind)
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1]
            rel_inds.append(rel_ind_i[rel_mask_i])
            rel_masks.append(rel_mask_i)
            obj_masks[start:end - 1] = True
            lo_masks[end - 1] = True
        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)
        subj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])
        obj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])
        subj_pred_map.scatter_(0, rel_inds[:, 0].view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, 1].view(1, -1), 1)
        return rel_masks, obj_masks, lo_masks, obj_obj_map, subj_pred_map, obj_pred_map

    def forward(self, output):
        maps = self._get_map(output)
        if maps is None:
            return {}
        rel_masks, obj_masks, lo_masks, obj_obj_map, subj_pred_map, obj_pred_map = maps
        x_obj, x_pred = self._get_object_features(output, 'obj'), self._get_object_features(output, 'rel')
        x_obj, x_pred = self.obj_embedding(x_obj), self.rel_embedding(x_pred)
        x_lo = self._get_layout_features(output)
        x_lo = self.lo_embedding(x_lo)
        x_obj_lo = []
        x_pred_objlo = []
        rel_pair = output['rel_pair_counts']
        for lo_index, (start, end) in enumerate(output['split']):
            x_obj_lo.append(x_obj[start:end])
            x_obj_lo.append(x_lo[lo_index:lo_index + 1])
            x_pred_objlo.append(x_pred[rel_pair[lo_index]:rel_pair[lo_index + 1]].reshape(end - start, end - start, -1))
            x_pred_objlo[-1] = F.pad(x_pred_objlo[-1].permute(2, 0, 1), [0, 1, 0, 1], 'constant', 0.001).permute(1, 2, 0)
            x_pred_objlo[-1] = x_pred_objlo[-1].reshape((end - start + 1) ** 2, -1)
        x_obj = torch.cat(x_obj_lo)
        x_pred = torch.cat(x_pred_objlo)
        x_pred = x_pred[rel_masks]
        """feature level agcn"""
        obj_feats = [x_obj]
        pred_feats = [x_pred]
        start = 0
        for group, (gcn_collect_feat, gcn_update_feat) in enumerate(zip(self.gcn_collect_feat, self.gcn_update_feat)):
            for t in range(start, start + self.feat_update_step):
                """update object features"""
                source_obj = gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map, 4)
                source_rel_sub = gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map, 0)
                source_rel_obj = gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map, 1)
                source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
                obj_feats.append(gcn_update_feat(obj_feats[t], source2obj_all, 0))
                """update predicate features"""
                source_obj_sub = gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(), 2)
                source_obj_obj = gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(), 3)
                source2rel_all = (source_obj_sub + source_obj_obj) / 2
                pred_feats.append(gcn_update_feat(pred_feats[t], source2rel_all, 1))
            if self.res_group and group > 0:
                obj_feats[-1] += obj_feats[start]
                pred_feats[-1] += pred_feats[start]
            start += self.feat_update_step
        obj_feats_wolo = obj_feats[-1][obj_masks]
        size = self.fc1(obj_feats_wolo)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)
        ori = self.fc3(obj_feats_wolo)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]
        centroid = self.fc5(obj_feats_wolo)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]
        offset = self.fc_off_1(obj_feats_wolo)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)
        obj_feats_lo = obj_feats[-1][lo_masks]
        cam = self.fc_1(obj_feats_lo)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0:self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN:self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2:self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN:self.PITCH_BIN * 2 + self.ROLL_BIN * 2]
        lo = self.fc_layout(obj_feats_lo)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]
        if self.res_output:
            size += output['size_reg_result']
            ori_reg += output['ori_reg_result']
            ori_cls += output['ori_cls_result']
            centroid_reg += output['centroid_reg_result']
            centroid_cls += output['centroid_cls_result']
            offset += output['offset_2D_result']
            pitch_reg += output['pitch_reg_result']
            pitch_cls += output['pitch_cls_result']
            roll_reg += output['roll_reg_result']
            roll_cls += output['roll_cls_result']
            lo_ori_reg += output['lo_ori_reg_result']
            lo_ori_cls += output['lo_ori_cls_result']
            lo_centroid += output['lo_centroid_result']
            lo_coeffs += output['lo_coeffs_result']
        return {'size_reg_result': size, 'ori_reg_result': ori_reg, 'ori_cls_result': ori_cls, 'centroid_reg_result': centroid_reg, 'centroid_cls_result': centroid_cls, 'offset_2D_result': offset, 'pitch_reg_result': pitch_reg, 'pitch_cls_result': pitch_cls, 'roll_reg_result': roll_reg, 'roll_cls_result': roll_cls, 'lo_ori_reg_result': lo_ori_reg, 'lo_ori_cls_result': lo_ori_cls, 'lo_centroid_result': lo_centroid, 'lo_coeffs_result': lo_coeffs}


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


class PoseNet(nn.Module):

    def __init__(self, cfg, optim_spec=None):
        super(PoseNet, self).__init__()
        """Optimizer parameters used in training"""
        self.optim_spec = optim_spec
        """Module parameters"""
        bin = cfg.dataset_config.bins
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])
        """Modules"""
        self.resnet = resnet.resnet34(pretrained=False)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        self.fc_layout = nn.Linear(2048, 2048)
        self.fc_3 = nn.Linear(2048, 1024)
        self.fc_4 = nn.Linear(1024, self.LO_ORI_BIN * 2)
        self.fc_5 = nn.Linear(2048, 1024)
        self.fc_6 = nn.Linear(1024, 6)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def forward(self, x):
        x = self.resnet(x)
        cam = self.fc_1(x)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0:self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN:self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2:self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN:self.PITCH_BIN * 2 + self.ROLL_BIN * 2]
        lo = self.fc_layout(x)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]
        return pitch_reg, roll_reg, pitch_cls, roll_cls, lo_ori_reg, lo_ori_cls, lo_centroid, lo_coeffs, x


class PointGenCon(nn.Module):

    def __init__(self, bottleneck_size=2500, output_dim=3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size // 4, output_dim, 1)
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class EREstimate(nn.Module):

    def __init__(self, bottleneck_size=2500, output_dim=3):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size // 4, output_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


number_pnts_on_template = 2562


def sample_points_on_edges(points, edges, quantity=1, mode='train'):
    n_batch = edges.shape[0]
    n_edges = edges.shape[1]
    if mode == 'train':
        weights = np.diff(np.sort(np.vstack([np.zeros((1, n_edges * quantity)), np.random.uniform(0, 1, size=(1, n_edges * quantity)), np.ones((1, n_edges * quantity))]), axis=0), axis=0)
    else:
        weights = 0.5 * np.ones((2, n_edges * quantity))
    weights = weights.reshape([2, quantity, n_edges])
    weights = torch.from_numpy(weights).float()
    weights = weights.transpose(1, 2)
    weights = weights.transpose(0, 1).contiguous()
    weights = weights.expand(n_batch, n_edges, 2, quantity).contiguous()
    weights = weights.view(n_batch * n_edges, 2, quantity)
    left_nodes = torch.gather(points.transpose(1, 2), 1, (edges[:, :, 0] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    right_nodes = torch.gather(points.transpose(1, 2), 1, (edges[:, :, 1] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    edge_points = torch.cat([left_nodes.unsqueeze(-1), right_nodes.unsqueeze(-1)], -1).view(n_batch * n_edges, 3, 2)
    new_point_set = torch.bmm(edge_points, weights).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges, 3, quantity)
    new_point_set = new_point_set.transpose(2, 3).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges * quantity, 3)
    new_point_set = new_point_set.transpose(1, 2).contiguous()
    return new_point_set


def load_template(number):
    file_name = './data/sphere%d.pkl' % number
    with open(file_name, 'rb') as file:
        sphere_obj = pickle.load(file)
        sphere_points_normals = torch.from_numpy(sphere_obj['v']).float()
        sphere_faces = torch.from_numpy(sphere_obj['f']).long()
        sphere_adjacency = torch.from_numpy(sphere_obj['adjacency'].todense()).long()
        sphere_edges = torch.from_numpy(sphere_obj['edges']).long()
        sphere_edge2face = torch.from_numpy(sphere_obj['edge2face'].todense()).type(torch.uint8)
    return sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DensTMNet(nn.Module):

    def __init__(self, cfg, optim_spec=None, bottleneck_size=1024, n_classes=pix3d_n_classes, pretrained_encoder=True):
        super(DensTMNet, self).__init__()
        """Optimizer parameters used in training"""
        self.optim_spec = optim_spec
        """Module parameters"""
        self.num_points = number_pnts_on_template
        self.subnetworks = cfg.config['data']['tmn_subnetworks']
        self.train_e_e = cfg.config['data']['with_edge_classifier']
        """Modules"""
        self.encoder = resnet.resnet18_full(pretrained=False, num_classes=1024, input_channels=4 if cfg.config['data'].get('mask', False) else 3)
        self.decoders = nn.ModuleList([PointGenCon(bottleneck_size=3 + bottleneck_size + n_classes) for i in range(0, self.subnetworks)])
        if self.train_e_e:
            self.error_estimators = nn.ModuleList([EREstimate(bottleneck_size=3 + bottleneck_size + n_classes, output_dim=1) for i in range(0, max(self.subnetworks - 1, 1))])
            self.face_samples = cfg.config['data']['face_samples']
        self.apply(weights_init)
        if pretrained_encoder:
            pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
            model_dict = self.encoder.state_dict()
            if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
                model_dict['conv1.weight'][:, :3, ...] = pretrained_dict['conv1.weight']
                pretrained_dict.pop('conv1.weight')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)

    def unfreeze_parts(self, loose_parts):
        for param in self.parameters():
            param.requires_grad = False
        None
        if 'encoder' in loose_parts:
            for param in self.encoder.parameters():
                param.requires_grad = True
            None

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        None

    def freeze_by_stage(self, stage, loose_parts):
        if stage >= 1:
            for param in self.parameters():
                param.requires_grad = False
            None
            if 'decoder' in loose_parts:
                for param in self.decoders[-1].parameters():
                    param.requires_grad = True
                None
            if 'ee' in loose_parts and hasattr(self, 'error_estimators'):
                for param in self.error_estimators[-1].parameters():
                    param.requires_grad = True
                None

    def forward(self, image, size_cls, threshold=0.1, factor=1.0, mask_status=None, reconstruction='mesh'):
        mode = 'train' if self.training else 'test'
        device = image.device
        n_batch = image.size(0)
        n_edges = sphere_edges.shape[0]
        image = image.contiguous()
        afeature = self.encoder(image)
        code = torch.cat([afeature, size_cls], 1)
        if mask_status is not None:
            code4recon = code[mask_status.nonzero()]
            n_batch = code4recon.size(0)
            if n_batch == 0:
                return {'mgn_afeature': afeature}
        else:
            code4recon = code
        if reconstruction is None:
            return {'mgn_afeature': afeature}
        if mode == 'test':
            current_faces = sphere_faces.clone().unsqueeze(0)
            current_faces = current_faces.repeat(n_batch, 1, 1)
        else:
            current_faces = None
        current_edges = sphere_edges.clone().unsqueeze(0)
        current_edges = current_edges.repeat(n_batch, 1, 1)
        current_shape_grid = sphere_points_normals[:, :3].t().expand(n_batch, 3, self.num_points)
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []
        boundary_point_ids = torch.zeros(size=(n_batch, self.num_points), dtype=torch.uint8)
        remove_edges_list = []
        for i in range(self.subnetworks):
            current_image_grid = code4recon.unsqueeze(2).expand(code4recon.size(0), code4recon.size(1), current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            current_shape_grid = current_shape_grid + self.decoders[i](current_image_grid)
            out_shape_points.append(current_shape_grid)
            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(n_batch):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1
                return {'mesh_coordinates_results': out_shape_points, 'points_from_edges': out_sampled_mesh_points, 'point_indicators': out_indicators, 'output_edges': current_edges, 'boundary_point_ids': boundary_point_ids, 'faces': current_faces, 'mgn_afeature': afeature}
            if self.train_e_e:
                sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)
                out_sampled_mesh_points.append(sampled_points)
                current_image_grid = code4recon.unsqueeze(2).expand(code4recon.size(0), code4recon.size(1), sampled_points.size(2)).contiguous()
                current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()
                indicators = self.error_estimators[i](current_image_grid)
                indicators = indicators.view(n_batch, 1, n_edges, self.face_samples)
                indicators = indicators.squeeze(1)
                indicators = torch.mean(indicators, dim=2)
                out_indicators.append(indicators)
                remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
                remove_edges_list.append(remove_edges)
                for batch_id in range(n_batch):
                    rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                    if len(rm_edges) > 0:
                        current_edges[batch_id][rm_edges, :] = 1
                        if mode == 'test':
                            current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1
                threshold *= factor
        return {'mesh_coordinates_results': out_shape_points, 'points_from_edges': out_sampled_mesh_points, 'point_indicators': out_indicators, 'output_edges': current_edges, 'boundary_point_ids': boundary_point_ids, 'faces': current_faces, 'mgn_afeature': afeature}


class BatchedCBNLayer(nn.Module):

    def __init__(self, f_dim=32):
        super(BatchedCBNLayer, self).__init__()
        self.fc_beta = nn.Linear(f_dim, f_dim)
        self.fc_gamma = nn.Linear(f_dim, f_dim)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

    def forward(self, shape_embedding, sample_embeddings):
        beta = self.fc_beta(shape_embedding)
        gamma = self.fc_gamma(shape_embedding)
        if self.training:
            batch_mean, batch_variance = sample_embeddings.mean().detach(), sample_embeddings.var().detach()
            self.running_mean = 0.995 * self.running_mean + 0.005 * batch_mean
            self.running_var = 0.995 * self.running_var + 0.005 * batch_variance
        sample_embeddings = (sample_embeddings - self.running_mean) / torch.sqrt(self.running_var + 1e-05)
        out = gamma.unsqueeze(1) * sample_embeddings + beta.unsqueeze(1)
        return out


class BatchedOccNetResnetLayer(nn.Module):

    def __init__(self, f_dim=32):
        super(BatchedOccNetResnetLayer, self).__init__()
        self.bn1 = BatchedCBNLayer(f_dim=f_dim)
        self.fc1 = nn.Linear(f_dim, f_dim)
        self.bn2 = BatchedCBNLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, f_dim)

    def forward(self, shape_embedding, sample_embeddings):
        sample_embeddings = self.bn1(shape_embedding, sample_embeddings)
        init_sample_embeddings = sample_embeddings
        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc1(sample_embeddings)
        sample_embeddings = self.bn2(shape_embedding, sample_embeddings)
        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc2(sample_embeddings)
        return init_sample_embeddings + sample_embeddings


class OccNetDecoder(nn.Module):

    def __init__(self, f_dim=32):
        super(OccNetDecoder, self).__init__()
        self.fc1 = nn.Linear(3, f_dim)
        self.resnet = BatchedOccNetResnetLayer(f_dim=f_dim)
        self.bn = BatchedCBNLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, 1)

    def write_occnet_file(self, path):
        """Serializes an occnet network and writes it to disk."""
        f = file_util.open_file(path, 'wb')

        def write_fc_layer(layer):
            weights = layer.weight.t().cpu().numpy()
            biases = layer.bias.cpu().numpy()
            f.write(weights.astype('f').tostring())
            f.write(biases.astype('f').tostring())

        def write_cbn_layer(layer):
            write_fc_layer(layer.fc_beta)
            write_fc_layer(layer.fc_gamma)
            running_mean = layer.running_mean.item()
            running_var = layer.running_var.item()
            f.write(struct.pack('ff', running_mean, running_var))
        f.write(struct.pack('ii', 1, self.fc1.out_features))
        write_fc_layer(self.fc1)
        write_cbn_layer(self.resnet.bn1)
        write_fc_layer(self.resnet.fc1)
        write_cbn_layer(self.resnet.bn2)
        write_fc_layer(self.resnet.fc2)
        write_cbn_layer(self.bn)
        weights = self.fc2.weight.t().cpu().numpy()
        bias = self.fc2.bias.data.item()
        f.write(weights.astype('f').tostring())
        f.write(struct.pack('f', bias))
        f.close()

    def forward(self, embedding, samples):
        sample_embeddings = self.fc1(samples)
        sample_embeddings = self.resnet(embedding, sample_embeddings)
        sample_embeddings = self.bn(embedding, sample_embeddings)
        vals = self.fc2(sample_embeddings)
        return vals


def _unflatten(config, vector):
    return torch.split(vector, [1, 3, 6, config['model']['mesh_reconstruction']['implicit_parameter_length']], -1)


def homogenize(m):
    m = F.pad(m, [0, 1, 0, 1], 'constant', 0)
    m[..., -1, -1] = 1
    return m


class StructuredImplicit(object):

    def __init__(self, config, constant, center, radius, iparam, net=None):
        self.config = config
        self.implicit_parameter_length = config['model']['mesh_reconstruction']['implicit_parameter_length']
        self.element_count = config['model']['mesh_reconstruction']['element_count']
        self.sym_element_count = config['model']['mesh_reconstruction']['sym_element_count']
        self.constants = constant
        self.radii = radius
        self.centers = center
        self.iparams = iparam
        self.effective_element_count = self.element_count + self.sym_element_count
        self.device = constant.device
        self.batch_size = constant.size(0)
        self.net = net
        self._packed_vector = None
        self._analytic_code = None
        self._all_centers = None

    @classmethod
    def from_packed_vector(cls, config, packed_vector, net):
        """Parse an already packed vector (NOT a network activation)."""
        constant, center, radius, iparam = _unflatten(config, packed_vector)
        return cls(config, constant, center, radius, iparam, net)

    @classmethod
    def from_activation(cls, config, activation, net):
        constant, center, radius, iparam = _unflatten(config, activation)
        constant = -torch.abs(constant)
        radius_var = torch.sigmoid(radius[..., :3])
        radius_var = 0.15 * radius_var
        radius_var = radius_var * radius_var
        max_euler_angle = np.pi / 4.0
        radius_rot = torch.clamp(radius[..., 3:], -max_euler_angle, max_euler_angle)
        radius = torch.cat([radius_var, radius_rot], -1)
        center /= 2
        return cls(config, constant, center, radius, iparam, net)

    def _tile_for_symgroups(self, elements):
        sym_elements = elements[:, :self.sym_element_count, ...]
        elements = torch.cat([elements, sym_elements], 1)
        return elements

    def _generate_symgroup_samples(self, samples):
        samples = samples.unsqueeze(1).expand(-1, self.element_count, -1, -1)
        sym_samples = samples[:, :self.sym_element_count].clone()
        sym_samples *= torch.tensor([-1, 1, 1], dtype=torch.float32, device=self.device)
        effective_samples = torch.cat([samples, sym_samples], 1)
        return effective_samples

    def compute_world2local(self):
        tx = torch.eye(3, device=self.device).expand(self.batch_size, self.element_count, -1, -1)
        centers = self.centers.unsqueeze(-1)
        tx = torch.cat([tx, -centers], -1)
        lower_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(self.batch_size, self.element_count, 1, -1)
        tx = torch.cat([tx, lower_row], -2)
        rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(self.radii[..., 3:6]).inverse()
        diag = 1.0 / (torch.sqrt(self.radii[..., :3] + 1e-08) + 1e-08)
        scale = torch.diag_embed(diag)
        tx3x3 = torch.matmul(scale, rotation)
        return torch.matmul(homogenize(tx3x3), tx)

    def implicit_values(self, local_samples):
        iparams = self._tile_for_symgroups(self.iparams)
        values = self.net.eval_implicit_parameters(iparams, local_samples)
        return values

    @property
    def all_centers(self):
        if self._all_centers is None:
            sym_centers = self.centers[:, :self.sym_element_count].clone()
            sym_centers[:, :, 0] *= -1
            self._all_centers = torch.cat([self.centers, sym_centers], 1)
        return self._all_centers

    def class_at_samples(self, samples, apply_class_transfer=True):
        effective_constants = self._tile_for_symgroups(self.constants)
        effective_centers = self._tile_for_symgroups(self.centers)
        effective_radii = self._tile_for_symgroups(self.radii)
        effective_samples = self._generate_symgroup_samples(samples)
        constants_quadrics = torch.zeros(self.batch_size, self.effective_element_count, 4, 4, device=self.device)
        constants_quadrics[:, :, -1:, -1] = effective_constants
        per_element_constants, per_element_weights = quadrics.compute_shape_element_influences(constants_quadrics, effective_centers, effective_radii, effective_samples)
        effective_world2local = self._tile_for_symgroups(self.compute_world2local())
        local_samples = torch.matmul(F.pad(effective_samples, [0, 1], 'constant', 1), effective_world2local.transpose(-1, -2))[..., :3]
        implicit_values = self.implicit_values(local_samples)
        residuals = 1 + implicit_values
        local_decisions = per_element_constants * per_element_weights * residuals
        local_weights = per_element_weights
        sdf = torch.sum(local_decisions, 1)
        if apply_class_transfer:
            sdf = torch.sigmoid(100 * (sdf + 0.07))
        return sdf, (local_decisions, local_weights)

    @property
    def vector(self):
        if self._packed_vector is None:
            self._packed_vector = torch.cat([self.constants, self.centers, self.radii, self.iparams], -1)
        return self._packed_vector

    @property
    def analytic_code(self):
        if self._analytic_code is None:
            self._analytic_code = torch.cat([self.constants, self.centers, self.radii], -1)
        return self._analytic_code

    def savetxt(self, path):
        assert self.vector.shape[0] == 1
        sif_vector = self.vector.squeeze().cpu().numpy()
        sif_vector[:, 4:7] = np.sqrt(np.maximum(sif_vector[:, 4:7], 0))
        out = 'SIF\n%i %i %i\n' % (self.element_count, 0, self.implicit_parameter_length)
        for row_idx in range(self.element_count):
            row = ' '.join(10 * ['%.9g']) % tuple(sif_vector[row_idx, :10].tolist())
            symmetry = int(row_idx < self.sym_element_count)
            row += ' %i' % symmetry
            implicit_params = ' '.join(self.implicit_parameter_length * ['%.9g']) % tuple(sif_vector[row_idx, 10:].tolist())
            row += ' ' + implicit_params
            row += '\n'
            out += row
        file_util.writetxt(path, out)

    def unbind(self):
        return [StructuredImplicit.from_packed_vector(self.config, self.vector[i:i + 1], self.net) for i in range(self.vector.size(0))]

    def __getitem__(self, item):
        return StructuredImplicit.from_packed_vector(self.config, self.vector[item], self.net)

    def dict(self):
        return {'constant': self.constants, 'radius': self.radii, 'center': self.centers, 'iparam': self.iparams}


class LDIF(nn.Module):

    def __init__(self, cfg, optim_spec=None, n_classes=pix3d_n_classes, pretrained_encoder=True):
        super(LDIF, self).__init__()
        """Optimizer parameters used in training"""
        self.optim_spec = optim_spec
        """Module parameters"""
        self.cfg = cfg
        self.bottleneck_size = cfg.config['model']['mesh_reconstruction'].get('bottleneck_size', 2048)
        cfg.config['model']['mesh_reconstruction']['bottleneck_size'] = self.bottleneck_size
        self.element_count = cfg.config['model']['mesh_reconstruction']['element_count']
        self.sym_element_count = cfg.config['model']['mesh_reconstruction']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        cfg.config['model']['mesh_reconstruction']['effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = cfg.config['model']['mesh_reconstruction']['implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        cfg.config['model']['mesh_reconstruction']['analytic_code_len'] = 10 * self.element_count
        cfg.config['model']['mesh_reconstruction']['structured_implicit_vector_len'] = self.element_embedding_length * self.element_count
        self._temp_folder = None
        """Modules"""
        self.encoder = resnet.resnet18_full(pretrained=False, num_classes=self.bottleneck_size, input_channels=4 if cfg.config['data'].get('mask', False) else 3)
        self.mlp = nn.Sequential(nn.Linear(self.bottleneck_size + n_classes, self.bottleneck_size), nn.LeakyReLU(0.2, True), nn.Linear(self.bottleneck_size, self.bottleneck_size), nn.LeakyReLU(0.2, True), nn.Linear(self.bottleneck_size, self.element_count * self.element_embedding_length))
        self.decoder = OccNetDecoder(f_dim=self.implicit_parameter_length)
        self.apply(weights_init)
        if pretrained_encoder:
            pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
            model_dict = self.encoder.state_dict()
            if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
                model_dict['conv1.weight'][:, :3, ...] = pretrained_dict['conv1.weight']
                pretrained_dict.pop('conv1.weight')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)

    def eval_implicit_parameters(self, implicit_parameters, samples):
        batch_size, element_count, element_embedding_length = list(implicit_parameters.shape)
        sample_count = samples.shape[-2]
        batched_parameters = torch.reshape(implicit_parameters, [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        vals = torch.reshape(batched_vals, [batch_size, element_count, sample_count, 1])
        return vals

    def extract_mesh(self, structured_implicit, resolution=64, extent=0.75, num_samples=10000, cuda=True, marching_cube=True):
        if cuda:
            mesh = []
            for s in structured_implicit.unbind():
                if self._temp_folder is None:
                    self._temp_folder = tempfile.mktemp(dir='/dev/shm')
                    os.makedirs(self._temp_folder)
                    self.decoder.write_occnet_file(os.path.join(self._temp_folder, 'serialized.occnet'))
                    shutil.copy('./external/ldif/ldif2mesh/ldif2mesh', self._temp_folder)
                si_path = os.path.join(self._temp_folder, 'ldif.txt')
                grd_path = os.path.join(self._temp_folder, 'grid.grd')
                s.savetxt(si_path)
                cmd = f"{os.path.join(self._temp_folder, 'ldif2mesh')} {si_path} {os.path.join(self._temp_folder, 'serialized.occnet')} {grd_path} -resolution {resolution} -extent {extent}"
                subprocess.check_output(cmd, shell=True)
                _, volume = file_util.read_grd(grd_path)
                _, m = extract_mesh.marching_cubes(volume, extent)
                mesh.append(m)
        else:
            mesh = mesh_util.reconstruction(structured_implicit=structured_implicit, resolution=resolution, b_min=np.array([-extent] * 3), b_max=np.array([extent] * 3), use_octree=True, num_samples=num_samples, marching_cube=marching_cube)
        return mesh

    def forward(self, image=None, size_cls=None, samples=None, occnet2gaps=None, structured_implicit=None, resolution=None, cuda=True, reconstruction='mesh', apply_class_transfer=True):
        return_dict = {}
        return_structured_implicit = structured_implicit
        if isinstance(structured_implicit, dict):
            structured_implicit = StructuredImplicit(config=self.cfg.config, **structured_implicit, net=self)
        elif structured_implicit is None or isinstance(structured_implicit, bool):
            embedding = self.encoder(image)
            return_dict['ldif_afeature'] = embedding
            embedding = torch.cat([embedding, size_cls], 1)
            structured_implicit_activations = self.mlp(embedding)
            structured_implicit_activations = torch.reshape(structured_implicit_activations, [-1, self.element_count, self.element_embedding_length])
            return_dict['structured_implicit_activations'] = structured_implicit_activations
            structured_implicit = StructuredImplicit.from_activation(self.cfg.config, structured_implicit_activations, self)
        else:
            raise NotImplementedError
        return_dict['structured_implicit'] = structured_implicit.dict()
        if return_structured_implicit is True:
            return return_dict
        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(samples, apply_class_transfer)
            return_dict.update({'global_decisions': global_decisions, 'element_centers': structured_implicit.centers})
            return return_dict
        elif reconstruction is not None:
            if resolution is None:
                resolution = self.cfg.config['data'].get('marching_cube_resolution', 128)
            mesh = self.extract_mesh(structured_implicit, extent=self.cfg.config['data']['bounding_box'], resolution=resolution, cuda=cuda, marching_cube=reconstruction == 'mesh')
            if reconstruction == 'mesh':
                if occnet2gaps is not None:
                    mesh = [(m.apply_transform(t.inverse().cpu().numpy()) if not isinstance(m, trimesh.primitives.Sphere) else m) for m, t in zip(mesh, occnet2gaps)]
                mesh_coordinates_results = []
                faces = []
                for m in mesh:
                    mesh_coordinates_results.append(torch.from_numpy(m.vertices).type(torch.float32).transpose(-1, -2))
                    faces.append(torch.from_numpy(m.faces) + 1)
                return_dict.update({'mesh': mesh, 'mesh_coordinates_results': [mesh_coordinates_results], 'faces': faces, 'element_centers': structured_implicit.centers})
            elif reconstruction == 'sdf':
                return_dict.update({'sdf': mesh[0], 'mat': mesh[1], 'element_centers': structured_implicit.centers})
            else:
                raise NotImplementedError
            return return_dict
        else:
            return return_dict

    def __del__(self):
        if self._temp_folder is not None:
            shutil.rmtree(self._temp_folder)


METHODS = Registry('method')


obj_cam_ratio = 1


class TOTAL3D(BaseNetwork):

    def __init__(self, cfg):
        """
        load submodules for the network.
        :param config: customized configurations.
        """
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            phase_names += ['layout_estimation']
        if cfg.config[cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            phase_names += ['object_detection']
        if cfg.config[cfg.config['mode']]['phase'] in ['joint']:
            phase_names += ['mesh_reconstruction']
            if 'output_adjust' in cfg.config['model'].keys():
                phase_names += ['output_adjust']
        if not cfg.config['model'] or not phase_names:
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')
        """load network blocks"""
        for phase_name in phase_names:
            if phase_name not in cfg.config['model'].keys():
                continue
            net_spec = cfg.config['model'][phase_name]
            method_name = net_spec['method']
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)
            """load corresponding loss functions"""
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(self.cfg.config['model'][phase_name].get('weight', 1), cfg.config))
        """Add joint loss"""
        setattr(self, 'joint_loss', LOSSES.get('JointLoss', 'Null')(1))
        """Multi-GPU setting"""
        if cfg.config[cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            self.layout_estimation = nn.DataParallel(self.layout_estimation)
        if cfg.config[cfg.config['mode']]['phase'] in ['joint']:
            self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)
        """freeze submodules or not"""
        self.freeze_modules(cfg)

    def get_extra_results(self, all_output):
        extra_results = {}
        if isinstance(self.mesh_reconstruction.module, MODULES.get('LDIF')):
            structured_implicit = all_output['structured_implicit']
            in_coor_min = structured_implicit.all_centers.min(dim=1)[0]
            in_coor_max = structured_implicit.all_centers.max(dim=1)[0]
            obj_center = (in_coor_max + in_coor_min) / 2.0
            obj_center[:, 2] *= -1
            obj_coef = (in_coor_max - in_coor_min) / 2.0
        else:
            if all_output['meshes'] is None:
                return {}
            obj_center = (all_output['meshes'].max(dim=0)[0] + all_output['meshes'].min(dim=0)[0]) / 2.0
            obj_center = obj_center.detach()
            obj_coef = (all_output['meshes'].max(dim=0)[0] - all_output['meshes'].min(dim=0)[0]) / 2.0
            obj_coef = obj_coef.detach()
        extra_results.update({'obj_center': obj_center, 'obj_coef': obj_coef})
        return extra_results

    def forward(self, data):
        all_output = {}
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            pitch_reg_result, roll_reg_result, pitch_cls_result, roll_cls_result, lo_ori_reg_result, lo_ori_cls_result, lo_centroid_result, lo_coeffs_result, a_features = self.layout_estimation(data['image'])
            layout_output = {'pitch_reg_result': pitch_reg_result, 'roll_reg_result': roll_reg_result, 'pitch_cls_result': pitch_cls_result, 'roll_cls_result': roll_cls_result, 'lo_ori_reg_result': lo_ori_reg_result, 'lo_ori_cls_result': lo_ori_cls_result, 'lo_centroid_result': lo_centroid_result, 'lo_coeffs_result': lo_coeffs_result, 'lo_afeatures': a_features}
            all_output.update(layout_output)
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            size_reg_result, ori_reg_result, ori_cls_result, centroid_reg_result, centroid_cls_result, offset_2D_result, a_features, r_features, a_r_features = self.object_detection(data['patch'], data['size_cls'], data['g_features'], data['split'], data['rel_pair_counts'])
            object_output = {'size_reg_result': size_reg_result, 'ori_reg_result': ori_reg_result, 'ori_cls_result': ori_cls_result, 'centroid_reg_result': centroid_reg_result, 'centroid_cls_result': centroid_cls_result, 'offset_2D_result': offset_2D_result, 'odn_afeature': a_features, 'odn_rfeatures': r_features, 'odn_arfeatures': a_r_features}
            all_output.update(object_output)
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            if self.cfg.config['mode'] == 'train':
                if isinstance(self.mesh_reconstruction.module, MODULES.get('LDIF')):
                    mesh_output = self.mesh_reconstruction(data['patch_for_mesh'], data['cls_codes'], structured_implicit=True)
                else:
                    mesh_output = self.mesh_reconstruction(data['patch_for_mesh'], data['cls_codes'], mask_status=data['mask_status'])
                    if data['mask_flag'] == 1:
                        meshes = mesh_output['mesh_coordinates_results'][-1]
                        mesh_output['meshes'] = meshes
                    else:
                        mesh_output['meshes'] = None
            else:
                reconstruction = 'mesh' if self.cfg.config['full'] else None
                mesh_output = self.mesh_reconstruction(data['patch_for_mesh'], data['cls_codes'], reconstruction=reconstruction)
                out_points = mesh_output.get('mesh_coordinates_results', [None])
                out_faces = mesh_output.get('faces', None)
                mesh_output.update({'meshes': out_points[-1], 'out_faces': out_faces})
            if 'structured_implicit' in mesh_output:
                mesh_output['structured_implicit'] = StructuredImplicit(config=self.cfg.config, **mesh_output['structured_implicit'])
            if mesh_output.get('meshes') is not None:
                if isinstance(mesh_output['meshes'], list):
                    for m in mesh_output['meshes']:
                        m[2, :] *= -1
                elif mesh_output['meshes'] is not None:
                    mesh_output['meshes'][:, 2, :] *= -1
            mesh_output['mgn'] = self.mesh_reconstruction
            all_output.update(mesh_output)
            all_output.update(self.get_extra_results(all_output))
            if hasattr(self, 'output_adjust'):
                input = all_output.copy()
                input['size_cls'] = data['size_cls']
                input['cls_codes'] = data['cls_codes']
                input['g_features'] = data['g_features']
                input['bdb2D_pos'] = data['bdb2D_pos']
                input['K'] = data['K']
                input['split'] = data['split']
                input['rel_pair_counts'] = data['rel_pair_counts']
                refined_output = self.output_adjust(input)
                all_output.update(refined_output)
        if all_output:
            return all_output
        else:
            raise NotImplementedError

    def loss(self, est_data, gt_data):
        """
        calculate loss of est_out given gt_out.
        """
        loss_weights = self.cfg.config.get('loss_weights', {})
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            layout_loss, layout_results = self.layout_estimation_loss(est_data, gt_data, self.cfg.bins_tensor)
            layout_loss_weighted = {k: (v * loss_weights.get(k, 1.0)) for k, v in layout_loss.items()}
            total_layout_loss = sum(layout_loss_weighted.values())
            total_layout_loss_unweighted = sum([v.detach() for v in layout_loss.values()])
            for key, value in layout_loss.items():
                layout_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            object_loss = self.object_detection_loss(est_data, gt_data)
            object_loss_weighted = {k: (v * loss_weights.get(k, 1.0)) for k, v in object_loss.items()}
            total_object_loss = sum(object_loss_weighted.values())
            total_object_loss_unweighted = sum([v.detach() for v in object_loss.values()])
            for key, value in object_loss.items():
                object_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            joint_loss, extra_results = self.joint_loss(est_data, gt_data, self.cfg.bins_tensor, layout_results)
            joint_loss_weighted = {k: (v * loss_weights.get(k, 1.0)) for k, v in joint_loss.items()}
            mesh_loss = self.mesh_reconstruction_loss(est_data, gt_data, extra_results)
            mesh_loss_weighted = {k: (v * loss_weights.get(k, 1.0)) for k, v in mesh_loss.items()}
            total_joint_loss = sum(joint_loss_weighted.values()) + sum(mesh_loss_weighted.values())
            total_joint_loss_unweighted = sum([v.detach() for v in joint_loss.values()]) + sum([(v.detach() if isinstance(v, torch.Tensor) else v) for v in mesh_loss.values()])
            for key, value in mesh_loss.items():
                mesh_loss[key] = float(value)
            for key, value in joint_loss.items():
                joint_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'layout_estimation':
            return {'total': total_layout_loss, **layout_loss, 'total_unweighted': total_layout_loss_unweighted}
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'object_detection':
            return {'total': total_object_loss, **object_loss, 'total_unweighted': total_object_loss_unweighted}
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            total3d_loss = total_object_loss + total_joint_loss + obj_cam_ratio * total_layout_loss
            total3d_loss_unweighted = total_object_loss_unweighted + total_joint_loss_unweighted + obj_cam_ratio * total_layout_loss_unweighted
            return {'total': total3d_loss, **layout_loss, **object_loss, **mesh_loss, **joint_loss, 'total_unweighted': total3d_loss_unweighted}
        else:
            raise NotImplementedError


class Relation_Config(object):

    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16


rel_cfg = Relation_Config()


class RelationNet(nn.Module):

    def __init__(self):
        super(RelationNet, self).__init__()
        self.fc_g = nn.Linear(rel_cfg.d_g, rel_cfg.Nr)
        self.threshold = nn.Threshold(1e-06, 1e-06)
        self.softmax = nn.Softmax(dim=1)
        self.fc_K = nn.Linear(2048, rel_cfg.d_k * rel_cfg.Nr)
        self.fc_Q = nn.Linear(2048, rel_cfg.d_k * rel_cfg.Nr)
        self.conv_s = nn.Conv1d(1, 1, 1)

    def forward(self, a_features, g_features, split, rel_pair_counts):
        """
        Extract relational features from appearance feature and geometric feature (see Hu et al. [2]).
        :param a_features: Patch_size x 2048 (Appearance_feature_size)
        a_features records the ResNet-34 feature for each object in Patch.
        :param g_features: SUM(N_i^2) x 64 (i.e. Number_of_object_pairs x Geometric_feature_size)
        g_features records the geometric features (64-D) between each pair of objects (see Hu et al. [2]). So the dimension
        is Number_of_pairs_in_images x 64 (or SUM(N_i^2) x 64). N_i is the number of objects in the i-th image.
        :param split: Batch_size x 2
        split records which batch a object belongs to.
        e.g. split = torch.tensor([[0, 5], [5, 8]]) when batch size is 2, and there are 5 objects in the first batch and
        3 objects in the second batch.
        Then the first 5 objects in the whole patch belongs to the first batch, and the rest belongs to the second batch.
        :param rel_pair_counts: (Batch_size + 1)
        rel_pair_counts records which batch a geometric feature belongs to, and gives the start and end index.
        e.g. rel_pair_counts = torch.tensor([0, 49, 113]).
        The batch size is two. The first 49 geometric features are from the first batch.
        The index begins from 0 and ends at 49. The second 64 geometric features are from the second batch.
        The index begins from 49 and ends at 113.
        :return: Relational features for each object.
        """
        g_weights = self.fc_g(g_features)
        g_weights = self.threshold(g_weights)
        g_weights = g_weights.transpose(0, 1)
        k_features = self.fc_K(a_features)
        q_features = self.fc_Q(a_features)
        k_features = k_features.view(-1, rel_cfg.Nr, rel_cfg.d_k).transpose(0, 1)
        q_features = q_features.view(-1, rel_cfg.Nr, rel_cfg.d_k).transpose(0, 1)
        v_features = a_features.view(a_features.size(0), rel_cfg.Nr, -1).transpose(0, 1)
        r_features = []
        for interval_idx, interval in enumerate(split):
            sample_k_features = k_features[:, interval[0]:interval[1], :]
            sample_q_features = q_features[:, interval[0]:interval[1], :]
            sample_a_weights = torch.div(torch.bmm(sample_k_features, sample_q_features.transpose(1, 2)), math.sqrt(rel_cfg.d_k))
            sample_g_weights = g_weights[:, rel_pair_counts[interval_idx]:rel_pair_counts[interval_idx + 1]]
            sample_g_weights = sample_g_weights.view(sample_g_weights.size(0), interval[1] - interval[0], interval[1] - interval[0])
            fin_weight = self.softmax(torch.log(sample_g_weights) + sample_a_weights)
            sample_v_features = v_features[:, interval[0]:interval[1], :]
            sample_r_feature = torch.bmm(sample_v_features.transpose(1, 2), fin_weight)
            sample_r_feature = sample_r_feature.view(sample_r_feature.size(0) * sample_r_feature.size(1), sample_r_feature.size(2)).transpose(0, 1)
            r_features.append(sample_r_feature)
        r_features = torch.cat(r_features, 0)
        r_features = self.conv_s(r_features.unsqueeze(1)).squeeze(1)
        return r_features


class Bdb3DNet(nn.Module):

    def __init__(self, cfg, optim_spec=None):
        super(Bdb3DNet, self).__init__()
        """Optimizer parameters used in training"""
        self.optim_spec = optim_spec
        """Module parameters"""
        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])
        self.resnet = nn.DataParallel(resnet.resnet34(pretrained=False))
        self.relnet = RelationNet()
        self.fc1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc2 = nn.Linear(128, 3)
        self.fc3 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc4 = nn.Linear(128, self.OBJ_ORI_BIN * 2)
        self.fc5 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc_centroid = nn.Linear(128, self.OBJ_CENTER_BIN * 2)
        self.fc_off_1 = nn.Linear(2048 + len(NYU40CLASSES), 128)
        self.fc_off_2 = nn.Linear(128, 2)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

    def forward(self, x, size_cls, g_features, split, rel_pair_counts):
        """
        Extract relational features for object bounding box estimation.

        The definition of 'batch' in train.py indicates the number of images we process in a single forward broadcasting.
        In this implementation, we speed-up the efficiency by processing all objects in a batch in parallel.

        As each image contains various number (N_i) of objects, it refers to an issue to assign which image an object belongs to.
        We address the problem by involving a look-up table in 'split'.

        Therefore, The meaning of 'batch' in this function actually refers to a 'patch' of objects.

        :param x: Patch_size x Channel_size x Height x Width
        :param size_cls: Patches x Number_of_classes.
        :param g_features: SUM(N_i^2) x 64
        g_features records the geometric features (64-D) between each pair of objects (see Hu et al. [2]). So the dimension
        is Number_of_pairs_in_images x 64 (or SUM(N_i^2) x 64). N_i is the number of objects in the i-th image.
        :param split: Batch_size x 2
        split records which batch a object belongs to.
        e.g. split = torch.tensor([[0, 5], [5, 8]]) when batch size is 2, and there are 5 objects in the first batch and
        3 objects in the second batch.
        Then the first 5 objects in the whole patch belongs to the first batch, and the rest belongs to the second batch.
        :param rel_pair_counts: (Batch_size + 1)
        rel_pair_counts records which batch a geometric feature belongs to, and gives the start and end index.
        e.g. rel_pair_counts = torch.tensor([0, 49, 113]).
        The batch size is two. The first 49 geometric features are from the first batch.
        The index begins from 0 and ends at 49. The second 64 geometric features are from the second batch.
        The index begins from 49 and ends at 113.
        :return: Object bounding box properties.
        """
        a_features = self.resnet(x)
        a_features = a_features.view(a_features.size(0), -1)
        r_features = self.relnet(a_features, g_features, split, rel_pair_counts)
        a_r_features = torch.add(a_features, r_features)
        a_r_features_cat = torch.cat([a_r_features, size_cls], 1)
        size = self.fc1(a_r_features_cat)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)
        ori = self.fc3(a_r_features_cat)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]
        centroid = self.fc5(a_r_features_cat)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]
        offset = self.fc_off_1(a_r_features_cat)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)
        return size, ori_reg, ori_cls, centroid_reg, centroid_cls, offset, a_features, r_features, a_r_features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Collection_Unit,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (_GraphConvolutionLayer_Collect,
     lambda: ([], {'dim_obj': 4, 'dim_rel': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (_GraphConvolutionLayer_Update,
     lambda: ([], {'dim_obj': 4, 'dim_rel': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (_Update_Unit,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_chengzhag_Implicit3DUnderstanding(_paritybench_base):
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

