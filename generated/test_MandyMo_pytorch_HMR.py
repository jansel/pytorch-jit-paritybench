import sys
_module = sys.modules[__name__]
del sys
main = _module
Discriminator = _module
HourGlass = _module
LinearModel = _module
PRNetEncoder = _module
Resnet = _module
SMPL = _module
config = _module
AICH_dataloader = _module
COCO2017_dataloader = _module
eval_dataloader = _module
hum36m_dataloader = _module
lsp_dataloader = _module
lsp_ext_dataloader = _module
mosh_dataloader = _module
mpi_inf_3dhp_dataloader = _module
densenet = _module
model = _module
timer = _module
trainer = _module
util = _module

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


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import torch.optim as optim


import math


import torchvision


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import scipy.io as scio


import random


import re


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


from torch.utils.data import ConcatDataset


import time


from torch.autograd import Variable


from scipy import interpolate


class PoseDiscriminator(nn.Module):

    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()
        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(channels[-1])
            sys.exit(msg)
        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(name='conv_{}'.format(idx), module=nn.Conv2d(in_channels=channels[idx], out_channels=channels[idx + 1], kernel_size=1, stride=1))
        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features=channels[l - 2], out_features=1))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2)
        internal_outputs = self.conv_blocks(inputs)
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))
        return torch.cat(o, 1), internal_outputs


class LinearModel(nn.Module):
    """
        input param:
            fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
            use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
            drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
            use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    """

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_ac_func = use_ac_func
        if not self._check():
            msg = 'wrong LinearModel parameters!'
            None
            sys.exit(msg)
        self.create_layers()

    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                None
                break
            if not isinstance(self.use_dropout, list):
                None
                break
            if not isinstance(self.drop_prob, list):
                None
                break
            if not isinstance(self.use_ac_func, list):
                None
                break
            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_porb = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)
            return l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_porb < l_fc_layer and l_use_ac_func < l_fc_layer and l_drop_porb == l_use_drop
        return False

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)
        self.fc_blocks = nn.Sequential()
        for _ in range(l_fc_layer - 1):
            self.fc_blocks.add_module(name='regressor_fc_{}'.format(_), module=nn.Linear(in_features=self.fc_layers[_], out_features=self.fc_layers[_ + 1]))
            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(name='regressor_af_{}'.format(_), module=nn.ReLU())
            if _ < l_use_drop and self.use_dropout[_]:
                self.fc_blocks.add_module(name='regressor_fc_dropout_{}'.format(_), module=nn.Dropout(p=self.drop_prob[_]))

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)


class FullPoseDiscriminator(LinearModel):

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)
        super(FullPoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class ShapeDiscriminator(LinearModel):

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)
        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self._read_configs()
        self._create_sub_modules()

    def _read_configs(self):
        self.beta_count = args.beta_count
        self.smpl_model = args.smpl_model
        self.smpl_mean_theta_path = args.smpl_mean_theta_path
        self.total_theta_count = args.total_theta_count
        self.joint_count = args.joint_count
        self.feature_count = args.feature_count

    def _create_sub_modules(self):
        """
            create theta discriminator for 23 joint
        """
        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])
        """
            create full pose discriminator for total 23 joints
        """
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)
        """
            shape discriminator for betas
        """
        fc_layers = [self.beta_count, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)
        None
    """
        inputs is N x 85(3 + 72 + 10)
    """

    def forward(self, thetas):
        batch_size = thetas.shape[0]
        cams, poses, shapes = thetas[:, :3], thetas[:, 3:75], thetas[:, 75:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)


class Residual(nn.Module):

    def __init__(self, use_bn, input_channels, out_channels, mid_channels, kernel_size=3, padding=1, stride=1):
        super(Residual, self).__init__()
        self.use_bn = use_bn
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.mid_channels = mid_channels
        self.down_channel = nn.Conv2d(input_channels, self.mid_channels, kernel_size=1)
        self.AcFunc = nn.ReLU()
        if use_bn:
            self.bn_0 = nn.BatchNorm2d(num_features=self.mid_channels)
            self.bn_1 = nn.BatchNorm2d(num_features=self.mid_channels)
            self.bn_2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.conv = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.up_channel = nn.Conv2d(self.mid_channels, out_channels, kernel_size=1)
        if input_channels != out_channels:
            self.trans = nn.Conv2d(input_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        x = self.down_channel(inputs)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.AcFunc(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.AcFunc(x)
        x = self.up_channel(x)
        if self.input_channels != self.out_channels:
            x += self.trans(inputs)
        else:
            x += inputs
        if self.use_bn:
            x = self.bn_2(x)
        return self.AcFunc(x)


class HourGlassBlock(nn.Module):

    def __init__(self, block_count, residual_each_block, input_channels, mid_channels, use_bn, stack_index):
        super(HourGlassBlock, self).__init__()
        self.block_count = block_count
        self.residual_each_block = residual_each_block
        self.use_bn = use_bn
        self.stack_index = stack_index
        self.input_channels = input_channels
        self.mid_channels = mid_channels
        if self.block_count == 0:
            self.process = nn.Sequential()
            for _ in range(residual_each_block * 3):
                self.process.add_module(name='inner_{}_{}'.format(self.stack_index, _), module=Residual(input_channels=input_channels, out_channels=input_channels, mid_channels=mid_channels, use_bn=use_bn))
        else:
            self.down_sampling = nn.Sequential()
            self.down_sampling.add_module(name='down_sample_{}_{}'.format(self.stack_index, self.block_count), module=nn.MaxPool2d(kernel_size=2, stride=2))
            for _ in range(residual_each_block):
                self.down_sampling.add_module(name='residual_{}_{}_{}'.format(self.stack_index, self.block_count, _), module=Residual(input_channels=input_channels, out_channels=input_channels, mid_channels=mid_channels, use_bn=use_bn))
            self.up_sampling = nn.Sequential()
            self.up_sampling.add_module(name='up_sample_{}_{}'.format(self.stack_index, self.block_count), module=nn.Upsample(scale_factor=2, mode='bilinear'))
            for _ in range(residual_each_block):
                self.up_sampling.add_module(name='residual_{}_{}_{}'.format(self.stack_index, self.block_count, _), module=Residual(input_channels=input_channels, out_channels=input_channels, mid_channels=mid_channels, use_bn=use_bn))
            self.sub_hg = HourGlassBlock(block_count=self.block_count - 1, residual_each_block=self.residual_each_block, input_channels=self.input_channels, mid_channels=self.mid_channels, use_bn=self.use_bn, stack_index=self.stack_index)
            self.trans = nn.Sequential()
            for _ in range(residual_each_block):
                self.trans.add_module(name='trans_{}_{}_{}'.format(self.stack_index, self.block_count, _), module=Residual(input_channels=input_channels, out_channels=input_channels, mid_channels=mid_channels, use_bn=use_bn))

    def forward(self, inputs):
        if self.block_count == 0:
            return self.process(inputs)
        else:
            down_sampled = self.down_sampling(inputs)
            transed = self.trans(down_sampled)
            sub_net_output = self.sub_hg(down_sampled)
            return self.up_sampling(transed + sub_net_output)


class HourGlass(nn.Module):

    def __init__(self, nStack, nBlockCount, nResidualEachBlock, nMidChannels, nChannels, nJointCount, bUseBn):
        super(HourGlass, self).__init__()
        self.nStack = nStack
        self.nBlockCount = nBlockCount
        self.nResidualEachBlock = nResidualEachBlock
        self.nChannels = nChannels
        self.nMidChannels = nMidChannels
        self.nJointCount = nJointCount
        self.bUseBn = bUseBn
        self.pre_process = nn.Sequential(nn.Conv2d(3, nChannels, kernel_size=3, padding=1), Residual(use_bn=bUseBn, input_channels=nChannels, out_channels=nChannels, mid_channels=nMidChannels), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=bUseBn, input_channels=nChannels, out_channels=nChannels, mid_channels=nMidChannels), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=bUseBn, input_channels=nChannels, out_channels=nChannels, mid_channels=nMidChannels))
        self.hg = nn.ModuleList()
        for _ in range(nStack):
            self.hg.append(HourGlassBlock(block_count=nBlockCount, residual_each_block=nResidualEachBlock, input_channels=nChannels, mid_channels=nMidChannels, use_bn=bUseBn, stack_index=_))
        self.blocks = nn.ModuleList()
        for _ in range(nStack - 1):
            self.blocks.append(nn.Sequential(Residual(use_bn=bUseBn, input_channels=nChannels, out_channels=nChannels, mid_channels=nMidChannels), Residual(use_bn=bUseBn, input_channels=nChannels, out_channels=nChannels, mid_channels=nMidChannels)))
        self.intermediate_supervision = nn.ModuleList()
        for _ in range(nStack):
            self.intermediate_supervision.append(nn.Conv2d(nChannels, nJointCount, kernel_size=1, stride=1))
        self.normal_feature_channel = nn.ModuleList()
        for _ in range(nStack - 1):
            self.normal_feature_channel.append(Residual(use_bn=bUseBn, input_channels=nJointCount, out_channels=nChannels, mid_channels=nMidChannels))

    def forward(self, inputs):
        o = []
        x = self.pre_process(inputs)
        for _ in range(self.nStack):
            o1 = self.hg[_](x)
            o2 = self.intermediate_supervision[_](o1)
            o.append(o2.view(-1, 4096))
            if _ == self.nStack - 1:
                break
            o2 = self.normal_feature_channel[_](o2)
            o1 = self.blocks[_](o1)
            x = o1 + o2 + x
        return o


class PRNetEncoder(nn.Module):

    def __init__(self):
        super(PRNetEncoder, self).__init__()
        self.conv_blocks = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1), nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), Residual(use_bn=True, input_channels=16, out_channels=32, mid_channels=16, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=32, out_channels=32, mid_channels=16, stride=1, padding=1), Residual(use_bn=True, input_channels=32, out_channels=32, mid_channels=16, stride=1, padding=1), Residual(use_bn=True, input_channels=32, out_channels=64, mid_channels=32, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=64, out_channels=64, mid_channels=32, stride=1, padding=1), Residual(use_bn=True, input_channels=64, out_channels=64, mid_channels=32, stride=1, padding=1), Residual(use_bn=True, input_channels=64, out_channels=128, mid_channels=64, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=128, out_channels=128, mid_channels=64, stride=1, padding=1), Residual(use_bn=True, input_channels=128, out_channels=128, mid_channels=64, stride=1, padding=1), Residual(use_bn=True, input_channels=128, out_channels=256, mid_channels=128, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=256, out_channels=256, mid_channels=128, stride=1, padding=1), Residual(use_bn=True, input_channels=256, out_channels=256, mid_channels=128, stride=1, padding=1), Residual(use_bn=True, input_channels=256, out_channels=512, mid_channels=256, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=512, out_channels=512, mid_channels=256, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=512, out_channels=512, mid_channels=256, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2), Residual(use_bn=True, input_channels=512, out_channels=512, mid_channels=256, stride=1, padding=1))

    def forward(self, inputs):
        return self.conv_blocks(inputs).view(-1, 2048)


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
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


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


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float())
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1))], dim=1)
        return torch.cat([R_homo, t_homo], 2)
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)
    results = torch.stack(results, dim=1)
    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1))], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone
    return new_J, A


def quat2mat(quat):
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


def batch_rodrigues(theta):
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-08, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


class SMPL(nn.Module):

    def __init__(self, model_path, joint_type='cocoplus', obj_saveable=False):
        super(SMPL, self).__init__()
        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)
        self.model_path = model_path
        self.joint_type = joint_type
        with open(model_path, 'r') as reader:
            model = json.load(reader)
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None
        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]
        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())
        np_J_regressor = np.array(model['J_regressor'], dtype=np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())
        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
        np_joint_regressor = np.array(model['cocoplus_regressor'], dtype=np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())
        np_weights = np.array(model['weights'], dtype=np.float)
        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]
        batch_size = max(args.batch_size + args.batch_3d_size, args.eval_batch_size)
        np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)
        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces:
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, beta, theta, get_skin=False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)
        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = Rs[:, 1:, :, :].sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)
        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        verts = v_homo[:, :, :3, 0]
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)
        if get_skin:
            return verts, joints, Rs
        else:
            return joints


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        return out


class ThetaRegressor(LinearModel):

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations):
        super(ThetaRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        batch_size = max(args.batch_size + args.batch_3d_size, args.eval_batch_size)
        mean_theta = np.tile(util.load_mean_theta(), batch_size).reshape((batch_size, -1))
        self.register_buffer('mean_theta', torch.from_numpy(mean_theta).float())
    """
        param:
            inputs: is the output of encoder, which has 2048 features
        
        return:
            a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations X N X 85(or other theta count)
    """

    def forward(self, inputs):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)
        return thetas


def _create_hourglass_net():
    return HourGlass(nStack=2, nBlockCount=4, nResidualEachBlock=1, nMidChannels=128, nChannels=256, nJointCount=1, bUseBn=True)


model_urls = {'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth', 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth', 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth', 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'}


def densenet121(pretrained=False, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    if pretrained:
        pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    if pretrained:
        pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if pretrained:
        pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if pretrained:
        pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def load_denseNet(net_type):
    if net_type == 'densenet121':
        return densenet121(pretrained=True)
    elif net_type == 'densenet169':
        return densenet169(pretrained=True)
    elif net_type == 'densenet201':
        return densenet201(pretrained=True)
    elif net_type == 'densenet161':
        return densenet161(pretrained=True)
    else:
        msg = 'invalid denset net type'
        sys.exit(msg)


class HMRNetBase(nn.Module):

    def __init__(self):
        super(HMRNetBase, self).__init__()
        self._read_configs()
        None
        self._create_sub_modules()

    def _read_configs(self):

        def _check_config():
            encoder_name = args.encoder_network
            enable_inter_supervions = args.enable_inter_supervision
            feature_count = args.feature_count
            if encoder_name == 'hourglass':
                assert args.crop_size == 256
            elif encoder_name == 'resnet50':
                assert args.crop_size == 224
                assert not enable_inter_supervions
            elif encoder_name.startswith('densenet'):
                assert args.crop_size == 224
                assert not enable_inter_supervions
            else:
                msg = 'invalid encoder network, only {} is allowd, got {}'.format(args.allowed_encoder_net, encoder_name)
                sys.exit(msg)
            assert config.encoder_feature_count[encoder_name] == feature_count
        _check_config()
        self.encoder_name = args.encoder_network
        self.beta_count = args.beta_count
        self.smpl_model = args.smpl_model
        self.smpl_mean_theta_path = args.smpl_mean_theta_path
        self.total_theta_count = args.total_theta_count
        self.joint_count = args.joint_count
        self.feature_count = args.feature_count

    def _create_sub_modules(self):
        """
            ddd smpl model, SMPL can create a mesh from beta & theta
        """
        self.smpl = SMPL(self.smpl_model, obj_saveable=True)
        """
            only resnet50 and hourglass is allowd currently, maybe other encoder will be allowd later.
        """
        if self.encoder_name == 'resnet50':
            None
            self.encoder = Resnet.load_Res50Model()
        elif self.encoder_name == 'hourglass':
            None
            self.encoder = _create_hourglass_net()
        elif self.encoder_name.startswith('densenet'):
            None
            self.encoder = load_denseNet(self.encoder_name)
        else:
            assert 0
        """
            regressor can predict betas(include beta and theta which needed by SMPL) from coder extracted from encoder in a iteratirve way
        """
        fc_layers = [self.feature_count + self.total_theta_count, 1024, 1024, 85]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations)
        self.iterations = iterations
        None

    def forward(self, inputs):
        if self.encoder_name == 'resnet50':
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name.startswith('densenet'):
            feature = self.encoder(inputs)
            thetas = self.regressor(feature)
            detail_info = []
            for theta in thetas:
                detail_info.append(self._calc_detail_info(theta))
            return detail_info
        elif self.encoder_name == 'hourglass':
            if args.enable_inter_supervision:
                features = self.encoder(inputs)
                detail_info = []
                for feature in features:
                    thetas = self.regressor(feature)
                    detail_info.append(self._calc_detail_info(thetas[-1]))
                return detail_info
            else:
                features = self.encoder(inputs)
                thetas = self.regressor(features[-1])
                detail_info = []
                for theta in thetas:
                    detail_info.append(self._calc_detail_info(theta))
                return detail_info
        else:
            assert 0
    """
        purpose:
            calc verts, joint2d, joint3d, Rotation matrix

        inputs:
            theta: N X (3 + 72 + 10)

        return:
            thetas, verts, j2d, j3d, Rs
    """

    def _calc_detail_info(self, theta):
        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:75].contiguous()
        shape = theta[:, 75:].contiguous()
        verts, j3d, Rs = self.smpl(beta=shape, theta=pose, get_skin=True)
        j2d = util.batch_orth_proj(j3d, cam)
        return theta, verts, j2d, j3d, Rs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (HourGlass,
     lambda: ([], {'nStack': 4, 'nBlockCount': 4, 'nResidualEachBlock': 4, 'nMidChannels': 4, 'nChannels': 4, 'nJointCount': 4, 'bUseBn': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (HourGlassBlock,
     lambda: ([], {'block_count': 4, 'residual_each_block': 4, 'input_channels': 4, 'mid_channels': 4, 'use_bn': 4, 'stack_index': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (PRNetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (Residual,
     lambda: ([], {'use_bn': 4, 'input_channels': 4, 'out_channels': 4, 'mid_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MandyMo_pytorch_HMR(_paritybench_base):
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

