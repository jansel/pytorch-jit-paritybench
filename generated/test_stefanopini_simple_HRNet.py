import sys
_module = sys.modules[__name__]
del sys
SimpleHRNet = _module
COCO = _module
HumanPoseEstimation = _module
LiveCamera = _module
datasets = _module
losses = _module
loss = _module
misc = _module
checkpoint = _module
nms = _module
setup_linux = _module
utils = _module
visualization = _module
models = _module
YOLOv3 = _module
hrnet = _module
modules = _module
poseresnet = _module
train_coco = _module
Test = _module
testing = _module
Train = _module
training = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


from torchvision.transforms import transforms


import torch.nn as nn


from torch import nn


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight=True):
        """
        MSE loss between output and GT body joints

        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                if target_weight is None:
                    raise NameError
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, (idx)]), heatmap_gt.mul(target_weight[:, (idx)]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):

    def __init__(self, use_target_weight=True, topk=8):
        """
        MSE loss between output and GT body joints

        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def forward(self, output, target, target_weight):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(heatmap_pred.mul(target_weight[:, (idx)]), heatmap_gt.mul(target_weight[:, (idx)])))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))
        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)
        return self.ohkm(loss, self.topk)


class StageModule(nn.Module):

    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * 2 ** i
            branch = nn.Sequential(BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum))
            self.branches.append(branch)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** i, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** j, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** j, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))
                    ops.append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** i, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused


class HRNet(nn.Module):

    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample=downsample), Bottleneck(256, 64), Bottleneck(256, 64), Bottleneck(256, 64))
        self.transition1 = nn.ModuleList([nn.Sequential(nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)), nn.Sequential(nn.Sequential(nn.Conv2d(256, c * 2 ** 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 1, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage2 = nn.Sequential(StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum))
        self.transition2 = nn.ModuleList([nn.Sequential(), nn.Sequential(), nn.Sequential(nn.Sequential(nn.Conv2d(c * 2 ** 1, c * 2 ** 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 2, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage3 = nn.Sequential(StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum))
        self.transition3 = nn.ModuleList([nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(nn.Sequential(nn.Conv2d(c * 2 ** 2, c * 2 ** 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 3, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage4 = nn.Sequential(StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum))
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [self.transition2[0](x[0]), self.transition2[1](x[1]), self.transition2[2](x[-1])]
        x = self.stage3(x)
        x = [self.transition3[0](x[0]), self.transition3[1](x[1]), self.transition3[2](x[2]), self.transition3[3](x[-1])]
        x = self.stage4(x)
        x = self.final_layer(x[0])
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
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


resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2]), (34): (BasicBlock, [3, 4, 6, 3]), (50): (Bottleneck, [3, 4, 6, 3]), (101): (Bottleneck, [3, 4, 23, 3]), (152): (Bottleneck, [3, 8, 36, 3])}


class PoseResNet(nn.Module):

    def __init__(self, resnet_size=50, nof_joints=17, bn_momentum=0.1):
        super(PoseResNet, self).__init__()
        assert resnet_size in resnet_spec.keys()
        block, layers = resnet_spec[resnet_size]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_momentum=bn_momentum)
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4], bn_momentum=bn_momentum)
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=nof_joints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum))
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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, bn_momentum=0.1):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=bn_momentum))
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HRNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (PoseResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_stefanopini_simple_HRNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

