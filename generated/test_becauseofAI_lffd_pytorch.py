import sys
_module = sys.modules[__name__]
del sys
ChasingTrainFramework_GeneralOneClassDetection = _module
data_iterator_base = _module
data_batch = _module
data_provider_base = _module
base_data_adapter = _module
base_provider = _module
pickle_provider = _module
text_list_adapter = _module
image_augmentation = _module
augmentor = _module
inference_speed_eval = _module
inference_speed_eval_with_mxnet_cudnn = _module
inference_speed_eval_with_tensorrt_cudnn = _module
logging_GOCD = _module
loss_layer_farm = _module
cross_entropy_with_focal_loss_for_one_class_detection = _module
cross_entropy_with_hnm_for_one_class_detection = _module
loss = _module
mean_squared_error_with_hnm_for_one_class_detection = _module
mean_squared_error_with_ohem_for_one_class_detection = _module
solver_GOCD = _module
train_GOCD = _module
evaluation_on_fddb = _module
evaluation_on_widerface = _module
predict = _module
config_farm = _module
configuration_10_320_20L_5scales_v2 = _module
configuration_10_560_25L_8scales_v1 = _module
data_iterator_farm = _module
multithread_dataiter_for_cross_entropy_v1 = _module
multithread_dataiter_for_cross_entropy_v2 = _module
data_provider_farm = _module
demo = _module
predict_tensorrt = _module
to_onnx = _module
metric_farm = _module
metric_default = _module
net_farm = _module
naivenet = _module
evaluation_on_brainwash = _module
configuration_10_160_17L_4scales_v1 = _module
reformat_brainwash = _module
symbol_farm = _module
symbol_10_160_17L_4scales_v1 = _module
evaluation_on_CCPD = _module
configuration_64_512_16L_3scales_v1 = _module
reformat_CCPD = _module
symbol_64_512_16L_3scales_v1 = _module
configuration_30_320_20L_4scales_v1 = _module
reformat_caltech = _module
symbol_30_320_20L_4scales_v1 = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn as nn


import torch.nn.functional as F


import logging


import time


import math


import torch.optim as optim


import random


import queue


import numpy


class cross_entropy_with_hnm_for_one_class_detection(nn.Module):

    def __init__(self, hnm_ratio, num_output_scales):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)
        self.num_output_scales = num_output_scales

    def forward(self, outputs, targets):
        loss_cls = 0
        loss_reg = 0
        loss_branch = []
        for i in range(self.num_output_scales):
            pred_score = outputs[i * 2]
            pred_bbox = outputs[i * 2 + 1]
            gt_mask = targets[i * 2]
            gt_label = targets[i * 2 + 1]
            pred_score_softmax = torch.softmax(pred_score, dim=1)
            loss_mask = torch.ones(pred_score_softmax.shape)
            if self.hnm_ratio > 0:
                pos_flag = gt_label[:, (0), :, :] > 0.5
                pos_num = torch.sum(pos_flag)
                if pos_num > 0:
                    neg_flag = gt_label[:, (1), :, :] > 0.5
                    neg_num = torch.sum(neg_flag)
                    neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                    neg_prob = torch.where(neg_flag, pred_score_softmax[:, (1), :, :], torch.zeros_like(pred_score_softmax[:, (1), :, :]))
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected - 1]
                    neg_grad_flag = neg_prob <= prob_threshold
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
                else:
                    neg_choice_ratio = 0.1
                    neg_num_selected = int(pred_score_softmax[:, (1), :, :].numel() * neg_choice_ratio)
                    neg_prob = pred_score_softmax[:, (1), :, :]
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected - 1]
                    neg_grad_flag = neg_prob <= prob_threshold
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
            pred_score_softmax_masked = pred_score_softmax[loss_mask]
            pred_score_log = torch.log(pred_score_softmax_masked)
            score_cross_entropy = -gt_label[:, :2, :, :][loss_mask] * pred_score_log
            loss_score = torch.sum(score_cross_entropy) / score_cross_entropy.numel()
            mask_bbox = gt_mask[:, 2:6, :, :]
            if torch.sum(mask_bbox) == 0:
                loss_bbox = torch.zeros_like(loss_score)
            else:
                predict_bbox = pred_bbox * mask_bbox
                label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
                loss_bbox = F.mse_loss(predict_bbox, label_bbox, reduction='sum') / torch.sum(mask_bbox)
            loss_cls += loss_score
            loss_reg += loss_bbox
            loss_branch.append(loss_score)
            loss_branch.append(loss_bbox)
        loss = loss_cls + loss_reg
        return loss, loss_branch


class cross_entropy_with_hnm_for_one_class_detection2(nn.Module):

    def __init__(self, hnm_ratio, num_output_scales):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)
        self.num_output_scales = num_output_scales

    def forward(self, outputs, targets):
        loss_branch_list = []
        for i in range(self.num_output_scales):
            pred_score = outputs[i * 2]
            pred_bbox = outputs[i * 2 + 1]
            gt_mask = targets[i * 2]
            gt_label = targets[i * 2 + 1]
            pred_score_softmax = torch.softmax(pred_score, dim=1)
            loss_mask = torch.ones(pred_score_softmax.shape)
            if self.hnm_ratio > 0:
                pos_flag = gt_label[:, (0), :, :] > 0.5
                pos_num = torch.sum(pos_flag)
                if pos_num > 0:
                    neg_flag = gt_label[:, (1), :, :] > 0.5
                    neg_num = torch.sum(neg_flag)
                    neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                    neg_prob = torch.where(neg_flag, pred_score_softmax[:, (1), :, :], torch.zeros_like(pred_score_softmax[:, (1), :, :]))
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected - 1]
                    neg_grad_flag = neg_prob <= prob_threshold
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
                else:
                    neg_choice_ratio = 0.1
                    neg_num_selected = int(pred_score_softmax[:, (1), :, :].numel() * neg_choice_ratio)
                    neg_prob = pred_score_softmax[:, (1), :, :]
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected - 1]
                    neg_grad_flag = neg_prob <= prob_threshold
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
            pred_score_softmax_masked = pred_score_softmax[loss_mask]
            pred_score_log = torch.log(pred_score_softmax_masked)
            score_cross_entropy = -gt_label[:, :2, :, :][loss_mask] * pred_score_log
            loss_score = torch.sum(score_cross_entropy) / score_cross_entropy.numel()
            mask_bbox = gt_mask[:, 2:6, :, :]
            if torch.sum(mask_bbox) == 0:
                loss_bbox = torch.zeros_like(loss_score)
            else:
                predict_bbox = pred_bbox * mask_bbox
                label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
                loss_bbox = F.mse_loss(predict_bbox, label_bbox, reduction='mean')
            loss_branch = loss_score + loss_bbox
            loss_branch_list.append(loss_branch)
        return loss_branch_list


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)


class Resv1Block(nn.Module):
    """ResNet v1 block without bn"""

    def __init__(self, inplanes, planes, stride=1):
        super(Resv1Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out


class Resv2Block(nn.Module):
    """ResNet v2 block without bn"""

    def __init__(self, inplanes, planes, stride=1, is_branch=False):
        super(Resv2Block, self).__init__()
        self.is_branch = is_branch
        self.relu1 = nn.ReLU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)

    def forward(self, x):
        out_branch = self.relu1(x)
        out = self.conv1(out_branch)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        if self.is_branch:
            return out, out_branch
        else:
            return out


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, 1, bias=True)


class BranchNet(nn.Module):
    """
    The branch of NaiveNet is the network output and 
    only consists of conv 1×1 and ReLU.
    """

    def __init__(self, inplanes, planes):
        super(BranchNet, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2_score = conv1x1(planes, planes)
        self.conv3_score = conv1x1(planes, 2)
        self.conv2_bbox = conv1x1(planes, planes)
        self.conv3_bbox = conv1x1(planes, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out_score = self.conv2_score(out)
        out_score = self.relu(out_score)
        out_score = self.conv3_score(out_score)
        out_bbox = self.conv2_bbox(out)
        out_bbox = self.relu(out_bbox)
        out_bbox = self.conv3_bbox(out_bbox)
        return out_score, out_bbox


num_filters_list = [32, 64, 128, 256]


class NaiveNet(nn.Module):
    """NaiveNet for Fast Single Class Object Detection. 
    The entire backbone and branches only consists of conv 3×3, 
    conv 1×1, ReLU and residual connection.
    """

    def __init__(self, arch, block, layers):
        super(NaiveNet, self).__init__()
        self.arch = arch
        self.block = block
        if self.arch == 'naivenet25':
            if self.block == Resv2Block:
                self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
                self.relu1 = nn.ReLU()
                self.stage1_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0] - 1, stride=2)
                self.stage1_2_branch1 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=True))
                self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage1_3_branch2 = nn.Sequential(nn.ReLU())
                self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_1 = nn.Sequential(conv3x3(num_filters_list[1], num_filters_list[1], stride=2, padding=0), Resv2Block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=False))
                self.stage2_2_branch3 = nn.Sequential(Resv2Block(num_filters_list[1], num_filters_list[1], stride=1, is_branch=True))
                self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_3_branch4 = nn.Sequential(nn.ReLU())
                self.branch4 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage3_1 = nn.Sequential(conv3x3(num_filters_list[1], num_filters_list[2], stride=2, padding=0), Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=False))
                self.stage3_2_branch5 = nn.Sequential(nn.ReLU())
                self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_1 = nn.Sequential(conv3x3(num_filters_list[2], num_filters_list[2], stride=2, padding=0), Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=False))
                self.stage4_2_branch6 = nn.Sequential(Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=True))
                self.branch6 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_3_branch7 = nn.Sequential(Resv2Block(num_filters_list[2], num_filters_list[2], stride=1, is_branch=True))
                self.branch7 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_4_branch8 = nn.Sequential(nn.ReLU())
                self.branch8 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            elif self.block == Resv1Block:
                self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
                self.relu1 = nn.ReLU()
                self.stage1_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0] - 1, stride=2)
                self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage1_2 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1))
                self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[1] - 1, stride=2)
                self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage2_2 = nn.Sequential(self.block(num_filters_list[1], num_filters_list[1], stride=1))
                self.branch4 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
                self.stage3_1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[2], layers[2], stride=2)
                self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_1 = self._make_layer(self.arch, self.block, num_filters_list[2], num_filters_list[2], layers[3] - 2, stride=2)
                self.branch6 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_2 = nn.Sequential(self.block(num_filters_list[2], num_filters_list[2], stride=1))
                self.branch7 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
                self.stage4_3 = nn.Sequential(self.block(num_filters_list[2], num_filters_list[2], stride=1))
                self.branch8 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        elif self.arch == 'naivenet20':
            self.conv1 = conv3x3(3, num_filters_list[1], stride=2, padding=0)
            self.relu1 = nn.ReLU()
            self.layer1 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[0], stride=2)
            self.branch1 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer2 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[1], stride=2)
            self.branch2 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer3 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[1], layers[2], stride=2)
            self.branch3 = nn.Sequential(BranchNet(num_filters_list[1], num_filters_list[2]))
            self.layer4 = self._make_layer(self.arch, self.block, num_filters_list[1], num_filters_list[2], layers[3], stride=2)
            self.branch4 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
            self.layer5 = self._make_layer(self.arch, self.block, num_filters_list[2], num_filters_list[2], layers[4], stride=2)
            self.branch5 = nn.Sequential(BranchNet(num_filters_list[2], num_filters_list[2]))
        else:
            raise TypeError('Unsupported NaiveNet Version.')

    def _make_layer(self, arch, block, inplanes, planes, blocks, stride=2):
        layers = []
        if self.arch == 'naivenet25':
            if block == Resv2Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            elif block == Resv1Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                layers.append(nn.ReLU())
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        elif self.arch == 'naivenet20':
            if block == Resv2Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
                layers.append(nn.ReLU())
            elif block == Resv1Block:
                layers.append(conv3x3(inplanes, planes, stride=stride, padding=0))
                layers.append(nn.ReLU())
                for _ in range(blocks):
                    layers.append(block(planes, planes, stride=1))
            else:
                raise TypeError('Unsupported ResNet Block Version.')
        else:
            raise TypeError('Unsupported NaiveNet Version.')
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.arch == 'naivenet25':
            if self.block == Resv2Block:
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.stage1_1(x)
                x, b1 = self.stage1_2_branch1(x)
                score1, bbox1 = self.branch1(b1)
                x = b2 = self.stage1_3_branch2(x)
                score2, bbox2 = self.branch2(b2)
                x = self.stage2_1(x)
                x, b3 = self.stage2_2_branch3(x)
                score3, bbox3 = self.branch3(b3)
                x = b4 = self.stage2_3_branch4(x)
                score4, bbox4 = self.branch4(b4)
                x = self.stage3_1(x)
                x = b5 = self.stage3_2_branch5(x)
                score5, bbox5 = self.branch5(b5)
                x = self.stage4_1(x)
                x, b6 = self.stage4_2_branch6(x)
                score6, bbox6 = self.branch6(b6)
                x, b7 = self.stage4_3_branch7(x)
                score7, bbox7 = self.branch7(b7)
                x = b8 = self.stage4_4_branch8(x)
                score8, bbox8 = self.branch8(b8)
            if self.block == Resv1Block:
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.stage1_1(x)
                score1, bbox1 = self.branch1(x)
                x = self.stage1_2(x)
                score2, bbox2 = self.branch2(x)
                x = self.stage2_1(x)
                score3, bbox3 = self.branch3(x)
                x = self.stage2_2(x)
                score4, bbox4 = self.branch4(x)
                x = self.stage3_1(x)
                score5, bbox5 = self.branch5(x)
                x = self.stage4_1(x)
                score6, bbox6 = self.branch6(x)
                x = self.stage4_2(x)
                score7, bbox7 = self.branch7(x)
                x = self.stage4_3(x)
                score8, bbox8 = self.branch8(x)
                outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5, score6, bbox6, score7, bbox7, score8, bbox8]
            return outs
        if self.arch == 'naivenet20':
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.layer1(x)
            score1, bbox1 = self.branch1(x)
            x = self.layer2(x)
            score2, bbox2 = self.branch2(x)
            x = self.layer3(x)
            score3, bbox3 = self.branch3(x)
            x = self.layer4(x)
            score4, bbox4 = self.branch4(x)
            x = self.layer5(x)
            score5, bbox5 = self.branch5(x)
            outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5]
            return outs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BranchNet,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resv1Block,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Resv2Block,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_becauseofAI_lffd_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

