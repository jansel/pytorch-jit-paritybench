import sys
_module = sys.modules[__name__]
del sys
data_processing = _module
psdb = _module
sampler = _module
backbone = _module
head = _module
network = _module
labeled_matching_layer = _module
unlabeled_matching_layer = _module
anchor_target_layer = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer = _module
rpn_layer = _module
boxes = _module
config = _module
evaluate = _module
utils = _module
_init_paths = _module
demo = _module
test_net = _module
train_net = _module

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


import logging


from scipy.io import loadmat


from torch.utils.data import Dataset


from torch.utils.data.sampler import Sampler


import torch.nn as nn


import torch.nn.functional as F


from torchvision.ops import RoIPool


from torchvision.ops import nms


from torch.autograd import Function


import matplotlib.pyplot as plt


import random


import time


import torch.optim as optim


from torch.utils.data import DataLoader


class Backbone(nn.Module):
    """
    Extract the basic features of the images (conv1 --> conv4_3).
    The name of each module is exactly the same as in the original caffe code.
    """

    def __init__(self):
        super(Backbone, self).__init__()
        self.SpatialConvolution_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.BN_1 = nn.BatchNorm2d(64)
        self.ReLU_2 = nn.ReLU(inplace=True)
        self.Pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.SpatialConvolution_4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.BN_5 = nn.BatchNorm2d(64)
        self.ReLU_6 = nn.ReLU(inplace=True)
        self.SpatialConvolution_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_8 = nn.BatchNorm2d(64)
        self.ReLU_9 = nn.ReLU(inplace=True)
        self.SpatialConvolution_10 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_11 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_13 = nn.BatchNorm2d(256)
        self.ReLU_14 = nn.ReLU(inplace=True)
        self.SpatialConvolution_15 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.BN_16 = nn.BatchNorm2d(64)
        self.ReLU_17 = nn.ReLU(inplace=True)
        self.SpatialConvolution_18 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_19 = nn.BatchNorm2d(64)
        self.ReLU_20 = nn.ReLU(inplace=True)
        self.SpatialConvolution_21 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_23 = nn.BatchNorm2d(256)
        self.ReLU_24 = nn.ReLU(inplace=True)
        self.SpatialConvolution_25 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.BN_26 = nn.BatchNorm2d(64)
        self.ReLU_27 = nn.ReLU(inplace=True)
        self.SpatialConvolution_28 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.BN_29 = nn.BatchNorm2d(64)
        self.ReLU_30 = nn.ReLU(inplace=True)
        self.SpatialConvolution_31 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.BN_33 = nn.BatchNorm2d(256)
        self.ReLU_34 = nn.ReLU(inplace=True)
        self.SpatialConvolution_35 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.BN_36 = nn.BatchNorm2d(128)
        self.ReLU_37 = nn.ReLU(inplace=True)
        self.SpatialConvolution_38 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.BN_39 = nn.BatchNorm2d(128)
        self.ReLU_40 = nn.ReLU(inplace=True)
        self.SpatialConvolution_41 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_42 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        self.BN_44 = nn.BatchNorm2d(512)
        self.ReLU_45 = nn.ReLU(inplace=True)
        self.SpatialConvolution_46 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_47 = nn.BatchNorm2d(128)
        self.ReLU_48 = nn.ReLU(inplace=True)
        self.SpatialConvolution_49 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_50 = nn.BatchNorm2d(128)
        self.ReLU_51 = nn.ReLU(inplace=True)
        self.SpatialConvolution_52 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_54 = nn.BatchNorm2d(512)
        self.ReLU_55 = nn.ReLU(inplace=True)
        self.SpatialConvolution_56 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_57 = nn.BatchNorm2d(128)
        self.ReLU_58 = nn.ReLU(inplace=True)
        self.SpatialConvolution_59 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_60 = nn.BatchNorm2d(128)
        self.ReLU_61 = nn.ReLU(inplace=True)
        self.SpatialConvolution_62 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_64 = nn.BatchNorm2d(512)
        self.ReLU_65 = nn.ReLU(inplace=True)
        self.SpatialConvolution_66 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.BN_67 = nn.BatchNorm2d(128)
        self.ReLU_68 = nn.ReLU(inplace=True)
        self.SpatialConvolution_69 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN_70 = nn.BatchNorm2d(128)
        self.ReLU_71 = nn.ReLU(inplace=True)
        self.SpatialConvolution_72 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.BN_74 = nn.BatchNorm2d(512)
        self.ReLU_75 = nn.ReLU(inplace=True)
        self.SpatialConvolution_76 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.BN_77 = nn.BatchNorm2d(256)
        self.ReLU_78 = nn.ReLU(inplace=True)
        self.SpatialConvolution_79 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.BN_80 = nn.BatchNorm2d(256)
        self.ReLU_81 = nn.ReLU(inplace=True)
        self.SpatialConvolution_82 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_83 = nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0)
        self.BN_85 = nn.BatchNorm2d(1024)
        self.ReLU_86 = nn.ReLU(inplace=True)
        self.SpatialConvolution_87 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_88 = nn.BatchNorm2d(256)
        self.ReLU_89 = nn.ReLU(inplace=True)
        self.SpatialConvolution_90 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_91 = nn.BatchNorm2d(256)
        self.ReLU_92 = nn.ReLU(inplace=True)
        self.SpatialConvolution_93 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_95 = nn.BatchNorm2d(1024)
        self.ReLU_96 = nn.ReLU(inplace=True)
        self.SpatialConvolution_97 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_98 = nn.BatchNorm2d(256)
        self.ReLU_99 = nn.ReLU(inplace=True)
        self.SpatialConvolution_100 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_101 = nn.BatchNorm2d(256)
        self.ReLU_102 = nn.ReLU(inplace=True)
        self.SpatialConvolution_103 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_105 = nn.BatchNorm2d(1024)
        self.ReLU_106 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.SpatialConvolution_0(x)
        x = self.BN_1(x)
        x = self.ReLU_2(x)
        x = self.Pooling_3(x)
        residual = self.SpatialConvolution_11(x)
        x = self.SpatialConvolution_4(x)
        x = self.BN_5(x)
        x = self.ReLU_6(x)
        x = self.SpatialConvolution_7(x)
        x = self.BN_8(x)
        x = self.ReLU_9(x)
        x = self.SpatialConvolution_10(x)
        x += residual
        x = self.BN_13(x)
        x = self.ReLU_14(x)
        residual = x
        x = self.SpatialConvolution_15(x)
        x = self.BN_16(x)
        x = self.ReLU_17(x)
        x = self.SpatialConvolution_18(x)
        x = self.BN_19(x)
        x = self.ReLU_20(x)
        x = self.SpatialConvolution_21(x)
        x += residual
        x = self.BN_23(x)
        x = self.ReLU_24(x)
        residual = x
        x = self.SpatialConvolution_25(x)
        x = self.BN_26(x)
        x = self.ReLU_27(x)
        x = self.SpatialConvolution_28(x)
        x = self.BN_29(x)
        x = self.ReLU_30(x)
        x = self.SpatialConvolution_31(x)
        x += residual
        x = self.BN_33(x)
        x = self.ReLU_34(x)
        residual = self.SpatialConvolution_42(x)
        x = self.SpatialConvolution_35(x)
        x = self.BN_36(x)
        x = self.ReLU_37(x)
        x = self.SpatialConvolution_38(x)
        x = self.BN_39(x)
        x = self.ReLU_40(x)
        x = self.SpatialConvolution_41(x)
        x += residual
        x = self.BN_44(x)
        x = self.ReLU_45(x)
        residual = x
        x = self.SpatialConvolution_46(x)
        x = self.BN_47(x)
        x = self.ReLU_48(x)
        x = self.SpatialConvolution_49(x)
        x = self.BN_50(x)
        x = self.ReLU_51(x)
        x = self.SpatialConvolution_52(x)
        x += residual
        x = self.BN_54(x)
        x = self.ReLU_55(x)
        residual = x
        x = self.SpatialConvolution_56(x)
        x = self.BN_57(x)
        x = self.ReLU_58(x)
        x = self.SpatialConvolution_59(x)
        x = self.BN_60(x)
        x = self.ReLU_61(x)
        x = self.SpatialConvolution_62(x)
        x += residual
        x = self.BN_64(x)
        x = self.ReLU_65(x)
        residual = x
        x = self.SpatialConvolution_66(x)
        x = self.BN_67(x)
        x = self.ReLU_68(x)
        x = self.SpatialConvolution_69(x)
        x = self.BN_70(x)
        x = self.ReLU_71(x)
        x = self.SpatialConvolution_72(x)
        x += residual
        x = self.BN_74(x)
        x = self.ReLU_75(x)
        residual = self.SpatialConvolution_83(x)
        x = self.SpatialConvolution_76(x)
        x = self.BN_77(x)
        x = self.ReLU_78(x)
        x = self.SpatialConvolution_79(x)
        x = self.BN_80(x)
        x = self.ReLU_81(x)
        x = self.SpatialConvolution_82(x)
        x += residual
        x = self.BN_85(x)
        x = self.ReLU_86(x)
        residual = x
        x = self.SpatialConvolution_87(x)
        x = self.BN_88(x)
        x = self.ReLU_89(x)
        x = self.SpatialConvolution_90(x)
        x = self.BN_91(x)
        x = self.ReLU_92(x)
        x = self.SpatialConvolution_93(x)
        x += residual
        x = self.BN_95(x)
        x = self.ReLU_96(x)
        residual = x
        x = self.SpatialConvolution_97(x)
        x = self.BN_98(x)
        x = self.ReLU_99(x)
        x = self.SpatialConvolution_100(x)
        x = self.BN_101(x)
        x = self.ReLU_102(x)
        x = self.SpatialConvolution_103(x)
        x += residual
        x = self.BN_105(x)
        x = self.ReLU_106(x)
        return x


class Head(nn.Module):
    """
    Extract the features of region proposals (conv4_4 --> conv5).
    The name of each module is exactly the same as in the original caffe code.
    """

    def __init__(self):
        super(Head, self).__init__()
        self.SpatialConvolution_107 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_108 = nn.BatchNorm2d(256)
        self.ReLU_109 = nn.ReLU(inplace=True)
        self.SpatialConvolution_110 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_111 = nn.BatchNorm2d(256)
        self.ReLU_112 = nn.ReLU(inplace=True)
        self.SpatialConvolution_113 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_115 = nn.BatchNorm2d(1024)
        self.ReLU_116 = nn.ReLU(inplace=True)
        self.SpatialConvolution_117 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_118 = nn.BatchNorm2d(256)
        self.ReLU_119 = nn.ReLU(inplace=True)
        self.SpatialConvolution_120 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_121 = nn.BatchNorm2d(256)
        self.ReLU_122 = nn.ReLU(inplace=True)
        self.SpatialConvolution_123 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_125 = nn.BatchNorm2d(1024)
        self.ReLU_126 = nn.ReLU(inplace=True)
        self.SpatialConvolution_127 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.BN_128 = nn.BatchNorm2d(256)
        self.ReLU_129 = nn.ReLU(inplace=True)
        self.SpatialConvolution_130 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.BN_131 = nn.BatchNorm2d(256)
        self.ReLU_132 = nn.ReLU(inplace=True)
        self.SpatialConvolution_133 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        self.BN_135 = nn.BatchNorm2d(1024)
        self.ReLU_136 = nn.ReLU(inplace=True)
        self.SpatialConvolution_137 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.BN_138 = nn.BatchNorm2d(512)
        self.ReLU_139 = nn.ReLU(inplace=True)
        self.SpatialConvolution_140 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.BN_141 = nn.BatchNorm2d(512)
        self.ReLU_142 = nn.ReLU(inplace=True)
        self.SpatialConvolution_143 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.SpatialConvolution_144 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2, padding=0)
        self.BN_146 = nn.BatchNorm2d(2048)
        self.ReLU_147 = nn.ReLU(inplace=True)
        self.SpatialConvolution_148 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.BN_149 = nn.BatchNorm2d(512)
        self.ReLU_150 = nn.ReLU(inplace=True)
        self.SpatialConvolution_151 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.BN_152 = nn.BatchNorm2d(512)
        self.ReLU_153 = nn.ReLU(inplace=True)
        self.SpatialConvolution_154 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.BN_156 = nn.BatchNorm2d(2048)
        self.ReLU_157 = nn.ReLU(inplace=True)
        self.SpatialConvolution_158 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.BN_159 = nn.BatchNorm2d(512)
        self.ReLU_160 = nn.ReLU(inplace=True)
        self.SpatialConvolution_161 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.BN_162 = nn.BatchNorm2d(512)
        self.ReLU_163 = nn.ReLU(inplace=True)
        self.SpatialConvolution_164 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.BN_166 = nn.BatchNorm2d(2048)
        self.ReLU_167 = nn.ReLU(inplace=True)
        self.Pooling_168 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        residual = x
        x = self.SpatialConvolution_107(x)
        x = self.BN_108(x)
        x = self.ReLU_109(x)
        x = self.SpatialConvolution_110(x)
        x = self.BN_111(x)
        x = self.ReLU_112(x)
        x = self.SpatialConvolution_113(x)
        x += residual
        x = self.BN_115(x)
        x = self.ReLU_116(x)
        residual = x
        x = self.SpatialConvolution_117(x)
        x = self.BN_118(x)
        x = self.ReLU_119(x)
        x = self.SpatialConvolution_120(x)
        x = self.BN_121(x)
        x = self.ReLU_122(x)
        x = self.SpatialConvolution_123(x)
        x += residual
        x = self.BN_125(x)
        x = self.ReLU_126(x)
        residual = x
        x = self.SpatialConvolution_127(x)
        x = self.BN_128(x)
        x = self.ReLU_129(x)
        x = self.SpatialConvolution_130(x)
        x = self.BN_131(x)
        x = self.ReLU_132(x)
        x = self.SpatialConvolution_133(x)
        x += residual
        x = self.BN_135(x)
        x = self.ReLU_136(x)
        residual = self.SpatialConvolution_144(x)
        x = self.SpatialConvolution_137(x)
        x = self.BN_138(x)
        x = self.ReLU_139(x)
        x = self.SpatialConvolution_140(x)
        x = self.BN_141(x)
        x = self.ReLU_142(x)
        x = self.SpatialConvolution_143(x)
        x += residual
        x = self.BN_146(x)
        x = self.ReLU_147(x)
        residual = x
        x = self.SpatialConvolution_148(x)
        x = self.BN_149(x)
        x = self.ReLU_150(x)
        x = self.SpatialConvolution_151(x)
        x = self.BN_152(x)
        x = self.ReLU_153(x)
        x = self.SpatialConvolution_154(x)
        x += residual
        x = self.BN_156(x)
        x = self.ReLU_157(x)
        residual = x
        x = self.SpatialConvolution_158(x)
        x = self.BN_159(x)
        x = self.ReLU_160(x)
        x = self.SpatialConvolution_161(x)
        x = self.BN_162(x)
        x = self.ReLU_163(x)
        x = self.SpatialConvolution_164(x)
        x += residual
        x = self.BN_166(x)
        x = self.ReLU_167(x)
        x = self.Pooling_168(x)
        return x


class LabeledMatching(Function):

    @staticmethod
    def forward(ctx, features, pid_labels, lookup_table, momentum=0.5):
        ctx.save_for_backward(features, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum
        scores = features.mm(lookup_table.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum
        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        for indx, label in enumerate(pid_labels):
            if label >= 0:
                lookup_table[label] = momentum * lookup_table[label] + (1 - momentum) * features[indx]
        return grad_feats, None, None, None


class LabeledMatchingLayer(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_persons=5532, feat_len=256):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(LabeledMatchingLayer, self).__init__()
        self.register_buffer('lookup_table', torch.zeros(num_persons, feat_len))

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        scores = LabeledMatching.apply(features, pid_labels, self.lookup_table)
        return scores


def bbox_overlaps(boxes1, boxes2):
    """
    Compute the overlaps between boxes1 and boxes2.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        overlaps (Tensor[N, M])
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    iw = (torch.min(boxes1[:, 2:3], boxes2[:, 2:3].t()) - torch.max(boxes1[:, 0:1], boxes2[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes1[:, 3:4], boxes2[:, 3:4].t()) - torch.max(boxes1[:, 1:2], boxes2[:, 1:2].t()) + 1).clamp(min=0)
    ua = area1.view(-1, 1) + area2.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return overlaps


def bbox_transform(boxes, gt_boxes):
    """
    Compute regression deltas of transforming boxes to gt_boxes.

    Args:
        boxes (Tensor[N, 4])
        gt_boxes (Tensor[N, 4])

    Returns:
        deltas (Tensor[N, 4])
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas


def torch_rand_choice(arr, size):
    """
    Generates a random sample from a given array, like numpy.random.choice.
    """
    idxs = torch.randperm(arr.size(0))[:size]
    return arr[idxs]


class ProposalTargetLayer(nn.Module):
    """
    Sample some proposals at the specified positive and negative ratio.

    And assign ground-truth targets (cls_labels, pid_labels, deltas, inside_weights,
    outside_weights) to these sampled proposals.

    BTW:
    pid_label = -1 -----> foreground proposals containing an unlabeled person.
    pid_label = -2 -----> background proposals.
    """

    def __init__(self, num_classes, bg_pid_label=-2):
        super(ProposalTargetLayer, self).__init__()
        self.num_classes = num_classes
        self.bg_pid_label = bg_pid_label

    def forward(self, proposals, gt_boxes):
        """
        Args:
            proposals (Tensor): Region proposals in (0, x1, y1, x2, y2) format coming from RPN.
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.

        Returns:
            proposals (Tensor[N, 5]): Sampled proposals.
            cls_labels (Tensor[N]): Ground-truth classification labels of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.
            deltas (Tensor[N, num_classes * 4]):  Ground-truth regression deltas of the proposals.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss.
        """
        assert torch.all(proposals[:, 0] == 0), 'Single batch only.'
        zeros = gt_boxes.new(gt_boxes.shape[0], 1).zero_()
        proposals = torch.cat((proposals, torch.cat((zeros, gt_boxes[:, :4]), dim=1)), dim=0)
        overlaps = bbox_overlaps(proposals[:, 1:5], gt_boxes[:, :4])
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        cls_labels = gt_boxes[argmax_overlaps, 4]
        pid_labels = gt_boxes[argmax_overlaps, 5]
        batch_size = cfg.TRAIN.BATCH_SIZE
        num_fg = round(cfg.TRAIN.FG_FRACTION * batch_size)
        fg_inds = torch.nonzero(max_overlaps >= cfg.TRAIN.FG_THRESH)[:, 0]
        num_fg = min(num_fg, fg_inds.numel())
        if fg_inds.numel() > 0:
            fg_inds = torch_rand_choice(fg_inds, num_fg)
        bg_inds = torch.nonzero((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[:, 0]
        num_bg = min(batch_size - num_fg, bg_inds.numel())
        if bg_inds.numel() > 0:
            bg_inds = torch_rand_choice(bg_inds, num_bg)
        keep = torch.cat((fg_inds, bg_inds))
        cls_labels = cls_labels[keep]
        pid_labels = pid_labels[keep]
        proposals = proposals[keep]
        cls_labels[num_fg:] = 0
        pid_labels[num_fg:] = self.bg_pid_label
        deltas, inside_weights, outside_weights = self.get_regression_targets(proposals[:, 1:5], gt_boxes[argmax_overlaps][keep, :4], cls_labels, self.num_classes)
        return proposals, cls_labels.long(), pid_labels.long(), deltas, inside_weights, outside_weights

    @staticmethod
    def get_regression_targets(proposals, gt_boxes, cls_labels, num_classes):
        """
        Args:
            proposals (Tensor): Sampled proposals in (x1, y1, x2, y2) format.
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2) format.
            cls_labels (Tensor): Classification labels of the proposals.
            num_classes (int): Number of classes.

        Returns:
            deltas ([N, num_classes * 4]): Proposal regression deltas.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss.
        """
        deltas_data = bbox_transform(proposals, gt_boxes)
        means = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        stds = gt_boxes.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        deltas_data = (deltas_data - means) / stds
        deltas = gt_boxes.new(proposals.size(0), 4 * num_classes).zero_()
        inside_weights = deltas.clone()
        outside_weights = deltas.clone()
        fg_inds = torch.nonzero(cls_labels > 0)[:, 0]
        for ind in fg_inds:
            cls = int(cls_labels[ind])
            start = 4 * cls
            end = start + 4
            deltas[ind, start:end] = deltas_data[ind]
            inside_weights[ind, start:end] = gt_boxes.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
            outside_weights[ind, start:end] = gt_boxes.new(4).fill_(1)
        return deltas, inside_weights, outside_weights


def mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around
    a center (x_ctr, y_ctr), output a set of anchors.
    """
    ws.unsqueeze_(1)
    hs.unsqueeze_(1)
    anchors = torch.cat((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)), dim=1)
    return anchors


def whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor.
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for a set of aspect ratios wrt an anchor.
    """
    w, h, x_ctr, y_ctr = whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for a set of scales wrt an anchor.
    """
    w, h, x_ctr, y_ctr = whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors():
    """
    Generate anchors by enumerating aspect ratios and
    scales wrt a reference (0, 0, 15, 15) window.
    """
    ratios = torch.Tensor(cfg.ANCHOR_RATIOS)
    scales = torch.Tensor(cfg.ANCHOR_SCALES)
    base_anchor = torch.Tensor([0, 0, 15, 15])
    ratio_anchors = ratio_enum(base_anchor, ratios)
    anchors = torch.cat([scale_enum(anchor, scales) for anchor in ratio_anchors], dim=0)
    return anchors


class AnchorTargetLayer(nn.Module):
    """
    Assign ground-truth targets (labels, deltas, inside_weights, outside_weights) to anchors.
    """

    def __init__(self):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, scores, gt_boxes, img_info):
        """
        Args:
            scores (Tensor[1, num_anchors * num_classes, H, W]): Classification scores.
            gt_boxes (Tensor[N, 6]): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format
            img_info (Tensor[3]): (height, width, scale)

        Returns:
            labels (Tensor): Ground-truth labels of the anchors.
            deltas (Tensor): Ground-truth regression deltas of the anchors.
            inside_weights, outside_weights (Tensor): Used to calculate smooth_l1_loss
        """
        assert scores.size(0) == 1, 'Single batch only.'
        height, width = scores.shape[-2:]
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack((shift_x.view(-1), shift_y.view(-1), shift_x.view(-1), shift_y.view(-1)), dim=1)
        shifts = shifts.type_as(gt_boxes)
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(gt_boxes)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)
        keep = torch.nonzero((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < img_info[1]) & (anchors[:, 3] < img_info[0]))[:, 0]
        anchors = anchors[keep]
        overlaps = bbox_overlaps(anchors, gt_boxes[:, :4])
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        gt_max_overlaps = overlaps.max(dim=0)[0]
        gt_argmax_overlaps = torch.nonzero(overlaps == gt_max_overlaps)[:, 0]
        labels = gt_boxes.new(len(keep)).fill_(-1)
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = torch.nonzero(labels == 1)[:, 0]
        if len(fg_inds) > num_fg:
            disable_inds = torch_rand_choice(fg_inds, len(fg_inds) - num_fg)
            labels[disable_inds] = -1
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum(labels == 1)
        bg_inds = torch.nonzero(labels == 0)[:, 0]
        if len(bg_inds) > num_bg:
            disable_inds = torch_rand_choice(bg_inds, len(bg_inds) - num_bg)
            labels[disable_inds] = -1
        deltas = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])
        inside_weights = gt_boxes.new(deltas.shape).zero_()
        inside_weights[labels == 1] = gt_boxes.new(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
        outside_weights = gt_boxes.new(deltas.shape).zero_()
        num_examples = torch.sum(labels >= 0)
        outside_weights[labels == 1] = gt_boxes.new(1, 4).fill_(1) / num_examples
        outside_weights[labels == 0] = gt_boxes.new(1, 4).fill_(1) / num_examples

        def map2origin(data, count=K * A, inds=keep, fill=0):
            """Map to original set."""
            shape = (count,) + data.shape[1:]
            origin = torch.empty(shape).fill_(fill).type_as(gt_boxes)
            origin[inds] = data
            return origin
        labels = map2origin(labels, fill=-1)
        deltas = map2origin(deltas)
        inside_weights = map2origin(inside_weights)
        outside_weights = map2origin(outside_weights)
        labels = labels.view(1, height, width, A).permute(0, 3, 1, 2)
        labels = labels.contiguous().view(1, 1, A * height, width)
        deltas = deltas.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        inside_weights = inside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        outside_weights = outside_weights.view(1, height, width, A * 4).permute(0, 3, 1, 2)
        return labels, deltas, inside_weights, outside_weights


def bbox_transform_inv(boxes, deltas):
    """
    Apply regression deltas on the boxes.

    Args:
        boxes (Tensor[N, 4])
        deltas (Tensor[N, 4])

    Returns:
        pred_boxes (Tensor[N, 4])
    """
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)
    pred_boxes = deltas.clone()
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
    return pred_boxes


def clip_boxes(boxes, img_shape):
    """
    Clip boxes to image boundaries.

    Args:
        boxes (Tensor[N, 4])
        img_shape (Tensor[2]): (height, width)

    Returns:
        boxes (Tensor[N, 4]): Clipped boxes.
    """
    boxes[:, 0::4].clamp_(0, img_shape[1] - 1)
    boxes[:, 1::4].clamp_(0, img_shape[0] - 1)
    boxes[:, 2::4].clamp_(0, img_shape[1] - 1)
    boxes[:, 3::4].clamp_(0, img_shape[0] - 1)
    return boxes


class ProposalLayer(nn.Module):
    """
    Outputs proposals by applying estimated regression deltas to a set of anchors.
    """

    def __init__(self):
        super(ProposalLayer, self).__init__()
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.size(0)

    def forward(self, probs, anchor_deltas, img_info):
        """
        Args:
            probs (Tensor): Classification probability of the anchors.
            anchor_deltas (Tensor): Anchor regression deltas.
            img_info (Tensor[3]): (height, width, scale)

        Returns:
            proposals (Tensor[N, 5]): Predicted region proposals in (0, x1, y1, x2, y2) format.
                                      0 means these proposals are from the first image in the batch.
        """
        assert probs.size(0) == 1, 'Single batch only.'
        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        probs = probs[:, self.num_anchors:, :, :]
        height, width = probs.shape[-2:]
        shift_x = torch.arange(0, width) * self.feat_stride
        shift_y = torch.arange(0, height) * self.feat_stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x, shift_y = shift_x.contiguous(), shift_y.contiguous()
        shifts = torch.stack((shift_x.view(-1), shift_y.view(-1), shift_x.view(-1), shift_y.view(-1)), dim=1)
        shifts = shifts.type_as(probs)
        A = self.num_anchors
        K = shifts.size(0)
        self.anchors = self.anchors.type_as(probs)
        anchors = self.anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2)
        anchors = anchors.view(K * A, 4)
        anchor_deltas = anchor_deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        anchor_deltas[:, 2:].clamp_(-10, 10)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        proposals = bbox_transform_inv(anchors, anchor_deltas)
        proposals = clip_boxes(proposals, img_info[:2])
        widths = proposals[:, 2] - proposals[:, 0] + 1
        heights = proposals[:, 3] - proposals[:, 1] + 1
        min_size = min_size * img_info[2]
        keep = torch.nonzero((widths >= min_size) & (heights >= min_size))[:, 0]
        proposals = proposals[keep]
        probs = probs[keep]
        order = probs.view(-1).argsort(descending=True)
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order]
        probs = probs[order]
        keep = nms(proposals, probs.squeeze(1), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep]
        probs = probs[keep]
        proposals = torch.cat((torch.zeros(proposals.size(0), 1).type_as(probs), proposals), dim=1)
        return proposals


def smooth_l1_loss(deltas, gt_deltas, inside_weights, outside_weights, sigma=1):
    """
    Calculate smooth L1 loss introduced by Fast-RCNN.

    f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
           |x| - 0.5 / sigma / sigma    otherwise

    Args:
        deltas (Tensor): Predicted regression deltas.
        gt_deltas (Tensor): Ground-truth regression deltas.
        inside_weights (Tensor): Calculate loss only for foreground proposals.
        outside_weights (Tensor): Weights of the smooth L1 loss relative to classification loss.
        sigma (float): Super parameter of smooth L1 loss.

    Returns:
        loss (Tensor)
    """
    sigma_2 = sigma ** 2
    x = inside_weights * torch.abs(deltas - gt_deltas)
    loss = torch.where(x < 1 / sigma_2, x ** 2 * sigma_2 * 0.5, x - 0.5 / sigma_2)
    loss *= outside_weights
    loss = loss.sum() / loss.size(0)
    return loss


class RPN(nn.Module):
    """
    Region proposal network.
    """

    def __init__(self, input_depth):
        super(RPN, self).__init__()
        self.num_anchors = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)
        self.rpn_conv = nn.Conv2d(input_depth, 512, 3, 1, 1)
        self.rpn_cls_score = nn.Conv2d(512, self.num_anchors * 2, 1, 1, 0)
        self.rpn_bbox_pred = nn.Conv2d(512, self.num_anchors * 4, 1, 1, 0)
        self.rpn_proposal = ProposalLayer()
        self.rpn_anchor_target = AnchorTargetLayer()

    @staticmethod
    def reshape(x, d):
        x = x.view(x.size(0), d, -1, x.size(3))
        return x

    def forward(self, base_feat, img_info, gt_boxes):
        """
        Args:
            base_feat (Tensor[1, C, H, W]): Basic feature extracted by backbone.
            img_info (Tensor[3]): (height, width, scale)
            gt_boxes (Tensor[N, 6]): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format

        Returns:
            proposals (Tensor[N, 5]): Predicted region proposals in (0, x1, y1, x2, y2) format.
            rpn_loss_cls, rpn_loss_bbox (Tensor): Training losses.
        """
        assert base_feat.size(0) == 1, 'Single batch only.'
        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)
        scores = self.rpn_cls_score(rpn_conv)
        scores_reshape = self.reshape(scores, 2)
        probs_reshape = F.softmax(scores_reshape, 1)
        probs = self.reshape(probs_reshape, self.num_anchors * 2)
        anchor_deltas = self.rpn_bbox_pred(rpn_conv)
        proposals = self.rpn_proposal(probs.data, anchor_deltas.data, img_info)
        rpn_loss_cls = 0
        rpn_loss_bbox = 0
        if self.training:
            assert gt_boxes is not None
            anchor_target = self.rpn_anchor_target(scores.data, gt_boxes, img_info)
            scores = scores_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            gt_anchor_labels = anchor_target[0].view(-1).long()
            rpn_loss_cls = F.cross_entropy(scores, gt_anchor_labels, ignore_index=-1)
            gt_anchor_deltas, anchor_inside_ws, anchor_outside_ws = anchor_target[1:]
            rpn_loss_bbox = smooth_l1_loss(anchor_deltas, gt_anchor_deltas, anchor_inside_ws, anchor_outside_ws, sigma=3)
        return proposals, rpn_loss_cls, rpn_loss_bbox


class UnlabeledMatching(Function):

    @staticmethod
    def forward(ctx, features, pid_labels, queue, tail):
        ctx.save_for_backward(features, pid_labels)
        ctx.queue = queue
        ctx.tail = tail
        scores = features.mm(queue.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail
        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)
        for indx, label in enumerate(pid_labels):
            if label == -1:
                queue[tail, :64] = features[indx, :64]
                tail += 1
                if tail >= queue.size(0):
                    tail -= queue.size(0)
        return grad_feats, None, None, None


class UnlabeledMatchingLayer(nn.Module):
    """
    Unlabeled matching of OIM loss function.
    """

    def __init__(self, queue_size=5000, feat_len=256):
        """
        Args:
            queue_size (int): Size of the queue saving the features of unlabeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(UnlabeledMatchingLayer, self).__init__()
        self.register_buffer('queue', torch.zeros(queue_size, feat_len))
        self.register_buffer('tail', torch.tensor(0))

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, queue_size]): Unlabeled matching scores, namely the similarities
                                            between proposals and unlabeled persons.
        """
        scores = UnlabeledMatching.apply(features, pid_labels, self.queue, self.tail)
        return scores


def img_preprocessing(img, flipped=False):
    """
    Image preprocessing: flip (optional), subtract mean, scale.

    Args:
        img (np.ndarray[H, W, C]): Origin image in BGR order.
        flipped (bool): Whether to flip the image.

    Returns:
        processed_img (np.ndarray[C, H, W]): Processed image.
        scale (float): The scale relative to the original image.
    """
    if flipped:
        img = img[:, ::-1, :]
    processed_img = img.astype(np.float32)
    processed_img -= cfg.PIXEL_MEANS
    img_size_min = np.min(processed_img.shape[0:2])
    img_size_max = np.max(processed_img.shape[0:2])
    scale = float(cfg.SCALE) / float(img_size_min)
    if np.round(scale * img_size_max) > cfg.MAX_SIZE:
        scale = float(cfg.MAX_SIZE) / float(img_size_max)
    processed_img = cv2.resize(processed_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    processed_img = processed_img.transpose((2, 0, 1))
    return processed_img, scale


class Network(nn.Module):
    """
    Person search network.

    Paper: Joint Detection and Identification Feature Learning for Person Search
           Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, Xiaogang Wang
    """

    def __init__(self):
        super(Network, self).__init__()
        rpn_depth = 1024
        num_classes = 2
        self.backbone = Backbone()
        self.head = Head()
        self.rpn = RPN(rpn_depth)
        self.roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.cls_score = nn.Linear(2048, num_classes)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feature = nn.Linear(2048, 256)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()
        self.freeze_blocks()

    def forward(self, img, img_info, gt_boxes, probe_roi=None):
        """
        Args:
            img (Tensor): Single image data.
            img_info (Tensor): (height, width, scale)
            gt_boxes (Tensor): Ground-truth boxes in (x1, y1, x2, y2, class, person_id) format.
            probe_roi (Tensor): Take probe_roi as proposal instead of using RPN.

        Returns:
            proposals (Tensor): Region proposals produced by RPN in (0, x1, y1, x2, y2) format.
            probs (Tensor): Classification probability of these proposals.
            proposal_deltas (Tensor): Proposal regression deltas.
            features (Tensor): Extracted features of these proposals.
            rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox and loss_oim (Tensor): Training losses.
        """
        assert img.size(0) == 1, 'Single batch only.'
        base_feat = self.backbone(img)
        if probe_roi is None:
            proposals, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, img_info, gt_boxes)
        else:
            proposals, rpn_loss_cls, rpn_loss_bbox = probe_roi, 0, 0
        if self.training:
            proposals, cls_labels, pid_labels, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws = self.proposal_target_layer(proposals, gt_boxes)
        else:
            cls_labels, pid_labels, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws = [None] * 5
        pooled_feat = self.roi_pool(base_feat, proposals)
        proposal_feat = self.head(pooled_feat).squeeze(2).squeeze(2)
        scores = self.cls_score(proposal_feat)
        probs = F.softmax(scores, dim=1)
        proposal_deltas = self.bbox_pred(proposal_feat)
        features = F.normalize(self.feature(proposal_feat))
        if self.training:
            loss_cls = F.cross_entropy(scores, cls_labels)
            loss_bbox = smooth_l1_loss(proposal_deltas, gt_proposal_deltas, proposal_inside_ws, proposal_outside_ws)
            labeled_matching_scores = self.labeled_matching_layer(features, pid_labels)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(features, pid_labels)
            unlabeled_matching_scores *= 10
            matching_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            pid_labels = pid_labels.clone()
            pid_labels[pid_labels == -2] = -1
            loss_oim = F.cross_entropy(matching_scores, pid_labels, ignore_index=-1)
        else:
            loss_cls, loss_bbox, loss_oim = 0, 0, 0
        return proposals, probs, proposal_deltas, features, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_oim

    def freeze_blocks(self):
        """
        The reason why we freeze all BNs in the backbone: The batch size is 1
        in the backbone, so BN is not stable.

        Reference: https://github.com/ShuangLI59/person_search/issues/87
        """
        for p in self.backbone.SpatialConvolution_0.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.backbone.apply(set_bn_fix)

    def train(self, mode=True):
        """
        It's not enough to just freeze all BNs in backbone.
        Setting them to eval mode is also needed.
        """
        nn.Module.train(self, mode)
        if mode:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)

    def inference(self, img, probe_roi=None, threshold=0.75):
        """
        End to end inference. Specific behavior depends on probe_roi.
        If probe_roi is None, detect persons in the image and extract their features.
        Otherwise, extract the feature of the probe RoI in the image.

        Args:
            img (np.ndarray[H, W, C]): Image of BGR order.
            probe_roi (np.ndarray[4]): The RoI to be extracting feature.
            threshold (float): The threshold used to remove those bounding boxes with low scores.

        Returns:
            detections (Tensor[N, 5]): Detected person bounding boxes in
                                       (x1, y1, x2, y2, score) format.
            features (Tensor[N, 256]): Features of these bounding boxes.
        """
        device = self.cls_score.weight.device
        processed_img, scale = img_preprocessing(img)
        processed_img = torch.from_numpy(processed_img).unsqueeze(0)
        img_info = torch.Tensor([processed_img.shape[2], processed_img.shape[3], scale])
        if probe_roi is not None:
            probe_roi = torch.from_numpy(probe_roi).float().view(1, 4)
            probe_roi *= scale
            probe_roi = torch.cat((torch.zeros(1, 1), probe_roi.float()), dim=1)
        with torch.no_grad():
            proposals, probs, proposal_deltas, features, _, _, _, _, _ = self.forward(processed_img, img_info, None, probe_roi)
        if probe_roi is not None:
            return features
        proposals = proposals[:, 1:5] / scale
        num_classes = proposal_deltas.shape[1] // 4
        stds = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(num_classes)
        means = torch.Tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(num_classes)
        proposal_deltas = proposal_deltas * stds + means
        boxes = bbox_transform_inv(proposals, proposal_deltas)
        boxes = clip_boxes(boxes, img.shape)
        j = 1
        keep = torch.nonzero(probs[:, j] > threshold)[:, 0]
        boxes = boxes[keep, j * 4:(j + 1) * 4]
        probs = probs[keep, j]
        features = features[keep]
        detections = torch.cat((boxes, probs.unsqueeze(1)), dim=1)
        keep = nms(boxes, probs, cfg.TEST.NMS)
        detections = detections[keep]
        features = features[keep]
        return detections, features


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Head,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
]

class Test_serend1p1ty_person_search(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

