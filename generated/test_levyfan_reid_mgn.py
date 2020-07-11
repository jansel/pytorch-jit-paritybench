import sys
_module = sys.modules[__name__]
del sys
mgn = _module
ide = _module
market1501 = _module
mgn = _module
test_market1501 = _module
test_triplet = _module
triplet = _module

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


from sklearn.metrics import average_precision_score


from scipy.spatial.distance import cdist


from sklearn.preprocessing import normalize


from torch import nn


from torch import optim


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.models import resnet50


from torchvision.transforms import functional


import collections


import random


import re


from torch.utils.data import dataset


from torch.utils.data import sampler


from torchvision.datasets.folder import default_loader


import copy


from torch.utils.data import dataloader


from torchvision.models.resnet import resnet50


from torchvision.models.resnet import Bottleneck


from torchvision.transforms import ToTensor


from torch.nn import functional as F


class IDE(nn.Module):

    def __init__(self, num_classes):
        super(IDE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(p=0.5), nn.Linear(512, num_classes))
        nn.init.kaiming_normal_(self.classifier[0].weight, mode='fan_out')
        nn.init.constant_(self.classifier[0].bias, 0.0)
        nn.init.normal_(self.classifier[1].weight, mean=1.0, std=0.02)
        nn.init.constant_(self.classifier[1].bias, 0.0)
        nn.init.normal_(self.classifier[4].weight, std=0.001)
        nn.init.constant_(self.classifier[4].bias, 0.0)

    def forward(self, x):
        """
        :param x: input image of (N, C, H, W)
        :return: (feature of N*2048, label predict of N*num_classes)
        """
        x = self.backbone(x)
        x = x.squeeze()
        y = self.classifier(x)
        return x, y


class MGN(nn.Module):
    """
    @ARTICLE{2018arXiv180401438W,
        author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
        title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1804.01438},
        primaryClass = "cs.CV",
        keywords = {Computer Science - Computer Vision and Pattern Recognition},
        year = 2018,
        month = apr,
        adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """

    def __init__(self, num_classes):
        super(MGN, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3[0])
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_g_conv5 = resnet.layer4
        res_p_conv5 = nn.Sequential(Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))), Bottleneck(2048, 512), Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)
        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        nn.init.normal_(reduction[1].weight, mean=1.0, std=0.02)
        nn.init.constant_(reduction[1].bias, 0.0)

    @staticmethod
    def _init_fc(fc):
        nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        predict = []
        triplet_losses = []
        softmax_losses = []
        if args.model in {'mgn', 'p1_single'}:
            p1 = self.p1(x)
            zg_p1 = self.maxpool_zg_p1(p1)
            fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
            l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
            predict.append(fg_p1)
            triplet_losses.append(fg_p1)
            softmax_losses.append(l_p1)
        if args.model in {'mgn', 'p2_single'}:
            p2 = self.p2(x)
            zg_p2 = self.maxpool_zg_p2(p2)
            fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
            l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
            zp2 = self.maxpool_zp2(p2)
            z0_p2 = zp2[:, :, 0:1, :]
            z1_p2 = zp2[:, :, 1:2, :]
            f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
            f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
            l0_p2 = self.fc_id_256_1_0(f0_p2)
            l1_p2 = self.fc_id_256_1_1(f1_p2)
            predict.extend([fg_p2, f0_p2, f1_p2])
            triplet_losses.append(fg_p2)
            softmax_losses.extend([l_p2, l0_p2, l1_p2])
        if args.model in {'mgn', 'p3_single'}:
            p3 = self.p3(x)
            zg_p3 = self.maxpool_zg_p3(p3)
            fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
            l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
            zp3 = self.maxpool_zp3(p3)
            z0_p3 = zp3[:, :, 0:1, :]
            z1_p3 = zp3[:, :, 1:2, :]
            z2_p3 = zp3[:, :, 2:3, :]
            f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
            f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
            f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)
            l0_p3 = self.fc_id_256_2_0(f0_p3)
            l1_p3 = self.fc_id_256_2_1(f1_p3)
            l2_p3 = self.fc_id_256_2_2(f2_p3)
            predict.extend([fg_p3, f0_p3, f1_p3, f2_p3])
            triplet_losses.append(fg_p3)
            softmax_losses.extend([l_p3, l0_p3, l1_p3, l2_p3])
        predict = torch.cat(predict, dim=1)
        return predict, triplet_losses, softmax_losses


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TripletSemihardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, margin=0, size_average=True):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)
        pos_mask = same_id
        neg_mask = 1 - same_id

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1000000.0 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1000000.0 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + torch.sum(input.t() ** 2, dim=0, keepdim=True) - 2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()
        pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
        neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)
        y = torch.ones(same_id.size()[0])
        return F.margin_ranking_loss(neg_min.float(), pos_max.float(), y, self.margin, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IDE,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_levyfan_reid_mgn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

