import sys
_module = sys.modules[__name__]
del sys
mobilenet_v1_1x = _module
res34_distill_res18_img = _module
res50_distill_mv1_img = _module
mobilenet_v1 = _module
train = _module
distillation = _module
builder = _module
distillers = _module
classification_distiller = _module
losses = _module
mgd = _module
backbones = _module
mobilenet_v1 = _module
train = _module
cascade_mask_rcnn_rx101_32x4d_distill_faster_rcnn_r50_fpn_2x_coco = _module
cascade_mask_rcnn_rx101_32x4d_distill_mask_rcnn_r50_fpn_2x_coco = _module
reppoints_rx101_64x4d_distill_reppoints_r50_fpn_2x_coco = _module
retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco = _module
solo_r101_ms_distill_solo_r50_coco = _module
train = _module
builder = _module
detection_distiller = _module
mgd = _module
train = _module
pth_transfer = _module
cityscapes_512x512 = _module
psp_r101_distill_deepv3_r18_40k_512x512_city = _module
psp_r101_distill_psp_r18_40k_512x512_city = _module
train = _module
builder = _module
segmentation_distiller = _module
mgd = _module
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


import random


import warnings


import numpy as np


import torch


import torch.distributed as dist


from torch import nn


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import copy


import time


class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00007
        lambda_mgd (float, optional): masked ratio. Defaults to 0.5
    """

    def __init__(self, student_channels, teacher_channels, name, alpha_mgd=7e-05, lambda_mgd=0.15):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        self.generation = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        if self.align is not None:
            preds_S = self.align(preds_S)
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, C, 1, 1))
        mat = torch.where(mat < self.lambda_mgd, 0, 1)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class FeatureLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self, student_channels, teacher_channels, name, alpha_mgd=2e-05, lambda_mgd=0.75):
        super(FeatureLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        self.generation = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        if self.align is not None:
            preds_S = self.align(preds_S)
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd
        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W))
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeatureLoss,
     lambda: ([], {'student_channels': 4, 'teacher_channels': 4, 'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MGDLoss,
     lambda: ([], {'student_channels': 4, 'teacher_channels': 4, 'name': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yzd_v_MGD(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

