import sys
_module = sys.modules[__name__]
del sys
get_spacing = _module
liver_slice_percentage = _module
liver_voxel_percentage = _module
get_training_set = _module
dataset = _module
BCE = _module
Dice = _module
ELDice = _module
Hybrid = _module
Jaccard = _module
SS = _module
Tversky = _module
WBCE = _module
ResUNet = _module
parameter = _module
train_ds = _module
calculate_metrics = _module
val = _module

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


import torch.nn as nn


import torch


import torch.nn.functional as F


from time import time


import numpy as np


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import copy


import collections


import scipy.ndimage as ndimage


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        return self.bce_loss(pred, target)


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        return torch.clamp((1 - dice).mean(), 0, 1)


class ELDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        return torch.clamp(torch.pow(-torch.log(dice + 1e-05), 0.3).mean(), 0, 2)


class HybridLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        return torch.clamp((1 - dice).mean(), 0, 1)


class SSLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        s1 = ((pred - target).pow(2) * target).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target.sum(dim=1).sum(dim=1).sum(dim=1))
        s2 = ((pred - target).pow(2) * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target).sum(dim=1).sum(dim=1).sum(dim=1))
        return (0.05 * s1 + 0.95 * s2).mean()


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        smooth = 1
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) + 0.3 * (pred * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        return torch.clamp((1 - dice).mean(), 0, 2)


class WCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        weight = torch.FloatTensor([0.05, 1])
        self.ce_loss = nn.CrossEntropyLoss(weight)

    def forward(self, pred, target):
        pred_ = torch.ones_like(pred) - pred
        pred = torch.cat((pred_, pred), dim=1)
        target = torch.long()
        return self.ce_loss(pred, target)


class ResUNet(nn.Module):
    """

    共9498260个可训练的参数, 接近九百五十万
    """

    def __init__(self, training):
        super().__init__()
        self.training = training
        self.encoder_stage1 = nn.Sequential(nn.Conv3d(1, 16, 3, 1, padding=1), nn.PReLU(16), nn.Conv3d(16, 16, 3, 1, padding=1), nn.PReLU(16))
        self.encoder_stage2 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32), nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32), nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32))
        self.encoder_stage3 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, padding=1), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4), nn.PReLU(64))
        self.encoder_stage4 = nn.Sequential(nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5), nn.PReLU(128))
        self.decoder_stage1 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, padding=1), nn.PReLU(256), nn.Conv3d(256, 256, 3, 1, padding=1), nn.PReLU(256), nn.Conv3d(256, 256, 3, 1, padding=1), nn.PReLU(256))
        self.decoder_stage2 = nn.Sequential(nn.Conv3d(128 + 64, 128, 3, 1, padding=1), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=1), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=1), nn.PReLU(128))
        self.decoder_stage3 = nn.Sequential(nn.Conv3d(64 + 32, 64, 3, 1, padding=1), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=1), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=1), nn.PReLU(64))
        self.decoder_stage4 = nn.Sequential(nn.Conv3d(32 + 16, 32, 3, 1, padding=1), nn.PReLU(32), nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32))
        self.down_conv1 = nn.Sequential(nn.Conv3d(16, 32, 2, 2), nn.PReLU(32))
        self.down_conv2 = nn.Sequential(nn.Conv3d(32, 64, 2, 2), nn.PReLU(64))
        self.down_conv3 = nn.Sequential(nn.Conv3d(64, 128, 2, 2), nn.PReLU(128))
        self.down_conv4 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, padding=1), nn.PReLU(256))
        self.up_conv2 = nn.Sequential(nn.ConvTranspose3d(256, 128, 2, 2), nn.PReLU(128))
        self.up_conv3 = nn.Sequential(nn.ConvTranspose3d(128, 64, 2, 2), nn.PReLU(64))
        self.up_conv4 = nn.Sequential(nn.ConvTranspose3d(64, 32, 2, 2), nn.PReLU(32))
        self.map4 = nn.Sequential(nn.Conv3d(32, 1, 1, 1), nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'), nn.Sigmoid())
        self.map3 = nn.Sequential(nn.Conv3d(64, 1, 1, 1), nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear'), nn.Sigmoid())
        self.map2 = nn.Sequential(nn.Conv3d(128, 1, 1, 1), nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'), nn.Sigmoid())
        self.map1 = nn.Sequential(nn.Conv3d(256, 1, 1, 1), nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'), nn.Sigmoid())

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, para.drop_rate, self.training)
        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, para.drop_rate, self.training)
        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, para.drop_rate, self.training)
        short_range4 = self.down_conv4(long_range4)
        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, para.drop_rate, self.training)
        output1 = self.map1(outputs)
        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, self.training)
        output2 = self.map2(outputs)
        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, self.training)
        output3 = self.map3(outputs)
        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        output4 = self.map4(outputs)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ELDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HybridLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (JaccardLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TverskyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_assassint2017_MICCAI_LITS2017(_paritybench_base):
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

