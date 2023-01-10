import sys
_module = sys.modules[__name__]
del sys
get_data = _module
get_threshold = _module
dataset = _module
dataset_with_aug = _module
ava_Dice_loss = _module
ava_Dice_loss_with_bg = _module
ce_loss = _module
focal_loss = _module
genernalised_DIce_loss = _module
ResUnet_CE = _module
ResUnet_dice = _module
train = _module
val = _module

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


import torch


from torch.utils.data import Dataset as dataset


import torch.nn.functional as F


from torch.autograd import Variable


import scipy.ndimage as ndimage


import torch.nn as nn


from time import time


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import numpy as np


num_organ = 13


organ_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_stage1, pred_stage2, target):
        """
        :param pred_stage1: 经过放大之后(B, 14, 48, 256, 256)
        :param pred_stage2: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)
        :return: Dice距离
        """
        organ_target = torch.zeros((target.size(0), num_organ, 48, 256, 256))
        for organ_index in range(1, num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index - 1, :, :, :] = temp_target
        organ_target = organ_target
        dice_stage1_numerator = 0.0
        dice_stage1_denominator = 0.0
        for organ_index in range(1, num_organ + 1):
            dice_stage1_numerator += 2 * (pred_stage1[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)
            dice_stage1_numerator *= organ_weight[organ_index - 1]
            dice_stage1_denominator += pred_stage1[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-05
            dice_stage1_denominator *= organ_weight[organ_index - 1]
        dice_stage1 = dice_stage1_numerator / dice_stage1_denominator
        dice_stage2_numerator = 0.0
        dice_stage2_denominator = 0.0
        for organ_index in range(1, num_organ + 1):
            dice_stage2_numerator += 2 * (pred_stage2[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)
            dice_stage2_numerator *= organ_weight[organ_index - 1]
            dice_stage2_denominator += pred_stage2[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-05
            dice_stage2_denominator *= organ_weight[organ_index - 1]
        dice_stage2 = dice_stage2_numerator / dice_stage2_denominator
        dice = dice_stage1 + dice_stage2
        return (2 - dice).mean()


class CELoss(nn.Module):

    def __init__(self, alpha=2):
        """

        :param alpha: focal loss中的指数项的次数
        """
        super().__init__()
        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred_stage1, pred_stage2, target):
        """

        :param pred_stage1: (B, 14, 48, 256, 256)
        :param pred_stage2: (B, 14, 48, 256, 256)
        :param target: (B, 48, 256, 256)
        """
        num_target = (target > 0).type(torch.FloatTensor).sum()
        loss_stage1 = self.loss(pred_stage1, target)
        loss_stage2 = self.loss(pred_stage2, target)
        exponential_term_stage1 = (1 - F.softmax(pred_stage1, dim=1).max(dim=1)[0]) ** self.alpha
        exponential_term_stage2 = (1 - F.softmax(pred_stage2, dim=1).max(dim=1)[0]) ** self.alpha
        loss_stage1 *= exponential_term_stage1
        loss_stage2 *= exponential_term_stage2
        loss = loss_stage1 + loss_stage2
        if num_target == 0:
            loss = loss.mean()
        else:
            loss = loss.sum() / num_target
        return loss


dropout_rate = 0.3


class ResUNet(nn.Module):
    """
    共9332094个可训练的参数, 九百三十万左右
    """

    def __init__(self, training, inchannel, stage):
        """
        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        """
        super().__init__()
        self.training = training
        self.stage = stage
        self.encoder_stage1 = nn.Sequential(nn.Conv3d(inchannel, 16, 3, 1, padding=1), nn.PReLU(16))
        self.encoder_stage2 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32), nn.Conv3d(32, 32, 3, 1, padding=1), nn.PReLU(32))
        self.encoder_stage3 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, padding=1), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4), nn.PReLU(64))
        self.encoder_stage4 = nn.Sequential(nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5), nn.PReLU(128))
        self.decoder_stage1 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, padding=1), nn.PReLU(256), nn.Conv3d(256, 256, 3, 1, padding=1), nn.PReLU(256), nn.Conv3d(256, 256, 3, 1, padding=1), nn.PReLU(256))
        self.decoder_stage2 = nn.Sequential(nn.Conv3d(128 + 64, 128, 3, 1, padding=1), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=1), nn.PReLU(128), nn.Conv3d(128, 128, 3, 1, padding=1), nn.PReLU(128))
        self.decoder_stage3 = nn.Sequential(nn.Conv3d(64 + 32, 64, 3, 1, padding=1), nn.PReLU(64), nn.Conv3d(64, 64, 3, 1, padding=1), nn.PReLU(64))
        self.decoder_stage4 = nn.Sequential(nn.Conv3d(32 + 16, 32, 3, 1, padding=1), nn.PReLU(32))
        self.down_conv1 = nn.Sequential(nn.Conv3d(16, 32, 2, 2), nn.PReLU(32))
        self.down_conv2 = nn.Sequential(nn.Conv3d(32, 64, 2, 2), nn.PReLU(64))
        self.down_conv3 = nn.Sequential(nn.Conv3d(64, 128, 2, 2), nn.PReLU(128))
        self.down_conv4 = nn.Sequential(nn.Conv3d(128, 256, 3, 1, padding=1), nn.PReLU(256))
        self.up_conv2 = nn.Sequential(nn.ConvTranspose3d(256, 128, 2, 2), nn.PReLU(128))
        self.up_conv3 = nn.Sequential(nn.ConvTranspose3d(128, 64, 2, 2), nn.PReLU(64))
        self.up_conv4 = nn.Sequential(nn.ConvTranspose3d(64, 32, 2, 2), nn.PReLU(32))
        self.map = nn.Sequential(nn.Conv3d(32, num_organ + 1, 1), nn.Softmax(dim=1))

    def forward(self, inputs):
        if self.stage is 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)
        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)
        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)
        short_range4 = self.down_conv4(long_range4)
        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, dropout_rate, self.training)
        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, dropout_rate, self.training)
        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, dropout_rate, self.training)
        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        outputs = self.map(outputs)
        return outputs


class Net(nn.Module):

    def __init__(self, training):
        super().__init__()
        self.training = training
        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')
        self.stage2 = ResUNet(training=training, inchannel=num_organ + 2, stage='stage2')

    def forward(self, inputs):
        """
        首先将输入数据在轴向上缩小一倍，然后送入第一阶段网络中
        得到一个粗糙尺度下的分割结果
        然后将原始尺度大小的数据与第一步中得到的分割结果进行拼接，共同送入第二阶段网络中
        得到最终的分割结果
        共18656348个可训练的参数，一千八百万左右
        """
        inputs_stage1 = F.upsample(inputs, (48, 128, 128), mode='trilinear')
        output_stage1 = self.stage1(inputs_stage1)
        output_stage1 = F.upsample(output_stage1, (48, 256, 256), mode='trilinear')
        inputs_stage2 = torch.cat((output_stage1, inputs), dim=1)
        output_stage2 = self.stage2(inputs_stage2)
        if self.training is True:
            return output_stage1, output_stage2
        else:
            return output_stage2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_assassint2017_abdominal_multi_organ_segmentation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

