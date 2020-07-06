import sys
_module = sys.modules[__name__]
del sys
normalizeImages = _module
progressBar = _module
utils = _module
config = _module
medicalDataLoader = _module
main = _module
attention = _module
my_stacked_danet = _module
resnext = _module
resnext101_regular = _module

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


import scipy.io as sio


import time


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


from random import random


from random import randint


import warnings


from torch.optim import Adam


import math


from torch.nn import functional as F


from functools import reduce


from torchvision.models import resnext50_32x4d


from torchvision.models import resnext101_32x8d


from torch import nn


def to_var(x):
    if torch.cuda.is_available():
        x = x
    return Variable(x)


class computeDiceOneHot(nn.Module):

    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-08) / (sum + 1e-08)
        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))
        DiceZ = to_var(torch.zeros(batchsize, 2))
        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])
            DiceZ[i, 0] = self.inter(pred[i, 4], GT[i, 4])
            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])
            DiceZ[i, 1] = self.sum(pred[i, 4], GT[i, 4])
        return DiceN, DiceB, DiceW, DiceT, DiceZ


class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """

    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))

    def forward(self, x):
        return self.decode(x)


class semanticModule(nn.Module):
    """
    Semantic attention module
    """

    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim
        self.enc1 = _EncoderBlock(in_dim, in_dim * 2)
        self.enc2 = _EncoderBlock(in_dim * 2, in_dim * 4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode='bilinear'))
        return enc2.view(-1), dec1


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention
    
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """

    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer, self).__init__()
        self.attn = nn.Sequential(nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch), nn.PReLU(), PAM_Module(in_ch) if use_pam else CAM_Module(in_ch), nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch), nn.PReLU())

    def forward(self, x):
        return self.attn(x)


class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.
    
    Parameters:
    ----------
    inputs:
        in_ch : input channels
        out_ch : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """

    def __init__(self, in_ch, out_ch, attn=True):
        super(MultiConv, self).__init__()
        self.fuse_attn = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(), nn.Conv2d(out_ch, out_ch, kernel_size=1), nn.BatchNorm2d(64), nn.Softmax2d() if attn else nn.PReLU())

    def forward(self, x):
        return self.fuse_attn(x)


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def resnext101():
    model = resnext101_32x8d()
    model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False)
    model.avgpool = nn.AvgPool2d((7, 7), (1, 1))
    model.fc = nn.Sequential(Lambda(lambda x: x.view(x.size(0), -1)), Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000))
    return model


class ResNeXt101(nn.Module):

    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext101()
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3:5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


class DAF_stack(nn.Module):

    def __init__(self):
        super(DAF_stack, self).__init__()
        self.resnext = ResNeXt101()
        self.down4 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down3 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down2 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        self.down1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU())
        inter_channels = 64
        out_channels = 64
        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv8_1 = nn.Conv2d(64, 64, 1)
        self.conv8_2 = nn.Conv2d(64, 64, 1)
        self.conv8_3 = nn.Conv2d(64, 64, 1)
        self.conv8_4 = nn.Conv2d(64, 64, 1)
        self.conv8_11 = nn.Conv2d(64, 64, 1)
        self.conv8_12 = nn.Conv2d(64, 64, 1)
        self.conv8_13 = nn.Conv2d(64, 64, 1)
        self.conv8_14 = nn.Conv2d(64, 64, 1)
        self.softmax_1 = nn.Softmax(dim=-1)
        self.pam_attention_1_1 = PAM_CAM_Layer(64, True)
        self.cam_attention_1_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_1_1 = semanticModule(128)
        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pam_attention_1_2 = PAM_CAM_Layer(64)
        self.cam_attention_1_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_3 = PAM_CAM_Layer(64)
        self.cam_attention_1_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_4 = PAM_CAM_Layer(64)
        self.cam_attention_1_4 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_1 = PAM_CAM_Layer(64)
        self.cam_attention_2_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_2_1 = semanticModule(128)
        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pam_attention_2_2 = PAM_CAM_Layer(64)
        self.cam_attention_2_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_3 = PAM_CAM_Layer(64)
        self.cam_attention_2_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_4 = PAM_CAM_Layer(64)
        self.cam_attention_2_4 = PAM_CAM_Layer(64, False)
        self.fuse1 = MultiConv(256, 64, False)
        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)
        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)
        self.predict4 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict4_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)
        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)
        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)
        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        semVector_1_1, semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1), 1))
        attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))
        attention1_4 = self.conv8_1((attn_cam4 + attn_pam4) * self.conv_sem_1_1(semanticModule_1_1))
        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3 = self.conv8_2((attn_cam3 + attn_pam3) * self.conv_sem_1_2(semanticModule_1_2))
        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2 = self.conv8_3((attn_cam2 + attn_pam2) * self.conv_sem_1_3(semanticModule_1_3))
        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv8_4((attn_cam1 + attn_pam1) * self.conv_sem_1_4(semanticModule_1_4))
        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_1 = self.pam_attention_2_4(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_2 = self.cam_attention_2_4(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4 = self.conv8_11((refine4_1 + refine4_2) * self.conv_sem_2_1(semanticModule_2_1))
        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.pam_attention_2_3(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_2 = self.cam_attention_2_3(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3 = self.conv8_12((refine3_1 + refine3_2) * self.conv_sem_2_2(semanticModule_2_2))
        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.pam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_2 = self.cam_attention_2_2(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2 = self.conv8_13((refine2_1 + refine2_2) * self.conv_sem_2_3(semanticModule_2_3))
        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.pam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_2 = self.cam_attention_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1 = self.conv8_14((refine1_1 + refine1_2) * self.conv_sem_2_4(semanticModule_2_4))
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)
        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')
        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        if self.training:
            return semVector_1_1, semVector_2_1, semVector_1_2, semVector_2_2, semVector_1_3, semVector_2_3, semVector_1_4, semVector_2_4, torch.cat((down1, fuse1), 1), torch.cat((down2, fuse1), 1), torch.cat((down3, fuse1), 1), torch.cat((down4, fuse1), 1), torch.cat((down1, attention1_1 * fuse1), 1), torch.cat((down2, attention1_2 * fuse1), 1), torch.cat((down3, attention1_3 * fuse1), 1), torch.cat((down4, attention1_4 * fuse1), 1), semanticModule_1_4, semanticModule_1_3, semanticModule_1_2, semanticModule_1_1, semanticModule_2_4, semanticModule_2_3, semanticModule_2_2, semanticModule_2_1, predict1, predict2, predict3, predict4, predict1_2, predict2_2, predict3_2, predict4_2
        else:
            return (predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4


class LambdaMap(LambdaBase):

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CAM_Module,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DAF_stack,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PAM_CAM_Layer,
     lambda: ([], {'in_ch': 64}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (PAM_Module,
     lambda: ([], {'in_dim': 64}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResNeXt101,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (_DecoderBlock,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_EncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (computeDiceOneHot,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 4, 4]), torch.rand([4, 5, 4, 4])], {}),
     False),
    (semanticModule,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sinAshish_Multi_Scale_Attention(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

