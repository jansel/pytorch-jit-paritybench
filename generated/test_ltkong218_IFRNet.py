import sys
_module = sys.modules[__name__]
del sys
Middlebury_Other = _module
SNU_FILM = _module
UCF101 = _module
Vimeo90K = _module
speed_parameters = _module
datasets = _module
demo_2x = _module
demo_8x = _module
generate_flow = _module
correlation = _module
run = _module
loss = _module
metric = _module
IFRNet = _module
IFRNet_L = _module
IFRNet_S = _module
train_gopro = _module
train_vimeo90k = _module
utils = _module

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


import torch.nn.functional as F


import time


import torch.nn as nn


import random


from torch.utils.data import Dataset


import math


import re


import numpy


from math import exp


import torch.optim as optim


from torch.utils.data import DataLoader


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP


import logging


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction].replace('{{intStride}}', str(objVariables['intStride']))
    while True:
        objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    return strKernel


class _FunctionCorrelation(torch.autograd.Function):

    @staticmethod
    def forward(self, one, two, intStride):
        rbot0 = one.new_zeros([one.shape[0], one.shape[2] + 6 * intStride, one.shape[3] + 6 * intStride, one.shape[1]])
        rbot1 = one.new_zeros([one.shape[0], one.shape[2] + 6 * intStride, one.shape[3] + 6 * intStride, one.shape[1]])
        self.intStride = intStride
        one = one.contiguous()
        two = two.contiguous()
        output = one.new_zeros([one.shape[0], 49, int(math.ceil(one.shape[2] / intStride)), int(math.ceil(one.shape[3] / intStride))])
        if one.is_cuda == True:
            n = one.shape[2] * one.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'intStride': self.intStride, 'input': one, 'output': rbot0}))(grid=tuple([int((n + 16 - 1) / 16), one.shape[1], one.shape[0]]), block=tuple([16, 1, 1]), args=[cupy.int32(n), one.data_ptr(), rbot0.data_ptr()])
            n = two.shape[2] * two.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'intStride': self.intStride, 'input': two, 'output': rbot1}))(grid=tuple([int((n + 16 - 1) / 16), two.shape[1], two.shape[0]]), block=tuple([16, 1, 1]), args=[cupy.int32(n), two.data_ptr(), rbot1.data_ptr()])
            n = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'top': output}))(grid=tuple([output.shape[3], output.shape[2], output.shape[0]]), block=tuple([32, 1, 1]), shared_mem=one.shape[1] * 4, args=[cupy.int32(n), rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()])
        elif one.is_cuda == False:
            raise NotImplementedError()
        self.save_for_backward(one, two, rbot0, rbot1)
        return output

    @staticmethod
    def backward(self, gradOutput):
        one, two, rbot0, rbot1 = self.saved_tensors
        gradOutput = gradOutput.contiguous()
        gradOne = one.new_zeros([one.shape[0], one.shape[1], one.shape[2], one.shape[3]]) if self.needs_input_grad[0] == True else None
        gradTwo = one.new_zeros([one.shape[0], one.shape[1], one.shape[2], one.shape[3]]) if self.needs_input_grad[1] == True else None
        if one.is_cuda == True:
            if gradOne is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch('kernel_Correlation_updateGradOne', cupy_kernel('kernel_Correlation_updateGradOne', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradOne': gradOne, 'gradTwo': None}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), gradOne.data_ptr(), None])
            if gradTwo is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch('kernel_Correlation_updateGradTwo', cupy_kernel('kernel_Correlation_updateGradTwo', {'intStride': self.intStride, 'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradOne': None, 'gradTwo': gradTwo}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None, gradTwo.data_ptr()])
        elif one.is_cuda == False:
            raise NotImplementedError()
        return gradOne, gradTwo, None


class ModuleCorrelation(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tenOne, tenTwo, intStride):
        return _FunctionCorrelation.apply(tenOne, tenTwo, intStride)


arguments_strModel = 'default'


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + 1.0 / tenFlow.shape[3], 1.0 - 1.0 / tenFlow.shape[3], tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + 1.0 / tenFlow.shape[2], 1.0 - 1.0 / tenFlow.shape[2], tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)


class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()


        class Features(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


        class Matching(torch.nn.Module):

            def __init__(self, intLevel):
                super().__init__()
                self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                if intLevel == 6:
                    self.netUpflow = None
                elif intLevel != 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)
                if intLevel >= 4:
                    self.netUpcorr = None
                elif intLevel < 4:
                    self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFlow = self.netUpflow(tenFlow)
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackwarp)
                if self.netUpcorr is None:
                    tenCorrelation = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=1), negative_slope=0.1, inplace=False)
                elif self.netUpcorr is not None:
                    tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenFeaturesOne, tenTwo=tenFeaturesTwo, intStride=2), negative_slope=0.1, inplace=False))
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)


        class Subpixel(torch.nn.Module):

            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                if intLevel != 2:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel == 2:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenFeaturesOne = self.netFeat(tenFeaturesOne)
                tenFeaturesTwo = self.netFeat(tenFeaturesTwo)
                if tenFlow is not None:
                    tenFeaturesTwo = backwarp(tenInput=tenFeaturesTwo, tenFlow=tenFlow * self.fltBackward)
                return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([tenFeaturesOne, tenFeaturesTwo, tenFlow], 1))


        class Regularization(torch.nn.Module):

            def __init__(self, intLevel):
                super().__init__()
                self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
                self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]
                if intLevel >= 5:
                    self.netFeat = torch.nn.Sequential()
                elif intLevel < 5:
                    self.netFeat = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                if intLevel >= 5:
                    self.netDist = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
                elif intLevel < 5:
                    self.netDist = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1, padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0)), torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]), stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])))
                self.netScaleX = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.netScaleY = torch.nn.Conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)

            def forward(self, tenOne, tenTwo, tenFeaturesOne, tenFeaturesTwo, tenFlow):
                tenDifference = ((tenOne - backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackward)) ** 2).sum(1, True).sqrt().detach()
                tenDist = self.netDist(self.netMain(torch.cat([tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), self.netFeat(tenFeaturesOne)], 1)))
                tenDist = (tenDist ** 2).neg()
                tenDist = (tenDist - tenDist.max(1, True)[0]).exp()
                tenDivisor = tenDist.sum(1, True).reciprocal()
                tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
                return torch.cat([tenScaleX, tenScaleY], 1)
        self.netFeatures = Features()
        self.netMatching = torch.nn.ModuleList([Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netSubpixel = torch.nn.ModuleList([Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.netRegularization = torch.nn.ModuleList([Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]])
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-liteflownet/network-' + arguments_strModel + '.pytorch').items()})

    def forward(self, tenOne, tenTwo):
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] - 0.411618
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] - 0.434631
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] - 0.454253
        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] - 0.410782
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] - 0.433645
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] - 0.452793
        tenFeaturesOne = self.netFeatures(tenOne)
        tenFeaturesTwo = self.netFeatures(tenTwo)
        tenOne = [tenOne]
        tenTwo = [tenTwo]
        for intLevel in [1, 2, 3, 4, 5]:
            tenOne.append(torch.nn.functional.interpolate(input=tenOne[-1], size=(tenFeaturesOne[intLevel].shape[2], tenFeaturesOne[intLevel].shape[3]), mode='bilinear', align_corners=False))
            tenTwo.append(torch.nn.functional.interpolate(input=tenTwo[-1], size=(tenFeaturesTwo[intLevel].shape[2], tenFeaturesTwo[intLevel].shape[3]), mode='bilinear', align_corners=False))
        tenFlow = None
        for intLevel in [-1, -2, -3, -4, -5]:
            tenFlow = self.netMatching[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netSubpixel[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
            tenFlow = self.netRegularization[intLevel](tenOne[intLevel], tenTwo[intLevel], tenFeaturesOne[intLevel], tenFeaturesTwo[intLevel], tenFlow)
        return tenFlow * 20.0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Ternary(nn.Module):

    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Geometry(nn.Module):

    def __init__(self, patch_size=3):
        super(Geometry, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        b, c, h, w = tensor.size()
        tensor_ = tensor.reshape(b * c, 1, h, w)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c * self.patch_size ** 2, h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_ ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class Charbonnier_L1(nn.Module):

    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-06) ** 0.5).mean()
        else:
            loss = ((diff ** 2 + 1e-06) ** 0.5 * mask).mean() / (mask.mean() + 1e-09)
        return loss


class Charbonnier_Ada(nn.Module):

    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss


class ResBlock(nn.Module):

    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.PReLU(in_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.PReLU(side_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.PReLU(in_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), nn.PReLU(side_channels))
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :])
        out = self.prelu(x + self.conv5(out))
        return out


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), nn.PReLU(out_channels))


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(convrelu(3, 24, 3, 2, 1), convrelu(24, 24, 3, 1, 1))
        self.pyramid2 = nn.Sequential(convrelu(24, 36, 3, 2, 1), convrelu(36, 36, 3, 1, 1))
        self.pyramid3 = nn.Sequential(convrelu(36, 54, 3, 2, 1), convrelu(54, 54, 3, 1, 1))
        self.pyramid4 = nn.Sequential(convrelu(54, 72, 3, 2, 1), convrelu(72, 72, 3, 1, 1))

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):

    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(convrelu(144 + 1, 144), ResBlock(144, 24), nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True))

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


class Decoder3(nn.Module):

    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(convrelu(166, 162), ResBlock(162, 24), nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):

    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(convrelu(112, 108), ResBlock(108, 24), nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):

    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(convrelu(76, 72), ResBlock(72, 24), nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True))

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


def get_robust_weight(flow_pred, flow_gt, beta):
    epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-beta * epe)
    return robust_weight


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)


class Model(nn.Module):

    def __init__(self, local_rank=-1, lr=0.0001):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def inference(self, img0, img1, embt, scale_factor=1.0):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)
        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]
        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]
        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]
        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        up_flow0_1 = resize(up_flow0_1, scale_factor=1.0 / scale_factor) * (1.0 / scale_factor)
        up_flow1_1 = resize(up_flow1_1, scale_factor=1.0 / scale_factor) * (1.0 / scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=1.0 / scale_factor)
        up_res_1 = resize(up_res_1, scale_factor=1.0 / scale_factor)
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred

    def forward(self, img0, img1, embt, imgt, flow=None):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)
        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]
        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]
        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]
        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1))
        else:
            loss_dis = 0.0 * loss_geo
        return imgt_pred, loss_rec, loss_geo, loss_dis


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Charbonnier_Ada,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Charbonnier_L1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Geometry,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'side_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Ternary,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_ltkong218_IFRNet(_paritybench_base):
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

