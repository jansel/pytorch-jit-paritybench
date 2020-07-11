import sys
_module = sys.modules[__name__]
del sys
SpatialCrossMapLRN_temp = _module
loadOpenFace = _module

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


import numpy


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


from collections import OrderedDict


import time


class SpatialCrossMapLRN_temp(Module):

    def __init__(self, size, alpha=0.0001, beta=0.75, k=1, gpuDevice=0):
        super(SpatialCrossMapLRN_temp, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.scale = None
        self.paddedRatio = None
        self.accumRatio = None
        self.gpuDevice = gpuDevice

    def updateOutput(self, input):
        assert input.dim() == 4
        if self.scale is None:
            self.scale = input.new()
        if self.output is None:
            self.output = input.new()
        batchSize = input.size(0)
        channels = input.size(1)
        inputHeight = input.size(2)
        inputWidth = input.size(3)
        if input.is_cuda:
            self.output = self.output
            self.scale = self.scale
        self.output.resize_as_(input)
        self.scale.resize_as_(input)
        inputSquare = self.output
        torch.pow(input, 2, out=inputSquare)
        prePad = int((self.size - 1) / 2 + 1)
        prePadCrop = channels if prePad > channels else prePad
        scaleFirst = self.scale.select(1, 0)
        scaleFirst.zero_()
        for c in range(prePadCrop):
            scaleFirst.add_(inputSquare.select(1, c))
        for c in range(1, channels):
            scalePrevious = self.scale.select(1, c - 1)
            scaleCurrent = self.scale.select(1, c)
            scaleCurrent.copy_(scalePrevious)
            if c < channels - prePad + 1:
                squareNext = inputSquare.select(1, c + prePad - 1)
                scaleCurrent.add_(1, squareNext)
            if c > prePad:
                squarePrevious = inputSquare.select(1, c - prePad)
                scaleCurrent.add_(-1, squarePrevious)
        self.scale.mul_(self.alpha / self.size).add_(self.k)
        torch.pow(self.scale, -self.beta, out=self.output)
        self.output.mul_(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 4
        batchSize = input.size(0)
        channels = input.size(1)
        inputHeight = input.size(2)
        inputWidth = input.size(3)
        if self.paddedRatio is None:
            self.paddedRatio = input.new()
        if self.accumRatio is None:
            self.accumRatio = input.new()
        self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
        self.accumRatio.resize_(inputHeight, inputWidth)
        cacheRatioValue = 2 * self.alpha * self.beta / self.size
        inversePrePad = int(self.size - (self.size - 1) / 2)
        self.gradInput.resize_as_(input)
        torch.pow(self.scale, -self.beta, out=self.gradInput).mul_(gradOutput)
        self.paddedRatio.zero_()
        paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
        for n in range(batchSize):
            torch.mul(gradOutput[n], self.output[n], out=paddedRatioCenter)
            paddedRatioCenter.div_(self.scale[n])
            torch.sum(self.paddedRatio.narrow(0, 0, self.size - 1), 0, out=self.accumRatio)
            for c in range(channels):
                self.accumRatio.add_(self.paddedRatio[c + self.size - 1])
                self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
                self.accumRatio.add_(-1, self.paddedRatio[c])
        return self.gradInput

    def clearState(self):
        clear(self, 'scale', 'paddedRatio', 'accumRatio')
        return super(SpatialCrossMapLRN_temp, self).clearState()


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


def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l


def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l


class Inception(nn.Module):

    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm, reduceStride=None, padding=True):
        super(Inception, self).__init__()
        self.seq_list = []
        self.outputSize = outputSize
        for i in range(len(kernelSize)):
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0, 0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))
        ii = len(kernelSize)
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0, 0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        self.seq_list.append(nn.Sequential(od))
        ii += 1
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0, 0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))
        self.seq_list = nn.ModuleList(self.seq_list)

    def forward(self, input):
        x = input
        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            y = seq(x)
            y_size = y.size()
            ys.append(y)
            if target_size is None:
                target_size = [0] * len(y_size)
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]
        target_size[1] = depth_dim
        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))
        output = torch.cat(ys, 1)
        return output


def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    if SpatialCrossMapLRN_temp is not None:
        lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k, gpuDevice=gpuDevice)
        n = Lambda(lambda x, lrn=lrn: Variable(lrn.forward(x.data)) if x.data.is_cuda else Variable(lrn.forward(x.data)))
    else:
        n = nn.LocalResponseNorm(size, alpha, beta, k)
    return n


def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l


class netOpenFace(nn.Module):

    def __init__(self, useCuda, gpuDevice=0):
        super(netOpenFace, self).__init__()
        self.gpuDevice = gpuDevice
        self.layer1 = Conv2d(3, 64, (7, 7), (2, 2), (3, 3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer6 = Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3, 3), (1, 1), (1, 1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer13 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        self.layer14 = Inception(192, (3, 5), (1, 1), (128, 32), (96, 16, 32, 64), nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer15 = Inception(256, (3, 5), (1, 1), (128, 64), (96, 32, 64, 64), nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer16 = Inception(320, (3, 5), (2, 2), (256, 64), (128, 32, None, None), nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer17 = Inception(640, (3, 5), (1, 1), (192, 64), (96, 32, 128, 256), nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer18 = Inception(640, (3, 5), (2, 2), (256, 128), (160, 64, None, None), nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96, 96, 256), nn.LPPool2d(2, (3, 3), stride=(3, 3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96, 96, 256), nn.MaxPool2d((3, 3), stride=(2, 2), padding=(0, 0)), True)
        self.layer22 = nn.AvgPool2d((3, 3), stride=(1, 1), padding=(0, 0))
        self.layer25 = Linear(736, 128)
        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)
        if useCuda:
            self

    def forward(self, input):
        x = input
        if x.data.is_cuda and self.gpuDevice != 0:
            x = x
        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))
        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))
        x_736 = x
        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x ** 2, 1) + 1e-06)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))
        return x, x_736


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_thnkim_OpenFacePytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

