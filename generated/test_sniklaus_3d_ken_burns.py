import sys
_module = sys.modules[__name__]
del sys
autozoom = _module
benchmark = _module
common = _module
depthestim = _module
interface = _module

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


import torch


import torchvision


import math


import numpy


import random


import re


import scipy


import scipy.io


import time


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Semantics(torch.nn.Module):

    def __init__(self):
        super(Semantics, self).__init__()
        moduleVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval()
        self.moduleVgg = torch.nn.Sequential(moduleVgg[0:3], moduleVgg[3:6], torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), moduleVgg[7:10], moduleVgg[10:13], torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), moduleVgg[14:17], moduleVgg[17:20], moduleVgg[20:23], moduleVgg[23:26], torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), moduleVgg[27:30], moduleVgg[30:33], moduleVgg[33:36], moduleVgg[36:39], torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

    def forward(self, tenInput):
        tenPreprocessed = tenInput[:, ([2, 1, 0]), :, :]
        tenPreprocessed[:, (0), :, :] = (tenPreprocessed[:, (0), :, :] - 0.485) / 0.229
        tenPreprocessed[:, (1), :, :] = (tenPreprocessed[:, (1), :, :] - 0.456) / 0.224
        tenPreprocessed[:, (2), :, :] = (tenPreprocessed[:, (2), :, :] - 0.406) / 0.225
        return self.moduleVgg(tenPreprocessed)


class Disparity(torch.nn.Module):

    def __init__(self):
        super(Disparity, self).__init__()
        self.moduleImage = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.moduleSemantics = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        for intRow, intFeatures in [(0, 32), (1, 48), (2, 64), (3, 512), (4, 512), (5, 512)]:
            self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
            self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
            self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
        for intCol in [0, 1]:
            self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([32, 48, 48]))
            self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([48, 64, 64]))
            self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([64, 512, 512]))
            self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol), Downsample([512, 512, 512]))
            self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol), Downsample([512, 512, 512]))
        for intCol in [2, 3]:
            self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol), Upsample([512, 512, 512]))
            self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol), Upsample([512, 512, 512]))
            self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([512, 64, 64]))
            self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([64, 48, 48]))
            self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([48, 32, 32]))
        self.moduleDisparity = Basic('conv-relu-conv', [32, 32, 1])

    def forward(self, tenImage, tenSemantics):
        tenColumn = [None, None, None, None, None, None]
        tenColumn[0] = self.moduleImage(tenImage)
        tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
        tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
        tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2]) + self.moduleSemantics(tenSemantics)
        tenColumn[4] = self._modules['3x0 - 4x0'](tenColumn[3])
        tenColumn[5] = self._modules['4x0 - 5x0'](tenColumn[4])
        intColumn = 1
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
        intColumn = 2
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        intColumn = 3
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        return torch.nn.functional.threshold(input=self.moduleDisparity(tenColumn[0]), threshold=0.0, value=0.0)


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Refine(torch.nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.moduleImageOne = Basic('conv-relu-conv', [3, 24, 24])
        self.moduleImageTwo = Downsample([24, 48, 48])
        self.moduleImageThr = Downsample([48, 96, 96])
        self.moduleDisparityOne = Basic('conv-relu-conv', [1, 96, 96])
        self.moduleDisparityTwo = Upsample([192, 96, 96])
        self.moduleDisparityThr = Upsample([144, 48, 48])
        self.moduleDisparityFou = Basic('conv-relu-conv', [72, 24, 24])
        self.moduleRefine = Basic('conv-relu-conv', [24, 24, 1])

    def forward(self, tenImage, tenDisparity):
        tenMean = [tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenStd = [tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenImage = tenImage.clone()
        tenImage -= tenMean[0]
        tenImage /= tenStd[0] + 1e-07
        tenDisparity = tenDisparity.clone()
        tenDisparity -= tenMean[1]
        tenDisparity /= tenStd[1] + 1e-07
        tenImageOne = self.moduleImageOne(tenImage)
        tenImageTwo = self.moduleImageTwo(tenImageOne)
        tenImageThr = self.moduleImageThr(tenImageTwo)
        tenUpsample = self.moduleDisparityOne(tenDisparity)
        if tenUpsample.shape != tenImageThr.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageThr.shape[2], tenImageThr.shape[3]), mode='bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityTwo(torch.cat([tenImageThr, tenUpsample], 1))
        tenImageThr = None
        if tenUpsample.shape != tenImageTwo.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageTwo.shape[2], tenImageTwo.shape[3]), mode='bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityThr(torch.cat([tenImageTwo, tenUpsample], 1))
        tenImageTwo = None
        if tenUpsample.shape != tenImageOne.shape:
            tenUpsample = torch.nn.functional.interpolate(input=tenUpsample, size=(tenImageOne.shape[2], tenImageOne.shape[3]), mode='bilinear', align_corners=False)
        tenUpsample = self.moduleDisparityFou(torch.cat([tenImageOne, tenUpsample], 1))
        tenImageOne = None
        tenRefine = self.moduleRefine(tenUpsample)
        tenRefine *= tenStd[1] + 1e-07
        tenRefine += tenMean[1]
        return torch.nn.functional.threshold(input=tenRefine, threshold=0.0, value=0.0)


class Basic(torch.nn.Module):

    def __init__(self, strType, intChannels):
        super(Basic, self).__init__()
        if strType == 'relu-conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        elif strType == 'conv-relu-conv':
            self.moduleMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))
        if intChannels[0] == intChannels[2]:
            self.moduleShortcut = None
        elif intChannels[0] != intChannels[2]:
            self.moduleShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

    def forward(self, tenInput):
        if self.moduleShortcut is None:
            return self.moduleMain(tenInput) + tenInput
        elif self.moduleShortcut is not None:
            return self.moduleMain(tenInput) + self.moduleShortcut(tenInput)


class Downsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Downsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


class Upsample(torch.nn.Module):

    def __init__(self, intChannels):
        super(Upsample, self).__init__()
        self.moduleMain = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), torch.nn.PReLU(num_parameters=intChannels[0], init=0.25), torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1), torch.nn.PReLU(num_parameters=intChannels[1], init=0.25), torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1))

    def forward(self, tenInput):
        return self.moduleMain(tenInput)


def depth_to_points(tenDepth, fltFocal):
    tenHorizontal = torch.linspace(-0.5 * tenDepth.shape[3] + 0.5, 0.5 * tenDepth.shape[3] - 0.5, tenDepth.shape[3]).view(1, 1, 1, tenDepth.shape[3]).expand(tenDepth.shape[0], -1, tenDepth.shape[2], -1)
    tenHorizontal = tenHorizontal * (1.0 / fltFocal)
    tenHorizontal = tenHorizontal.type_as(tenDepth)
    tenVertical = torch.linspace(-0.5 * tenDepth.shape[2] + 0.5, 0.5 * tenDepth.shape[2] - 0.5, tenDepth.shape[2]).view(1, 1, tenDepth.shape[2], 1).expand(tenDepth.shape[0], -1, -1, tenDepth.shape[3])
    tenVertical = tenVertical * (1.0 / fltFocal)
    tenVertical = tenVertical.type_as(tenDepth)
    return torch.cat([tenDepth * tenHorizontal, tenDepth * tenVertical, tenDepth], 1)


objCommon = {}


def preprocess_kernel(strKernel, objVariables):
    with open('./common.cuda', 'r') as objFile:
        strKernel = objFile.read() + strKernel
    for strVariable in objVariables:
        objValue = objVariables[strVariable]
        if type(objValue) == int:
            strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))
        elif type(objValue) == float:
            strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))
        elif type(objValue) == str:
            strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)
    while True:
        objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    while True:
        objMatch = re.search('(STRIDE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intStrides = objVariables[strTensor].stride()
        strKernel = strKernel.replace(objMatch.group(), str(intStrides[intArg]))
    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    return strKernel


def render_pointcloud(tenInput, tenData, intWidth, intHeight, fltFocal, fltBaseline):
    tenData = torch.cat([tenData, tenData.new_ones([tenData.shape[0], 1, tenData.shape[2]])], 1)
    tenZee = tenInput.new_zeros([tenData.shape[0], 1, intHeight, intWidth]).fill_(1000000.0)
    tenOutput = tenInput.new_zeros([tenData.shape[0], tenData.shape[1], intHeight, intWidth])
    n = tenInput.shape[0] * tenInput.shape[2]
    launch_kernel('kernel_pointrender_updateZee', preprocess_kernel('\n\t\textern "C" __global__ void kernel_pointrender_updateZee(\n\t\t\tconst int n,\n\t\t\tconst float* input,\n\t\t\tconst float* data,\n\t\t\tconst float* zee\n\t\t) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {\n\t\t\tconst int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);\n\t\t\tconst int intPoint  = ( intIndex                 ) % SIZE_2(input);\n\n\t\t\tassert(SIZE_1(input) == 3);\n\t\t\tassert(SIZE_1(zee) == 1);\n\n\t\t\tfloat3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});\n\t\t\tfloat3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);\n\n\t\t\tfloat3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));\n\t\t\tfloat3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;\n\n\t\t\tif (fltLinePoint.z < 0.001) {\n\t\t\t\treturn;\n\t\t\t}\n\n\t\t\tfloat fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);\n\t\t\tfloat fltDenominator = dot(fltLineVector, fltPlaneNormal);\n\t\t\tfloat fltDistance = fltNumerator / fltDenominator;\n\n\t\t\tif (fabs(fltDenominator) < 0.001) {\n\t\t\t\treturn;\n\t\t\t}\n\n\t\t\tfloat3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection\n\n\t\t\tfloat fltOutputX = fltIntersection.x + (0.5 * SIZE_3(zee)) - 0.5;\n\t\t\tfloat fltOutputY = fltIntersection.y + (0.5 * SIZE_2(zee)) - 0.5;\n\n\t\t\tfloat fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));\n\n\t\t\tint intNorthwestX = (int) (floor(fltOutputX));\n\t\t\tint intNorthwestY = (int) (floor(fltOutputY));\n\t\t\tint intNortheastX = intNorthwestX + 1;\n\t\t\tint intNortheastY = intNorthwestY;\n\t\t\tint intSouthwestX = intNorthwestX;\n\t\t\tint intSouthwestY = intNorthwestY + 1;\n\t\t\tint intSoutheastX = intNorthwestX + 1;\n\t\t\tint intSoutheastY = intNorthwestY + 1;\n\n\t\t\tfloat fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);\n\t\t\tfloat fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);\n\t\t\tfloat fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);\n\t\t\tfloat fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);\n\n\t\t\tif ((fltNorthwest >= fltNortheast) & (fltNorthwest >= fltSouthwest) & (fltNorthwest >= fltSoutheast)) {\n\t\t\t\tif ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(zee)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(zee))) {\n\t\t\t\t\tatomicMin(&zee[OFFSET_4(zee, intSample, 0, intNorthwestY, intNorthwestX)], fltError);\n\t\t\t\t}\n\n\t\t\t} else if ((fltNortheast >= fltNorthwest) & (fltNortheast >= fltSouthwest) & (fltNortheast >= fltSoutheast)) {\n\t\t\t\tif ((intNortheastX >= 0) & (intNortheastX < SIZE_3(zee)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(zee))) {\n\t\t\t\t\tatomicMin(&zee[OFFSET_4(zee, intSample, 0, intNortheastY, intNortheastX)], fltError);\n\t\t\t\t}\n\n\t\t\t} else if ((fltSouthwest >= fltNorthwest) & (fltSouthwest >= fltNortheast) & (fltSouthwest >= fltSoutheast)) {\n\t\t\t\tif ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(zee)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(zee))) {\n\t\t\t\t\tatomicMin(&zee[OFFSET_4(zee, intSample, 0, intSouthwestY, intSouthwestX)], fltError);\n\t\t\t\t}\n\n\t\t\t} else if ((fltSoutheast >= fltNorthwest) & (fltSoutheast >= fltNortheast) & (fltSoutheast >= fltSouthwest)) {\n\t\t\t\tif ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(zee)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(zee))) {\n\t\t\t\t\tatomicMin(&zee[OFFSET_4(zee, intSample, 0, intSoutheastY, intSoutheastX)], fltError);\n\t\t\t\t}\n\n\t\t\t}\n\t\t} }\n\t', {'intWidth': intWidth, 'intHeight': intHeight, 'fltFocal': fltFocal, 'fltBaseline': fltBaseline, 'input': tenInput, 'data': tenData, 'zee': tenZee}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr()])
    n = tenZee.nelement()
    launch_kernel('kernel_pointrender_updateDegrid', preprocess_kernel('\n\t\textern "C" __global__ void kernel_pointrender_updateDegrid(\n\t\t\tconst int n,\n\t\t\tconst float* input,\n\t\t\tconst float* data,\n\t\t\tfloat* zee\n\t\t) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {\n\t\t\tconst int intN = ( intIndex / SIZE_3(zee) / SIZE_2(zee) / SIZE_1(zee) ) % SIZE_0(zee);\n\t\t\tconst int intC = ( intIndex / SIZE_3(zee) / SIZE_2(zee)               ) % SIZE_1(zee);\n\t\t\tconst int intY = ( intIndex / SIZE_3(zee)                             ) % SIZE_2(zee);\n\t\t\tconst int intX = ( intIndex                                           ) % SIZE_3(zee);\n\n\t\t\tassert(SIZE_1(input) == 3);\n\t\t\tassert(SIZE_1(zee) == 1);\n\n\t\t\tint intCount = 0;\n\t\t\tfloat fltSum = 0.0;\n\n\t\t\tint intOpposingX[] = {  1,  0,  1,  1 };\n\t\t\tint intOpposingY[] = {  0,  1,  1, -1 };\n\n\t\t\tfor (int intOpposing = 0; intOpposing < 4; intOpposing += 1) {\n\t\t\t\tint intOneX = intX + intOpposingX[intOpposing];\n\t\t\t\tint intOneY = intY + intOpposingY[intOpposing];\n\t\t\t\tint intTwoX = intX - intOpposingX[intOpposing];\n\t\t\t\tint intTwoY = intY - intOpposingY[intOpposing];\n\n\t\t\t\tif ((intOneX < 0) | (intOneX >= SIZE_3(zee)) | (intOneY < 0) | (intOneY >= SIZE_2(zee))) {\n\t\t\t\t\tcontinue;\n\n\t\t\t\t} else if ((intTwoX < 0) | (intTwoX >= SIZE_3(zee)) | (intTwoY < 0) | (intTwoY >= SIZE_2(zee))) {\n\t\t\t\t\tcontinue;\n\n\t\t\t\t}\n\n\t\t\t\tif (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intOneY, intOneX) + 1.0) {\n\t\t\t\t\tif (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intTwoY, intTwoX) + 1.0) {\n\t\t\t\t\t\tintCount += 2;\n\t\t\t\t\t\tfltSum += VALUE_4(zee, intN, intC, intOneY, intOneX);\n\t\t\t\t\t\tfltSum += VALUE_4(zee, intN, intC, intTwoY, intTwoX);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tif (intCount > 0) {\n\t\t\t\tzee[OFFSET_4(zee, intN, intC, intY, intX)] = min(VALUE_4(zee, intN, intC, intY, intX), fltSum / intCount);\n\t\t\t}\n\t\t} }\n\t', {'intWidth': intWidth, 'intHeight': intHeight, 'fltFocal': fltFocal, 'fltBaseline': fltBaseline, 'input': tenInput, 'data': tenData, 'zee': tenZee}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr()])
    n = tenInput.shape[0] * tenInput.shape[2]
    launch_kernel('kernel_pointrender_updateOutput', preprocess_kernel('\n\t\textern "C" __global__ void kernel_pointrender_updateOutput(\n\t\t\tconst int n,\n\t\t\tconst float* input,\n\t\t\tconst float* data,\n\t\t\tconst float* zee,\n\t\t\tfloat* output\n\t\t) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {\n\t\t\tconst int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);\n\t\t\tconst int intPoint  = ( intIndex                 ) % SIZE_2(input);\n\n\t\t\tassert(SIZE_1(input) == 3);\n\t\t\tassert(SIZE_1(zee) == 1);\n\n\t\t\tfloat3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});\n\t\t\tfloat3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);\n\n\t\t\tfloat3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));\n\t\t\tfloat3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;\n\n\t\t\tif (fltLinePoint.z < 0.001) {\n\t\t\t\treturn;\n\t\t\t}\n\n\t\t\tfloat fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);\n\t\t\tfloat fltDenominator = dot(fltLineVector, fltPlaneNormal);\n\t\t\tfloat fltDistance = fltNumerator / fltDenominator;\n\n\t\t\tif (fabs(fltDenominator) < 0.001) {\n\t\t\t\treturn;\n\t\t\t}\n\n\t\t\tfloat3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection\n\n\t\t\tfloat fltOutputX = fltIntersection.x + (0.5 * SIZE_3(output)) - 0.5;\n\t\t\tfloat fltOutputY = fltIntersection.y + (0.5 * SIZE_2(output)) - 0.5;\n\n\t\t\tfloat fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));\n\n\t\t\tint intNorthwestX = (int) (floor(fltOutputX));\n\t\t\tint intNorthwestY = (int) (floor(fltOutputY));\n\t\t\tint intNortheastX = intNorthwestX + 1;\n\t\t\tint intNortheastY = intNorthwestY;\n\t\t\tint intSouthwestX = intNorthwestX;\n\t\t\tint intSouthwestY = intNorthwestY + 1;\n\t\t\tint intSoutheastX = intNorthwestX + 1;\n\t\t\tint intSoutheastY = intNorthwestY + 1;\n\n\t\t\tfloat fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);\n\t\t\tfloat fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);\n\t\t\tfloat fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);\n\t\t\tfloat fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);\n\n\t\t\tif ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {\n\t\t\t\tif (fltError <= VALUE_4(zee, intSample, 0, intNorthwestY, intNorthwestX) + 1.0) {\n\t\t\t\t\tfor (int intData = 0; intData < SIZE_1(data); intData += 1) {\n\t\t\t\t\t\tatomicAdd(&output[OFFSET_4(output, intSample, intData, intNorthwestY, intNorthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltNorthwest);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tif ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {\n\t\t\t\tif (fltError <= VALUE_4(zee, intSample, 0, intNortheastY, intNortheastX) + 1.0) {\n\t\t\t\t\tfor (int intData = 0; intData < SIZE_1(data); intData += 1) {\n\t\t\t\t\t\tatomicAdd(&output[OFFSET_4(output, intSample, intData, intNortheastY, intNortheastX)], VALUE_3(data, intSample, intData, intPoint) * fltNortheast);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tif ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {\n\t\t\t\tif (fltError <= VALUE_4(zee, intSample, 0, intSouthwestY, intSouthwestX) + 1.0) {\n\t\t\t\t\tfor (int intData = 0; intData < SIZE_1(data); intData += 1) {\n\t\t\t\t\t\tatomicAdd(&output[OFFSET_4(output, intSample, intData, intSouthwestY, intSouthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltSouthwest);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\n\t\t\tif ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {\n\t\t\t\tif (fltError <= VALUE_4(zee, intSample, 0, intSoutheastY, intSoutheastX) + 1.0) {\n\t\t\t\t\tfor (int intData = 0; intData < SIZE_1(data); intData += 1) {\n\t\t\t\t\t\tatomicAdd(&output[OFFSET_4(output, intSample, intData, intSoutheastY, intSoutheastX)], VALUE_3(data, intSample, intData, intPoint) * fltSoutheast);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t} }\n\t', {'intWidth': intWidth, 'intHeight': intHeight, 'fltFocal': fltFocal, 'fltBaseline': fltBaseline, 'input': tenInput, 'data': tenData, 'zee': tenZee, 'output': tenOutput}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr(), tenOutput.data_ptr()])
    return tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 1e-07), tenOutput[:, -1:, :, :].detach().clone()


def spatial_filter(tenInput, strType):
    tenOutput = None
    if strType == 'laplacian':
        tenLaplacian = tenInput.new_zeros(tenInput.shape[1], tenInput.shape[1], 3, 3)
        for intKernel in range(tenInput.shape[1]):
            tenLaplacian[intKernel, intKernel, 0, 1] = -1.0
            tenLaplacian[intKernel, intKernel, 0, 2] = -1.0
            tenLaplacian[intKernel, intKernel, 1, 1] = 4.0
            tenLaplacian[intKernel, intKernel, 1, 0] = -1.0
            tenLaplacian[intKernel, intKernel, 2, 0] = -1.0
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[1, 1, 1, 1], mode='replicate')
        tenOutput = torch.nn.functional.conv2d(input=tenOutput, weight=tenLaplacian)
    elif strType == 'median-3':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[1, 1, 1, 1], mode='reflect')
        tenOutput = tenOutput.unfold(2, 3, 1).unfold(3, 3, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 3 * 3)
        tenOutput = tenOutput.median(-1, False)[0]
    elif strType == 'median-5':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[2, 2, 2, 2], mode='reflect')
        tenOutput = tenOutput.unfold(2, 5, 1).unfold(3, 5, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 5 * 5)
        tenOutput = tenOutput.median(-1, False)[0]
    return tenOutput


class Inpaint(torch.nn.Module):

    def __init__(self):
        super(Inpaint, self).__init__()
        self.moduleContext = torch.nn.Sequential(torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True), torch.nn.PReLU(num_parameters=64, init=0.25), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True), torch.nn.PReLU(num_parameters=64, init=0.25))
        self.moduleInput = Basic('conv-relu-conv', [3 + 1 + 64 + 1, 32, 32])
        for intRow, intFeatures in [(0, 32), (1, 64), (2, 128), (3, 256)]:
            self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
            self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
            self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures]))
        for intCol in [0, 1]:
            self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([32, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([64, 128, 128]))
            self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([128, 256, 256]))
        for intCol in [2, 3]:
            self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([256, 128, 128]))
            self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([128, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([64, 32, 32]))
        self.moduleImage = Basic('conv-relu-conv', [32, 32, 3])
        self.moduleDisparity = Basic('conv-relu-conv', [32, 32, 1])

    def forward(self, tenImage, tenDisparity, tenShift):
        tenDepth = objCommon['fltFocal'] * objCommon['fltBaseline'] / (tenDisparity + 1e-07)
        tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
        tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
        tenPoints = tenPoints.view(1, 3, -1)
        tenMean = [tenImage.view(tenImage.shape[0], -1).mean(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).mean(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenStd = [tenImage.view(tenImage.shape[0], -1).std(1, True).view(tenImage.shape[0], 1, 1, 1), tenDisparity.view(tenDisparity.shape[0], -1).std(1, True).view(tenDisparity.shape[0], 1, 1, 1)]
        tenImage = tenImage.clone()
        tenImage -= tenMean[0]
        tenImage /= tenStd[0] + 1e-07
        tenDisparity = tenDisparity.clone()
        tenDisparity -= tenMean[1]
        tenDisparity /= tenStd[1] + 1e-07
        tenContext = self.moduleContext(torch.cat([tenImage, tenDisparity], 1))
        tenRender, tenExisting = render_pointcloud(tenPoints + tenShift, torch.cat([tenImage, tenDisparity, tenContext], 1).view(1, 68, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])
        tenExisting = (tenExisting > 0.0).float()
        tenExisting = tenExisting * spatial_filter(tenExisting, 'median-5')
        tenRender = tenRender * tenExisting.clone().detach()
        tenColumn = [None, None, None, None]
        tenColumn[0] = self.moduleInput(torch.cat([tenRender, tenExisting], 1))
        tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
        tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
        tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])
        intColumn = 1
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
        intColumn = 2
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        intColumn = 3
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)
                tenColumn[intRow] += tenUp
        tenImage = self.moduleImage(tenColumn[0])
        tenImage *= tenStd[0] + 1e-07
        tenImage += tenMean[0]
        tenDisparity = self.moduleDisparity(tenColumn[0])
        tenDisparity *= tenStd[1] + 1e-07
        tenDisparity += tenMean[1]
        return {'tenExisting': tenExisting, 'tenImage': tenImage.clamp(0.0, 1.0) if self.training == False else tenImage, 'tenDisparity': torch.nn.functional.threshold(input=tenDisparity, threshold=0.0, value=0.0)}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Disparity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 512, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'intChannels': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Refine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 1, 64, 64])], {}),
     True),
    (Semantics,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'intChannels': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sniklaus_3d_ken_burns(_paritybench_base):
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

