import sys
_module = sys.modules[__name__]
del sys
comparison = _module
correlation = _module
run = _module

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


import re


import torch


import math


import numpy


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]
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
    def forward(self, one, two):
        rbot0 = one.new_zeros([one.shape[0], one.shape[2] + 8, one.shape[3] + 8, one.shape[1]])
        rbot1 = one.new_zeros([one.shape[0], one.shape[2] + 8, one.shape[3] + 8, one.shape[1]])
        one = one.contiguous()
        two = two.contiguous()
        output = one.new_zeros([one.shape[0], 81, one.shape[2], one.shape[3]])
        if one.is_cuda == True:
            n = one.shape[2] * one.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'input': one, 'output': rbot0}))(grid=tuple([int((n + 16 - 1) / 16), one.shape[1], one.shape[0]]), block=tuple([16, 1, 1]), args=[cupy.int32(n), one.data_ptr(), rbot0.data_ptr()])
            n = two.shape[2] * two.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {'input': two, 'output': rbot1}))(grid=tuple([int((n + 16 - 1) / 16), two.shape[1], two.shape[0]]), block=tuple([16, 1, 1]), args=[cupy.int32(n), two.data_ptr(), rbot1.data_ptr()])
            n = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {'rbot0': rbot0, 'rbot1': rbot1, 'top': output}))(grid=tuple([output.shape[3], output.shape[2], output.shape[0]]), block=tuple([32, 1, 1]), shared_mem=one.shape[1] * 4, args=[cupy.int32(n), rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()])
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
                    cupy_launch('kernel_Correlation_updateGradOne', cupy_kernel('kernel_Correlation_updateGradOne', {'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradOne': gradOne, 'gradTwo': None}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), gradOne.data_ptr(), None])
            if gradTwo is not None:
                for intSample in range(one.shape[0]):
                    n = one.shape[1] * one.shape[2] * one.shape[3]
                    cupy_launch('kernel_Correlation_updateGradTwo', cupy_kernel('kernel_Correlation_updateGradTwo', {'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput': gradOutput, 'gradOne': None, 'gradTwo': gradTwo}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[cupy.int32(n), intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None, gradTwo.data_ptr()])
        elif one.is_cuda == False:
            raise NotImplementedError()
        return gradOne, gradTwo


class ModuleCorrelation(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, tenOne, tenTwo):
        return _FunctionCorrelation.apply(tenOne, tenTwo)


arguments_strModel = 'default'


backwarp_tenGrid = {}


backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + 1.0 / tenFlow.shape[3], 1.0 - 1.0 / tenFlow.shape[3], tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + 1.0 / tenFlow.shape[2], 1.0 - 1.0 / tenFlow.shape[2], tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1)
    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)
    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0
    return tenOutput[:, :-1, :, :] * tenMask


class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()


        class Extractor(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


        class Decoder(torch.nn.Module):

            def __init__(self, intLevel):
                super().__init__()
                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]
                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6:
                    self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6:
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1))

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None
                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None
                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)
                    tenFeat = torch.cat([tenVolume], 1)
                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])
                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)
                    tenFeat = torch.cat([tenVolume, tenOne, tenFlow, tenFeat], 1)
                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)
                tenFlow = self.netSix(tenFeat)
                return {'tenFlow': tenFlow, 'tenFeat': tenFeat}


        class Refiner(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1))

            def forward(self, tenInput):
                return self.netMain(tenInput)
        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.netRefiner = Refiner()
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items()})

    def forward(self, tenOne, tenTwo):
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)
        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)
        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0

