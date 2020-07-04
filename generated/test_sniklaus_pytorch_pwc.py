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
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import re


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
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace(
            '}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for
            intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' +
            str.join('+', strIndex) + ']')
    return strKernel


class _FunctionCorrelation(torch.autograd.Function):

    @staticmethod
    def forward(self, first, second):
        rbot0 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.
            shape[3] + 8, first.shape[1]])
        rbot1 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.
            shape[3] + 8, first.shape[1]])
        self.save_for_backward(first, second, rbot0, rbot1)
        assert first.is_contiguous() == True
        assert second.is_contiguous() == True
        output = first.new_zeros([first.shape[0], 81, first.shape[2], first
            .shape[3]])
        if first.is_cuda == True:
            n = first.shape[2] * first.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel(
                'kernel_Correlation_rearrange', {'input': first, 'output':
                rbot0}))(grid=tuple([int((n + 16 - 1) / 16), first.shape[1],
                first.shape[0]]), block=tuple([16, 1, 1]), args=[n, first.
                data_ptr(), rbot0.data_ptr()])
            n = second.shape[2] * second.shape[3]
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel(
                'kernel_Correlation_rearrange', {'input': second, 'output':
                rbot1}))(grid=tuple([int((n + 16 - 1) / 16), second.shape[1
                ], second.shape[0]]), block=tuple([16, 1, 1]), args=[n,
                second.data_ptr(), rbot1.data_ptr()])
            n = output.shape[1] * output.shape[2] * output.shape[3]
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel(
                'kernel_Correlation_updateOutput', {'rbot0': rbot0, 'rbot1':
                rbot1, 'top': output}))(grid=tuple([output.shape[3], output
                .shape[2], output.shape[0]]), block=tuple([32, 1, 1]),
                shared_mem=first.shape[1] * 4, args=[n, rbot0.data_ptr(),
                rbot1.data_ptr(), output.data_ptr()])
        elif first.is_cuda == False:
            raise NotImplementedError()
        return output

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors
        assert gradOutput.is_contiguous() == True
        gradFirst = first.new_zeros([first.shape[0], first.shape[1], first.
            shape[2], first.shape[3]]) if self.needs_input_grad[0
            ] == True else None
        gradSecond = first.new_zeros([first.shape[0], first.shape[1], first
            .shape[2], first.shape[3]]) if self.needs_input_grad[1
            ] == True else None
        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.shape[0]):
                    n = first.shape[1] * first.shape[2] * first.shape[3]
                    cupy_launch('kernel_Correlation_updateGradFirst',
                        cupy_kernel('kernel_Correlation_updateGradFirst', {
                        'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput':
                        gradOutput, 'gradFirst': gradFirst, 'gradSecond':
                        None}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]
                        ), block=tuple([512, 1, 1]), args=[n, intSample,
                        rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.
                        data_ptr(), gradFirst.data_ptr(), None])
            if gradSecond is not None:
                for intSample in range(first.shape[0]):
                    n = first.shape[1] * first.shape[2] * first.shape[3]
                    cupy_launch('kernel_Correlation_updateGradSecond',
                        cupy_kernel('kernel_Correlation_updateGradSecond',
                        {'rbot0': rbot0, 'rbot1': rbot1, 'gradOutput':
                        gradOutput, 'gradFirst': None, 'gradSecond':
                        gradSecond}))(grid=tuple([int((n + 512 - 1) / 512),
                        1, 1]), block=tuple([512, 1, 1]), args=[n,
                        intSample, rbot0.data_ptr(), rbot1.data_ptr(),
                        gradOutput.data_ptr(), None, gradSecond.data_ptr()])
        elif first.is_cuda == False:
            raise NotImplementedError()
        return gradFirst, gradSecond


class ModuleCorrelation(torch.nn.Module):

    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tenFirst, tenSecond):
        return _FunctionCorrelation.apply(tenFirst, tenSecond)


arguments_strModel = 'default'


backwarp_tenGrid = {}


backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1,
            1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.
            shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1,
            tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.
            shape[3])
        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal,
            tenVertical], 1).cuda()
    if str(tenFlow.size()) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.size())] = tenFlow.new_ones([
            tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) /
        2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.size())
        ]], 1)
    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(
        backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1
        ), mode='bilinear', padding_mode='zeros', align_corners=True)
    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0
    return tenOutput[:, :-1, :, :] * tenMask


class Network(torch.nn.Module):

    def __init__(self):
        super(Network, self).__init__()


        class Extractor(torch.nn.Module):

            def __init__(self):
                super(Extractor, self).__init__()
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=2,
                    padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=16,
                    out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=32,
                    out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=64,
                    out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=64, out_channels=96, kernel_size=3, stride=
                    2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=96,
                    out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=96, out_channels=128, kernel_size=3, stride
                    =2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=128,
                    out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=128, out_channels=196, kernel_size=3,
                    stride=2, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=196,
                    out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))

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
                super(Decoder, self).__init__()
                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]
                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2,
                        out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6:
                    self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=
                        intPrevious + 128 + 128 + 96 + 64 + 32,
                        out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6:
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 
                        0.625, None][intLevel + 1]
                self.netOne = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent, out_channels=128, kernel_size=3,
                    stride=1, padding=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent + 128, out_channels=128,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent + 128 + 128, out_channels=96,
                    kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU
                    (inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent + 128 + 128 + 96, out_channels=
                    64, kernel_size=3, stride=1, padding=1), torch.nn.
                    LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent + 128 + 128 + 96 + 64,
                    out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                    out_channels=2, kernel_size=3, stride=1, padding=1))

            def forward(self, tenFirst, tenSecond, objPrevious):
                tenFlow = None
                tenFeat = None
                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None
                    tenVolume = torch.nn.functional.leaky_relu(input=
                        correlation.FunctionCorrelation(tenFirst=tenFirst,
                        tenSecond=tenSecond), negative_slope=0.1, inplace=False
                        )
                    tenFeat = torch.cat([tenVolume], 1)
                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])
                    tenVolume = torch.nn.functional.leaky_relu(input=
                        correlation.FunctionCorrelation(tenFirst=tenFirst,
                        tenSecond=backwarp(tenInput=tenSecond, tenFlow=
                        tenFlow * self.fltBackwarp)), negative_slope=0.1,
                        inplace=False)
                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow,
                        tenFeat], 1)
                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)
                tenFlow = self.netSix(tenFeat)
                return {'tenFlow': tenFlow, 'tenFeat': tenFeat}


        class Refiner(torch.nn.Module):

            def __init__(self):
                super(Refiner, self).__init__()
                self.netMain = torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                    out_channels=128, kernel_size=3, stride=1, padding=1,
                    dilation=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=128,
                    out_channels=128, kernel_size=3, stride=1, padding=2,
                    dilation=2), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=128,
                    out_channels=128, kernel_size=3, stride=1, padding=4,
                    dilation=4), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=128,
                    out_channels=96, kernel_size=3, stride=1, padding=8,
                    dilation=8), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=96,
                    out_channels=64, kernel_size=3, stride=1, padding=16,
                    dilation=16), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=64,
                    out_channels=32, kernel_size=3, stride=1, padding=1,
                    dilation=1), torch.nn.LeakyReLU(inplace=False,
                    negative_slope=0.1), torch.nn.Conv2d(in_channels=32,
                    out_channels=2, kernel_size=3, stride=1, padding=1,
                    dilation=1))

            def forward(self, tenInput):
                return self.netMain(tenInput)
        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.netRefiner = Refiner()
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for
            strKey, tenWeight in torch.load(__file__.replace('run.py', 
            'network-' + arguments_strModel + '.pytorch')).items()})

    def forward(self, tenFirst, tenSecond):
        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)
        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)
        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sniklaus_pytorch_pwc(_paritybench_base):
    pass
