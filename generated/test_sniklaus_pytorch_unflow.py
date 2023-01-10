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
        rbot0 = one.new_zeros([one.shape[0], one.shape[2] + 40, one.shape[3] + 40, one.shape[1]])
        rbot1 = one.new_zeros([one.shape[0], one.shape[2] + 40, one.shape[3] + 40, one.shape[1]])
        one = one.contiguous()
        two = two.contiguous()
        output = one.new_zeros([one.shape[0], 441, one.shape[2], one.shape[3]])
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


arguments_strModel = 'css'


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + 1.0 / tenFlow.shape[3], 1.0 - 1.0 / tenFlow.shape[3], tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + 1.0 / tenFlow.shape[2], 1.0 - 1.0 / tenFlow.shape[2], tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)


class Network(torch.nn.Module):

    def __init__(self):
        super().__init__()


        class Upconv(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.netSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.netFivNext = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.netFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.netFouNext = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.netFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.netThrNext = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.netThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                self.netTwoNext = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)
                self.netUpscale = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1, bias=False), torch.nn.ReplicationPad2d(padding=[0, 1, 0, 1]))

            def forward(self, tenOne, tenTwo, objInput):
                objOutput = {}
                tenInput = objInput['conv6']
                objOutput['flow6'] = self.netSixOut(tenInput)
                tenInput = torch.cat([objInput['conv5'], self.netFivNext(tenInput), self.netSixUp(objOutput['flow6'])], 1)
                objOutput['flow5'] = self.netFivOut(tenInput)
                tenInput = torch.cat([objInput['conv4'], self.netFouNext(tenInput), self.netFivUp(objOutput['flow5'])], 1)
                objOutput['flow4'] = self.netFouOut(tenInput)
                tenInput = torch.cat([objInput['conv3'], self.netThrNext(tenInput), self.netFouUp(objOutput['flow4'])], 1)
                objOutput['flow3'] = self.netThrOut(tenInput)
                tenInput = torch.cat([objInput['conv2'], self.netTwoNext(tenInput), self.netThrUp(objOutput['flow3'])], 1)
                objOutput['flow2'] = self.netTwoOut(tenInput)
                return self.netUpscale(self.netUpscale(objOutput['flow2'])) * 20.0


        class Complex(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(torch.nn.ZeroPad2d([2, 4, 2, 4]), torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.ZeroPad2d([1, 3, 1, 3]), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.ZeroPad2d([1, 3, 1, 3]), torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netRedir = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netCorrelation = correlation.ModuleCorrelation()
                self.netCombined = torch.nn.Sequential(torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netUpconv = Upconv()

            def forward(self, tenOne, tenTwo, tenFlow):
                objOutput = {}
                assert tenFlow is None
                objOutput['conv1'] = self.netOne(tenOne)
                objOutput['conv2'] = self.netTwo(objOutput['conv1'])
                objOutput['conv3'] = self.netThr(objOutput['conv2'])
                tenRedir = self.netRedir(objOutput['conv3'])
                tenOther = self.netThr(self.netTwo(self.netOne(tenTwo)))
                tenCorr = self.netCorrelation(objOutput['conv3'], tenOther)
                objOutput['conv3'] = self.netCombined(torch.cat([tenRedir, tenCorr], 1))
                objOutput['conv4'] = self.netFou(objOutput['conv3'])
                objOutput['conv5'] = self.netFiv(objOutput['conv4'])
                objOutput['conv6'] = self.netSix(objOutput['conv5'])
                return self.netUpconv(tenOne, tenTwo, objOutput)


        class Simple(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.netOne = torch.nn.Sequential(torch.nn.ZeroPad2d([2, 4, 2, 4]), torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = torch.nn.Sequential(torch.nn.ZeroPad2d([1, 3, 1, 3]), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = torch.nn.Sequential(torch.nn.ZeroPad2d([1, 3, 1, 3]), torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = torch.nn.Sequential(torch.nn.ZeroPad2d([0, 2, 0, 2]), torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1), torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netUpconv = Upconv()

            def forward(self, tenOne, tenTwo, tenFlow):
                objOutput = {}
                tenWarp = backwarp(tenInput=tenTwo, tenFlow=tenFlow)
                objOutput['conv1'] = self.netOne(torch.cat([tenOne, tenTwo, tenFlow, tenWarp, (tenOne - tenWarp).abs()], 1))
                objOutput['conv2'] = self.netTwo(objOutput['conv1'])
                objOutput['conv3'] = self.netThr(objOutput['conv2'])
                objOutput['conv4'] = self.netFou(objOutput['conv3'])
                objOutput['conv5'] = self.netFiv(objOutput['conv4'])
                objOutput['conv6'] = self.netSix(objOutput['conv5'])
                return self.netUpconv(tenOne, tenTwo, objOutput)
        self.netFlownets = torch.nn.ModuleList([Complex(), Simple(), Simple()])
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-unflow/network-' + arguments_strModel + '.pytorch', file_name='unflow-' + arguments_strModel).items()})

    def forward(self, tenOne, tenTwo):
        tenOne = tenOne[:, [2, 1, 0], :, :]
        tenTwo = tenTwo[:, [2, 1, 0], :, :]
        tenOne[:, 0, :, :] = tenOne[:, 0, :, :] - 104.920005 / 255.0
        tenOne[:, 1, :, :] = tenOne[:, 1, :, :] - 110.1753 / 255.0
        tenOne[:, 2, :, :] = tenOne[:, 2, :, :] - 114.785955 / 255.0
        tenTwo[:, 0, :, :] = tenTwo[:, 0, :, :] - 104.920005 / 255.0
        tenTwo[:, 1, :, :] = tenTwo[:, 1, :, :] - 110.1753 / 255.0
        tenTwo[:, 2, :, :] = tenTwo[:, 2, :, :] - 114.785955 / 255.0
        tenFlow = None
        for netFlownet in self.netFlownets:
            tenFlow = netFlownet(tenOne, tenTwo, tenFlow)
        return tenFlow

