import sys
_module = sys.modules[__name__]
del sys
run = _module
softsplat = _module

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


import torch


import numpy


import re


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


class _FunctionSoftsplat(torch.autograd.Function):

    @staticmethod
    def forward(self, input, flow):
        self.save_for_backward(input, flow)
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]
        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth
        assert input.is_contiguous() == True
        assert flow.is_contiguous() == True
        output = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth])
        if input.is_cuda == True:
            n = output.nelement()
            cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {'input': input, 'flow': flow, 'output': output}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), output.data_ptr()])
        elif input.is_cuda == False:
            raise NotImplementedError()
        return output

    @staticmethod
    def backward(self, gradOutput):
        input, flow = self.saved_tensors
        intSamples = input.shape[0]
        intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
        intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]
        assert intFlowDepth == 2
        assert intInputHeight == intFlowHeight
        assert intInputWidth == intFlowWidth
        assert gradOutput.is_contiguous() == True
        gradInput = input.new_zeros([intSamples, intInputDepth, intInputHeight, intInputWidth]) if self.needs_input_grad[0] == True else None
        gradFlow = input.new_zeros([intSamples, intFlowDepth, intFlowHeight, intFlowWidth]) if self.needs_input_grad[1] == True else None
        if input.is_cuda == True:
            if gradInput is not None:
                n = gradInput.nelement()
                cupy_launch('kernel_Softsplat_updateGradInput', cupy_kernel('kernel_Softsplat_updateGradInput', {'input': input, 'flow': flow, 'gradOutput': gradOutput, 'gradInput': gradInput, 'gradFlow': gradFlow}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None])
            if gradFlow is not None:
                n = gradFlow.nelement()
                cupy_launch('kernel_Softsplat_updateGradFlow', cupy_kernel('kernel_Softsplat_updateGradFlow', {'input': input, 'flow': flow, 'gradOutput': gradOutput, 'gradInput': gradInput, 'gradFlow': gradFlow}))(grid=tuple([int((n + 512 - 1) / 512), 1, 1]), block=tuple([512, 1, 1]), args=[n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr()])
        elif input.is_cuda == False:
            raise NotImplementedError()
        return gradInput, gradFlow


def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ['summation', 'average', 'linear', 'softmax']
    if strType == 'average':
        tenInput = torch.cat([tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])], 1)
    elif strType == 'linear':
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)
    elif strType == 'softmax':
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)
    tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)
    if strType != 'summation':
        tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 1e-07)
    return tenOutput


class ModuleSoftsplat(torch.nn.Module):

    def __init__(self, strType):
        super(ModuleSoftsplat, self).__init__()
        self.strType = strType

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)

