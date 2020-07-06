import sys
_module = sys.modules[__name__]
del sys
spatial_correlation_sampler = _module
spatial_correlation_sampler = _module
benchmark = _module
check = _module
grad_check = _module
setup = _module
setup_cpu = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch import nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import time


import torch


import numpy as np


from torch.autograd import gradcheck


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


class SpatialCorrelationSamplerFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, patch_size=1, stride=1, padding=0, dilation_patch=1):
        ctx.save_for_backward(input1, input2)
        kH, kW = ctx.kernel_size = _pair(kernel_size)
        patchH, patchW = ctx.patch_size = _pair(patch_size)
        padH, padW = ctx.padding = _pair(padding)
        dilation_patchH, dilation_patchW = ctx.dilation_patch = _pair(dilation_patch)
        dH, dW = ctx.stride = _pair(stride)
        output = correlation.forward(input1, input2, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_variables
        kH, kW = ctx.kernel_size
        patchH, patchW = ctx.patch_size
        padH, padW = ctx.padding
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride
        grad_input1, grad_input2 = correlation.backward(input1, input2, grad_output, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW)
        return grad_input1, grad_input2, None, None, None, None, None


class SpatialCorrelationSampler(nn.Module):

    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(SpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return SpatialCorrelationSamplerFunction.apply(input1, input2, self.kernel_size, self.patch_size, self.stride, self.padding, self.dilation_patch)

