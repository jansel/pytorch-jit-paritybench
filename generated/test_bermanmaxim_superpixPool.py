import sys
_module = sys.modules[__name__]
del sys
supvoxpool = _module
pytorch_superpixpool = _module
setup = _module
suppixpool_layer = _module
suppixpool_orig = _module
test_GPUpool = _module

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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import torch


import numpy as np


import torch.nn as nn


import time


class SupPixPoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, img, spx):
        spx = spx
        K = spx.max() + 1
        assert spx.size()[-2:] == img.size()[-2:]
        out = spx_gpu.forward(img, spx, K)
        outputs, indices = out
        ctx.save_for_backward(indices, img, spx, K)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        indices, img, spx, K = ctx.saved_tensors
        grad_input, = spx_gpu.backward(grad_output.contiguous(), img, spx, indices, K)
        return grad_input, torch.zeros_like(spx)


class SupPixPool(torch.nn.Module):

    def __init__(self):
        super(SupPixPool, self).__init__()

    def forward(self, img, spx):
        return SupPixPoolFunction.apply(img, spx)


class SupPixUnpool(torch.nn.Module):

    def __init__(self):
        super(SupPixUnpool, self).__init__()

    def forward(self, pooled, spx):
        outShape = pooled.size()[0:2] + spx.size()[-2:]
        out = pooled.new_zeros(outShape)
        for batch in xrange(pooled.size()[0]):
            out[batch, :, :, :] = pooled[batch, :, spx[batch, :, :]]
        return out

