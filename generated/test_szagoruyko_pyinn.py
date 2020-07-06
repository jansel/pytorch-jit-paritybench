import sys
_module = sys.modules[__name__]
del sys
pyinn = _module
cdgmm = _module
conv2d_depthwise = _module
dgmm = _module
im2col = _module
modules = _module
ncrelu = _module
utils = _module
setup = _module
benchmark = _module
test = _module

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


import torch


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import torch.nn.functional as F


from torch import nn


from collections import namedtuple


from string import Template


from torch.autograd import Variable


from torch.nn.init import kaiming_normal


from torch.backends import cudnn


from functools import partial


from torch.autograd import gradcheck


class Conv2dDepthwise(nn.Conv2d):
    """Depthwise 2D convolution.

    Implements depthwise convolution as in https://arxiv.org/pdf/1704.04861v1.pdf
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

    CUDA kernels from https://github.com/BVLC/caffe/pull/5665
    CPU side is done by F.conv2d

    Equivalent to:
        `nn.Conv2d(channels, channels, kernel_size, groups=channels)`

    Args:
        channels (int): Number of channels in the input image
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        dilation (int or tuple, optional): Spacing between kernel elements
        bias (bool, optional): If True, adds a learnable bias to the output

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\\_size[1] - 1) - 1) / stride[1] + 1)`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (channels, 1, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (channels)
    """

    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dDepthwise, self).__init__(channels, channels, kernel_size, stride, padding, dilation, groups=channels, bias=bias)

    def forward(self, input):
        return P.conv2d_depthwise(input, self.weight, self.bias, self.stride, self.padding, self.dilation)

