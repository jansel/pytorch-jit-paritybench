import sys
_module = sys.modules[__name__]
del sys
VOC_dataset = _module
data = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
nyuv2_dataset = _module
nyuv2_dataset_crop = _module
stanfordindoor_dataset = _module
sunrgbd_dataset = _module
Deeplab_HHA = _module
VGG_Deeplab = _module
models = _module
base_model = _module
losses = _module
model_utils = _module
models = _module
build = _module
functions = _module
depthavgpooling = _module
modules = _module
depthavgpooling = _module
depthconv = _module
build = _module
depthconv = _module
depthconv = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
test_ops = _module
train = _module
utils = _module
gradcheck = _module
html = _module
util = _module
visualizer = _module

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


import numpy as np


import torchvision


import torchvision.transforms as transforms


import torch


import time


import math


import random


import torch.utils.data as data


import torch.utils.data


import torch.utils.data as torchdata


from torchvision import transforms


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


from collections import OrderedDict


import torch.nn.functional as F


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.nn.modules.module import Module


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _triple


from collections import Iterable


class Deeplab_VGG(nn.Module):

    def __init__(self, num_classes, depthconv=False):
        super(Deeplab_VGG, self).__init__()
        self.Scale = VGG_Deeplab.vgg16(num_classes=num_classes, depthconv=depthconv)

    def forward(self, x, depth=None):
        output = self.Scale(x, depth)
        return output


class ConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bn=False, maxpool=False, pool_kernel=3, pool_stride=2, pool_pad=1):
        super(ConvModule, self).__init__()
        conv2d = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        if maxpool:
            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_pad)]
        self.layers = nn.Sequential(*([conv2d] + layers))

    def forward(self, x):
        x = self.layers(x)
        return x


class DepthconvFunction(Function):

    def __init__(self, stride, padding, dilation, bias=True):
        super(DepthconvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        ffi_ = cffi.FFI()
        self.null = ffi_.NULL
        self.bias = bias

    def forward(self, input, depth, weight, bias=None):
        self.save_for_backward(input, depth, weight, bias)
        if not self.bias or bias is None:
            bias = self.null
        output_size = [int((input.size()[i + 2] + 2 * self.padding[i] - weight.size()[i + 2]) / self.stride[i] + 1) for i in range(2)]
        output = input.new(*self._output_size(input, weight))
        self.columns = input.new(weight.size(1) * weight.size(2) * weight.size(3), output_size[0] * output_size[1]).zero_()
        self.ones = input.new(output_size[0] * output_size[1]).zero_()
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.FloatTensor):
                raise NotImplementedError
            depthconv.depthconv_forward_cuda(input, depth, weight, bias, output, self.columns, self.ones, weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])
        return output

    def backward(self, grad_output):
        input, depth, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0]:
                grad_input = input.new(*input.size()).zero_()
                depthconv.depthconv_backward_input_cuda(input, depth, grad_output, grad_input, weight, self.columns, weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])
            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                if len(self.needs_input_grad) == 4:
                    if self.needs_input_grad[3]:
                        grad_bias = weight.new(*bias.size()).zero_()
                    else:
                        grad_bias = self.null
                else:
                    grad_bias = self.null
                depthconv.depthconv_backward_parameters_cuda(input, depth, grad_output, grad_weight, grad_bias, self.columns, self.ones, weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0], 1)
                if len(self.needs_input_grad) == 4:
                    if not self.needs_input_grad[3]:
                        grad_bias = None
                else:
                    grad_bias = None
        return grad_input, None, grad_weight, grad_bias

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


def depth_conv(input, depth, weight, bias, stride=1, padding=0, dilation=1):
    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    f = DepthconvFunction(_pair(stride), _pair(padding), _pair(dilation))
    if isinstance(bias, torch.nn.Parameter):
        return f(input, depth, weight, bias)
    else:
        return f(input, depth, weight)


class DepthConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, depth):
        return depth_conv(input, depth, self.weight, self.bias, self.stride, self.padding, self.dilation)


class DepthConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bn=False):
        super(DepthConvModule, self).__init__()
        conv2d = DepthConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*([conv2d] + layers))

    def forward(self, x, depth):
        for im, module in enumerate(self.layers._modules.values()):
            if im == 0:
                x = module(x, depth)
            else:
                x = module(x)
        return x


class VGG_layer2(nn.Module):

    def __init__(self, batch_norm=False, depthconv=False):
        super(VGG_layer2, self).__init__()
        in_channels = 3
        self.depthconv = depthconv
        self.conv1_1 = ConvModule(3, 64, bn=batch_norm)
        self.conv1_2 = ConvModule(64, 64, bn=batch_norm, maxpool=True)
        self.downsample_depth2_1 = nn.AvgPool2d(3, padding=1, stride=2)
        self.conv2_1 = ConvModule(64, 128, bn=batch_norm)
        self.conv2_2 = ConvModule(128, 128, bn=batch_norm, maxpool=True)
        self.downsample_depth3_1 = nn.AvgPool2d(3, padding=1, stride=2)
        self.conv3_1 = ConvModule(128, 256, bn=batch_norm)
        self.conv3_2 = ConvModule(256, 256, bn=batch_norm)
        self.conv3_3 = ConvModule(256, 256, bn=batch_norm, maxpool=True)
        if self.depthconv:
            self.conv4_1_depthconvweight = 1.0
            self.downsample_depth4_1 = nn.AvgPool2d(3, padding=1, stride=2)
            self.conv4_1 = DepthConvModule(256, 512, bn=batch_norm)
        else:
            self.conv4_1 = ConvModule(256, 512, bn=batch_norm)
        self.conv4_2 = ConvModule(512, 512, bn=batch_norm)
        self.conv4_3 = ConvModule(512, 512, bn=batch_norm, maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        if self.depthconv:
            self.conv5_1_depthconvweight = 1.0
            self.conv5_1 = DepthConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        else:
            self.conv5_1 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_2 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_3 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2, maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x, depth=None):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        depth = self.downsample_depth2_1(depth)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        depth = self.downsample_depth3_1(depth)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.depthconv:
            depth = self.downsample_depth4_1(depth)
            x = self.conv4_1(x, self.conv4_1_depthconvweight * depth)
        else:
            x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.depthconv:
            x = self.conv5_1(x, self.conv5_1_depthconvweight * depth)
        else:
            x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5a(x)
        return x, depth


class DepthavgpoolingFunction(Function):

    def __init__(self, kernel_size, stride, padding):
        super(DepthavgpoolingFunction, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, input, depth):
        self.save_for_backward(input, depth)
        self.depth = depth
        output = input.new(*self._output_size(input))
        self.depthweightcount = input.new(*depth.size()).zero_()
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.FloatTensor):
                raise NotImplementedError
            depthavgpooling.depthavgpooling_forward_cuda(input, depth, output, self.depthweightcount, self.kernel_size[1], self.kernel_size[0], self.stride[1], self.stride[0], self.padding[1], self.padding[0])
        return output

    def backward(self, grad_output):
        input, depth = self.saved_tensors
        grad_input = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0]:
                grad_input = input.new(*input.size()).zero_()
                depthavgpooling.depthavgpooling_backward_input_cuda(input, depth, self.depthweightcount, grad_output, grad_input, self.kernel_size[1], self.kernel_size[0], self.stride[1], self.stride[0], self.padding[1], self.padding[0])
        return grad_input, None

    def _output_size(self, input):
        output_size = input.size(0), input.size(0)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.kernel_size[d]
            stride = self.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('avgpooling input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


def depth_avgpooling(input, depth, kernel_size=3, stride=1, padding=0):
    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
    f = DepthavgpoolingFunction(_pair(kernel_size), _pair(stride), _pair(padding))
    return f(input, depth)


class Depthavgpooling(Module):

    def __init__(self, kernel_size, stride=1, padding=0):
        super(Depthavgpooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input, depth):
        return depth_avgpooling(input, depth, self.kernel_size, self.stride, self.padding)


class VGG_layer(nn.Module):

    def __init__(self, batch_norm=False, depthconv=False):
        super(VGG_layer, self).__init__()
        in_channels = 3
        self.depthconv = depthconv
        if self.depthconv:
            self.conv1_1_depthconvweight = 1.0
            self.conv1_1 = DepthConvModule(3, 64, bn=batch_norm)
        else:
            self.conv1_1 = ConvModule(3, 64, bn=batch_norm)
        self.conv1_2 = ConvModule(64, 64, bn=batch_norm, maxpool=True)
        if self.depthconv:
            self.conv2_1_depthconvweight = 1.0
            self.downsample_depth2_1 = nn.AvgPool2d(3, padding=1, stride=2)
            self.conv2_1 = DepthConvModule(64, 128, bn=batch_norm)
        else:
            self.conv2_1 = ConvModule(64, 128, bn=batch_norm)
        self.conv2_2 = ConvModule(128, 128, bn=batch_norm, maxpool=True)
        if self.depthconv:
            self.conv3_1_depthconvweight = 1.0
            self.downsample_depth3_1 = nn.AvgPool2d(3, padding=1, stride=2)
            self.conv3_1 = DepthConvModule(128, 256, bn=batch_norm)
        else:
            self.conv3_1 = ConvModule(128, 256, bn=batch_norm)
        self.conv3_2 = ConvModule(256, 256, bn=batch_norm)
        self.conv3_3 = ConvModule(256, 256, bn=batch_norm, maxpool=True)
        if self.depthconv:
            self.conv4_1_depthconvweight = 1.0
            self.downsample_depth4_1 = nn.AvgPool2d(3, padding=1, stride=2)
            self.conv4_1 = DepthConvModule(256, 512, bn=batch_norm)
        else:
            self.conv4_1 = ConvModule(256, 512, bn=batch_norm)
        self.conv4_2 = ConvModule(512, 512, bn=batch_norm)
        self.conv4_3 = ConvModule(512, 512, bn=batch_norm, maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        if self.depthconv:
            self.conv5_1_depthconvweight = 1.0
            self.conv5_1 = DepthConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        else:
            self.conv5_1 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_2 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_3 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2, maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a_d = Depthavgpooling(kernel_size=3, stride=1, padding=1)

    def forward(self, x, depth=None):
        if self.depthconv:
            x = self.conv1_1(x, self.conv1_1_depthconvweight * depth)
        else:
            x = self.conv1_1(x)
        x = self.conv1_2(x)
        if self.depthconv:
            depth = self.downsample_depth2_1(depth)
            x = self.conv2_1(x, self.conv2_1_depthconvweight * depth)
        else:
            x = self.conv2_1(x)
        x = self.conv2_2(x)
        if self.depthconv:
            depth = self.downsample_depth3_1(depth)
            x = self.conv3_1(x, self.conv3_1_depthconvweight * depth)
        else:
            x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.depthconv:
            depth = self.downsample_depth4_1(depth)
            x = self.conv4_1(x, self.conv4_1_depthconvweight * depth)
        else:
            x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.depthconv:
            x = self.conv5_1(x, self.conv5_1_depthconvweight * depth)
        else:
            x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        if self.depthconv:
            x = self.pool5a_d(x, depth)
        else:
            x = self.pool5a(x)
        return x, depth


class Classifier_Module(nn.Module):

    def __init__(self, num_classes, inplanes, depthconv=False):
        super(Classifier_Module, self).__init__()
        self.depthconv = depthconv
        if depthconv:
            self.fc6_1_depthconvweight = 1.0
            self.fc6_1 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
        else:
            self.fc6_1 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=6, dilation=6)
        self.fc7_1 = nn.Sequential(*[nn.ReLU(True), nn.Dropout(), nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])
        self.fc8_1 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
        if depthconv:
            self.fc6_2_depthconvweight = 1.0
            self.fc6_2 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
        else:
            self.fc6_2 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
        self.fc7_2 = nn.Sequential(*[nn.ReLU(True), nn.Dropout(), nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])
        self.fc8_2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
        if depthconv:
            self.fc6_3_depthconvweight = 1.0
            self.fc6_3 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=18, dilation=18)
        else:
            self.fc6_3 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=18, dilation=18)
        self.fc7_3 = nn.Sequential(*[nn.ReLU(True), nn.Dropout(), nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])
        self.fc8_3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)
        if depthconv:
            self.fc6_4_depthconvweight = 1.0
            self.fc6_4 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=24, dilation=24)
        else:
            self.fc6_4 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=24, dilation=24)
        self.fc7_4 = nn.Sequential(*[nn.ReLU(True), nn.Dropout(), nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])
        self.fc8_4 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x, depth=None):
        if self.depthconv:
            out1 = self.fc6_1(x, self.fc6_1_depthconvweight * depth)
        else:
            out1 = self.fc6_1(x)
        out1 = self.fc7_1(out1)
        out1 = self.fc8_1(out1)
        if self.depthconv:
            out2 = self.fc6_2(x, self.fc6_2_depthconvweight * depth)
        else:
            out2 = self.fc6_2(x)
        out2 = self.fc7_2(out2)
        out2 = self.fc8_2(out2)
        if self.depthconv:
            out3 = self.fc6_3(x, self.fc6_3_depthconvweight * depth)
        else:
            out3 = self.fc6_3(x)
        out3 = self.fc7_3(out3)
        out3 = self.fc8_3(out3)
        if self.depthconv:
            out4 = self.fc6_4(x, self.fc6_4_depthconvweight * depth)
        else:
            out4 = self.fc6_4(x)
        out4 = self.fc7_4(out4)
        out4 = self.fc8_4(out4)
        return out1 + out2 + out3 + out4


class Classifier_Module2(nn.Module):

    def __init__(self, num_classes, inplanes, depthconv=False):
        super(Classifier_Module2, self).__init__()
        self.depthconv = depthconv
        if depthconv:
            self.fc6_2_depthconvweight = 1.0
            self.fc6_2 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
            self.downsample_depth = None
        else:
            self.downsample_depth = nn.AvgPool2d(9, padding=1, stride=8)
            self.fc6_2 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
        self.fc7_2 = nn.Sequential(*[nn.ReLU(True), nn.Dropout(), nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc8_2 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x, depth=None):
        if self.depthconv:
            out2 = self.fc6_2(x, self.fc6_2_depthconvweight * depth)
        else:
            out2 = self.fc6_2(x)
        out2 = self.fc7_2(out2)
        out2_size = out2.size()
        globalpool = self.globalpooling(out2)
        globalpool = self.dropout(globalpool)
        upsample = nn.Upsample((out2_size[2], out2_size[3]), mode='bilinear')
        globalpool = upsample(globalpool)
        out2 = torch.cat([out2, globalpool], 1)
        out2 = self.fc8_2(out2)
        return out2


class CaffeNormalize(nn.Module):

    def __init__(self, features, eps=1e-07):
        super(CaffeNormalize, self).__init__()
        self.scale = nn.Parameter(10.0 * torch.ones(features))
        self.eps = eps

    def forward(self, x):
        x_size = x.size()
        norm = x.norm(2, dim=1, keepdim=True)
        x = x.div(norm + self.eps)
        return x.mul(self.scale.view(1, x_size[1], 1, 1))


class VGG(nn.Module):

    def __init__(self, num_classes=20, init_weights=True, depthconv=False, bn=False):
        super(VGG, self).__init__()
        self.features = VGG_layer(batch_norm=bn, depthconv=depthconv)
        self.classifier = Classifier_Module2(num_classes, 512, depthconv=depthconv)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, depth=None):
        x, depth = self.features(x, depth)
        x = self.classifier(x, depth)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_normalize_params(self):
        b = []
        b.append(self.classifier.norm)
        for i in b:
            if isinstance(i, CaffeNormalize):
                yield i.scale

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        b.append(self.classifier.fc6_2)
        b.append(self.classifier.fc7_2)
        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.weight.requires_grad:
                        yield j.weight
                elif isinstance(j, DepthConv):
                    if j.weight.requires_grad:
                        yield j.weight

    def get_2x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        b.append(self.classifier.fc6_2)
        b.append(self.classifier.fc7_2)
        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias
                elif isinstance(j, DepthConv):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.classifier.fc8_2.weight)
        for i in b:
            yield i

    def get_20x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.classifier.fc8_2.bias)
        for i in b:
            yield i

    def get_100x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.features.conv1_1_depthconvweight)
        b.append(self.features.conv2_1_depthconvweight)
        b.append(self.features.conv3_1_depthconvweight)
        b.append(self.features.conv4_1_depthconvweight)
        b.append(self.features.conv5_1_depthconvweight)
        b.append(self.classifier.fc6_1_depthconvweight)
        b.append(self.classifier.fc6_2_depthconvweight)
        b.append(self.classifier.fc6_3_depthconvweight)
        b.append(self.classifier.fc6_4_depthconvweight)
        for j in range(len(b)):
            yield b[j]


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2.0, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num + 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.softmax(inputs)
        b, c, h, w = inputs.size()
        class_mask = Variable(torch.zeros([b, c + 1, h, w]))
        class_mask.scatter_(1, targets.long(), 1.0)
        class_mask = class_mask[:, :-1, :, :]
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
        probs = (P * class_mask).sum(1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06, gamma=1.0, beta=0.0, learnable=False):
        super(LayerNorm, self).__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
        else:
            self.gamma = gamma
            self.beta = beta
        self.eps = eps

    def forward(self, x):
        x_size = x.size()
        mean = x.view(x_size[0], x_size[1], x_size[2] * x_size[3]).mean(2).view(x_size[0], x_size[1], 1, 1).repeat(1, 1, x_size[2], x_size[3])
        std = x.view(x_size[0], x_size[1], x_size[2] * x_size[3]).std(2).view(x_size[0], x_size[1], 1, 1).repeat(1, 1, x_size[2], x_size[3])
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class DepthGlobalPool(nn.Module):

    def __init__(self, n_features, n_out):
        super(DepthGlobalPool, self).__init__()
        self.model = nn.Conv2d(n_features, n_out, kernel_size=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = CaffeNormalize(n_out)
        self.dropout = nn.Dropout(0.3)
        n = self.model.kernel_size[0] * self.model.kernel_size[1] * self.model.out_channels
        self.model.weight.data.normal_(0, np.sqrt(2.0 / n))
        if self.model.bias is not None:
            self.model.bias.data.zero_()

    def forward(self, features, depth, depthpool=False):
        out2_size = features.size()
        features = self.model(features)
        if isinstance(depth, Variable) and depthpool:
            outfeatures = features.clone()
            n_c = features.size()[1]
            depth = depth.data.cpu().numpy()
            _, depth_bin = np.histogram(depth)
            bin_low = depth_bin[0]
            for bin_high in depth_bin[1:]:
                indices = ((depth <= bin_high) & (depth >= bin_low)).nonzero()
                if indices[0].shape[0] != 0:
                    for j in range(n_c):
                        output_ins = features[indices[0], indices[1] + j, indices[2], indices[3]]
                        mean_feat = torch.mean(output_ins).expand_as(output_ins)
                        outfeatures[indices[0], indices[1] + j, indices[2], indices[3]] = mean_feat
                    bin_low = bin_high
            outfeatures = self.dropout(outfeatures)
        else:
            features = self.pool(features)
            outfeatures = self.dropout(features)
            self.upsample = nn.UpsamplingBilinear2d((out2_size[2], out2_size[3]))
            outfeatures = self.upsample(outfeatures)
        return outfeatures


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CaffeNormalize,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier_Module,
     lambda: ([], {'num_classes': 4, 'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Classifier_Module2,
     lambda: ([], {'num_classes': 4, 'inplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvModule,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthGlobalPool,
     lambda: ([], {'n_features': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FocalLoss,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4, 4, 4], dtype=torch.int64)], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VGG_layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_laughtervv_DepthAwareCNN(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

