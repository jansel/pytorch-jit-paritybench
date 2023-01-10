import sys
_module = sys.modules[__name__]
del sys
FlowNet2_src = _module
datasets = _module
flowlib = _module
losses = _module
main = _module
models = _module
FlowNetC = _module
FlowNetFusion = _module
FlowNetS = _module
FlowNetSD = _module
components = _module
misc = _module
ops = _module
channelnorm = _module
build = _module
functions = _module
channelnorm = _module
modules = _module
channelnorm = _module
correlation = _module
build = _module
correlation = _module
correlation = _module
resample2d = _module
build = _module
resample2d = _module
resample2d = _module
flownet2 = _module
utils = _module
flow_utils = _module
frame_utils = _module
param_utils = _module
tools = _module
demo = _module

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


import torch.utils.data as data


import math


import random


import numpy as np


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.autograd import Variable


import torch.nn.init as nn_init


from torch import nn


from torch.autograd import Function


from torch.nn.modules.module import Module


import time


from inspect import isclass


import inspect


import matplotlib


import matplotlib.pyplot as plt


class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue


def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


class L1Loss(nn.Module):

    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):

    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):

    def __init__(self, args, startScale=4, numScales=5, l_weight=0.32, norm='L1'):
        super(MultiScale, self).__init__()
        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert len(self.loss_weights) == self.numScales
        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()
        self.multiScales = [nn.AvgPool2d(self.startScale * 2 ** scale, self.startScale * 2 ** scale) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-' + self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i] * EPE(output_, target_)
                lossvalue += self.loss_weights[i] * self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return [lossvalue, epevalue]


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        rbot1 = input1.new()
        rbot2 = input2.new()
        output = input1.new()
        correlation.Correlation_forward_cuda(input1, input2, rbot1, rbot2, output, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        rbot1 = input1.new()
        rbot2 = input2.new()
        grad_input1 = Variable(input1.new())
        grad_input2 = Variable(input2.new())
        correlation.Correlation_backward_cuda(input1, input2, rbot1, rbot2, grad_output.data, grad_input1.data, grad_input2.data, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return (grad_input1, grad_input2) + (None,) * 6


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True, with_bn=True, with_relu=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True)]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if with_relu:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*tuple(layers))


def deconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class FlowNetC(nn.Module):

    def __init__(self, with_bn=True, fp16=False):
        super(FlowNetC, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.conv1 = conv(3, 64, kernel_size=7, stride=2, with_bn=with_bn)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv_redir = conv(256, 32, kernel_size=1, stride=1, with_bn=with_bn)
        corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr = nn.Sequential(tofp32(), corr, tofp16()) if fp16 else corr
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(473, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetFusion(nn.Module):

    def __init__(self, with_bn=True):
        super(FlowNetFusion, self).__init__()
        self.with_bn = with_bn
        self.conv0 = conv(11, 64, with_bn=with_bn)
        self.conv1 = conv(64, 64, stride=2, with_bn=with_bn)
        self.conv1_1 = conv(64, 128, with_bn=with_bn)
        self.conv2 = conv(128, 128, stride=2, with_bn=with_bn)
        self.conv2_1 = conv(128, 128, with_bn=with_bn)
        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)
        self.inter_conv1 = conv(162, 32, with_bn=with_bn, with_relu=False)
        self.inter_conv0 = conv(82, 16, with_bn=with_bn, with_relu=False)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)
        return flow0


class FlowNetS(nn.Module):

    def __init__(self, input_channels=12, with_bn=True):
        super(FlowNetS, self).__init__()
        self.with_bn = with_bn
        self.conv1 = conv(input_channels, 64, kernel_size=7, stride=2, with_bn=with_bn)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, with_bn=with_bn)
        self.conv3_1 = conv(256, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetSD(nn.Module):

    def __init__(self, with_bn=True):
        super(FlowNetSD, self).__init__()
        self.with_bn = with_bn
        self.conv0 = conv(6, 64, with_bn=with_bn)
        self.conv1 = conv(64, 64, stride=2, with_bn=with_bn)
        self.conv1_1 = conv(64, 128, with_bn=with_bn)
        self.conv2 = conv(128, 128, stride=2, with_bn=with_bn)
        self.conv2_1 = conv(128, 128, with_bn=with_bn)
        self.conv3 = conv(128, 256, stride=2, with_bn=with_bn)
        self.conv3_1 = conv(256, 256, with_bn=with_bn)
        self.conv4 = conv(256, 512, stride=2, with_bn=with_bn)
        self.conv4_1 = conv(512, 512, with_bn=with_bn)
        self.conv5 = conv(512, 512, stride=2, with_bn=with_bn)
        self.conv5_1 = conv(512, 512, with_bn=with_bn)
        self.conv6 = conv(512, 1024, stride=2, with_bn=with_bn)
        self.conv6_1 = conv(1024, 1024, with_bn=with_bn)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.inter_conv5 = conv(1026, 512, with_bn=with_bn, with_relu=False)
        self.inter_conv4 = conv(770, 256, with_bn=with_bn, with_relu=False)
        self.inter_conv3 = conv(386, 128, with_bn=with_bn, with_relu=False)
        self.inter_conv2 = conv(194, 64, with_bn=with_bn, with_relu=False)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()
        channelnorm.ChannelNorm_cuda_forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        channelnorm.ChannelNorm_cuda_backward(input1, output, grad_output.data, grad_input1.data, ctx.norm_deg)
        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        resample2d.Resample2d_cuda_forward(input1, input2, output, kernel_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        resample2d.Resample2d_cuda_backward(input1, input2, grad_output.data, grad_input1.data, grad_input2.data, ctx.kernel_size)
        return grad_input1, grad_input2, None


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)


def save_grad(grads, name):

    def hook(grad):
        grads[name] = grad
    return hook


class FlowNet2(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255.0, div_flow=20.0, grads=None):
        super(FlowNet2, self).__init__()
        self.with_bn = with_bn
        self.div_flow = div_flow
        self.rgb_max = rgb_max
        self.grads = {} if grads is None else grads
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16()) if fp16 else Resample2d()
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16()) if fp16 else Resample2d()
        self.flownets_2 = FlowNetS(with_bn=with_bn)
        self.flownets_d = FlowNetSD(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.resample3 = nn.Sequential(tofp32(), Resample2d(), tofp16()) if fp16 else Resample2d()
        self.resample4 = nn.Sequential(tofp32(), Resample2d(), tofp16()) if fp16 else Resample2d()
        self.flownetfusion = FlowNetFusion(with_bn=with_bn)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat([x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0], dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        req_grad = diff_flownets2_flow.requires_grad
        if req_grad:
            diff_flownets2_flow.register_hook(save_grad(self.grads, 'diff_flownets2_flow'))
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownets2_flow)
        if req_grad:
            diff_flownets2_img1.register_hook(save_grad(self.grads, 'diff_flownets2_img1'))
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        if req_grad:
            diff_flownetsd_flow.register_hook(save_grad(self.grads, 'diff_flownetsd_flow'))
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownetsd_flow)
        if req_grad:
            diff_flownetsd_img1.register_hook(save_grad(self.grads, 'diff_flownetsd_img1'))
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        if req_grad:
            flownetfusion_flow.register_hook(save_grad(self.grads, 'flownetfusion_flow'))
        return flownetfusion_flow


class FlowNet2C(FlowNetC):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255.0, div_flow=20):
        super(FlowNet2C, self).__init__(with_bn, fp16)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        flows = super(FlowNet2C, self).forward(x1, x2)
        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2S(FlowNetS):

    def __init__(self, with_bn=False, rgb_max=255.0, div_flow=20):
        super(FlowNet2S, self).__init__(input_channels=6, with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        flows = super(FlowNet2S, self).forward(x)
        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2SD(FlowNetSD):

    def __init__(self, with_bn=False, rgb_max=255.0, div_flow=20):
        super(FlowNet2SD, self).__init__(with_bn=with_bn)
        self.rgb_max = rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        flows = super(FlowNet2SD, self).forward(x)
        if self.training:
            return flows
        else:
            return self.upsample1(flows[0] * self.div_flow)


class FlowNet2CS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255.0, div_flow=20):
        super(FlowNet2CS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16()) if fp16 else Resample2d()
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat([x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0], dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, with_bn=False, fp16=False, rgb_max=255.0, div_flow=20):
        super(FlowNet2CSS, self).__init__()
        self.with_bn = with_bn
        self.fp16 = fp16
        self.rgb_max = rgb_max
        self.div_flow = div_flow
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC(with_bn=with_bn, fp16=fp16)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS(with_bn=with_bn)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS(with_bn=with_bn)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn_init.uniform(m.bias)
                nn_init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat([x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0], dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FlowNetFusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 11, 64, 64])], {}),
     True),
    (FlowNetS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (FlowNetSD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     False),
    (L1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Loss,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiScale,
     lambda: ([], {'args': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp32,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_vt_vl_lab_flownet2_pytorch(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

