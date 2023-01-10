import sys
_module = sys.modules[__name__]
del sys
SimpleHigherHRNet = _module
HeatmapParser = _module
misc = _module
tensorrt_utils = _module
utils = _module
visualization = _module
higherhrnet = _module
modules = _module
yolov5 = _module
dataloaders = _module
general = _module
torch_utils = _module

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


from collections import OrderedDict


import numpy as np


import torch


from torchvision.transforms import transforms


from collections import defaultdict


from collections import namedtuple


import matplotlib.pyplot as plt


import torchvision


from torch import nn


import time


import warnings


import pandas as pd


import math


import random


from itertools import repeat


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import dataloader


from torch.utils.data import distributed


import inspect


import logging


import re


from copy import deepcopy


from typing import Optional


import torch.distributed as dist


import torch.nn as nn


from torch.nn.parallel import DistributedDataParallel as DDP


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


class TRTModule_HigherHRNet(torch.nn.Module):
    """
    TensorRT wrapper for HigherHRNet.
    Args:
        path (str): Path to the .engine file for trt inference.
        device (:class:`torch.device` or str): The cuda device to be used (cpu not supported)
    """

    def __init__(self, path=None, device=None):
        super(TRTModule_HigherHRNet, self).__init__()
        logger = trt.Logger(trt.Logger.INFO)
        with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
        self.input_names = ['images']
        self.output_names = []
        self.input_flattener = None
        self.output_flattener = None
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.bindings = OrderedDict()
        fp16 = False
        dynamic = False
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if self.engine.binding_is_input(i):
                if -1 in tuple(self.engine.get_binding_shape(i)):
                    dynamic = True
                    self.context.set_binding_shape(i, tuple(self.engine.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype))
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]

    def forward(self, *inputs):
        """Forward of the model. For more details, please refer to models.higherhrnet.HigherHRNet.forward ."""
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]
        return outputs

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class StageModule(nn.Module):

    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * 2 ** i
            branch = nn.Sequential(BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum), BasicBlock(w, w, bn_momentum=bn_momentum))
            self.branches.append(branch)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** i, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** j, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** j, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))
                    ops.append(nn.Sequential(nn.Conv2d(c * 2 ** j, c * 2 ** i, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** i, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HigherHRNet(nn.Module):

    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HigherHRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True))
        self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample=downsample), Bottleneck(256, 64), Bottleneck(256, 64), Bottleneck(256, 64))
        self.transition1 = nn.ModuleList([nn.Sequential(nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)), nn.Sequential(nn.Sequential(nn.Conv2d(256, c * 2 ** 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 1, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage2 = nn.Sequential(StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum))
        self.transition2 = nn.ModuleList([nn.Sequential(), nn.Sequential(), nn.Sequential(nn.Sequential(nn.Conv2d(c * 2 ** 1, c * 2 ** 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 2, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage3 = nn.Sequential(StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum))
        self.transition3 = nn.ModuleList([nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(nn.Sequential(nn.Conv2d(c * 2 ** 2, c * 2 ** 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), nn.BatchNorm2d(c * 2 ** 3, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True), nn.ReLU(inplace=True)))])
        self.stage4 = nn.Sequential(StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum))
        self.num_deconvs = 1
        self.final_layers = []
        self.final_layers.append(nn.Conv2d(c, nof_joints * 2, kernel_size=(1, 1), stride=(1, 1)))
        for i in range(self.num_deconvs):
            self.final_layers.append(nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1)))
        self.final_layers = nn.ModuleList(self.final_layers)
        self.deconv_layers = []
        input_channels = c
        for i in range(self.num_deconvs):
            if True:
                if i == 0:
                    input_channels += nof_joints * 2
                else:
                    input_channels += nof_joints
            output_channels = c
            deconv_kernel, padding, output_padding = 4, 1, 0
            layers = []
            layers.append(nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=deconv_kernel, stride=2, padding=padding, output_padding=output_padding, bias=False), nn.BatchNorm2d(output_channels, momentum=bn_momentum), nn.ReLU(inplace=True)))
            for _ in range(4):
                layers.append(nn.Sequential(BasicBlock(output_channels, output_channels)))
            self.deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels
        self.deconv_layers = nn.ModuleList(self.deconv_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [self.transition2[0](x[0]), self.transition2[1](x[1]), self.transition2[2](x[-1])]
        x = self.stage3(x)
        x = [self.transition3[0](x[0]), self.transition3[1](x[1]), self.transition3[2](x[2]), self.transition3[3](x[-1])]
        x = self.stage4(x)
        final_outputs = []
        x = x[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)
        for i in range(self.num_deconvs):
            if True:
                x = torch.cat((x, y), 1)
            x = self.deconv_layers[i](x)
            y = self.final_layers[i + 1](x)
            final_outputs.append(y)
        return final_outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HigherHRNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_stefanopini_simple_HigherHRNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

