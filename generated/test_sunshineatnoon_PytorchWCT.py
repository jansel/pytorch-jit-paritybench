import sys
_module = sys.modules[__name__]
del sys
Loader = _module
WCT = _module
convert = _module
modelsNIPS = _module
util = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.utils.data as data


import numpy as np


import torch


import torch.nn as nn


from torch.autograd import Variable


import time


class encoder1(nn.Module):

    def __init__(self, vgg1):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class decoder1(nn.Module):

    def __init__(self, d1):
        super(decoder1, self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(d1.get(1).weight.float())
        self.conv3.bias = torch.nn.Parameter(d1.get(1).bias.float())

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


class encoder2(nn.Module):

    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices
            =True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out


class decoder2(nn.Module):

    def __init__(self, d):
        super(decoder2, self).__init__()
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv5.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv6.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv7.bias = torch.nn.Parameter(d.get(8).bias.float())

    def forward(self, x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out


class encoder3(nn.Module):

    def __init__(self, vgg):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices
            =True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out


class decoder3(nn.Module):

    def __init__(self, d):
        super(decoder3, self).__init__()
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv7.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv8.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv9.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(d.get(12).weight.float())
        self.conv10.bias = torch.nn.Parameter(d.get(12).bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(d.get(15).weight.float())
        self.conv11.bias = torch.nn.Parameter(d.get(15).bias.float())

    def forward(self, x):
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out


class encoder4(nn.Module):

    def __init__(self, vgg):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices
            =True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        pool3 = self.relu9(out)
        out, pool_idx3 = self.maxPool3(pool3)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out


class decoder4(nn.Module):

    def __init__(self, d):
        super(decoder4, self).__init__()
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv11.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu11 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv12.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv12.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu12 = nn.ReLU(inplace=True)
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv13.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv13.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu13 = nn.ReLU(inplace=True)
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv14.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv14.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu14 = nn.ReLU(inplace=True)
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv15.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv16.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv17.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv18.weight = torch.nn.Parameter(d.get(25).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(25).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv19.weight = torch.nn.Parameter(d.get(28).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(28).bias.float())

    def forward(self, x):
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class encoder5(nn.Module):

    def __init__(self, vgg):
        super(encoder5, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices
            =True)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(vgg.get(32).weight.float())
        self.conv11.bias = torch.nn.Parameter(vgg.get(32).bias.float())
        self.relu11 = nn.ReLU(inplace=True)
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv12.weight = torch.nn.Parameter(vgg.get(35).weight.float())
        self.conv12.bias = torch.nn.Parameter(vgg.get(35).bias.float())
        self.relu12 = nn.ReLU(inplace=True)
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv13.weight = torch.nn.Parameter(vgg.get(38).weight.float())
        self.conv13.bias = torch.nn.Parameter(vgg.get(38).bias.float())
        self.relu13 = nn.ReLU(inplace=True)
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv14.weight = torch.nn.Parameter(vgg.get(42).weight.float())
        self.conv14.bias = torch.nn.Parameter(vgg.get(42).bias.float())
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out, pool_idx = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out, pool_idx2 = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out, pool_idx3 = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out, pool_idx4 = self.maxPool4(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out


class decoder5(nn.Module):

    def __init__(self, d):
        super(decoder5, self).__init__()
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)
        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
        self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
        self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
        self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
        self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
        self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
        self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x):
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


class WCT(nn.Module):

    def __init__(self, args):
        super(WCT, self).__init__()
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)
        vgg2 = load_lua(args.vgg2)
        decoder2_torch = load_lua(args.decoder2)
        vgg3 = load_lua(args.vgg3)
        decoder3_torch = load_lua(args.decoder3)
        vgg4 = load_lua(args.vgg4)
        decoder4_torch = load_lua(args.decoder4)
        vgg5 = load_lua(args.vgg5)
        decoder5_torch = load_lua(args.decoder5)
        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self, cF, sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(
            cFSize[0]).double()
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 1e-05:
                k_c = i
                break
        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 1e-05:
                k_s = i
                break
        c_d = c_e[0:k_c].pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, c_v[:, 0:k_c].t())
        whiten_cF = torch.mm(step2, cF)
        s_d = s_e[0:k_s].pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.
            diag(s_d)), s_v[:, 0:k_s].t()), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(
            targetFeature)
        return targetFeature

    def transform(self, cF, sF, csF, alpha):
        cF = cF.double()
        sF = sF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        csF.data.resize_(ccsF.size()).copy_(ccsF)
        return csF


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sunshineatnoon_PytorchWCT(_paritybench_base):
    pass
