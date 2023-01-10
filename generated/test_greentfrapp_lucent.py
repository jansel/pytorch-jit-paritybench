import sys
_module = sys.modules[__name__]
del sys
demo = _module
lucent = _module
misc = _module
channel_reducer = _module
io = _module
collapse_channels = _module
serialize_array = _module
showing = _module
modelzoo = _module
InceptionV1 = _module
inceptionv1 = _module
helper_layers = _module
util = _module
optvis = _module
objectives = _module
objectives_util = _module
param = _module
color = _module
cppn = _module
gan = _module
images = _module
lowres = _module
resize_bilinear_nd = _module
spatial = _module
render = _module
transform = _module
util = _module
setup = _module
test_collapse_channels = _module
test_showing = _module
test_channel_reducer = _module
test_inceptionv1 = _module
test_cppn = _module
test_gan = _module
test_lowres = _module
test_spatial = _module
test_integration = _module
test_objectives = _module
test_render = _module
test_transform = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from collections import OrderedDict


import warnings


from torchvision.transforms import Normalize


import random


model_urls = {'pool5': 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA', 'fc6': 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4', 'fc7': 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw', 'fc8': 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU'}


class InceptionV1(nn.Module):

    def __init__(self, pretrained=False, progress=True, redirected_ReLU=True):
        super(InceptionV1, self).__init__()
        self.conv2d0_pre_relu_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.conv2d1_pre_relu_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv2d2_pre_relu_conv = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_1x1_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_3x3_pre_relu_conv = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3a_5x5_pre_relu_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_1x1_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_3x3_pre_relu_conv = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed3b_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_1x1_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=16, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=480, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_3x3_pre_relu_conv = nn.Conv2d(in_channels=96, out_channels=204, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4a_5x5_pre_relu_conv = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_1x1_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=508, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_3x3_pre_relu_conv = nn.Conv2d(in_channels=112, out_channels=224, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4b_5x5_pre_relu_conv = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_1x1_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_3x3_pre_relu_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4c_5x5_pre_relu_conv = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_1x1_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=112, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=144, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_3x3_pre_relu_conv = nn.Conv2d(in_channels=144, out_channels=288, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4d_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_1x1_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=32, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=528, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_3x3_pre_relu_conv = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed4e_5x5_pre_relu_conv = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_1x1_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=160, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_3x3_pre_relu_conv = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed5a_5x5_pre_relu_conv = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_1x1_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=192, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=48, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels=832, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_3x3_pre_relu_conv = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.mixed5b_5x5_pre_relu_conv = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=(1, 1), groups=1, bias=True)
        self.softmax2_pre_activation_matmul = nn.Linear(in_features=1024, out_features=1008, bias=True)
        self.add_layers(redirected_ReLU)
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['inceptionv1'], progress=progress))

    def add_layers(self, redirected_ReLU=True):
        if redirected_ReLU:
            relu = helper_layers.RedirectedReluLayer
        else:
            relu = helper_layers.ReluLayer
        self.conv2d0 = relu()
        self.maxpool0 = helper_layers.MaxPool2dLayer()
        self.conv2d1 = relu()
        self.conv2d2 = relu()
        self.maxpool1 = helper_layers.MaxPool2dLayer()
        self.mixed3a_pool = helper_layers.MaxPool2dLayer()
        self.mixed3a_1x1 = relu()
        self.mixed3a_3x3_bottleneck = relu()
        self.mixed3a_5x5_bottleneck = relu()
        self.mixed3a_pool_reduce = relu()
        self.mixed3a_3x3 = relu()
        self.mixed3a_5x5 = relu()
        self.mixed3a = helper_layers.CatLayer()
        self.mixed3b_pool = helper_layers.MaxPool2dLayer()
        self.mixed3b_1x1 = relu()
        self.mixed3b_3x3_bottleneck = relu()
        self.mixed3b_5x5_bottleneck = relu()
        self.mixed3b_pool_reduce = relu()
        self.mixed3b_3x3 = relu()
        self.mixed3b_5x5 = relu()
        self.mixed3b = helper_layers.CatLayer()
        self.maxpool4 = helper_layers.MaxPool2dLayer()
        self.mixed4a_pool = helper_layers.MaxPool2dLayer()
        self.mixed4a_1x1 = relu()
        self.mixed4a_3x3_bottleneck = relu()
        self.mixed4a_5x5_bottleneck = relu()
        self.mixed4a_pool_reduce = relu()
        self.mixed4a_3x3 = relu()
        self.mixed4a_5x5 = relu()
        self.mixed4a = helper_layers.CatLayer()
        self.mixed4b_pool = helper_layers.MaxPool2dLayer()
        self.mixed4b_1x1 = relu()
        self.mixed4b_3x3_bottleneck = relu()
        self.mixed4b_5x5_bottleneck = relu()
        self.mixed4b_pool_reduce = relu()
        self.mixed4b_3x3 = relu()
        self.mixed4b_5x5 = relu()
        self.mixed4b = helper_layers.CatLayer()
        self.mixed4c_pool = helper_layers.MaxPool2dLayer()
        self.mixed4c_1x1 = relu()
        self.mixed4c_3x3_bottleneck = relu()
        self.mixed4c_5x5_bottleneck = relu()
        self.mixed4c_pool_reduce = relu()
        self.mixed4c_3x3 = relu()
        self.mixed4c_5x5 = relu()
        self.mixed4c = helper_layers.CatLayer()
        self.mixed4d_pool = helper_layers.MaxPool2dLayer()
        self.mixed4d_1x1 = relu()
        self.mixed4d_3x3_bottleneck = relu()
        self.mixed4d_5x5_bottleneck = relu()
        self.mixed4d_pool_reduce = relu()
        self.mixed4d_3x3 = relu()
        self.mixed4d_5x5 = relu()
        self.mixed4d = helper_layers.CatLayer()
        self.mixed4e_pool = helper_layers.MaxPool2dLayer()
        self.mixed4e_1x1 = relu()
        self.mixed4e_3x3_bottleneck = relu()
        self.mixed4e_5x5_bottleneck = relu()
        self.mixed4e_pool_reduce = relu()
        self.mixed4e_3x3 = relu()
        self.mixed4e_5x5 = relu()
        self.mixed4e = helper_layers.CatLayer()
        self.maxpool10 = helper_layers.MaxPool2dLayer()
        self.mixed5a_pool = helper_layers.MaxPool2dLayer()
        self.mixed5a_1x1 = relu()
        self.mixed5a_3x3_bottleneck = relu()
        self.mixed5a_5x5_bottleneck = relu()
        self.mixed5a_pool_reduce = relu()
        self.mixed5a_3x3 = relu()
        self.mixed5a_5x5 = relu()
        self.mixed5a = helper_layers.CatLayer()
        self.mixed5b_pool = helper_layers.MaxPool2dLayer()
        self.mixed5b_1x1 = relu()
        self.mixed5b_3x3_bottleneck = relu()
        self.mixed5b_5x5_bottleneck = relu()
        self.mixed5b_pool_reduce = relu()
        self.mixed5b_3x3 = relu()
        self.mixed5b_5x5 = relu()
        self.mixed5b = helper_layers.CatLayer()
        self.softmax2 = helper_layers.SoftMaxLayer()

    def forward(self, x):
        conv2d0_pre_relu_conv_pad = F.pad(x, (2, 3, 2, 3))
        conv2d0_pre_relu_conv = self.conv2d0_pre_relu_conv(conv2d0_pre_relu_conv_pad)
        conv2d0 = self.conv2d0(conv2d0_pre_relu_conv)
        maxpool0_pad = F.pad(conv2d0, (0, 1, 0, 1), value=float('-inf'))
        maxpool0 = self.maxpool0(maxpool0_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        localresponsenorm0 = F.local_response_norm(maxpool0, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
        conv2d1_pre_relu_conv = self.conv2d1_pre_relu_conv(localresponsenorm0)
        conv2d1 = self.conv2d1(conv2d1_pre_relu_conv)
        conv2d2_pre_relu_conv_pad = F.pad(conv2d1, (1, 1, 1, 1))
        conv2d2_pre_relu_conv = self.conv2d2_pre_relu_conv(conv2d2_pre_relu_conv_pad)
        conv2d2 = self.conv2d2(conv2d2_pre_relu_conv)
        localresponsenorm1 = F.local_response_norm(conv2d2, size=9, alpha=9.99999974738e-05, beta=0.5, k=1)
        maxpool1_pad = F.pad(localresponsenorm1, (0, 1, 0, 1), value=float('-inf'))
        maxpool1 = self.maxpool1(maxpool1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed3a_1x1_pre_relu_conv = self.mixed3a_1x1_pre_relu_conv(maxpool1)
        mixed3a_3x3_bottleneck_pre_relu_conv = self.mixed3a_3x3_bottleneck_pre_relu_conv(maxpool1)
        mixed3a_5x5_bottleneck_pre_relu_conv = self.mixed3a_5x5_bottleneck_pre_relu_conv(maxpool1)
        mixed3a_pool_pad = F.pad(maxpool1, (1, 1, 1, 1), value=float('-inf'))
        mixed3a_pool = self.mixed3a_pool(mixed3a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed3a_1x1 = self.mixed3a_1x1(mixed3a_1x1_pre_relu_conv)
        mixed3a_3x3_bottleneck = self.mixed3a_3x3_bottleneck(mixed3a_3x3_bottleneck_pre_relu_conv)
        mixed3a_5x5_bottleneck = self.mixed3a_5x5_bottleneck(mixed3a_5x5_bottleneck_pre_relu_conv)
        mixed3a_pool_reduce_pre_relu_conv = self.mixed3a_pool_reduce_pre_relu_conv(mixed3a_pool)
        mixed3a_3x3_pre_relu_conv_pad = F.pad(mixed3a_3x3_bottleneck, (1, 1, 1, 1))
        mixed3a_3x3_pre_relu_conv = self.mixed3a_3x3_pre_relu_conv(mixed3a_3x3_pre_relu_conv_pad)
        mixed3a_5x5_pre_relu_conv_pad = F.pad(mixed3a_5x5_bottleneck, (2, 2, 2, 2))
        mixed3a_5x5_pre_relu_conv = self.mixed3a_5x5_pre_relu_conv(mixed3a_5x5_pre_relu_conv_pad)
        mixed3a_pool_reduce = self.mixed3a_pool_reduce(mixed3a_pool_reduce_pre_relu_conv)
        mixed3a_3x3 = self.mixed3a_3x3(mixed3a_3x3_pre_relu_conv)
        mixed3a_5x5 = self.mixed3a_5x5(mixed3a_5x5_pre_relu_conv)
        mixed3a = self.mixed3a((mixed3a_1x1, mixed3a_3x3, mixed3a_5x5, mixed3a_pool_reduce), 1)
        mixed3b_1x1_pre_relu_conv = self.mixed3b_1x1_pre_relu_conv(mixed3a)
        mixed3b_3x3_bottleneck_pre_relu_conv = self.mixed3b_3x3_bottleneck_pre_relu_conv(mixed3a)
        mixed3b_5x5_bottleneck_pre_relu_conv = self.mixed3b_5x5_bottleneck_pre_relu_conv(mixed3a)
        mixed3b_pool_pad = F.pad(mixed3a, (1, 1, 1, 1), value=float('-inf'))
        mixed3b_pool = self.mixed3b_pool(mixed3b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed3b_1x1 = self.mixed3b_1x1(mixed3b_1x1_pre_relu_conv)
        mixed3b_3x3_bottleneck = self.mixed3b_3x3_bottleneck(mixed3b_3x3_bottleneck_pre_relu_conv)
        mixed3b_5x5_bottleneck = self.mixed3b_5x5_bottleneck(mixed3b_5x5_bottleneck_pre_relu_conv)
        mixed3b_pool_reduce_pre_relu_conv = self.mixed3b_pool_reduce_pre_relu_conv(mixed3b_pool)
        mixed3b_3x3_pre_relu_conv_pad = F.pad(mixed3b_3x3_bottleneck, (1, 1, 1, 1))
        mixed3b_3x3_pre_relu_conv = self.mixed3b_3x3_pre_relu_conv(mixed3b_3x3_pre_relu_conv_pad)
        mixed3b_5x5_pre_relu_conv_pad = F.pad(mixed3b_5x5_bottleneck, (2, 2, 2, 2))
        mixed3b_5x5_pre_relu_conv = self.mixed3b_5x5_pre_relu_conv(mixed3b_5x5_pre_relu_conv_pad)
        mixed3b_pool_reduce = self.mixed3b_pool_reduce(mixed3b_pool_reduce_pre_relu_conv)
        mixed3b_3x3 = self.mixed3b_3x3(mixed3b_3x3_pre_relu_conv)
        mixed3b_5x5 = self.mixed3b_5x5(mixed3b_5x5_pre_relu_conv)
        mixed3b = self.mixed3b((mixed3b_1x1, mixed3b_3x3, mixed3b_5x5, mixed3b_pool_reduce), 1)
        maxpool4_pad = F.pad(mixed3b, (0, 1, 0, 1), value=float('-inf'))
        maxpool4 = self.maxpool4(maxpool4_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed4a_1x1_pre_relu_conv = self.mixed4a_1x1_pre_relu_conv(maxpool4)
        mixed4a_3x3_bottleneck_pre_relu_conv = self.mixed4a_3x3_bottleneck_pre_relu_conv(maxpool4)
        mixed4a_5x5_bottleneck_pre_relu_conv = self.mixed4a_5x5_bottleneck_pre_relu_conv(maxpool4)
        mixed4a_pool_pad = F.pad(maxpool4, (1, 1, 1, 1), value=float('-inf'))
        mixed4a_pool = self.mixed4a_pool(mixed4a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4a_1x1 = self.mixed4a_1x1(mixed4a_1x1_pre_relu_conv)
        mixed4a_3x3_bottleneck = self.mixed4a_3x3_bottleneck(mixed4a_3x3_bottleneck_pre_relu_conv)
        mixed4a_5x5_bottleneck = self.mixed4a_5x5_bottleneck(mixed4a_5x5_bottleneck_pre_relu_conv)
        mixed4a_pool_reduce_pre_relu_conv = self.mixed4a_pool_reduce_pre_relu_conv(mixed4a_pool)
        mixed4a_3x3_pre_relu_conv_pad = F.pad(mixed4a_3x3_bottleneck, (1, 1, 1, 1))
        mixed4a_3x3_pre_relu_conv = self.mixed4a_3x3_pre_relu_conv(mixed4a_3x3_pre_relu_conv_pad)
        mixed4a_5x5_pre_relu_conv_pad = F.pad(mixed4a_5x5_bottleneck, (2, 2, 2, 2))
        mixed4a_5x5_pre_relu_conv = self.mixed4a_5x5_pre_relu_conv(mixed4a_5x5_pre_relu_conv_pad)
        mixed4a_pool_reduce = self.mixed4a_pool_reduce(mixed4a_pool_reduce_pre_relu_conv)
        mixed4a_3x3 = self.mixed4a_3x3(mixed4a_3x3_pre_relu_conv)
        mixed4a_5x5 = self.mixed4a_5x5(mixed4a_5x5_pre_relu_conv)
        mixed4a = self.mixed4a((mixed4a_1x1, mixed4a_3x3, mixed4a_5x5, mixed4a_pool_reduce), 1)
        mixed4b_1x1_pre_relu_conv = self.mixed4b_1x1_pre_relu_conv(mixed4a)
        mixed4b_3x3_bottleneck_pre_relu_conv = self.mixed4b_3x3_bottleneck_pre_relu_conv(mixed4a)
        mixed4b_5x5_bottleneck_pre_relu_conv = self.mixed4b_5x5_bottleneck_pre_relu_conv(mixed4a)
        mixed4b_pool_pad = F.pad(mixed4a, (1, 1, 1, 1), value=float('-inf'))
        mixed4b_pool = self.mixed4b_pool(mixed4b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4b_1x1 = self.mixed4b_1x1(mixed4b_1x1_pre_relu_conv)
        mixed4b_3x3_bottleneck = self.mixed4b_3x3_bottleneck(mixed4b_3x3_bottleneck_pre_relu_conv)
        mixed4b_5x5_bottleneck = self.mixed4b_5x5_bottleneck(mixed4b_5x5_bottleneck_pre_relu_conv)
        mixed4b_pool_reduce_pre_relu_conv = self.mixed4b_pool_reduce_pre_relu_conv(mixed4b_pool)
        mixed4b_3x3_pre_relu_conv_pad = F.pad(mixed4b_3x3_bottleneck, (1, 1, 1, 1))
        mixed4b_3x3_pre_relu_conv = self.mixed4b_3x3_pre_relu_conv(mixed4b_3x3_pre_relu_conv_pad)
        mixed4b_5x5_pre_relu_conv_pad = F.pad(mixed4b_5x5_bottleneck, (2, 2, 2, 2))
        mixed4b_5x5_pre_relu_conv = self.mixed4b_5x5_pre_relu_conv(mixed4b_5x5_pre_relu_conv_pad)
        mixed4b_pool_reduce = self.mixed4b_pool_reduce(mixed4b_pool_reduce_pre_relu_conv)
        mixed4b_3x3 = self.mixed4b_3x3(mixed4b_3x3_pre_relu_conv)
        mixed4b_5x5 = self.mixed4b_5x5(mixed4b_5x5_pre_relu_conv)
        mixed4b = self.mixed4b((mixed4b_1x1, mixed4b_3x3, mixed4b_5x5, mixed4b_pool_reduce), 1)
        mixed4c_1x1_pre_relu_conv = self.mixed4c_1x1_pre_relu_conv(mixed4b)
        mixed4c_3x3_bottleneck_pre_relu_conv = self.mixed4c_3x3_bottleneck_pre_relu_conv(mixed4b)
        mixed4c_5x5_bottleneck_pre_relu_conv = self.mixed4c_5x5_bottleneck_pre_relu_conv(mixed4b)
        mixed4c_pool_pad = F.pad(mixed4b, (1, 1, 1, 1), value=float('-inf'))
        mixed4c_pool = self.mixed4c_pool(mixed4c_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4c_1x1 = self.mixed4c_1x1(mixed4c_1x1_pre_relu_conv)
        mixed4c_3x3_bottleneck = self.mixed4c_3x3_bottleneck(mixed4c_3x3_bottleneck_pre_relu_conv)
        mixed4c_5x5_bottleneck = self.mixed4c_5x5_bottleneck(mixed4c_5x5_bottleneck_pre_relu_conv)
        mixed4c_pool_reduce_pre_relu_conv = self.mixed4c_pool_reduce_pre_relu_conv(mixed4c_pool)
        mixed4c_3x3_pre_relu_conv_pad = F.pad(mixed4c_3x3_bottleneck, (1, 1, 1, 1))
        mixed4c_3x3_pre_relu_conv = self.mixed4c_3x3_pre_relu_conv(mixed4c_3x3_pre_relu_conv_pad)
        mixed4c_5x5_pre_relu_conv_pad = F.pad(mixed4c_5x5_bottleneck, (2, 2, 2, 2))
        mixed4c_5x5_pre_relu_conv = self.mixed4c_5x5_pre_relu_conv(mixed4c_5x5_pre_relu_conv_pad)
        mixed4c_pool_reduce = self.mixed4c_pool_reduce(mixed4c_pool_reduce_pre_relu_conv)
        mixed4c_3x3 = self.mixed4c_3x3(mixed4c_3x3_pre_relu_conv)
        mixed4c_5x5 = self.mixed4c_5x5(mixed4c_5x5_pre_relu_conv)
        mixed4c = self.mixed4c((mixed4c_1x1, mixed4c_3x3, mixed4c_5x5, mixed4c_pool_reduce), 1)
        mixed4d_1x1_pre_relu_conv = self.mixed4d_1x1_pre_relu_conv(mixed4c)
        mixed4d_3x3_bottleneck_pre_relu_conv = self.mixed4d_3x3_bottleneck_pre_relu_conv(mixed4c)
        mixed4d_5x5_bottleneck_pre_relu_conv = self.mixed4d_5x5_bottleneck_pre_relu_conv(mixed4c)
        mixed4d_pool_pad = F.pad(mixed4c, (1, 1, 1, 1), value=float('-inf'))
        mixed4d_pool = self.mixed4d_pool(mixed4d_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4d_1x1 = self.mixed4d_1x1(mixed4d_1x1_pre_relu_conv)
        mixed4d_3x3_bottleneck = self.mixed4d_3x3_bottleneck(mixed4d_3x3_bottleneck_pre_relu_conv)
        mixed4d_5x5_bottleneck = self.mixed4d_5x5_bottleneck(mixed4d_5x5_bottleneck_pre_relu_conv)
        mixed4d_pool_reduce_pre_relu_conv = self.mixed4d_pool_reduce_pre_relu_conv(mixed4d_pool)
        mixed4d_3x3_pre_relu_conv_pad = F.pad(mixed4d_3x3_bottleneck, (1, 1, 1, 1))
        mixed4d_3x3_pre_relu_conv = self.mixed4d_3x3_pre_relu_conv(mixed4d_3x3_pre_relu_conv_pad)
        mixed4d_5x5_pre_relu_conv_pad = F.pad(mixed4d_5x5_bottleneck, (2, 2, 2, 2))
        mixed4d_5x5_pre_relu_conv = self.mixed4d_5x5_pre_relu_conv(mixed4d_5x5_pre_relu_conv_pad)
        mixed4d_pool_reduce = self.mixed4d_pool_reduce(mixed4d_pool_reduce_pre_relu_conv)
        mixed4d_3x3 = self.mixed4d_3x3(mixed4d_3x3_pre_relu_conv)
        mixed4d_5x5 = self.mixed4d_5x5(mixed4d_5x5_pre_relu_conv)
        mixed4d = self.mixed4d((mixed4d_1x1, mixed4d_3x3, mixed4d_5x5, mixed4d_pool_reduce), 1)
        mixed4e_1x1_pre_relu_conv = self.mixed4e_1x1_pre_relu_conv(mixed4d)
        mixed4e_3x3_bottleneck_pre_relu_conv = self.mixed4e_3x3_bottleneck_pre_relu_conv(mixed4d)
        mixed4e_5x5_bottleneck_pre_relu_conv = self.mixed4e_5x5_bottleneck_pre_relu_conv(mixed4d)
        mixed4e_pool_pad = F.pad(mixed4d, (1, 1, 1, 1), value=float('-inf'))
        mixed4e_pool = self.mixed4e_pool(mixed4e_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed4e_1x1 = self.mixed4e_1x1(mixed4e_1x1_pre_relu_conv)
        mixed4e_3x3_bottleneck = self.mixed4e_3x3_bottleneck(mixed4e_3x3_bottleneck_pre_relu_conv)
        mixed4e_5x5_bottleneck = self.mixed4e_5x5_bottleneck(mixed4e_5x5_bottleneck_pre_relu_conv)
        mixed4e_pool_reduce_pre_relu_conv = self.mixed4e_pool_reduce_pre_relu_conv(mixed4e_pool)
        mixed4e_3x3_pre_relu_conv_pad = F.pad(mixed4e_3x3_bottleneck, (1, 1, 1, 1))
        mixed4e_3x3_pre_relu_conv = self.mixed4e_3x3_pre_relu_conv(mixed4e_3x3_pre_relu_conv_pad)
        mixed4e_5x5_pre_relu_conv_pad = F.pad(mixed4e_5x5_bottleneck, (2, 2, 2, 2))
        mixed4e_5x5_pre_relu_conv = self.mixed4e_5x5_pre_relu_conv(mixed4e_5x5_pre_relu_conv_pad)
        mixed4e_pool_reduce = self.mixed4e_pool_reduce(mixed4e_pool_reduce_pre_relu_conv)
        mixed4e_3x3 = self.mixed4e_3x3(mixed4e_3x3_pre_relu_conv)
        mixed4e_5x5 = self.mixed4e_5x5(mixed4e_5x5_pre_relu_conv)
        mixed4e = self.mixed4e((mixed4e_1x1, mixed4e_3x3, mixed4e_5x5, mixed4e_pool_reduce), 1)
        maxpool10_pad = F.pad(mixed4e, (0, 1, 0, 1), value=float('-inf'))
        maxpool10 = self.maxpool10(maxpool10_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        mixed5a_1x1_pre_relu_conv = self.mixed5a_1x1_pre_relu_conv(maxpool10)
        mixed5a_3x3_bottleneck_pre_relu_conv = self.mixed5a_3x3_bottleneck_pre_relu_conv(maxpool10)
        mixed5a_5x5_bottleneck_pre_relu_conv = self.mixed5a_5x5_bottleneck_pre_relu_conv(maxpool10)
        mixed5a_pool_pad = F.pad(maxpool10, (1, 1, 1, 1), value=float('-inf'))
        mixed5a_pool = self.mixed5a_pool(mixed5a_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed5a_1x1 = self.mixed5a_1x1(mixed5a_1x1_pre_relu_conv)
        mixed5a_3x3_bottleneck = self.mixed5a_3x3_bottleneck(mixed5a_3x3_bottleneck_pre_relu_conv)
        mixed5a_5x5_bottleneck = self.mixed5a_5x5_bottleneck(mixed5a_5x5_bottleneck_pre_relu_conv)
        mixed5a_pool_reduce_pre_relu_conv = self.mixed5a_pool_reduce_pre_relu_conv(mixed5a_pool)
        mixed5a_3x3_pre_relu_conv_pad = F.pad(mixed5a_3x3_bottleneck, (1, 1, 1, 1))
        mixed5a_3x3_pre_relu_conv = self.mixed5a_3x3_pre_relu_conv(mixed5a_3x3_pre_relu_conv_pad)
        mixed5a_5x5_pre_relu_conv_pad = F.pad(mixed5a_5x5_bottleneck, (2, 2, 2, 2))
        mixed5a_5x5_pre_relu_conv = self.mixed5a_5x5_pre_relu_conv(mixed5a_5x5_pre_relu_conv_pad)
        mixed5a_pool_reduce = self.mixed5a_pool_reduce(mixed5a_pool_reduce_pre_relu_conv)
        mixed5a_3x3 = self.mixed5a_3x3(mixed5a_3x3_pre_relu_conv)
        mixed5a_5x5 = self.mixed5a_5x5(mixed5a_5x5_pre_relu_conv)
        mixed5a = self.mixed5a((mixed5a_1x1, mixed5a_3x3, mixed5a_5x5, mixed5a_pool_reduce), 1)
        mixed5b_1x1_pre_relu_conv = self.mixed5b_1x1_pre_relu_conv(mixed5a)
        mixed5b_3x3_bottleneck_pre_relu_conv = self.mixed5b_3x3_bottleneck_pre_relu_conv(mixed5a)
        mixed5b_5x5_bottleneck_pre_relu_conv = self.mixed5b_5x5_bottleneck_pre_relu_conv(mixed5a)
        mixed5b_pool_pad = F.pad(mixed5a, (1, 1, 1, 1), value=float('-inf'))
        mixed5b_pool = self.mixed5b_pool(mixed5b_pool_pad, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False)
        mixed5b_1x1 = self.mixed5b_1x1(mixed5b_1x1_pre_relu_conv)
        mixed5b_3x3_bottleneck = self.mixed5b_3x3_bottleneck(mixed5b_3x3_bottleneck_pre_relu_conv)
        mixed5b_5x5_bottleneck = self.mixed5b_5x5_bottleneck(mixed5b_5x5_bottleneck_pre_relu_conv)
        mixed5b_pool_reduce_pre_relu_conv = self.mixed5b_pool_reduce_pre_relu_conv(mixed5b_pool)
        mixed5b_3x3_pre_relu_conv_pad = F.pad(mixed5b_3x3_bottleneck, (1, 1, 1, 1))
        mixed5b_3x3_pre_relu_conv = self.mixed5b_3x3_pre_relu_conv(mixed5b_3x3_pre_relu_conv_pad)
        mixed5b_5x5_pre_relu_conv_pad = F.pad(mixed5b_5x5_bottleneck, (2, 2, 2, 2))
        mixed5b_5x5_pre_relu_conv = self.mixed5b_5x5_pre_relu_conv(mixed5b_5x5_pre_relu_conv_pad)
        mixed5b_pool_reduce = self.mixed5b_pool_reduce(mixed5b_pool_reduce_pre_relu_conv)
        mixed5b_3x3 = self.mixed5b_3x3(mixed5b_3x3_pre_relu_conv)
        mixed5b_5x5 = self.mixed5b_5x5(mixed5b_5x5_pre_relu_conv)
        mixed5b = self.mixed5b((mixed5b_1x1, mixed5b_3x3, mixed5b_5x5, mixed5b_pool_reduce), 1)
        avgpool0 = F.avg_pool2d(mixed5b, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False)
        avgpool0_reshape = torch.reshape(input=avgpool0, shape=(-1, 1024))
        softmax2_pre_activation_matmul = self.softmax2_pre_activation_matmul(avgpool0_reshape)
        softmax2 = self.softmax2(softmax2_pre_activation_matmul)
        return softmax2


class AdditionLayer(nn.Module):

    def forward(self, t_1, t_2):
        return t_1 + t_2


class MaxPool2dLayer(nn.Module):

    def forward(self, tensor, kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False):
        return F.max_pool2d(tensor, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


class PadLayer(nn.Module):

    def forward(self, tensor, padding=(1, 1, 1, 1), value=None):
        if value is None:
            return F.pad(tensor, padding)
        return F.pad(tensor, padding, value=value)


class ReluLayer(nn.Module):

    def forward(self, tensor):
        return F.relu(tensor)


class RedirectedReLU(torch.autograd.Function):
    """
    A workaround when there is no gradient flow from an initial random input
    See https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
    Note: this means that the gradient is technically "wrong"
    TODO: the original Lucid library has a more sophisticated way of doing this
    """

    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = grad_input[input_tensor < 0] * 0.1
        return grad_input


class RedirectedReluLayer(nn.Module):

    def forward(self, tensor):
        return RedirectedReLU.apply(tensor)


class SoftMaxLayer(nn.Module):

    def forward(self, tensor, dim=1):
        return F.softmax(tensor, dim=dim)


class DropoutLayer(nn.Module):

    def forward(self, tensor, p=0.4000000059604645, training=False, inplace=True):
        return F.dropout(input=tensor, p=p, training=training, inplace=inplace)


class CatLayer(nn.Module):

    def forward(self, tensor_list, dim=1):
        return torch.cat(tensor_list, dim)


class LocalResponseNormLayer(nn.Module):

    def forward(self, tensor, size=5, alpha=9.999999747378752e-05, beta=0.75, k=1.0):
        return F.local_response_norm(tensor, size=size, alpha=alpha, beta=beta, k=k)


class AVGPoolLayer(nn.Module):

    def forward(self, tensor, kernel_size=(7, 7), stride=(1, 1), padding=(0,), ceil_mode=False, count_include_pad=False):
        return F.avg_pool2d(tensor, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x / 0.67, x * x / 0.6], 1)


class View(nn.Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def load_statedict_from_online(name='fc6'):
    torchhome = torch.hub._get_torch_home()
    ckpthome = join(torchhome, 'checkpoints')
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, 'upconvGAN_%s.pt' % name)
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None, progress=True)
    SD = torch.load(filepath)
    return SD


load_urls = True


netsdir = '~'


class upconvGAN(nn.Module):

    def __init__(self, name='fc6', pretrained=True):
        """ `name`: can be ["fc6", "fc7", "fc8", "pool5"] """
        super(upconvGAN, self).__init__()
        self.name = name
        if name == 'fc6' or name == 'fc7':
            self.G = nn.Sequential(OrderedDict([('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)), ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)), ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)), ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('reshape', View((-1, 256, 4, 4))), ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))]))
            self.codelen = self.G[0].in_features
        elif name == 'fc8':
            self.G = nn.Sequential(OrderedDict([('defc7', nn.Linear(in_features=1000, out_features=4096, bias=True)), ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)), ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)), ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('reshape', View((-1, 256, 4, 4))), ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))]))
            self.codelen = self.G[0].in_features
        elif name == 'pool5':
            self.G = nn.Sequential(OrderedDict([('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))), ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)), ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))]))
            self.codelen = self.G[0].in_channels
        if pretrained:
            if load_urls:
                SDnew = load_statedict_from_online(name)
            else:
                savepath = {'fc6': join(netsdir, 'upconvGAN_%s.pt' % name), 'fc7': join(netsdir, 'upconvGAN_%s.pt' % name), 'fc8': join(netsdir, 'upconvGAN_%s.pt' % name), 'pool5': join(netsdir, 'upconvGAN_%s.pt' % name)}
                SD = torch.load(savepath[name])
                SDnew = OrderedDict()
                for name, W in SD.items():
                    name = name.replace('.1.', '.')
                    SDnew[name] = W
            self.G.load_state_dict(SDnew)
        self.G.requires_grad_(False)

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)[:, [2, 1, 0], :, :]
        return torch.clamp(raw + RGB_mean, 0, 255.0) / 255.0 * scale


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AVGPoolLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (AdditionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CompositeActivation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropoutLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (LocalResponseNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaxPool2dLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PadLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RedirectedReluLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReluLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftMaxLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_greentfrapp_lucent(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

