import sys
_module = sys.modules[__name__]
del sys
data = _module
image_iterator = _module
image_transforms = _module
iterator_factory = _module
video_iterator = _module
video_sampler = _module
video_transforms = _module
convert_videos = _module
remove_spaces = _module
dataset = _module
config = _module
network = _module
initializer = _module
mfnet_3d = _module
symbol_builder = _module
evaluate_speed = _module
evaluate_video_ucf101_split1 = _module
train = _module
callback = _module
lr_scheduler = _module
metric = _module
model = _module
train_hmdb51 = _module
train_kinetics = _module
train_model = _module
train_ucf101 = _module

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


import logging


import numpy as np


import torch


from collections import OrderedDict


import torch.nn as nn


import time


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.distributed as dist


class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter, kernel=(1, 1, 1), pad=(0, 0, 0),
        stride=(1, 1, 1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel,
            padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1),
        first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid / 4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_ix,
            kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix, num_filter=num_in,
            kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_mid,
            kernel=(kt, 3, 3), pad=(pt, 1, 1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out,
                kernel=(1, 1, 1), pad=(0, 0, 0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out,
                kernel=(1, 3, 3), pad=(0, 1, 1), g=g)
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_out,
                kernel=(1, 1, 1), pad=(0, 0, 0), stride=stride)

    def forward(self, x):
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)
        h = self.conv_m1(x_in)
        h = self.conv_m2(h)
        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)
        return h + x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cypw_PyTorch_MFNet(_paritybench_base):
    pass
