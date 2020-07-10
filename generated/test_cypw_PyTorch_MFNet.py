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
xrange = range
wraps = functools.wraps


import numpy as np


import torch.utils.data as data


import logging


import torch


from collections import OrderedDict


import torch.nn as nn


import time


import torch.backends.cudnn as cudnn


import torch.nn.parallel


import torch.distributed as dist


import torchvision


class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter, kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid / 4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_ix, kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix, num_filter=num_in, kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_mid, kernel=(kt, 3, 3), pad=(pt, 1, 1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 3, 3), pad=(0, 1, 1), g=g)
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0), stride=stride)

    def forward(self, x):
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)
        h = self.conv_m1(x_in)
        h = self.conv_m2(h)
        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)
        return h + x


class MFNET_3D(nn.Module):

    def __init__(self, num_classes, pretrained=False, **kwargs):
        super(MFNET_3D, self).__init__()
        groups = 16
        k_sec = {(2): 3, (3): 4, (4): 6, (5): 3}
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv3d(3, conv1_num_out, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2), bias=False)), ('bn', nn.BatchNorm3d(conv1_num_out)), ('relu', nn.ReLU(inplace=True))]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(OrderedDict([('B%02d' % i, MF_UNIT(num_in=conv1_num_out if i == 1 else conv2_num_out, num_mid=num_mid, num_out=conv2_num_out, stride=(2, 1, 1) if i == 1 else (1, 1, 1), g=groups, first_block=i == 1)) for i in range(1, k_sec[2] + 1)]))
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([('B%02d' % i, MF_UNIT(num_in=conv2_num_out if i == 1 else conv3_num_out, num_mid=num_mid, num_out=conv3_num_out, stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=i == 1)) for i in range(1, k_sec[3] + 1)]))
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([('B%02d' % i, MF_UNIT(num_in=conv3_num_out if i == 1 else conv4_num_out, num_mid=num_mid, num_out=conv4_num_out, stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=i == 1)) for i in range(1, k_sec[4] + 1)]))
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([('B%02d' % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out, num_mid=num_mid, num_out=conv5_num_out, stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=i == 1)) for i in range(1, k_sec[5] + 1)]))
        self.tail = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(conv5_num_out)), ('relu', nn.ReLU(inplace=True))]))
        self.globalpool = nn.Sequential(OrderedDict([('avg', nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1)))]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)
        initializer.xavier(net=self)
        if pretrained:
            import torch
            load_method = 'inflation'
            pretrained_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/MFNet2D_ImageNet1k-0000.pth')
            logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
            assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
            state_dict_2d = torch.load(pretrained_model)
            initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method=load_method)
        else:
            logging.info('Network:: graph initialized, use random inilization!')

    def forward(self, x):
        assert x.shape[2] == 16
        h = self.conv1(x)
        h = self.maxpool(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.tail(h)
        h = self.globalpool(h)
        h = h.view(h.shape[0], -1)
        h = self.classifier(h)
        return h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BN_AC_CONV3D,
     lambda: ([], {'num_in': 4, 'num_filter': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (MF_UNIT,
     lambda: ([], {'num_in': 4, 'num_mid': 4, 'num_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_cypw_PyTorch_MFNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

