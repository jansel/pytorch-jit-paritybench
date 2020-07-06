import sys
_module = sys.modules[__name__]
del sys
dataset = _module
eval = _module
model = _module
pano = _module
pano_lsd_align = _module
pano_opt = _module
torch2pytorch_data = _module
torch2pytorch_pretrained_weight = _module
train = _module
utils = _module
utils_eval = _module
visual = _module
visual_3d_layout = _module
visual_preprocess = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.utils.data as data


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


import torchvision


import torch.optim as optim


from scipy.io import loadmat


from scipy.ndimage import convolve


from itertools import chain


from torch import optim


def conv3x3(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1), nn.ReLU(inplace=True))


def conv3x3_down(in_planes, out_planes):
    return nn.Sequential(conv3x3(in_planes, out_planes), nn.MaxPool2d(kernel_size=2, stride=2))


class Encoder(nn.Module):

    def __init__(self, in_planes=6):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([conv3x3_down(in_planes, 32), conv3x3_down(32, 64), conv3x3_down(64, 128), conv3x3_down(128, 256), conv3x3_down(256, 512), conv3x3_down(512, 1024), conv3x3_down(1024, 2048)])

    def forward(self, x):
        conv_out = []
        for conv in self.convs:
            x = conv(x)
            conv_out.append(x)
        return conv_out


class Decoder(nn.Module):

    def __init__(self, skip_num=2, out_planes=3):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList([conv3x3(2048, 1024), conv3x3(1024 * skip_num, 512), conv3x3(512 * skip_num, 256), conv3x3(256 * skip_num, 128), conv3x3(128 * skip_num, 64), conv3x3(64 * skip_num, 32)])
        self.last_conv = nn.Conv2d(32 * skip_num, out_planes, kernel_size=3, padding=1)

    def forward(self, f_list):
        conv_out = []
        f_last = f_list[0]
        for conv, f in zip(self.convs, f_list[1:]):
            f_last = F.interpolate(f_last, scale_factor=2, mode='nearest')
            f_last = conv(f_last)
            f_last = torch.cat([f_last, f], dim=1)
            conv_out.append(f_last)
        conv_out.append(self.last_conv(F.interpolate(f_last, scale_factor=2, mode='nearest')))
        return conv_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 128, 128])], {}),
     True),
]

class Test_sunset1995_pytorch_layoutnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

