import sys
_module = sys.modules[__name__]
del sys
binvox_rw = _module
get_gt_LOD = _module
get_gt_LOD_sdf_only = _module
setup = _module
utils = _module
simplify_obj = _module
get_mesh = _module
dataset = _module
eval_cd_nc_f1_ecd_ef1 = _module
eval_v_t_count = _module
generate_loss_float_gradient_code = _module
loss_float_gradient_code = _module
main = _module
model = _module
template = _module
update_template_to_configs = _module

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


import numpy as np


import time


import torch


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


class resnet_block(nn.Module):

    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class CNN_3d_rec7(nn.Module):

    def __init__(self, out_bool, out_float):
        super(CNN_3d_rec7, self).__init__()
        self.ef_dim = 256
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_3 = resnet_block(self.ef_dim)
        self.res_4 = resnet_block(self.ef_dim)
        self.res_5 = resnet_block(self.ef_dim)
        self.res_6 = resnet_block(self.ef_dim)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            self.conv_out_bool = nn.Conv3d(self.ef_dim, 5, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 51, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


class CNN_3d_rec15(nn.Module):

    def __init__(self, out_bool, out_float):
        super(CNN_3d_rec15, self).__init__()
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_1 = resnet_block(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_2 = resnet_block(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_3 = resnet_block(self.ef_dim)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.res_4 = resnet_block(self.ef_dim)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_5 = resnet_block(self.ef_dim)
        self.conv_5 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_6 = resnet_block(self.ef_dim)
        self.conv_6 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)
        self.res_9 = resnet_block(self.ef_dim)
        self.res_10 = resnet_block(self.ef_dim)
        self.conv_7 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_8 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        if self.out_bool:
            self.conv_out_bool = nn.Conv3d(self.ef_dim, 5, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 51, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_3(out)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_4(out)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_5(out)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_6(out)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.res_7(out)
        out = self.res_8(out)
        out = self.res_9(out)
        out = self.res_10(out)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNN_3d_rec15,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (CNN_3d_rec7,
     lambda: ([], {'out_bool': 4, 'out_float': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (resnet_block,
     lambda: ([], {'ef_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_czq142857_NMC(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

