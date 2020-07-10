import sys
_module = sys.modules[__name__]
del sys
data_gen = _module
gen_bone_data = _module
kinetics_gendata = _module
merge_joint_bone_data = _module
ntu_gen_bone_data = _module
ntu_gen_joint_data = _module
ntu_gen_motion_data = _module
preprocess = _module
rotation = _module
ensemble = _module
feeders = _module
feeder = _module
tools = _module
graph = _module
directed_ntu_rgb_d = _module
kinetics = _module
ntu_rgb_d = _module
main = _module
model = _module
dgnn = _module

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


from torch.utils.data import Dataset


import torch


import time


from collections import OrderedDict


from collections import defaultdict


import torch.nn as nn


import torch.optim as optim


from torch.autograd import Variable


from torch.optim.lr_scheduler import MultiStepLR


import random


import inspect


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import math


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


class TemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BiTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.tempconv = TemporalConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv(fv), self.tempconv(fe)


class DGNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, source_M, target_M):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        self.source_M = nn.Parameter(torch.from_numpy(source_M.astype('float32')))
        self.target_M = nn.Parameter(torch.from_numpy(target_M.astype('float32')))
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)
        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fv, fe):
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape
        fv = fv.view(N, -1, V_node)
        fe = fe.view(N, -1, V_edge)
        fe_in_agg = torch.einsum('nce,ev->ncv', fe, self.source_M.transpose(0, 1))
        fe_out_agg = torch.einsum('nce,ev->ncv', fe, self.target_M.transpose(0, 1))
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)
        fvp = fvp.view(N, 3 * C, T, V_node).contiguous().permute(0, 2, 3, 1)
        fvp = self.H_v(fvp).permute(0, 3, 1, 2)
        fvp = self.bn_v(fvp)
        fvp = self.relu(fvp)
        fv_in_agg = torch.einsum('ncv,ve->nce', fv, self.source_M)
        fv_out_agg = torch.einsum('ncv,ve->nce', fv, self.target_M)
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)
        fep = fep.view(N, 3 * C, T, V_edge).contiguous().permute(0, 2, 3, 1)
        fep = self.H_e(fep).permute(0, 3, 1, 2)
        fep = self.bn_e(fep)
        fep = self.relu(fep)
        return fvp, fep


class GraphTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.dgn = DGNBlock(in_channels, out_channels, source_M, target_M)
        self.tcn = BiTemporalConv(out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = BiTemporalConv(in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv_res, fe_res = self.residual(fv, fe)
        fv, fe = self.dgn(fv, fe)
        fv, fe = self.tcn(fv, fe)
        fv += fv_res
        fe += fe_res
        return self.relu(fv), self.relu(fe)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):

    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        source_M, target_M = self.graph.source_M, self.graph.target_M
        self.data_bn_v = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_e = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = GraphTemporalConv(3, 64, source_M, target_M, residual=False)
        self.l2 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l3 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l4 = GraphTemporalConv(64, 64, source_M, target_M)
        self.l5 = GraphTemporalConv(64, 128, source_M, target_M, stride=2)
        self.l6 = GraphTemporalConv(128, 128, source_M, target_M)
        self.l7 = GraphTemporalConv(128, 128, source_M, target_M)
        self.l8 = GraphTemporalConv(128, 256, source_M, target_M, stride=2)
        self.l9 = GraphTemporalConv(256, 256, source_M, target_M)
        self.l10 = GraphTemporalConv(256, 256, source_M, target_M)
        self.fc = nn.Linear(256 * 2, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn_v, 1)
        bn_init(self.data_bn_e, 1)

        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        for module in self.modules():
            None
            None
            None
        None

    def forward(self, fv, fe):
        N, C, T, V_node, M = fv.shape
        _, _, _, V_edge, _ = fe.shape
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * C, T)
        fv = self.data_bn_v(fv)
        fv = fv.view(N, M, V_node, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_node)
        fe = fe.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_edge * C, T)
        fe = self.data_bn_e(fe)
        fe = fe.view(N, M, V_edge, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_edge)
        fv, fe = self.l1(fv, fe)
        fv, fe = self.l2(fv, fe)
        fv, fe = self.l3(fv, fe)
        fv, fe = self.l4(fv, fe)
        fv, fe = self.l5(fv, fe)
        fv, fe = self.l6(fv, fe)
        fv, fe = self.l7(fv, fe)
        fv, fe = self.l8(fv, fe)
        fv, fe = self.l9(fv, fe)
        fv, fe = self.l10(fv, fe)
        out_channels = fv.size(1)
        fv = fv.view(N, M, out_channels, -1).mean(3).mean(1)
        fe = fe.view(N, M, out_channels, -1).mean(3).mean(1)
        out = torch.cat((fv, fe), dim=-1)
        return self.fc(out)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiTemporalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemporalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kenziyuliu_Unofficial_DGNN_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

