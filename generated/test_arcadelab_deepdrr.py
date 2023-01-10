import sys
_module = sys.modules[__name__]
del sys
deepdrr = _module
annotations = _module
line_annotation = _module
device = _module
carm = _module
mobile_carm = _module
downsample_tool = _module
geo = _module
camera_intrinsic_transform = _module
camera_projection = _module
core = _module
exceptions = _module
random = _module
load_dicom = _module
load_dicom_tool = _module
logging = _module
network_segmentation = _module
projector = _module
analytic_generators = _module
conv_to_mcgpu = _module
cuda_scatter_structs = _module
mass_attenuation = _module
material_coefficients = _module
mcgpu_compton_data = _module
mcgpu_density = _module
PMMA_compton_data = _module
mcgpu_incoherent_scatter_data = _module
adipose_compton_data = _module
air_compton_data = _module
blood_compton_data = _module
bone_compton_data = _module
brain_compton_data = _module
breast_compton_data = _module
cartilage_compton_data = _module
connective_compton_data = _module
glands_others_compton_data = _module
liver_compton_data = _module
lung_compton_data = _module
muscle_compton_data = _module
red_marrow_compton_data = _module
skin_compton_data = _module
soft_tissue_compton_data = _module
stomach_intestines_compton_data = _module
titanium_compton_data = _module
water_compton_data = _module
PMMA_mfp = _module
mcgpu_mean_free_path_data = _module
adipose_mfp = _module
air_mfp = _module
blood_mfp = _module
bone_mfp = _module
brain_mfp = _module
breast_mfp = _module
cartilage_mfp = _module
connective_mfp = _module
glands_others_mfp = _module
liver_mfp = _module
lung_mfp = _module
muscle_mfp = _module
red_marrow_mfp = _module
skin_mfp = _module
soft_tissue_mfp = _module
stomach_intestines_mfp = _module
titanium_mfp = _module
water_mfp = _module
mcgpu_mfp_data = _module
PMMA_rita_params = _module
mcgpu_rita_params = _module
adipose_rita_params = _module
air_rita_params = _module
blood_rita_params = _module
bone_rita_params = _module
brain_rita_params = _module
breast_rita_params = _module
cartilage_rita_params = _module
connective_rita_params = _module
glands_others_rita_params = _module
liver_rita_params = _module
lung_rita_params = _module
muscle_rita_params = _module
red_marrow_rita_params = _module
skin_rita_params = _module
soft_tissue_rita_params = _module
stomach_intestines_rita_params = _module
titanium_rita_params = _module
water_rita_params = _module
mcgpu_rita_samplers = _module
plane_surface = _module
projector = _module
rita = _module
scatter = _module
spectral_data = _module
segmentation = _module
utils = _module
data_utils = _module
image_utils = _module
mesh_utils = _module
test_utils = _module
vis = _module
vol = _module
kwire = _module
volume = _module
conf = _module
example_projector = _module
geometry_testing = _module
setup = _module
test_core = _module
test_kwire = _module
test_mcgpu = _module
test_multivolume = _module
test_phantom = _module
test_scatter = _module

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


import logging


import time


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import warnings


import numpy as np


from torch.autograd import Variable


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class InputTransition(nn.Module):

    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(4, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


def passthrough(x, **kwargs):
    return x


class DownTransition(nn.Module):

    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(self, inChans, outChans, elu, nll):
        self.outChans = outChans
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        self.softmax = F.softmax

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.softmax(out, dim=1)
        return out


class VNet(nn.Module):

    def __init__(self, elu=False, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, 3, elu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (DownTransition,
     lambda: ([], {'inChans': 4, 'nConvs': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (InputTransition,
     lambda: ([], {'outChans': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LUConv,
     lambda: ([], {'nchan': 4, 'elu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (OutputTransition,
     lambda: ([], {'inChans': 4, 'outChans': 4, 'elu': 4, 'nll': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_arcadelab_deepdrr(_paritybench_base):
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

