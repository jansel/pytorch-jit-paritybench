import sys
_module = sys.modules[__name__]
del sys
eval_proposal = _module
utils = _module
data_process = _module
ldb_process = _module
dataset = _module
eval = _module
loss_function = _module
main = _module
models = _module
opts = _module
post_processing = _module

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


import torch


import torch.nn.functional as F


import torch.nn.parallel


import torch.optim as optim


import math


import torch.nn as nn


class BMN(nn.Module):

    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt['temporal_scale']
        self.prop_boundary_ratio = opt['prop_boundary_ratio']
        self.num_sample = opt['num_sample']
        self.num_sample_perbin = opt['num_sample_perbin']
        self.feat_dim = opt['feat_dim']
        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512
        self._get_interp1d_mask()
        self.x_1d_b = nn.Sequential(nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True), nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True))
        self.x_1d_s = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True), nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())
        self.x_1d_e = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4), nn.ReLU(inplace=True), nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1), nn.Sigmoid())
        self.x_1d_p = nn.Sequential(nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.x_3d_p = nn.Sequential(nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)), nn.ReLU(inplace=True))
        self.x_2d_p = nn.Sequential(nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0], input_size[1], self.num_sample, self.tscale, self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [(seg_xmin + plen_sample * ii) for ii in range(num_sample * num_sample_perbin)]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= tscale - 1 and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= tscale - 1 and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.tscale):
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(sample_xmin, sample_xmax, self.tscale, self.num_sample, self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BMN,
     lambda: ([], {'opt': _mock_config(temporal_scale=1, prop_boundary_ratio=4, num_sample=4, num_sample_perbin=4, feat_dim=4)}),
     lambda: ([torch.rand([4, 4, 1])], {}),
     True),
]

class Test_JJBOY_BMN_Boundary_Matching_Network(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

