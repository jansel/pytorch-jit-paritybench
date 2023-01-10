import sys
_module = sys.modules[__name__]
del sys
convert = _module
module_edge_detector = _module
module_photo2pixel = _module
module_pixel_effect = _module
img_common_util = _module

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


class EdgeDetectorModule(nn.Module):

    def __init__(self):
        super(EdgeDetectorModule, self).__init__()
        self.pad = nn.ReflectionPad2d(padding=(1, 1, 1, 1))
        kernel_sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_h = torch.from_numpy(kernel_sobel_h).reshape([1, 1, 3, 3]).repeat([3, 1, 1, 1])
        self.conv_h = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)
        self.conv_h.weight = nn.Parameter(kernel_sobel_h)
        kernel_sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_v = torch.from_numpy(kernel_sobel_v).reshape([1, 1, 3, 3]).repeat([3, 1, 1, 1])
        self.conv_v = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)
        self.conv_v.weight = nn.Parameter(kernel_sobel_v)

    def forward(self, rgb, param_edge_thresh, param_edge_dilate):
        """
        :param rgb: [1, c(3), H, W]
        :param param_edge_thresh: int
        :param param_edge_dilate: odd number
        :return: [1,c(1),H,W]
        """
        rgb = self.pad(rgb)
        edge_h = self.conv_h(rgb)
        edge_w = self.conv_v(rgb)
        edge = torch.stack([torch.abs(edge_h), torch.abs(edge_w)], dim=1)
        edge = torch.max(edge, dim=1)[0]
        edge = torch.mean(edge, dim=1, keepdim=True)
        edge = torch.gt(edge, param_edge_thresh).float()
        edge = F.max_pool2d(edge, kernel_size=param_edge_dilate, stride=1, padding=param_edge_dilate // 2)
        return edge


class PixelEffectModule(nn.Module):

    def __init__(self):
        super(PixelEffectModule, self).__init__()

    def create_mask_by_idx(self, idx_z, max_z):
        """
        :param idx_z: [H, W]
        :return:
        """
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z])
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        """
        :param data: [h,w,C]
        :param idx_z: [h,w]
        :return:
        """
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def forward(self, rgb, param_num_bins, param_kernel_size, param_pixel_size):
        """
        :param rgb:[b(1), c(3), H, W]
        :return: [b(1), c(3), H, W]
        """
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256.0 * param_num_bins
        intensity_idx = intensity_idx.long()
        intensity = self.create_mask_by_idx(intensity_idx, max_z=param_num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0)
        r, g, b = r * intensity, g * intensity, b * intensity
        kernel_conv = torch.ones([param_num_bins, 1, param_kernel_size, param_kernel_size])
        r = F.conv2d(input=r, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        g = F.conv2d(input=g, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        b = F.conv2d(input=b, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        intensity = F.conv2d(input=intensity, weight=kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size, groups=param_num_bins, bias=None)[0, :, :, :]
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)
        r = torch.permute(r, dims=[1, 2, 0])
        g = torch.permute(g, dims=[1, 2, 0])
        b = torch.permute(b, dims=[1, 2, 0])
        r = self.select_by_idx(r, intensity_argmax)
        g = self.select_by_idx(g, intensity_argmax)
        b = self.select_by_idx(b, intensity_argmax)
        r = r / intensity_max
        g = g / intensity_max
        b = b / intensity_max
        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb = F.interpolate(result_rgb, scale_factor=param_pixel_size)
        return result_rgb


class Photo2PixelModel(nn.Module):

    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()
        self.module_edge_detect = EdgeDetectorModule()

    def forward(self, rgb, param_kernel_size=10, param_pixel_size=16, param_edge_thresh=112):
        """
        :param rgb: [b(1), c(3), H, W]
        :param param_kernel_size:
        :param param_pixel_size:
        :param param_edge_thresh: 0~255
        :return:
        """
        rgb = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)
        edge_mask = self.module_edge_detect(rgb, param_edge_thresh, param_edge_dilate=3)
        rgb = torch.masked_fill(rgb, torch.gt(edge_mask, 0.5), 0)
        return rgb


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Photo2PixelModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Jzou44_photo2pixel(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

