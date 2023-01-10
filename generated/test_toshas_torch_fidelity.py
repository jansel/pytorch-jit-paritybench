import sys
_module = sys.modules[__name__]
del sys
conf = _module
sngan_cifar10 = _module
reference_lpips = _module
reference_metric_fid_ttur = _module
reference_metric_isc_ttur = _module
reference_metric_kid_mmdgan = _module
setup = _module
test_feature_layer = _module
test_metric_isc_genmodel = _module
test_metric_ppl_genmodel = _module
test_convolution = _module
test_inception = _module
test_interpolation = _module
test_lpips = _module
test_metric_fid_determinism = _module
test_metric_fid_fidelity = _module
test_metric_isc_determinism = _module
test_metric_isc_fidelity = _module
test_metric_kid_determinism = _module
test_metric_kid_fidelity = _module
test_metrics_all = _module
torch_fidelity = _module
datasets = _module
defaults = _module
feature_extractor_base = _module
feature_extractor_inceptionv3 = _module
fidelity = _module
generative_model_base = _module
generative_model_modulewrapper = _module
generative_model_onnx = _module
helpers = _module
interpolate_compat_tensorflow = _module
metric_fid = _module
metric_isc = _module
metric_kid = _module
metric_ppl = _module
metrics = _module
noise = _module
registry = _module
sample_similarity_base = _module
sample_similarity_lpips = _module
utils = _module
version = _module
util_convert_inception_weights = _module
util_dump_dataset_as_images = _module

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


import torchvision


from torch.utils import tensorboard


from torch.hub import load_state_dict_from_url


from torch.hub import download_url_to_file


import numpy as np


import tensorflow as tf


import torch.nn.functional as F


from torch.utils.data import Dataset


from torchvision.datasets import CIFAR10


from torchvision.datasets import STL10


import torch.nn as nn


from abc import ABC


from abc import abstractmethod


import copy


import math


from torch.nn.modules.utils import _ntuple


import scipy.linalg


import torch.hub


from torch.utils.data import DataLoader


from tensorflow.python.framework import tensor_util


class Generator(torch.nn.Module):

    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.model = torch.nn.Sequential(torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(), torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=(1, 1)), torch.nn.Tanh())

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = 255 * (fake.clamp(-1, 1) * 0.5 + 0.5)
            fake = fake
        return fake


class Discriminator(torch.nn.Module):

    def __init__(self, sn=True):
        super(Discriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1, 1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        return self.fc(m.view(-1, 4 * 4 * 512))


URL_VGG16_LPIPS_STYLEGAN = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'


class LPIPS_reference(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg16 = load_state_dict_from_url(URL_VGG16_LPIPS_STYLEGAN, file_name='vgg16_stylegan.pth')

    def forward(self, in0, in1):
        out0 = self.vgg16(in0, resize_images=False, return_lpips=True)
        out1 = self.vgg16(in1, resize_images=False, return_lpips=True)
        out = (out0 - out1).square().sum(dim=-1)
        return out


def vassert(truecond, message):
    if not truecond:
        raise ValueError(message)


class FeatureExtractorBase(nn.Module):

    def __init__(self, name, features_list):
        """
        Base class for feature extractors that can be used in :func:`calculate_metrics`.

        Args:

            name (str): Unique name of the subclassed feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): List of feature names, provided by the subclassed feature extractor.
        """
        super(FeatureExtractorBase, self).__init__()
        vassert(type(name) is str, 'Feature extractor name must be a string')
        vassert(type(features_list) in (list, tuple), 'Wrong features list type')
        vassert(all(a in self.get_provided_features_list() for a in features_list), f'Requested features {tuple(features_list)} are not on the list provided by the selected feature extractor {self.get_provided_features_list()}')
        vassert(len(features_list) == len(set(features_list)), 'Duplicate features requested')
        self.name = name
        self.features_list = features_list

    def get_name(self):
        return self.name

    @staticmethod
    def get_provided_features_list():
        """
        Returns a tuple of feature names, extracted by the subclassed feature extractor.
        """
        raise NotImplementedError

    def get_requested_features_list(self):
        return self.features_list

    def convert_features_tuple_to_dict(self, features):
        vassert(type(features) is tuple and len(features) == len(self.features_list), 'Features must be the output of forward function')
        return dict((name, feature) for name, feature in zip(self.features_list, features))

    def forward(self, input):
        """
        Returns a tuple of tensors extracted from the `input`, in the same order as they are provided by
        `get_provided_features_list()`.
        """
        raise NotImplementedError


class BasicConv2d(nn.Module):
    """Original BasicConv2d block"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    """Block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    """Original block"""

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    """Block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    """Original block"""

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE_1(nn.Module):
    """First InceptionE block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels):
        super(InceptionE_1, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE_2(nn.Module):
    """Second InceptionE block from torchvision patched to be compatible with TensorFlow implementation"""

    def __init__(self, in_channels):
        super(InceptionE_2, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


URL_INCEPTION_V3 = 'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth'


def interpolate_bilinear_2d_like_tensorflow1x(input, size=None, scale_factor=None, align_corners=None, method='slow'):
    """Down/up samples the input to either the given :attr:`size` or the given :attr:`scale_factor`

    Epsilon-exact bilinear interpolation as it is implemented in TensorFlow 1.x:
    https://github.com/tensorflow/tensorflow/blob/f66daa493e7383052b2b44def2933f61faf196e0/tensorflow/core/kernels/image_resizer_state.h#L41
    https://github.com/tensorflow/tensorflow/blob/6795a8c3a3678fb805b6a8ba806af77ddfe61628/tensorflow/core/kernels/resize_bilinear_op.cc#L85
    as per proposal:
    https://github.com/pytorch/pytorch/issues/10604#issuecomment-465783319

    Related materials:
    https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://machinethink.net/blog/coreml-upsampling/

    Currently only 2D spatial sampling is supported, i.e. expected inputs are 4-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x height x width`.

    Args:
        input (Tensor): the input tensor
        size (Tuple[int, int]): output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        align_corners (bool, optional): Same meaning as in TensorFlow 1.x.
        method (str, optional):
            'slow' (1e-4 L_inf error on GPU, bit-exact on CPU, with checkerboard 32x32->299x299), or
            'fast' (1e-3 L_inf error on GPU and CPU, with checkerboard 32x32->299x299)
    """
    if method not in ('slow', 'fast'):
        raise ValueError('how_exact can only be one of "slow", "fast"')
    if input.dim() != 4:
        raise ValueError('input must be a 4-D tensor')
    if not torch.is_floating_point(input):
        raise ValueError('input must be of floating point dtype')
    if size is not None and (type(size) not in (tuple, list) or len(size) != 2):
        raise ValueError('size must be a list or a tuple of two elements')
    if align_corners is None:
        raise ValueError('align_corners is not specified (use this function for a complete determinism)')

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple) and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))
    is_tracing = torch._C._get_tracing_state()

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            if is_tracing:
                return [torch.tensor(i) for i in size]
            else:
                return size
        scale_factors = _ntuple(dim)(scale_factor)
        if is_tracing:
            return [torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()) for i in range(dim)]
        else:
            return [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]

    def tf_calculate_resize_scale(in_size, out_size):
        if align_corners:
            if is_tracing:
                return (in_size - 1) / (out_size.float() - 1).clamp(min=1)
            else:
                return (in_size - 1) / max(1, out_size - 1)
        elif is_tracing:
            return in_size / out_size.float()
        else:
            return in_size / out_size
    out_size = _output_size(2)
    scale_x = tf_calculate_resize_scale(input.shape[3], out_size[1])
    scale_y = tf_calculate_resize_scale(input.shape[2], out_size[0])

    def resample_using_grid_sample():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * (2 * scale_x / (input.shape[3] - 1)) - 1
        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * (2 * scale_y / (input.shape[2] - 1)) - 1
        grid_x = grid_x.view(1, out_size[1]).repeat(out_size[0], 1)
        grid_y = grid_y.view(out_size[0], 1).repeat(1, out_size[1])
        grid_xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=2).unsqueeze(0)
        grid_xy = grid_xy.repeat(input.shape[0], 1, 1, 1)
        out = F.grid_sample(input, grid_xy, mode='bilinear', padding_mode='border', align_corners=True)
        return out

    def resample_manually():
        grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
        grid_x = grid_x * torch.tensor(scale_x, dtype=torch.float32)
        grid_x_lo = grid_x.long()
        grid_x_hi = (grid_x_lo + 1).clamp_max(input.shape[3] - 1)
        grid_dx = grid_x - grid_x_lo.float()
        grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
        grid_y = grid_y * torch.tensor(scale_y, dtype=torch.float32)
        grid_y_lo = grid_y.long()
        grid_y_hi = (grid_y_lo + 1).clamp_max(input.shape[2] - 1)
        grid_dy = grid_y - grid_y_lo.float()
        in_00 = input[:, :, grid_y_lo, :][:, :, :, grid_x_lo]
        in_01 = input[:, :, grid_y_lo, :][:, :, :, grid_x_hi]
        in_10 = input[:, :, grid_y_hi, :][:, :, :, grid_x_lo]
        in_11 = input[:, :, grid_y_hi, :][:, :, :, grid_x_hi]
        in_0 = in_00 + (in_01 - in_00) * grid_dx.view(1, 1, 1, out_size[1])
        in_1 = in_10 + (in_11 - in_10) * grid_dx.view(1, 1, 1, out_size[1])
        out = in_0 + (in_1 - in_0) * grid_dy.view(1, 1, out_size[0], 1)
        return out
    if method == 'slow':
        out = resample_manually()
    else:
        out = resample_using_grid_sample()
    return out


class FeatureExtractorInceptionV3(FeatureExtractorBase):
    INPUT_IMAGE_SIZE = 299

    def __init__(self, name, features_list, feature_extractor_weights_path=None, **kwargs):
        """
        InceptionV3 feature extractor for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - '64'
                - '192'
                - '768'
                - '2048'
                - 'logits_unbiased'
                - 'logits'

            feature_extractor_weights_path (str): Path to the pretrained InceptionV3 model weights in PyTorch format.
                Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.
        """
        super(FeatureExtractorInceptionV3, self).__init__(name, features_list)
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 1008)
        if feature_extractor_weights_path is None:
            with redirect_stdout(sys.stderr):
                state_dict = load_state_dict_from_url(URL_INCEPTION_V3, progress=True)
        else:
            state_dict = torch.load(feature_extractor_weights_path)
        self.load_state_dict(state_dict)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, 'Expecting image as torch.Tensor with dtype=torch.uint8')
        features = {}
        remaining_features = self.features_list.copy()
        x = x.float()
        x = interpolate_bilinear_2d_like_tensorflow1x(x, size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), align_corners=False)
        x = (x - 128) / 128
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)
        if '64' in remaining_features:
            features['64'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('64')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)
        if '192' in remaining_features:
            features['192'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('192')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if '768' in remaining_features:
            features['768'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('768')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        if '2048' in remaining_features:
            features['2048'] = x
            remaining_features.remove('2048')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
        if 'logits_unbiased' in remaining_features:
            x = x.mm(self.fc.weight.T)
            features['logits_unbiased'] = x
            remaining_features.remove('logits_unbiased')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)
            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)
        features['logits'] = x
        return tuple(features[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return '64', '192', '768', '2048', 'logits_unbiased', 'logits'


class GenerativeModelBase(ABC, torch.nn.Module):
    """
    Base class for generative models that can be used as inputs in :func:`calculate_metrics`.
    """

    @property
    @abstractmethod
    def z_size(self):
        """
        Size of the noise dimension of the generative model (positive integer).
        """
        pass

    @property
    @abstractmethod
    def z_type(self):
        """
        Type of the noise used by the generative model (see :ref:`registry <Registry>` for a list of preregistered noise
        types, see :func:`register_noise_source` for registering a new noise type).
        """
        pass

    @property
    @abstractmethod
    def num_classes(self):
        """
        Number of classes used by a conditional generative model. Must return zero for unconditional models.
        """
        pass


class GenerativeModelModuleWrapper(GenerativeModelBase):

    def __init__(self, module, z_size, z_type, num_classes, make_copy=False, make_eval=True, cuda=None):
        """
        Wraps any generative model :class:`torch.nn.Module`, implements the :class:`GenerativeModelBase` interface, and
        provides a few convenience functions.

        Args:

            module (torch.nn.Module): A generative model module, taking a batch of noise samples, and producing
                generative samples.

            z_size (int): Size of the noise dimension of the generative model (positive integer).

            z_type (str): Type of the noise used by the generative model (see :ref:`registry <Registry>` for a list of
                preregistered noise types, see :func:`register_noise_source` for registering a new noise type).

            num_classes (int): Number of classes used by a conditional generative model. Must return zero for
                unconditional models.

            make_copy (bool): Makes a copy of the model weights if `True`. Default: `False`.

            make_eval (bool): Switches to :class:`torch.nn.Module` evaluation mode upon construction if `True`. Default:
                `True`.

            cuda (bool): Moves the module on a CUDA device if `True`, moves to CPU if `False`, does nothing if `None`.
                Default: `None`.
        """
        super().__init__()
        vassert(isinstance(module, torch.nn.Module), 'Not an instance of torch.nn.Module')
        vassert(type(z_size) is int and z_size > 0, 'z_size must be a positive integer')
        vassert(z_type in ('normal', 'unit', 'uniform_0_1'), f'z_type={z_type} not implemented')
        vassert(type(num_classes) is int and num_classes >= 0, 'num_classes must be a non-negative integer')
        self.module = module
        if make_copy:
            self.module = copy.deepcopy(self.module)
        if make_eval:
            self.module.eval()
        if cuda is not None:
            if cuda:
                self.module = self.module
            else:
                self.module = self.module.cpu()
        self._z_size = z_size
        self._z_type = z_type
        self._num_classes = num_classes

    @property
    def z_size(self):
        return self._z_size

    @property
    def z_type(self):
        return self._z_type

    @property
    def num_classes(self):
        return self._num_classes

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class SampleSimilarityBase(nn.Module):

    def __init__(self, name):
        """
        Base class for samples similarity measures that can be used in :func:`calculate_metrics`.

        Args:

            name (str): Unique name of the subclassed sample similarity measure, must be the same as used in
                :func:`register_sample_similarity`.
        """
        super(SampleSimilarityBase, self).__init__()
        vassert(type(name) is str, 'Sample similarity name must be a string')
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, *args):
        """
        Returns the value of sample similarity between the inputs.
        """
        raise NotImplementedError


URL_VGG16_BASE = 'https://download.pytorch.org/models/vgg16-397923af.pth'


class VGG16features(torch.nn.Module):

    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=False)
        with redirect_stdout(sys.stderr):
            vgg_pretrained_features.load_state_dict(load_state_dict_from_url(URL_VGG16_BASE))
        vgg_pretrained_features = vgg_pretrained_features.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


class NetLinLayer(nn.Module):

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


URL_VGG16_LPIPS = 'https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth'


def normalize_tensor(in_features, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_features ** 2, dim=1, keepdim=True))
    return in_features / (norm_factor + eps)


def spatial_average(in_tensor):
    return in_tensor.mean([2, 3]).squeeze(1)


class SampleSimilarityLPIPS(SampleSimilarityBase):
    SUPPORTED_DTYPES = {'uint8': torch.uint8, 'float32': torch.float32}

    def __init__(self, name, sample_similarity_resize=None, sample_similarity_dtype=None, **kwargs):
        """
        LPIPS sample similarity measure for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the sample similarity measure, must be the same as used in
                :func:`register_sample_similarity`.

            sample_similarity_resize (int or None): Resizes inputs to this size if set, keeps as is if `None`.

            sample_similarity_dtype (str): Coerces tensor dtype to one of the following: 'uint8', 'float32'.
                This is useful when the inputs are generated by a generative model, to ensure the proper data range and
                quantization.
        """
        super(SampleSimilarityLPIPS, self).__init__(name)
        self.sample_similarity_resize = sample_similarity_resize
        self.sample_similarity_dtype = sample_similarity_dtype
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        with redirect_stdout(sys.stderr):
            state_dict = load_state_dict_from_url(URL_VGG16_LPIPS, map_location='cpu', progress=True)
        self.load_state_dict(state_dict)
        self.net = VGG16features()
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize(x):
        mean_rescaled = (1 + torch.tensor([-0.03, -0.088, -0.188], device=x.device)[None, :, None, None]) * 255 / 2
        inv_std_rescaled = 2 / (torch.tensor([0.458, 0.448, 0.45], device=x.device)[None, :, None, None] * 255)
        x = (x.float() - mean_rescaled) * inv_std_rescaled
        return x

    @staticmethod
    def resize(x, size):
        if x.shape[-1] > size and x.shape[-2] > size:
            x = torch.nn.functional.interpolate(x, (size, size), mode='area')
        else:
            x = torch.nn.functional.interpolate(x, (size, size), mode='bilinear', align_corners=False)
        return x

    def forward(self, in0, in1):
        vassert(torch.is_tensor(in0) and torch.is_tensor(in1), 'Inputs must be torch tensors')
        vassert(in0.dim() == 4 and in0.shape[1] == 3, 'Input 0 is not Bx3xHxW')
        vassert(in1.dim() == 4 and in1.shape[1] == 3, 'Input 1 is not Bx3xHxW')
        if self.sample_similarity_dtype is not None:
            dtype = self.SUPPORTED_DTYPES.get(self.sample_similarity_dtype, None)
            vassert(dtype is not None and in0.dtype == dtype and in1.dtype == dtype, f'Unexpected input dtype ({in0.dtype})')
        in0_input = self.normalize(in0)
        in1_input = self.normalize(in1)
        if self.sample_similarity_resize is not None:
            in0_input = self.resize(in0_input, self.sample_similarity_resize)
            in1_input = self.resize(in1_input, self.sample_similarity_resize)
        outs0 = self.net.forward(in0_input)
        outs1 = self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(self.lins[kk].model(diffs[kk])) for kk in range(self.L)]
        val = sum(res)
        return val


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Generator,
     lambda: ([], {'z_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionB,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionD,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_toshas_torch_fidelity(_paritybench_base):
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

