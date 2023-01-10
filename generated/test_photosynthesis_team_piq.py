import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
feature_metrics = _module
image_metrics = _module
piq = _module
base = _module
brisque = _module
dss = _module
feature_extractors = _module
fid_inception = _module
fid = _module
fsim = _module
functional = _module
base = _module
colour_conversion = _module
filters = _module
layers = _module
resize = _module
gmsd = _module
gs = _module
haarpsi = _module
isc = _module
iw_ssim = _module
kid = _module
mdsi = _module
ms_ssim = _module
msid = _module
perceptual = _module
pieapp = _module
pr = _module
psnr = _module
srsim = _module
ssim = _module
tv = _module
utils = _module
common = _module
vif = _module
vsi = _module
setup = _module
tests = _module
conftest = _module
results_benchmark = _module
test_brisque = _module
test_dss = _module
test_examples = _module
test_fid = _module
test_fsim = _module
test_gmsd = _module
test_gs = _module
test_haarpsi = _module
test_is = _module
test_iw_ssim = _module
test_kid = _module
test_mdsi = _module
test_ms_ssim = _module
test_msid = _module
test_perceptual = _module
test_pieapp = _module
test_pr = _module
test_psnr = _module
test_srsim = _module
test_ssim = _module
test_tv = _module
test_utils = _module
test_vif = _module
test_vsi = _module

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


from typing import Union


from typing import Tuple


import warnings


from torch.nn.modules.loss import _Loss


from torch.utils.model_zoo import load_url


import torch.nn.functional as F


import math


import functools


import torch.nn as nn


from torchvision import models


from typing import List


from typing import Dict


import numpy as np


from typing import Optional


import typing


from torch.nn import functional as F


from warnings import warn


from torch.nn.functional import pad


from torch.nn.functional import avg_pool2d


from typing import Collection


from torchvision.models import vgg16


from torchvision.models import vgg19


import re


from typing import Any


from torch.nn.functional import interpolate


import torchvision


import pandas as pd


from typing import Callable


from scipy.stats import spearmanr


from scipy.stats import kendalltau


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch import nn


from itertools import chain


from scipy.stats import entropy


import itertools


import tensorflow as tf


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation."""

    def __init__(self, in_channels: int, pool_features: int) ->None:
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
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


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation."""

    def __init__(self, in_channels: int, channels_7x7: int) ->None:
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
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


class FIDInceptionE1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation."""

    def __init__(self, in_channels: int) ->None:
        super(FIDInceptionE1, self).__init__(in_channels)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
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


class FIDInceptionE2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation."""

    def __init__(self, in_channels: int) ->None:
        super(FIDInceptionE2, self).__init__(in_channels)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
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


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks: List[int]=[DEFAULT_BLOCK_INDEX], resize_input: bool=True, normalize_input: bool=True, requires_grad: bool=False, use_fid_inception: bool=True) ->None:
        """Build pretrained InceptionV3

        Args:
            output_blocks: Indices of blocks to return features of. Possible values are:
                    - 0: corresponds to output of first max pooling
                    - 1: corresponds to output of second max pooling
                    - 2: corresponds to output which is fed to aux classifier
                    - 3: corresponds to output of final average pooling
            resize_input:  If true, bilinearly resizes input to width and height 299 before
                feeding input to model. As the network without fully connected
                layers is fully convolutional, it should be able to handle inputs
                of arbitrary size, so resizing might not be strictly needed
            normalize_input: If true, scales the input from range (0, 1) to the range the
                pretrained Inception network expects, namely (-1, 1)
            requires_grad: If true, parameters of the model require gradients.
                Possibly useful for finetuning the network
            use_fid_inception: If true, uses the pretrained Inception model used in Tensorflow's
                FID implementation. If false, uses the pretrained Inception model
                available in torchvision. The FID Inception model has different
                weights and a slightly different structure from torchvision's
                Inception model. If you want to compute FID scores, you are
                strongly advised to set this parameter to true to get comparable
                results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = self.fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) ->List[torch.Tensor]:
        """Get Inception feature maps

        Args:
            x: Batch of images with shape (N, 3, H, W). RGB colour order.
                Values are expected to be in range (0, 1) if `normalize_input` is True,
                and in range (-1, 1) otherwise.

        Returns:
            List of torch.autograd.Variable, corresponding to the selected output block, sorted ascending by index.
        """
        outp = []
        input_range = (0, 1) if self.normalize_input else (-1, 1)
        assert x.min() >= input_range[0] and x.max() <= input_range[1], f'Input tensor should be normalized in ({input_range[0]}, {input_range[0]}) range.'
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp

    @staticmethod
    def fid_inception_v3() ->nn.Module:
        """Build pretrained Inception model for FID computation

        The Inception model for FID computation uses a different set of weights
        and has a slightly different structure than torchvision's Inception.
        This method first constructs torchvision's Inception and then patches the
        necessary parts that are different in the FID Inception model.
        """
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
        inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = FIDInceptionE1(1280)
        inception.Mixed_7c = FIDInceptionE2(2048)
        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
        inception.load_state_dict(state_dict)
        return inception


class BaseFeatureMetric(torch.nn.Module):
    """Base class for all metrics, which require computation of per image features.
     For example: FID, KID, MSID etc.
     """

    def __init__(self) ->None:
        super(BaseFeatureMetric, self).__init__()

    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        return self.compute_metric(x_features, y_features)

    @torch.no_grad()
    def compute_feats(self, loader: torch.utils.data.DataLoader, feature_extractor: torch.nn.Module=None, device: str='cuda') ->torch.Tensor:
        """Generate low-dimensional image descriptors

        Args:
            loader: Should return dict with key `images` in it
            feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
                Model should return a list with features from one of the network layers.
            out_features: size of `feature_extractor` output
            device: Device on which to compute inference of the model
        """
        if feature_extractor is None:
            None
            feature_extractor = InceptionV3()
        else:
            assert isinstance(feature_extractor, torch.nn.Module), f'Feature extractor must be PyTorch module. Got {type(feature_extractor)}'
        feature_extractor
        feature_extractor.eval()
        total_feats = []
        for batch in loader:
            images = batch['images']
            N = images.shape[0]
            images = images.float()
            features = feature_extractor(images)
            assert len(features) == 1, f'feature_encoder must return list with features from one layer. Got {len(features)}'
            total_feats.append(features[0].view(N, -1))
        return torch.cat(total_feats, dim=0)

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError('This function should be defined for each children class')


def _aggd_parameters(x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = torch.arange(start=0.2, end=10.001, step=0.001, dtype=x.dtype, device=x.device)
    r_table = torch.exp(2 * torch.lgamma(2.0 / gamma) - torch.lgamma(1.0 / gamma) - torch.lgamma(3.0 / gamma))
    r_table = r_table.repeat(x.size(0), 1)
    mask_left = x < 0
    mask_right = x > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=x.dtype)
    count_right = mask_right.sum(dim=(-1, -2), dtype=x.dtype)
    assert (count_left > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients)  with values below zero to compute parameters of AGGD'
    assert (count_right > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients) with values above zero to compute parameters of AGGD'
    left_sigma = ((x * mask_left).pow(2).sum(dim=(-1, -2)) / count_left).sqrt()
    right_sigma = ((x * mask_right).pow(2).sum(dim=(-1, -2)) / count_right).sqrt()
    assert (left_sigma > 0).all() and (right_sigma > 0).all(), f'Expected non-zero left and right variances, got {left_sigma} and {right_sigma}'
    gamma_hat = left_sigma / right_sigma
    ro_hat = x.abs().mean(dim=(-1, -2)).pow(2) / x.pow(2).mean(dim=(-1, -2))
    ro_hat_norm = ro_hat * (gamma_hat.pow(3) + 1) * (gamma_hat + 1) / (gamma_hat.pow(2) + 1).pow(2)
    indexes = (ro_hat_norm - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, left_sigma.squeeze(dim=-1), right_sigma.squeeze(dim=-1)


def _ggd_parameters(x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.arange(0.2, 10 + 0.001, 0.001, dtype=x.dtype, device=x.device)
    r_table = (torch.lgamma(1.0 / gamma) + torch.lgamma(3.0 / gamma) - 2 * torch.lgamma(2.0 / gamma)).exp()
    r_table = r_table.repeat(x.size(0), 1)
    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)
    assert not torch.isclose(sigma, torch.zeros_like(sigma)).all(), 'Expected image with non zero variance of pixel values'
    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E ** 2
    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, sigma


def gaussian_filter(kernel_size: int, sigma: float, dtype: torch.dtype=torch.float32) ->torch.Tensor:
    """Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        dtype: type of tensor to return
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()
    g /= g.sum()
    return g.unsqueeze(0)


def _natural_scene_statistics(luma: torch.Tensor, kernel_size: int=7, sigma: float=7.0 / 6) ->torch.Tensor:
    kernel = gaussian_filter(kernel_size=kernel_size, sigma=sigma, dtype=luma.dtype).view(1, 1, kernel_size, kernel_size)
    C = 1
    mu = F.conv2d(luma, kernel, padding=kernel_size // 2)
    mu_sq = mu ** 2
    std = F.conv2d(luma ** 2, kernel, padding=kernel_size // 2)
    std = (std - mu_sq).abs().sqrt()
    luma_nrmlzd = (luma - mu) / (std + C)
    alpha, sigma = _ggd_parameters(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = _aggd_parameters(luma_nrmlzd * shifted_luma_nrmlzd)
        eta = (sigma_r - sigma_l) * torch.exp(torch.lgamma(2.0 / alpha) - (torch.lgamma(1.0 / alpha) + torch.lgamma(3.0 / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))
    return torch.stack(features, dim=-1)


def _reduce(x: torch.Tensor, reduction: str='mean') ->torch.Tensor:
    """Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _scale_features(features: torch.Tensor) ->torch.Tensor:
    lower_bound = -1
    upper_bound = 1
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612], [0.236, 1.642], [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257], [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456], [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858], [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561], [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508], [0.471, 3.264], [0.012809, 0.703171], [0.218, 1.046], [-0.094876, 0.187459], [1.5e-05, 0.442057], [0.001272, 0.40803], [0.222, 1.042], [-0.115772, 0.162604], [1.6e-05, 0.444362], [0.001374, 0.40243], [0.227, 0.996], [-0.117188, 0.098323], [3e-05, 0.531903], [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658], [2.8e-05, 0.530092], [0.001118, 0.370399]])
    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (feature_ranges[..., 1] - feature_ranges[..., 0])
    return scaled_features


def _rbf_kernel(features: torch.Tensor, sv: torch.Tensor, gamma: float=0.05) ->torch.Tensor:
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


def _score_svr(features: torch.Tensor) ->torch.Tensor:
    url = 'https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt'
    sv_coef, sv = load_url(url, map_location=features.device)
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = _rbf_kernel(features=features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def _validate_input(tensors: List[torch.Tensor], dim_range: Tuple[int, int]=(0, -1), data_range: Tuple[float, float]=(0.0, -1.0), size_range: Optional[Tuple[int, int]]=None) ->None:
    """Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """
    if not __debug__:
        return
    x = tensors[0]
    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'
        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]:size_range[1]] == x.size()[size_range[0]:size_range[1]], f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'
        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'
        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'


_D = typing.Optional[torch.dtype]


def cast_input(x: torch.Tensor) ->typing.Tuple[torch.Tensor, _D]:
    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None
    return x, dtype


def cast_output(x: torch.Tensor, dtype: _D) ->torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        if dtype is torch.uint8:
            x = x.clamp(0, 255)
        x = x
    return x


def reflect_padding(x: torch.Tensor, dim: int, pad_pre: int, pad_post: int) ->torch.Tensor:
    """
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    """
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:h + pad_pre, :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:w + pad_pre].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])
    return padding_buffer


def padding(x: torch.Tensor, dim: int, pad_pre: int, pad_post: int, padding_type: typing.Optional[str]='reflect') ->torch.Tensor:
    if padding_type is None:
        return x
    elif padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))
    return x_pad


def downsampling_2d(x: torch.Tensor, k: torch.Tensor, scale: int, padding_type: str='reflect') ->torch.Tensor:
    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)
    k = k
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e
    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y


_I = typing.Optional[int]


def reshape_input(x: torch.Tensor) ->typing.Tuple[torch.Tensor, _I, _I, int, int]:
    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))
    x = x.view(-1, 1, h, w)
    return x, b, c, h, w


def reshape_output(x: torch.Tensor, b: _I, c: _I) ->torch.Tensor:
    rh = x.size(-2)
    rw = x.size(-1)
    if b is not None:
        x = x.view(b, c, rh, rw)
    elif c is not None:
        x = x.view(c, rh, rw)
    else:
        x = x.view(rh, rw)
    return x


def get_padding(base: torch.Tensor, kernel_size: int, x_size: int) ->typing.Tuple[int, int, torch.Tensor]:
    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1
    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0
    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0
    return pad_pre, pad_post, base


def cubic_contribution(x: torch.Tensor, a: float=-0.5) ->torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2
    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))
    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01
    cont_12 = a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a
    cont_12 = cont_12 * range_12
    cont = cont_01 + cont_12
    return cont


def gaussian_contribution(x: torch.Tensor, sigma: float=2.0) ->torch.Tensor:
    range_3sigma = x.abs() <= 3 * sigma + 1
    cont = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    cont = cont * range_3sigma
    return cont


def get_weight(dist: torch.Tensor, kernel_size: int, kernel: str='cubic', sigma: float=2.0, antialiasing_factor: float=1) ->torch.Tensor:
    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))
    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) ->torch.Tensor:
    if dim == 2 or dim == -2:
        k = kernel_size, 1
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    else:
        k = 1, kernel_size
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1
    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def resize_1d(x: torch.Tensor, dim: int, size: int, scale: float, kernel: str='cubic', sigma: float=2.0, padding_type: str='reflect', antialiasing: bool=True) ->torch.Tensor:
    """
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):
    Return:
    """
    if scale == 1:
        return x
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)
    if antialiasing and scale < 1:
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1
    kernel_size += 2
    with torch.no_grad():
        pos = torch.linspace(0, size - 1, steps=size, dtype=x.dtype, device=x.device)
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - kernel_size // 2 + 1
        dist = pos - base
        weight = get_weight(dist, kernel_size, kernel=kernel, sigma=sigma, antialiasing_factor=antialiasing_factor)
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x


def imresize(x: torch.Tensor, scale: typing.Optional[float]=None, sizes: typing.Optional[typing.Tuple[int, int]]=None, kernel: typing.Union[str, torch.Tensor]='cubic', sigma: float=2, rotation_degree: float=0, padding_type: str='reflect', antialiasing: bool=True) ->torch.Tensor:
    """
    Args:
        x (torch.Tensor):
        scale (float):
        sizes (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    """
    if scale is None and sizes is None:
        raise ValueError('One of scale or sizes must be specified!')
    if scale is not None and sizes is not None:
        raise ValueError('Please specify scale or sizes to avoid conflict!')
    x, b, c, h, w = reshape_input(x)
    if sizes is None and scale is not None:
        """
        # Check if we can apply the convolution algorithm
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )
        """
        sizes = math.ceil(h * scale), math.ceil(w * scale)
        scales = scale, scale
    if scale is None and sizes is not None:
        scales = sizes[0] / h, sizes[1] / w
    x, dtype = cast_input(x)
    if isinstance(kernel, str) and sizes is not None:
        x = resize_1d(x, -2, size=sizes[0], scale=scales[0], kernel=kernel, sigma=sigma, padding_type=padding_type, antialiasing=antialiasing)
        x = resize_1d(x, -1, size=sizes[1], scale=scales[1], kernel=kernel, sigma=sigma, padding_type=padding_type, antialiasing=antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale is not None:
        x = downsampling_2d(x, kernel, scale=int(1 / scale))
    x = reshape_output(x, b, c)
    x = cast_output(x, dtype)
    return x


def rgb2yiq(x: torch.Tensor) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]).t()
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def brisque(x: torch.Tensor, kernel_size: int=7, kernel_sigma: float=7 / 6, data_range: Union[int, float]=1.0, reduction: str='mean') ->torch.Tensor:
    """Interface of BRISQUE index.
    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``

    Returns:
        Value of BRISQUE index.

    References:
        Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
        https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    Note:
        The back propagation is not available using ``torch=1.5.0`` due to bug in ``argmin`` and ``argmax``
        backpropagation. Update the torch and torchvision to the latest versions.
    """
    if '1.5.0' in torch.__version__:
        warnings.warn(f'BRISQUE does not support back propagation due to bug in torch={torch.__version__}.Update torch to the latest version to access full functionality of the BRIQSUE.More info is available at https://github.com/photosynthesis-team/piq/pull/79 andhttps://github.com/pytorch/pytorch/issues/38869.')
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x], dim_range=(4, 4), data_range=(0, data_range))
    x = x / float(data_range) * 255
    if x.size(1) == 3:
        x = torch.round(rgb2yiq(x)[:, :1])
    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, sizes=(x.size(2) // 2, x.size(3) // 2))
    features = torch.cat(features, dim=-1)
    scaled_features = _scale_features(features)
    score = _score_svr(scaled_features)
    return _reduce(score, reduction)


class BRISQUELoss(_Loss):
    """Creates a criterion that measures the BRISQUE score for input :math:`x`.
    :math:`x` is 4D tensor (N, C, H, W).
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided by setting ``reduction = 'sum'``.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma: Standard deviation for Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    Examples:
        >>> loss = BRISQUELoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(x)
        >>> output.backward()
    References:
        Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
        https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    Note:
        The back propagation is not available using ``torch=1.5.0`` due to bug in ``argmin`` and ``argmax``
        backpropagation. Update the torch and torchvision to the latest versions.
    """

    def __init__(self, kernel_size: int=7, kernel_sigma: float=7 / 6, data_range: Union[int, float]=1.0, reduction: str='mean', interpolation: str='nearest') ->None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        self.kernel_sigma = kernel_sigma
        self.data_range = data_range

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computation of BRISQUE score as a loss function.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of BRISQUE loss to be minimized.
        """
        return brisque(x, reduction=self.reduction, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, data_range=self.data_range)


def _dct_matrix(size: int) ->torch.Tensor:
    """ Computes the matrix coefficients for DCT transform using the following formula:
    https://fr.mathworks.com/help/images/discrete-cosine-transform.html

    Args:
       size : size of DCT matrix to create.  (`size`, `size`)
    """
    p = torch.arange(1, size).reshape((size - 1, 1))
    q = torch.arange(1, 2 * size, 2)
    return torch.cat((math.sqrt(1 / size) * torch.ones((1, size)), math.sqrt(2 / size) * torch.cos(math.pi / (2 * size) * p * q)), 0)


def _dct_decomp(x: torch.Tensor, dct_size: int=8) ->torch.Tensor:
    """ Computes 2D Discrete Cosine Transform on 8x8 blocks of an image

    Args:
        x: input image. Shape :math:`(N, 1, H, W)`.
        dct_size: size of DCT performed. DCT size must be in (0, input size]. Default: 8
    Returns:
        decomp: the result of DCT on NxN blocks of the image, same shape.
    Note:
        Inspired by https://gitlab.com/Queuecumber/torchjpeg
    """
    bs, _, h, w = x.size()
    x = x.view(bs, 1, h, w)
    blocks = F.unfold(x, kernel_size=(dct_size, dct_size), stride=(dct_size, dct_size))
    blocks = blocks.transpose(1, 2)
    blocks = blocks.view(bs, 1, -1, dct_size, dct_size)
    coeffs = _dct_matrix(dct_size)
    blocks = coeffs @ blocks @ coeffs.t()
    blocks = blocks.reshape(bs, -1, dct_size ** 2)
    blocks = blocks.transpose(1, 2)
    blocks = F.fold(blocks, output_size=x.size()[-2:], kernel_size=(dct_size, dct_size), stride=(dct_size, dct_size))
    decomp = blocks.reshape(bs, 1, x.size(-2), x.size(-1))
    return decomp


def _subband_similarity(x: torch.Tensor, y: torch.Tensor, first_term: bool, kernel_size: int=3, sigma: float=1.5, percentile: float=0.05) ->torch.Tensor:
    """Compute similarity between 2 subbands

    Args:
        x: First input subband. Shape (N, 1, H, W).
        y: Second input subband. Shape (N, 1, H, W).
        first_term: whether this is is the first element of subband sim matrix to be calculated
        kernel_size: Size of gaussian kernel for computing local variance. Kernels size must be in (0, input size].
            Default: 3
        sigma: STD of gaussian kernel for computing local variance. Default: 1.5
        percentile: % in [0,1] of worst similarity scores which should be kept. Default: 0.05
    Returns:
        DSS: Index of similarity between two images. In [0, 1] interval.
    Note:
        This implementation is based on the original MATLAB code (see header).
    """
    dc_coeff, ac_coeff = 1000, 300
    c = dc_coeff if first_term else ac_coeff
    kernel = gaussian_filter(kernel_size=kernel_size, sigma=sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    mu_x = F.conv2d(x, kernel, padding=kernel_size // 2)
    mu_y = F.conv2d(y, kernel, padding=kernel_size // 2)
    sigma_xx = F.conv2d(x * x, kernel, padding=kernel_size // 2) - mu_x ** 2
    sigma_yy = F.conv2d(y * y, kernel, padding=kernel_size // 2) - mu_y ** 2
    sigma_xx[sigma_xx < 0] = 0
    sigma_yy[sigma_yy < 0] = 0
    left_term = (2 * torch.sqrt(sigma_xx * sigma_yy) + c) / (sigma_xx + sigma_yy + c)
    percentile_index = round(percentile * (left_term.size(-2) * left_term.size(-1)))
    sorted_left = torch.sort(left_term.flatten(start_dim=1)).values
    similarity = torch.mean(sorted_left[:, :percentile_index], dim=1)
    if first_term:
        sigma_xy = F.conv2d(x * y, kernel, padding=kernel_size // 2) - mu_x * mu_y
        right_term = (sigma_xy + c) / (torch.sqrt(sigma_xx * sigma_yy) + c)
        sorted_right = torch.sort(right_term.flatten(start_dim=1)).values
        similarity *= torch.mean(sorted_right[:, :percentile_index], dim=1)
    return similarity


def dss(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, dct_size: int=8, sigma_weight: float=1.55, kernel_size: int=3, sigma_similarity: float=1.5, percentile: float=0.05) ->torch.Tensor:
    """Compute DCT Subband Similarity index for a batch of images.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        dct_size: Size of blocks in 2D Discrete Cosine Transform. DCT sizes must be in (0, input size].
        sigma_weight: STD of gaussian that determines the proportion of weight given to low freq and high freq.
            Default: 1.55
        kernel_size: Size of gaussian kernel for computing subband similarity. Kernels size must be in (0, input size].
            Default: 3
        sigma_similarity: STD of gaussian kernel for computing subband similarity. Default: 1.55
        percentile: % in (0, 1] of the worst similarity scores which should be kept. Default: 0.05
    Returns:
        DSS: Index of similarity between two images. In [0, 1] interval.
    Note:
        This implementation is based on the original MATLAB code (see header).
        Image will be scaled to [0, 255] because all constants are computed for this range.
        Make sure you know what you are doing when changing default coefficient values.
    """
    if sigma_weight == 0 or sigma_similarity == 0:
        raise ValueError(f'Gaussian sigmas must not be 0, got sigma_weight: {sigma_weight} and sigma_similarity: {sigma_similarity}')
    if percentile <= 0 or percentile > 1:
        raise ValueError(f'Percentile must be in (0,1], got {percentile}')
    _validate_input(tensors=[x, y], dim_range=(4, 4))
    for size in (dct_size, kernel_size):
        if size <= 0 or size > min(x.size(-1), x.size(-2)):
            raise ValueError('DCT and kernels sizes must be included in (0, input size]')
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    num_channels = x.size(1)
    if num_channels == 3:
        x_lum = rgb2yiq(x)[:, :1]
        y_lum = rgb2yiq(y)[:, :1]
    else:
        x_lum = x
        y_lum = y
    rows, cols = x_lum.size()[-2:]
    rows = dct_size * (rows // dct_size)
    cols = dct_size * (cols // dct_size)
    x_lum = x_lum[:, :, 0:rows, 0:cols]
    y_lum = y_lum[:, :, 0:rows, 0:cols]
    dct_x = _dct_decomp(x_lum, dct_size)
    dct_y = _dct_decomp(y_lum, dct_size)
    coords = torch.arange(1, dct_size + 1)
    weight = (coords - 0.5) ** 2
    weight = (-(weight.unsqueeze(0) + weight.unsqueeze(1)) / (2 * sigma_weight ** 2)).exp()
    subband_sim_matrix = torch.zeros((x.size(0), dct_size, dct_size), device=x.device)
    threshold = 0.01
    for m in range(dct_size):
        for n in range(dct_size):
            first_term = m == 0 and n == 0
            if weight[m, n] < threshold:
                weight[m, n] = 0
                continue
            subband_sim_matrix[:, m, n] = _subband_similarity(dct_x[:, :, m::dct_size, n::dct_size], dct_y[:, :, m::dct_size, n::dct_size], first_term, kernel_size, sigma_similarity, percentile)
    eps = torch.finfo(weight.dtype).eps
    similarity_scores = torch.sum(subband_sim_matrix * (weight / torch.sum(weight) + eps), dim=[1, 2])
    dss_val = _reduce(similarity_scores, reduction)
    return dss_val


class DSSLoss(_Loss):
    """Creates a criterion that measures the DSS for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value `1 - clip(DSS, min=0, max=1)` is returned. If you need DSS value,
    use function `dss` instead.

    Args:

        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        dct_size: Size of blocks in 2D Discrete Cosine Transform. DCT sizes must be in (0, input size].
        sigma_weight: STD of gaussian that determines the proportion of weight given to low freq and high freq.
            Default: 1.55
        kernel_size: Size of gaussian kernel for computing subband similarity. Kernels size must be in (0, input size].
            Default: 3
        sigma_similarity: STD of gaussian kernel for computing subband similarity. Default: 1.5
        percentile: % in (0,1] of worst similarity scores which should be kept. Default: 0.05

    Shape:
        - Input: Required to be 4D (N, C, H, W). RGB channel order for colour images.
        - Target: Required to be 4D (N, C, H, W). RGB channel order for colour images.

    Examples::
        >>> loss = DSSLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf
        """

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, dct_size: int=8, sigma_weight: float=1.55, kernel_size: int=3, sigma_similarity: float=1.5, percentile: float=0.05) ->None:
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.dss = functools.partial(dss, reduction=reduction, data_range=data_range, dct_size=dct_size, sigma_weight=sigma_weight, kernel_size=kernel_size, sigma_similarity=sigma_similarity, percentile=percentile)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Computation of DSS as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        Returns:
            Value of DSS loss to be minimized. 0 <= DSS <= 1.
        """
        score = self.dss(prediction, target)
        return 1 - torch.clamp(score, 0, 1)


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) ->torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int=100) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)
        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.0], device=error.device, dtype=error.dtype), atol=1e-05):
            break
    return s_matrix, error


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps=1e-06) ->torch.Tensor:
    """
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.mm(sigma2))
    if not torch.isfinite(covmean).all():
        None
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma1 + offset).mm(sigma2 + offset))
    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def _cov(m: torch.Tensor, rowvar: bool=True) ->torch.Tensor:
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def _compute_statistics(samples: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the statistics used by FID
    Args:
        samples:  Low-dimension representation of image set.
            Shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
        mu: mean over all activations from the encoder.
        sigma: covariance matrix over all activations from the encoder.
    """
    mu = torch.mean(samples, dim=0)
    sigma = _cov(samples, rowvar=False)
    return mu, sigma


class FID(BaseFeatureMetric):
    """Interface of Frechet Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. FID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Examples:
        >>> fid_metric = FID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> fid: torch.Tensor = fid_metric(x_feats, y_feats)

    References:
        Heusel M. et al. (2017).
        Gans trained by a two time-scale update rule converge to a local nash equilibrium.
        Advances in neural information processing systems,
        https://arxiv.org/abs/1706.08500
    """

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        """
        Fits multivariate Gaussians: :math:`X \\sim \\mathcal{N}(\\mu_x, \\sigma_x)` and
        :math:`Y \\sim \\mathcal{N}(\\mu_y, \\sigma_y)` to image stacks.
        Then computes FID as :math:`d^2 = ||\\mu_x - \\mu_y||^2 + Tr(\\sigma_x + \\sigma_y - 2\\sqrt{\\sigma_x \\sigma_y})`.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            The Frechet Distance.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        mu_x, sigma_x = _compute_statistics(x_features.detach())
        mu_y, sigma_y = _compute_statistics(y_features.detach())
        score = _compute_fid(mu_x, sigma_x, mu_y, sigma_y)
        return score


def get_meshgrid(size: Tuple[int, int]) ->torch.Tensor:
    """Return coordinate grid matrices centered at zero point.
    Args:
        size: Shape of meshgrid to create
    """
    if size[0] % 2:
        x = torch.arange(-(size[0] - 1) / 2, size[0] / 2) / (size[0] - 1)
    else:
        x = torch.arange(-size[0] / 2, size[0] / 2) / size[0]
    if size[1] % 2:
        y = torch.arange(-(size[1] - 1) / 2, size[1] / 2) / (size[1] - 1)
    else:
        y = torch.arange(-size[1] / 2, size[1] / 2) / size[1]
    return torch.meshgrid(x, y)


def ifftshift(x: torch.Tensor) ->torch.Tensor:
    """ Similar to np.fft.ifftshift but applies to PyTorch Tensors"""
    shift = [(-(ax // 2)) for ax in x.size()]
    return torch.roll(x, shift, tuple(range(len(shift))))


def _lowpassfilter(size: Tuple[int, int], cutoff: float, n: int) ->torch.Tensor:
    """
    Constructs a low-pass Butterworth filter.

    Args:
        size: Tuple with height and width of filter to construct
        cutoff: Cutoff frequency of the filter in (0, 0.5()
        n: Filter order. Higher `n` means sharper transition.
            Note that `n` is doubled so that it is always an even integer.

    Returns:
        f = 1 / (1 + w/cutoff) ^ 2n

    Note:
        The frequency origin of the returned filter is at the corners.

    """
    assert 0 < cutoff <= 0.5, 'Cutoff frequency must be between 0 and 0.5'
    assert n > 1 and int(n) == n, 'n must be an integer >= 1'
    grid_x, grid_y = get_meshgrid(size)
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    return ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def _construct_filters(x: torch.Tensor, scales: int=4, orientations: int=4, min_length: int=6, mult: int=2, sigma_f: float=0.55, delta_theta: float=1.2, k: float=2.0):
    """Creates a stack of filters used for computation of phase congruensy maps

    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        scales: Number of wavelets
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
        """
    N, _, H, W = x.shape
    theta_sigma = math.pi / (orientations * delta_theta)
    grid_x, grid_y = get_meshgrid((H, W))
    grid_x, grid_y = grid_x, grid_y
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    theta = torch.atan2(-grid_y, grid_x)
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)
    lp = _lowpassfilter(size=(H, W), cutoff=0.45, n=15)
    log_gabor = []
    for s in range(scales):
        wavelength = min_length * mult ** s
        omega_0 = 1.0 / wavelength
        gabor_filter = torch.exp(-torch.log(radius / omega_0) ** 2 / (2 * math.log(sigma_f) ** 2))
        gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        log_gabor.append(gabor_filter)
    spread = []
    for o in range(orientations):
        angl = o * math.pi / orientations
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp(-dtheta ** 2 / (2 * theta_sigma ** 2)))
    spread = torch.stack(spread)
    log_gabor = torch.stack(log_gabor)
    filters = (spread.repeat_interleave(scales, dim=0) * log_gabor.repeat(orientations, 1, 1)).unsqueeze(0)
    return filters


PEP_440_VERSION_PATTERN = """
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\\.]?
                (?P<post_l>post|rev|r)
                [-_\\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\\.]?
            (?P<dev_l>dev)
            [-_\\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version
"""


SEMVER_VERSION_PATTERN = re.compile("""
        ^
        (?P<major>0|[1-9]\\d*)
        \\.
        (?P<minor>0|[1-9]\\d*)
        \\.
        (?P<patch>0|[1-9]\\d*)
        (?:-(?P<prerelease>
            (?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)
            (?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*
        ))?
        (?:\\+(?P<build>
            [0-9a-zA-Z-]+
            (?:\\.[0-9a-zA-Z-]+)*
        ))?
        $
    """, re.VERBOSE)


def _parse_version(version: Union[str, bytes]) ->Tuple[int, ...]:
    """ Parses valid Python versions according to Semver and PEP 440 specifications.
    For more on Semver check: https://semver.org/
    For more on PEP 440 check: https://www.python.org/dev/peps/pep-0440/.

    Implementation is inspired by:
    - https://github.com/python-semver
    - https://github.com/pypa/packaging

    Args:
        version: unparsed information about the library of interest.

    Returns:
        parsed information about the library of interest.
    """
    if isinstance(version, bytes):
        version = version.decode('UTF-8')
    elif not isinstance(version, str) and not isinstance(version, bytes):
        raise TypeError(f'not expecting type {type(version)}')
    match = SEMVER_VERSION_PATTERN.match(version)
    if match:
        matched_version_parts: Dict[str, Any] = match.groupdict()
        release = tuple([int(matched_version_parts[k]) for k in ['major', 'minor', 'patch']])
        return release
    regex = re.compile('^\\s*' + PEP_440_VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)
    match = regex.search(version)
    if match is None:
        warnings.warn(f'{version} is not a valid SemVer or PEP 440 string')
        return tuple()
    release = tuple(int(i) for i in match.group('release').split('.'))
    return release


def _phase_congruency(x: torch.Tensor, scales: int=4, orientations: int=4, min_length: int=6, mult: int=2, sigma_f: float=0.55, delta_theta: float=1.2, k: float=2.0) ->torch.Tensor:
    """Compute Phase Congruence for a batch of greyscale images

    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        scales: Number of wavelet scales
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.

    Returns:
        Phase Congruency map with shape :math:`(N, H, W)`

    """
    EPS = torch.finfo(x.dtype).eps
    N, _, H, W = x.shape
    filters = _construct_filters(x, scales, orientations, min_length, mult, sigma_f, delta_theta, k)
    recommended_torch_version = _parse_version('1.8.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        imagefft = torch.fft.fft2(x)
        filters_ifft = torch.fft.ifft2(filters)
        filters_ifft = filters_ifft.real * math.sqrt(H * W)
        even_odd = torch.view_as_real(torch.fft.ifft2(imagefft * filters)).view(N, orientations, scales, H, W, 2)
    else:
        imagefft = torch.rfft(x, 2, onesided=False)
        filters_ifft = torch.ifft(torch.stack([filters, torch.zeros_like(filters)], dim=-1), 2)[..., 0]
        filters_ifft *= math.sqrt(H * W)
        even_odd = torch.ifft(imagefft * filters.unsqueeze(-1), 2).view(N, orientations, scales, H, W, 2)
    an = torch.sqrt(torch.sum(even_odd ** 2, dim=-1))
    em_n = (filters.view(1, orientations, scales, H, W)[:, :, :1, ...] ** 2).sum(dim=[-2, -1], keepdims=True)
    sum_e = even_odd[..., 0].sum(dim=2, keepdims=True)
    sum_o = even_odd[..., 1].sum(dim=2, keepdims=True)
    x_energy = torch.sqrt(sum_e ** 2 + sum_o ** 2) + EPS
    mean_e = sum_e / x_energy
    mean_o = sum_o / x_energy
    even = even_odd[..., 0]
    odd = even_odd[..., 1]
    energy = (even * mean_e + odd * mean_o - torch.abs(even * mean_o - odd * mean_e)).sum(dim=2, keepdim=True)
    abs_eo = torch.sqrt(torch.sum(even_odd[:, :, :1, ...] ** 2, dim=-1)).reshape(N, orientations, 1, 1, H * W)
    median_e2n = torch.median(abs_eo ** 2, dim=-1, keepdim=True).values
    mean_e2n = -median_e2n / math.log(0.5)
    noise_power = mean_e2n / em_n
    filters_ifft = filters_ifft.view(1, orientations, scales, H, W)
    sum_an2 = torch.sum(filters_ifft ** 2, dim=-3, keepdim=True)
    sum_ai_aj = torch.zeros(N, orientations, 1, H, W)
    for s in range(scales - 1):
        sum_ai_aj = sum_ai_aj + (filters_ifft[:, :, s:s + 1] * filters_ifft[:, :, s + 1:]).sum(dim=-3, keepdim=True)
    sum_an2 = torch.sum(sum_an2, dim=[-1, -2], keepdim=True)
    sum_ai_aj = torch.sum(sum_ai_aj, dim=[-1, -2], keepdim=True)
    noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_ai_aj
    tau = torch.sqrt(noise_energy2 / 2)
    noise_energy = tau * math.sqrt(math.pi / 2)
    moise_energy_sigma = torch.sqrt((2 - math.pi / 2) * tau ** 2)
    T = noise_energy + k * moise_energy_sigma
    T = T / 1.7
    energy = torch.max(energy - T, torch.zeros_like(T))
    eps = torch.finfo(energy.dtype).eps
    energy_all = energy.sum(dim=[1, 2]) + eps
    an_all = an.sum(dim=[1, 2]) + eps
    result_pc = energy_all / an_all
    return result_pc.unsqueeze(1)


def gradient_map(x: torch.Tensor, kernels: torch.Tensor) ->torch.Tensor:
    """ Compute gradient map for a given tensor and stack of kernels.

    Args:
        x: Tensor with shape (N, C, H, W).
        kernels: Stack of tensors for gradient computation with shape (k_N, k_H, k_W)
    Returns:
        Gradients of x per-channel with shape (N, C, H, W)
    """
    padding = kernels.size(-1) // 2
    grads = torch.nn.functional.conv2d(x, kernels, padding=padding)
    return torch.sqrt(torch.sum(grads ** 2, dim=-3, keepdim=True))


def scharr_filter() ->torch.Tensor:
    """Utility function that returns a normalized 3x3 Scharr kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)
    """
    return torch.tensor([[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]]) / 16


def similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float, alpha: float=0.0) ->torch.Tensor:
    """ Compute similarity_map between two tensors using Dice-like equation.

    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
        alpha: Masking coefficient. Subtracts - `alpha` * map_x * map_y from denominator and nominator
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / (map_x ** 2 + map_y ** 2 - alpha * map_x * map_y + constant)


def fsim(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, chromatic: bool=True, scales: int=4, orientations: int=4, min_length: int=6, mult: int=2, sigma_f: float=0.55, delta_theta: float=1.2, k: float=2.0) ->torch.Tensor:
    """Compute Feature Similarity Index Measure for a batch of images.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.

    Returns:
        Index of similarity between two images. Usually in [0, 1] interval.
        Can be bigger than 1 for predicted :math:`x` images with higher contrast than the original ones.

    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575

    Note:
        This implementation is based on the original MATLAB code.
        https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm

    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    kernel_size = max(1, round(min(x.shape[-2:]) / 256))
    x = torch.nn.functional.avg_pool2d(x, kernel_size)
    y = torch.nn.functional.avg_pool2d(y, kernel_size)
    num_channels = x.size(1)
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)
        x_lum = x_yiq[:, :1]
        y_lum = y_yiq[:, :1]
        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]
    else:
        x_lum = x
        y_lum = y
    pc_x = _phase_congruency(x_lum, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)
    pc_y = _phase_congruency(y_lum, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)
    T1, T2, T3, T4, lmbda = 0.85, 160, 200, 200, 0.03
    PC = similarity_map(pc_x, pc_y, T1)
    GM = similarity_map(grad_map_x, grad_map_y, T2)
    pc_max = torch.where(pc_x > pc_y, pc_x, pc_y)
    score = GM * PC * pc_max
    if chromatic:
        assert num_channels == 3, 'Chromatic component can be computed only for RGB images!'
        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
    result = score.sum(dim=[1, 2, 3]) / pc_max.sum(dim=[1, 2, 3])
    return _reduce(result, reduction)


class FSIMLoss(_Loss):
    """Creates a criterion that measures the FSIM or FSIMc for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value ``1 - clip(FSIM, min=0, max=1)`` is returned. If you need FSIM value,
    use function `fsim` instead.
    Supports greyscale and colour images with RGB channel order.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.

    Examples:
        >>> loss = FSIMLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575
    """

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, chromatic: bool=True, scales: int=4, orientations: int=4, min_length: int=6, mult: int=2, sigma_f: float=0.55, delta_theta: float=1.2, k: float=2.0) ->None:
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.fsim = functools.partial(fsim, reduction=reduction, data_range=data_range, chromatic=chromatic, scales=scales, orientations=orientations, min_length=min_length, mult=mult, sigma_f=sigma_f, delta_theta=delta_theta, k=k)

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of FSIM as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of FSIM loss to be minimized in [0, 1] range.
        """
        score = self.fsim(x, y)
        return 1 - torch.clamp(score, 0, 1)


def hann_filter(kernel_size: int) ->torch.Tensor:
    """Creates  Hann kernel
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
    kernel = window[:, None] * window[None, :]
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


class L2Pool2d(torch.nn.Module):
    """Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12

    def __init__(self, kernel_size: int=3, stride: int=2, padding=1) ->None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.kernel is None:
            C = x.size(1)
            self.kernel = hann_filter(self.kernel_size).repeat((C, 1, 1, 1))
        out = torch.nn.functional.conv2d(x ** 2, self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + self.EPS).sqrt()


def prewitt_filter() ->torch.Tensor:
    """Utility function that returns a normalized 3x3 Prewitt kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)"""
    return torch.tensor([[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]]) / 3


def _gmsd(x: torch.Tensor, y: torch.Tensor, t: float=170 / 255.0 ** 2, alpha: float=0.0) ->torch.Tensor:
    """Compute Gradient Magnitude Similarity Deviation
    Supports greyscale images in [0, 1] range.

    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        y: Tensor. Shape :math:`(N, 1, H, W)`.
        t: Constant from the reference paper numerical stability of similarity map
        alpha: Masking coefficient for similarity masks computation

    Returns:
        gmsd : Gradient Magnitude Similarity Deviation between given tensors.

    References:
        https://arxiv.org/pdf/1308.3052.pdf
    """
    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(-1, -2)])
    x_grad = gradient_map(x, kernels)
    y_grad = gradient_map(y, kernels)
    gms = similarity_map(x_grad, y_grad, constant=t, alpha=alpha)
    mean_gms = torch.mean(gms, dim=[1, 2, 3], keepdims=True)
    score = torch.pow(gms - mean_gms, 2).mean(dim=[1, 2, 3]).sqrt()
    return score


def gmsd(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, t: float=170 / 255.0 ** 2) ->torch.Tensor:
    """Compute Gradient Magnitude Similarity Deviation.

    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        t: Constant from the reference paper numerical stability of similarity map.

    Returns:
        Gradient Magnitude Similarity Deviation between given tensors.

    References:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    x = x / float(data_range)
    y = y / float(data_range)
    num_channels = x.size(1)
    if num_channels == 3:
        x = rgb2yiq(x)[:, :1]
        y = rgb2yiq(y)[:, :1]
    up_pad = 0
    down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
    pad_to_use = [up_pad, down_pad, up_pad, down_pad]
    x = F.pad(x, pad=pad_to_use)
    y = F.pad(y, pad=pad_to_use)
    x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
    y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
    score = _gmsd(x=x, y=y, t=t)
    return _reduce(score, reduction)


class GMSDLoss(_Loss):
    """Creates a criterion that measures Gradient Magnitude Similarity Deviation
    between each element in the input and target.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        t: Constant from the reference paper numerical stability of similarity map

    Examples:
        >>> loss = GMSDLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf

    """

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, t: float=170 / 255.0 ** 2) ->None:
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.t = t

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Gradient Magnitude Similarity Deviation (GMSD) as a loss function.
        Supports greyscale and colour images with RGB channel order.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of GMSD loss to be minimized in [0, 1] range.
        """
        return gmsd(x=x, y=y, reduction=self.reduction, data_range=self.data_range, t=self.t)


def multi_scale_gmsd(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float]=1.0, reduction: str='mean', scale_weights: Optional[torch.Tensor]=None, chromatic: bool=False, alpha: float=0.5, beta1: float=0.01, beta2: float=0.32, beta3: float=15.0, t: float=170) ->torch.Tensor:
    """Computation of Multi scale GMSD.

    Supports greyscale and colour images with RGB channel order.
    The height and width should be at least ``2 ** scales + 1``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        scale_weights: Weights for different scales. Can contain any number of floating point values.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        alpha: Masking coefficient. See references for details.
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, see references.
        beta3: Algorithm parameter. Small constant, see references.
        t: Constant from the reference paper numerical stability of similarity map

    Returns:
        Value of MS-GMSD in [0, 1] range.

    References:
        Bo Zhang et al. Gradient Magnitude Similarity Deviation on Multiple Scales (2017).
        http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    if scale_weights is None:
        scale_weights = torch.tensor([0.096, 0.596, 0.289, 0.019], device=x.device)
    else:
        scale_weights = scale_weights / scale_weights.sum()
    num_scales = scale_weights.size(0)
    min_size = 2 ** num_scales + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    num_channels = x.size(1)
    if num_channels == 3:
        x = rgb2yiq(x)
        y = rgb2yiq(y)
    ms_gmds = []
    for scale in range(num_scales):
        if scale > 0:
            up_pad = 0
            down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
            pad_to_use = [up_pad, down_pad, up_pad, down_pad]
            x = F.pad(x, pad=pad_to_use)
            y = F.pad(y, pad=pad_to_use)
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)
        score = _gmsd(x[:, :1], y[:, :1], t=t, alpha=alpha)
        ms_gmds.append(score)
    ms_gmds_val = scale_weights.view(1, num_scales) * torch.stack(ms_gmds, dim=1) ** 2
    ms_gmds_val = torch.sqrt(torch.sum(ms_gmds_val, dim=1))
    score = ms_gmds_val
    if chromatic:
        assert x.size(1) == 3, 'Chromatic component can be computed only for RGB images!'
        x_iq = x[:, 1:]
        y_iq = y[:, 1:]
        rmse_iq = torch.sqrt(torch.mean((x_iq - y_iq) ** 2, dim=[2, 3]))
        rmse_chrome = torch.sqrt(torch.sum(rmse_iq ** 2, dim=1))
        gamma = 2 / (1 + beta2 * torch.exp(-beta3 * ms_gmds_val)) - 1
        score = gamma * ms_gmds_val + (1 - gamma) * beta1 * rmse_chrome
    return _reduce(score, reduction)


class MultiScaleGMSDLoss(_Loss):
    """Creates a criterion that measures multi scale Gradient Magnitude Similarity Deviation
    between each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        scale_weights: Weights for different scales. Can contain any number of floating point values.
            By default weights are initialized with values from the paper.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, references.
        beta3: Algorithm parameter. Small constant, references.
        t: Constant from the reference paper numerical stability of similarity map

    Examples:
        >>> loss = MultiScaleGMSDLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Bo Zhang et al. Gradient Magnitude Similarity Deviation on Multiple Scales (2017).
        http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, scale_weights: Optional[torch.Tensor]=None, chromatic: bool=False, alpha: float=0.5, beta1: float=0.01, beta2: float=0.32, beta3: float=15.0, t: float=170) ->None:
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.scale_weights = scale_weights
        self.chromatic = chromatic
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.t = t

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Multi Scale GMSD index as a loss function.
        Supports greyscale and colour images with RGB channel order.
        The height and width should be at least 2 ** scales + 1.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of MS-GMSD loss to be minimized in [0, 1] range.
        """
        return multi_scale_gmsd(x=x, y=y, data_range=self.data_range, reduction=self.reduction, chromatic=self.chromatic, alpha=self.alpha, beta1=self.beta1, beta2=self.beta2, beta3=self.beta3, scale_weights=self.scale_weights, t=self.t)


def relative(intervals: np.ndarray, alpha_max: float, i_max: int=100) ->np.ndarray:
    """
    For a collection of intervals this functions computes
    RLT by formulas (2) and (3) from the paper. This function will be typically called
    on the output of the gudhi persistence_intervals_in_dimension function.
    Args:
      intervals: list of intervals e.g. [[0, 1], [0, 2], [0, np.inf]].
      alpha_max: The maximal persistence value
      i_max: Upper bound on the value of beta_1 to compute.
    Returns:
        rlt: Array of size (i_max, ) containing desired RLT.
    """
    persistence_intervals = []
    for interval in intervals:
        if np.isinf(interval[1]):
            persistence_intervals.append([interval[0], alpha_max])
        else:
            persistence_intervals.append(list(interval))
    if len(persistence_intervals) == 0:
        rlt = np.zeros(i_max)
        rlt[0] = 1.0
        return rlt
    persistence_intervals_ext = np.array(persistence_intervals + [[0, alpha_max]])
    persistence_intervals = np.array(persistence_intervals)
    switch_points = np.sort(np.unique(persistence_intervals_ext.flatten()))
    rlt = np.zeros(i_max)
    for i in range(switch_points.shape[0] - 1):
        midpoint = (switch_points[i] + switch_points[i + 1]) / 2
        s = 0
        for interval in persistence_intervals:
            if midpoint >= interval[0] and midpoint < interval[1]:
                s = s + 1
        if s < i_max:
            rlt[s] += switch_points[i + 1] - switch_points[i]
    return rlt / alpha_max


def lmrk_table(witnesses: np.ndarray, landmarks: np.ndarray) ->Tuple[np.ndarray, np.ndarray]:
    """Construct an input for the gudhi.WitnessComplex function.
    Args:
        witnesses: Array with shape (w, d), containing witnesses.
        landmarks: Array with shape (l, d), containing landmarks.
    Returns:
        distances: 3D array with shape (w, l, 2). It satisfies the property that
            distances[i, :, :] is [idx_i, dists_i], where dists_i are the sorted distances
            from the i-th witness to each point in L and idx_i are the indices of the corresponding points
            in L, e.g., D[i, :, :] = [[0, 0.1], [1, 0.2], [3, 0.3], [2, 0.4]]
        max_dist: Maximal distance between W and L
    """
    try:
        import scipy
    except ImportError:
        raise ImportError('Scipy is required for computation of the Geometry Score but not installed. Please install scipy using the following command: pip install --user scipy')
    recommended_scipy_version = _parse_version('1.3.3')
    scipy_version = _parse_version(scipy.__version__)
    if len(scipy_version) != 0 and scipy_version < recommended_scipy_version:
        warn(f'Scipy of version {scipy.__version__} is used while version >= {recommended_scipy_version} is recommended. Consider updating scipy to avoid potential long compute time with older versions.')
    from scipy.spatial.distance import cdist
    a = cdist(witnesses, landmarks)
    max_dist = np.max(a)
    idx = np.argsort(a)
    b = a[np.arange(np.shape(a)[0])[:, np.newaxis], idx]
    distances = np.dstack([idx, b])
    return distances, max_dist


class GS(BaseFeatureMetric):
    """Interface of Geometry Score.
    It's computed for a whole set of data and can use features from encoder instead of images itself to decrease
    computation cost. GS can compare two data distributions with different number of samples.
    Dimensionalities of features should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        sample_size: Number of landmarks to use on each iteration.
            Higher values can give better accuracy, but increase computation cost.
        num_iters: Number of iterations.
            Higher values can reduce variance, but increase computation cost.
        gamma: Parameter determining maximum persistence value. Default is ``1.0 / 128 * N_imgs / 5000``
        i_max: Upper bound on i in RLT(i, 1, X, L)
        num_workers: Number of processes used for GS computation.

    Examples:
        >>> gs_metric = GS()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> gs: torch.Tensor = gs_metric(x_feats, y_feats)

    References:
        Khrulkov V., Oseledets I. (2018).
        Geometry score: A method for comparing generative adversarial networks.
        arXiv preprint, 2018.
        https://arxiv.org/abs/1802.02664

    Note:
        Computation is heavily CPU dependent, adjust ``num_workers`` parameter according to your system configuration.
        GS metric requiers ``gudhi`` library which is not installed by default.
        For conda, write: ``conda install -c conda-forge gudhi``,
        otherwise follow installation guide: http://gudhi.gforge.inria.fr/python/latest/installation.html
    """

    def __init__(self, sample_size: int=64, num_iters: int=1000, gamma: Optional[float]=None, i_max: int=100, num_workers: int=4) ->None:
        super().__init__()
        self.sample_size = sample_size
        self.num_iters = num_iters
        self.gamma = gamma
        self.i_max = i_max
        self.num_workers = num_workers

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        """Implements Algorithm 2 from the paper.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            Scalar value of the distance between distributions.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        with Pool(self.num_workers) as p:
            self.features = x_features.detach().cpu().numpy()
            pool_results = p.map(self._relative_living_times, range(self.num_iters))
            mean_rlt_x = np.vstack(pool_results).mean(axis=0)
            self.features = y_features.detach().cpu().numpy()
            pool_results = p.map(self._relative_living_times, range(self.num_iters))
            mean_rlt_y = np.vstack(pool_results).mean(axis=0)
        score = np.sum((mean_rlt_x - mean_rlt_y) ** 2)
        return torch.tensor(score, device=x_features.device) * 1000

    def _relative_living_times(self, idx: int) ->Union[np.ndarray, np.ndarray, np.ndarray]:
        """Implements Algorithm 1 for two samples of landmarks.

        Args:
            idx : Dummy argument. Used for multiprocessing.Pool to work correctly

        Returns:
            An array of size (i_max, ) containing RLT(i, 1, X, L)
            for randomly sampled landmarks.
        """
        intervals, alpha_max = witness(self.features, sample_size=self.sample_size, gamma=self.gamma)
        rlt = relative(intervals, alpha_max, i_max=self.i_max)
        return rlt


def haar_filter(kernel_size: int) ->torch.Tensor:
    """Creates Haar kernel
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    kernel = torch.ones((kernel_size, kernel_size)) / kernel_size
    kernel[kernel_size // 2:, :] = -kernel[kernel_size // 2:, :]
    return kernel.unsqueeze(0)


def haarpsi(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, scales: int=3, subsample: bool=True, c: float=30.0, alpha: float=4.2) ->torch.Tensor:
    """Compute Haar Wavelet-Based Perceptual Similarity
    Inputs supposed to be in range ``[0, data_range]`` with RGB channels order for colour images.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        scales: Number of Haar wavelets used for image decomposition.
        subsample: Flag to apply average pooling before HaarPSI computation. See references for details.
        c: Constant from the paper. See references for details
        alpha: Exponent used for similarity maps weighting. See references for details

    Returns:
        HaarPSI Wavelet-Based Perceptual Similarity between two tensors

    References:
        R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand (2017)
        'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment'
        http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf

        Code from authors on MATLAB and Python
        https://github.com/rgcda/haarpsi
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    kernel_size = 2 ** (scales + 1)
    if x.size(-1) < kernel_size or x.size(-2) < kernel_size:
        raise ValueError(f"Kernel size can't be greater than actual input size. Input size: {x.size()}. Kernel size: {kernel_size}")
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    num_channels = x.size(1)
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)
    else:
        x_yiq = x
        y_yiq = y
    if subsample:
        up_pad = 0
        down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x_yiq = F.pad(x_yiq, pad=pad_to_use)
        y_yiq = F.pad(y_yiq, pad=pad_to_use)
        x_yiq = F.avg_pool2d(x_yiq, kernel_size=2, stride=2, padding=0)
        y_yiq = F.avg_pool2d(y_yiq, kernel_size=2, stride=2, padding=0)
    coefficients_x, coefficients_y = [], []
    for scale in range(scales):
        kernel_size = 2 ** (scale + 1)
        kernels = torch.stack([haar_filter(kernel_size), haar_filter(kernel_size).transpose(-1, -2)])
        upper_pad = kernel_size // 2 - 1
        bottom_pad = kernel_size // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        coeff_x = torch.nn.functional.conv2d(F.pad(x_yiq[:, :1], pad=pad_to_use, mode='constant'), kernels)
        coeff_y = torch.nn.functional.conv2d(F.pad(y_yiq[:, :1], pad=pad_to_use, mode='constant'), kernels)
        coefficients_x.append(coeff_x)
        coefficients_y.append(coeff_y)
    coefficients_x = torch.cat(coefficients_x, dim=1)
    coefficients_y = torch.cat(coefficients_y, dim=1)
    weights = torch.max(torch.abs(coefficients_x[:, 4:]), torch.abs(coefficients_y[:, 4:]))
    sim_map = []
    for orientation in range(2):
        magnitude_x = torch.abs(coefficients_x[:, (orientation, orientation + 2)])
        magnitude_y = torch.abs(coefficients_y[:, (orientation, orientation + 2)])
        sim_map.append(similarity_map(magnitude_x, magnitude_y, constant=c).sum(dim=1, keepdims=True) / 2)
    if num_channels == 3:
        pad_to_use = [0, 1, 0, 1]
        x_yiq = F.pad(x_yiq, pad=pad_to_use)
        y_yiq = F.pad(y_yiq, pad=pad_to_use)
        coefficients_x_iq = torch.abs(F.avg_pool2d(x_yiq[:, 1:], kernel_size=2, stride=1, padding=0))
        coefficients_y_iq = torch.abs(F.avg_pool2d(y_yiq[:, 1:], kernel_size=2, stride=1, padding=0))
        weights = torch.cat([weights, weights.mean(dim=1, keepdims=True)], dim=1)
        sim_map.append(similarity_map(coefficients_x_iq, coefficients_y_iq, constant=c).sum(dim=1, keepdims=True) / 2)
    sim_map = torch.cat(sim_map, dim=1)
    eps = torch.finfo(sim_map.dtype).eps
    score = (((sim_map * alpha).sigmoid() * weights).sum(dim=[1, 2, 3]) + eps) / (torch.sum(weights, dim=[1, 2, 3]) + eps)
    score = (torch.log(score / (1 - score)) / alpha) ** 2
    return _reduce(score, reduction)


class HaarPSILoss(_Loss):
    """Creates a criterion that measures  Haar Wavelet-Based Perceptual Similarity loss between
    each element in the input and target.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        scales: Number of Haar wavelets used for image decomposition.
        subsample: Flag to apply average pooling before HaarPSI computation. See references for details.
        c: Constant from the paper. See references for details
        alpha: Exponent used for similarity maps weightning. See references for details

    Examples:

        >>> loss = HaarPSILoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand (2017)
        'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment'
        http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf
    """

    def __init__(self, reduction: Optional[str]='mean', data_range: Union[int, float]=1.0, scales: int=3, subsample: bool=True, c: float=30.0, alpha: float=4.2) ->None:
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.haarpsi = functools.partial(haarpsi, scales=scales, subsample=subsample, c=c, alpha=alpha, data_range=data_range, reduction=reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of HaarPSI as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of HaarPSI loss to be minimized in [0, 1] range.
        """
        return 1.0 - self.haarpsi(x=x, y=y)


def inception_score(features: torch.Tensor, num_splits: int=10):
    """Compute Inception Score for a list of image features.
    Expects raw logits from Inception-V3 as input.

    Args:
        features (torch.Tensor): Low-dimension representation of image set. Shape (N_samples, encoder_dim).
        num_splits: Number of parts to divide features. Inception Score is computed for them separately and
            results are then averaged.

    Returns:
        score

        variance

    References:
        "A Note on the Inception Score"
        https://arxiv.org/pdf/1801.01973.pdf

    """
    assert len(features.shape) == 2, f'Features must have shape (N_samples, encoder_dim), got {features.shape}'
    N = features.size(0)
    probas = F.softmax(features)
    partial_scores = []
    for i in range(num_splits):
        subset = probas[i * (N // num_splits):(i + 1) * (N // num_splits), :]
        p_y = torch.mean(subset, dim=0)
        scores = []
        for k in range(subset.shape[0]):
            p_yx = subset[k, :]
            scores.append(F.kl_div(p_y.log(), p_yx, reduction='sum'))
        partial_scores.append(torch.tensor(scores).mean().exp())
    partial_scores = torch.tensor(partial_scores)
    return torch.mean(partial_scores), torch.std(partial_scores)


class IS(BaseFeatureMetric):
    """Creates a criterion that measures difference of Inception Score between two datasets.

    IS is computed separately for predicted :math:`x` and target :math:`y` features and expects raw InceptionV3 model
    logits as inputs.

    Args:
        num_splits: Number of parts to divide features.
            IS is computed for them separately and results are then averaged.
        distance: How to measure distance between scores: ``'l1'`` | ``'l2'``. Default: ``'l1'``.

    Examples:
        >>> is_metric = IS()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> is: torch.Tensor = is_metric(x_feats, y_feats)

    References:
        "A Note on the Inception Score" https://arxiv.org/pdf/1801.01973.pdf
    """

    def __init__(self, num_splits: int=10, distance: str='l1') ->None:
        """

        """
        super(IS, self).__init__()
        self.num_splits = num_splits
        self.distance = distance

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        """Compute IS.

        Both features should have shape (N_samples, encoder_dim).

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            L1 or L2 distance between scores for datasets :math:`x` and :math:`y`.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(0, 2))
        x_is, _ = inception_score(x_features, num_splits=self.num_splits)
        y_is, _ = inception_score(y_features, num_splits=self.num_splits)
        if self.distance == 'l1':
            return torch.dist(x_is, y_is, 1)
        elif self.distance == 'l2':
            return torch.dist(x_is, y_is, 2)
        else:
            raise ValueError('Distance should be one of {`l1`, `l2`}')


def _image_enlarge(x: torch.Tensor) ->torch.Tensor:
    """Custom bilinear upscaling of an image.
    The function upscales an input image with upscaling factor 4x-3, adds padding on boundaries as difference
    and downscaled by the factor of 2.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.

    Returns:
        Upscaled tensor.
    """
    t1 = F.interpolate(x, size=(int(4 * x.size(-2) - 3), int(4 * x.size(-1) - 3)), mode='bilinear', align_corners=False)
    t2 = torch.zeros([x.size(0), 1, 4 * x.size(-2) - 1, 4 * x.size(-1) - 1])
    t2[:, :, 1:-1, 1:-1] = t1
    t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
    t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
    t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
    t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
    out = t2[:, :, ::2, ::2]
    return out


def _shift(x: torch.Tensor, shift: list) ->torch.Tensor:
    """ Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-vector), such that  RES(POS) = MTX(POS-OFFSET).

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        shift: Offset list.

    Returns:
        The circular shiftet tensor.
    """
    x_shifted = torch.cat((x[..., -shift[0]:, :], x[..., :-shift[0], :]), dim=-2)
    x_shifted = torch.cat((x_shifted[..., -shift[1]:], x_shifted[..., :-shift[1]]), dim=-1)
    return x_shifted


def average_filter2d(kernel_size: int) ->torch.Tensor:
    """Creates 2D normalized average filter

    Args:
        kernel_size (int):

    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    window = torch.ones(kernel_size) / kernel_size
    kernel = window[:, None] * window[None, :]
    return kernel.unsqueeze(0)


def _information_content(x: torch.Tensor, y: torch.Tensor, y_parent: torch.Tensor=None, kernel_size: int=3, sigma_nsq: float=0.4) ->torch.Tensor:
    """Computes Information Content Map for weighting the Structural Similarity.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        y_parent: Flag to control dependency on previous layer of pyramid.
        kernel_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Parameter of visual distortion model.

    Returns:
        Information Content Maps.
    """
    EPS = torch.finfo(x.dtype).eps
    n_channels = x.size(1)
    kernel = average_filter2d(kernel_size=kernel_size).repeat(x.size(1), 1, 1, 1)
    padding_up = kernel.size(-1) // 2
    padding_down = kernel.size(-1) - padding_up
    mu_x = F.conv2d(input=F.pad(x, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, padding=0, groups=n_channels)
    mu_y = F.conv2d(input=F.pad(y, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, padding=0, groups=n_channels)
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_xx = F.conv2d(F.pad(x ** 2, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(F.pad(y ** 2, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(F.pad(x * y, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy
    sigma_xx = F.relu(sigma_xx)
    sigma_yy = F.relu(sigma_yy)
    g = sigma_xy / (sigma_yy + EPS)
    vv = sigma_xx - g * sigma_xy
    g = g.masked_fill(sigma_yy < EPS, 0)
    vv[sigma_yy < EPS] = sigma_xx[sigma_yy < EPS]
    g = g.masked_fill(sigma_xx < EPS, 0)
    vv = vv.masked_fill(sigma_xx < EPS, 0)
    block = [kernel_size, kernel_size]
    nblv = y.size(-2) - block[0] + 1
    nblh = y.size(-1) - block[1] + 1
    nexp = nblv * nblh
    N = block[0] * block[1]
    assert block[0] % 2 == 1 and block[1] % 2 == 1, f'Expected odd block dimensions, got {block}'
    Ly = (block[0] - 1) // 2
    Lx = (block[1] - 1) // 2
    if y_parent is not None:
        y_parent_up = _image_enlarge(y_parent)[:, :, :y.size(-2), :y.size(-1)]
        N = N + 1
    Y = torch.zeros(y.size(0), y.size(1), nexp, N, dtype=y.dtype, device=y.device)
    n = -1
    for ny in range(-Ly, Ly + 1):
        for nx in range(-Lx, Lx + 1):
            n = n + 1
            foo = _shift(y, [ny, nx])
            foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
            Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)
    if y_parent is not None:
        n = n + 1
        foo = y_parent_up
        foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
        Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)
    C_u = torch.matmul(Y.transpose(-2, -1), Y) / nexp
    recommended_torch_version = _parse_version('1.10.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        eig_values, eig_vectors = torch.linalg.eigh(C_u)
    else:
        eig_values, eig_vectors = torch.symeig(C_u, eigenvectors=True)
    sum_eig_values = torch.sum(eig_values, dim=-1).view(y.size(0), y.size(1), 1, 1)
    non_zero_eig_values_matrix = torch.diag_embed(eig_values * (eig_values > 0))
    sum_non_zero_eig_values = torch.sum(non_zero_eig_values_matrix, dim=(-2, -1), keepdim=True)
    L = non_zero_eig_values_matrix * sum_eig_values / (sum_non_zero_eig_values + (sum_non_zero_eig_values == 0))
    C_u = torch.matmul(torch.matmul(eig_vectors, L), eig_vectors.transpose(-2, -1))
    C_u_inv = torch.inverse(C_u)
    ss = torch.matmul(Y, C_u_inv) * Y / N
    ss = torch.sum(ss, dim=-1, keepdim=True)
    ss = ss.view(y.size(0), y.size(1), nblv, nblh)
    g = g[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
    vv = vv[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
    scaled_eig_values = torch.diagonal(L, offset=0, dim1=-2, dim2=-1).unsqueeze(2).unsqueeze(3)
    iw_map = torch.sum(torch.log2(1 + ((vv.unsqueeze(-1) + (1 + g.unsqueeze(-1) * g.unsqueeze(-1)) * sigma_nsq) * ss.unsqueeze(-1) * scaled_eig_values + sigma_nsq * vv.unsqueeze(-1)) / (sigma_nsq * sigma_nsq)), dim=-1)
    iw_map[iw_map < EPS] = 0
    return iw_map


def _pyr_step(x: torch.Tensor, kernel: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """ Computes one step of Laplacian pyramid generation.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel: Kernel to perform blurring.

    Returns:
        Tuple of tensors with downscaled low resolution image and high-resolution difference.
    """
    up_pad = (kernel.size(-1) - 1) // 2
    down_pad = kernel.size(-1) - 1 - up_pad
    kernel_t = kernel.transpose(-2, -1)
    lo_x = x
    if x.size(-1) > 1:
        lo_x = F.pad(lo_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel.unsqueeze(0), padding=0)[:, :, :, ::2]
    if x.size(-2) > 1:
        lo_x = F.pad(lo_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel_t.unsqueeze(0), padding=0)[:, :, ::2, :]
    up_pad = (kernel.size(-1) - 1) // 2
    down_pad = kernel.size(-1) - 1 - up_pad
    hi_x = lo_x
    if x.size(-1) > 1:
        upsampling_kernel = torch.tensor([[[[1.0, 0.0]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(1, 2), padding=0)
        hi_x = F.pad(hi_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel.unsqueeze(0), padding=0)[:, :, :, :x.size(-1)]
    if x.size(-2) > 1:
        upsampling_kernel = torch.tensor([[[[1.0], [0.0]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(2, 1), padding=0)
        hi_x = F.pad(hi_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel_t.unsqueeze(0), padding=0)[:, :, :x.size(-2), :]
    hi_x = x - hi_x
    return lo_x, hi_x


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int]=1.0, k1: float=0.01, k2: float=0.03) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f"Kernel size can't be greater than actual input size. Input size: {x.size()}. Kernel size: {kernel.size()}")
    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def binomial_filter1d(kernel_size: int) ->torch.Tensor:
    """Creates 1D normalized binomial filter

    Args:
        kernel_size (int): kernel size

    Returns:
        Binomial kernel with shape (1, 1, kernel_size)
    """
    kernel = np.poly1d([0.5, 0.5]) ** (kernel_size - 1)
    return torch.tensor(kernel.c).view(1, 1, kernel_size)


def information_weighted_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float]=1.0, kernel_size: int=11, kernel_sigma: float=1.5, k1: float=0.01, k2: float=0.03, parent: bool=True, blk_size: int=3, sigma_nsq: float=0.4, scale_weights: Optional[torch.Tensor]=None, reduction: str='mean') ->torch.Tensor:
    """Interface of Information Content Weighted Structural Similarity (IW-SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution for sliding window used in comparison.
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        parent: Flag to control dependency on previous layer of pyramid.
        blk_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Parameter of visual distortion model.
        scale_weights: Weights for scaling.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Returns:
        Value of Information Content Weighted Structural Similarity (IW-SSIM) index.

    References:
        Wang, Zhou, and Qiang Li..
        Information content weighting for perceptual image quality assessment.
        IEEE Transactions on image processing 20.5 (2011): 1185-1198.
        https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf DOI:`10.1109/TIP.2010.2092435`

    Note:
        Lack of content in target image could lead to RuntimeError due to singular information content matrix,
        which cannot be inverted.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input(tensors=[x, y], dim_range=(4, 4), data_range=(0.0, data_range))
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    if x.size(1) == 3:
        x = rgb2yiq(x)[:, :1]
        y = rgb2yiq(y)[:, :1]
    if scale_weights is None:
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=x.dtype, device=x.device)
    scale_weights = scale_weights / scale_weights.sum()
    if scale_weights.size(0) != scale_weights.numel():
        raise ValueError(f'Expected a vector of weights, got {scale_weights.dim()}D tensor')
    levels = scale_weights.size(0)
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    blur_pad = math.ceil((kernel_size - 1) / 2)
    iw_pad = blur_pad - math.floor((blk_size - 1) / 2)
    gauss_kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1)
    pyramid_kernel_size = 5
    bin_filter = binomial_filter1d(kernel_size=pyramid_kernel_size) * 2 ** 0.5
    lo_x, x_diff_old = _pyr_step(x, bin_filter)
    lo_y, y_diff_old = _pyr_step(y, bin_filter)
    x = lo_x
    y = lo_y
    wmcs = []
    for i in range(levels):
        if i < levels - 2:
            lo_x, x_diff = _pyr_step(x, bin_filter)
            lo_y, y_diff = _pyr_step(y, bin_filter)
            x = lo_x
            y = lo_y
        else:
            x_diff = x
            y_diff = y
        ssim_map, cs_map = _ssim_per_channel(x=x_diff_old, y=y_diff_old, kernel=gauss_kernel, data_range=255, k1=k1, k2=k2)
        if parent and i < levels - 2:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=y_diff, kernel_size=blk_size, sigma_nsq=sigma_nsq)
            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]
        elif i == levels - 1:
            iw_map = torch.ones_like(cs_map)
            cs_map = ssim_map
        else:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=None, kernel_size=blk_size, sigma_nsq=sigma_nsq)
            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]
        wmcs.append(torch.sum(cs_map * iw_map, dim=(-2, -1)) / torch.sum(iw_map, dim=(-2, -1)))
        x_diff_old = x_diff
        y_diff_old = y_diff
    wmcs = torch.stack(wmcs, dim=0).abs()
    score = torch.prod(wmcs ** scale_weights.view(-1, 1, 1), dim=0)[:, 0]
    return _reduce(x=score, reduction=reduction)


class InformationWeightedSSIMLoss(_Loss):
    """Creates a criterion that measures the Interface of Information Content Weighted Structural Similarity (IW-SSIM)
    index error betweeneach element in the input :math:`x` and target :math:`y`.

    Inputs supposed to be in range ``[0, data_range]``.

    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        InformationWeightedSSIMLoss(x, y) =
        \\begin{cases}
            \\operatorname{mean}(1 - IWSSIM), &  \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(1 - IWSSIM),  &  \\text{if reduction} = \\text{'sum'.}
        \\end{cases}

    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution for sliding window used in comparison.
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        parent: Flag to control dependency on previous layer of pyramid.
        blk_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Sigma of normal distribution for sliding window used in comparison for information content.
        scale_weights: Weights for scaling.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:
        >>> loss = InformationWeightedSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        Wang, Zhou, and Qiang Li..
        Information content weighting for perceptual image quality assessment.
        IEEE Transactions on image processing 20.5 (2011): 1185-1198.
        https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf DOI:`10.1109/TIP.2010.2092435`

    """

    def __init__(self, data_range: Union[int, float]=1.0, kernel_size: int=11, kernel_sigma: float=1.5, k1: float=0.01, k2: float=0.03, parent: bool=True, blk_size: int=3, sigma_nsq: float=0.4, scale_weights: Optional[torch.Tensor]=None, reduction: str='mean'):
        super().__init__()
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.parent = parent
        self.blk_size = blk_size
        self.sigma_nsq = sigma_nsq
        self.scale_weights = scale_weights
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Information Content Weighted Structural Similarity (IW-SSIM) index as a loss function.
        For colour images channel order is RGB.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of IW-SSIM loss to be minimized, i.e. ``1 - information_weighted_ssim`` in [0, 1] range.
        """
        score = information_weighted_ssim(x=x, y=y, data_range=self.data_range, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, k1=self.k1, k2=self.k2, parent=self.parent, blk_size=self.blk_size, sigma_nsq=self.sigma_nsq, scale_weights=self.scale_weights, reduction=self.reduction)
        return torch.ones_like(score) - score


def _sqn(tensor: torch.Tensor) ->torch.Tensor:
    flat = tensor.flatten()
    return flat.dot(flat)


def _mmd2_and_variance(K_XX: torch.Tensor, K_XY: torch.Tensor, K_YY: torch.Tensor, unit_diagonal: bool=False, mmd_est: str='unbiased', var_at_m: Optional[int]=None, ret_var: bool=False) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    m = K_XX.size(0)
    assert K_XX.size() == (m, m)
    assert K_XY.size() == (m, m)
    assert K_YY.size() == (m, m)
    if var_at_m is None:
        var_at_m = m
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = torch.diagonal(K_XX)
        diag_Y = torch.diagonal(K_YY)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)
    K_XY_sums_1 = K_XY.sum(dim=1)
    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    if mmd_est == 'biased':
        mmd2 = (Kt_XX_sum + sum_diag_X) / (m * m) + (Kt_YY_sum + sum_diag_Y) / (m * m) - 2 * K_XY_sum / (m * m)
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - torch.trace(K_XY)) / (m * (m - 1))
    if not ret_var:
        return mmd2
    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)
    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)
    m1 = m - 1
    m2 = m - 2
    zeta1_est = 1 / (m * m1 * m2) * (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum) - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2) + 1 / (m * m * m1) * (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum) - 2 / m ** 4 * K_XY_sum ** 2 - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX) + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    zeta2_est = 1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum) - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2) + 2 / (m * m) * K_XY_2_sum - 2 / m ** 4 * K_XY_sum ** 2 - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX) + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    var_est = 4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est
    return mmd2, var_est


def _polynomial_kernel(X: torch.Tensor, Y: torch.Tensor=None, degree: int=3, gamma: Optional[float]=None, coef0: float=1.0) ->torch.Tensor:
    """
    Compute the polynomial kernel between x and y
    K(X, Y) = (gamma <X, Y> + coef0)^degree

    Args:
        X: Tensor with shape (n_samples_1, n_features)
        Y: torch.Tensor of shape (n_samples_2, n_features)
        degree: default 3
        gamma: if None, defaults to 1.0 / n_features.
        coef0 : default 1

    Returns:
        Gram matrix : Array with shape (n_samples_1, n_samples_2)

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.size(1)
    K = torch.mm(X, Y.T)
    K *= gamma
    K += coef0
    K.pow_(degree)
    return K


class KID(BaseFeatureMetric):
    """Interface of Kernel Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. KID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        degree: Degree of a polynomial functions used in kernels. Default: 3
        gamma: Kernel parameter. See paper for details
        coef0: Kernel parameter. See paper for details
        var_at_m: Kernel variance. Default is `None`
        average: If `True` recomputes metric `n_subsets` times using `subset_size` elements.
        n_subsets: Number of repeats. Ignored if `average` is False
        subset_size: Size of each subset for repeat. Ignored if `average` is False
        ret_var: Whether to return variance after the distance is computed.
            This function will return ``Tuple[torch.Tensor, torch.Tensor]`` in this case. Default: False

    Examples:
        >>> kid_metric = KID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> kid: torch.Tensor = kid_metric(x_feats, y_feats)

    References:
        Demystifying MMD GANs https://arxiv.org/abs/1801.01401
    """

    def __init__(self, degree: int=3, gamma: Optional[float]=None, coef0: int=1, var_at_m: Optional[int]=None, average: bool=False, n_subsets: int=50, subset_size: Optional[int]=1000, ret_var: bool=False) ->None:
        super().__init__()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.ret_var = ret_var
        if average:
            self.n_subsets = n_subsets
            self.subset_size = subset_size
        else:
            self.n_subsets = 1
            self.subset_size = None

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes KID (polynomial MMD) for given sets of features, obtained from Inception net
        or any other feature extractor.
        Samples must be in range [0, 1].

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            KID score and variance (optional).
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        var_at_m = min(x_features.size(0), y_features.size(0))
        if self.subset_size is None:
            subset_size = x_features.size(0)
        else:
            subset_size = self.subset_size
        results = []
        for _ in range(self.n_subsets):
            x_subset = x_features[torch.randperm(len(x_features))[:subset_size]]
            y_subset = y_features[torch.randperm(len(y_features))[:subset_size]]
            K_XX = _polynomial_kernel(x_subset, None, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
            K_YY = _polynomial_kernel(y_subset, None, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
            K_XY = _polynomial_kernel(x_subset, y_subset, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
            out = _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=self.ret_var)
            results.append(out)
        if self.ret_var:
            score = torch.mean(torch.stack([p[0] for p in results], dim=0))
            variance = torch.mean(torch.stack([p[1] for p in results], dim=0))
            return score, variance
        else:
            score = torch.mean(torch.stack(results, dim=0))
            return score


def pow_for_complex(base: torch.Tensor, exp: Union[int, float]) ->torch.Tensor:
    """ Takes the power of each element in a 4D tensor with negative values or 5D tensor with complex values.
    Complex numbers are represented by modulus and argument: r * \\exp(i * \\phi).

    It will likely to be redundant with introduction of torch.ComplexTensor.

    Args:
        base: Tensor with shape (N, C, H, W) or (N, C, H, W, 2).
        exp: Exponent
    Returns:
        Complex tensor with shape (N, C, H, W, 2).
    """
    if base.dim() == 4:
        x_complex_r = base.abs()
        x_complex_phi = torch.atan2(torch.zeros_like(base), base)
    elif base.dim() == 5 and base.size(-1) == 2:
        x_complex_r = base.pow(2).sum(dim=-1).sqrt()
        x_complex_phi = torch.atan2(base[..., 1], base[..., 0])
    else:
        raise ValueError(f'Expected real or complex tensor, got {base.size()}')
    x_complex_pow_r = x_complex_r ** exp
    x_complex_pow_phi = x_complex_phi * exp
    x_real_pow = x_complex_pow_r * torch.cos(x_complex_pow_phi)
    x_imag_pow = x_complex_pow_r * torch.sin(x_complex_pow_phi)
    return torch.stack((x_real_pow, x_imag_pow), dim=-1)


def rgb2lhm(x: torch.Tensor) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LHM images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.

    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = torch.tensor([[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t()
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm


def mdsi(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float]=1.0, reduction: str='mean', c1: float=140.0, c2: float=55.0, c3: float=550.0, combination: str='sum', alpha: float=0.6, beta: float=0.1, gamma: float=0.2, rho: float=1.0, q: float=0.25, o: float=0.25):
    """Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: ``'sum'`` | ``'mult'``.
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Returns:
        Mean Deviation Similarity Index (MDSI) between 2 tensors.

    References:
        Nafchi, Hossein Ziaei and Shahkolaei, Atena and Hedjam, Rachid and Cheriet, Mohamed (2016).
        Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
        IEEE Ieee Access, 4, 5579--5590.
        https://arxiv.org/pdf/1608.07433.pdf,
        DOI:`10.1109/ACCESS.2016.2604042`

    Note:
        The ratio between constants is usually equal :math:`c_3 = 4c_1 = 10c_2`

    Note:
        Both inputs are supposed to have RGB channels order in accordance with the original approach.
        Nevertheless, the method supports greyscale images, which are converted to RGB by copying the grey
        channel 3 times.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        warnings.warn('The original MDSI supports only RGB images. The input images were converted to RGB by copying the grey channel 3 times.')
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    kernel_size = max(1, round(min(x.size()[-2:]) / 256))
    padding = kernel_size // 2
    if padding:
        up_pad = (kernel_size - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x = pad(x, pad=pad_to_use)
        y = pad(y, pad=pad_to_use)
    x = avg_pool2d(x, kernel_size=kernel_size)
    y = avg_pool2d(y, kernel_size=kernel_size)
    x_lhm = rgb2lhm(x)
    y_lhm = rgb2lhm(y)
    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(1, 2)])
    gm_x = gradient_map(x_lhm[:, :1], kernels)
    gm_y = gradient_map(y_lhm[:, :1], kernels)
    gm_avg = gradient_map((x_lhm[:, :1] + y_lhm[:, :1]) / 2.0, kernels)
    gs_x_y = similarity_map(gm_x, gm_y, c1)
    gs_x_average = similarity_map(gm_x, gm_avg, c2)
    gs_y_average = similarity_map(gm_y, gm_avg, c2)
    gs_total = gs_x_y + gs_x_average - gs_y_average
    cs_total = (2 * (x_lhm[:, 1:2] * y_lhm[:, 1:2] + x_lhm[:, 2:] * y_lhm[:, 2:]) + c3) / (x_lhm[:, 1:2] ** 2 + y_lhm[:, 1:2] ** 2 + x_lhm[:, 2:] ** 2 + y_lhm[:, 2:] ** 2 + c3)
    if combination == 'sum':
        gcs = alpha * gs_total + (1 - alpha) * cs_total
    elif combination == 'mult':
        gs_total_pow = pow_for_complex(base=gs_total, exp=gamma)
        cs_total_pow = pow_for_complex(base=cs_total, exp=beta)
        gcs = torch.stack((gs_total_pow[..., 0] * cs_total_pow[..., 0], gs_total_pow[..., 1] + cs_total_pow[..., 1]), dim=-1)
    else:
        raise ValueError(f'Expected combination method "sum" or "mult", got {combination}')
    mct_complex = pow_for_complex(base=gcs, exp=q)
    mct_complex = mct_complex.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    score = (pow_for_complex(base=gcs, exp=q) - mct_complex).pow(2).sum(dim=-1).sqrt()
    score = ((score ** rho).mean(dim=(-1, -2)) ** (o / rho)).squeeze(1)
    return _reduce(score, reduction)


class MDSILoss(_Loss):
    """Creates a criterion that measures Mean Deviation Similarity Index (MDSI) error between the prediction :math:`x`
    and target :math:`y`.
    Supports greyscale and colour images with RGB channel order.

    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: ``'sum'`` | ``'mult'``.
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Examples:
        >>> loss = MDSILoss(data_range=1.)
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Nafchi, Hossein Ziaei and Shahkolaei, Atena and Hedjam, Rachid and Cheriet, Mohamed (2016).
        Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
        IEEE Ieee Access, 4, 5579--5590.
        https://arxiv.org/pdf/1608.07433.pdf
        DOI:`10.1109/ACCESS.2016.2604042`

    Note:
        The ratio between constants is usually equal :math:`c_3 = 4c_1 = 10c_2`
    """

    def __init__(self, data_range: Union[int, float]=1.0, reduction: str='mean', c1: float=140.0, c2: float=55.0, c3: float=550.0, alpha: float=0.6, rho: float=1.0, q: float=0.25, o: float=0.25, combination: str='sum', beta: float=0.1, gamma: float=0.2):
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.mdsi = functools.partial(mdsi, c1=c1, c2=c2, c3=c3, alpha=alpha, rho=rho, q=q, o=o, combination=combination, beta=beta, gamma=gamma, data_range=self.data_range, reduction=self.reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Mean Deviation Similarity Index (MDSI) as a loss function.

        Both inputs are supposed to have RGB channels order.
        Greyscale images converted to RGB by copying the grey channel 3 times.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of MDSI loss to be minimized in [0, 1] range.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which are converted to RGB by copying the grey
            channel 3 times.
        """
        return self.mdsi(x=x, y=y)


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor, scale_weights: torch.Tensor, k1: float, k2: float) ->torch.Tensor:
    """Calculates Multi scale Structural Similarity (MS-SSIM) index for X and Y.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel: 2D Gaussian kernel.
        scale_weights: Weights for scaled SSIM
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    mcs = []
    ssim_val = None
    for iteration in range(levels):
        if iteration > 0:
            padding = max(x.shape[2] % 2, x.shape[3] % 2)
            x = F.pad(x, pad=[padding, 0, padding, 0], mode='replicate')
            y = F.pad(y, pad=[padding, 0, padding, 0], mode='replicate')
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)
        ssim_val, cs = _ssim_per_channel(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))
    msssim_val = torch.prod(mcs_ssim ** scale_weights.view(-1, 1, 1), dim=0).mean(1)
    return msssim_val


def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int]=1.0, k1: float=0.01, k2: float=0.03) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        kernel: 2-D gauss kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(f"Kernel size can't be greater than actual input size. Input size: {x.size()}. Kernel size: {kernel.size()}")
    c1 = k1 ** 2
    c2 = k2 ** 2
    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]
    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real
    compensation = 1.0
    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real
    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map
    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))
    return ssim_val, cs


def _multi_scale_ssim_complex(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor, scale_weights: torch.Tensor, k1: float, k2: float) ->torch.Tensor:
    """Calculate Multi scale Structural Similarity (MS-SSIM) index for Complex X and Y.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel: 2-D gauss kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Complex Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-2) < min_size or x.size(-3) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    mcs = []
    ssim_val = None
    for iteration in range(levels):
        x_real = x[..., 0]
        x_imag = x[..., 1]
        y_real = y[..., 0]
        y_imag = y[..., 1]
        if iteration > 0:
            padding = max(x.size(2) % 2, x.size(3) % 2)
            x_real = F.pad(x_real, pad=[padding, 0, padding, 0], mode='replicate')
            x_imag = F.pad(x_imag, pad=[padding, 0, padding, 0], mode='replicate')
            y_real = F.pad(y_real, pad=[padding, 0, padding, 0], mode='replicate')
            y_imag = F.pad(y_imag, pad=[padding, 0, padding, 0], mode='replicate')
            x_real = F.avg_pool2d(x_real, kernel_size=2, padding=0)
            x_imag = F.avg_pool2d(x_imag, kernel_size=2, padding=0)
            y_real = F.avg_pool2d(y_real, kernel_size=2, padding=0)
            y_imag = F.avg_pool2d(y_imag, kernel_size=2, padding=0)
            x = torch.stack((x_real, x_imag), dim=-1)
            y = torch.stack((y_real, y_imag), dim=-1)
        ssim_val, cs = _ssim_per_channel_complex(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))
    mcs_ssim_real = mcs_ssim[..., 0]
    mcs_ssim_imag = mcs_ssim[..., 1]
    mcs_ssim_abs = (mcs_ssim_real.pow(2) + mcs_ssim_imag.pow(2)).sqrt()
    mcs_ssim_deg = torch.atan2(mcs_ssim_imag, mcs_ssim_real)
    mcs_ssim_pow_abs = mcs_ssim_abs ** scale_weights.view(-1, 1, 1)
    mcs_ssim_pow_deg = mcs_ssim_deg * scale_weights.view(-1, 1, 1)
    msssim_val_abs = torch.prod(mcs_ssim_pow_abs, dim=0)
    msssim_val_deg = torch.sum(mcs_ssim_pow_deg, dim=0)
    msssim_val_real = msssim_val_abs * torch.cos(msssim_val_deg)
    msssim_val_imag = msssim_val_abs * torch.sin(msssim_val_deg)
    msssim_val = torch.stack((msssim_val_real, msssim_val_imag), dim=-1).mean(dim=1)
    return msssim_val


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int=11, kernel_sigma: float=1.5, data_range: Union[int, float]=1.0, reduction: str='mean', scale_weights: Optional[torch.Tensor]=None, k1: float=0.01, k2: float=0.03) ->torch.Tensor:
    """ Interface of Multi-scale Structural Similarity (MS-SSIM) index.
    Inputs supposed to be in range ``[0, data_range]`` with RGB channels order for colour images.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        scale_weights: Weights for different scales.
            If ``None``, default weights from the paper will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi-scale Structural Similarity (MS-SSIM) index. In case of 5D input tensors,
        complex value is returned as a tensor of size 2.

    References:
        Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
        Multi-scale Structural Similarity for Image Quality Assessment.
        IEEE Asilomar Conference on Signals, Systems and Computers, 37,
        https://ieeexplore.ieee.org/document/1292216 DOI:`10.1109/ACSSC.2003.1292216`

        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    Note:
        The size of the image should be at least ``(kernel_size - 1) * 2 ** (levels - 1) + 1``.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))
    x = x / float(data_range)
    y = y / float(data_range)
    if scale_weights is None:
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    else:
        scale_weights = scale_weights / scale_weights.sum()
    if scale_weights.size(0) != scale_weights.numel():
        raise ValueError(f'Expected a vector of weights, got {scale_weights.dim()}D tensor')
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1)
    _compute_msssim = _multi_scale_ssim_complex if x.dim() == 5 else _multi_scale_ssim
    msssim_val = _compute_msssim(x=x, y=y, data_range=data_range, kernel=kernel, scale_weights=scale_weights, k1=k1, k2=k2)
    return _reduce(msssim_val, reduction)


class MultiScaleSSIMLoss(_Loss):
    """Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        MSSIM = \\{mssim_1,\\dots,mssim_{N \\times C}\\}, \\\\
        mssim_{l}(x, y) = \\frac{(2 \\mu_{x,m} \\mu_{y,m} + c_1) }
        {(\\mu_{x,m}^2 +\\mu_{y,m}^2 + c_1)} \\prod_{j=1}^{m - 1}
        \\frac{(2 \\sigma_{xy,j} + c_2)}{(\\sigma_{x,j}^2 +\\sigma_{y,j}^2 + c_2)}

    where :math:`N` is the batch size, `C` is the channel size, `m` is the scale level (Default: 5).
    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        MultiscaleSSIMLoss(x, y) =
        \\begin{cases}
            \\operatorname{mean}(1 - MSSIM), &  \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(1 - MSSIM),  &  \\text{if reduction} = \\text{'sum'.}
        \\end{cases}

    For colour images channel order is RGB.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        scale_weights:  Weights for different scales.
            If ``None``, default weights from the paper will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        reduction: Specifies the reduction type: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> loss = MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
        Multi-scale Structural Similarity for Image Quality Assessment.
        IEEE Asilomar Conference on Signals, Systems and Computers, 37,
        https://ieeexplore.ieee.org/document/1292216
        DOI:`10.1109/ACSSC.2003.1292216`

        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI:`10.1109/TIP.2003.819861`

    Note:
        The size of the image should be at least ``(kernel_size - 1) * 2 ** (levels - 1) + 1``.
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int=11, kernel_sigma: float=1.5, k1: float=0.01, k2: float=0.03, scale_weights: Optional[torch.Tensor]=None, reduction: str='mean', data_range: Union[int, float]=1.0) ->None:
        super().__init__()
        self.reduction = reduction
        if scale_weights is None:
            self.scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.scale_weights = scale_weights
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Multi-scale Structural Similarity (MS-SSIM) index as a loss function.
        For colour images channel order is RGB.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
            y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.

        Returns:
            Value of MS-SSIM loss to be minimized, i.e. ``1 - ms_ssim`` in [0, 1] range. In case of 5D tensor,
            complex value is returned as a tensor of size 2.
        """
        score = multi_scale_ssim(x=x, y=y, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, data_range=self.data_range, reduction=self.reduction, scale_weights=self.scale_weights, k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


NORMALIZATION = 1000000.0


def _construct_graph_sparse(data: np.ndarray, k: int) ->np.ndarray:
    n = len(data)
    from scipy.sparse import lil_matrix
    spmat = lil_matrix((n, n))
    dd = np.sum(data * data, axis=1)
    for i in range(n):
        dists = dd - 2 * data[i, :].dot(data.T)
        inds = np.argpartition(dists, k + 1)[:k + 1]
        inds = inds[inds != i]
        spmat[i, inds] = 1
    return spmat.tocsr()


def _laplacian_sparse(matrix: np.ndarray, normalized: bool=True) ->np.ndarray:
    from scipy.sparse import diags
    from scipy.sparse import eye
    row_sum = matrix.sum(1).A1
    if not normalized:
        return diags(row_sum) - matrix
    row_sum_sqrt = diags(1 / np.sqrt(row_sum))
    return eye(matrix.shape[0]) - row_sum_sqrt.dot(matrix).dot(row_sum_sqrt)


def _build_graph(data: np.ndarray, k: int=5, normalized: bool=True):
    """Return Laplacian from data or load preconstructed from path

    Args:
        data: Samples.
        k: Number of neighbours for graph construction.
        normalized: if True, use nnormalized Laplacian.

    Returns:
        L: Laplacian of the graph constructed with data.
    """
    A = _construct_graph_sparse(data, k)
    A = (A + A.T) / 2
    A.data = np.ones(A.data.shape)
    L = _laplacian_sparse(A, normalized)
    return L


EPSILON = 1e-06


def _normalize_msid(msid: np.ndarray, normalization: str, n: int, k: int, ts: np.ndarray):
    normed_msid = msid.copy()
    if normalization == 'empty':
        normed_msid /= n
    elif normalization == 'complete':
        normed_msid /= 1 + (n - 1) * np.exp(-(1 + 1 / (n - 1)) * ts)
    elif normalization == 'er':
        xs = np.linspace(0, 1, n)
        er_spectrum = 4 / np.sqrt(k) * xs + 1 - 2 / np.sqrt(k)
        er_msid = np.exp(-np.outer(ts, er_spectrum)).sum(-1)
        normed_msid = normed_msid / (er_msid + EPSILON)
    elif not (normalization == 'none' or normalization is None):
        raise ValueError('Unknown normalization parameter!')
    return normed_msid


def _lanczos_m(A: np.ndarray, m: int, nv: int, rademacher: bool, starting_vectors: Optional[np.ndarray]=None) ->Tuple[np.ndarray, np.ndarray]:
    """Lanczos algorithm computes symmetric m x m tridiagonal matrix T and matrix V with orthogonal rows
        constituting the basis of the Krylov subspace K_m(A, x),
        where x is an arbitrary starting unit vector.
        This implementation parallelizes `nv` starting vectors.

    Args:
        A: matrix based on which the Krylov subspace will be built.
        m: Number of Lanczos steps.
        nv: Number of random vectors.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors
        starting_vectors: Specified starting vectors.

    Returns:
        T: Array with shape (nv, m, m), where T[i, :, :] is the i-th symmetric tridiagonal matrix.
        V: Array with shape (n, m, nv) where, V[:, :, i] is the i-th matrix with orthogonal rows.
    """
    orthtol = 1e-05
    if starting_vectors is None:
        if rademacher:
            starting_vectors = np.sign(np.random.randn(A.shape[0], nv))
        else:
            starting_vectors = np.random.randn(A.shape[0], nv)
    V = np.zeros((starting_vectors.shape[0], m, nv))
    T = np.zeros((nv, m, m))
    np.divide(starting_vectors, np.linalg.norm(starting_vectors, axis=0), out=starting_vectors)
    V[:, 0, :] = starting_vectors
    w = A.dot(starting_vectors)
    alpha = np.einsum('ij,ij->j', w, starting_vectors)
    w -= alpha[None, :] * starting_vectors
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)
    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta
    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w
    t = np.zeros((m, nv))
    for i in range(1, m):
        old_starting_vectors = V[:, i - 1, :]
        starting_vectors = V[:, i, :]
        w = A.dot(starting_vectors)
        w -= beta[None, :] * old_starting_vectors
        np.einsum('ij,ij->j', w, starting_vectors, out=alpha)
        T[:, i, i] = alpha
        if i < m - 1:
            w -= alpha[None, :] * starting_vectors
            np.einsum('ijk,ik->jk', V, w, out=t)
            w -= np.einsum('ijk,jk->ik', V, t)
            np.einsum('ij,ij->j', w, w, out=beta)
            np.sqrt(beta, beta)
            np.divide(w, beta[None, :], out=w)
            T[:, i, i + 1] = beta
            T[:, i + 1, i] = beta
            innerprod = np.einsum('ijk,ik->jk', V, w)
            reortho = False
            for _ in range(100):
                if not (innerprod > orthtol).sum():
                    reortho = True
                    break
                np.einsum('ijk,ik->jk', V, w, out=t)
                w -= np.einsum('ijk,jk->ik', V, t)
                np.divide(w, np.linalg.norm(w, axis=0)[None, :], out=w)
                innerprod = np.einsum('ijk,ik->jk', V, w)
            V[:, i + 1, :] = w
            if (np.abs(beta) > 1e-06).sum() == 0 or not reortho:
                break
    return T, V


def _slq_ts_fs(A: np.ndarray, m: int, niters: int, ts: np.ndarray, rademacher: bool, fs: List) ->np.ndarray:
    """Compute the trace of matrix functions

    Args:
        A: Square matrix in trace(exp(-t*A)), where t is temperature.
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        ts: Array with temperatures.
        rademacher: True to use Rademacher distribution, else - standard normal for random vectors in Hutchinson
        fs: A list of functions.

    Returns:
        traces: Estimate of traces for each of the functions in `fs`.
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    traces = np.zeros((len(fs), len(ts)))
    for i, f in enumerate(fs):
        expeig = f(-np.outer(ts, eigvals)).reshape(ts.shape[0], niters, m)
        sqeigv1 = np.power(eigvecs[:, 0, :], 2)
        traces[i, :] = A.shape[-1] * (expeig * sqeigv1).sum(-1).mean(-1)
    return traces


def _slq_red_var(A: np.ndarray, m: int, niters: int, ts: np.ndarray, rademacher: bool) ->np.ndarray:
    """Compute the trace of matrix exponential with reduced variance

    Args:
        A: Square matrix in trace(exp(-t*A)), where t is temperature.
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        ts: Array with temperatures.

    Returns:
        traces: Estimate of trace for each temperature value in `ts`.
    """
    fs = [np.exp, lambda x: x]
    traces = _slq_ts_fs(A, m, niters, ts, rademacher, fs)
    subee = traces[0, :] - traces[1, :] / np.exp(ts)
    sub = -ts * A.shape[0] / np.exp(ts)
    return subee + sub


def _msid_descriptor(x: np.ndarray, ts: np.ndarray=np.logspace(-1, 1, 256), k: int=5, m: int=10, niters: int=100, rademacher: bool=False, normalized_laplacian: bool=True, normalize: str='empty') ->np.ndarray:
    """Compute the msid descriptor for a single set of samples

    Args:
        x: Samples from data distribution. Shape (N_samples, data_dim)
        ts: Temperature values.
        k: Number of neighbours for graph construction.
        m: Lanczos steps in SLQ.
        niters: Number of starting random vectors for SLQ.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors in Hutchinson.
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi normalization, 'none' for no normalization
    Returns:
        normed_msidx: normalized msid descriptor
    """
    try:
        import scipy
    except ImportError:
        raise ImportError('Scipy is required for computation of the Geometry Score but not installed. Please install scipy using the following command: pip install --user scipy')
    recommended_scipy_version = _parse_version('1.3.3')
    scipy_version = _parse_version(scipy.__version__)
    if len(scipy_version) != 0 and scipy_version < recommended_scipy_version:
        warn(f'Scipy of version {scipy.__version__} is used while version >= {recommended_scipy_version} is recommended. Consider updating scipy to avoid potential long compute time with older versions.')
    Lx = _build_graph(x, k, normalized_laplacian)
    nx = Lx.shape[0]
    msidx = _slq_red_var(Lx, m, niters, ts, rademacher)
    normed_msidx = _normalize_msid(msidx, normalize, nx, k, ts) * NORMALIZATION
    return normed_msidx


class MSID(BaseFeatureMetric):
    """Creates a criterion that measures MSID score for two batches of images
    It's computed for a whole set of data and uses features from encoder instead of images itself
    to decrease computation cost. MSID can compare two data distributions with different
    number of samples or different dimensionalities.

    Args:
        ts: Temperature values. If ``None``, the default value ``torch.logspace(-1, 1, 256)`` is used.
        k: Number of neighbours for graph construction.
        m: Lanczos steps in SLQ.
        niters: Number of starting random vectors for SLQ.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors in Hutchinson.
        normalized_laplacian: if True, use normalized Laplacian.
        normalize: ``'empty'`` for average heat kernel (corresponds to the empty graph normalization of NetLSD),
            ``'complete'`` for the complete, ``'er'`` for Erdos-Renyi normalization, ``'none'`` for no normalization
        msid_mode: ``'l2'`` to compute the L2 norm of the distance between `msid1` and `msid2`;
            ``'max'`` to find the maximum absolute difference between two descriptors over temperature

    Examples:
        >>> msid_metric = MSID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> msid: torch.Tensor = msid_metric(x_feats, y_feats)

    References:
        Tsitsulin, A., Munkhoeva, M., Mottin, D., Karras, P., Bronstein, A., Oseledets, I., & Müller, E. (2019).
        The shape of data: Intrinsic distance for data distributions.
        https://arxiv.org/abs/1905.11141
    """

    def __init__(self, ts: torch.Tensor=None, k: int=5, m: int=10, niters: int=100, rademacher: bool=False, normalized_laplacian: bool=True, normalize: str='empty', msid_mode: str='max') ->None:
        super(MSID, self).__init__()
        if ts is None:
            ts = torch.logspace(-1, 1, 256)
        self.ts = ts.numpy()
        self.k = k
        self.m = m
        self.niters = niters
        self.rademacher = rademacher
        self.msid_mode = msid_mode
        self.normalized_laplacian = normalized_laplacian
        self.normalize = normalize

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) ->torch.Tensor:
        """Compute MSID score between two sets of samples.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D_x)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D_y)`

        Returns:
            Scalar value of the distance between distributions.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        normed_msid_x = _msid_descriptor(x_features.detach().cpu().numpy(), ts=self.ts, k=self.k, m=self.m, niters=self.niters, rademacher=self.rademacher, normalized_laplacian=self.normalized_laplacian, normalize=self.normalize)
        normed_msid_y = _msid_descriptor(y_features.detach().cpu().numpy(), ts=self.ts, k=self.k, m=self.m, niters=self.niters, rademacher=self.rademacher, normalized_laplacian=self.normalized_laplacian, normalize=self.normalize)
        c = np.exp(-2 * (self.ts + 1 / self.ts))
        if self.msid_mode == 'l2':
            score = np.linalg.norm(normed_msid_x - normed_msid_y)
        elif self.msid_mode == 'max':
            score = np.amax(c * np.abs(normed_msid_x - normed_msid_y))
        else:
            raise ValueError('Mode must be in {`l2`, `max`}')
        return torch.tensor(score, device=x_features.device)


EPS = 1e-10


IMAGENET_MEAN = [0.485, 0.456, 0.406]


IMAGENET_STD = [0.229, 0.224, 0.225]


VGG16_LAYERS = {'conv1_1': '0', 'relu1_1': '1', 'conv1_2': '2', 'relu1_2': '3', 'pool1': '4', 'conv2_1': '5', 'relu2_1': '6', 'conv2_2': '7', 'relu2_2': '8', 'pool2': '9', 'conv3_1': '10', 'relu3_1': '11', 'conv3_2': '12', 'relu3_2': '13', 'conv3_3': '14', 'relu3_3': '15', 'pool3': '16', 'conv4_1': '17', 'relu4_1': '18', 'conv4_2': '19', 'relu4_2': '20', 'conv4_3': '21', 'relu4_3': '22', 'pool4': '23', 'conv5_1': '24', 'relu5_1': '25', 'conv5_2': '26', 'relu5_2': '27', 'conv5_3': '28', 'relu5_3': '29', 'pool5': '30'}


VGG19_LAYERS = {'conv1_1': '0', 'relu1_1': '1', 'conv1_2': '2', 'relu1_2': '3', 'pool1': '4', 'conv2_1': '5', 'relu2_1': '6', 'conv2_2': '7', 'relu2_2': '8', 'pool2': '9', 'conv3_1': '10', 'relu3_1': '11', 'conv3_2': '12', 'relu3_2': '13', 'conv3_3': '14', 'relu3_3': '15', 'conv3_4': '16', 'relu3_4': '17', 'pool3': '18', 'conv4_1': '19', 'relu4_1': '20', 'conv4_2': '21', 'relu4_2': '22', 'conv4_3': '23', 'relu4_3': '24', 'conv4_4': '25', 'relu4_4': '26', 'pool4': '27', 'conv5_1': '28', 'relu5_1': '29', 'conv5_2': '30', 'relu5_2': '31', 'conv5_3': '32', 'relu5_3': '33', 'conv5_4': '34', 'relu5_4': '35', 'pool5': '36'}


class ContentLoss(_Loss):
    """Creates Content loss that can be used for image style transfer or as a measure for image to image tasks.
    Uses pretrained VGG models from torchvision.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]

    Args:
        feature_extractor: Model to extract features or model name: ``'vgg16'`` | ``'vgg19'``.
        layers: List of strings with layer names. Default: ``'relu3_3'``
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See references for details.
        distance: Method to compute distance between features: ``'mse'`` | ``'mae'``.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
        normalize_features: If true, unit-normalize each feature in channel dimension before scaling
            and computing distance. See references for details.

    Examples:
        >>> loss = ContentLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Gatys, Leon and Ecker, Alexander and Bethge, Matthias (2016).
        A Neural Algorithm of Artistic Style
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576

        Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
    """

    def __init__(self, feature_extractor: Union[str, torch.nn.Module]='vgg16', layers: Collection[str]=('relu3_3',), weights: List[Union[float, torch.Tensor]]=[1.0], replace_pooling: bool=False, distance: str='mse', reduction: str='mean', mean: List[float]=IMAGENET_MEAN, std: List[float]=IMAGENET_STD, normalize_features: bool=False, allow_layers_weights_mismatch: bool=False) ->None:
        assert allow_layers_weights_mismatch or len(layers) == len(weights), f'Lengths of provided layers and weighs mismatch ({len(weights)} weights and {len(layers)} layers), which will cause incorrect results. Please provide weight for each layer.'
        super().__init__()
        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        elif feature_extractor == 'vgg16':
            self.model = vgg16(pretrained=True, progress=False).features
            self.layers = [VGG16_LAYERS[l] for l in layers]
        elif feature_extractor == 'vgg19':
            self.model = vgg19(pretrained=True, progress=False).features
            self.layers = [VGG19_LAYERS[l] for l in layers]
        else:
            raise ValueError('Unknown feature extractor')
        if replace_pooling:
            self.model = self.replace_pooling(self.model)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.distance = {'mse': nn.MSELoss, 'mae': nn.L1Loss}[distance](reduction='none')
        self.weights = [(torch.tensor(w) if not isinstance(w, torch.Tensor) else w) for w in weights]
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        self.normalize_features = normalize_features
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Content loss between feature representations of prediction :math:`x` and
        target :math:`y` tensors.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Content loss between feature representations
        """
        _validate_input([x, y], dim_range=(4, 4), data_range=(0, -1))
        self.model
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        distances = self.compute_distance(x_features, y_features)
        loss = torch.cat([(d * w).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)
        return _reduce(loss, self.reduction)

    def compute_distance(self, x_features: List[torch.Tensor], y_features: List[torch.Tensor]) ->List[torch.Tensor]:
        """Take L2 or L1 distance between feature maps depending on ``distance``.

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Distance between feature maps
        """
        return [self.distance(x, y) for x, y in zip(x_features, y_features)]

    def get_features(self, x: torch.Tensor) ->List[torch.Tensor]:
        """
        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            List of features extracted from intermediate layers
        """
        x = (x - self.mean) / self.std
        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(self.normalize(x) if self.normalize_features else x)
        return features

    @staticmethod
    def normalize(x: torch.Tensor) ->torch.Tensor:
        """Normalize feature maps in channel direction to unit length.

        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Normalized input
        """
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + EPS)

    def replace_pooling(self, module: torch.nn.Module) ->torch.nn.Module:
        """Turn All MaxPool layers into AveragePool

        Args:
            module: Module to change MaxPool int AveragePool

        Returns:
            Module with AveragePool instead MaxPool

        """
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output


class StyleLoss(ContentLoss):
    """Creates Style loss that can be used for image style transfer or as a measure in
    image to image tasks. Computes distance between Gram matrices of feature maps.
    Uses pretrained VGG models from torchvision.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        feature_extractor: Model to extract features or model name: ``'vgg16'`` | ``'vgg19'``.
        layers: List of strings with layer names. Default: ``'relu3_3'``
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See references for details.
        distance: Method to compute distance between features: ``'mse'`` | ``'mae'``.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
        normalize_features: If true, unit-normalize each feature in channel dimension before scaling
            and computing distance. See references for details.

    Examples:
        >>> loss = StyleLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Gatys, Leon and Ecker, Alexander and Bethge, Matthias (2016).
        A Neural Algorithm of Artistic Style
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576

        Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
    """

    def compute_distance(self, x_features: torch.Tensor, y_features: torch.Tensor):
        """Take L2 or L1 distance between Gram matrices of feature maps depending on ``distance``.

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Distance between Gram matrices
        """
        x_gram = [self.gram_matrix(x) for x in x_features]
        y_gram = [self.gram_matrix(x) for x in y_features]
        return [self.distance(x, y) for x, y in zip(x_gram, y_gram)]

    @staticmethod
    def gram_matrix(x: torch.Tensor) ->torch.Tensor:
        """Compute Gram matrix for batch of features.

        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Gram matrix for given input
        """
        B, C, H, W = x.size()
        gram = []
        for i in range(B):
            features = x[i].view(C, H * W)
            gram.append(torch.mm(features, features.t()).unsqueeze(0))
        return torch.stack(gram)


class LPIPS(ContentLoss):
    """Learned Perceptual Image Patch Similarity metric. Only VGG16 learned weights are supported.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See references for details.
        distance: Method to compute distance between features: ``'mse'`` | ``'mae'``.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].

    Examples:
        >>> loss = LPIPS()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Gatys, Leon and Ecker, Alexander and Bethge, Matthias (2016).
        A Neural Algorithm of Artistic Style
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576

        Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
        https://github.com/richzhang/PerceptualSimilarity
    """
    _weights_url = 'https://github.com/photosynthesis-team/' + 'photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt'

    def __init__(self, replace_pooling: bool=False, distance: str='mse', reduction: str='mean', mean: List[float]=IMAGENET_MEAN, std: List[float]=IMAGENET_STD) ->None:
        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        super().__init__('vgg16', layers=lpips_layers, weights=lpips_weights, replace_pooling=replace_pooling, distance=distance, reduction=reduction, mean=mean, std=std, normalize_features=True)


class DISTS(ContentLoss):
    """Deep Image Structure and Texture Similarity metric.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].

    Examples:
        >>> loss = DISTS()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli (2020).
        Image Quality Assessment: Unifying Structure and Texture Similarity.
        https://arxiv.org/abs/2004.07728
        https://github.com/dingkeyan93/DISTS
    """
    _weights_url = 'https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt'

    def __init__(self, reduction: str='mean', mean: List[float]=IMAGENET_MEAN, std: List[float]=IMAGENET_STD) ->None:
        dists_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]
        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        dists_weights = list(torch.split(weights['alpha'], channels, dim=1))
        dists_weights.extend(torch.split(weights['beta'], channels, dim=1))
        super().__init__('vgg16', layers=dists_layers, weights=dists_weights, replace_pooling=True, reduction=reduction, mean=mean, std=std, normalize_features=False, allow_layers_weights_mismatch=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Deep Image Structure and Texture Similarity loss, i.e. ``1-DISTS`` in range [0, 1].
        """
        _, _, H, W = x.shape
        if min(H, W) > 256:
            x = torch.nn.functional.interpolate(x, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')
            y = torch.nn.functional.interpolate(y, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')
        loss = super().forward(x, y)
        return 1 - loss

    def compute_distance(self, x_features: torch.Tensor, y_features: torch.Tensor) ->List[torch.Tensor]:
        """Compute structure similarity between feature maps

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Structural similarity distance between feature maps
        """
        structure_distance, texture_distance = [], []
        EPS = 1e-06
        for x, y in zip(x_features, y_features):
            x_mean = x.mean([2, 3], keepdim=True)
            y_mean = y.mean([2, 3], keepdim=True)
            structure_distance.append(similarity_map(x_mean, y_mean, constant=EPS))
            x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
            texture_distance.append((2 * xy_cov + EPS) / (x_var + y_var + EPS))
        return structure_distance + texture_distance

    def get_features(self, x: torch.Tensor) ->List[torch.Tensor]:
        """

        Args:
            x: Input tensor

        Returns:
            List of features extracted from input tensor
        """
        features = super().get_features(x)
        features.insert(0, x)
        return features

    def replace_pooling(self, module: torch.nn.Module) ->torch.nn.Module:
        """Turn All MaxPool layers into L2Pool

        Args:
            module: Module to change MaxPool into L2Pool

        Returns:
            Module with L2Pool instead of MaxPool
        """
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = L2Pool2d(kernel_size=3, stride=2, padding=1)
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output


class PieAPPModel(nn.Module):
    """ Model used for PieAPP score computation """
    FEATURES = 64

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)
        self.conv1 = nn.Conv2d(3, self.FEATURES, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.FEATURES, self.FEATURES, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.FEATURES, self.FEATURES, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.FEATURES, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 8, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(self.FEATURES * 8, self.FEATURES * 8, kernel_size=3, padding=1)
        self.fc1_score = nn.Linear(in_features=120832, out_features=512, bias=True)
        self.fc2_score = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc1_weight = nn.Linear(in_features=2048, out_features=512)
        self.fc2_weight = nn.Linear(in_features=512, out_features=1, bias=True)
        self.ref_score_subtract = nn.Linear(in_features=1, out_features=1, bias=True)
        self.EPS = 1e-06

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass a batch of square patches with shape :math:`(N, C, F, F)`.

        Returns:
            features: Concatenation of model features from different scales
            x11: Outputs of the last convolutional layer used as weights
        """
        _validate_input([x], dim_range=(4, 4), data_range=(0, -1))
        assert x.shape[2] == x.shape[3] == self.FEATURES, f'Expected square input with shape {self.FEATURES, self.FEATURES}, got {x.shape}'
        x3 = F.relu(self.conv3(self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))))
        x5 = F.relu(self.conv5(self.pool(F.relu(self.conv4(x3)))))
        x7 = F.relu(self.conv7(self.pool(F.relu(self.conv6(x5)))))
        x9 = F.relu(self.conv9(self.pool(F.relu(self.conv8(x7)))))
        x11 = self.flatten(F.relu(self.conv11(self.pool(F.relu(self.conv10(x9))))))
        features = torch.cat((self.flatten(x3), self.flatten(x5), self.flatten(x7), self.flatten(x9), x11), dim=1)
        return features, x11

    def compute_difference(self, features_diff: torch.Tensor, weights_diff: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_diff: Tensor. Shape :math:`(N, C_1)`
            weights_diff: Tensor. Shape :math:`(N, C_2)`

        Returns:
            distances
            weights
        """
        distances = self.ref_score_subtract(0.01 * self.fc2_score(F.relu(self.fc1_score(features_diff))))
        weights = self.fc2_weight(F.relu(self.fc1_weight(weights_diff))) + self.EPS
        return distances, weights


def crop_patches(images: torch.Tensor, size: int=64, stride: int=32) ->torch.Tensor:
    """Crop input images into smaller patches.

    Args:
        images: Tensor of images with shape (batch x 3 x H x W)
        size: size of a square patch
        stride: Step between patches
    Returns:
        A tensor on cropped patches of shape (-1, 3, size, size)
    """
    patches = images.data.unfold(1, 3, 3).unfold(2, size, stride).unfold(3, size, stride)
    patches = patches.reshape(-1, 3, size, size)
    return patches


class PieAPP(_Loss):
    """
    Implementation of Perceptual Image-Error Assessment through Pairwise Preference.

    Expects input to be in range ``[0, data_range]`` with no normalization and RGB channel order.
    Input images are cropped into smaller patches. Score for each individual image is mean of it's patch scores.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        stride: Step between cropped patches. Smaller values lead to better quality,
            but cause higher memory consumption. Default: 27 (`sparse` sampling in original implementation)
        enable_grad: Flag to compute gradients. Useful when PieAPP used as a loss. Default: False.

    Examples:
        >>> loss = PieAPP()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen (2018).
        PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
        https://arxiv.org/abs/1806.02067

        https://github.com/prashnani/PerceptualImageError

    """
    _weights_url = 'https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth'

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, stride: int=27, enable_grad: bool=False) ->None:
        super().__init__()
        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        weights['ref_score_subtract.weight'] = weights['ref_score_subtract.weight'].unsqueeze(1)
        self.model = PieAPPModel()
        self.model.load_state_dict(weights)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.data_range = data_range
        self.reduction = reduction
        self.stride = stride
        self.enable_grad = enable_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """
        Computation of PieAPP  between feature representations of prediction :math:`x` and target :math:`y` tensors.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Perceptual Image-Error Assessment through Pairwise Preference
        """
        _validate_input([x, y], dim_range=(4, 4), data_range=(0, self.data_range))
        N, C, _, _ = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            warnings.warn('The original PieAPP supports only RGB images.The input images were converted to RGB by copying the grey channel 3 times.')
        self.model
        x_features, x_weights = self.get_features(x)
        y_features, y_weights = self.get_features(y)
        distances, weights = self.model.compute_difference(y_features - x_features, y_weights - x_weights)
        distances = distances.reshape(N, -1)
        weights = weights.reshape(N, -1)
        loss = torch.stack([((d * w).sum() / w.sum()) for d, w in zip(distances, weights)])
        return _reduce(loss, self.reduction)

    def get_features(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            List of features extracted from intermediate layers weights
        """
        x = x / float(self.data_range) * 255
        x_patches = crop_patches(x, size=64, stride=self.stride)
        with torch.autograd.set_grad_enabled(self.enable_grad):
            features, weights = self.model(x_patches)
        return features, weights


def _compute_pairwise_distance(data_x: torch.Tensor, data_y: Optional[torch.Tensor]=None) ->torch.Tensor:
    """Compute Euclidean distance between :math:`x` and :math:`y`.

    Args:
        data_x: Tensor of shape :math:`(N, feature_dim)`
        data_y: Tensor of shape :math:`(N, feature_dim)`
    Returns:
        Tensor of shape :math:`(N, N)` of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = torch.cdist(data_x, data_y, p=2)
    return dists


def _get_kth_value(unsorted: torch.Tensor, k: int, axis: int=-1) ->torch.Tensor:
    """
    Args:
        unsorted: Tensor of any dimensionality.
        k: Int of the :math:`k`-th value to retrieve.
    Returns:
        kth values along the designated axis.
    """
    k_smallests = torch.topk(unsorted, k, dim=axis, largest=False)[0]
    kth_values = k_smallests.max(dim=axis)[0]
    return kth_values


def _compute_nearest_neighbour_distances(input_features: torch.Tensor, nearest_k: int) ->torch.Tensor:
    """Compute K-nearest neighbour distances.

    Args:
        input_features: Tensor of shape :math:`(N, feature_dim)`
        nearest_k: Int of the :math:`k`-th nearest neighbour.
    Returns:
        Distances to :math:`k`-th nearest neighbours.
    """
    distances = _compute_pairwise_distance(input_features)
    radii = _get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


class PR(BaseFeatureMetric):
    """Interface of Improved Precision and Recall.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. Precision and Recall can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        nearest_k: Nearest neighbor to compute the non-parametric representation. Shape :math:`1`

    Examples:
        >>> pr_metric = PR()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> precision, recall = pr_metric(x_feats, y_feats)

    References:
        Kynkäänniemi T. et al. (2019).
        Improved Precision and Recall Metric for Assessing Generative Models.
        Advances in Neural Information Processing Systems,
        https://arxiv.org/abs/1904.06991
    """

    def __init__(self, nearest_k: int=5) ->None:
        """
        Args:
            nearest_k: Nearest neighbor to compute the non-parametric representation. Shape :math:`1`
        """
        super(PR, self).__init__()
        self.nearest_k = nearest_k

    def compute_metric(self, real_features: torch.Tensor, fake_features: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Creates non-parametric representations of the manifolds of real and generated data and computes
        the precision and recall between them.

        Args:
            real_features: Samples from data distribution. Shape :math:`(N_x, D)`
            fake_features: Samples from fake distribution. Shape :math:`(N_x, D)`
        Returns:
            Scalar value of the precision of the generated images.

            Scalar value of the recall of the generated images.
        """
        _validate_input([real_features, fake_features], dim_range=(2, 2), size_range=(1, 2))
        real_nearest_neighbour_distances = _compute_nearest_neighbour_distances(real_features, self.nearest_k).unsqueeze(1)
        fake_nearest_neighbour_distances = _compute_nearest_neighbour_distances(fake_features, self.nearest_k).unsqueeze(0)
        distance_real_fake = _compute_pairwise_distance(real_features, fake_features)
        precision = torch.logical_or(distance_real_fake < real_nearest_neighbour_distances, torch.isclose(distance_real_fake, real_nearest_neighbour_distances)).any(dim=0).float().mean()
        recall = torch.logical_or(distance_real_fake < fake_nearest_neighbour_distances, torch.isclose(distance_real_fake, real_nearest_neighbour_distances)).any(dim=1).float().mean()
        return precision, recall


def _spectral_residual_visual_saliency(x: torch.Tensor, scale: float=0.25, kernel_size: int=3, sigma: float=3.8, gaussian_size: int=10) ->torch.Tensor:
    """Compute Spectral Residual Visual Saliency
    Credits X. Hou and L. Zhang, CVPR 07, 2007
    Reference:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.5641&rep=rep1&type=pdf

    Args:
        x: Tensor with shape (N, 1, H, W).
        scale: Resizing factor
        kernel_size: Kernel size of average blur filter
        sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        saliency_map: Tensor with shape BxHxW

    """
    eps = torch.finfo(x.dtype).eps
    for kernel in (kernel_size, gaussian_size):
        if x.size(-1) * scale < kernel or x.size(-2) * scale < kernel:
            raise ValueError(f"Kernel size can't be greater than actual input size. Input size: {x.size()} x {scale}. Kernel size: {kernel}")
    in_img = imresize(x, scale=scale)
    recommended_torch_version = _parse_version('1.8.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        imagefft = torch.fft.fft2(in_img)
        log_amplitude = torch.log(imagefft.abs() + eps)
        phase = torch.angle(imagefft)
    else:
        imagefft = torch.rfft(in_img, 2, onesided=False)
        log_amplitude = torch.log(imagefft.pow(2).sum(dim=-1).sqrt() + eps)
        phase = torch.atan2(imagefft[..., 1], imagefft[..., 0] + eps)
    padding = kernel_size // 2
    if padding:
        up_pad = (kernel_size - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        spectral_residual = F.pad(log_amplitude, pad=pad_to_use, mode='replicate')
    else:
        spectral_residual = log_amplitude
    spectral_residual = log_amplitude - F.avg_pool2d(spectral_residual, kernel_size=kernel_size, stride=1)
    compx = torch.stack((torch.exp(spectral_residual) * torch.cos(phase), torch.exp(spectral_residual) * torch.sin(phase)), -1)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        saliency_map = torch.abs(torch.fft.ifft2(torch.view_as_complex(compx))) ** 2
    else:
        saliency_map = torch.sum(torch.ifft(compx, 2) ** 2, dim=-1)
    kernel = gaussian_filter(gaussian_size, sigma)
    if gaussian_size % 2 == 0:
        kernel = torch.cat((torch.zeros(1, 1, gaussian_size), kernel), 1)
        kernel = torch.cat((torch.zeros(1, gaussian_size + 1, 1), kernel), 2)
        gaussian_size += 1
    kernel = kernel.view(1, 1, gaussian_size, gaussian_size)
    saliency_map = F.conv2d(saliency_map, kernel, padding=(gaussian_size - 1) // 2)
    min_sal = torch.min(saliency_map[:])
    max_sal = torch.max(saliency_map[:])
    saliency_map = (saliency_map - min_sal) / (max_sal - min_sal + eps)
    saliency_map = imresize(saliency_map, sizes=x.size()[-2:])
    return saliency_map


def srsim(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, chromatic: bool=False, scale: float=0.25, kernel_size: int=3, sigma: float=3.8, gaussian_size: int=10) ->torch.Tensor:
    """Compute Spectral Residual based Similarity for a batch of images.

    Args:
        x: Predicted images. Shape (H, W), (C, H, W) or (N, C, H, W).
        y: Target images. Shape (H, W), (C, H, W) or (N, C, H, W).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        chromatic: Flag to compute SR-SIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        SR-SIM: Index of similarity between two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than the original ones.
    Note:
        This implementation is based on the original MATLAB code.
        https://sse.tongji.edu.cn/linzhang/IQA/SR-SIM/Files/SR_SIM.m

    """
    _validate_input(tensors=[x, y], dim_range=(4, 4), data_range=(0, data_range))
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    ksize = max(1, round(min(x.size()[-2:]) / 256))
    padding = ksize // 2
    if padding:
        up_pad = (ksize - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x = F.pad(x, pad=pad_to_use)
        y = F.pad(y, pad=pad_to_use)
    x = F.avg_pool2d(x, ksize)
    y = F.avg_pool2d(y, ksize)
    num_channels = x.size(1)
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)
        x_lum = x_yiq[:, :1]
        y_lum = y_yiq[:, :1]
        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]
    else:
        if chromatic:
            raise ValueError('Chromatic component can be computed only for RGB images!')
        x_lum = x
        y_lum = y
    svrs_x = _spectral_residual_visual_saliency(x_lum, scale=scale, kernel_size=kernel_size, sigma=sigma, gaussian_size=gaussian_size)
    svrs_y = _spectral_residual_visual_saliency(y_lum, scale=scale, kernel_size=kernel_size, sigma=sigma, gaussian_size=gaussian_size)
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)
    C1, C2, alpha = 0.4, 225, 0.5
    SVRS = similarity_map(svrs_x, svrs_y, C1)
    GM = similarity_map(grad_map_x, grad_map_y, C2)
    svrs_max = torch.where(svrs_x > svrs_y, svrs_x, svrs_y)
    score = SVRS * GM ** alpha * svrs_max
    if chromatic:
        T3, T4, lmbda = 200, 200, 0.03
        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
    eps = torch.finfo(score.dtype).eps
    result = score.sum(dim=[1, 2, 3]) / (svrs_max.sum(dim=[1, 2, 3]) + eps)
    return _reduce(result, reduction)


class SRSIMLoss(_Loss):
    """Creates a criterion that measures the SR-SIM or SR-SIMc for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value `1 - clip(SR-SIM, min=0, max=1)` is returned. If you need SR-SIM value,
    use function `srsim` instead.

    Args:
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        chromatic: Flag to compute SRSIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map

    Shape:
        - Input: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.
        - Target: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.

    Examples::

        >>> loss = SRSIMLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf
        """

    def __init__(self, reduction: str='mean', data_range: Union[int, float]=1.0, chromatic: bool=False, scale: float=0.25, kernel_size: int=3, sigma: float=3.8, gaussian_size: int=10) ->None:
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.srsim = functools.partial(srsim, reduction=reduction, data_range=data_range, chromatic=chromatic, scale=scale, kernel_size=kernel_size, sigma=sigma, gaussian_size=gaussian_size)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Computation of SR-SIM as a loss function.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        Returns:
            Value of SR-SIM loss to be minimized. 0 <= SR-SIM <= 1.
        """
        score = self.srsim(prediction, target)
        return 1 - torch.clamp(score, 0, 1)


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int=11, kernel_sigma: float=1.5, data_range: Union[int, float]=1.0, reduction: str='mean', full: bool=False, downsample: bool=True, k1: float=0.01, k2: float=0.03) ->List[torch.Tensor]:
    """Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.
    To match performance with skimage and tensorflow set ``'downsample' = True``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        full: Return cs map or not.
        downsample: Perform average pool before SSIM computation. Default: True
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))
    x = x / float(data_range)
    y = y / float(data_range)
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1 and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)
    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)
    if full:
        return [ssim_val, cs]
    return ssim_val


class SSIMLoss(_Loss):
    """Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    To match performance with skimage and tensorflow set ``'downsample' = True``.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \\{ssim_1,\\dots,ssim_{N \\times C}\\}\\\\
        ssim_{l}(x, y) = \\frac{(2 \\mu_x \\mu_y + c_1) (2 \\sigma_{xy} + c_2)}
        {(\\mu_x^2 +\\mu_y^2 + c_1)(\\sigma_x^2 +\\sigma_y^2 + c_2)},

    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        SSIMLoss(x, y) =
        \\begin{cases}
            \\operatorname{mean}(1 - SSIM), &  \\text{if reduction} = \\text{'mean';}\\\\
            \\operatorname{sum}(1 - SSIM),  &  \\text{if reduction} = \\text{'sum'.}
        \\end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        downsample: Perform average pool before SSIM computation. Default: True
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> loss = SSIMLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int=11, kernel_sigma: float=1.5, k1: float=0.01, k2: float=0.03, downsample: bool=True, reduction: str='mean', data_range: Union[int, float]=1.0) ->None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
            y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.

        Returns:
            Value of SSIM loss to be minimized, i.e ``1 - ssim`` in [0, 1] range. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """
        score = ssim(x=x, y=y, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma, downsample=self.downsample, data_range=self.data_range, reduction=self.reduction, full=False, k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def total_variation(x: torch.Tensor, reduction: str='mean', norm_type: str='l2') ->torch.Tensor:
    """Compute Total Variation metric

    Args:
        x: Tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        norm_type: ``'l1'`` | ``'l2'`` | ``'l2_squared'``,
            defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        Total variation of a given tensor

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """
    _validate_input([x], dim_range=(4, 4), data_range=(0, -1))
    if norm_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=[1, 2, 3])
        score = h_variance + w_variance
    elif norm_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = torch.sqrt(h_variance + w_variance)
    elif norm_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = h_variance + w_variance
    else:
        raise ValueError("Incorrect norm type, should be one of {'l1', 'l2', 'l2_squared'}")
    return _reduce(score, reduction)


class TVLoss(_Loss):
    """Creates a criterion that measures the total variation of the
    the given input :math:`x`.


    If :attr:`norm_type` set to ``'l2'`` the loss can be described as:

    .. math::
        TV(x) = \\sum_{N}\\sqrt{\\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}

    Else if :attr:`norm_type` set to ``'l1'``:

    .. math::
        TV(x) = \\sum_{N}\\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|)

    where :math:`N` is the batch size, `C` is the channel size.

    Args:
        norm_type: one of ``'l1'`` | ``'l2'`` | ``'l2_squared'``
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:

        >>> loss = TVLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(x)
        >>> output.backward()

    References:
        https://www.wikiwand.com/en/Total_variation_denoising

        https://remi.flamary.com/demos/proxtv.html
    """

    def __init__(self, norm_type: str='l2', reduction: str='mean'):
        super().__init__()
        self.norm_type = norm_type
        self.reduction = reduction

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computation of Total Variation (TV) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of TV loss to be minimized.
        """
        score = total_variation(x, reduction=self.reduction, norm_type=self.norm_type)
        return score


def vif_p(x: torch.Tensor, y: torch.Tensor, sigma_n_sq: float=2.0, data_range: Union[int, float]=1.0, reduction: str='mean') ->torch.Tensor:
    """Compute Visiual Information Fidelity in **pixel** domain for a batch of images.
    This metric isn't symmetric, so make sure to place arguments in correct order.
    Both inputs supposed to have RGB channels order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        sigma_n_sq: HVS model parameter (variance of the visual noise).
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Returns:
        VIF Index of similarity between two images. Usually in [0, 1] interval.
        Can be bigger than 1 for predicted :math:`x` images with higher contrast than original one.

    References:
        H. R. Sheikh and A. C. Bovik, "Image information and visual quality,"
        IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, Feb. 2006
        https://ieeexplore.ieee.org/abstract/document/1576816/
        DOI: 10.1109/TIP.2005.859378.

    Note:
        In original paper this method was used for bands in discrete wavelet decomposition.
        Later on authors released code to compute VIF approximation in pixel domain.
        See https://live.ece.utexas.edu/research/Quality/VIF.htm for details.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    min_size = 41
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    num_channels = x.size(1)
    if num_channels == 3:
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        y = 0.299 * y[:, 0, :, :] + 0.587 * y[:, 1, :, :] + 0.114 * y[:, 2, :, :]
        x = x[:, None, :, :]
        y = y[:, None, :, :]
    EPS = 1e-08
    x_vif, y_vif = 0, 0
    for scale in range(4):
        kernel_size = 2 ** (4 - scale) + 1
        kernel = gaussian_filter(kernel_size, sigma=kernel_size / 5)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        if scale > 0:
            x = F.conv2d(x, kernel)[:, :, ::2, ::2]
            y = F.conv2d(y, kernel)[:, :, ::2, ::2]
        mu_x, mu_y = F.conv2d(x, kernel), F.conv2d(y, kernel)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x_sq = F.conv2d(x ** 2, kernel) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, kernel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel) - mu_xy
        sigma_x_sq = torch.relu(sigma_x_sq)
        sigma_y_sq = torch.relu(sigma_y_sq)
        g = sigma_xy / (sigma_y_sq + EPS)
        sigma_v_sq = sigma_x_sq - g * sigma_xy
        g = torch.where(sigma_y_sq >= EPS, g, torch.zeros_like(g))
        sigma_v_sq = torch.where(sigma_y_sq >= EPS, sigma_v_sq, sigma_x_sq)
        sigma_y_sq = torch.where(sigma_y_sq >= EPS, sigma_y_sq, torch.zeros_like(sigma_y_sq))
        g = torch.where(sigma_x_sq >= EPS, g, torch.zeros_like(g))
        sigma_v_sq = torch.where(sigma_x_sq >= EPS, sigma_v_sq, torch.zeros_like(sigma_v_sq))
        sigma_v_sq = torch.where(g >= 0, sigma_v_sq, sigma_x_sq)
        g = torch.relu(g)
        sigma_v_sq = torch.where(sigma_v_sq > EPS, sigma_v_sq, torch.ones_like(sigma_v_sq) * EPS)
        x_vif_scale = torch.log10(1.0 + g ** 2.0 * sigma_y_sq / (sigma_v_sq + sigma_n_sq))
        x_vif = x_vif + torch.sum(x_vif_scale, dim=[1, 2, 3])
        y_vif = y_vif + torch.sum(torch.log10(1.0 + sigma_y_sq / sigma_n_sq), dim=[1, 2, 3])
    score: torch.Tensor = (x_vif + EPS) / (y_vif + EPS)
    return _reduce(score, reduction)


class VIFLoss(_Loss):
    """Creates a criterion that measures the Visual Information Fidelity loss
    between predicted (x) and target (y) image. In order to be considered as a loss,
    value ``1 - clip(VIF, min=0, max=1)`` is returned.

    Args:
        sigma_n_sq: HVS model parameter (variance of the visual noise).
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:
        >>> loss = VIFLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        H. R. Sheikh and A. C. Bovik, "Image information and visual quality,"
        IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, Feb. 2006
        https://ieeexplore.ieee.org/abstract/document/1576816/
        DOI: 10.1109/TIP.2005.859378.
    """

    def __init__(self, sigma_n_sq: float=2.0, data_range: Union[int, float]=1.0, reduction: str='mean'):
        super().__init__()
        self.sigma_n_sq = sigma_n_sq
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """Computation of Visual Information Fidelity (VIF) index as a loss function.
        Colour images are expected to have RGB channel order.
        Order of inputs is important! First tensor must contain distorted images, second reference images.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of VIF loss to be minimized in [0, 1] range.
        """
        score = vif_p(x, y, sigma_n_sq=self.sigma_n_sq, data_range=self.data_range, reduction=self.reduction)
        loss = 1 - torch.clamp(score, 0, 1)
        return loss


def rgb2lmn(x: torch.Tensor) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LMN images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LMN colour space.
    """
    weights_rgb_to_lmn = torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t()
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_lmn).permute(0, 3, 1, 2)
    return x_lmn


def _log_gabor(size: Tuple[int, int], omega_0: float, sigma_f: float) ->torch.Tensor:
    """Creates log Gabor filter

    Args:
        size: size of the requires log Gabor filter
        omega_0: center frequency of the filter
        sigma_f: bandwidth of the filter

    Returns:
        log Gabor filter
    """
    xx, yy = get_meshgrid(size)
    radius = (xx ** 2 + yy ** 2).sqrt()
    mask = radius <= 0.5
    r = radius * mask
    r = ifftshift(r)
    r[0, 0] = 1
    lg = torch.exp(-(r / omega_0).log().pow(2) / (2 * sigma_f ** 2))
    lg[0, 0] = 0
    return lg


def rgb2xyz(x: torch.Tensor) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of XYZ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). XYZ colour space.
    """
    mask_below = x <= 0.04045
    mask_above = x > 0.04045
    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above
    weights_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175], [0.0193339, 0.119192, 0.9503041]])
    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_rgb_to_xyz.t()).permute(0, 3, 1, 2)
    return x_xyz


def xyz2lab(x: torch.Tensor, illuminant: str='D50', observer: str='2') ->torch.Tensor:
    """Convert a batch of XYZ images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants: Dict[str, Dict] = {'A': {'2': (1.098466069456375, 1, 0.3558228003436005), '10': (1.111420406956693, 1, 0.3519978321919493)}, 'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288), '10': (0.9672062750333777, 1, 0.8142801513128616)}, 'D55': {'2': (0.956797052643698, 1, 0.9214805860173273), '10': (0.9579665682254781, 1, 0.9092525159847462)}, 'D65': {'2': (0.95047, 1.0, 1.08883), '10': (0.94809667673716, 1, 1.0730513595166162)}, 'D75': {'2': (0.9497220898840717, 1, 1.226393520724154), '10': (0.9441713925645873, 1, 1.2064272211720228)}, 'E': {'2': (1.0, 1.0, 1.0), '10': (1.0, 1.0, 1.0)}}
    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).view(1, 3, 1, 1)
    tmp = x / illuminants_to_use
    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = torch.pow(tmp, 1.0 / 3.0) * mask_above + (kappa * tmp + 16.0) / 116.0 * mask_below
    weights_xyz_to_lab = torch.tensor([[0, 116.0, 0], [500.0, -500.0, 0], [0, 200.0, -200.0]])
    bias_xyz_to_lab = torch.tensor([-16.0, 0.0, 0.0]).view(1, 3, 1, 1)
    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_xyz_to_lab.t()).permute(0, 3, 1, 2) + bias_xyz_to_lab
    return x_lab


def rgb2lab(x: torch.Tensor, data_range: Union[int, float]=255) ->torch.Tensor:
    """Convert a batch of RGB images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
        data_range: dynamic range of the input image.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    return xyz2lab(rgb2xyz(x / float(data_range)))


def sdsp(x: torch.Tensor, data_range: Union[int, float]=255, omega_0: float=0.021, sigma_f: float=1.34, sigma_d: float=145.0, sigma_c: float=0.001) ->torch.Tensor:
    """SDSP algorithm for salient region detection from a given image.

    Supports only colour images with RGB channel order.

    Args:
        x: Tensor. Shape :math:`(N, 3, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        omega_0: coefficient for log Gabor filter
        sigma_f: coefficient for log Gabor filter
        sigma_d: coefficient for the central areas, which have a bias towards attention
        sigma_c: coefficient for the warm colors, which have a bias towards attention

    Returns:
        torch.Tensor: Visual saliency map
    """
    x = x / float(data_range) * 255
    size = x.size()
    size_to_use = 256, 256
    x = interpolate(input=x, size=size_to_use, mode='bilinear', align_corners=False)
    x_lab = rgb2lab(x, data_range=255)
    lg = _log_gabor(size_to_use, omega_0, sigma_f).view(1, 1, *size_to_use)
    recommended_torch_version = _parse_version('1.8.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        x_fft = torch.fft.fft2(x_lab)
        x_ifft_real = torch.fft.ifft2(x_fft * lg).real
    else:
        x_fft = torch.rfft(x_lab, 2, onesided=False)
        x_ifft_real = torch.ifft(x_fft * lg.unsqueeze(-1), 2)[..., 0]
    s_f = x_ifft_real.pow(2).sum(dim=1, keepdim=True).sqrt()
    coordinates = torch.stack(get_meshgrid(size_to_use), dim=0)
    coordinates = coordinates * size_to_use[0] + 1
    s_d = torch.exp(-torch.sum(coordinates ** 2, dim=0) / sigma_d ** 2).view(1, 1, *size_to_use)
    eps = torch.finfo(x_lab.dtype).eps
    min_x = x_lab.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_x = x_lab.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    normalized = (x_lab - min_x) / (max_x - min_x + eps)
    norm = normalized[:, 1:].pow(2).sum(dim=1, keepdim=True)
    s_c = 1 - torch.exp(-norm / sigma_c ** 2)
    vs_m = s_f * s_d * s_c
    vs_m = interpolate(vs_m, size[-2:], mode='bilinear', align_corners=True)
    min_vs_m = vs_m.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_vs_m = vs_m.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    return (vs_m - min_vs_m) / (max_vs_m - min_vs_m + eps)


def vsi(x: torch.Tensor, y: torch.Tensor, reduction: str='mean', data_range: Union[int, float]=1.0, c1: float=1.27, c2: float=386.0, c3: float=130.0, alpha: float=0.4, beta: float=0.02, omega_0: float=0.021, sigma_f: float=1.34, sigma_d: float=145.0, sigma_c: float=0.001) ->torch.Tensor:
    """Compute Visual Saliency-induced Index for a batch of images.

    Both inputs are supposed to have RGB channels order in accordance with the original approach.
    Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
    channel 3 times.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    Returns:
        Index of similarity between two images. Usually in [0, 1] range.

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 23, no. 10, pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260

    Note:
        The original method supports only RGB image.
        See https://ieeexplore.ieee.org/document/6873260 for details.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        warnings.warn('The original VSI supports only RGB images. The input images were converted to RGB by copying the grey channel 3 times.')
    x = x * 255.0 / float(data_range)
    y = y * 255.0 / float(data_range)
    vs_x = sdsp(x, data_range=255, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    vs_y = sdsp(y, data_range=255, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    x_lmn = rgb2lmn(x)
    y_lmn = rgb2lmn(y)
    kernel_size = max(1, round(min(vs_x.size()[-2:]) / 256))
    padding = kernel_size // 2
    if padding:
        upper_pad = padding
        bottom_pad = (kernel_size - 1) // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        mode = 'replicate'
        vs_x = pad(vs_x, pad=pad_to_use, mode=mode)
        vs_y = pad(vs_y, pad=pad_to_use, mode=mode)
        x_lmn = pad(x_lmn, pad=pad_to_use, mode=mode)
        y_lmn = pad(y_lmn, pad=pad_to_use, mode=mode)
    vs_x = avg_pool2d(vs_x, kernel_size=kernel_size)
    vs_y = avg_pool2d(vs_y, kernel_size=kernel_size)
    x_lmn = avg_pool2d(x_lmn, kernel_size=kernel_size)
    y_lmn = avg_pool2d(y_lmn, kernel_size=kernel_size)
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(1, 2)])
    gm_x = gradient_map(x_lmn[:, :1], kernels)
    gm_y = gradient_map(y_lmn[:, :1], kernels)
    s_vs = similarity_map(vs_x, vs_y, c1)
    s_gm = similarity_map(gm_x, gm_y, c2)
    s_m = similarity_map(x_lmn[:, 1:2], y_lmn[:, 1:2], c3)
    s_n = similarity_map(x_lmn[:, 2:], y_lmn[:, 2:], c3)
    s_c = s_m * s_n
    s_c_complex = [s_c.abs(), torch.atan2(torch.zeros_like(s_c), s_c)]
    s_c_complex_pow = [s_c_complex[0] ** beta, s_c_complex[1] * beta]
    s_c_real_pow = s_c_complex_pow[0] * torch.cos(s_c_complex_pow[1])
    s = s_vs * s_gm.pow(alpha) * s_c_real_pow
    vs_max = torch.max(vs_x, vs_y)
    eps = torch.finfo(vs_max.dtype).eps
    output = s * vs_max
    output = ((output.sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)).squeeze(-1)
    return _reduce(output, reduction)


class VSILoss(_Loss):
    """Creates a criterion that measures Visual Saliency-induced Index error between
    each element in the input and target.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    Examples:

        >>> loss = VSILoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 23, no. 10, pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260
    """

    def __init__(self, reduction: str='mean', c1: float=1.27, c2: float=386.0, c3: float=130.0, alpha: float=0.4, beta: float=0.02, data_range: Union[int, float]=1.0, omega_0: float=0.021, sigma_f: float=1.34, sigma_d: float=145.0, sigma_c: float=0.001) ->None:
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.vsi = functools.partial(vsi, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta, omega_0=omega_0, sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c, data_range=data_range, reduction=reduction)

    def forward(self, x, y):
        """Computation of VSI as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of VSI loss to be minimized in [0, 1] range.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
            channel 3 times.
        """
        return 1.0 - self.vsi(x=x, y=y)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BRISQUELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (FID,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (KID,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (L2Pool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_photosynthesis_team_piq(_paritybench_base):
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

