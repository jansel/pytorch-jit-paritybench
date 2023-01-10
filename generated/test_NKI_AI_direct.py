import sys
_module = sys.modules[__name__]
del sys
direct = _module
checkpointer = _module
cli = _module
predict = _module
train = _module
upload = _module
utils = _module
common = _module
subsample = _module
subsample_config = _module
config = _module
defaults = _module
constants = _module
data = _module
bbox = _module
datasets = _module
datasets_config = _module
fake = _module
h5_data = _module
lr_scheduler = _module
mri_transforms = _module
samplers = _module
sens = _module
transforms = _module
engine = _module
environment = _module
exceptions = _module
functionals = _module
challenges = _module
grad = _module
nmae = _module
nmse = _module
psnr = _module
ssim = _module
inference = _module
launch = _module
nn = _module
cirim = _module
cirim = _module
cirim_engine = _module
conjgradnet = _module
conjgrad = _module
conjgradnet = _module
conjgradnet_engine = _module
conv = _module
conv = _module
crossdomain = _module
crossdomain = _module
multicoil = _module
didn = _module
didn = _module
get_nn_model_config = _module
iterdualnet = _module
iterdualnet = _module
iterdualnet_engine = _module
jointicnet = _module
jointicnet = _module
jointicnet_engine = _module
kikinet = _module
kikinet = _module
kikinet_engine = _module
lpd = _module
lpd = _module
lpd_engine = _module
mobilenet = _module
mobilenet = _module
mri_models = _module
multidomainnet = _module
multidomain = _module
multidomainnet = _module
multidomainnet_engine = _module
mwcnn = _module
mwcnn = _module
recurrent = _module
recurrent = _module
recurrentvarnet = _module
recurrentvarnet = _module
recurrentvarnet_engine = _module
resnet = _module
resnet = _module
rim = _module
rim = _module
rim_engine = _module
types = _module
unet = _module
unet_2d = _module
unet_engine = _module
varnet = _module
varnet = _module
varnet_engine = _module
varsplitnet = _module
varsplitnet = _module
varsplitnet_engine = _module
xpdnet = _module
xpdnet = _module
xpdnet_engine = _module
predict = _module
train = _module
types = _module
utils = _module
asserts = _module
bbox = _module
communication = _module
dataset = _module
events = _module
imports = _module
io = _module
logging = _module
models = _module
writers = _module
jupyter_notebook_config = _module
conf = _module
doi_role = _module
compute_masks = _module
predict_test = _module
predict_val = _module
compute_metrics = _module
plot_zoomed = _module
setup = _module
tests = _module
test_checkpointer = _module
test_cli = _module
test_utils = _module
test_train = _module
tests_common = _module
test_subsample = _module
tests_data = _module
test_datasets = _module
test_fake = _module
test_lr_scheduler = _module
test_mri_transforms = _module
test_samplers = _module
test_sens = _module
test_transforms = _module
tests_functionals = _module
test_gradloss = _module
test_nmae = _module
test_nmse = _module
test_psnr = _module
test_ssim = _module
tests_nn = _module
test_cirim = _module
test_cirim_engine = _module
test_conjgradnet = _module
test_conjgradnet_engine = _module
test_conv = _module
test_didn = _module
test_iterdualnet = _module
test_iterdualnet_engine = _module
test_jointicnet = _module
test_jointicnet_engine = _module
test_kikinet = _module
test_kikinet_engine = _module
test_lpd = _module
test_lpd_engine = _module
test_mri_models = _module
test_multidomainnet = _module
test_multidomainnet_engine = _module
test_mwcnn = _module
test_recurrent = _module
test_recurrentvarnet = _module
test_recurrentvarnet_engine = _module
test_resnet = _module
test_rim = _module
test_rim_engine = _module
test_unet_2d = _module
test_unet_engine = _module
test_varnet = _module
test_varnet_engine = _module
test_varsplitnet = _module
test_varsplitnet_engine = _module
test_xpdnet = _module
test_xpdnet_engine = _module
tests_utils = _module
test_imports = _module
test_io = _module
test_utils = _module
parse_metrics_log = _module

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


import logging


import re


import warnings


from typing import Dict


from typing import Mapping


from typing import Optional


from typing import Union


from typing import get_args


import torch


import torch.nn as nn


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


from abc import abstractmethod


from enum import Enum


from typing import Iterable


from typing import List


from typing import Tuple


import numpy as np


from typing import Any


from typing import Callable


from typing import Sequence


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


import math


import functools


import itertools


import random


from torch.utils.data.sampler import Sampler


import torch.fft


from numpy.typing import ArrayLike


from abc import ABC


from collections import namedtuple


from torch import nn


from torch.cuda.amp import GradScaler


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torchvision.utils import make_grid


from torch.utils import collect_env


import torch.nn.functional as F


from functools import partial


from typing import DefaultDict


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.cuda.amp import autocast


import time


from collections import defaultdict


from torch.nn import functional as F


from collections import OrderedDict


from typing import NewType


from torch import nn as nn


import abc


from typing import KeysView


import inspect


from typing import IO


from torch.utils.data import ConcatDataset


from sklearn.datasets import load_sample_image


class SobelGradLossType(str, Enum):
    l1 = 'l1'
    l2 = 'l2'


def get_sobel_kernel2d() ->torch.Tensor:
    """Returns the Sobel kernel matrices :math:`G_{x}` and :math:`G_{y}`:

    ..math::

        G_{x} = \\begin{matrix}
                    -1 & 0 & 1 \\\\
                    -2 & 0 & 2 \\\\
                    -1 & 0 & 1
                \\end{matrix}, \\quad
        G_{y} = \\begin{matrix}
                    -1 & -2 & -1 \\\\
                     0 & 0 & 0 \\\\
                     1 & 2 & 1
                \\end{matrix}.
    """
    kernel_x: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def normalize_kernel(input: torch.Tensor) ->torch.Tensor:
    """Normalize both derivative kernel.

    Parameters
    ----------
    input: torch.Tensor

    Returns
    -------
    torch.Tensor
        Normalized kernel.
    """
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / norm.unsqueeze(-1).unsqueeze(-1)


def spatial_gradient(input: torch.Tensor, normalized: bool=True) ->Tuple[torch.Tensor, torch.Tensor]:
    """Computes the first order image derivatives in :math:`x` and :math:`y` directions using a Sobel operator.

    Parameters
    ----------
    input: torch.Tensor
        Input image tensor with shape :math:`(B, C, H, W)`.
    normalized: bool
        Whether the output is normalized. Default: True.

    Returns
    -------
    grad_x, grad_y: (torch.Tensor, torch.Tensor)
        The derivatives in :math:`x` and :math:`y:` directions of the input each of same shape as input.
    """
    if not len(input.shape) == 4:
        raise ValueError(f'Invalid input shape, we expect BxCxHxW. Got: {input.shape}')
    kernel: torch.Tensor = get_sobel_kernel2d()
    if normalized:
        kernel = normalize_kernel(kernel)
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]
    grad = F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, 2, h, w)
    grad_x, grad_y = grad[:, :, 0], grad[:, :, 1]
    return grad_x, grad_y


class SobelGradLoss(nn.Module):
    """Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_k^k + ||u_y - v_y||_k^k

    where :math:`u` and :math:`v` denote the input and target images and :math:`k` is 1 if `type_loss`="l1" or 2 if
    `type_loss`="l2". The gradients w.r.t. to :math:`x` and :math:`y` directions are computed using the Sobel operators.
    """

    def __init__(self, type_loss: SobelGradLossType, reduction: str='mean', normalized_grad: bool=True):
        """Inits :class:`SobelGradLoss`.

        Parameters
        ----------
        type_loss: SobelGradLossType
            Type of loss to be used. Can be "l1" or "l2".
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__()
        self.reduction = reduction
        if type_loss == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        else:
            self.loss = nn.MSELoss(reduction=reduction)
        self.normalized_grad = normalized_grad

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`SobelGradLoss`.

        Parameters
        ----------
        input: torch.Tensor
            Input tensor.
        target: torch.Tensor
            Target tensor.

        Returns
        -------
        loss: torch.Tensor
            Sum of the l1-loss between the gradient of input and target.
        """
        input_grad_x, input_grad_y = spatial_gradient(input, self.normalized_grad)
        target_grad_x, target_grad_y = spatial_gradient(target, self.normalized_grad)
        return self.loss(input_grad_x, target_grad_x) + self.loss(input_grad_y, target_grad_y)


class SobelGradL1Loss(SobelGradLoss):
    """Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_1 + ||u_y - v_y||_1

    where :math:`u` and :math:`v` denote the input and target images. The gradients w.r.t. to :math:`x` and :math:`y`
    directions are computed using the Sobel operators.
    """

    def __init__(self, reduction: str='mean', normalized_grad: bool=True):
        """Inits :class:`SobelGradL1Loss`.

        Parameters
        ----------
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__(SobelGradLossType.l1, reduction, normalized_grad)


class SobelGradL2Loss(SobelGradLoss):
    """Computes the sum of the l1-loss between the gradient of input and target:

    It returns

    .. math ::

        ||u_x - v_x ||_2^2 + ||u_y - v_y||_2^2

    where :math:`u` and :math:`v` denote the input and target images. The gradients w.r.t. to :math:`x` and :math:`y`
    directions are computed using the Sobel operators.
    """

    def __init__(self, reduction: str='mean', normalized_grad: bool=True):
        """Inits :class:`SobelGradL2Loss`.

        Parameters
        ----------
        reduction: str
            Loss reduction. Can be 'mean' or "sum". Default: "mean".
        normalized_grad: bool
            Whether the computed gradients are normalized. Default: True.
        """
        super().__init__(SobelGradLossType.l2, reduction, normalized_grad)


class NMAELoss(nn.Module):
    """Computes the Normalized Mean Absolute Error (NMAE), i.e.:

    .. math::
        rac{||u - v||_1}{||u||_1},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction='mean'):
        """Inits :class:`NMAE`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NMAE`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return self.mae_loss(input, target) / self.mae_loss(torch.zeros_like(target, dtype=target.dtype, device=target.device), target)


class NMSELoss(nn.Module):
    """Computes the Normalized Mean Squared Error (NMSE), i.e.:

    .. math::
        rac{||u - v||_2^2}{||u||_2^2},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction='mean'):
        """Inits :class:`NMSE`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NMSE`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return self.mse_loss(input, target) / self.mse_loss(torch.zeros_like(target, dtype=target.dtype, device=target.device), target)


class NRMSELoss(nn.Module):
    """Computes the Normalized Root Mean Squared Error (NRMSE), i.e.:

    .. math::
        rac{||u - v||_2}{||u||_2},

    where :math:`u` and :math:`v` denote the target and the input.
    """

    def __init__(self, reduction='mean'):
        """Inits :class:`NRMSE`

        Parameters
        ----------
        reduction: str
             Specifies the reduction to apply to the output. Can be "none", "mean" or "sum".
             Note that "mean" or "sum" will yield the same output. Default: "mean".
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward method of :class:`NRMSE`.

        Parameters
        ----------
        input: torch.Tensor
            Tensor of shape (*), where * means any number of dimensions.
        target: torch.Tensor
            Tensor of same shape as the input.
        """
        return torch.sqrt(self.mse_loss(input, target) / self.mse_loss(torch.zeros_like(target, dtype=target.dtype, device=target.device), target))


def batch_psnr(input_data, target_data, reduction='mean'):
    """This function is a torch implementation of skimage.metrics.compare_psnr.

    Parameters
    ----------
    input_data: torch.Tensor
    target_data: torch.Tensor
    reduction: str

    Returns
    -------
    torch.Tensor
    """
    batch_size = target_data.size(0)
    input_view = input_data.view(batch_size, -1)
    target_view = target_data.view(batch_size, -1)
    maximum_value = torch.max(input_view, 1)[0]
    mean_square_error = torch.mean((input_view - target_view) ** 2, 1)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)
    if reduction == 'mean':
        return psnrs.mean()
    if reduction == 'sum':
        return psnrs.sum()
    if reduction == 'none':
        return psnrs
    raise ValueError(f'Reduction is either `mean`, `sum` or `none`. Got {reduction}.')


class PSNRLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, input_data, target_data):
        return batch_psnr(input_data, target_data, reduction=self.reduction)


class SSIMLoss(nn.Module):
    """SSIM loss module.

    From: https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Parameters
        ----------
        win_size: int
            Window size for SSIM calculation. Default: 7.
        k1: float
            k1 parameter for SSIM calculation. Default: 0.1.
        k2: float
            k2 parameter for SSIM calculation. Default: 0.03.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = 2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2
        D = B1 * B2
        S = A1 * A2 / D
        return 1 - S.mean()


class ConvRNNStack(nn.Module):
    """
    A stack of convolutional RNNs.

    Takes as input a sequence of recurrent and convolutional layers.
    """

    def __init__(self, convs, recurrent):
        """
        Parameters
        ----------
        convs: List[torch.nn.Module]
            List of convolutional layers.
        recurrent: torch.nn.Module
            Recurrent layer.
        """
        super().__init__()
        self.convs = convs
        self.recurrent = recurrent

    def forward(self, _input, hidden):
        """
        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size)
        hidden: torch.Tensor
            Hidden state. (num_layers * num_directions, batch_size, hidden_size)

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, hidden_size)
        """
        return self.recurrent(self.convs(_input), hidden)


class ConvNonlinear(nn.Module):
    """A convolutional layer with nonlinearity."""

    def __init__(self, input_size, features, kernel_size, dilation, bias):
        """
        Initializes the convolutional layer.

        Parameters
        ----------
        input_size: int
            Size of the input.
        features: int
            Number of features.
        kernel_size: int
            Size of the kernel.
        dilation: int
            Dilation of the kernel.
        bias: bool
            Whether to use bias.
        """
        super().__init__()
        self.padding = torch.nn.ReplicationPad2d(torch.div(dilation * (kernel_size - 1), 2, rounding_mode='trunc').item())
        self.conv_layer = nn.Conv2d(in_channels=input_size, out_channels=features, kernel_size=kernel_size, padding=0, dilation=dilation, bias=bias)
        self.nonlinear = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the convolutional layer."""
        torch.nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='relu')
        if self.conv_layer.bias is not None:
            nn.init.zeros_(self.conv_layer.bias)

    def forward(self, _input):
        """
        Forward pass of the convolutional layer.

        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size)

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, features)
        """
        return self.nonlinear(self.conv_layer(self.padding(_input)))


class IndRNNCell(nn.Module):
    """
    Base class for Independently RNN cells as presented in [1]_.

    References
    ----------

    .. [1] Li, S. et al. (2018) ‘Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN’,
        Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, (1),
        pp. 5457–5466. doi: 10.1109/CVPR.2018.00572.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int=1, dilation: int=1, bias: bool=True):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels
        hidden_channels : int
            Number of hidden channels
        kernel_size : int
            Kernel size. Default: 1.
        dilation : int
            Dilation size. Default: 1.
        bias : bool
            Whether to use bias. Default: True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.ih = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=torch.div(dilation * (kernel_size - 1), 2, rounding_mode='trunc').item(), dilation=dilation, bias=bias)
        self.hh = nn.Parameter(nn.init.normal_(torch.empty(1, hidden_channels, 1, 1), std=1.0 / (hidden_channels * (1 + kernel_size ** 2))))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        nn.init.normal_(self.ih.weight, std=1.0 / (self.hidden_channels * (1 + self.kernel_size ** 2)))
        if self.bias is True:
            nn.init.zeros_(self.ih.bias)

    @staticmethod
    def orthotogonalize_weights(weights, chunks=1):
        """
        Orthogonalize weights.

        Parameters
        ----------
        weights: torch.Tensor
            The weights to orthogonalize.
        chunks: int
            Number of chunks. Default: 1.

        Returns
        -------
        weights: torch.Tensor
            The orthogonalized weights.
        """
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    def forward(self, _input, hx):
        """
        Forward pass of the cell.

        Parameters
        ----------
        _input: torch.Tensor
            Input tensor. (batch_size, seq_len, input_size), tensor containing input features.
        hx: torch.Tensor
            Hidden state. (batch_size, hidden_channels, 1, 1), tensor containing hidden state features.

        Returns
        -------
        output: torch.Tensor
            Output tensor. (batch_size, seq_len, hidden_channels), tensor containing the next hidden state.
        """
        return nn.ReLU()(self.ih(_input) + self.hh * hx)


class MRILogLikelihood(nn.Module):
    """Defines the MRI loglikelihood assuming one noise vector for the complex images for all coils:

    .. math::
         \\frac{1}{\\sigma^2} \\sum_{i}^{N_c} {S}_i^{\\text{H}} \\mathcal{F}^{-1} P^{*} (P \\mathcal{F} S_i x_{\\tau} - y_{\\tau})

    for each time step :math:`\\tau`.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable):
        """Inits :class:`MRILogLikelihood`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._spatial_dims = 2, 3

    def forward(self, input_image, masked_kspace, sensitivity_map, sampling_mask, loglikelihood_scaling=None) ->torch.Tensor:
        """Performs forward pass of :class:`MRILogLikelihood`.

        Parameters
        ----------
        input_image: torch.Tensor
            Initial or previous iteration of image with complex first
            of shape (N, complex, height, width).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex).
        sensitivity_map: torch.Tensor
            Sensitivity Map of shape (N, coil, height, width, complex).
        sampling_mask: torch.Tensor
        loglikelihood_scaling: torch.Tensor
            Multiplier for loglikelihood, for instance for the k-space noise, of shape (1,).

        Returns
        -------
        out: torch.Tensor
            The MRI Loglikelihood.
        """
        input_image = input_image.permute(0, 2, 3, 1)
        if loglikelihood_scaling is not None:
            loglikelihood_scaling = loglikelihood_scaling
        else:
            loglikelihood_scaling = torch.tensor([1.0], dtype=masked_kspace.dtype)
        loglikelihood_scaling = loglikelihood_scaling.reshape(-1, *torch.ones(len(sensitivity_map.shape) - 1).int())
        mul = loglikelihood_scaling * T.complex_multiplication(sensitivity_map, input_image.unsqueeze(1))
        mr_forward = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), self.forward_operator(mul, dim=self._spatial_dims))
        error = mr_forward - loglikelihood_scaling * torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), masked_kspace)
        mr_backward = self.backward_operator(error, dim=self._spatial_dims)
        if sensitivity_map is not None:
            out = T.complex_multiplication(T.conjugate(sensitivity_map), mr_backward).sum(self._coil_dim)
        else:
            out = mr_backward.sum(self._coil_dim)
        out = out.permute(0, 3, 1, 2)
        return out


COMPLEX_DIM = 2


def is_complex_data(data: torch.Tensor, complex_axis: int=-1) ->bool:
    """Returns True if data is a complex tensor at a specified dimension, i.e. complex_axis of data is of size 2,
    corresponding to real and imaginary channels..

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the check will be done. Default: -1 (last).

    Returns
    -------
    bool
        True if data is a complex tensor.
    """
    return data.size(complex_axis) == COMPLEX_DIM


def assert_complex(data: torch.Tensor, complex_axis: int=-1, complex_last: Optional[bool]=None) ->None:
    """Assert if a tensor is complex (has complex dimension of size 2 corresponding to real and imaginary channels).

    Parameters
    ----------
    data: torch.Tensor
    complex_axis: int
        Complex dimension along which the assertion will be done. Default: -1 (last).
    complex_last: Optional[bool]
        If true, will override complex_axis with -1 (last). Default: None.
    """
    if complex_last:
        complex_axis = -1
    assert is_complex_data(data, complex_axis), f'Complex dimension assumed to be 2 (complex valued), but not found in shape {data.shape}.'


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) ->torch.Tensor:
    """Multiplies two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input data
    other_tensor: torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)
    complex_index = -1
    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]
    multiplication = torch.cat([real_part.unsqueeze(dim=complex_index), imaginary_part.unsqueeze(dim=complex_index)], dim=complex_index)
    return multiplication


def expand_operator(data: torch.Tensor, sensitivity_map: torch.Tensor, dim: int=0) ->torch.Tensor:
    """
    Given a reconstructed image :math:`x` and coil sensitivity maps :math:`\\{S_i\\}_{i=1}^{N_c}`, it returns

        .. math::
            E(x) = (S_1 \\times x, .., S_{N_c} \\times x) = (x_1, .., x_{N_c}).

    Adapted from [1]_.

    Parameters
    ----------
    data: torch.Tensor
        Image data. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.

    Returns
    -------
    torch.Tensor:
        Zero-filled reconstructions from each coil.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.

    """
    assert_complex(data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)
    return complex_multiplication(sensitivity_map, data.unsqueeze(dim))


def conjugate(data: torch.Tensor) ->torch.Tensor:
    """Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part (last axis
    has dimension 2).

    Parameters
    ----------
    data: torch.Tensor

    Returns
    -------
    conjugate_tensor: torch.Tensor
    """
    assert_complex(data, complex_last=True)
    data = data.clone()
    data[..., 1] = data[..., 1] * -1.0
    return data


def reduce_operator(coil_data: torch.Tensor, sensitivity_map: torch.Tensor, dim: int=0) ->torch.Tensor:
    """
    Given zero-filled reconstructions from multiple coils :math:`\\{x_i\\}_{i=1}^{N_c}` and
    coil sensitivity maps :math:`\\{S_i\\}_{i=1}^{N_c}` it returns:

        .. math::
            R(x_{1}, .., x_{N_c}, S_1, .., S_{N_c}) = \\sum_{i=1}^{N_c} {S_i}^{*} \\times x_i.

    Adapted from [1]_.

    Parameters
    ----------
    coil_data: torch.Tensor
        Zero-filled reconstructions from coils. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.

    Returns
    -------
    torch.Tensor:
        Combined individual coil images.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.

    """
    assert_complex(coil_data, complex_last=True)
    assert_complex(sensitivity_map, complex_last=True)
    return complex_multiplication(conjugate(sensitivity_map), coil_data).sum(dim)


class RIMBlock(nn.Module):
    """
    Recurrent Inference Machines block as presented in [1]_.

    References
    ----------

    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently
        Recurrent Inference Machines for fast and robust accelerated MRI reconstruction’.
        Available at: https://arxiv.org/abs/2111.15498v1
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, depth: int=2, in_channels: int=2, hidden_channels: int=64, time_steps: int=4, no_parameter_sharing: bool=False):
        """
        Parameters
        ----------
        forward_operator: Callable
            Forward Fourier Transform.
        backward_operator: Callable
            Backward Fourier Transform.
        depth: int
            Number of layers in the RIM block.
        in_channels: int,
            Input channel number. Default is 2 for complex data.
        hidden_channels: int,
            Hidden channels. Default: 64.
        time_steps: int,
            Number of layers of :math:`n_l` recurrent unit. Default: 4.
        data_consistency: bool,
            If False, the DC component is removed from the input.
        """
        super().__init__()
        self.input_size = in_channels * 2
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.time_steps = time_steps
        self.layers = nn.ModuleList()
        for i in range(depth):
            conv_layer = None
            if i != depth:
                conv_layer = ConvNonlinear(self.input_size, hidden_channels, kernel_size=5 if i == 0 else 3, dilation=2 if i == 1 else 1, bias=True)
                self.input_size = hidden_channels
            if i != depth:
                rnn_layer = IndRNNCell(self.input_size, hidden_channels, kernel_size=1, dilation=1, bias=True)
                self.input_size = hidden_channels
                self.layers.append(ConvRNNStack(conv_layer, rnn_layer))
        self.final_layer = torch.nn.Sequential(ConvNonlinear(self.input_size, 2, kernel_size=3, dilation=1, bias=False))
        self.no_parameter_sharing = no_parameter_sharing
        if not self.no_parameter_sharing:
            self.dc_weight = nn.Parameter(torch.ones(1))
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.grad_likelihood = MRILogLikelihood(self.forward_operator, self.backward_operator)

    def forward(self, current_prediction: torch.Tensor, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor, hidden_state: Union[None, torch.Tensor], parameter_sharing: bool=False, coil_dim: int=1, spatial_dims: Tuple[int, int]=(2, 3)) ->Union[Tuple[List, None], Tuple[List, Union[List, torch.Tensor]]]:
        """
        Parameters
        ----------
        current_prediction : torch.Tensor
            Current k-space.
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            IndRNN hidden state of shape (N, hidden_channels, height, width, num_layers) if not None. Optional.
        parameter_sharing: bool
            If True, the weights of the convolutional layers are shared between the forward and backward pass.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).

        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            Next hidden state of shape (N, hidden_channels, height, width, num_layers) if parameter_sharing else None.
        """
        if hidden_state is None:
            hidden_state = [masked_kspace.new_zeros((masked_kspace.size(0), self.hidden_channels, *masked_kspace.size()[2:-1])) for _ in range(self.depth)]
        if isinstance(current_prediction, list):
            current_prediction = current_prediction[-1].detach()
        intermediate_image = reduce_operator(self.backward_operator(current_prediction, dim=spatial_dims), sensitivity_map, coil_dim) if not parameter_sharing else current_prediction
        intermediate_image = intermediate_image.permute(0, 3, 1, 2)
        intermediate_images = []
        for _ in range(self.time_steps):
            llg = self.grad_likelihood(intermediate_image, masked_kspace, sensitivity_map, sampling_mask)
            llg_eta = torch.cat([llg, intermediate_image], dim=coil_dim).contiguous()
            for hs, convrnn in enumerate(self.layers):
                hidden_state[hs] = convrnn(llg_eta, hidden_state[hs])
                llg_eta = hidden_state[hs]
            llg_eta = self.final_layer(llg_eta)
            llg_eta = (intermediate_image + llg_eta).permute(0, 2, 3, 1)
            intermediate_images.append(llg_eta)
        if self.no_parameter_sharing:
            return intermediate_images, None
        soft_dc = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), current_prediction - masked_kspace)
        current_kspace = [(masked_kspace - soft_dc - self.forward_operator(expand_operator(x, sensitivity_map, dim=coil_dim), dim=spatial_dims)) for x in intermediate_images]
        return current_kspace, hidden_state


class CIRIM(nn.Module):
    """
    Cascades of Independently Recurrent Inference Machines implementation as presented in [1]_.

    References
    ----------

    .. [1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent
        Inference Machines for fast and robust accelerated MRI reconstruction’.
        Available at: https://arxiv.org/abs/2111.15498v1
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, depth: int=2, in_channels: int=2, time_steps: int=8, recurrent_hidden_channels: int=64, num_cascades: int=8, no_parameter_sharing: bool=True, **kwargs):
        """

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        depth: int
            Number of layers.
        time_steps : int
            Number of iterations :math:`T`.
        in_channels : int
            Input channel number. Default is 2 for complex data.
        recurrent_hidden_channels : int
            Hidden channels number for the recurrent unit of the CIRIM Blocks. Default: 64.
        recurrent_num_layers : int
            Number of layers for the recurrent unit of the CIRIM Block (:math:`n_l`). Default: 4.
        no_parameter_sharing : bool
            If False, the same CIRIM Block is used for all time_steps. Default: True.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in ['model_name']:
                raise ValueError(f'{type(self).__name__} got key `{extra_key}` which is not supported.')
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList([RIMBlock(forward_operator=self.forward_operator, backward_operator=self.backward_operator, depth=depth, in_channels=in_channels, hidden_channels=recurrent_hidden_channels, time_steps=time_steps, no_parameter_sharing=self.no_parameter_sharing) for _ in range(num_cascades)])
        self._coil_dim = 1
        self._spatial_dims = 2, 3
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->List[List[Union[torch.Tensor, Any]]]:
        """
        Parameters
        ----------
        masked_kspace : torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask : torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map : torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).

        Returns
        -------
        imspace_prediction: torch.Tensor
            imspace prediction.
        """
        previous_state: Optional[torch.Tensor] = None
        current_prediction = masked_kspace.clone()
        cascades_etas = []
        for i, cascade in enumerate(self.block_list):
            current_prediction, previous_state = cascade(current_prediction, masked_kspace, sampling_mask, sensitivity_map, previous_state, parameter_sharing=False if i == 0 else self.no_parameter_sharing, coil_dim=self._coil_dim, spatial_dims=self._spatial_dims)
            if self.no_parameter_sharing:
                _current_prediction = [torch.abs(torch.view_as_complex(x)) for x in current_prediction]
            else:
                _current_prediction = [torch.abs(torch.view_as_complex(reduce_operator(self.backward_operator(x, dim=self._spatial_dims), sensitivity_map, self._coil_dim))) for x in current_prediction]
            cascades_etas.append(_current_prediction)
        yield cascades_etas


def safe_divide(input_tensor: torch.Tensor, other_tensor: torch.Tensor) ->torch.Tensor:
    """Divide input_tensor and other_tensor safely, set the output to zero where the divisor b is zero.

    Parameters
    ----------
    input_tensor: torch.Tensor
    other_tensor: torch.Tensor

    Returns
    -------
    torch.Tensor: the division.
    """
    data = torch.where(other_tensor == 0, torch.tensor([0.0], dtype=input_tensor.dtype), input_tensor / other_tensor)
    return data


def complex_division(input_tensor: torch.Tensor, other_tensor: torch.Tensor) ->torch.Tensor:
    """Divides two complex-valued tensors. Assumes input tensors are complex (last axis has dimension 2).

    Parameters
    ----------
    input_tensor: torch.Tensor
        Input data
    other_tensor: torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert_complex(input_tensor, complex_last=True)
    assert_complex(other_tensor, complex_last=True)
    complex_index = -1
    denominator = other_tensor[..., 0] ** 2 + other_tensor[..., 1] ** 2
    real_part = safe_divide(input_tensor[..., 0] * other_tensor[..., 0] + input_tensor[..., 1] * other_tensor[..., 1], denominator)
    imaginary_part = safe_divide(input_tensor[..., 1] * other_tensor[..., 0] - input_tensor[..., 0] * other_tensor[..., 1], denominator)
    division = torch.cat([real_part.unsqueeze(dim=complex_index), imaginary_part.unsqueeze(dim=complex_index)], dim=complex_index)
    return division


def complex_dot_product(a: torch.Tensor, b: torch.Tensor, dim: List[int]) ->torch.Tensor:
    """Computes the dot product of the complex tensors :math:`a` and :math:`b`: :math:`a^{*}b = <a, b>`.

    Parameters
    ----------
    a : torch.Tensor
        Input :math:`a`.
    b : torch.Tensor
        Input :math:`b`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    complex_dot_product : torch.Tensor
        Dot product of :math:`a` and :math:`b`.
    """
    return complex_multiplication(conjugate(a), b).sum(dim)


def _BAN(rk_new: torch.Tensor, rk_old: torch.Tensor, dim: List[int]) ->torch.Tensor:
    """Bamigbola-Ali-Nwaeze (BAN) update method for :math:`b_k`:

    .. math ::

        b_k = \\frac{ r_{k+1}^{*} y_k }{ r_{k}^{*} y_k }, y_k = r_{k+1} - r_k.

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        BAN computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, yk, dim), complex_dot_product(rk_old, yk, dim))


def _DY(rk_new: torch.Tensor, rk_old: torch.Tensor, pk: torch.Tensor, dim: List[int]) ->torch.Tensor:
    """Dai-Yuan (DY) update method for :math:`b_k`:

    .. math ::

        b_k = \\frac{ ||r_{k+1}||_2^2 }{ p_{k}^{*} y_k } , y_k = r_{k+1} - r_k.

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    pk : torch.Tensor
        Input fot :math:`p_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        DY computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, rk_new, dim), complex_dot_product(pk, yk, dim))


def _PRP(rk_new: torch.Tensor, rk_old: torch.Tensor, dim: List[int]) ->torch.Tensor:
    """Polak-Ribiere-Polyak (PRP) update method for :math:`b_k`:

    .. math ::

        b_k = \\frac{ r_{k+1}^{*}(r_{k+1} - r_k) }{ ||r_k||_2^2 } ,

    Parameters
    ----------
    rk_new : torch.Tensor
        Input for :math:`r_{k+1}`.
    rk_old : torch.Tensor
        Input for :math:`r_k`.
    dim : List[int]
        Dimensions which will be suppressed. Useful when inputs are batched.

    Returns
    -------
    bk : torch.Tensor
        PRP computation for :math:`b_k`.
    """
    yk = rk_new - rk_old
    return complex_division(complex_dot_product(rk_new, yk, dim), complex_dot_product(rk_old, rk_old, dim))


COMPLEX_SIZE = 2


class Conv2d(nn.Module):
    """Implementation of a simple cascade of 2D convolutions.

    If `batchnorm` is set to True, batch normalization layer is applied after each convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_convs: int=3, activation: nn.Module=nn.PReLU(), batchnorm: bool=False):
        """Inits :class:`Conv2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels.
        n_convs: int
            Number of convolutional layers.
        activation: nn.Module
            Activation function.
        batchnorm: bool
            If True a batch normalization layer is applied after every convolution.
        """
        super().__init__()
        conv: List[nn.Module] = []
        for idx in range(n_convs):
            conv.append(nn.Conv2d(in_channels if idx == 0 else hidden_channels, hidden_channels if idx != n_convs - 1 else out_channels, kernel_size=3, padding=1))
            if batchnorm:
                conv.append(nn.BatchNorm2d(hidden_channels if idx != n_convs - 1 else out_channels, eps=0.0001))
            if idx != n_convs - 1:
                conv.append(activation)
        self.conv = nn.Sequential(*conv)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Performs the forward pass of :class:`Conv2d`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        out: torch.Tensor
            Convoluted output.
        """
        out = self.conv(x)
        return out


class Subpixel(nn.Module):
    """Subpixel convolution layer for up-scaling of low resolution features at super-resolution as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int, kernel_size: Union[int, Tuple[int, int]], padding: int=0):
        """Inits :class:`Subpixel`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        upscale_factor: int
            Subpixel upscale factor.
        kernel_size: int or (int, int)
            Convolution kernel size.
        padding: int
            Padding size. Default: 0.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, kernel_size=kernel_size, padding=padding)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computes :class:`Subpixel` convolution on input torch.Tensor ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        return self.pixelshuffle(self.conv(x))


class DUB(nn.Module):
    """Down-up block (DUB) for :class:`DIDN` model as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`DUB`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_1 = nn.Sequential(*([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()] * 2))
        self.down1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Sequential(*[nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1), nn.PReLU()])
        self.down2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Sequential(*[nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=3, padding=1), nn.PReLU()])
        self.up1 = nn.Sequential(*[Subpixel(in_channels * 4, in_channels * 2, 2, 1, 0)])
        self.conv_agg_1 = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)
        self.conv2_2 = nn.Sequential(*[nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1), nn.PReLU()])
        self.up2 = nn.Sequential(*[Subpixel(in_channels * 2, in_channels, 2, 1, 0)])
        self.conv_agg_2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv1_2 = nn.Sequential(*([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()] * 2))
        self.conv_out = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.PReLU()])

    @staticmethod
    def pad(x: torch.Tensor) ->torch.Tensor:
        """Pads input to height and width dimensions if odd.

        Parameters
        ----------
        x: torch.Tensor
            Input to pad.

        Returns
        -------
        x: torch.Tensor
            Padded tensor.
        """
        padding = [0, 0, 0, 0]
        if x.shape[-2] % 2 != 0:
            padding[3] = 1
        if x.shape[-1] % 2 != 0:
            padding[1] = 1
        if sum(padding) != 0:
            x = F.pad(x, padding, 'reflect')
        return x

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) ->torch.Tensor:
        """Crops ``x`` to specified shape.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (\\*, H, W).
        shape: Tuple(int, int)
            Crop shape corresponding to H, W.

        Returns
        -------
        cropped_output: torch.Tensor
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, :shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, :shape[1]]
        return x

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        out: torch.Tensor
            DUB output.
        """
        x1 = self.pad(x.clone())
        x1 = x1 + self.conv1_1(x1)
        x2 = self.down1(x1)
        x2 = x2 + self.conv2_1(x2)
        out = self.down2(x2)
        out = out + self.conv3_1(out)
        out = self.up1(out)
        out = torch.cat([x2, self.crop_to_shape(out, (x2.shape[-2], x2.shape[-1]))], dim=1)
        out = self.conv_agg_1(out)
        out = out + self.conv2_2(out)
        out = self.up2(out)
        out = torch.cat([x1, self.crop_to_shape(out, (x1.shape[-2], x1.shape[-1]))], dim=1)
        out = self.conv_agg_2(out)
        out = out + self.conv1_2(out)
        out = x + self.crop_to_shape(self.conv_out(out), (x.shape[-2], x.shape[-1]))
        return out


class ReconBlock(nn.Module):
    """Reconstruction Block of :class:`DIDN` model as implemented in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(self, in_channels: int, num_convs: int):
        """Inits :class:`ReconBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        num_convs: int
            Number of convolution blocks.
        """
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1), nn.PReLU()]) for _ in range(num_convs - 1)])
        self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))
        self.num_convs = num_convs

    def forward(self, input_data: torch.Tensor) ->torch.Tensor:
        """Computes num_convs convolutions followed by PReLU activation on `input_data`.

        Parameters
        ----------
        input_data: torch.Tensor
            Input tensor.
        """
        output = input_data.clone()
        for idx in range(self.num_convs):
            output = self.convs[idx](output)
        return input_data + output


class DIDN(nn.Module):
    """Deep Iterative Down-up convolutional Neural network (DIDN) implementation as in [1]_.

    References
    ----------

    .. [1] Yu, Songhyun, et al. “Deep Iterative Down-Up CNN for Image Denoising.” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, pp. 2095–103. IEEE Xplore, https://doi.org/10.1109/CVPRW.2019.00262.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int=128, num_dubs: int=6, num_convs_recon: int=9, skip_connection: bool=False):
        """Inits :class:`DIDN`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        hidden_channels: int
            Number of hidden channels. First convolution out_channels. Default: 128.
        num_dubs: int
            Number of DUB networks. Default: 6.
        num_convs_recon: int
            Number of ReconBlock convolutions. Default: 9.
        skip_connection: bool
            Use skip connection. Default: False.
        """
        super().__init__()
        self.conv_in = nn.Sequential(*[nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1), nn.PReLU()])
        self.down = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1)
        self.dubs = nn.ModuleList([DUB(in_channels=hidden_channels, out_channels=hidden_channels) for _ in range(num_dubs)])
        self.recon_block = ReconBlock(in_channels=hidden_channels, num_convs=num_convs_recon)
        self.recon_agg = nn.Conv2d(in_channels=hidden_channels * num_dubs, out_channels=hidden_channels, kernel_size=1)
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1), nn.PReLU()])
        self.up2 = Subpixel(hidden_channels, hidden_channels, 2, 1)
        self.conv_out = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.num_dubs = num_dubs
        self.skip_connection = in_channels == out_channels and skip_connection

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: Tuple[int, int]) ->torch.Tensor:
        """Crops ``x`` to specified shape.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (\\*, H, W).
        shape: Tuple(int, int)
            Crop shape corresponding to H, W.

        Returns
        -------
        cropped_output: torch.Tensor
            Cropped tensor.
        """
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, :shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, :shape[1]]
        return x

    def forward(self, x: torch.Tensor, channel_dim: int=1) ->torch.Tensor:
        """Takes as input a torch.Tensor `x` and computes DIDN(x).

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        channel_dim: int
            Channel dimension. Default: 1.

        Returns
        -------
        out: torch.Tensor
            DIDN output tensor.
        """
        out = self.conv_in(x)
        out = self.down(out)
        dub_outs = []
        for dub in self.dubs:
            out = dub(out)
            dub_outs.append(out)
        out = [self.recon_block(dub_out) for dub_out in dub_outs]
        out = self.recon_agg(torch.cat(out, dim=channel_dim))
        out = self.conv(out)
        out = self.up2(out)
        out = self.conv_out(out)
        out = self.crop_to_shape(out, (x.shape[-2], x.shape[-1]))
        if self.skip_connection:
            out = x + out
        return out


class ConvBlock(nn.Module):
    """U-Net convolutional block.

    It consists of two convolution layers each followed by instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits ConvBlock.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(dropout_probability), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(dropout_probability))

    def forward(self, input_data: torch.Tensor) ->torch.Tensor:
        """Performs the forward pass of :class:`ConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of :class:`ConvBlock`."""
        return f'ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, dropout_probability={self.dropout_probability})'


class TransposeConvBlock(nn.Module):
    """U-Net Transpose Convolutional Block.

    It consists of one convolution transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Inits :class:`TransposeConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, input_data: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`TransposeConvBlock`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        """Representation of "class:`TransposeConvBlock`."""
        return f'ConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})'


class UnetModel2d(nn.Module):
    """PyTorch implementation of a U-Net model based on [1]_.

    References
    ----------

    .. [1] Ronneberger, Olaf, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab et al., Springer International Publishing, 2015, pp. 234–41. Springer Link, https://doi.org/10.1007/978-3-319-24574-4_28.
    """

    def __init__(self, in_channels: int, out_channels: int, num_filters: int, num_pool_layers: int, dropout_probability: float):
        """Inits :class:`UnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, dropout_probability)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, dropout_probability)]
            ch //= 2
        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [nn.Sequential(ConvBlock(ch * 2, ch, dropout_probability), nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1))]

    def forward(self, input_data: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`UnetModel2d`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) != 0:
                output = F.pad(output, padding, 'reflect')
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class NormUnetModel2d(nn.Module):
    """Implementation of a Normalized U-Net model."""

    def __init__(self, in_channels: int, out_channels: int, num_filters: int, num_pool_layers: int, dropout_probability: float, norm_groups: int=2):
        """Inits :class:`NormUnetModel2d`.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()
        self.unet2d = UnetModel2d(in_channels=in_channels, out_channels=out_channels, num_filters=num_filters, num_pool_layers=num_pool_layers, dropout_probability=dropout_probability)
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, groups: int) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)
        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)
        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, groups: int) ->torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    @staticmethod
    def pad(input_data: torch.Tensor) ->Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = input_data.shape
        w_mult = (w - 1 | 15) + 1
        h_mult = (h - 1 | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        output = F.pad(input_data, w_pad + h_pad)
        return output, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(input_data: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) ->torch.Tensor:
        return input_data[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, input_data: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`NormUnetModel2d`.

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        output, mean, std = self.norm(input_data, self.norm_groups)
        output, pad_sizes = self.pad(output)
        output = self.unet2d(output)
        output = self.unpad(output, *pad_sizes)
        output = self.unnorm(output, mean, std, self.norm_groups)
        return output


class ResNetBlock(nn.Module):
    """Main block of :class:`ResNet`.

    Consisted of a convolutional layer followed by a relu activation, a second convolution, and finally a scaled
    skip connection with the input.
    """

    def __init__(self, in_channels: int, hidden_channels: int, scale: Optional[float]=0.1):
        """Inits :class:`ResNetBlock`.

        Parameters
        ----------
        in_channels : int
            Input channels.
        hidden_channels : int
            Hidden channels (output channels of firs conv).
        scale : float
            Float that will scale the output of the convolutions before adding the input. Default: 0.1.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x.clone())))
        if self.scale:
            out = self.scale * out
        return x + out


class ResNet(nn.Module):
    """Simple residual network.

    Consisted of a sequence of :class:`ResNetBlocks` followed optionally by batch normalization blocks, followed by
    an output convolution layer.
    """

    def __init__(self, hidden_channels: int, in_channels: int=2, out_channels: Optional[int]=None, num_blocks: int=15, batchnorm: bool=True, scale: Optional[float]=0.1):
        """Inits :class:`ResNet`.

        Parameters
        ----------
        hidden_channels : int
            Hidden dimension.
        in_channels : int
            Input dimension. Default: 2 (for MRI).
        out_channels : int, optional
            Output dimension. If None, will be the same as `in_channels`.
        num_blocks : int
            Number of :class:`ResNetBlocks`. Default: 15.
        batchnorm : bool
            If True, batch normalization will be performed after each :class:`ResNetBlock`.
        scale : float, optional
            Scale parameter for :class:`ResNetBlock`. Default: 0.1
        """
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), padding=(1, 1))
        self.resblocks = []
        for _ in range(num_blocks):
            self.resblocks.append(ResNetBlock(in_channels=hidden_channels, hidden_channels=hidden_channels, scale=scale))
            if batchnorm:
                self.resblocks.append(nn.BatchNorm2d(num_features=hidden_channels))
        self.resblocks = nn.Sequential(*self.resblocks)
        if out_channels is None:
            out_channels = in_channels
        self.conv_out = nn.Sequential(*[nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))])

    def forward(self, input_image: torch.Tensor) ->torch.Tensor:
        """Computes forward pass of :class:`ResNet`.

        Parameters
        ----------
        input_image: torch.Tensor
            Masked k-space of shape (N, in_channels, height, width).

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        return self.conv_out(self.conv_in(input_image) + self.resblocks(self.conv_in(input_image)))


class CrossDomainNetwork(nn.Module):
    """This performs optimisation in both, k-space ("K") and image ("I") domains according to domain_sequence."""

    def __init__(self, forward_operator: Callable, backward_operator: Callable, image_model_list: nn.ModuleList, kspace_model_list: Optional[Union[nn.ModuleList, None]]=None, domain_sequence: str='KIKI', image_buffer_size: int=1, kspace_buffer_size: int=1, normalize_image: bool=False, **kwargs):
        """Inits CrossDomainNetwork.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        image_model_list: nn.ModuleList
            Image domain model list.
        kspace_model_list: Optional[nn.ModuleList]
            K-space domain model list. If set to None, a correction step is applied. Default: None.
        domain_sequence: str
            Domain sequence containing only "K" (k-space domain) and/or "I" (image domain). Default: "KIKI".
        image_buffer_size: int
            Image buffer size. Default: 1.
        kspace_buffer_size: int
            K-space buffer size. Default: 1.
        normalize_image: bool
            If True, input is normalized. Default: False.
        kwargs: dict
            Keyword Arguments.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.domain_sequence = [domain_name for domain_name in domain_sequence.strip()]
        if not set(self.domain_sequence).issubset({'K', 'I'}):
            raise ValueError(f"Invalid domain sequence. Got {domain_sequence}. Should only contain 'K' and 'I'.")
        if kspace_model_list is not None:
            if len(kspace_model_list) != self.domain_sequence.count('K'):
                raise ValueError('K-space domain steps do not match k-space model list length.')
        if len(image_model_list) != self.domain_sequence.count('I'):
            raise ValueError('Image domain steps do not match image model list length.')
        self.kspace_model_list = kspace_model_list
        self.kspace_buffer_size = kspace_buffer_size
        self.image_model_list = image_model_list
        self.image_buffer_size = image_buffer_size
        self.normalize_image = normalize_image
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3

    def kspace_correction(self, block_idx: int, image_buffer: torch.Tensor, kspace_buffer: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor, masked_kspace: torch.Tensor) ->torch.Tensor:
        forward_buffer = torch.cat([self._forward_operator(image.clone(), sampling_mask, sensitivity_map) for image in torch.split(image_buffer, 2, self._complex_dim)], self._complex_dim)
        kspace_buffer = torch.cat([kspace_buffer, forward_buffer, masked_kspace], self._complex_dim)
        if self.kspace_model_list is not None:
            kspace_buffer = self.kspace_model_list[block_idx](kspace_buffer.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        else:
            kspace_buffer = kspace_buffer[..., :2] - kspace_buffer[..., 2:4]
        return kspace_buffer

    def image_correction(self, block_idx: int, image_buffer: torch.Tensor, kspace_buffer: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        backward_buffer = torch.cat([self._backward_operator(kspace.clone(), sampling_mask, sensitivity_map) for kspace in torch.split(kspace_buffer, 2, self._complex_dim)], self._complex_dim)
        image_buffer = torch.cat([image_buffer, backward_buffer], self._complex_dim).permute(0, 3, 1, 2)
        image_buffer = self.image_model_list[block_idx](image_buffer).permute(0, 2, 3, 1)
        return image_buffer

    def _forward_operator(self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        forward = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=image.dtype), self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims))
        return forward

    def _backward_operator(self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        backward = T.reduce_operator(self.backward_operator(torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype), kspace).contiguous(), self._spatial_dims), sensitivity_map, self._coil_dim)
        return backward

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor, scaling_factor: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Computes the forward pass of :class:`CrossDomainNetwork`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        scaling_factor: Optional[torch.Tensor]
            Scaling factor of shape (N,). If None, no scaling is applied. Default: None.

        Returns
        -------
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        if self.normalize_image and scaling_factor is not None:
            input_image = input_image / scaling_factor ** 2
            masked_kspace = masked_kspace / scaling_factor ** 2
        image_buffer = torch.cat([input_image] * self.image_buffer_size, self._complex_dim)
        kspace_buffer = torch.cat([masked_kspace] * self.kspace_buffer_size, self._complex_dim)
        kspace_block_idx, image_block_idx = 0, 0
        for block_domain in self.domain_sequence:
            if block_domain == 'K':
                kspace_buffer = self.kspace_correction(kspace_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map, masked_kspace)
                kspace_block_idx += 1
            else:
                image_buffer = self.image_correction(image_block_idx, image_buffer, kspace_buffer, sampling_mask, sensitivity_map)
                image_block_idx += 1
        if self.normalize_image and scaling_factor is not None:
            image_buffer = image_buffer * scaling_factor ** 2
        out_image = image_buffer[..., :2]
        return out_image


class MultiCoil(nn.Module):
    """This makes the forward pass of multi-coil data of shape (N, N_coils, H, W, C) to a model.

    If coil_to_batch is set to True, coil dimension is moved to the batch dimension. Otherwise, it passes to the model
    each coil-data individually.
    """

    def __init__(self, model: nn.Module, coil_dim: int=1, coil_to_batch: bool=False):
        """Inits :class:`MultiCoil`.

        Parameters
        ----------
        model: nn.Module
            Any nn.Module that takes as input with 4D data (N, H, W, C). Typically a convolutional-like model.
        coil_dim: int
            Coil dimension. Default: 1.
        coil_to_batch: bool
            If True batch and coil dimensions are merged when forwarded by the model and unmerged when outputted.
            Otherwise, input is forwarded to the model per coil.
        """
        super().__init__()
        self.model = model
        self.coil_to_batch = coil_to_batch
        self._coil_dim = coil_dim

    def _compute_model_per_coil(self, data: torch.Tensor) ->torch.Tensor:
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(self.model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Performs the forward pass of MultiCoil.

        Parameters
        ----------
        x: torch.Tensor
            Multi-coil input of shape (N, coil, height, width, in_channels).

        Returns
        -------
        out: torch.Tensor
            Multi-coil output of shape (N, coil, height, width, out_channels).
        """
        if self.coil_to_batch:
            x = x.clone()
            batch, coil, height, width, channels = x.size()
            x = x.reshape(batch * coil, height, width, channels).permute(0, 3, 1, 2).contiguous()
            x = self.model(x).permute(0, 2, 3, 1)
            x = x.reshape(batch, coil, height, width, -1)
        else:
            x = self._compute_model_per_coil(x).contiguous()
        return x


class IterDualNet(nn.Module):
    """Iterative Dual Network solves iteratively the following problem

    .. math ::

        \\min_{x} ||A(x) - y||_2^2 + \\lambda_I ||x - D_I(x)||_2^2 + \\lambda_F ||x - \\mathcal{Q}(D_F(f))||_2^2, \\quad
        \\left\\{ \\begin{array} Q = \\mathcal{F}^{-1}, f = \\mathcal{F}(x) & \\text{if compute_per_coil is False} \\\\
        Q = \\mathcal{F}^{-1} \\circ \\mathcal{E}, f = \\mathcal{R} \\circ \\mathcal{F}(x) & \\text{otherwise} \\end{array}

    by unrolling a gradient descent scheme where :math:`\\mathcal{E}` and :math:`\\mathcal{R}` are the expand and
    reduce operators which use the sensitivity maps. :math:`D_I` and :math:`D_F` are trainable U-Nets operating
    in the image and k-space domain.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_iter: int=10, image_normunet: bool=False, kspace_normunet: bool=False, image_no_parameter_sharing: bool=True, kspace_no_parameter_sharing: bool=True, compute_per_coil: bool=True, **kwargs):
        """Inits :class:`IterDualNet`.

        Parameters
        ----------
        forward_operator : Callable
            Forward Operator.
        backward_operator : Callable
            Backward Operator.
        num_iter : int
            Number of iterations. Default: 10.
        image_normunet : bool
            If True will use NormUNet for the image model. Default: False.
        kspace_normunet : bool
            If True will use NormUNet for the kspace model. Default: False.
        image_no_parameter_sharing : bool
            If False, a single image model will be shared across all iterations. Default: True.
        kspace_no_parameter_sharing : bool
            If False, a single kspace model will be shared across all iterations. Default: True.
        compute_per_coil : bool
            If True :math:`f` will be transformed into a multi-coil kspace.
        kwargs : dict
            Kwargs for unet models.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.image_no_parameter_sharing = image_no_parameter_sharing
        self.kspace_no_parameter_sharing = kspace_no_parameter_sharing
        image_unet_architecture = NormUnetModel2d if image_normunet else UnetModel2d
        kspace_unet_architecture = NormUnetModel2d if kspace_normunet else UnetModel2d
        self.image_block_list = nn.ModuleList()
        self.kspace_block_list = nn.ModuleList()
        for _ in range(self.num_iter if self.image_no_parameter_sharing else 1):
            self.image_block_list.append(image_unet_architecture(in_channels=COMPLEX_SIZE, out_channels=COMPLEX_SIZE, num_filters=kwargs.get('image_unet_num_filters', 8), num_pool_layers=kwargs.get('image_unet_num_pool_layers', 4), dropout_probability=kwargs.get('image_unet_dropout', 0.0)))
        for _ in range(self.num_iter if self.kspace_no_parameter_sharing else 1):
            self.kspace_block_list.append(kspace_unet_architecture(in_channels=COMPLEX_SIZE, out_channels=COMPLEX_SIZE, num_filters=kwargs.get('kspace_unet_num_filters', 8), num_pool_layers=kwargs.get('kspace_unet_num_pool_layers', 4), dropout_probability=kwargs.get('kspace_unet_dropout', 0.0)))
        self.compute_per_coil = compute_per_coil
        self.lr = nn.Parameter(torch.ones(num_iter))
        self.reg_param_I = nn.Parameter(torch.ones(num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(num_iter))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3

    def _image_model(self, image: torch.Tensor, step: int) ->torch.Tensor:
        image = image.permute(0, 3, 1, 2)
        block_idx = step if self.image_no_parameter_sharing else 0
        return self.image_block_list[block_idx](image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace: torch.Tensor, step: int) ->torch.Tensor:
        block_idx = step if self.kspace_no_parameter_sharing else 0
        if self.compute_per_coil:
            kspace = self._compute_model_per_coil(self.kspace_block_list[block_idx], kspace.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2).contiguous()
        else:
            kspace = self.kspace_block_list[block_idx](kspace.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return kspace

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) ->torch.Tensor:
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def _forward_operator(self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        return T.apply_mask(self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims), sampling_mask, return_mask=False)

    def _backward_operator(self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        return T.reduce_operator(self.backward_operator(T.apply_mask(kspace, sampling_mask, return_mask=False), self._spatial_dims), sensitivity_map, self._coil_dim)

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Computes forward pass of :class:`IterDualNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        x = T.reduce_operator(self.backward_operator(masked_kspace, self._spatial_dims), sensitivity_map, self._coil_dim)
        for step in range(self.num_iter):
            f = self.forward_operator(T.expand_operator(x, sensitivity_map, self._coil_dim), dim=self._spatial_dims) if self.compute_per_coil else self.forward_operator(x, dim=[(d - 1) for d in self._spatial_dims])
            kspace_model_out = self._kspace_model(f, step)
            kspace_model_out = T.reduce_operator(self.backward_operator(kspace_model_out, self._spatial_dims), sensitivity_map, self._coil_dim) if self.compute_per_coil else self.backward_operator(kspace_model_out, dim=[(d - 1) for d in self._spatial_dims])
            img_model_out = self._image_model(x, step)
            dc_out = self._backward_operator(self._forward_operator(x, sampling_mask, sensitivity_map) - masked_kspace, sampling_mask, sensitivity_map)
            x = (1 - self.lr[step] * (self.reg_param_I[step] + self.reg_param_F[step])) * x + self.lr[step] * (self.reg_param_I[step] * img_model_out + self.reg_param_F[step] * kspace_model_out - dc_out)
        return x


class JointICNet(nn.Module):
    """Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) implementation as
    presented in [1]_.

    References
    ----------
    .. [1] Jun, Yohan, et al. “Joint Deep Model-Based MR Image and Coil Sensitivity Reconstruction Network (Joint-ICNet) for Fast MRI.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2021, pp. 5266–75. DOI.org (Crossref), https://doi.org/10.1109/CVPR46437.2021.00523.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_iter: int=10, use_norm_unet: bool=False, **kwargs):
        """Inits :class:`JointICNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Transform.
        backward_operator: Callable
            Backward Transform.
        num_iter: int
            Number of unrolled iterations. Default: 10.
        use_norm_unet: bool
            If True, a Normalized U-Net is used. Default: False.
        kwargs: dict
            Image, k-space and sensitivity-map U-Net models keyword-arguments.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        unet_architecture = NormUnetModel2d if use_norm_unet else UnetModel2d
        self.image_model = unet_architecture(in_channels=2, out_channels=2, num_filters=kwargs.get('image_unet_num_filters', 8), num_pool_layers=kwargs.get('image_unet_num_pool_layers', 4), dropout_probability=kwargs.get('image_unet_dropout', 0.0))
        self.kspace_model = unet_architecture(in_channels=2, out_channels=2, num_filters=kwargs.get('kspace_unet_num_filters', 8), num_pool_layers=kwargs.get('kspace_unet_num_pool_layers', 4), dropout_probability=kwargs.get('kspace_unet_dropout', 0.0))
        self.sens_model = unet_architecture(in_channels=2, out_channels=2, num_filters=kwargs.get('sens_unet_num_filters', 8), num_pool_layers=kwargs.get('sens_unet_num_pool_layers', 4), dropout_probability=kwargs.get('sens_unet_dropout', 0.0))
        self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.reg_param_I = nn.Parameter(torch.ones(num_iter))
        self.reg_param_F = nn.Parameter(torch.ones(num_iter))
        self.reg_param_C = nn.Parameter(torch.ones(num_iter))
        self.lr_image = nn.Parameter(torch.ones(num_iter))
        self.lr_sens = nn.Parameter(torch.ones(num_iter))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3

    def _image_model(self, image: torch.Tensor) ->torch.Tensor:
        image = image.permute(0, 3, 1, 2)
        return self.image_model(image).permute(0, 2, 3, 1).contiguous()

    def _kspace_model(self, kspace: torch.Tensor) ->torch.Tensor:
        kspace = kspace.permute(0, 3, 1, 2)
        return self.kspace_model(kspace).permute(0, 2, 3, 1).contiguous()

    def _sens_model(self, sensitivity_map: torch.Tensor) ->torch.Tensor:
        return self._compute_model_per_coil(self.sens_model, sensitivity_map.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2).contiguous()

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) ->torch.Tensor:
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        return torch.stack(output, dim=self._coil_dim)

    def _forward_operator(self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        forward = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=image.dtype), self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims))
        return forward

    def _backward_operator(self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        backward = T.reduce_operator(self.backward_operator(torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype), kspace), self._spatial_dims), sensitivity_map, self._coil_dim)
        return backward

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Computes forward pass of :class:`JointICNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        out_image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        input_image = input_image / T.modulus(input_image).unsqueeze(self._coil_dim).amax(dim=self._spatial_dims).view(-1, 1, 1, 1)
        for curr_iter in range(self.num_iter):
            step_sensitivity_map = 2 * self.lr_sens[curr_iter] * (T.complex_multiplication(self.backward_operator(torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), self._forward_operator(input_image, sampling_mask, sensitivity_map) - masked_kspace), self._spatial_dims), T.conjugate(input_image).unsqueeze(self._coil_dim)) + self.reg_param_C[curr_iter] * (sensitivity_map - self._sens_model(self.backward_operator(masked_kspace, dim=self._spatial_dims))))
            sensitivity_map = sensitivity_map - step_sensitivity_map
            sensitivity_map_norm = torch.sqrt((sensitivity_map ** 2).sum(self._complex_dim).sum(self._coil_dim))
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self._complex_dim).unsqueeze(self._coil_dim)
            sensitivity_map = T.safe_divide(sensitivity_map, sensitivity_map_norm)
            input_kspace = self.forward_operator(input_image, dim=tuple(d - 1 for d in self._spatial_dims))
            step_image = 2 * self.lr_image[curr_iter] * (self._backward_operator(self._forward_operator(input_image, sampling_mask, sensitivity_map) - masked_kspace, sampling_mask, sensitivity_map) + self.reg_param_I[curr_iter] * (input_image - self._image_model(input_image)) + self.reg_param_F[curr_iter] * (input_image - self.backward_operator(self._kspace_model(input_kspace), dim=tuple(d - 1 for d in self._spatial_dims))))
            input_image = input_image - step_image
            input_image = input_image / T.modulus(input_image).unsqueeze(self._coil_dim).amax(dim=self._spatial_dims).view(-1, 1, 1, 1)
        out_image = self.conv_out(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_image


class DWT(nn.Module):
    """2D Discrete Wavelet Transform as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self):
        """Inits :class:`DWT`."""
        super().__init__()
        self.requires_grad = False

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computes DWT(`x`) given tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        out: torch.Tensor
            DWT of `x`.
        """
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class DilatedConvBlock(nn.Module):
    """Double dilated Convolution Block fpr :class:`MWCNN` as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self, in_channels: int, dilations: Tuple[int, int], kernel_size: int, out_channels: Optional[int]=None, bias: bool=True, batchnorm: bool=False, activation: nn.Module=nn.ReLU(True), scale: Optional[float]=1.0):
        """Inits :class:`DilatedConvBlock`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        dilations: (int, int)
            Number of dilations.
        kernel_size: int
            Conv kernel size.
        out_channels: int
            Number of output channels.
        bias: bool
            Use convolution bias. Default: True.
        batchnorm: bool
            Use batch normalization. Default: False.
        activation: nn.Module
            Activation function. Default: nn.ReLU(True).
        scale: float, optional
            Scale. Default: 1.0.
        """
        super().__init__()
        net: List[nn.Module] = []
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, dilation=dilations[0], padding=kernel_size // 2 + dilations[0] - 1))
        if batchnorm:
            net.append(nn.BatchNorm2d(num_features=in_channels, eps=0.0001, momentum=0.95))
        net.append(activation)
        if out_channels is None:
            out_channels = in_channels
        net.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, dilation=dilations[1], padding=kernel_size // 2 + dilations[1] - 1))
        if batchnorm:
            net.append(nn.BatchNorm2d(num_features=in_channels, eps=0.0001, momentum=0.95))
        net.append(activation)
        self.net = nn.Sequential(*net)
        self.scale = scale

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`DilatedConvBlock`.

        Parameters
        ----------
        x: torch.Tensor
            Input with shape (N, C, H, W).

        Returns
        -------
        output: torch.Tensor
            Output with shape (N, C', H', W').
        """
        output = self.net(x) * self.scale
        return output


class IWT(nn.Module):
    """2D Inverse Wavelet Transform as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self):
        """Inits :class:`IWT`."""
        super().__init__()
        self.requires_grad = False
        self._r = 2

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computes IWT(`x`) given tensor `x`.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        h: torch.Tensor
            IWT of `x`.
        """
        batch, in_channel, in_height, in_width = x.size()
        out_channel, out_height, out_width = int(in_channel / self._r ** 2), self._r * in_height, self._r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([batch, out_channel, out_height, out_width], dtype=x.dtype)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h


class MWCNN(nn.Module):
    """Multi-level Wavelet CNN (MWCNN) implementation as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018. arXiv.org, http://arxiv.org/abs/1805.07071.
    """

    def __init__(self, input_channels: int, first_conv_hidden_channels: int, num_scales: int=4, bias: bool=True, batchnorm: bool=False, activation: nn.Module=nn.ReLU(True)):
        """Inits :class:`MWCNN`.

        Parameters
        ----------
        input_channels: int
            Input channels dimension.
        first_conv_hidden_channels: int
            First convolution output channels dimension.
        num_scales: int
            Number of scales. Default: 4.
        bias: bool
            Convolution bias. If True, adds a learnable bias to the output. Default: True.
        batchnorm: bool
            If True, a batchnorm layer is added after each convolution. Default: False.
        activation: nn.Module
            Activation function applied after each convolution. Default: nn.ReLU().
        """
        super().__init__()
        self._kernel_size = 3
        self.DWT = DWT()
        self.IWT = IWT()
        self.down = nn.ModuleList()
        for idx in range(0, num_scales):
            in_channels = input_channels if idx == 0 else first_conv_hidden_channels * 2 ** (idx + 1)
            out_channels = first_conv_hidden_channels * 2 ** idx
            dilations = (2, 1) if idx != num_scales - 1 else (2, 3)
            self.down.append(nn.Sequential(OrderedDict([(f'convblock{idx}', ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=self._kernel_size, bias=bias, batchnorm=batchnorm, activation=activation)), (f'dilconvblock{idx}', DilatedConvBlock(in_channels=out_channels, dilations=dilations, kernel_size=self._kernel_size, bias=bias, batchnorm=batchnorm, activation=activation))])))
        self.up = nn.ModuleList()
        for idx in range(num_scales)[::-1]:
            in_channels = first_conv_hidden_channels * 2 ** idx
            out_channels = input_channels if idx == 0 else first_conv_hidden_channels * 2 ** (idx + 1)
            dilations = (2, 1) if idx != num_scales - 1 else (3, 2)
            self.up.append(nn.Sequential(OrderedDict([(f'invdilconvblock{num_scales - 2 - idx}', DilatedConvBlock(in_channels=in_channels, dilations=dilations, kernel_size=self._kernel_size, bias=bias, batchnorm=batchnorm, activation=activation)), (f'invconvblock{num_scales - 2 - idx}', ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=self._kernel_size, bias=bias, batchnorm=batchnorm, activation=activation))])))
        self.num_scales = num_scales

    @staticmethod
    def pad(x: torch.Tensor) ->torch.Tensor:
        padding = [0, 0, 0, 0]
        if x.shape[-2] % 2 != 0:
            padding[3] = 1
        if x.shape[-1] % 2 != 0:
            padding[1] = 1
        if sum(padding) != 0:
            x = F.pad(x, padding, 'reflect')
        return x

    @staticmethod
    def crop_to_shape(x: torch.Tensor, shape: tuple) ->torch.Tensor:
        h, w = x.shape[-2:]
        if h > shape[0]:
            x = x[:, :, :shape[0], :]
        if w > shape[1]:
            x = x[:, :, :, :shape[1]]
        return x

    def forward(self, input_tensor: torch.Tensor, res: bool=False) ->torch.Tensor:
        """Computes forward pass of :class:`MWCNN`.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        res: bool
            If True, residual connection is applied to the output. Default: False.

        Returns
        -------
        x: torch.Tensor
            Output tensor.
        """
        res_values = []
        x = self.pad(input_tensor.clone())
        for idx in range(self.num_scales):
            if idx == 0:
                x = self.pad(self.down[idx](x))
                res_values.append(x)
            elif idx == self.num_scales - 1:
                x = self.down[idx](self.DWT(x))
            else:
                x = self.pad(self.down[idx](self.DWT(x)))
                res_values.append(x)
        for idx in range(self.num_scales):
            if idx != self.num_scales - 1:
                x = self.crop_to_shape(self.IWT(self.up[idx](x)), res_values[self.num_scales - 2 - idx].shape[-2:]) + res_values[self.num_scales - 2 - idx]
            else:
                x = self.crop_to_shape(self.up[idx](x), input_tensor.shape[-2:])
                if res:
                    x += input_tensor
        return x


class KIKINet(nn.Module):
    """Based on KIKINet implementation [1]_. Modified to work with multi-coil k-space data.

    References
    ----------

    .. [1] Eo, Taejoon, et al. “KIKI-Net: Cross-Domain Convolutional Neural Networks for Reconstructing Undersampled Magnetic Resonance Images.” Magnetic Resonance in Medicine, vol. 80, no. 5, Nov. 2018, pp. 2188–201. PubMed, https://doi.org/10.1002/mrm.27201.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, image_model_architecture: str='MWCNN', kspace_model_architecture: str='DIDN', num_iter: int=2, normalize: bool=False, **kwargs):
        """Inits :class:`KIKINet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        image_model_architecture: str
            Image model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        kspace_model_architecture: str
            Kspace model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        num_iter: int
            Number of unrolled iterations.
        normalize: bool
            If true, input is normalised based on input scaling_factor.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        super().__init__()
        image_model: nn.Module
        if image_model_architecture == 'MWCNN':
            image_model = MWCNN(input_channels=2, first_conv_hidden_channels=kwargs.get('image_mwcnn_hidden_channels', 32), num_scales=kwargs.get('image_mwcnn_num_scales', 4), bias=kwargs.get('image_mwcnn_bias', False), batchnorm=kwargs.get('image_mwcnn_batchnorm', False))
        elif image_model_architecture in ['UNET', 'NORMUNET']:
            unet = UnetModel2d if image_model_architecture == 'UNET' else NormUnetModel2d
            image_model = unet(in_channels=2, out_channels=2, num_filters=kwargs.get('image_unet_num_filters', 8), num_pool_layers=kwargs.get('image_unet_num_pool_layers', 4), dropout_probability=kwargs.get('image_unet_dropout_probability', 0.0))
        else:
            raise NotImplementedError(f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET.Got {image_model_architecture}.")
        kspace_model: nn.Module
        if kspace_model_architecture == 'CONV':
            kspace_model = Conv2d(in_channels=2, out_channels=2, hidden_channels=kwargs.get('kspace_conv_hidden_channels', 16), n_convs=kwargs.get('kspace_conv_n_convs', 4), batchnorm=kwargs.get('kspace_conv_batchnorm', False))
        elif kspace_model_architecture == 'DIDN':
            kspace_model = DIDN(in_channels=2, out_channels=2, hidden_channels=kwargs.get('kspace_didn_hidden_channels', 16), num_dubs=kwargs.get('kspace_didn_num_dubs', 6), num_convs_recon=kwargs.get('kspace_didn_num_convs_recon', 9))
        elif kspace_model_architecture in ['UNET', 'NORMUNET']:
            unet = UnetModel2d if kspace_model_architecture == 'UNET' else NormUnetModel2d
            kspace_model = unet(in_channels=2, out_channels=2, num_filters=kwargs.get('kspace_unet_num_filters', 8), num_pool_layers=kwargs.get('kspace_unet_num_pool_layers', 4), dropout_probability=kwargs.get('kspace_unet_dropout_probability', 0.0))
        else:
            raise NotImplementedError(f"XPDNet is currently implemented for kspace_model_architecture == 'CONV', 'DIDN', 'UNET' or 'NORMUNET'. Got kspace_model_architecture == {kspace_model_architecture}.")
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3
        self.image_model_list = nn.ModuleList([image_model] * num_iter)
        self.kspace_model_list = nn.ModuleList([MultiCoil(kspace_model, self._coil_dim)] * num_iter)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.normalize = normalize

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor, scaling_factor: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Computes forward pass of :class:`KIKINet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        scaling_factor: Optional[torch.Tensor]
            Scaling factor of shape (N,). If None, no scaling is applied. Default: None.

        Returns
        -------
        image: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        kspace = masked_kspace.clone()
        if self.normalize and scaling_factor is not None:
            kspace = kspace / (scaling_factor ** 2).view(-1, 1, 1, 1, 1)
        for idx in range(self.num_iter):
            kspace = self.kspace_model_list[idx](kspace.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
            image = T.reduce_operator(self.backward_operator(torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype), kspace).contiguous(), self._spatial_dims), sensitivity_map, self._coil_dim)
            image = self.image_model_list[idx](image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if idx < self.num_iter - 1:
                kspace = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=image.dtype), self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims))
        if self.normalize and scaling_factor is not None:
            image = image * (scaling_factor ** 2).view(-1, 1, 1, 1)
        return image


class DualNet(nn.Module):
    """Dual Network for Learned Primal Dual Network."""

    def __init__(self, num_dual: int, **kwargs):
        """Inits :class:`DualNet`.

        Parameters
        ----------
        num_dual: int
            Number of dual for LPD algorithm.
        kwargs: dict
        """
        super().__init__()
        if kwargs.get('dual_architectue') is None:
            n_hidden = kwargs.get('n_hidden')
            if n_hidden is None:
                raise ValueError('Missing argument n_hidden.')
            self.dual_block = nn.Sequential(*[nn.Conv2d(2 * (num_dual + 2), n_hidden, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(n_hidden, 2 * num_dual, kernel_size=3, padding=1)])
        else:
            self.dual_block = kwargs.get('dual_architectue')

    @staticmethod
    def compute_model_per_coil(model: nn.Module, data: torch.Tensor) ->torch.Tensor:
        """Computes model per coil.

        Parameters
        ----------
        model: nn.Module
            Model to compute.
        data: torch.Tensor
            Multi-coil input.

        Returns
        -------
        output: torch.Tensor
            Multi-coil output.
        """
        output = []
        for idx in range(data.size(1)):
            subselected_data = data.select(1, idx)
            output.append(model(subselected_data))
        return torch.stack(output, dim=1)

    def forward(self, h: torch.Tensor, forward_f: torch.Tensor, g: torch.Tensor) ->torch.Tensor:
        inp = torch.cat([h, forward_f, g], dim=-1).permute(0, 1, 4, 2, 3)
        return self.compute_model_per_coil(self.dual_block, inp).permute(0, 1, 3, 4, 2)


class PrimalNet(nn.Module):
    """Primal Network for Learned Primal Dual Network."""

    def __init__(self, num_primal: int, **kwargs):
        """Inits :class:`PrimalNet`.

        Parameters
        ----------
        num_primal: int
            Number of primal for LPD algorithm.
        """
        super().__init__()
        if kwargs.get('primal_architectue') is None:
            n_hidden = kwargs.get('n_hidden')
            if n_hidden is None:
                raise ValueError('Missing argument n_hidden.')
            self.primal_block = nn.Sequential(*[nn.Conv2d(2 * (num_primal + 1), n_hidden, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(n_hidden, 2 * num_primal, kernel_size=3, padding=1)])
        else:
            self.primal_block = kwargs.get('primal_architectue')

    def forward(self, f: torch.Tensor, backward_h: torch.Tensor) ->torch.Tensor:
        inp = torch.cat([f, backward_h], dim=-1).permute(0, 3, 1, 2)
        return self.primal_block(inp).permute(0, 2, 3, 1)


class LPDNet(nn.Module):
    """Learned Primal Dual network implementation inspired by [1]_.

    References
    ----------

    .. [1] Adler, Jonas, and Ozan Öktem. “Learned Primal-Dual Reconstruction.” IEEE Transactions on Medical Imaging, vol. 37, no. 6, June 2018, pp. 1322–32. arXiv.org, https://doi.org/10.1109/TMI.2018.2799231.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_iter: int, num_primal: int, num_dual: int, primal_model_architecture: str='MWCNN', dual_model_architecture: str='DIDN', **kwargs):
        """Inits :class:`LPDNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_iter: int
            Number of unrolled iterations.
        num_primal: int
            Number of primal networks.
        num_dual: int
            Number of dual networks.
        primal_model_architecture: str
            Primal model architecture. Currently only implemented for MWCNN and (NORM)UNET. Default: 'MWCNN'.
        dual_model_architecture: str
            Dual model architecture. Currently only implemented for CONV and DIDN and (NORM)UNET. Default: 'DIDN'.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.num_iter = num_iter
        self.num_primal = num_primal
        self.num_dual = num_dual
        primal_model: nn.Module
        if primal_model_architecture == 'MWCNN':
            primal_model = nn.Sequential(*[MWCNN(input_channels=2 * (num_primal + 1), first_conv_hidden_channels=kwargs.get('primal_mwcnn_hidden_channels', 32), num_scales=kwargs.get('primal_mwcnn_num_scales', 4), bias=kwargs.get('primal_mwcnn_bias', False), batchnorm=kwargs.get('primal_mwcnn_batchnorm', False)), nn.Conv2d(2 * (num_primal + 1), 2 * num_primal, kernel_size=1)])
        elif primal_model_architecture in ['UNET', 'NORMUNET']:
            unet = UnetModel2d if primal_model_architecture == 'UNET' else NormUnetModel2d
            primal_model = unet(in_channels=2 * (num_primal + 1), out_channels=2 * num_primal, num_filters=kwargs.get('primal_unet_num_filters', 8), num_pool_layers=kwargs.get('primal_unet_num_pool_layers', 4), dropout_probability=kwargs.get('primal_unet_dropout_probability', 0.0))
        else:
            raise NotImplementedError(f"XPDNet is currently implemented only with primal_model_architecture == 'MWCNN', 'UNET' or 'NORMUNET.Got {primal_model_architecture}.")
        dual_model: nn.Module
        if dual_model_architecture == 'CONV':
            dual_model = Conv2d(in_channels=2 * (num_dual + 2), out_channels=2 * num_dual, hidden_channels=kwargs.get('dual_conv_hidden_channels', 16), n_convs=kwargs.get('dual_conv_n_convs', 4), batchnorm=kwargs.get('dual_conv_batchnorm', False))
        elif dual_model_architecture == 'DIDN':
            dual_model = DIDN(in_channels=2 * (num_dual + 2), out_channels=2 * num_dual, hidden_channels=kwargs.get('dual_didn_hidden_channels', 16), num_dubs=kwargs.get('dual_didn_num_dubs', 6), num_convs_recon=kwargs.get('dual_didn_num_convs_recon', 9))
        elif dual_model_architecture in ['UNET', 'NORMUNET']:
            unet = UnetModel2d if dual_model_architecture == 'UNET' else NormUnetModel2d
            dual_model = unet(in_channels=2 * (num_dual + 2), out_channels=2 * num_dual, num_filters=kwargs.get('dual_unet_num_filters', 8), num_pool_layers=kwargs.get('dual_unet_num_pool_layers', 4), dropout_probability=kwargs.get('dual_unet_dropout_probability', 0.0))
        else:
            raise NotImplementedError(f"XPDNet is currently implemented for dual_model_architecture == 'CONV', 'DIDN', 'UNET' or 'NORMUNET'. Got dual_model_architecture == {dual_model_architecture}.")
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3
        self.primal_net = nn.ModuleList([PrimalNet(num_primal, primal_architectue=primal_model) for _ in range(num_iter)])
        self.dual_net = nn.ModuleList([DualNet(num_dual, dual_architectue=dual_model) for _ in range(num_iter)])

    def _forward_operator(self, image: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        forward = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=image.dtype), self.forward_operator(T.expand_operator(image, sensitivity_map, self._coil_dim), dim=self._spatial_dims))
        return forward

    def _backward_operator(self, kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        backward = T.reduce_operator(self.backward_operator(torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=kspace.dtype), kspace).contiguous(), self._spatial_dims), sensitivity_map, self._coil_dim)
        return backward

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor, sampling_mask: torch.Tensor) ->torch.Tensor:
        """Computes forward pass of :class:`LPDNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        input_image = self._backward_operator(masked_kspace, sampling_mask, sensitivity_map)
        dual_buffer = torch.cat([masked_kspace] * self.num_dual, self._complex_dim)
        primal_buffer = torch.cat([input_image] * self.num_primal, self._complex_dim)
        for curr_iter in range(self.num_iter):
            f_2 = primal_buffer[..., 2:4].clone()
            dual_buffer = self.dual_net[curr_iter](dual_buffer, self._forward_operator(f_2, sampling_mask, sensitivity_map), masked_kspace)
            h_1 = dual_buffer[..., 0:2].clone()
            primal_buffer = self.primal_net[curr_iter](primal_buffer, self._backward_operator(h_1, sampling_mask, sensitivity_map))
        output = primal_buffer[..., 0:2]
        return output


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), norm_layer(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise AssertionError
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def str_to_class(module_name: str, function_name: str) ->Callable:
    """Convert a string to a class Base on: https://stackoverflow.com/a/1176180/576363.

    Also support function arguments, e.g. ifft(dim=2) will be parsed as a partial and return ifft where dim has been
    set to 2.


    Examples
    --------
    >>> def mult(f, mul=2):
    >>>    return f*mul

    >>> str_to_class(".", "mult(mul=4)")
    >>> str_to_class(".", "mult(mul=4)")
    will return a function which multiplies the input times 4, while

    >>> str_to_class(".", "mult")
    just returns the function itself.

    Parameters
    ----------
    module_name: str
        e.g. direct.data.transforms
    function_name: str
        e.g. Identity
    Returns
    -------
    object
    """
    tree = ast.parse(function_name)
    func_call = tree.body[0].value
    args = [ast.literal_eval(arg) for arg in func_call.args] if hasattr(func_call, 'args') else []
    kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords} if hasattr(func_call, 'keywords') else {}
    module = importlib.import_module(module_name)
    if not args and not kwargs:
        return getattr(module, function_name)
    return functools.partial(getattr(module, func_call.func.id), *args, **kwargs)


class MobileNetV2(nn.Module):

    def __init__(self, num_channels=2, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer: Callable[..., Any]=None):
        """MobileNet V2 main class.

        Parameters
        ----------
        num_channels: int
            Number of channels.
        num_classes: int
            Number of classes.
        width_mult: float
            Width multiplier - adjusts number of channels in each layer by this amount.
        inverted_residual_setting: Network structure
        round_nearest: int
            Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        block: str
            Module specifying inverted residual building block for mobilenet.
        norm_layer: str
            Module specifying the normalization layer to use.
        """
        super().__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        else:
            module_name = '.'.join(str(norm_layer).split('.')[:-1])
            norm_layer = str_to_class(f'torch.{module_name}', str(norm_layer).split('.')[-1])
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(f'inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}')
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(num_channels, input_channel, stride=2, norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for idx in range(n):
                stride = s if idx == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class MultiDomainConv2d(nn.Module):

    def __init__(self, forward_operator: Callable, backward_operator: Callable, in_channels: int, out_channels: int, **kwargs):
        """Inits :class:`MultiDomainConv2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.image_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._channels_dim = 1
        self._spatial_dims = 1, 2

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConv2d`.

        Parameters
        ----------
        image: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        kspace = torch.cat(tensors=[self.forward_operator(im, dim=self._spatial_dims) for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)], dim=-1).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)
        backward = torch.cat(tensors=[self.backward_operator(ks, dim=self._spatial_dims) for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)], dim=-1).permute(0, 3, 1, 2)
        image = self.image_conv(image)
        image = torch.cat([image, backward], dim=self._channels_dim)
        return image


class MultiDomainConvTranspose2d(nn.Module):

    def __init__(self, forward_operator: Callable, backward_operator: Callable, in_channels: int, out_channels: int, **kwargs):
        """Inits :class:`MultiDomainConvTranspose2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.image_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.kspace_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels // 2, **kwargs)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._channels_dim = 1
        self._spatial_dims = 1, 2

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConvTranspose2d`.

        Parameters
        ----------
        image: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        kspace = torch.cat(tensors=[self.forward_operator(im, dim=self._spatial_dims) for im in torch.split(image.permute(0, 2, 3, 1).contiguous(), 2, -1)], dim=-1).permute(0, 3, 1, 2)
        kspace = self.kspace_conv(kspace)
        backward = torch.cat(tensors=[self.backward_operator(ks, dim=self._spatial_dims) for ks in torch.split(kspace.permute(0, 2, 3, 1).contiguous(), 2, -1)], dim=-1).permute(0, 3, 1, 2)
        image = self.image_conv(image)
        return torch.cat([image, backward], dim=self._channels_dim)


class MultiDomainConvBlock(nn.Module):
    """A multi-domain convolutional block that consists of two multi-domain convolution layers each followed by instance
    normalization, LeakyReLU activation and dropout."""

    def __init__(self, forward_operator: Callable, backward_operator: Callable, in_channels: int, out_channels: int, dropout_probability: float):
        """Inits :class:`MultiDomainConvBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_probability = dropout_probability
        self.layers = nn.Sequential(MultiDomainConv2d(forward_operator, backward_operator, in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(dropout_probability), MultiDomainConv2d(forward_operator, backward_operator, out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Dropout2d(dropout_probability))

    def forward(self, _input: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of of :class:`MultiDomainConvBlock`.

        Parameters
        ----------
        _input: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(_input)

    def __repr__(self):
        """Representation of :class:`MultiDomainConvBlock`."""
        return f'MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, dropout_probability={self.dropout_probability})'


class TransposeMultiDomainConvBlock(nn.Module):
    """A Transpose Convolutional Block that consists of one convolution transpose layers followed by instance
    normalization and LeakyReLU activation."""

    def __init__(self, forward_operator, backward_operator, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(MultiDomainConvTranspose2d(forward_operator, backward_operator, in_channels, out_channels, kernel_size=2, stride=2, bias=False), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, input_data: torch.Tensor):
        """

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.layers(input_data)

    def __repr__(self):
        return f'MultiDomainConvBlock(in_channels={self.in_channels}, out_channels={self.out_channels})'


class MultiDomainUnet2d(nn.Module):
    """Unet modification to be used with Multi-domain network as in AIRS Medical submission to the Fast MRI 2020
    challenge."""

    def __init__(self, forward_operator: Callable, backward_operator: Callable, in_channels: int, out_channels: int, num_filters: int, num_pool_layers: int, dropout_probability: float):
        """

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        in_channels: int
            Number of input channels to the u-net.
        out_channels: int
            Number of output channels to the u-net.
        num_filters: int
            Number of output channels of the first convolutional layer.
        num_pool_layers: int
            Number of down-sampling and up-sampling layers (depth).
        dropout_probability: float
            Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_pool_layers = num_pool_layers
        self.dropout_probability = dropout_probability
        self.down_sample_layers = nn.ModuleList([MultiDomainConvBlock(forward_operator, backward_operator, in_channels, num_filters, dropout_probability)])
        ch = num_filters
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [MultiDomainConvBlock(forward_operator, backward_operator, ch, ch * 2, dropout_probability)]
            ch *= 2
        self.conv = MultiDomainConvBlock(forward_operator, backward_operator, ch, ch * 2, dropout_probability)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeMultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch)]
            self.up_conv += [MultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch, dropout_probability)]
            ch //= 2
        self.up_transpose_conv += [TransposeMultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch)]
        self.up_conv += [nn.Sequential(MultiDomainConvBlock(forward_operator, backward_operator, ch * 2, ch, dropout_probability), nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1))]

    def forward(self, input_data: torch.Tensor):
        """

        Parameters
        ----------
        input_data: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        stack = []
        output = input_data
        for _, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) != 0:
                output = F.pad(output, padding, 'reflect')
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class StandardizationLayer(nn.Module):
    """Multi-channel data standardization method. Inspired by AIRS model submission to the Fast MRI 2020 challenge. Given individual coil images :math:`\\{x_i\\}_{i=1}^{N_c}` and sensitivity coil maps :math:`\\{S_i\\}_{i=1}^{N_c}` it returns

    .. math::
        [(x_{\\text{sense}}, {x_{\\text{res}}}_1), ..., (x_{\\text{sense}}, {x_{\\text{res}}}_{N_c})]

    where :math:`{x_{\\text{res}}}_i = xi - S_i \\times x_{\\text{sense}}` and :math:`x_{\\text{sense}} = \\sum_{i=1}^{N_c} {S_i}^{*} \\times x_i`.

    """

    def __init__(self, coil_dim: int=1, channel_dim: int=-1):
        """Inits :class:`StandardizationLayer`.

        Parameters
        ----------
        coil_dim: int
            Coil dimension. Default: 1.
        channel_dim: int
            Channel dimension. Default: -1.
        """
        super().__init__()
        self.coil_dim = coil_dim
        self.channel_dim = channel_dim

    def forward(self, coil_images: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`StandardizationLayer`.

        Parameters
        ----------
        coil_images: torch.Tensor
            Coil images tensor.
        sensitivity_map: torch.Tensor
            Sensitivity maps.

        Returns
        -------
        torch.Tensor
        """
        combined_image = T.reduce_operator(coil_images, sensitivity_map, self.coil_dim)
        residual_image = combined_image.unsqueeze(self.coil_dim) - T.complex_multiplication(sensitivity_map, combined_image.unsqueeze(self.coil_dim))
        concat = torch.cat([torch.cat([combined_image, residual_image.select(self.coil_dim, idx)], self.channel_dim).unsqueeze(self.coil_dim) for idx in range(coil_images.size(self.coil_dim))], self.coil_dim)
        return concat


class MultiDomainNet(nn.Module):
    """Feature-level multi-domain module.

    Inspired by AIRS Medical submission to the Fast MRI 2020 challenge.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, standardization: bool=True, num_filters: int=16, num_pool_layers: int=4, dropout_probability: float=0.0, **kwargs):
        """Inits :class:`MultiDomainNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        standardization: bool
            If True standardization is used. Default: True.
        num_filters: int
            Number of filters for the :class:`MultiDomainUnet` module. Default: 16.
        num_pool_layers: int
            Number of pooling layers for the :class:`MultiDomainUnet` module. Default: 4.
        dropout_probability: float
            Dropout probability for the :class:`MultiDomainUnet` module. Default: 0.0.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3
        if standardization:
            self.standardization = StandardizationLayer(self._coil_dim, self._complex_dim)
        self.unet = MultiDomainUnet2d(forward_operator, backward_operator, in_channels=4 if standardization else 2, out_channels=2, num_filters=num_filters, num_pool_layers=num_pool_layers, dropout_probability=dropout_probability)

    def _compute_model_per_coil(self, model: nn.Module, data: torch.Tensor) ->torch.Tensor:
        """Computes model per coil.

        Parameters
        ----------
        model: nn.Module
            Model to compute.
        data: torch.Tensor
            Data to pass in the model.

        Returns
        -------
        output: torch.Tensor
        """
        output = []
        for idx in range(data.size(self._coil_dim)):
            subselected_data = data.select(self._coil_dim, idx)
            output.append(model(subselected_data))
        output = torch.stack(output, dim=self._coil_dim)
        return output

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Performs forward pass of :class:`MultiDomainNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        output_image: torch.Tensor
            Multi-coil output image of shape (N, coil, height, width, complex=2).
        """
        input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims)
        if hasattr(self, 'standardization'):
            input_image = self.standardization(input_image, sensitivity_map)
        output_image = self._compute_model_per_coil(self.unet, input_image.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        return output_image


class Conv2dGRU(nn.Module):
    """2D Convolutional GRU Network."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: Optional[int]=None, num_layers: int=2, gru_kernel_size=1, orthogonal_initialization: bool=True, instance_norm: bool=False, dense_connect: int=0, replication_padding: bool=True):
        """Inits :class:`Conv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dense_connect = dense_connect
        self.reset_gates = nn.ModuleList([])
        self.update_gates = nn.ModuleList([])
        self.out_gates = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        for idx in range(num_layers + 1):
            in_ch = in_channels if idx == 0 else (1 + min(idx, dense_connect)) * hidden_channels
            out_ch = hidden_channels if idx < num_layers else out_channels
            padding = 0 if replication_padding else 2 if idx == 0 else 1
            block: List[nn.Module] = []
            if replication_padding:
                if idx == 1:
                    block.append(nn.ReplicationPad2d(2))
                else:
                    block.append(nn.ReplicationPad2d(2 if idx == 0 else 1))
            block.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5 if idx == 0 else 3, dilation=2 if idx == 1 else 1, padding=padding))
            self.conv_blocks.append(nn.Sequential(*block))
        for idx in range(num_layers):
            for gru_part in [self.reset_gates, self.update_gates, self.out_gates]:
                gru_block: List[nn.Module] = []
                if instance_norm:
                    gru_block.append(nn.InstanceNorm2d(2 * hidden_channels))
                gru_block.append(nn.Conv2d(in_channels=2 * hidden_channels, out_channels=hidden_channels, kernel_size=gru_kernel_size, padding=gru_kernel_size // 2))
                gru_part.append(nn.Sequential(*gru_block))
        if orthogonal_initialization:
            for reset_gate, update_gate, out_gate in zip(self.reset_gates, self.update_gates, self.out_gates):
                nn.init.orthogonal_(reset_gate[-1].weight)
                nn.init.orthogonal_(update_gate[-1].weight)
                nn.init.orthogonal_(out_gate[-1].weight)
                nn.init.constant_(reset_gate[-1].bias, -1.0)
                nn.init.constant_(update_gate[-1].bias, 0.0)
                nn.init.constant_(out_gate[-1].bias, 0.0)

    def forward(self, cell_input: torch.Tensor, previous_state: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Computes Conv2dGRU forward pass given tensors `cell_input` and `previous_state`.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.
        """
        new_states: List[torch.Tensor] = []
        conv_skip: List[torch.Tensor] = []
        if previous_state is None:
            batch_size, spatial_size = cell_input.size(0), (cell_input.size(2), cell_input.size(3))
            state_size = [batch_size, self.hidden_channels] + list(spatial_size) + [self.num_layers]
            previous_state = torch.zeros(*state_size, dtype=cell_input.dtype)
        for idx in range(self.num_layers):
            if len(conv_skip) > 0:
                cell_input = F.relu(self.conv_blocks[idx](torch.cat([*conv_skip[-self.dense_connect:], cell_input], dim=1)), inplace=True)
            else:
                cell_input = F.relu(self.conv_blocks[idx](cell_input), inplace=True)
            if self.dense_connect > 0:
                conv_skip.append(cell_input)
            stacked_inputs = torch.cat([cell_input, previous_state[:, :, :, :, idx]], dim=1)
            update = torch.sigmoid(self.update_gates[idx](stacked_inputs))
            reset = torch.sigmoid(self.reset_gates[idx](stacked_inputs))
            delta = torch.tanh(self.out_gates[idx](torch.cat([cell_input, previous_state[:, :, :, :, idx] * reset], dim=1)))
            cell_input = previous_state[:, :, :, :, idx] * (1 - update) + delta * update
            new_states.append(cell_input)
            cell_input = F.relu(cell_input, inplace=False)
        if len(conv_skip) > 0:
            out = self.conv_blocks[self.num_layers](torch.cat([*conv_skip[-self.dense_connect:], cell_input], dim=1))
        else:
            out = self.conv_blocks[self.num_layers](cell_input)
        return out, torch.stack(new_states, dim=-1)


class NormConv2dGRU(nn.Module):
    """Normalized 2D Convolutional GRU Network.

    Normalization methods adapted from NormUnet of [1]_.

    References
    ----------

    .. [1] https://github.com/facebookresearch/fastMRI/blob/
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: Optional[int]=None, num_layers: int=2, gru_kernel_size=1, orthogonal_initialization: bool=True, instance_norm: bool=False, dense_connect: int=0, replication_padding: bool=True, norm_groups: int=2):
        """Inits :class:`NormConv2dGRU`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        hidden_channels: int
            Number of hidden channels.
        out_channels: Optional[int]
            Number of output channels. If None, same as in_channels. Default: None.
        num_layers: int
            Number of layers. Default: 2.
        gru_kernel_size: int
            Size of the GRU kernel. Default: 1.
        orthogonal_initialization: bool
            Orthogonal initialization is used if set to True. Default: True.
        instance_norm: bool
            Instance norm is used if set to True. Default: False.
        dense_connect: int
            Number of dense connections.
        replication_padding: bool
            If set to true replication padding is applied.
        norm_groups: int,
            Number of normalization groups.
        """
        super().__init__()
        self.convgru = Conv2dGRU(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, gru_kernel_size=gru_kernel_size, orthogonal_initialization=orthogonal_initialization, instance_norm=instance_norm, dense_connect=dense_connect, replication_padding=replication_padding)
        self.norm_groups = norm_groups

    @staticmethod
    def norm(input_data: torch.Tensor, num_groups: int) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs group normalization."""
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        mean = input_data.mean(-1, keepdim=True)
        std = input_data.std(-1, keepdim=True)
        output = (input_data - mean) / std
        output = output.reshape(b, c, h, w)
        return output, mean, std

    @staticmethod
    def unnorm(input_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, num_groups: int) ->torch.Tensor:
        b, c, h, w = input_data.shape
        input_data = input_data.reshape(b, num_groups, -1)
        return (input_data * std + mean).reshape(b, c, h, w)

    def forward(self, cell_input: torch.Tensor, previous_state: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Computes :class:`NormConv2dGRU` forward pass given tensors `cell_input` and `previous_state`.

        It performs group normalization on the input before the forward pass to the Conv2dGRU.
        Output of Conv2dGRU is then un-normalized.

        Parameters
        ----------
        cell_input: torch.Tensor
            Input tensor.
        previous_state: torch.Tensor
            Tensor of previous hidden state.

        Returns
        -------
        out, new_states: (torch.Tensor, torch.Tensor)
            Output and new states.

        """
        cell_input, mean, std = self.norm(cell_input, self.norm_groups)
        cell_input, previous_state = self.convgru(cell_input, previous_state)
        cell_input = self.unnorm(cell_input, mean, std, self.norm_groups)
        return cell_input, previous_state


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.

    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock
    of the RecurrentVarNet.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(self, in_channels: int, out_channels: int, channels: Tuple[int, ...], dilations: Tuple[int, ...], depth: int=2, multiscale_depth: int=1):
        """Inits :class:`RecurrentInit`.

        Parameters
        ----------
        in_channels: int
            Input channels.
        out_channels: int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
        channels: tuple
            Channels :math:`n_d` in the convolutional layers of initializer.
        dilations: tuple
            Dilations :math:`p` of the convolutional layers of the initializer.
        depth: int
            RecurrentVarNet Block number of layers :math:`n_l`.
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [nn.ReplicationPad2d(curr_dilations), nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations)]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Computes initialization for recurrent unit given input `x`.

        Parameters
        ----------
        x: torch.Tensor
            Initialization for RecurrentInit.

        Returns
        -------
        out: torch.Tensor
            Initial recurrent hidden state from input `x`.
        """
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth:], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class RecurrentVarNetBlock(nn.Module):
    """Recurrent Variational Network Block :math:`\\mathcal{H}_{\\theta_{t}}` as presented in [1]_.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.

    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, in_channels: int=2, hidden_channels: int=64, num_layers: int=4, normalized: bool=False):
        """Inits RecurrentVarNetBlock.

        Parameters
        ----------
        forward_operator: Callable
            Forward Fourier Transform.
        backward_operator: Callable
            Backward Fourier Transform.
        in_channels: int,
            Input channel number. Default is 2 for complex data.
        hidden_channels: int,
            Hidden channels. Default: 64.
        num_layers: int,
            Number of layers of :math:`n_l` recurrent unit. Default: 4.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used as a regularizer. Default: False.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        regularizer_params = {'in_channels': in_channels, 'hidden_channels': hidden_channels, 'num_layers': num_layers, 'replication_padding': True}
        self.regularizer = NormConv2dGRU(**regularizer_params) if normalized else Conv2dGRU(**regularizer_params)

    def forward(self, current_kspace: torch.Tensor, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor, hidden_state: Union[None, torch.Tensor], coil_dim: int=1, spatial_dims: Tuple[int, int]=(2, 3)) ->Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass of RecurrentVarNetBlock.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            Recurrent unit hidden state of shape (N, hidden_channels, height, width, num_layers) if not None. Optional.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).

        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor
            Next hidden state of shape (N, hidden_channels, height, width, num_layers).
        """
        kspace_error = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), current_kspace - masked_kspace)
        recurrent_term = reduce_operator(self.backward_operator(current_kspace, dim=spatial_dims), sensitivity_map, dim=coil_dim).permute(0, 3, 1, 2)
        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)
        recurrent_term = self.forward_operator(expand_operator(recurrent_term, sensitivity_map, dim=coil_dim), dim=spatial_dims)
        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term
        return new_kspace, hidden_state


class RIMInit(nn.Module):
    """Learned initializer for RIM, based on multi-scale context aggregation with dilated convolutions, that replaces
    zero initializer for the RIM hidden vector. Inspired by [1]_.

    References
    ----------

    .. [1] Yu, Fisher, and Vladlen Koltun. “Multi-Scale Context Aggregation by Dilated Convolutions.” ArXiv:1511.07122 [Cs], Apr. 2016. arXiv.org, http://arxiv.org/abs/1511.07122.
    """

    def __init__(self, x_ch: int, out_ch: int, channels: Tuple[int, ...], dilations: Tuple[int, ...], depth: int=2, multiscale_depth: int=1):
        """Inits :class:`RIMInit`.

        Parameters
        ----------
        x_ch: int
            Input channels.
        out_ch: int
            Number of hidden channels in the RIM.
        channels: tuple
            Channels in the convolutional layers of initializer. Typical it could be e.g. (32, 32, 64, 64).
        dilations: tuple
            Dilations of the convolutional layers of the initializer. Typically it could be e.g. (1, 1, 2, 4).
        depth: int
            RIM depth
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = x_ch
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [nn.ReplicationPad2d(curr_dilations), nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations)]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_ch, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth:], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


def assert_positive_integer(*variables, strict: bool=False) ->None:
    """Assert if given variables are positive integer.

    Parameters
    ----------
    variables: Any
    strict: bool
        If true, will allow zero values.
    """
    if not strict:
        type_name = 'positive integer'
    else:
        type_name = 'positive integer larger than zero'
    for variable in variables:
        if not isinstance(variable, int) or variable <= 0 and strict or variable < 0 and not strict:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            variable_name = [var_name for var_name, var_val in callers_local_vars if var_val is variable][0]
            raise ValueError(f'{variable_name} has to be a {type_name}. Got {variable} of type {type(variable)}.')


class RIM(nn.Module):
    """Recurrent Inference Machine Module as in [1]_.

    References
    ----------

    .. [1] Putzky, Patrick, and Max Welling. “Recurrent Inference Machines for Solving Inverse Problems.” ArXiv:1706.04008 [Cs], June 2017. arXiv.org, http://arxiv.org/abs/1706.04008.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, hidden_channels: int, x_channels: int=2, length: int=8, depth: int=1, no_parameter_sharing: bool=True, instance_norm: bool=False, dense_connect: bool=False, skip_connections: bool=True, replication_padding: bool=True, image_initialization: str='zero_filled', learned_initializer: bool=False, initializer_channels: Optional[Tuple[int, ...]]=(32, 32, 64, 64), initializer_dilations: Optional[Tuple[int, ...]]=(1, 1, 2, 4), initializer_multiscale: int=1, normalized: bool=False, **kwargs):
        """Inits :class:`RIM`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        hidden_channels: int
            Number of hidden channels in recurrent unit of RIM.
        x_channels: int
            Number of input channels. Default: 2 (complex data).
        length: int
            Number of time-steps. Default: 8.
        depth: int
            Number of layers of recurrent unit of RIM. Default: 1.
        no_parameter_sharing: bool
            If False, a single recurrent unit will be used for each time-step. Default: True.
        instance_norm: bool
            If True, instance normalization is applied in the recurrent unit of RIM. Default: False.
        dense_connect: bool
            Use dense connection in the recurrent unit of RIM. Default: False.
        skip_connections: bool
            If True, the previous prediction is added to the next. Default: True.
        replication_padding: bool
            Replication padding for the recurrent unit of RIM. Defaul: True.
        image_initialization: str
            Input image initialization for RIM. Can be "sense", "input_kspace", "input_image" or "zero_filled". Default: "zero_filled".
        learned_initializer: bool
            If True, an initializer is trained to learn image initialization. Default: False.
        initializer_channels: Optional[Tuple[int, ...]]
            Number of channels for learned_initializer. If "learned_initializer=False" this is ignored. Default: (32, 32, 64, 64).
        initializer_dilations: Optional[Tuple[int, ...]]
            Number of dilations for learned_initializer. Must have the same length as "initialize_channels".
            If "learned_initializer=False" this is ignored. Default: (1, 1, 2, 4)
        initializer_multiscale: int
            Number of initializer multiscale. If "learned_initializer=False" this is ignored. Default: 1.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used instead of :class:`Conv2dGRU`. Default: False.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in ['steps', 'sensitivity_map_model', 'model_name', 'z_reduction_frequency', 'kspace_context', 'scale_loglikelihood', 'whiten_input']:
                raise ValueError(f'{type(self).__name__} got key `{extra_key}` which is not supported.')
        assert_positive_integer(x_channels, hidden_channels, length, depth)
        self.initializer: Optional[nn.Module] = None
        if learned_initializer and initializer_channels is not None and initializer_dilations is not None:
            self.initializer = RIMInit(x_channels, hidden_channels, channels=initializer_channels, dilations=initializer_dilations, depth=depth, multiscale_depth=initializer_multiscale)
        self.image_initialization = image_initialization
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.grad_likelihood = MRILogLikelihood(forward_operator, backward_operator)
        self.skip_connections = skip_connections
        self.x_channels = x_channels
        self.hidden_channels = hidden_channels
        self.cell_list = nn.ModuleList()
        self.no_parameter_sharing = no_parameter_sharing
        conv_unit_params = {'in_channels': x_channels * 2, 'out_channels': x_channels, 'hidden_channels': hidden_channels, 'num_layers': depth, 'instance_norm': instance_norm, 'dense_connect': dense_connect, 'replication_padding': replication_padding}
        for _ in range(length if no_parameter_sharing else 1):
            self.cell_list.append(NormConv2dGRU(**conv_unit_params) if normalized else Conv2dGRU(**conv_unit_params))
        self.length = length
        self.depth = depth
        self._coil_dim = 1
        self._spatial_dims = 2, 3

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        input_image = T.complex_multiplication(T.conjugate(sensitivity_map), self.backward_operator(kspace, dim=self._spatial_dims))
        input_image = input_image.sum(self._coil_dim)
        return input_image

    def forward(self, input_image: torch.Tensor, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: Optional[torch.Tensor]=None, previous_state: Optional[torch.Tensor]=None, loglikelihood_scaling: Optional[torch.Tensor]=None, **kwargs):
        """Performs forward pass of :class:`RIM`.

        Parameters
        ----------
        input_image: torch.Tensor
            Initial or intermediate guess of input. Has shape (N, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        previous_state: torch.Tensor
        loglikelihood_scaling: torch.Tensor
            Float tensor of shape (1,).

        Returns
        -------
        torch.Tensor
        """
        if input_image is None:
            if self.image_initialization == 'sense':
                input_image = self.compute_sense_init(kspace=masked_kspace, sensitivity_map=sensitivity_map)
            elif self.image_initialization == 'input_kspace':
                if 'initial_kspace' not in kwargs:
                    raise ValueError(f"`'initial_kspace` is required as input if initialization is {self.image_initialization}.")
                input_image = self.compute_sense_init(kspace=kwargs['initial_kspace'], sensitivity_map=sensitivity_map)
            elif self.image_initialization == 'input_image':
                if 'initial_image' not in kwargs:
                    raise ValueError(f"`'initial_image` is required as input if initialization is {self.image_initialization}.")
                input_image = kwargs['initial_image']
            elif self.image_initialization == 'zero_filled':
                input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
            else:
                raise ValueError(f'Unknown image_initialization. Expected `sense`, `input_kspace`, `input_image` or `zero_filled`. Got {self.image_initialization}.')
        if self.initializer is not None and previous_state is None:
            previous_state = self.initializer(input_image.permute(0, 3, 1, 2))
        input_image = input_image.permute(0, 3, 1, 2).contiguous()
        batch_size = input_image.size(0)
        spatial_shape = [input_image.size(self._spatial_dims[0]), input_image.size(self._spatial_dims[1])]
        state_size = [batch_size, self.hidden_channels] + list(spatial_shape) + [self.depth]
        if previous_state is None:
            previous_state = torch.zeros(*state_size, dtype=input_image.dtype)
        cell_outputs = []
        intermediate_image = input_image
        for cell_idx in range(self.length):
            cell = self.cell_list[cell_idx] if self.no_parameter_sharing else self.cell_list[0]
            grad_loglikelihood = self.grad_likelihood(intermediate_image, masked_kspace, sensitivity_map, sampling_mask, loglikelihood_scaling)
            if grad_loglikelihood.abs().max() > 150.0:
                warnings.warn(f'Very large values for the gradient loglikelihood ({grad_loglikelihood.abs().max()}). Might cause difficulties.')
            cell_input = torch.cat([intermediate_image, grad_loglikelihood], dim=1)
            cell_output, previous_state = cell(cell_input, previous_state)
            if self.skip_connections:
                intermediate_image = intermediate_image + cell_output
            else:
                intermediate_image = cell_output
            if not self.training:
                cell_output.set_()
                grad_loglikelihood.set_()
                del cell_output, grad_loglikelihood
            if self.training or cell_idx == self.length - 1:
                cell_outputs.append(intermediate_image)
        return cell_outputs, previous_state


class Unet2d(nn.Module):
    """PyTorch implementation of a U-Net model for MRI Reconstruction."""

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_filters: int, num_pool_layers: int, dropout_probability: float, skip_connection: bool=False, normalized: bool=False, image_initialization: str='zero_filled', **kwargs):
        """Inits :class:`Unet2d`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_filters: int
            Number of first layer filters.
        num_pool_layers: int
            Number of pooling layers.
        dropout_probability: float
            Dropout probability.
        skip_connection: bool
            If True, skip connection is used for the output. Default: False.
        normalized: bool
            If True, Normalized Unet is used. Default: False.
        image_initialization: str
            Type of image initialization. Default: "zero-filled".
        kwargs: dict
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in ['sensitivity_map_model', 'model_name']:
                raise ValueError(f'{type(self).__name__} got key `{extra_key}` which is not supported.')
        self.unet: nn.Module
        if normalized:
            self.unet = NormUnetModel2d(in_channels=2, out_channels=2, num_filters=num_filters, num_pool_layers=num_pool_layers, dropout_probability=dropout_probability)
        else:
            self.unet = UnetModel2d(in_channels=2, out_channels=2, num_filters=num_filters, num_pool_layers=num_pool_layers, dropout_probability=dropout_probability)
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.skip_connection = skip_connection
        self.image_initialization = image_initialization
        self._coil_dim = 1
        self._spatial_dims = 2, 3

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Computes sense initialization :math:`x_{\\text{SENSE}}`:

        .. math::
            x_{\\text{SENSE}} = \\sum_{k=1}^{n_c} {S^{k}}^* \\times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\\text{SENSE}}`.
        """
        input_image = T.complex_multiplication(T.conjugate(sensitivity_map), self.backward_operator(kspace, dim=self._spatial_dims))
        input_image = input_image.sum(self._coil_dim)
        return input_image

    def forward(self, masked_kspace: torch.Tensor, sensitivity_map: Optional[torch.Tensor]=None) ->torch.Tensor:
        """Computes forward pass of Unet2d.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2). Default: None.

        Returns
        -------
        output: torch.Tensor
            Output image of shape (N, height, width, complex=2).
        """
        if self.image_initialization == 'sense':
            if sensitivity_map is None:
                raise ValueError("Expected sensitivity_map not to be None with 'sense' image_initialization.")
            input_image = self.compute_sense_init(kspace=masked_kspace, sensitivity_map=sensitivity_map)
        elif self.image_initialization == 'zero_filled':
            input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims).sum(self._coil_dim)
        else:
            raise ValueError(f'Unknown image_initialization. Expected `sense` or `zero_filled`. Got {self.image_initialization}.')
        output = self.unet(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if self.skip_connection:
            output += input_image
        return output


class EndToEndVarNetBlock(nn.Module):
    """End-to-End Variational Network block."""

    def __init__(self, forward_operator: Callable, backward_operator: Callable, regularizer_model: nn.Module):
        """Inits :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        regularizer_model: nn.Module
            Regularizer model.
        """
        super().__init__()
        self.regularizer_model = regularizer_model
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        self._coil_dim = 1
        self._complex_dim = -1
        self._spatial_dims = 2, 3

    def forward(self, current_kspace: torch.Tensor, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNetBlock`.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        torch.Tensor
            Next k-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_error = torch.where(sampling_mask == 0, torch.tensor([0.0], dtype=masked_kspace.dtype), current_kspace - masked_kspace)
        regularization_term = torch.cat([reduce_operator(self.backward_operator(kspace, dim=self._spatial_dims), sensitivity_map, dim=self._coil_dim) for kspace in torch.split(current_kspace, 2, self._complex_dim)], dim=self._complex_dim).permute(0, 3, 1, 2)
        regularization_term = self.regularizer_model(regularization_term).permute(0, 2, 3, 1)
        regularization_term = torch.cat([self.forward_operator(expand_operator(image, sensitivity_map, dim=self._coil_dim), dim=self._spatial_dims) for image in torch.split(regularization_term, 2, self._complex_dim)], dim=self._complex_dim)
        return current_kspace - self.learning_rate * kspace_error + regularization_term


class EndToEndVarNet(nn.Module):
    """End-to-End Variational Network based on [1]_.

    References
    ----------

    .. [1] Sriram, Anuroop, et al. “End-to-End Variational Networks for Accelerated MRI Reconstruction.”
        ArXiv:2004.06688 [Cs, Eess], Apr. 2020. arXiv.org, http://arxiv.org/abs/2004.06688.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_layers: int, regularizer_num_filters: int=18, regularizer_num_pull_layers: int=4, regularizer_dropout: float=0.0, in_channels: int=2, **kwargs):
        """Inits :class:`EndToEndVarNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_layers: int
            Number of cascades.
        regularizer_num_filters: int
            Regularizer model number of filters.
        regularizer_num_pull_layers: int
            Regularizer model number of pulling layers.
        regularizer_dropout: float
            Regularizer model dropout probability.
        """
        super().__init__()
        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in ['model_name']:
                raise ValueError(f'{type(self).__name__} got key `{extra_key}` which is not supported.')
        self.layers_list = nn.ModuleList()
        for _ in range(num_layers):
            self.layers_list.append(EndToEndVarNetBlock(forward_operator=forward_operator, backward_operator=backward_operator, regularizer_model=UnetModel2d(in_channels=in_channels, out_channels=in_channels, num_filters=regularizer_num_filters, num_pool_layers=regularizer_num_pull_layers, dropout_probability=regularizer_dropout)))

    def forward(self, masked_kspace: torch.Tensor, sampling_mask: torch.Tensor, sensitivity_map: torch.Tensor) ->torch.Tensor:
        """Performs the forward pass of :class:`EndToEndVarNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        kspace_prediction: torch.Tensor
            K-space prediction of shape (N, coil, height, width, complex=2).
        """
        kspace_prediction = masked_kspace.clone()
        for layer in self.layers_list:
            kspace_prediction = layer(kspace_prediction, masked_kspace, sampling_mask, sensitivity_map)
        return kspace_prediction


class XPDNet(CrossDomainNetwork):
    """XPDNet as implemented in [1]_.

    References
    ----------

    .. [1] Ramzi, Zaccharie, et al. “XPDNet for MRI Reconstruction: An Application to the 2020 FastMRI Challenge.” ArXiv:2010.07290 [Physics, Stat], July 2021. arXiv.org, http://arxiv.org/abs/2010.07290.
    """

    def __init__(self, forward_operator: Callable, backward_operator: Callable, num_primal: int=5, num_dual: int=1, num_iter: int=10, use_primal_only: bool=True, image_model_architecture: str='MWCNN', kspace_model_architecture: Optional[str]=None, normalize: bool=False, **kwargs):
        """Inits :class:`XPDNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_primal: int
            Number of primal networks.
        num_dual: int
            Number of dual networks.
        num_iter: int
            Number of unrolled iterations.
        use_primal_only: bool
            If set to True no dual-kspace model is used. Default: True.
        image_model_architecture: str
            Primal-image model architecture. Currently only implemented for MWCNN. Default: 'MWCNN'.
        kspace_model_architecture: str
            Dual-kspace model architecture. Currently only implemented for CONV and DIDN.
        normalize: bool
            Normalize input. Default: False.
        kwargs: dict
            Keyword arguments for model architectures.
        """
        if use_primal_only:
            kspace_model_list = None
            num_dual = 1
        elif kspace_model_architecture == 'CONV':
            kspace_model_list = nn.ModuleList([MultiCoil(Conv2d(2 * (num_dual + num_primal + 1), 2 * num_dual, kwargs.get('dual_conv_hidden_channels', 16), kwargs.get('dual_conv_n_convs', 4), batchnorm=kwargs.get('dual_conv_batchnorm', False))) for _ in range(num_iter)])
        elif kspace_model_architecture == 'DIDN':
            kspace_model_list = nn.ModuleList([MultiCoil(DIDN(in_channels=2 * (num_dual + num_primal + 1), out_channels=2 * num_dual, hidden_channels=kwargs.get('dual_didn_hidden_channels', 16), num_dubs=kwargs.get('dual_didn_num_dubs', 6), num_convs_recon=kwargs.get('dual_didn_num_convs_recon', 9))) for _ in range(num_iter)])
        else:
            raise NotImplementedError(f"XPDNet is currently implemented for kspace_model_architecture == 'CONV' or 'DIDN'.Got kspace_model_architecture == {kspace_model_architecture}.")
        if image_model_architecture == 'MWCNN':
            image_model_list = nn.ModuleList([nn.Sequential(MWCNN(input_channels=2 * (num_primal + num_dual), first_conv_hidden_channels=kwargs.get('mwcnn_hidden_channels', 32), num_scales=kwargs.get('mwcnn_num_scales', 4), bias=kwargs.get('mwcnn_bias', False), batchnorm=kwargs.get('mwcnn_batchnorm', False)), nn.Conv2d(2 * (num_primal + num_dual), 2 * num_primal, kernel_size=3, padding=1)) for _ in range(num_iter)])
        else:
            raise NotImplementedError(f"XPDNet is currently implemented only with image_model_architecture == 'MWCNN'.Got {image_model_architecture}.")
        super().__init__(forward_operator=forward_operator, backward_operator=backward_operator, image_model_list=image_model_list, kspace_model_list=kspace_model_list, domain_sequence='KI' * num_iter, image_buffer_size=num_primal, kspace_buffer_size=num_dual, normalize_image=normalize)


class DirectTransform:
    """Direct transform class.

    Defines :meth:`__repr__` method for Direct transforms.
    """

    def __init__(self):
        """Inits DirectTransform."""
        super().__init__()

    def __repr__(self):
        """Representation of DirectTransform."""
        repr_string = self.__class__.__name__ + '('
        for k, v in self.__dict__.items():
            if k == 'logger':
                continue
            repr_string += f'{k}='
            if callable(v):
                if hasattr(v, '__class__'):
                    repr_string += type(v).__name__ + ', '
                else:
                    repr_string += str(v) + ', '
            elif isinstance(v, (dict, OrderedDict)):
                repr_string += f'{k}=dict(len={len(v)}), '
            elif isinstance(v, list):
                repr_string = f'{k}=list(len={len(v)}), '
            elif isinstance(v, tuple):
                repr_string = f'{k}=tuple(len={len(v)}), '
            else:
                repr_string += str(v) + ', '
        if repr_string[-2:] == ', ':
            repr_string = repr_string[:-2]
        return repr_string + ')'


class DirectModule(torch.nn.Module, DirectTransform, abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(self, sample: Dict):
        pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CIRIM,
     lambda: ([], {'forward_operator': 4, 'backward_operator': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dropout_probability': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNonlinear,
     lambda: ([], {'input_size': 4, 'features': 4, 'kernel_size': 4, 'dilation': 1, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DIDN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DUB,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilatedConvBlock,
     lambda: ([], {'in_channels': 4, 'dilations': [4, 4], 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (IWT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IndRNNCell,
     lambda: ([], {'in_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64])], {}),
     True),
    (MultiCoil,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NMAELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NRMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormUnetModel2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'num_filters': 4, 'num_pool_layers': 1, 'dropout_probability': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RIMInit,
     lambda: ([], {'x_ch': 4, 'out_ch': 4, 'channels': [4, 4], 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReconBlock,
     lambda: ([], {'in_channels': 4, 'num_convs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RecurrentInit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'channels': [4, 4], 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNet,
     lambda: ([], {'hidden_channels': 4}),
     lambda: ([torch.rand([4, 2, 64, 64])], {}),
     True),
    (ResNetBlock,
     lambda: ([], {'in_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSIMLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 1])], {}),
     True),
    (SobelGradL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SobelGradL2Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SobelGradLoss,
     lambda: ([], {'type_loss': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransposeConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetModel2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'num_filters': 4, 'num_pool_layers': 1, 'dropout_probability': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NKI_AI_direct(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

