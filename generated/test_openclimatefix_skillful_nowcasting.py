import sys
_module = sys.modules[__name__]
del sys
dgmr = _module
common = _module
dgmr = _module
discriminators = _module
generators = _module
hub = _module
Attention = _module
ConvGRU = _module
CoordConv = _module
layers = _module
utils = _module
losses = _module
setup = _module
test_losses = _module
test_model = _module
run = _module

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


from typing import Tuple


import torch


import torch.nn.functional as F


from torch.distributions import normal


from torch.nn.utils.parametrizations import spectral_norm


from torch.nn.modules.pixelshuffle import PixelUnshuffle


import torchvision


from torch.nn.modules.pixelshuffle import PixelShuffle


from typing import List


import logging


from functools import partial


import torch.nn as nn


from torch.nn import functional as F


import numpy as np


import torch.utils.data.dataset


from torch.utils.data import DataLoader


from numpy.random import default_rng


import tensorflow as tf


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


def get_conv_layer(conv_type: str='standard') ->torch.nn.Module:
    if conv_type == 'standard':
        conv_layer = torch.nn.Conv2d
    elif conv_type == 'coord':
        conv_layer = CoordConv
    elif conv_type == '3d':
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f'{conv_type} is not a recognized Conv method')
    return conv_layer


class GBlock(torch.nn.Module):
    """Residual generator block without upsampling"""

    def __init__(self, input_channels: int=12, output_channels: int=12, conv_type: str='standard', spectral_normalized_eps=0.0001):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1), eps=spectral_normalized_eps)
        self.first_conv_3x3 = spectral_norm(conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1), eps=spectral_normalized_eps)
        self.last_conv_3x3 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1), eps=spectral_normalized_eps)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if x.shape[1] != self.output_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + sc
        return x


class UpsampleGBlock(torch.nn.Module):
    """Residual generator block with upsampling"""

    def __init__(self, input_channels: int=12, output_channels: int=12, conv_type: str='standard', spectral_normalized_eps=0.0001):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.output_channels = output_channels
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1), eps=spectral_normalized_eps)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.first_conv_3x3 = spectral_norm(conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1), eps=spectral_normalized_eps)
        self.last_conv_3x3 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1), eps=spectral_normalized_eps)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        sc = self.upsample(x)
        sc = self.conv_1x1(sc)
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.first_conv_3x3(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + sc
        return x


class DBlock(torch.nn.Module):

    def __init__(self, input_channels: int=12, output_channels: int=12, conv_type: str='standard', first_relu: bool=True, keep_same_output: bool=False):
        """
        D and 3D Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Convolution type, see satflow/models/utils.py for options
            first_relu: Whether to have an ReLU before the first 3x3 convolution
            keep_same_output: Whether the output should have the same spatial dimensions as input, if False, downscales by 2
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == '3d':
            self.pooling = torch.nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1))
        self.first_conv_3x3 = spectral_norm(conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1))
        self.last_conv_3x3 = spectral_norm(conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1))
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.input_channels != self.output_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x
        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)
        if not self.keep_same_output:
            x = self.pooling(x)
        x = x1 + x
        return x


class LBlock(torch.nn.Module):
    """Residual block for the Latent Stack."""

    def __init__(self, input_channels: int=12, output_channels: int=12, kernel_size: int=3, conv_type: str='standard'):
        """
        L-Block for increasing the number of channels in the input
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(in_channels=input_channels, out_channels=output_channels - input_channels, kernel_size=1)
        self.first_conv_3x3 = conv2d(input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.last_conv_3x3 = conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=1, stride=1)

    def forward(self, x) ->torch.Tensor:
        if self.input_channels < self.output_channels:
            sc = self.conv_1x1(x)
            sc = torch.cat([x, sc], dim=1)
        else:
            sc = x
        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        return x2 + sc


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""
    k = einops.rearrange(k, 'h w c -> (h w) c')
    v = einops.rearrange(v, 'h w c -> (h w) c')
    beta = F.softmax(torch.einsum('hwc, Lc->hwL', q, k), dim=-1)
    out = torch.einsum('hwL, Lc->hwc', beta, v)
    return out


class AttentionLayer(torch.nn.Module):
    """Attention Module"""

    def __init__(self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8):
        super(AttentionLayer, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.query = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.output_channels // self.ratio_kq, kernel_size=(1, 1), padding='valid', bias=False)
        self.key = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.output_channels // self.ratio_kq, kernel_size=(1, 1), padding='valid', bias=False)
        self.value = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.output_channels // self.ratio_v, kernel_size=(1, 1), padding='valid', bias=False)
        self.last_conv = torch.nn.Conv2d(in_channels=self.output_channels // 8, out_channels=self.output_channels, kernel_size=(1, 1), padding='valid', bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        out = []
        for b in range(x.shape[0]):
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = torch.stack(out, dim=0)
        out = self.gamma * self.last_conv(out)
        return out + x


class ConvGRUCell(torch.nn.Module):
    """A ConvGRU implementation."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size=3, sn_eps=0.0001):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps
        self.read_gate_conv = spectral_norm(torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(kernel_size, kernel_size), padding=1), eps=sn_eps)
        self.update_gate_conv = spectral_norm(torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(kernel_size, kernel_size), padding=1), eps=sn_eps)
        self.output_conv = spectral_norm(torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(kernel_size, kernel_size), padding=1), eps=sn_eps)

    def forward(self, x, prev_state):
        """
        ConvGRU forward, returning the current+new state

        Args:
            x: Input tensor
            prev_state: Previous state

        Returns:
            New tensor plus the new state
        """
        xh = torch.cat([x, prev_state], dim=1)
        read_gate = F.sigmoid(self.read_gate_conv(xh))
        update_gate = F.sigmoid(self.update_gate_conv(xh))
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)
        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out
        return out, new_state


class ConvGRU(torch.nn.Module):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation"""

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int=3, sn_eps=0.0001):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size, sn_eps)

    def forward(self, x: torch.Tensor, hidden_state=None) ->torch.Tensor:
        outputs = []
        for step in range(len(x)):
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs


class SSIMLoss(nn.Module):

    def __init__(self, convert_range: bool=False, **kwargs):
        """
        SSIM Loss, optionally converting input range from [-1,1] to [0,1]
        Args:
            convert_range:
            **kwargs:
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):

    def __init__(self, convert_range: bool=False, **kwargs):
        """
        Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1]
        Args:
            convert_range:
            **kwargs:
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class SSIMLossDynamic(nn.Module):

    def __init__(self, convert_range: bool=False, **kwargs):
        """
        SSIM Loss on only dynamic part of the images, optionally converting input range from [-1,1] to [0,1]

        In Mathieu et al. to stop SSIM regressing towards the mean and predicting only the background, they only
        run SSIM on the dynamic parts of the image. We can accomplish that by subtracting the current image from the future ones

        Args:
            convert_range:
            **kwargs:
        """
        super(SSIMLossDynamic, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, curr_image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            curr_image = torch.div(torch.add(curr_image, 1), 2)
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        x = x - curr_image
        y = y - curr_image
        return 1.0 - self.ssim_module(x, y)


def tv_loss(img, tv_weight):
    """
    Taken from https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class TotalVariationLoss(nn.Module):

    def __init__(self, tv_weight: float=1.0):
        super(TotalVariationLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x: torch.Tensor):
        return tv_loss(x, self.tv_weight)


class GradientDifferenceLoss(nn.Module):
    """"""

    def __init__(self, alpha: int=2):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        t1 = torch.pow(torch.abs(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]) - torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])), self.alpha)
        t2 = torch.pow(torch.abs(torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:]) - torch.abs(y[:, :, :, :, :-1] - y[:, :, :, :, 1:])), self.alpha)
        loss = t1 + t2
        None
        return loss


class GridCellLoss(nn.Module):
    """
    Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

    """

    def __init__(self, weight_fn=None):
        super().__init__()
        self.weight_fn = weight_fn

    def forward(self, generated_images, targets):
        """
        Calculates the grid cell regularizer value, assumes generated images are the mean predictions from
        6 calls to the generater (Monte Carlo estimation of the expectations for the latent variable)
        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            difference *= self.weight_fn(targets)
        difference /= targets.size(1) * targets.size(3) * targets.size(4)
        return difference.mean()


class NowcastingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, real_flag):
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-05, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        if alpha.device != logit.device:
            alpha = alpha
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow(1 - pt, gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AddCoords,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 2, 2])], {}),
     False),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([16, 4]), torch.rand([4, 4])], {}),
     True),
    (GBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (GridCellLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     True),
    (NowcastingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TotalVariationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UpsampleGBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 4, 4])], {}),
     False),
]

class Test_openclimatefix_skillful_nowcasting(_paritybench_base):
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

