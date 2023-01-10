import sys
_module = sys.modules[__name__]
del sys
ckpt_manager = _module
config = _module
config_RefVSR_IR_L1 = _module
config_RefVSR_IR_MFID = _module
config_RefVSR_L1 = _module
config_RefVSR_MFID = _module
config_RefVSR_MFID_8K = _module
config_RefVSR_small_L1 = _module
config_RefVSR_small_MFID = _module
config_RefVSR_small_MFID_8K = _module
FastDataLoader = _module
data_sampler = _module
datasets = _module
utils = _module
eval = _module
eval_qual_quan = _module
eval_quan_FOV = _module
eval_quan_conf_map = _module
init = _module
metrics = _module
common = _module
aspp = _module
contextual_attention = _module
conv = _module
flow_warp = _module
gated_conv_module = _module
gca_module = _module
generation_model_utils = _module
img_normalize = _module
linear_module = _module
mask_conv_module = _module
model_utils = _module
partial_conv = _module
separable_conv_module = _module
sr_backbone_utils = _module
upsample = _module
collect_env = _module
logger = _module
SRNet = _module
RefVSR = _module
alignment = _module
attention = _module
common = _module
utils = _module
RefVSR_IR = _module
SPyNet = _module
edvr_net = _module
Loss = _module
contextual = _module
contextual_X_mu = _module
gaussian = _module
vgg = _module
utils = _module
replicate = _module
run = _module
trainers = _module
baseTrainer = _module
lr_scheduler = _module
trainer = _module

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


import numpy as np


import collections


import math


import torch.utils.data


from torch.utils.data.sampler import Sampler


import torch.distributed as dist


import random


import torch.utils.data as data


import torchvision.transforms.functional as TF


import torch.nn.functional as F


import torchvision.utils as vutils


from torch import nn


from torch.nn import functional as F


from functools import partial


import torch.nn as nn


import copy


from torch.nn import init


from torchvision import models


from torch.nn.modules.utils import _pair


import scipy.ndimage


import torchvision.models as models


from collections import namedtuple


import torchvision.models.vgg as vgg


import time


import torch.multiprocessing as mp


import numpy


import warnings


import torch.nn.utils as torch_utils


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.parallel import DataParallel as DP


from torch.nn.parallel import DistributedDataParallel as DDP


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__(nn.AdaptiveAvgPool2d(1), ConvModule(in_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if ``norm_cfg`` and ``act_cfg`` are specified.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        padding (int or tuple[int]): Same as nn.Conv2d. Default: 0.
        dilation (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as ``norm_cfg``. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as ``act_cfg``. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as ``act_cfg``. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_cfg=None, act_cfg=dict(type='ReLU'), dw_norm_cfg='default', dw_act_cfg='default', pw_norm_cfg='default', pw_act_cfg='default', **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, norm_cfg=dw_norm_cfg, act_cfg=dw_act_cfg, **kwargs)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=pw_norm_cfg, act_cfg=pw_act_cfg, **kwargs)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ASPP(nn.Module):
    """ASPP module from DeepLabV3.

    The code is adopted from
    https://github.com/pytorch/vision/blob/master/torchvision/models/
    segmentation/deeplabv3.py

    For more information about the module:
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        in_channels (int): Input channels of the module.
        out_channels (int): Output channels of the module.
        mid_channels (int): Output channels of the intermediate ASPP conv
            modules.
        dilations (Sequence[int]): Dilation rate of three ASPP conv module.
            Default: [12, 24, 36].
        conv_cfg (dict): Config dict for convolution layer. If "None",
            nn.Conv2d will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        separable_conv (bool): Whether replace normal conv with depthwise
            separable conv which is faster. Default: False.
    """

    def __init__(self, in_channels, out_channels=256, mid_channels=256, dilations=(12, 24, 36), conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), separable_conv=False):
        super().__init__()
        if separable_conv:
            conv_module = DepthwiseSeparableConvModule
        else:
            conv_module = ConvModule
        modules = []
        modules.append(ConvModule(in_channels, mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        for dilation in dilations:
            modules.append(conv_module(in_channels, mid_channels, 3, padding=dilation, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        modules.append(ASPPPooling(in_channels, mid_channels, conv_cfg, norm_cfg, act_cfg))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(ConvModule(5 * mid_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), nn.Dropout(0.5))

    def forward(self, x):
        """Forward function for ASPP module.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ContextualAttentionModule(nn.Module):
    """Contexture attention module.

    The details of this module can be found in:
    Generative Image Inpainting with Contextual Attention

    Args:
        unfold_raw_kernel_size (int): Kernel size used in unfolding raw
            feature. Default: 4.
        unfold_raw_stride (int): Stride used in unfolding raw feature. Default:
            2.
        unfold_raw_padding (int): Padding used in unfolding raw feature.
            Default: 1.
        unfold_corr_kernel_size (int): Kernel size used in unfolding
            context for computing correlation maps. Default: 3.
        unfold_corr_stride (int): Stride used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_dilation (int): Dilation used in unfolding context for
            computing correlation maps. Default: 1.
        unfold_corr_padding (int): Padding used in unfolding context for
            computing correlation maps. Default: 1.
        scale (float): The resale factor used in resize input features.
            Default: 0.5.
        fuse_kernel_size (int): The kernel size used in fusion module.
            Default: 3.
        softmax_scale (float): The scale factor for softmax function.
            Default: 10.
        return_attention_score (bool): If True, the attention score will be
            returned. Default: True.
    """

    def __init__(self, unfold_raw_kernel_size=4, unfold_raw_stride=2, unfold_raw_padding=1, unfold_corr_kernel_size=3, unfold_corr_stride=1, unfold_corr_dilation=1, unfold_corr_padding=1, scale=0.5, fuse_kernel_size=3, softmax_scale=10, return_attention_score=True):
        super().__init__()
        self.unfold_raw_kernel_size = unfold_raw_kernel_size
        self.unfold_raw_stride = unfold_raw_stride
        self.unfold_raw_padding = unfold_raw_padding
        self.unfold_corr_kernel_size = unfold_corr_kernel_size
        self.unfold_corr_stride = unfold_corr_stride
        self.unfold_corr_dilation = unfold_corr_dilation
        self.unfold_corr_padding = unfold_corr_padding
        self.scale = scale
        self.fuse_kernel_size = fuse_kernel_size
        self.with_fuse_correlation = fuse_kernel_size > 1
        self.softmax_scale = softmax_scale
        self.return_attention_score = return_attention_score
        if self.with_fuse_correlation:
            assert fuse_kernel_size % 2 == 1
            fuse_kernel = torch.eye(fuse_kernel_size).view(1, 1, fuse_kernel_size, fuse_kernel_size)
            self.register_buffer('fuse_kernel', fuse_kernel)
            padding = int((fuse_kernel_size - 1) // 2)
            self.fuse_conv = partial(F.conv2d, padding=padding, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, context, mask=None):
        """Forward Function.

        Args:
            x (torch.Tensor): Tensor with shape (n, c, h, w).
            context (torch.Tensor): Tensor with shape (n, c, h, w).
            mask (torch.Tensor): Tensor with shape (n, 1, h, w). Default: None.

        Returns:
            tuple(torch.Tensor): Features after contextural attention.
        """
        raw_context = context
        raw_context_cols = self.im2col(raw_context, kernel_size=self.unfold_raw_kernel_size, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding, normalize=False, return_cols=True)
        x = F.interpolate(x, scale_factor=self.scale)
        context = F.interpolate(context, scale_factor=self.scale)
        context_cols = self.im2col(context, kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation, normalize=True, return_cols=True)
        h_unfold, w_unfold = self.calculate_unfold_hw(context.size()[-2:], kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation)
        context_cols = context_cols.reshape(-1, *context_cols.shape[2:])
        correlation_map = self.patch_correlation(x, context_cols)
        if self.with_fuse_correlation:
            correlation_map = self.fuse_correlation_map(correlation_map, h_unfold, w_unfold)
        correlation_map = self.mask_correlation_map(correlation_map, mask=mask)
        attention_score = self.softmax(correlation_map * self.softmax_scale)
        raw_context_filter = raw_context_cols.reshape(-1, *raw_context_cols.shape[2:])
        output = self.patch_copy_deconv(attention_score, raw_context_filter)
        overlap_factor = self.calculate_overlap_factor(attention_score)
        output /= overlap_factor
        if self.return_attention_score:
            n, _, h_s, w_s = attention_score.size()
            attention_score = attention_score.view(n, h_unfold, w_unfold, h_s, w_s)
            return output, attention_score
        return output

    def patch_correlation(self, x, kernel):
        """Calculate patch correlation.

        Args:
            x (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.

        Returns:
            torch.Tensor: Tensor with shape of (n, l, h, w).
        """
        n, _, h_in, w_in = x.size()
        patch_corr = F.conv2d(x.view(1, -1, h_in, w_in), kernel, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation, groups=n)
        h_out, w_out = patch_corr.size()[-2:]
        return patch_corr.view(n, -1, h_out, w_out)

    def patch_copy_deconv(self, attention_score, context_filter):
        """Copy patches using deconv.

        Args:
            attention_score (torch.Tensor): Tensor with shape of (n, l , h, w).
            context_filter (torch.Tensor): Filter kernel.

        Returns:
            torch.Tensor: Tensor with shape of (n, c, h, w).
        """
        n, _, h, w = attention_score.size()
        attention_score = attention_score.view(1, -1, h, w)
        output = F.conv_transpose2d(attention_score, context_filter, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding, groups=n)
        h_out, w_out = output.size()[-2:]
        return output.view(n, -1, h_out, w_out)

    def fuse_correlation_map(self, correlation_map, h_unfold, w_unfold):
        """Fuse correlation map.

        This operation is to fuse correlation map for increasing large
        consistent correlation regions.

        The mechanism behind this op is simple and easy to understand. A
        standard 'Eye' matrix will be applied as a filter on the correlation
        map in horizontal and vertical direction.

        The shape of input correlation map is (n, h_unfold*w_unfold, h, w).
        When adopting fusing, we will apply convolutional filter in the
        reshaped feature map with shape of (n, 1, h_unfold*w_fold, h*w).

        A simple specification for horizontal direction is shown below:

        .. code-block:: python

                   (h, (h, (h, (h,
                    0)  1)  2)  3)  ...
            (h, 0)
            (h, 1)      1
            (h, 2)          1
            (h, 3)              1
            ...

        """
        n, _, h_map, w_map = correlation_map.size()
        map_ = correlation_map.permute(0, 2, 3, 1)
        map_ = map_.reshape(n, h_map * w_map, h_unfold * w_unfold, 1)
        map_ = map_.permute(0, 3, 1, 2).contiguous()
        map_ = self.fuse_conv(map_, self.fuse_kernel)
        correlation_map = map_.view(n, h_unfold, w_unfold, h_map, w_map)
        map_ = correlation_map.permute(0, 2, 1, 4, 3).reshape(n, 1, h_unfold * w_unfold, h_map * w_map)
        map_ = self.fuse_conv(map_, self.fuse_kernel)
        correlation_map = map_.view(n, w_unfold, h_unfold, w_map, h_map).permute(0, 4, 3, 2, 1)
        correlation_map = correlation_map.reshape(n, -1, h_unfold, w_unfold)
        return correlation_map

    def calculate_unfold_hw(self, input_size, kernel_size=3, stride=1, dilation=1, padding=0):
        """Calculate (h, w) after unfolding

        The official implementation of `unfold` in pytorch will put the
        dimension (h, w) into `L`. Thus, this function is just to calculate the
        (h, w) according to the equation in:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        """
        h_in, w_in = input_size
        h_unfold = int((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        w_unfold = int((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        return h_unfold, w_unfold

    def calculate_overlap_factor(self, attention_score):
        """Calculate the overlap factor after applying deconv.

        Args:
            attention_score (torch.Tensor): The attention score with shape of
                (n, c, h, w).

        Returns:
            torch.Tensor: The overlap factor will be returned.
        """
        h, w = attention_score.shape[-2:]
        kernel_size = self.unfold_raw_kernel_size
        ones_input = torch.ones(1, 1, h, w)
        ones_filter = torch.ones(1, 1, kernel_size, kernel_size)
        overlap = F.conv_transpose2d(ones_input, ones_filter, stride=self.unfold_raw_stride, padding=self.unfold_raw_padding)
        overlap[overlap == 0] = 1.0
        return overlap

    def mask_correlation_map(self, correlation_map, mask):
        """Add mask weight for correlation map.

        Add a negative infinity number to the masked regions so that softmax
        function will result in 'zero' in those regions.

        Args:
            correlation_map (torch.Tensor): Correlation map with shape of
                (n, h_unfold*w_unfold, h_map, w_map).
            mask (torch.Tensor): Mask tensor with shape of (n, c, h, w). '1'
                in the mask indicates masked region while '0' indicates valid
                region.

        Returns:
            torch.Tensor: Updated correlation map with mask.
        """
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=self.scale)
            mask_cols = self.im2col(mask, kernel_size=self.unfold_corr_kernel_size, stride=self.unfold_corr_stride, padding=self.unfold_corr_padding, dilation=self.unfold_corr_dilation)
            mask_cols = (mask_cols.sum(dim=1, keepdim=True) > 0).float()
            mask_cols = mask_cols.permute(0, 2, 1).reshape(mask.size(0), -1, 1, 1)
            mask_cols[mask_cols == 1] = -float('inf')
            correlation_map += mask_cols
        return correlation_map

    def im2col(self, img, kernel_size, stride=1, padding=0, dilation=1, normalize=False, return_cols=False):
        """Reshape image-style feature to columns.

        This function is used for unfold feature maps to columns. The
        details of this function can be found in:
        https://pytorch.org/docs/1.1.0/nn.html?highlight=unfold#torch.nn.Unfold

        Args:
            img (torch.Tensor): Features to be unfolded. The shape of this
                feature should be (n, c, h, w).
            kernel_size (int): In this function, we only support square kernel
                with same height and width.
            stride (int): Stride number in unfolding. Default: 1.
            padding (int): Padding number in unfolding. Default: 0.
            dilation (int): Dilation number in unfolding. Default: 1.
            normalize (bool): If True, the unfolded feature will be normalized.
                Default: False.
            return_cols (bool): The official implementation in PyTorch of
                unfolding will return features with shape of
                (n, c*$prod{kernel_size}$, L). If True, the features will be
                reshaped to (n, L, c, kernel_size, kernel_size). Otherwise,
                the results will maintain the shape as the official
                implementation.

        Returns:
            torch.Tensor: Unfolded columns. If `return_cols` is True, the                 shape of output tensor is                 `(n, L, c, kernel_size, kernel_size)`. Otherwise, the shape                 will be `(n, c*$prod{kernel_size}$, L)`.
        """
        img_unfold = F.unfold(img, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if normalize:
            norm = torch.sqrt((img_unfold ** 2).sum(dim=1, keepdim=True))
            eps = torch.tensor([0.0001])
            img_unfold = img_unfold / torch.max(norm, eps)
        if return_cols:
            img_unfold_ = img_unfold.permute(0, 2, 1)
            n, num_cols = img_unfold_.size()[:2]
            img_cols = img_unfold_.view(n, num_cols, img.size(1), kernel_size, kernel_size)
            return img_cols
        return img_unfold


class SimpleGatedConvModule(nn.Module):
    """Simple Gated Convolutional Module.

    This module is a simple gated convolutional module. The detailed formula
    is:

    .. math::
        y = \\phi(conv1(x)) * \\sigma(conv2(x)),

    where `phi` is the feature activation function and `sigma` is the gate
    activation function. In default, the gate activation function is sigmoid.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): The number of channels of the output feature. Note
            that `out_channels` in the conv module is doubled since this module
            contains two convolutions for feature and gate seperately.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        feat_act_cfg (dict): Config dict for feature activation layer.
        gate_act_cfg (dict): Config dict for gate activation layer.
        kwargs (keyword arguments): Same as `ConvModule`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, feat_act_cfg=dict(type='ELU'), gate_act_cfg=dict(type='Sigmoid'), **kwargs):
        super().__init__()
        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['act_cfg'] = None
        self.with_feat_act = feat_act_cfg is not None
        self.with_gate_act = gate_act_cfg is not None
        self.conv = ConvModule(in_channels, out_channels * 2, kernel_size, **kwargs_)
        if self.with_feat_act:
            self.feat_act = build_activation_layer(feat_act_cfg)
        if self.with_gate_act:
            self.gate_act = build_activation_layer(gate_act_cfg)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.conv(x)
        x, gate = torch.split(x, x.size(1) // 2, dim=1)
        if self.with_feat_act:
            x = self.feat_act(x)
        if self.with_gate_act:
            gate = self.gate_act(gate)
        x = x * gate
        return x


class GCAModule(nn.Module):
    """Guided Contextual Attention Module.

    From https://arxiv.org/pdf/2001.04069.pdf.
    Based on https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting.
    This module use image feature map to augment the alpha feature map with
    guided contextual attention score.

    Image feature and alpha feature are unfolded to small patches and later
    used as conv kernel. Thus, we refer the unfolding size as kernel size.
    Image feature patches have a default kernel size 3 while the kernel size of
    alpha feature patches could be specified by `rate` (see `rate` below). The
    image feature patches are used to convolve with the image feature itself
    to calculate the contextual attention. Then the attention feature map is
    convolved by alpha feature patches to obtain the attention alpha feature.
    At last, the attention alpha feature is added to the input alpha feature.

    Args:
        in_channels (int): Input channels of the guided contextual attention
            module.
        out_channels (int): Output channels of the guided contextual attention
            module.
        kernel_size (int): Kernel size of image feature patches. Default 3.
        stride (int): Stride when unfolding the image feature. Default 1.
        rate (int): The downsample rate of image feature map. The corresponding
            kernel size and stride of alpha feature patches will be `rate x 2`
            and `rate`. It could be regarded as the granularity of the gca
            module. Default: 2.
        pad_args (dict): Parameters of padding when convolve image feature with
            image feature patches or alpha feature patches. Allowed keys are
            `mode` and `value`. See torch.nn.functional.pad() for more
            information. Default: dict(mode='reflect').
        interpolation (str): Interpolation method in upsampling and
            downsampling.
        penalty (float): Punishment hyperparameter to avoid a large correlation
            between each unknown patch and itself.
        eps (float): A small number to avoid dividing by 0 when calculating
            the normed image feature patch. Default: 1e-4.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, rate=2, pad_args=dict(mode='reflect'), interpolation='nearest', penalty=-10000.0, eps=0.0001):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.pad_args = pad_args
        self.interpolation = interpolation
        self.penalty = penalty
        self.eps = eps
        self.guidance_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out_conv = ConvModule(out_channels, out_channels, 1, norm_cfg=dict(type='BN'), act_cfg=None)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.guidance_conv, distribution='uniform')
        xavier_init(self.out_conv.conv, distribution='uniform')
        constant_init(self.out_conv.norm, 0.001)

    def forward(self, img_feat, alpha_feat, unknown=None, softmax_scale=1.0):
        """Forward function of GCAModule.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, ori_c, ori_h, ori_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap.
                If specified, this tensor should have shape
                (N, 1, ori_h, ori_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            Tensor: The augmented alpha feature.
        """
        if alpha_feat.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(f'image feature size does not align with alpha feature size: image feature size {img_feat.shape[2:4]}, alpha feature size {alpha_feat.shape[2:4]}')
        if unknown is not None and unknown.shape[2:4] != img_feat.shape[2:4]:
            raise ValueError(f'image feature size does not align with unknown mask size: image feature size {img_feat.shape[2:4]}, unknown mask size {unknown.shape[2:4]}')
        img_feat = self.guidance_conv(img_feat)
        img_feat = F.interpolate(img_feat, scale_factor=1 / self.rate, mode=self.interpolation)
        unknown, softmax_scale = self.process_unknown_mask(unknown, img_feat, softmax_scale)
        img_ps, alpha_ps, unknown_ps = self.extract_feature_maps_patches(img_feat, alpha_feat, unknown)
        self_mask = self.get_self_correlation_mask(img_feat)
        img_groups = torch.split(img_feat, 1, dim=0)
        img_ps_groups = torch.split(img_ps, 1, dim=0)
        alpha_ps_groups = torch.split(alpha_ps, 1, dim=0)
        unknown_ps_groups = torch.split(unknown_ps, 1, dim=0)
        scale_groups = torch.split(softmax_scale, 1, dim=0)
        groups = img_groups, img_ps_groups, alpha_ps_groups, unknown_ps_groups, scale_groups
        out = []
        for img_i, img_ps_i, alpha_ps_i, unknown_ps_i, scale_i in zip(*groups):
            similarity_map = self.compute_similarity_map(img_i, img_ps_i)
            gca_score = self.compute_guided_attention_score(similarity_map, unknown_ps_i, scale_i, self_mask)
            out_i = self.propagate_alpha_feature(gca_score, alpha_ps_i)
            out.append(out_i)
        out = torch.cat(out, dim=0)
        out.reshape_as(alpha_feat)
        out = self.out_conv(out) + alpha_feat
        return out

    def extract_feature_maps_patches(self, img_feat, alpha_feat, unknown):
        """Extract image feature, alpha feature unknown patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (N, img_c, img_h, img_w).
            alpha_feat (Tensor): Alpha feature map of shape
                (N, alpha_c, ori_h, ori_w).
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, img_h, img_w).

        Returns:
            tuple: 3-tuple of

                ``Tensor``: Image feature patches of shape                     (N, img_h*img_w, img_c, img_ks, img_ks).

                ``Tensor``: Guided contextual attention alpha feature map.                     (N, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

                ``Tensor``: Unknown mask of shape (N, img_h*img_w, 1, 1).
        """
        img_ks = self.kernel_size
        img_ps = self.extract_patches(img_feat, img_ks, self.stride)
        alpha_ps = self.extract_patches(alpha_feat, self.rate * 2, self.rate)
        unknown_ps = self.extract_patches(unknown, img_ks, self.stride)
        unknown_ps = unknown_ps.squeeze(dim=2)
        unknown_ps = unknown_ps.mean(dim=[2, 3], keepdim=True)
        return img_ps, alpha_ps, unknown_ps

    def compute_similarity_map(self, img_feat, img_ps):
        """Compute similarity between image feature patches.

        Args:
            img_feat (Tensor): Image feature map of shape
                (1, img_c, img_h, img_w).
            img_ps (Tensor): Image feature patches tensor of shape
                (1, img_h*img_w, img_c, img_ks, img_ks).

        Returns:
            Tensor: Similarity map between image feature patches with shape                 (1, img_h*img_w, img_h, img_w).
        """
        img_ps = img_ps[0]
        escape_NaN = torch.FloatTensor([self.eps])
        img_ps_normed = img_ps / torch.max(self.l2_norm(img_ps), escape_NaN)
        img_feat = self.pad(img_feat, self.kernel_size, self.stride)
        similarity_map = F.conv2d(img_feat, img_ps_normed)
        return similarity_map

    def compute_guided_attention_score(self, similarity_map, unknown_ps, scale, self_mask):
        """Compute guided attention score.

        Args:
            similarity_map (Tensor): Similarity map of image feature with shape
                (1, img_h*img_w, img_h, img_w).
            unknown_ps (Tensor): Unknown area patches tensor of shape
                (1, img_h*img_w, 1, 1).
            scale (Tensor): Softmax scale of known and unknown area:
                [unknown_scale, known_scale].
            self_mask (Tensor): Self correlation mask of shape
                (1, img_h*img_w, img_h, img_w). At (1, i*i, i, i) mask value
                equals -1e4 for i in [1, img_h*img_w] and other area is all
                zero.

        Returns:
            Tensor: Similarity map between image feature patches with shape                 (1, img_h*img_w, img_h, img_w).
        """
        unknown_scale, known_scale = scale[0]
        out = similarity_map * (unknown_scale * unknown_ps.gt(0.0).float() + known_scale * unknown_ps.le(0.0).float())
        out = out + self_mask * unknown_ps
        gca_score = F.softmax(out, dim=1)
        return gca_score

    def propagate_alpha_feature(self, gca_score, alpha_ps):
        """Propagate alpha feature based on guided attention score.

        Args:
            gca_score (Tensor): Guided attention score map of shape
                (1, img_h*img_w, img_h, img_w).
            alpha_ps (Tensor): Alpha feature patches tensor of shape
                (1, img_h*img_w, alpha_c, alpha_ks, alpha_ks).

        Returns:
            Tensor: Propagated alpha feature map of shape                 (1, alpha_c, alpha_h, alpha_w).
        """
        alpha_ps = alpha_ps[0]
        if self.rate == 1:
            gca_score = self.pad(gca_score, kernel_size=2, stride=1)
            alpha_ps = alpha_ps.permute(1, 0, 2, 3)
            out = F.conv2d(gca_score, alpha_ps) / 4.0
        else:
            out = F.conv_transpose2d(gca_score, alpha_ps, stride=self.rate, padding=1) / 4.0
        return out

    def process_unknown_mask(self, unknown, img_feat, softmax_scale):
        """Process unknown mask.

        Args:
            unknown (Tensor, optional): Unknown area map generated by trimap of
                shape (N, 1, ori_h, ori_w)
            img_feat (Tensor): The interpolated image feature map of shape
                (N, img_c, img_h, img_w).
            softmax_scale (float, optional): The softmax scale of the attention
                if unknown area is not provided in forward. Default: 1.

        Returns:
            tuple: 2-tuple of

                ``Tensor``: Interpolated unknown area map of shape                     (N, img_h*img_w, img_h, img_w).

                ``Tensor``: Softmax scale tensor of known and unknown area of                     shape (N, 2).
        """
        n, _, h, w = img_feat.shape
        if unknown is not None:
            unknown = unknown.clone()
            unknown = F.interpolate(unknown, scale_factor=1 / self.rate, mode=self.interpolation)
            unknown_mean = unknown.mean(dim=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(torch.sqrt(unknown_mean / known_mean), 0.1, 10)
            known_scale = torch.clamp(torch.sqrt(known_mean / unknown_mean), 0.1, 10)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones((n, 1, h, w))
            softmax_scale = torch.FloatTensor([softmax_scale, softmax_scale]).view(1, 2).repeat(n, 1)
        return unknown, softmax_scale

    def extract_patches(self, x, kernel_size, stride):
        """Extract feature patches.

        The feature map will be padded automatically to make sure the number of
        patches is equal to `(H / stride) * (W / stride)`.

        Args:
            x (Tensor): Feature map of shape (N, C, H, W).
            kernel_size (int): Size of each patches.
            stride (int): Stride between patches.

        Returns:
            Tensor: Extracted patches of shape                 (N, (H / stride) * (W / stride) , C, kernel_size, kernel_size).
        """
        n, c, _, _ = x.shape
        x = self.pad(x, kernel_size, stride)
        x = F.unfold(x, (kernel_size, kernel_size), stride=(stride, stride))
        x = x.permute(0, 2, 1)
        x = x.reshape(n, -1, c, kernel_size, kernel_size)
        return x

    def pad(self, x, kernel_size, stride):
        left = (kernel_size - stride + 1) // 2
        right = (kernel_size - stride) // 2
        pad = left, right, left, right
        return F.pad(x, pad, **self.pad_args)

    def get_self_correlation_mask(self, img_feat):
        _, _, h, w = img_feat.shape
        self_mask = F.one_hot(torch.arange(h * w).view(h, w), num_classes=int(h * w))
        self_mask = self_mask.permute(2, 0, 1).view(1, h * w, h, w)
        self_mask = self_mask * self.penalty
        return self_mask

    @staticmethod
    def l2_norm(x):
        x = x ** 2
        x = x.sum(dim=[1, 2, 3], keepdim=True)
        return torch.sqrt(x)


class UnetSkipConnectionBlock(nn.Module):
    """Construct a Unet submodule with skip connections, with the following
    structure: downsampling - `submodule` - upsampling.

    Args:
        outer_channels (int): Number of channels at the outer conv layer.
        inner_channels (int): Number of channels at the inner conv layer.
        in_channels (int): Number of channels in input images/features. If is
            None, equals to `outer_channels`. Default: None.
        submodule (UnetSkipConnectionBlock): Previously constructed submodule.
            Default: None.
        is_outermost (bool): Whether this module is the outermost module.
            Default: False.
        is_innermost (bool): Whether this module is the innermost module.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Default: False.
    """

    def __init__(self, outer_channels, inner_channels, in_channels=None, submodule=None, is_outermost=False, is_innermost=False, norm_cfg=dict(type='BN'), use_dropout=False):
        super().__init__()
        assert not (is_outermost and is_innermost), "'is_outermost' and 'is_innermost' cannot be Trueat the same time."
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), f"'norm_cfg' should be dict, butgot {type(norm_cfg)}"
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        use_bias = norm_cfg['type'] == 'IN'
        kernel_size = 4
        stride = 2
        padding = 1
        if in_channels is None:
            in_channels = outer_channels
        down_conv_cfg = dict(type='Conv2d')
        down_norm_cfg = norm_cfg
        down_act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        up_conv_cfg = dict(type='Deconv')
        up_norm_cfg = norm_cfg
        up_act_cfg = dict(type='ReLU')
        up_in_channels = inner_channels * 2
        up_bias = use_bias
        middle = [submodule]
        upper = []
        if is_outermost:
            down_act_cfg = None
            down_norm_cfg = None
            up_bias = True
            up_norm_cfg = None
            upper = [nn.Tanh()]
        elif is_innermost:
            down_norm_cfg = None
            up_in_channels = inner_channels
            middle = []
        else:
            upper = [nn.Dropout(0.5)] if use_dropout else []
        down = [ConvModule(in_channels=in_channels, out_channels=inner_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, conv_cfg=down_conv_cfg, norm_cfg=down_norm_cfg, act_cfg=down_act_cfg, order=('act', 'conv', 'norm'))]
        up = [ConvModule(in_channels=up_in_channels, out_channels=outer_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=up_bias, conv_cfg=up_conv_cfg, norm_cfg=up_norm_cfg, act_cfg=up_act_cfg, order=('act', 'conv', 'norm'))]
        model = down + middle + up + upper
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.is_outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class ResidualBlockWithDropout(nn.Module):
    """Define a Residual Block with dropout layers.

    Ref:
    Deep Residual Learning for Image Recognition

    A residual block is a conv block with skip connections. A dropout layer is
    added between two common conv modules.

    Args:
        channels (int): Number of channels in the conv layer.
        padding_mode (str): The name of padding layer:
            'reflect' | 'replicate' | 'zeros'.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='IN')`.
        use_dropout (bool): Whether to use dropout layers. Default: True.
    """

    def __init__(self, channels, padding_mode, norm_cfg=dict(type='BN'), use_dropout=True):
        super().__init__()
        assert isinstance(norm_cfg, dict), f"'norm_cfg' should be dict, butgot {type(norm_cfg)}"
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        use_bias = norm_cfg['type'] == 'IN'
        block = [ConvModule(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=use_bias, norm_cfg=norm_cfg, padding_mode=padding_mode)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [ConvModule(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=use_bias, norm_cfg=norm_cfg, act_cfg=None, padding_mode=padding_mode)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward function. Add skip connections without final ReLU.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = x + self.block(x)
        return out


class ImgNormalize(nn.Conv2d):
    """Normalize images with the given mean and std value.

    Based on Conv2d layer, can work in GPU.

    Args:
        pixel_range (float): Pixel range of feature.
        img_mean (Tuple[float]): Image mean of each channel.
        img_std (Tuple[float]): Image std of each channel.
        sign (int): Sign of bias. Default -1.
    """

    def __init__(self, pixel_range, img_mean, img_std, sign=-1):
        assert len(img_mean) == len(img_std)
        num_channels = len(img_mean)
        super().__init__(num_channels, num_channels, kernel_size=1)
        std = torch.Tensor(img_std)
        self.weight.data = torch.eye(num_channels).view(num_channels, num_channels, 1, 1)
        self.weight.data.div_(std.view(num_channels, 1, 1, 1))
        self.bias.data = sign * pixel_range * torch.Tensor(img_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class LinearModule(nn.Module):
    """A linear block that contains linear/norm/activation layers.

    For low level vision, we add spectral norm and padding layer.

    Args:
        in_features (int): Same as nn.Linear.
        out_features (int): Same as nn.Linear.
        bias (bool): Same as nn.Linear.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        inplace (bool): Whether to use inplace mode for activation.
        with_spectral_norm (bool): Whether use spectral norm in linear module.
        order (tuple[str]): The order of linear/activation layers. It is a
            sequence of "linear", "norm" and "act". Examples are
            ("linear", "act") and ("act", "linear").
    """

    def __init__(self, in_features, out_features, bias=True, act_cfg=dict(type='ReLU'), inplace=True, with_spectral_norm=False, order=('linear', 'act')):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 2
        assert set(order) == set(['linear', 'act'])
        self.with_activation = act_cfg is not None
        self.with_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        if self.with_spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
        self.init_weights()

    def init_weights(self):
        if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
            a = self.act_cfg.get('negative_slope', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        kaiming_init(self.linear, a=a, nonlinearity=nonlinearity)

    def forward(self, x, activate=True):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, \\*,  # noqa: W605
                c). Same as ``torch.nn.Linear``.
            activate (bool, optional): Whether to use activation layer.
                Defaults to True.

        Returns:
            torch.Tensor: Same as ``torch.nn.Linear``.
        """
        for layer in self.order:
            if layer == 'linear':
                x = self.linear(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class PartialConv2d(nn.Conv2d):
    """Implementation for partial convolution.

    Image Inpainting for Irregular Holes Using Partial Convolutions
    [https://arxiv.org/abs/1804.07723]

    Args:
        multi_channel (bool): If True, the mask is multi-channel. Otherwise,
            the mask is single-channel.
        eps (float): Need to be changed for mixed precision training.
            For mixed precision training, you need change 1e-8 to 1e-6.
    """

    def __init__(self, *args, multi_channel=False, eps=1e-08, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_channel = multi_channel
        self.eps = eps
        if self.multi_channel:
            out_channels, in_channels = self.out_channels, self.in_channels
        else:
            out_channels, in_channels = 1, 1
        self.register_buffer('weight_mask_updater', torch.ones(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.mask_kernel_numel = np.prod(self.weight_mask_updater.shape[1:4])
        self.mask_kernel_numel = np.asscalar(self.mask_kernel_numel)

    def forward(self, input, mask=None, return_mask=True):
        """Forward function for partial conv2d.

        Args:
            input (torch.Tensor): Tensor with shape of (n, c, h, w).
            mask (torch.Tensor): Tensor with shape of (n, c, h, w) or
                (n, 1, h, w). If mask is not given, the function will
                work as standard conv2d. Default: None.
            return_mask (bool): If True and mask is not None, the updated
                mask will be returned. Default: True.

        Returns:
            torch.Tensor : Results after partial conv.            torch.Tensor : Updated mask will be returned if mask is given and                 ``return_mask`` is True.
        """
        assert input.dim() == 4
        if mask is not None:
            assert mask.dim() == 4
            if self.multi_channel:
                assert mask.shape[1] == input.shape[1]
            else:
                assert mask.shape[1] == 1
        if mask is not None:
            with torch.no_grad():
                updated_mask = F.conv2d(mask, self.weight_mask_updater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
                mask_ratio = self.mask_kernel_numel / (updated_mask + self.eps)
                updated_mask = torch.clamp(updated_mask, 0, 1)
                mask_ratio = mask_ratio * updated_mask
        if mask is not None:
            input = input * mask
        raw_out = super().forward(input)
        if mask is not None:
            if self.bias is None:
                output = raw_out * mask_ratio
            else:
                bias_view = self.bias.view(1, self.out_channels, 1, 1)
                output = (raw_out - bias_view) * mask_ratio + bias_view
                output = output * updated_mask
        else:
            output = raw_out
        if return_mask and mask is not None:
            return output, updated_mask
        return output


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(self.in_channels, self.out_channels * scale_factor * scale_factor, self.upsample_kernel, padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def toRed(content):
    return termcolor.colored(content, 'red', attrs=['bold'])


class SRNet(nn.Module):

    def __init__(self, config):
        super(SRNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.config = config
        self.device = config.device
        if self.rank <= 0:
            None
        lib = importlib.import_module('models.archs.{}'.format(config.network))
        self.Network = lib.Network(config)
        self.data = collections.OrderedDict()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=self.config.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, self.config.win)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init(self):
        if self.config.wi is not None and self.config.win is not None:
            self.Network.apply(self.weights_init)
            if self.Network.FlowNet is not None:
                self.Network.FlowNet.load_ckpt(pretrained='./ckpt/SPyNet.pytorch')

    def input_constructor(self, res):
        b, f, c, h, w = res[:]
        imgs = torch.FloatTensor(np.random.randn(b, f, c, h, w))
        flows = torch.FloatTensor(np.random.randn(b, f - 1, c, h, w))
        return {'x': imgs, 'ref': imgs}

    def forward(self, x, ref, is_first_frame=True, is_log=False, is_train=False):
        outs = self.Network.forward(x, ref, is_first_frame, is_log=is_log, is_train=is_train)
        return outs


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class AlignedConv2d(nn.Module):

    def __init__(self, inc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(AlignedConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ReflectionPad2d(padding)
        head = [nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True)]
        head2 = [nn.Conv2d(2 * 32, 32, kernel_size=5, padding=2, stride=stride), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1)]
        self.p_conv = nn.Sequential(*head2)
        self.conv1 = nn.Sequential(*head)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.conv1.register_full_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(2 * inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x, query, ref):
        query = F.interpolate(query, scale_factor=2, mode='bicubic', align_corners=False)
        query = self.conv1(query)
        ref = self.conv1(ref)
        affine = self.p_conv(torch.cat((ref, query), 1)) + 1.0
        if self.modulation:
            m = torch.sigmoid(self.m_conv(torch.cat((ref, query), 1)))
        dtype = affine.data.type()
        ks = self.kernel_size
        N = ks * ks
        if self.padding:
            x = self.zero_padding(x)
        affine = torch.clamp(affine, -3, 3)
        p = self._get_p(affine, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        alignment = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(alignment.size(1))], dim=1)
            alignment *= m
        alignment = self._reshape_alignment(alignment, ks)
        return alignment

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-1 * ((self.kernel_size - 1) // 2) - 0.5, (self.kernel_size - 1) // 2 + 0.6, 1.0), torch.arange(-1 * ((self.kernel_size - 1) // 2) - 0.5, (self.kernel_size - 1) // 2 + 0.6, 1.0))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, affine, dtype):
        N, h, w = self.kernel_size * self.kernel_size, affine.size(2), affine.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_n.repeat(affine.size(0), 1, h, w)
        p = p.permute(0, 2, 3, 1)
        affine = affine.permute(0, 2, 3, 1)
        s_x = affine[:, :, :, 0:1]
        s_y = affine[:, :, :, 1:2]
        p[:, :, :, :N] = p[:, :, :, :N].clone() * s_x.type(dtype)
        p[:, :, :, N:] = p[:, :, :, N:].clone() * s_y.type(dtype)
        p = p.view(p.shape[0], p.shape[1], p.shape[2], 1, p.shape[3])
        p = torch.cat((p[:, :, :, :, :N], p[:, :, :, :, N:]), 3)
        p = p.permute(0, 1, 2, 4, 3)
        theta = (affine[:, :, :, 2:] - 1.0) * 1.0472
        rm = torch.cat((torch.cos(theta), torch.sin(theta), -1 * torch.sin(theta), torch.cos(theta)), 3)
        rm = rm.view(affine.shape[0], affine.shape[1], affine.shape[2], 2, 2)
        result = torch.matmul(p, rm)
        result = torch.cat((result[:, :, :, :, 0], result[:, :, :, :, 1]), 3)
        result = result.permute(0, 3, 1, 2) + (self.kernel_size - 1) // 2 + 0.5 + p_0
        return result

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        padded_h = x.size(2)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        result = x.gather(dim=-1, index=index.long()).contiguous().view(b, c, h, w, N)
        return result

    @staticmethod
    def _reshape_alignment(alignment, ks):
        b, c, h, w, N = alignment.size()
        alignment = torch.cat([alignment[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        alignment = alignment.contiguous().view(b, c, h * ks, w * ks)
        return alignment


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = padding_left, padding_right, padding_top, padding_bottom
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


class AlignedAttention(nn.Module):

    def __init__(self, ksize=3, k_vsize=1, scale=1, stride=1, align=False):
        super(AlignedAttention, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0
        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        self.align = align
        if align:
            self.align = AlignedConv2d(inc=128, kernel_size=self.scale * self.k_vsize, padding=1, stride=self.scale * 1, bias=None, modulation=False)

    def warp(self, input, dim, index):
        views = [input.size(0)] + [(1 if i != dim else -1) for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, ref, index_map, value, name, return_fm=False):
        shape_out = list(lr.size())
        kernel = self.scale * self.k_vsize
        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel], strides=[self.stride * self.scale, self.stride * self.scale], rates=[1, 1], padding='same')
        warpped_value = self.warp(unfolded_value, 2, index_map)
        warpped_features_ = F.fold(warpped_value, output_size=(shape_out[2] * 2, shape_out[3] * 2), kernel_size=(kernel, kernel), padding=0, stride=self.scale)
        if return_fm:
            return warpped_features_
        if self.align:
            unfolded_ref = extract_image_patches(ref, ksizes=[kernel, kernel], strides=[self.stride * self.scale, self.stride * self.scale], rates=[1, 1], padding='same')
            warpped_ref = self.warp(unfolded_ref, 2, index_map)
            warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2] * 2, shape_out[3] * 2), kernel_size=(kernel, kernel), padding=0, stride=self.scale)
            warpped_features = self.align(warpped_features_, lr, warpped_ref)
        else:
            return warpped_features_
        return warpped_features


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, In=False, act=nn.PReLU()):
        m = [conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if In:
            m.append(nn.InstanceNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        deform_groups (int): Deformable groups. Defaults: 8.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self, mid_channels=64, deform_groups=8, act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
            if i == 3:
                self.offset_conv2[level] = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            else:
                self.offset_conv2[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
                self.offset_conv3[level] = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
            self.dcn_pack[level] = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
            if i < 3:
                act_cfg_ = act_cfg if i == 2 else None
                self.feat_conv[level] = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg_)
        self.cas_offset_conv1 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_offset_conv2 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, neighbor_feats, ref_feats):
        """Forward function for PCDAlignment.

        Align neighboring frames to the reference frame in the feature level.

        Args:
            neighbor_feats (list[Tensor]): List of neighboring features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).
            ref_feats (list[Tensor]): List of reference features. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (n, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        assert len(neighbor_feats) == 3 and len(ref_feats) == 3, f'The length of neighbor_feats and ref_feats must be both 3, but got {len(neighbor_feats)} and {len(ref_feats)}'
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([neighbor_feats[i - 1], ref_feats[i - 1]], dim=1)
            offset = self.offset_conv1[level](offset)
            if i == 3:
                offset = self.offset_conv2[level](offset)
            else:
                offset = self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1))
                offset = self.offset_conv3[level](offset)
            feat = self.dcn_pack[level](neighbor_feats[i - 1], offset)
            if i == 3:
                feat = self.lrelu(feat)
            else:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        offset = torch.cat([feat, ref_feats[0]], dim=1)
        offset = self.cas_offset_conv2(self.cas_offset_conv1(offset))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.

    Args:
        mid_channels (int): Number of the channels of middle features.
            Default: 64.
        num_frames (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
        act_cfg (dict): Activation function config for ConvModule.
            Default: LeakyReLU with negative_slope=0.1.
    """

    def __init__(self, mid_channels=64, num_frames=5, center_frame_idx=2, act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.temporal_attn2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.feat_fusion = ConvModule(num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = ConvModule(num_frames * mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn2 = ConvModule(mid_channels * 2, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn3 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn4 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn5 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.spatial_attn_l1 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_l2 = ConvModule(mid_channels * 2, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_l3 = ConvModule(mid_channels, mid_channels, 3, padding=1, act_cfg=act_cfg)
        self.spatial_attn_add1 = ConvModule(mid_channels, mid_channels, 1, act_cfg=act_cfg)
        self.spatial_attn_add2 = nn.Conv2d(mid_channels, mid_channels, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """Forward function for TSAFusion.

        Args:
            aligned_feat (Tensor): Aligned features with shape (n, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (n, c, h, w).
        """
        n, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        emb = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        emb = emb.view(n, t, -1, h, w)
        corr_l = []
        for i in range(t):
            emb_neighbor = emb[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))
        corr_prob = corr_prob.unsqueeze(2).expand(n, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(n, -1, h, w)
        aligned_feat = aligned_feat.view(n, -1, h, w) * corr_prob
        feat = self.feat_fusion(aligned_feat)
        attn = self.spatial_attn1(aligned_feat)
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l1(attn)
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1))
        attn_level = self.spatial_attn_l3(attn_level)
        attn_level = self.upsample(attn_level)
        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.spatial_attn_add1(attn))
        attn = torch.sigmoid(attn)
        feat = feat * attn * 2 + attn_add
        return feat


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmedit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.
    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
        pretrained (str): The pretrained model path. Default: None.
    """

    def __init__(self, in_channels=3, out_channel=3, mid_channels=64, num_frames=5, deform_groups=8, num_blocks_extraction=5, num_blocks_reconstruction=10, center_frame_idx=2, with_tsa=True, pretrained=None):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_blocks_extraction, mid_channels=mid_channels)
        self.feat_l2_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.pcd_alignment = PCDAlignment(mid_channels=mid_channels, deform_groups=deform_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(mid_channels=mid_channels, num_frames=num_frames, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """
        n, t, c, h, w = x.size()
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))
        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)
        ref_feats = [l1_feat[:, self.center_frame_idx].clone(), l2_feat[:, self.center_frame_idx].clone(), l3_feat[:, self.center_frame_idx].clone()]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [l1_feat[:, i].clone(), l2_feat[:, i].clone(), l3_feat[:, i].clone()]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)
        return feat


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias)


class FeatureMatching(nn.Module):

    def __init__(self, scale=2, stride=1, flag_HD_in=False):
        super(FeatureMatching, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0
        self.ksize = 3
        self.scale = scale
        self.stride = stride
        self.flag_HD_in = flag_HD_in
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.feature_extract = torch.nn.Sequential()
        if self.flag_HD_in is False:
            vgg_range = 4 if self.scale == 4 else 7
        else:
            vgg_range = 7
        self.vgg_range = vgg_range
        for x in range(vgg_range):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])
        match0 = BasicBlock(default_conv, 64 if vgg_range == 4 else 128, 16, 1, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))
        self.feature_extract.add_module('map{}'.format(64 if vgg_range == 4 else 128), match0)
        for param in self.feature_extract.parameters():
            param.requires_grad = True
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229, 0.224, 0.225
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std)
        self.avgpool = nn.AvgPool2d((2, 2), (2, 2))
        if self.flag_HD_in:
            self.scale_factor_x2 = 1 / (self.scale // 2)
        else:
            self.scale_factor_x2 = self.scale / 2

    def forward(self, lr, ref, ref_downsample=True):
        h, w = lr.size()[2:]
        lr = self.sub_mean(lr)
        ref = self.sub_mean(ref)
        if self.flag_HD_in:
            lr = F.interpolate(lr, scale_factor=self.scale_factor_x2, mode='nearest')
            ref = F.interpolate(ref, scale_factor=self.scale_factor_x2, mode='nearest')
        lr_f = self.feature_extract(lr)
        lr_p = extract_image_patches(lr_f, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        if ref_downsample:
            ref_down = self.avgpool(ref)
        else:
            ref_down = ref
        ref_f = self.feature_extract(ref_down)
        ref_p = extract_image_patches(ref_f, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        ref_p = ref_p.permute(0, 2, 1)
        ref_p = F.normalize(ref_p, dim=2)
        lr_p = F.normalize(lr_p, dim=1)
        N, hrwr, _ = ref_p.size()
        _, _, hw = lr_p.size()
        relavance_maps, hard_indices = torch.max(torch.einsum('bij,bjk->bik', ref_p.contiguous(), lr_p.contiguous()), dim=1)
        shape_lr = lr_f.shape
        relavance_maps = relavance_maps.view(shape_lr[0], 1, shape_lr[2], shape_lr[3])
        h_c, w_c = relavance_maps.size()[2:]
        if h / h_c != 1.0:
            relavance_maps = F.interpolate(relavance_maps, scale_factor=h / h_c, mode='bicubic', align_corners=False).clamp(0, 1)
        return relavance_maps, hard_indices


class ResList(nn.Module):

    def __init__(self, num_res_blocks, n_feats, res_scale=1):
        super(ResList, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(ConvModule(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')), ConvModule(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')), ConvModule(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')), ConvModule(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')), ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained, device='cuda'):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        if isinstance(pretrained, str):
            logger = get_root_logger(log_level=0)
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'[pretrained] should be str or None, but got {type(pretrained)}.')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_ckpt(self, pretrained):
        logger = get_root_logger(log_level=0)
        load_checkpoint(self, pretrained, strict=False, logger=logger)

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border'), flow_up], 1))
        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)
        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


def norm_res_vis(res):
    res = res.detach().clone()
    b, c, h, w = res.size()
    res = res.view(res.size(0), -1)
    res = res - res.min(1, keepdim=True)[0]
    res = res / res.max(1, keepdim=True)[0]
    res = res.view(b, c, h, w)
    return res


Backward_tensorGrid = {}


def warp(tensorInput, tensorFlow, mode='bilinear', padding_mode='zeros', align_corners=False):
    if str(tensorFlow.size()[2:]) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(-1, -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(-1, -1, -1, tensorFlow.size(3))
        Backward_tensorGrid[str(tensorFlow.size()[2:])] = torch.cat([tensorHorizontal, tensorVertical], 1)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size()[2:])] + tensorFlow).permute(0, 2, 3, 1), mode=mode, padding_mode=padding_mode, align_corners=align_corners)


class Network(nn.Module):

    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.scale = config.scale
        self.flag_HD_in = config.flag_HD_in
        num_blocks = config.num_blocks
        mid_channels = config.mid_channels
        self.mid_channels = mid_channels
        self.keyframe_stride = config.keyframe_stride
        self.padding = 2
        self.edvr = EDVRFeatureExtractor(num_frames=self.padding * 2 + 1, center_frame_idx=self.padding, pretrained='https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth')
        self.FlowNet = SPyNet(pretrained='./ckpt/SPyNet.pytorch', device=self.config.device)
        for name, param in self.FlowNet.named_parameters():
            param.requires_grad = False
        self.feature_match = FeatureMatching(scale=self.scale, stride=1, flag_HD_in=self.flag_HD_in)
        self.aa1 = AlignedAttention(scale=config.matching_ksize // 2, align=True if config.matching_ksize // 2 > 1 else False)
        self.aa2 = AlignedAttention(scale=config.matching_ksize, align=True)
        m_head1 = [BasicBlock(default_conv, 3, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        m_head2 = [BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=2, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        conf_fusion = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        conf_fusion2 = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        conf_fusion_BWFW = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion = [BasicBlock(default_conv, 2 * mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion2_1 = [BasicBlock(default_conv, 2 * mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion2 = [BasicBlock(default_conv, 2 * mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion_BWFW = [BasicBlock(default_conv, 2 * mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)), BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        self.ref_encoder1 = nn.Sequential(*m_head1)
        self.res1 = ResList(4, mid_channels)
        self.ref_encoder2 = nn.Sequential(*m_head2)
        self.res2 = ResList(4, mid_channels)
        self.conf_fusion = nn.Sequential(*conf_fusion)
        self.feat_fusion = nn.Sequential(*feat_fusion)
        self.feat_decoder = ResList(8, mid_channels)
        self.conf_fusion2 = nn.Sequential(*conf_fusion2)
        self.feat_fusion2_1 = nn.Sequential(*feat_fusion2_1)
        self.feat_fusion2 = nn.Sequential(*feat_fusion2)
        self.feat_decoder2 = ResList(4, mid_channels)
        self.conf_fusion_BWFW = nn.Sequential(*conf_fusion_BWFW)
        self.feat_fusion_BWFW = nn.Sequential(*feat_fusion_BWFW)
        self.feat_decoder_BWFW = ResList(4, mid_channels)
        self.backward_fusion = nn.Conv2d(64 + mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.forward_fusion = nn.Conv2d(64 + mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(2 * mid_channels + 3, mid_channels, num_blocks)
        self.fusion_UP = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        if self.scale == 4:
            self.upsample2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.forward_feat_prop_prev = None
        self.forward_flow_prev = None
        self.forward_feat_prop_UP_prev = None
        self.forward_conf_map_prop_prev = None
        self.frame_itr_num = 0
        self.max_frame_itr_num = self.config.reset_branch

    def compute_up(self, backward_feat_UP, forward_feat_UP, conf_map_backward, conf_map_forward, base):
        conf_map_backward = F.interpolate(conf_map_backward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_forward = F.interpolate(conf_map_forward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        cat_features = torch.cat([backward_feat_UP, forward_feat_UP], dim=1)
        out = self.fusion_UP(cat_features)
        alpha = self.conf_fusion_BWFW(torch.cat([conf_map_backward, conf_map_forward], dim=1))
        out = out + alpha * self.feat_fusion_BWFW(cat_features)
        out = self.feat_decoder_BWFW(out)
        if self.scale == 4:
            out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out) + base
        return out

    def AA_AF_conf_prop(self, lr, ref, conf_map, conf_map_prop, index_map, feat_prop, feat_prop_UP, ref_feat_down, ref_feat):
        lr_down = F.interpolate(lr, scale_factor=1 / 2, mode='bicubic', align_corners=False).clamp(0, 1)
        ref_feat_aligned = self.aa1(lr_down, ref, index_map, ref_feat_down, 'aa1')
        cat_features = torch.cat([feat_prop, ref_feat_aligned], dim=1)
        alpha = self.conf_fusion(torch.cat([conf_map_prop, conf_map], dim=1))
        feat_prop = feat_prop + alpha * self.feat_fusion(cat_features)
        feat_prop = self.feat_decoder(feat_prop)
        ref_feat_aligned_UP = self.aa2(lr, ref, index_map, ref_feat, 'aa2')
        feat_prop_UP = self.feat_fusion2_1(torch.cat([feat_prop_UP, self.upsample1(feat_prop)], dim=1))
        cat_features = torch.cat([feat_prop_UP, ref_feat_aligned_UP], dim=1)
        conf_map_prop_UP = F.interpolate(conf_map_prop, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_UP = F.interpolate(conf_map, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        alpha = self.conf_fusion2(torch.cat([conf_map_prop_UP, conf_map_UP], dim=1))
        feat_prop_UP = feat_prop_UP + alpha * self.feat_fusion2(cat_features)
        feat_prop_UP = self.feat_decoder2(feat_prop_UP)
        conf_map_prop, _ = torch.max(torch.cat([conf_map_prop, conf_map], dim=1), dim=1, keepdim=True)
        return feat_prop, feat_prop_UP, conf_map_prop

    def spatial_padding(self, lrs):
        """ Apply pdding spatially.
        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = lrs.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        lrs = lrs.reshape(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')
        return lrs.view(n, t, c, h + pad_h, w + pad_w)

    def compute_refill_features(self, lrs, keyframe_idx, h, w):
        """ Compute keyframe features for information-refill.
        Since EDVR-M is used, padding is performed before feature computation.
        Args:
            lrs (Tensor): Input LR images with shape (n, t, c, h, w)
            keyframe_idx (list(int)): The indices specifying the keyframes.
        Return:
            dict(Tensor): The keyframe features. Each key corresponds to the
                indices in keyframe_idx.
        """
        if self.padding == 2:
            lrs = [lrs[:, [4, 3]], lrs, lrs[:, [-4, -5]]]
        elif self.padding == 3:
            lrs = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]
        lrs = torch.cat(lrs, dim=1)
        num_frames = 2 * self.padding + 1
        feats_refill = {}
        for i in keyframe_idx:
            feats_refill[i] = self.edvr(lrs[:, i:i + num_frames].contiguous())[:, :, :h, :w]
        return feats_refill

    def forward(self, lrs, refs, is_first_frame, is_log=False, is_train=False):
        """Forward function for RefVSR.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            refs (Tensor): Input Ref sequence with shape (n, t, c, h, w). None: Spatial resolution of the tensors are twice as the LR during training, equal during validation and evaluation.
            is_first_frame (boolean): whether lrs[:, 0] and refs[:, 0] are the first frame of a video sequence.
            is_log (boolean): whether to return samples (e.g., confidence map, warppred image, etc. for debugging. The return is enabled when config.save_sample is True.)
            is_train (boolean): whether it is trainig phase.
        Returns:
            Tensor: Output HR (SR result of the center LR frame) with shape (n, c, 4h, 4w).
        """
        outs = collections.OrderedDict()
        if is_log:
            outs['vis'] = collections.OrderedDict()
        if is_train == False:
            if self.max_frame_itr_num is not None and self.frame_itr_num == self.max_frame_itr_num:
                is_first_frame = True
        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, f'The height and width of inputs should be at least 64, but got {h} and {w}.'
        with torch.no_grad():
            forward_flows = []
            backward_flows = []
            for j in range(0, t - 1):
                forward_flows.append(F.interpolate(self.FlowNet(lrs[:, j + 1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            for j in range(t - 1, 0, -1):
                backward_flows.insert(0, F.interpolate(self.FlowNet(lrs[:, j - 1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            forward_flows = torch.cat(forward_flows, dim=1)
            backward_flows = torch.cat(backward_flows, dim=1)
        lrs = self.spatial_padding(lrs)
        if is_first_frame:
            self.keyframe_idx = np.arange(0, t, self.keyframe_stride)
        else:
            new_ki = self.keyframe_idx - 1
            new_ki = new_ki[new_ki >= 0]
            self.keyframe_idx = np.arange(new_ki[0], t, self.keyframe_stride)
        if self.keyframe_idx[-1] != t - 1:
            self.keyframe_idx = np.append(self.keyframe_idx, t - 1)
        feats_refill = self.compute_refill_features(lrs, self.keyframe_idx, h, w)
        lrs = lrs[:, :, :, :h, :w]
        conf_maps = []
        index_maps = []
        for i in range(0, t):
            conf_map, index_map = self.feature_match(lrs[:, i], refs[:, i])
            conf_maps.append(conf_map)
            index_maps.append(index_map)
        if is_train is False:
            gc.collect()
            torch.cuda.empty_cache()
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_UP = lrs.new_zeros(n, self.mid_channels, h * 2, w * 2)
        conf_map_prop = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = backward_flows[:, i]
                feat_prop = warp(feat_prop, flow)
                conf_map_prop = warp(conf_map_prop, flow)
                feat_prop_UP = warp(feat_prop_UP, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
            if i in self.keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            conf_map = conf_maps[i]
            index_map = index_maps[i]
            ref_feat = self.res1(self.ref_encoder1(refs[:, i]))
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat))
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1)), feat_prop_UP, ref_feat_down, ref_feat)
            if i == t // 2:
                backward_feat_UP = feat_prop_UP
                conf_map_prop_backward = conf_map_prop
            outputs.append(feat_prop)
        outputs = outputs[::-1]
        if is_first_frame:
            feat_prop = torch.zeros_like(feat_prop)
            feat_prop_UP = torch.zeros_like(backward_feat_UP)
            conf_map_prop = torch.zeros_like(conf_map)
        for i in range(0, t // 2 + 1):
            lr_curr = lrs[:, i]
            if i > 0:
                feat_prop = warp(feat_prop, forward_flows[:, i - 1])
                feat_prop_UP = warp(feat_prop, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(conf_map_prop, flow)
            elif i == 0 and is_first_frame is False:
                feat_prop = warp(self.forward_feat_prop_prev, self.forward_flow_prev)
                feat_prop_UP = warp(self.forward_feat_prop_UP_prev, F.interpolate(input=self.forward_flow_prev, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(self.forward_conf_map_prop_prev, self.forward_flow_prev)
            if i in self.keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)
            conf_map = conf_maps[i]
            index_map = index_maps[i]
            ref_feat = self.res1(self.ref_encoder1(refs[:, i]))
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat))
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, self.forward_resblocks(torch.cat([lr_curr, outputs[i], feat_prop], dim=1)), feat_prop_UP, ref_feat_down, ref_feat)
            if i == 0:
                self.forward_feat_prop_prev = feat_prop.detach().clone()
                self.forward_flow_prev = forward_flows[:, i, :, :, :].detach().clone()
                self.forward_feat_prop_UP_prev = feat_prop_UP.detach().clone()
                self.forward_conf_map_prop_prev = conf_map_prop.detach().clone()
        base = F.interpolate(lrs[:, t // 2], scale_factor=self.scale, mode='bicubic', align_corners=False).clamp(0, 1)
        out = self.compute_up(backward_feat_UP, feat_prop_UP, conf_map_prop_backward, conf_map_prop, base)
        if is_train is False:
            if is_first_frame:
                self.frame_itr_num = 0
            self.frame_itr_num += 1
            out = out.clamp(0, 1)
        outs['result'] = out
        if is_log and self.config.save_sample:
            with torch.no_grad():
                lr_down = F.interpolate(lrs[:, t // 2], scale_factor=1 / 2, mode='bicubic', align_corners=False).clamp(0, 1)
                conf_map_prop_forward = conf_map_prop
                conf_map_prop, _ = torch.max(torch.cat([conf_map_prop_backward, conf_map_prop_forward], dim=1), dim=1, keepdim=True)
                ref_downsampled = F.interpolate(refs[:, t // 2], scale_factor=1 / 2, mode='bicubic', align_corners=False).clamp(0, 1)
                outs['vis']['FW_aa1_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t // 2], index_map, ref_downsampled, 'aa1', return_fm=True).detach().clone()
                if self.aa1.align:
                    outs['vis']['FW_aa1_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t // 2], index_map, ref_downsampled, 'aa1').detach().clone()
                outs['vis']['FW_aa2_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t // 2], refs[:, t // 2], index_map, refs[:, t // 2], 'aa2', return_fm=True).detach().clone()
                if self.aa2.align:
                    outs['vis']['FW_aa2_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t // 2], refs[:, t // 2], index_map, refs[:, t // 2], 'aa2').detach().clone()
                outs['vis']['conf_map_norm'] = norm_res_vis(conf_map)
                outs['vis']['conf_map_prop_backward_norm'] = norm_res_vis(conf_map_prop_backward)
                outs['vis']['conf_map_prop_forward_norm'] = norm_res_vis(conf_map_prop_forward)
                outs['vis']['conf_map_prop_norm'] = norm_res_vis(conf_map_prop)
        return outs


class PatchSelect(nn.Module):

    def __init__(self, stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride

    def forward(self, lr, ref):
        shape_lr = lr.shape
        shape_ref = ref.shape
        P = shape_ref[3] - shape_lr[3] + 1
        ref = extract_image_patches(ref, ksizes=[shape_lr[2], shape_lr[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')
        lr = lr.view(shape_lr[0], shape_lr[1] * shape_lr[2] * shape_lr[3], 1)
        y = torch.mean(torch.abs(ref - lr), 1)
        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True)
        return hard_indices.view(-1), P, relavance_maps


class Encoder_input(nn.Module):

    def __init__(self, num_res_blocks, n_feats, img_channel, res_scale=1):
        super(Encoder_input, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(img_channel, n_feats)
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv_tail = conv3x3(n_feats, n_feats)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class EDVRNet(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """

    def __init__(self, in_channels, out_channels, mid_channels=64, num_frames=5, deform_groups=8, num_blocks_extraction=5, num_blocks_reconstruction=10, center_frame_idx=2, with_tsa=True):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_blocks_extraction, mid_channels=mid_channels)
        self.feat_l2_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.pcd_alignment = PCDAlignment(mid_channels=mid_channels, deform_groups=deform_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(mid_channels=mid_channels, num_frames=num_frames, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)
        self.reconstruction = make_layer(ResidualBlockNoBN, num_blocks_reconstruction, mid_channels=mid_channels)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function for EDVRNet.

        Args:
            x (Tensor): Input tensor with shape (n, t, c, h, w).

        Returns:
            Tensor: SR center frame with shape (n, c, h, w).
        """
        n, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, f'The height and width of inputs should be a multiple of 4, but got {h} and {w}.'
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))
        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)
        ref_feats = [l1_feat[:, self.center_frame_idx, :, :, :].clone(), l2_feat[:, self.center_frame_idx, :, :, :].clone(), l3_feat[:, self.center_frame_idx, :, :, :].clone()]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(), l3_feat[:, i, :, :, :].clone()]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)
        out = self.reconstruction(feat)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = self.img_upsample(x_center)
        out += base
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            if self.with_tsa:
                for module in [self.fusion.feat_fusion, self.fusion.spatial_attn1, self.fusion.spatial_attn2, self.fusion.spatial_attn3, self.fusion.spatial_attn4, self.fusion.spatial_attn_l1, self.fusion.spatial_attn_l2, self.fusion.spatial_attn_l3, self.fusion.spatial_attn_add1]:
                    kaiming_init(module.conv, a=0.1, mode='fan_out', nonlinearity='leaky_relu', bias=0, distribution='uniform')
        else:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class VGG19(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
        return out


LOSS_TYPES = ['cosine']


def compute_cosine_distance(x, y):
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_mu = x.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - x_mu
    y_centered = y - y_mu
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)
    y_normalized = y_normalized.reshape(N, C, -1)
    dist = torch.clamp(1 - torch.bmm(x_normalized.transpose(1, 2), y_normalized), min=0)
    return dist


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)
    cx = w / (torch.sum(w, dim=2, keepdim=True) + 1e-05)
    return cx


def compute_l1_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_vec_ = x.view(N, C, 1, -1).sum(dim=1)
    y_vec_ = x.view(N, C, -1, 1).sum(dim=1)
    dist = (x_vec_ - y_vec_).abs()
    dist = dist.reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.0)
    return dist


def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)
    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.0)
    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)
    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)
    return feature_grid


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-05)
    return dist_tilde


def contextual_bilateral_loss(x=torch.Tensor, y=torch.Tensor, weight_sp=0.1, band_width=0.5, loss_type='cosine'):
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
    N, C, H, W = x.size()
    grid = compute_meshgrid(x.shape)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'L2':
        dist_raw = compute_l2_distance(x, y)
    elif loss_type == 'L1':
        dist_raw = compute_l1_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_ = compute_cx(dist_tilde, band_width)
    cx_ = (1.0 - weight_sp) * cx_ + weight_sp * cx_sp
    r_m = torch.max(cx_, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / band_width), 1, r_m[1])
    rank = torch.distributed.get_rank()
    cx = torch.sum(torch.squeeze(r_m[0] * c, 1), dim=1) / torch.sum(torch.squeeze(c, 1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-05))
    c = c.view(N, 1, y.shape[2], y.shape[3])
    return cx_loss, c


def contextual_loss(x=torch.Tensor, y=torch.Tensor, band_width=0.5, loss_type='cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
    N, C, H, W = x.size()
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_ = compute_cx(dist_tilde, band_width)
    r_m = torch.max(cx_, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / 0.5), 1, r_m[1])
    rank = torch.distributed.get_rank()
    cx = torch.sum(torch.squeeze(r_m[0] * c, 1), dim=1) / torch.sum(torch.squeeze(c, 1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-05))
    c = c.view(N, 1, y.shape[2], y.shape[3])
    return cx_loss, c


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.
    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width=0.5, loss_type='cosine', is_CoBi=False, use_vgg=True, vgg_layer='relu3_4'):
        super(ContextualLoss, self).__init__()
        self.band_width = band_width
        self.is_CoBi = is_CoBi
        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
            self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 channel images.'
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        if self.is_CoBi:
            return contextual_bilateral_loss(x, y, band_width=self.band_width)
        else:
            return contextual_loss(x, y, band_width=self.band_width)


class GaussianLayer(nn.Module):

    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(nn.ReflectionPad2d(2), nn.Conv2d(3, 3, 3, stride=1, padding=0, bias=None, groups=3))
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((3, 3))
        n[1, 1] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=1)
        for name, f in self.named_parameters():
            weight_torch = torch.from_numpy(k)
            f.data.copy_(weight_torch)
            f.requires_grad = False


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContextualAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Encoder_input,
     lambda: ([], {'num_res_blocks': 4, 'n_feats': 4, 'img_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeatureMatching,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (GaussianLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (ImgNormalize,
     lambda: ([], {'pixel_range': 4, 'img_mean': [4, 4], 'img_std': [4, 4]}),
     lambda: ([torch.rand([4, 2, 64, 64])], {}),
     True),
    (PatchSelect,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResList,
     lambda: ([], {'num_res_blocks': 4, 'n_feats': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_codeslake_RefVSR(_paritybench_base):
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

