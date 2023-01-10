import sys
_module = sys.modules[__name__]
del sys
basicsr = _module
archs = _module
arch_util = _module
basicvsr_arch = _module
basicvsrpp_arch = _module
dfdnet_arch = _module
dfdnet_util = _module
discriminator_arch = _module
duf_arch = _module
ecbsr_arch = _module
edsr_arch = _module
edvr_arch = _module
hifacegan_arch = _module
hifacegan_util = _module
inception = _module
rcan_arch = _module
ridnet_arch = _module
rrdbnet_arch = _module
spynet_arch = _module
srresnet_arch = _module
srvgg_arch = _module
stylegan2_arch = _module
stylegan2_bilinear_arch = _module
swinir_arch = _module
tof_arch = _module
vgg_arch = _module
data = _module
data_sampler = _module
data_util = _module
degradations = _module
ffhq_dataset = _module
paired_image_dataset = _module
prefetch_dataloader = _module
realesrgan_dataset = _module
realesrgan_paired_dataset = _module
reds_dataset = _module
single_image_dataset = _module
transforms = _module
video_test_dataset = _module
vimeo90k_dataset = _module
losses = _module
basic_loss = _module
gan_loss = _module
loss_util = _module
metrics = _module
fid = _module
metric_util = _module
niqe = _module
psnr_ssim = _module
test_psnr_ssim = _module
models = _module
base_model = _module
edvr_model = _module
esrgan_model = _module
hifacegan_model = _module
lr_scheduler = _module
realesrgan_model = _module
realesrnet_model = _module
sr_model = _module
srgan_model = _module
stylegan2_model = _module
swinir_model = _module
video_base_model = _module
video_gan_model = _module
video_recurrent_gan_model = _module
video_recurrent_model = _module
ops = _module
dcn = _module
deform_conv = _module
fused_act = _module
fused_act = _module
upfirdn2d = _module
upfirdn2d = _module
test = _module
train = _module
utils = _module
color_util = _module
diffjpeg = _module
dist_util = _module
download_util = _module
file_client = _module
flow_util = _module
img_process_util = _module
img_util = _module
lmdb_util = _module
logger = _module
matlab_functions = _module
misc = _module
options = _module
plot_util = _module
registry = _module
auto_generate_api = _module
conf = _module
inference_basicvsr = _module
inference_basicvsrpp = _module
inference_dfdnet = _module
inference_esrgan = _module
inference_ridnet = _module
inference_stylegan2 = _module
inference_swinir = _module
create_lmdb = _module
download_datasets = _module
extract_images_from_tfrecords = _module
extract_subimages = _module
generate_meta_info = _module
prepare_hifacegan_dataset = _module
regroup_reds_dataset = _module
download_gdrive = _module
download_pretrained_models = _module
calculate_fid_folder = _module
calculate_fid_stats_from_datasets = _module
calculate_lpips = _module
calculate_niqe = _module
calculate_psnr_ssim = _module
calculate_stylegan2_fid = _module
convert_dfdnet = _module
convert_models = _module
convert_ridnet = _module
convert_stylegan = _module
model_complexity_cmp_bsrn = _module
publish_models = _module
setup = _module
test_discriminator_backward = _module
test_ffhq_dataset = _module
test_lr_scheduler = _module
test_niqe = _module
test_paired_image_dataset = _module
test_reds_dataset = _module
test_vimeo90k_dataset = _module
test_basicvsr_arch = _module
test_discriminator_arch = _module
test_duf_arch = _module
test_ecbsr_arch = _module
test_srresnet_arch = _module
test_single_image_dataset = _module
test_losses = _module
test_sr_model = _module

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


import collections.abc


import math


import torch


import torchvision


import warnings


from itertools import repeat


from torch import nn as nn


from torch.nn import functional as F


from torch.nn import init as init


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torch.nn.utils.spectral_norm import spectral_norm


from torch.autograd import Function


from torch.nn.utils import spectral_norm


import re


from torch.nn import init


from torch.utils.model_zoo import load_url


from torchvision import models


import random


from torch import nn


import torch.utils.checkpoint as checkpoint


from collections import OrderedDict


from torchvision.models import vgg as vgg


import torch.utils.data


from copy import deepcopy


from functools import partial


from torch.utils.data.sampler import Sampler


from scipy import special


from scipy.stats import multivariate_normal


from torchvision.transforms.functional_tensor import rgb_to_grayscale


import time


from torch.utils import data as data


from torchvision.transforms.functional import normalize


import queue as Queue


from torch.utils.data import DataLoader


from torch import autograd as autograd


import functools


from scipy import linalg


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from collections import Counter


from torch.optim.lr_scheduler import _LRScheduler


from torch import distributed as dist


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


import logging


import itertools


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.hub import download_url_to_file


from torch.hub import get_dir


from torchvision.utils import make_grid


import torchvision.transforms as transforms


from torchvision import utils


from torch.serialization import _is_zipfile


from torch.serialization import _open_file_like


import copy


import torchvision.utils


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:

            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class
            return deco
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            None
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True), make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class BasicModule(nn.Module):
    """Basic module of SPyNet.

    Note that unlike the architecture in spynet_arch.py, the basic module
    here contains batch normalization.
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Estimated flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = ref[0].new_zeros([ref[0].size(0), 2, int(math.floor(ref[0].size(2) / 2.0)), int(math.floor(ref[0].size(3) / 2.0))])
        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')
            flow = self.basic_module[level](torch.cat([ref[level], flow_warp(supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'), upsampled_flow], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()
        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)
        return flow


class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        self.spynet = SpyNet(spynet_path)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
        return torch.stack(out_l, dim=1)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_ext.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_ext.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


initialized_logger = {}


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        feat = self.lrelu(self.feat_fusion(aligned_feat))
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)
        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)
        feat = feat * attn * 2 + attn_add
        return feat


class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor used in IconVSR.

    Args:
        num_input_frame (int): Number of input frames.
        num_feat (int): Number of feature channels
        load_path (str): Path to the pretrained weights of EDVR. Default: None.
    """

    def __init__(self, num_input_frame, num_feat, load_path):
        super(EDVRFeatureExtractor, self).__init__()
        self.center_frame_idx = num_input_frame // 2
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)
        ref_feat_l = [feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(), feat_l3[:, self.center_frame_idx, :, :, :].clone()]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        return self.fusion(aligned_feat)


class IconVSR(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, keyframe_stride=5, temporal_padding=2, spynet_path=None, edvr_path=None):
        super().__init__()
        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        self.spynet = SpyNet(spynet_path)
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        """Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)
        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()
        x = self.pad_spatial(x)
        h, w = x.shape[3:]
        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1))
        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)
        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped. Besides, we adopt the official DCN
    implementation and the version of torch need to be higher than 1.9.

    ``Paper: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment``

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self, mid_channels=64, num_blocks=7, max_residue_magnitude=10, is_low_res_input=True, spynet_path=None, cpu_cache_length=100):
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.spynet = SpyNet(spynet_path)
        if is_low_res_input:
            self.feat_extract = ConvResidualBlocks(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), ConvResidualBlocks(mid_channels, mid_channels, 5))
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(2 * mid_channels, mid_channels, 3, padding=1, deformable_groups=16, max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)
        self.reconstruction = ConvResidualBlocks(5 * mid_channels, mid_channels, 5)
        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False
        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. Probably your CUDA is not configured correctly. DCN can only be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation                 (current to previous). 'flows_backward' corresponds to the flows used for backward-time                 propagation (current to next).
        """
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated                 features. Each key in the dictionary corresponds to a                 propagation branch, which is represented by a list of tensors.
        """
        n, t, _, h, w = flows.size()
        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current
                feat_prop = feat_prop
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f for f in feat]
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)
            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        outputs = []
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]
        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]
            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()
            outputs.append(hr)
        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False
        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)
        self.check_if_mirror_extended(lqs)
        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, f'The height and width of low-res inputs must be at least 64, but got {h} and {w}.'
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)
                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()
        return self.upsample(lqs, feats)


class BlurFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        grad_input = F.conv2d(grad_output, kernel_flip, padding=1, groups=grad_output.shape[1])
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, _ = ctx.saved_tensors
        grad_input = F.conv2d(gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1])
        return grad_input, None, None


class BlurFunction(Function):

    @staticmethod
    def forward(ctx, x, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)
        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):

    def __init__(self, channel):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel / kernel.sum()
        kernel_flip = torch.flip(kernel, [2, 3])
        self.kernel = kernel.repeat(channel, 1, 1, 1)
        self.kernel_flip = kernel_flip.repeat(channel, 1, 1, 1)

    def forward(self, x):
        return blur(x, self.kernel.type_as(x), self.kernel_flip.type_as(x))


class SFTUpBlock(nn.Module):
    """Spatial feature transform (SFT) with upsampling block.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size in convolutions. Default: 3.
        padding (int): Padding in convolutions. Default: 1.
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(SFTUpBlock, self).__init__()
        self.conv1 = nn.Sequential(Blur(in_channel), spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)), nn.LeakyReLU(0.04, True))
        self.convup = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)), nn.LeakyReLU(0.2, True))
        self.scale_block = nn.Sequential(spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)))
        self.shift_block = nn.Sequential(spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)), nn.Sigmoid())

    def forward(self, x, updated_feat):
        out = self.conv1(x)
        scale = self.scale_block(updated_feat)
        shift = self.shift_block(updated_feat)
        out = out * scale + shift
        out = self.convup(out)
        return out


def AttentionBlock(in_channel):
    return nn.Sequential(spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True), spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)))


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """Conv block used in MSDilationBlock."""
    return nn.Sequential(spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation, bias=bias)), nn.LeakyReLU(0.2), spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation, bias=bias)))


class MSDilationBlock(nn.Module):
    """Multi-scale dilation block."""

    def __init__(self, in_channels, kernel_size=3, dilation=(1, 1, 1, 1), bias=True):
        super(MSDilationBlock, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(4):
            self.conv_blocks.append(conv_block(in_channels, in_channels, kernel_size, dilation=dilation[i], bias=bias))
        self.conv_fusion = spectral_norm(nn.Conv2d(in_channels * 4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias))

    def forward(self, x):
        out = []
        for i in range(4):
            out.append(self.conv_blocks[i](x))
        out = torch.cat(out, 1)
        out = self.conv_fusion(out) + x
        return out


class UpResBlock(nn.Module):

    def __init__(self, in_channel):
        super(UpResBlock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(in_channel, in_channel, 3, 1, 1))

    def forward(self, x):
        out = x + self.body(x)
        return out


NAMES = {'vgg11': ['conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg13': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg16': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], 'vgg19': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']}


VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, range_norm=False, requires_grad=False, remove_pooling=False, pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)
        features = vgg_net.features[:max_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v
        self.vgg_net = nn.Sequential(modified_net)
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output


def calc_mean_std(feat, eps=1e-05):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class DFDNet(nn.Module):
    """DFDNet: Deep Face Dictionary Network.

    It only processes faces with 512x512 size.

    Args:
        num_feat (int): Number of feature channels.
        dict_path (str): Path to the facial component dictionary.
    """

    def __init__(self, num_feat, dict_path):
        super().__init__()
        self.parts = ['left_eye', 'right_eye', 'nose', 'mouth']
        channel_sizes = [128, 256, 512, 512]
        self.feature_sizes = np.array([256, 128, 64, 32])
        self.vgg_layers = ['relu2_2', 'relu3_4', 'relu4_4', 'conv5_4']
        self.flag_dict_device = False
        self.dict = torch.load(dict_path)
        self.vgg_extractor = VGGFeatureExtractor(layer_name_list=self.vgg_layers, vgg_type='vgg19', use_input_norm=True, range_norm=True, requires_grad=False)
        self.attn_blocks = nn.ModuleDict()
        for idx, feat_size in enumerate(self.feature_sizes):
            for name in self.parts:
                self.attn_blocks[f'{name}_{feat_size}'] = AttentionBlock(channel_sizes[idx])
        self.multi_scale_dilation = MSDilationBlock(num_feat * 8, dilation=[4, 3, 2, 1])
        self.upsample0 = SFTUpBlock(num_feat * 8, num_feat * 8)
        self.upsample1 = SFTUpBlock(num_feat * 8, num_feat * 4)
        self.upsample2 = SFTUpBlock(num_feat * 4, num_feat * 2)
        self.upsample3 = SFTUpBlock(num_feat * 2, num_feat)
        self.upsample4 = nn.Sequential(spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1)), nn.LeakyReLU(0.2, True), UpResBlock(num_feat), UpResBlock(num_feat), nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def swap_feat(self, vgg_feat, updated_feat, dict_feat, location, part_name, f_size):
        """swap the features from the dictionary."""
        part_feat = vgg_feat[:, :, location[1]:location[3], location[0]:location[2]].clone()
        part_resize_feat = F.interpolate(part_feat, dict_feat.size()[2:4], mode='bilinear', align_corners=False)
        dict_feat = adaptive_instance_normalization(dict_feat, part_resize_feat)
        similarity_score = F.conv2d(part_resize_feat, dict_feat)
        similarity_score = F.softmax(similarity_score.view(-1), dim=0)
        select_idx = torch.argmax(similarity_score)
        swap_feat = F.interpolate(dict_feat[select_idx:select_idx + 1], part_feat.size()[2:4])
        attn = self.attn_blocks[f'{part_name}_' + str(f_size)](swap_feat - part_feat)
        attn_feat = attn * swap_feat
        updated_feat[:, :, location[1]:location[3], location[0]:location[2]] = attn_feat + part_feat
        return updated_feat

    def put_dict_to_device(self, x):
        if self.flag_dict_device is False:
            for k, v in self.dict.items():
                for kk, vv in v.items():
                    self.dict[k][kk] = vv
            self.flag_dict_device = True

    def forward(self, x, part_locations):
        """
        Now only support testing with batch size = 0.

        Args:
            x (Tensor): Input faces with shape (b, c, 512, 512).
            part_locations (list[Tensor]): Part locations.
        """
        self.put_dict_to_device(x)
        vgg_features = self.vgg_extractor(x)
        updated_vgg_features = []
        batch = 0
        for vgg_layer, f_size in zip(self.vgg_layers, self.feature_sizes):
            dict_features = self.dict[f'{f_size}']
            vgg_feat = vgg_features[vgg_layer]
            updated_feat = vgg_feat.clone()
            for part_idx, part_name in enumerate(self.parts):
                location = (part_locations[part_idx][batch] // (512 / f_size)).int()
                updated_feat = self.swap_feat(vgg_feat, updated_feat, dict_features[part_name], location, part_name, f_size)
            updated_vgg_features.append(updated_feat)
        vgg_feat_dilation = self.multi_scale_dilation(vgg_features['conv5_4'])
        upsampled_feat = self.upsample0(vgg_feat_dilation, updated_vgg_features[3])
        upsampled_feat = self.upsample1(upsampled_feat, updated_vgg_features[2])
        upsampled_feat = self.upsample2(upsampled_feat, updated_vgg_features[1])
        upsampled_feat = self.upsample3(upsampled_feat, updated_vgg_features[0])
        out = self.upsample4(upsampled_feat)
        return out


class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, f'input size must be 128 or 256, but received {input_size}'
        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)
        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, f'Input size must be identical to input_size, but received {x.size()}.'
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))
        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out


class DenseBlocksTemporalReduce(nn.Module):
    """A concatenation of 3 dense blocks with reduction in temporal dimension.

    Note that the output temporal dimension is 6 fewer the input temporal dimension, since there are 3 blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, adapt_official_weights=False):
        super(DenseBlocksTemporalReduce, self).__init__()
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.temporal_reduce1 = nn.Sequential(nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat, num_feat, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
        self.temporal_reduce2 = nn.Sequential(nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + num_grow_ch, num_feat + num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))
        self.temporal_reduce3 = nn.Sequential(nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + 2 * num_grow_ch, num_feat + 2 * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_grow_ch * 3, 1, h, w).
        """
        x1 = self.temporal_reduce1(x)
        x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        x2 = self.temporal_reduce2(x1)
        x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        x3 = self.temporal_reduce3(x2)
        x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        return x3


class DenseBlocks(nn.Module):
    """ A concatenation of N dense blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32.
        num_block (int): Number of dense blocks. The values are:
            DUF-S (16 layers): 3
            DUF-M (18 layers): 9
            DUF-L (52 layers): 21
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_block, num_feat=64, num_grow_ch=16, adapt_official_weights=False):
        super(DenseBlocks, self).__init__()
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(nn.Sequential(nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + i * num_grow_ch, num_feat + i * num_grow_ch, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True), nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True), nn.Conv3d(num_feat + i * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_block * num_grow_ch, t, h, w).
        """
        for i in range(0, len(self.dense_blocks)):
            y = self.dense_blocks[i](x)
            x = torch.cat((x, y), 1)
        return x


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Reference: https://github.com/yhjo09/VSR-DUF

    It only supports input with 3 channels. And it applies the same filters to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters. The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError(f'The type of filter_size must be tuple, but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError(f'The length of filter size must be 2, but got {len(filter_size)}.')
        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        expansion_filter = torch.eye(int(filter_prod)).view(filter_prod, 1, *filter_size)
        self.expansion_filter = expansion_filter.repeat(3, 1, 1, 1)

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            filters (Tensor): Generated dynamic filters. The shape is (n, filter_prod, upsampling_square, h, w).
                filter_prod: prod of filter kernel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle, upsampling_square = upsampling * upsampling.
                e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            Tensor: Filtered image with shape (n, 3*upsampling_square, h, w)
        """
        n, filter_prod, upsampling_square, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = F.conv2d(x, self.expansion_filter, padding=(kh // 2, kw // 2), groups=3)
        expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(0, 3, 4, 1, 2)
        filters = filters.permute(0, 3, 4, 1, 2)
        out = torch.matmul(expanded_input, filters)
        return out.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)


class DUF(nn.Module):
    """Network architecture for DUF

    ``Paper: Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation``

    Reference: https://github.com/yhjo09/VSR-DUF

    For all the models below, 'adapt_official_weights' is only necessary when
    loading the weights converted from the official TensorFlow weights.
    Please set it to False if you are training the model from scratch.

    There are three models with different model size: DUF16Layers, DUF28Layers,
    and DUF52Layers. This class is the base class for these models.

    Args:
        scale (int): The upsampling factor. Default: 4.
        num_layer (int): The number of layers. Default: 52.
        adapt_official_weights_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, scale=4, num_layer=52, adapt_official_weights=False):
        super(DUF, self).__init__()
        self.scale = scale
        if adapt_official_weights:
            eps = 0.001
            momentum = 0.001
        else:
            eps = 1e-05
            momentum = 0.1
        self.conv3d1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dynamic_filter = DynamicUpsamplingFilter((5, 5))
        if num_layer == 16:
            num_block = 3
            num_grow_ch = 32
        elif num_layer == 28:
            num_block = 9
            num_grow_ch = 16
        elif num_layer == 52:
            num_block = 21
            num_grow_ch = 16
        else:
            raise ValueError(f'Only supported (16, 28, 52) layers, but got {num_layer}.')
        self.dense_block1 = DenseBlocks(num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch, adapt_official_weights=adapt_official_weights)
        self.dense_block2 = DenseBlocksTemporalReduce(64 + num_grow_ch * num_block, num_grow_ch, adapt_official_weights=adapt_official_weights)
        channels = 64 + num_grow_ch * num_block + num_grow_ch * 3
        self.bn3d2 = nn.BatchNorm3d(channels, eps=eps, momentum=momentum)
        self.conv3d2 = nn.Conv3d(channels, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * scale ** 2, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f2 = nn.Conv3d(512, 1 * 5 * 5 * scale ** 2, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape (b, 7, c, h, w)

        Returns:
            Tensor: Output with shape (b, c, h * scale, w * scale)
        """
        num_batches, num_imgs, _, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x_center = x[:, :, num_imgs // 2, :, :]
        x = self.conv3d1(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = F.relu(self.bn3d2(x), inplace=True)
        x = F.relu(self.conv3d2(x), inplace=True)
        res = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))
        filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))
        filter_ = F.softmax(filter_.view(num_batches, 25, self.scale ** 2, h, w), dim=1)
        out = self.dynamic_filter(x_center, filter_)
        out += res.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)
        return out


class SeqConv3x3(nn.Module):
    """The re-parameterizable block used in the ECBSR architecture.

    ``Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices``

    Reference: https://github.com/xindongzhang/ECBSR

    Args:
        seq_type (str): Sequence type, option: conv1x1-conv3x3 | conv1x1-sobelx | conv1x1-sobely | conv1x1-laplacian.
        in_channels (int): Channel number of input.
        out_channels (int): Channel number of output.
        depth_multiplier (int): Width multiplier in the expand-and-squeeze conv. Default: 1.
    """

    def __init__(self, seq_type, in_channels, out_channels, depth_multiplier=1):
        super(SeqConv3x3, self).__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.seq_type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_channels * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.in_channels, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_channels, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.seq_type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 0.001
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_channels) * 0.001
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.seq_type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 0.001
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 0.001
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.seq_type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 0.001
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 0.001
            bias = torch.reshape(bias, (self.out_channels,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('The type of seqconv is not supported!')

    def forward(self, x):
        if self.seq_type == 'conv1x1-conv3x3':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_channels)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        if self.seq_type == 'conv1x1-conv3x3':
            rep_weight = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            rep_bias = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=self.k1).view(-1) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=device)
            for i in range(self.out_channels):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            rep_weight = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            rep_bias = torch.ones(1, self.out_channels, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=k1).view(-1) + b1
        return rep_weight, rep_bias


class ECB(nn.Module):
    """The ECB block used in the ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        in_channels (int): Channel number of input.
        out_channels (int): Channel number of output.
        depth_multiplier (int): Width multiplier in the expand-and-squeeze conv. Default: 1.
        act_type (str): Activation type. Option: prelu | relu | rrelu | softplus | linear. Default: prelu.
        with_idt (bool): Whether to use identity connection. Default: False.
    """

    def __init__(self, in_channels, out_channels, depth_multiplier, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type
        if with_idt and self.in_channels == self.out_channels:
            self.with_idt = True
        else:
            self.with_idt = False
        self.conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.in_channels, self.out_channels, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_channels)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            rep_weight, rep_bias = self.rep_params()
            y = F.conv2d(input=x, weight=rep_weight, bias=rep_bias, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        weight0, bias0 = self.conv3x3.weight, self.conv3x3.bias
        weight1, bias1 = self.conv1x1_3x3.rep_params()
        weight2, bias2 = self.conv1x1_sbx.rep_params()
        weight3, bias3 = self.conv1x1_sby.rep_params()
        weight4, bias4 = self.conv1x1_lpl.rep_params()
        rep_weight, rep_bias = weight0 + weight1 + weight2 + weight3 + weight4, bias0 + bias1 + bias2 + bias3 + bias4
        if self.with_idt:
            device = rep_weight.get_device()
            if device < 0:
                device = None
            weight_idt = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=device)
            for i in range(self.out_channels):
                weight_idt[i, i, 1, 1] = 1.0
            bias_idt = 0.0
            rep_weight, rep_bias = rep_weight + weight_idt, rep_bias + bias_idt
        return rep_weight, rep_bias


class ECBSR(nn.Module):
    """ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_block (int): Block number in the trunk network.
        num_channel (int): Channel number.
        with_idt (bool): Whether use identity in convolution layers.
        act_type (str): Activation type.
        scale (int): Upsampling factor.
    """

    def __init__(self, num_in_ch, num_out_ch, num_block, num_channel, with_idt, act_type, scale):
        super(ECBSR, self).__init__()
        self.num_in_ch = num_in_ch
        self.scale = scale
        backbone = []
        backbone += [ECB(num_in_ch, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        for _ in range(num_block):
            backbone += [ECB(num_channel, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        backbone += [ECB(num_channel, num_out_ch * scale * scale, depth_multiplier=2.0, act_type='linear', with_idt=with_idt)]
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.scale * self.scale, dim=1)
        else:
            shortcut = x
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return y


class EDSR(nn.Module):
    """EDSR network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=16, upscale=4, res_scale=1, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.404)):
        super(EDSR, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))
        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))
        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_frame=5, deformable_groups=8, num_extract_block=5, num_reconstruct_block=10, center_frame_idx=None, hr_in=False, with_predeblur=False, with_tsa=True):
        super(EDVR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, 'The height and width must be multiple of 16.'
        else:
            assert h % 4 == 0 and w % 4 == 0, 'The height and width must be multiple of 4.'
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)
        ref_feat_l = [feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(), feat_l3[:, self.center_frame_idx, :, :, :].clone()]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out


class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'syncbatch':
            None
            self.param_free_norm = nn.InstanceNorm2d(norm_nc)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(f'{param_free_norm_type} is not a recognized param-free norm type in SPADE')
        nhidden = 128 if norm_nc > 128 else norm_nc
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, bias=False)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * gamma + beta
        return out


class SPADEResnetBlock(nn.Module):
    """
    ResNet block that uses SPADE. It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    This architecture seemed like a standard architecture for unconditional or
    class-conditional GAN architecture using residual block.
    The code was inspired from https://github.com/LMescheder/GAN_stability.
    """

    def __init__(self, fin, fout, norm_g='spectralspadesyncbatch3x3', semantic_nc=3):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        if 'spectral' in norm_g:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        spade_config_str = norm_g.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.act(self.norm_0(x, seg)))
        dx = self.conv_1(self.act(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def act(self, x):
        return F.leaky_relu(x, 0.2)


class BaseNetwork(nn.Module):
    """ A basis for hifacegan archs with custom initialization """

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def forward(self, x):
        pass


class SoftGate(nn.Module):
    COEFF = 12.0

    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


class SimplifiedLIP(nn.Module):

    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()
        self.logit = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.InstanceNorm2d(channels, affine=True), SoftGate())

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac


class LIPEncoder(BaseNetwork):
    """Local Importance-based Pooling (Ziteng Gao et.al.,ICCV 2019)"""

    def __init__(self, input_nc, ngf, sw, sh, n_2xdown, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        kw = 3
        pw = (kw - 1) // 2
        model = [nn.Conv2d(input_nc, ngf, kw, stride=1, padding=pw, bias=False), norm_layer(ngf), nn.ReLU()]
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio * 2, self.max_ratio)
            model += [SimplifiedLIP(ngf * cur_ratio), nn.Conv2d(ngf * cur_ratio, ngf * next_ratio, kw, stride=1, padding=pw), norm_layer(ngf * next_ratio)]
            cur_ratio = next_ratio
            if i < n_2xdown - 1:
                model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

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


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

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


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

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


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

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


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


LOCAL_FID_WEIGHTS = 'experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation.

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    try:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False, init_weights=False)
    except TypeError:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    if os.path.exists(LOCAL_FID_WEIGHTS):
        state_dict = torch.load(LOCAL_FID_WEIGHTS, map_location=lambda storage, loc: storage)
    else:
        state_dict = load_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=DEFAULT_BLOCK_INDEX, resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3.

        Args:
            output_blocks (list[int]): Indices of blocks to return features of.
                Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
            resize_input (bool): If true, bilinearly resizes input to width and
                height 299 before feeding input to model. As the network
                without fully connected layers is fully convolutional, it
                should be able to handle inputs of arbitrary size, so resizing
                might not be strictly needed. Default: True.
            normalize_input (bool): If true, scales the input from range (0, 1)
                to the range the pretrained Inception network expects,
                namely (-1, 1). Default: True.
            requires_grad (bool): If true, parameters of the model require
                gradients. Possibly useful for finetuning the network.
                Default: False.
            use_fid_inception (bool): If true, uses the pretrained Inception
                model used in Tensorflow's FID implementation.
                If false, uses the pretrained Inception model available in
                torchvision. The FID Inception model has different weights
                and a slightly different structure from torchvision's
                Inception model. If you want to compute FID scores, you are
                strongly advised to set this parameter to true to get
                comparable results. Default: True.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            try:
                inception = models.inception_v3(pretrained=True, init_weights=False)
            except TypeError:
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

    def forward(self, x):
        """Get Inception feature maps.

        Args:
            x (Tensor): Input tensor of shape (b, 3, h, w).
                Values are expected to be in range (-1, 1). You can also input
                (0, 1) with setting normalize_input = True.

        Returns:
            list[Tensor]: Corresponding to the selected output block, sorted
            ascending by index.
        """
        output = []
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                output.append(x)
            if idx == self.last_needed_block:
                break
        return output


class ChannelAttention(nn.Module):
    """Channel attention.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default:
    """

    def __init__(self, mid_channels, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid_channels, mid_channels // squeeze_factor, 1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(mid_channels // squeeze_factor, mid_channels, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale
        self.rcab = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()
        self.residual_group = make_layer(RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class RCAN(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_group=10, num_block=16, squeeze_factor=16, upscale=4, res_scale=1, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.404)):
        super(RCAN, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualGroup, num_group, num_feat=num_feat, num_block=num_block, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x


class MeanShift(nn.Conv2d):
    """ Data normalization with mean and std.

    Args:
        rgb_range (int): Maximum value of RGB.
        rgb_mean (list[float]): Mean for RGB channels.
        rgb_std (list[float]): Std for RGB channels.
        sign (int): For subtraction, sign is -1, for addition, sign is 1.
            Default: -1.
        requires_grad (bool): Whether to update the self.weight and self.bias.
            Default: True.
    """

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1, requires_grad=True):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = requires_grad


class EResidualBlockNoBN(nn.Module):
    """Enhanced Residual block without BN.

    There are three convolution layers in residual branch.
    """

    def __init__(self, in_channels, out_channels):
        super(EResidualBlockNoBN, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 1, 1, 0))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        out = self.relu(out + x)
        return out


class MergeRun(nn.Module):
    """ Merge-and-run unit.

    This unit contains two branches with different dilated convolutions,
    followed by a convolution to process the concatenated features.

    Paper: Real Image Denoising with Feature Attention
    Ref git repo: https://github.com/saeed-anwar/RIDNet
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MergeRun, self).__init__()
        self.dilation1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size, stride, 2, 2), nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 3, 3), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size, stride, 4, 4), nn.ReLU(inplace=True))
        self.aggregation = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True))

    def forward(self, x):
        dilation1 = self.dilation1(x)
        dilation2 = self.dilation2(x)
        out = torch.cat([dilation1, dilation2], dim=1)
        out = self.aggregation(out)
        out = out + x
        return out


class EAM(nn.Module):
    """Enhancement attention modules (EAM) in RIDNet.

    This module contains a merge-and-run unit, a residual block,
    an enhanced residual block and a feature attention unit.

    Attributes:
        merge: The merge-and-run unit.
        block1: The residual block.
        block2: The enhanced residual block.
        ca: The feature/channel attention unit.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(EAM, self).__init__()
        self.merge = MergeRun(in_channels, mid_channels)
        self.block1 = ResidualBlockNoBN(mid_channels)
        self.block2 = EResidualBlockNoBN(mid_channels, out_channels)
        self.ca = ChannelAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.merge(x)
        out = self.relu(self.block1(out))
        out = self.block2(out)
        out = self.ca(out)
        return out


class RIDNet(nn.Module):
    """RIDNet: Real Image Denoising with Feature Attention.

    Ref git repo: https://github.com/saeed-anwar/RIDNet

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of EAM modules.
            Default: 64.
        out_channels (int): Channel number of outputs.
        num_block (int): Number of EAM. Default: 4.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self, in_channels, mid_channels, out_channels, num_block=4, img_range=255.0, rgb_mean=(0.4488, 0.4371, 0.404), rgb_std=(1.0, 1.0, 1.0)):
        super(RIDNet, self).__init__()
        self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)
        self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)
        self.head = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(EAM, num_block, in_channels=mid_channels, mid_channels=mid_channels, out_channels=mid_channels)
        self.tail = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.sub_mean(x)
        res = self.tail(self.body(self.relu(self.head(res))))
        res = self.add_mean(res)
        out = x + res
        return out


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * scale ** 2
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
        grad_input = upfirdn2d_ext.upfirdn2d(grad_output, grad_kernel, down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
        ctx.save_for_backward(kernel)
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors
        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
        gradgrad_out = upfirdn2d_ext.upfirdn2d(gradgrad_input, kernel, ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y, ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1)
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1])
        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad
        kernel_h, kernel_w = kernel.shape
        _, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape
        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = out_h, out_w
        ctx.up = up_x, up_y
        ctx.down = down_x, down_y
        ctx.pad = pad_x0, pad_x1, pad_y0, pad_y1
        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        ctx.g_pad = g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
        out = upfirdn2d_ext.upfirdn2d(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
        out = out.view(-1, channel, out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors
        grad_input = UpFirDn2dBackward.apply(grad_output, kernel, grad_kernel, ctx.up, ctx.down, ctx.pad, ctx.g_pad, ctx.in_size, ctx.out_size)
        return grad_input, None, None, None, None


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if input.device.type == 'cpu':
        out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    else:
        out = UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return out


class UpFirDnUpsample(nn.Module):
    """Upsample, FIR filter, and downsample (upsampole version).

    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * factor ** 2
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2 + factor - 1, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnDownsample(nn.Module):
    """Upsample, FIR filter, and downsample (downsampole version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor
        pad = self.kernel.shape[0] - factor
        self.pad = (pad + 1) // 2, pad // 2

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Default: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * upsample_factor ** 2
        if upsample_factor > 1:
            pad = self.kernel.shape[0] - upsample_factor - (kernel_size - 1)
            self.pad = (pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1
        elif downsample_factor > 1:
            pad = self.kernel.shape[0] - downsample_factor + (kernel_size - 1)
            self.pad = (pad + 1) // 2, pad // 2
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}, downsample_factor={self.downsample_factor})'


class FusedLeakyReLUFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        empty = grad_output.new_empty(0)
        grad_input = fused_act_ext.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)
        dim = [0]
        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))
        grad_bias = grad_input.sum(dim).detach()
        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused_act_ext.fused_bias_act(gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale)
        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused_act_ext.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(grad_output, out, ctx.negative_slope, ctx.scale)
        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f"Wrong activation value in EqualLinear: {activation}Supported ones are: ['fused_lrelu', None].")
        self.scale = 1 / math.sqrt(in_channels) * lr_mul
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias is not None})'


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None, eps=1e-08, interpolation_mode='bilinear'):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.interpolation_mode = interpolation_mode
        if self.interpolation_mode == 'nearest':
            self.align_corners = None
        else:
            self.align_corners = False
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.modulation = EqualLinear(num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.sample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode, align_corners=self.align_corners)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x, scale_factor=0.5, mode=self.interpolation_mode, align_corners=self.align_corners)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})'


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


class StyleConv(nn.Module):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None, interpolation_mode='bilinear'):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode, interpolation_mode=interpolation_mode)
        self.weight = nn.Parameter(torch.zeros(1))
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, interpolation_mode='bilinear'):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.interpolation_mode = interpolation_mode
        if self.interpolation_mode == 'nearest':
            self.align_corners = None
        else:
            self.align_corners = False
        self.modulated_conv = ModulatedConv2d(in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None, interpolation_mode=interpolation_mode)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode=self.interpolation_mode, align_corners=self.align_corners)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), lr_mlp=0.01, narrow=1):
        super(StyleGAN2Generator, self).__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(EqualLinear(num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        channels = {'4': int(512 * narrow), '8': int(512 * narrow), '16': int(512 * narrow), '32': int(512 * narrow), '64': int(256 * channel_multiplier * narrow), '128': int(128 * channel_multiplier * narrow), '256': int(64 * channel_multiplier * narrow), '512': int(32 * channel_multiplier * narrow), '1024': int(16 * channel_multiplier * narrow)}
        self.channels = channels
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(channels['4'], channels['4'], kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False, resample_kernel=resample_kernel)
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channels = channels['4']
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.style_convs.append(StyleConv(in_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode='upsample', resample_kernel=resample_kernel))
            self.style_convs.append(StyleConv(out_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self, styles, input_is_latent=False, noise=None, randomize_noise=True, truncation=1, truncation_latent=None, inject_index=None, return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})'


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, bias=True, activate=True, interpolation_mode='bilinear'):
        layers = []
        self.interpolation_mode = interpolation_mode
        if downsample:
            if self.interpolation_mode == 'nearest':
                self.align_corners = None
            else:
                self.align_corners = False
            layers.append(torch.nn.Upsample(scale_factor=0.5, mode=interpolation_mode, align_corners=self.align_corners))
        stride = 1
        self.padding = kernel_size // 2
        layers.append(EqualConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias and not activate))
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))
        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels, interpolation_mode='bilinear'):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(in_channels, out_channels, 3, downsample=True, interpolation_mode=interpolation_mode, bias=True, activate=True)
        self.skip = ConvLayer(in_channels, out_channels, 1, downsample=True, interpolation_mode=interpolation_mode, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator, self).__init__()
        channels = {'4': int(512 * narrow), '8': int(512 * narrow), '16': int(512 * narrow), '32': int(512 * narrow), '64': int(256 * channel_multiplier * narrow), '128': int(128 * channel_multiplier * narrow), '256': int(64 * channel_multiplier * narrow), '512': int(32 * channel_multiplier * narrow), '1024': int(16 * channel_multiplier * narrow)}
        log_size = int(math.log(out_size, 2))
        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]
        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2 ** (i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)
        self.final_conv = ConvLayer(in_channels + 1, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(EqualLinear(channels['4'] * 4 * 4, channels['4'], bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'), EqualLinear(channels['4'], 1, bias=True, bias_init_val=0, lr_mul=1, activation=None))
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)
        b, c, h, w = out.shape
        group = min(b, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-08)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)
        return out


class StyleGAN2GeneratorBilinear(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, lr_mlp=0.01, narrow=1, interpolation_mode='bilinear'):
        super(StyleGAN2GeneratorBilinear, self).__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(EqualLinear(num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        channels = {'4': int(512 * narrow), '8': int(512 * narrow), '16': int(512 * narrow), '32': int(512 * narrow), '64': int(256 * channel_multiplier * narrow), '128': int(128 * channel_multiplier * narrow), '256': int(64 * channel_multiplier * narrow), '512': int(32 * channel_multiplier * narrow), '1024': int(16 * channel_multiplier * narrow)}
        self.channels = channels
        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(channels['4'], channels['4'], kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, interpolation_mode=interpolation_mode)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False, interpolation_mode=interpolation_mode)
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        in_channels = channels['4']
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            self.style_convs.append(StyleConv(in_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode='upsample', interpolation_mode=interpolation_mode))
            self.style_convs.append(StyleConv(out_channels, out_channels, kernel_size=3, num_style_feat=num_style_feat, demodulate=True, sample_mode=None, interpolation_mode=interpolation_mode))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, interpolation_mode=interpolation_mode))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self, styles, input_is_latent=False, noise=None, randomize_noise=True, truncation=1, truncation_latent=None, inject_index=None, return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        flops = 0
        flops += n * self.dim * 3 * self.dim
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += h // 2 * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """ Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, scale ** 2 * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    """ SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.0, upsampler='', resi_connection='1conv', **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = 0.4488, 0.4371, 0.404
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint, img_size=img_size, patch_size=patch_size, resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
        x = x / self.img_range + self.mean
        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class SPyNetTOF(nn.Module):
    """SPyNet architecture for TOF.

    Note that this implementation is specifically for TOFlow. Please use :file:`spynet_arch.py` for general use.
    They differ in the following aspects:

    1. The basic modules here contain BatchNorm.
    2. Normalization and denormalization are not done here, as they are done in TOFlow.

    ``Paper: Optical Flow Estimation using a Spatial Pyramid Network``

    Reference: https://github.com/Coldog2333/pytoflow

    Args:
        load_path (str): Path for pretrained SPyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SPyNetTOF, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(4)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, ref, supp):
        """
        Args:
            ref (Tensor): Reference image with shape of (b, 3, h, w).
            supp: The supporting image to be warped: (b, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (b, 2, h, w).
        """
        num_batches, _, h, w = ref.size()
        ref = [ref]
        supp = [supp]
        for _ in range(3):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = ref[0].new_zeros(num_batches, 2, h // 16, w // 16)
        for i in range(4):
            flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[i](torch.cat([ref[i], flow_warp(supp[i], flow_up.permute(0, 2, 3, 1)), flow_up], 1))
        return flow


class TOFlow(nn.Module):
    """PyTorch implementation of TOFlow.

    In TOFlow, the LR frames are pre-upsampled and have the same size with the GT frames.

    ``Paper: Video Enhancement with Task-Oriented Flow``

    Reference: https://github.com/anchen1011/toflow

    Reference: https://github.com/Coldog2333/pytoflow

    Args:
        adapt_official_weights (bool): Whether to adapt the weights translated
            from the official implementation. Set to false if you want to
            train from scratch. Default: False
    """

    def __init__(self, adapt_official_weights=False):
        super(TOFlow, self).__init__()
        self.adapt_official_weights = adapt_official_weights
        self.ref_idx = 0 if adapt_official_weights else 3
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.spynet = SPyNetTOF()
        self.conv_1 = nn.Conv2d(3 * 7, 64, 9, 1, 4)
        self.conv_2 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_3 = nn.Conv2d(64, 64, 1)
        self.conv_4 = nn.Conv2d(64, 3, 1)
        self.relu = nn.ReLU(inplace=True)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, lrs):
        """
        Args:
            lrs: Input lr frames: (b, 7, 3, h, w).

        Returns:
            Tensor: SR frame: (b, 3, h, w).
        """
        if self.adapt_official_weights:
            lrs = lrs[:, [3, 0, 1, 2, 4, 5, 6], :, :, :]
        num_batches, num_lrs, _, h, w = lrs.size()
        lrs = self.normalize(lrs.view(-1, 3, h, w))
        lrs = lrs.view(num_batches, num_lrs, 3, h, w)
        lr_ref = lrs[:, self.ref_idx, :, :, :]
        lr_aligned = []
        for i in range(7):
            if i == self.ref_idx:
                lr_aligned.append(lr_ref)
            else:
                lr_supp = lrs[:, i, :, :, :]
                flow = self.spynet(lr_ref, lr_supp)
                lr_aligned.append(flow_warp(lr_supp, flow.permute(0, 2, 3, 1)))
        hr = torch.stack(lr_aligned, dim=1)
        hr = hr.view(num_batches, -1, h, w)
        hr = self.relu(self.conv_1(hr))
        hr = self.relu(self.conv_2(hr))
        hr = self.relu(self.conv_3(hr))
        hr = self.conv_4(hr) + lr_ref
        return self.denormalize(hr)


LOSS_REGISTRY = Registry('loss')


_reduction_modes = ['none', 'mean', 'sum']


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss
    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]
        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)
        loss = x_diff + y_diff
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True, range_norm=False, perceptual_weight=1.0, style_weight=0.0, criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, range_norm=range_norm)
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:
            loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_ext.deform_conv_forward(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_ext.deform_conv_backward_input(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_ext.deform_conv_backward_parameters(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(f"convolution input is too small (output would be {'x'.join(map(str, output_size))})")
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, f'in_channels {in_channels} is not divisible by groups {groups}'
        assert out_channels % groups == 0, f'out_channels {out_channels} is not divisible by groups {groups}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        input_pad = x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1]
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()
        return out


class DeformConvPack(DeformConv):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class RGB2YCbCrJpeg(nn.Module):
    """ Converts RGB image to YCbCr
    """

    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0.0, 128.0, 128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(Tensor): batch x 3 x height x width

        Returns:
            Tensor: batch x height x width x 3
        """
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class ChromaSubsampling(nn.Module):
    """ Chroma subsampling on CbCr channels
    """

    def __init__(self):
        super(ChromaSubsampling, self).__init__()

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image_2[:, 1, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cr = F.avg_pool2d(image_2[:, 2, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Module):
    """ Splitting image into patches
    """

    def __init__(self):
        super(BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class DCT8x8(nn.Module):
    """ Discrete Cosine Transformation
    """

    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class YQuantize(nn.Module):
    """ JPEG Quantization for Y channel

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CQuantize(nn.Module):
    """ JPEG Quantization for CbCr channels

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CompressJpeg(nn.Module):
    """Full JPEG compression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x 3 x height x width

        Returns:
            dict(tensor): Compressed tensor with batch x h*w/64 x 8 x 8.
        """
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)
            components[k] = comp
        return components['y'], components['cb'], components['cr']


class YDequantize(nn.Module):
    """Dequantize Y channel
    """

    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class CDequantize(nn.Module):
    """Dequantize CbCr channel
    """

    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class iDCT8x8(nn.Module):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self):
        super(iDCT8x8, self).__init__()
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class BlockMerging(nn.Module):
    """Merge patches into image
    """

    def __init__(self):
        super(BlockMerging, self).__init__()

    def forward(self, patches, height, width):
        """
        Args:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)

        Returns:
            Tensor: batch x height x width
        """
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
    """Upsample chroma layers
    """

    def __init__(self):
        super(ChromaUpsampling, self).__init__()

    def forward(self, y, cb, cr):
        """
        Args:
            y(tensor): y channel image
            cb(tensor): cb channel
            cr(tensor): cr channel

        Returns:
            Tensor: batch x height x width x 3
        """

        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x
        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCr2RGBJpeg(nn.Module):
    """Converts YCbCr image to RGB JPEG
    """

    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()
        matrix = np.array([[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128.0, -128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            Tensor: batch x 3 x height x width
        """
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class DeCompressJpeg(nn.Module):
    """Full JPEG decompression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        """
        Args:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            imgh(int)
            imgw(int)
            factor(float)

        Returns:
            Tensor: batch x 3 x height x width
        """
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255


def diff_round(x):
    """ Differentiable rounding function
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality

    Args:
        quality(float): Quality for jpeg compression.

    Returns:
        float: Compression factor.
    """
    if quality < 50:
        quality = 5000.0 / quality
    else:
        quality = 200.0 - quality * 2
    return quality / 100.0


class DiffJPEG(nn.Module):
    """This JPEG algorithm result is slightly different from cv2.
    DiffJPEG supports batch processing.

    Args:
        differentiable(bool): If True, uses custom differentiable rounding function, if False, uses standard torch.round
    """

    def __init__(self, differentiable=True):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def forward(self, x, quality):
        """
        Args:
            x (Tensor): Input image, bchw, rgb, [0, 1]
            quality(float): Quality factor for jpeg compression scheme.
        """
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            for i in range(factor.size(0)):
                factor[i] = quality_to_factor(factor[i])
        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, h + h_pad, w + w_pad, factor=factor)
        recovered = recovered[:, :, 0:h, 0:w]
        return recovered


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')
    ph, pw = img.size()[-2:]
    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur
        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


class ToyDiscriminator(nn.Module):

    def __init__(self):
        super(ToyDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(3, 4, 3, 1, 1, bias=True)
        self.bn0 = nn.BatchNorm2d(4, affine=True)
        self.conv1 = nn.Conv2d(4, 4, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(4, affine=True)
        self.linear = nn.Linear(4 * 6 * 6, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.bn0(self.conv0(x)))
        feat = self.lrelu(self.bn1(self.conv1(feat)))
        feat = feat.view(feat.size(0), -1)
        out = torch.sigmoid(self.linear(feat))
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (Blur,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChromaSubsampling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvResidualBlocks,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ECB,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'depth_multiplier': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EResidualBlockNoBN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSDilationBlock,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSRResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MergeRun,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModulatedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'num_style_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (NormStyleCode,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PredeblurModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RRDB,
     lambda: ([], {'num_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RRDBNet,
     lambda: ([], {'num_in_ch': 4, 'num_out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualBlockNoBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualDenseBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SFTUpBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPADEResnetBlock,
     lambda: ([], {'fin': 4, 'fout': 4}),
     lambda: ([torch.rand([4, 1, 64, 64]), torch.rand([4, 3, 4, 4])], {}),
     False),
    (SPyNetTOF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 64, 64]), torch.rand([4, 4, 64, 64])], {}),
     False),
    (SRVGGNetCompact,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ScaledLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimplifiedLIP,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TOFlow,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 7, 3, 64, 64])], {}),
     False),
    (TSAFusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 4, 4])], {}),
     True),
    (ToRGB,
     lambda: ([], {'in_channels': 4, 'num_style_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (UNetDiscriminatorSN,
     lambda: ([], {'num_in_ch': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (UpResBlock,
     lambda: ([], {'in_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedTVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (YCbCr2RGBJpeg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 3])], {}),
     True),
    (iDCT8x8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     True),
]

class Test_XPixelGroup_BasicSR(_paritybench_base):
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

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

