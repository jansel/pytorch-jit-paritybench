import sys
_module = sys.modules[__name__]
del sys
create_data = _module
AFHQ_128 = _module
AFHQ_128_test = _module
Fid = _module
fid_score = _module
inception = _module
celebA_128 = _module
celebA_128_test = _module
cifar10_test = _module
cifar10_train = _module
Model2 = _module
deblurring_diffusion_pytorch = _module
cal_Fid = _module
deblurring_diffusion_pytorch = _module
dispatch = _module
mnist_test = _module
mnist_train = _module
fid_score = _module
inception = _module
diffusion = _module
diffusion = _module
forward_process_impl = _module
get_dataset = _module
model = _module
get_model = _module
unet_convnext = _module
unet_resnet = _module
utils = _module
setup = _module
test = _module
train = _module
fid_score = _module
inception = _module
celebA_test = _module
celebA_train = _module
cifar10_train_wd = _module
defading_diffusion_pytorch = _module
defading_diffusion_gaussian = _module
defading_diffusion_naive = _module
new_model = _module
fid_score = _module
inception = _module
celebA_128 = _module
celebA_128_test = _module
Model2 = _module
defading_diffusion_pytorch = _module
AFHQ_128_to_celebA_128 = _module
AFHQ_128_to_celebA_128_test = _module
fid_score = _module
inception = _module
Model2 = _module
demixing_diffusion_pytorch = _module
demixing_diffusion_pytorch = _module
AFHQ_noise_128 = _module
AFHQ_noise_128_test = _module
fid_score = _module
inception = _module
celebA_noise_128 = _module
celebA_noise_128_test = _module
Model2 = _module
denoising_diffusion_pytorch = _module
denoising_diffusion_pytorch = _module
fid_score = _module
inception = _module
celebA = _module
celebA_128 = _module
celebA_128_test = _module
cifar10_test = _module
Model2 = _module
resolution_diffusion_pytorch = _module
resolution_diffusion_pytorch = _module
fid_score = _module
inception = _module
diffusion = _module
forward_process_impl = _module
unet_convnext = _module
unet_resnet = _module
utils = _module
setup = _module
test = _module

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


import torchvision


import torch


import random


import numpy as np


import torchvision.transforms as TF


from scipy import linalg


from torch.nn.functional import adaptive_avg_pool2d


import torch.nn as nn


import torch.nn.functional as F


import math


import copy


from torch import nn


from torch import einsum


from inspect import isfunction


from functools import partial


from torch.utils import data


from torch.optim import Adam


from torchvision import transforms


from torchvision import utils


from collections import OrderedDict


import time


from torchvision import datasets


import matplotlib.pyplot as plt


import matplotlib.image as mpimg


from torch import linalg as LA


import torch.linalg


from scipy.ndimage import zoom as scizoom


import torch.nn.functional as func


from sklearn.mixture import GaussianMixture


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`

    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
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


class FIDInceptionC(torchvision.models.inception.InceptionC):
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


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
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


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
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


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=(DEFAULT_BLOCK_INDEX,), resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
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
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)
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

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
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


class Upsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-06, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * int(c) ** -0.5
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Model(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch), torch.nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def exists(x):
    return x is not None


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(LayerNorm(dim) if norm else nn.Identity(), nn.Conv2d(dim, dim_out * mult, 3, padding=1), nn.GELU(), nn.Conv2d(dim_out * mult, dim_out, 3, padding=1))
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')
        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Unet(nn.Module):

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, with_time_emb=True, residual=False):
        super().__init__()
        self.channels = channels
        self.residual = residual
        None
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        else:
            time_dim = None
            self.time_mlp = None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0), ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Downsample(dim_out) if not is_last else nn.Identity()]))
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(nn.ModuleList([ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim), ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Upsample(dim_in) if not is_last else nn.Identity()]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(ConvNextBlock(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []
        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x
        return self.final_conv(x)


class ForwardProcessBase:

    def forward(self, x, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass


def lab2rgb(image: torch.Tensor, clip: bool=True) ->torch.Tensor:
    """Convert a Lab image to RGB.

    Args:
        image: Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - _b / 200.0
    fz = fz.clamp(min=0.0)
    fxyz = torch.stack([fx, fy, fz], dim=-3)
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white
    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)
    rgb_im = linear_rgb_to_rgb(rgbs_im)
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)
    rgb_im = 2.0 * rgb_im - 1
    return rgb_im


def rgb2lab(image_old: torch.Tensor) ->torch.Tensor:
    """Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -127..127.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    image = (image_old + 1) * 0.5
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(image)}')
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {image.shape}')
    lin_rgb = rgb_to_linear_rgb(image)
    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)
    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)
    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]
    L: torch.Tensor = 116.0 * y - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)
    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)
    return out


class DeColorization(ForwardProcessBase):

    def __init__(self, decolor_routine='Constant', decolor_ema_factor=0.9, decolor_total_remove=False, num_timesteps=50, channels=3, to_lab=False):
        self.decolor_routine = decolor_routine
        self.decolor_ema_factor = decolor_ema_factor
        self.decolor_total_remove = decolor_total_remove
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.device_of_kernel = 'cuda'
        self.kernels = self.get_kernels()
        self.to_lab = to_lab

    def get_conv(self, decolor_ema_factor):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular', bias=False)
        with torch.no_grad():
            ori_color_weight = torch.eye(self.channels)[:, :, None, None]
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_ema_factor * ori_color_weight + (1.0 - decolor_ema_factor) * decolor_weight
            conv.weight = nn.Parameter(kernel)
        if self.device_of_kernel == 'cuda':
            conv = conv
        return conv

    def get_kernels(self):
        kernels = []
        if self.decolor_routine == 'Constant':
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0))
                else:
                    kernels.append(self.get_conv(self.decolor_ema_factor))
        elif self.decolor_routine == 'Linear':
            diff = 1.0 / self.num_timesteps
            start = 1.0
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0))
                else:
                    ema_factor = 1 - diff / start
                    start = start * ema_factor
                    kernels.append(self.get_conv(ema_factor))
        return kernels

    def forward(self, x, i, og=None):
        if self.to_lab:
            x_rgb = lab2rgb(x)
            x_next = self.kernels[i](x_rgb)
            return rgb2lab(x_next)
        else:
            return self.kernels[i](x)

    def total_forward(self, x_in):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular', bias=False)
        if self.to_lab:
            x = lab2rgb(x_in)
        else:
            x = x_in
        with torch.no_grad():
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_weight
            conv.weight = nn.Parameter(kernel)
        if self.device_of_kernel == 'cuda':
            conv = conv
        x_out = conv(x)
        if self.to_lab:
            x_out = rgb2lab(x_out)
        return x_out


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


class Snow(ForwardProcessBase):

    def __init__(self, image_size=(32, 32), snow_level=1, num_timesteps=50, snow_base_path=None, random_snow=False, single_snow=False, batch_size=32, load_snow_base=False, fix_brightness=False):
        self.num_timesteps = num_timesteps
        self.random_snow = random_snow
        self.snow_level = snow_level
        self.image_size = image_size
        self.single_snow = single_snow
        self.batch_size = batch_size
        self.generate_snow_layer()
        self.fix_brightness = fix_brightness

    @torch.no_grad()
    def reset_parameters(self, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size
        if self.random_snow:
            self.generate_snow_layer()

    @torch.no_grad()
    def generate_snow_layer(self):
        if not self.random_snow:
            rstate = np.random.get_state()
            np.random.seed(123321)
        if self.snow_level == 1:
            c = 0.1, 0.3, 3, 0.5, 5, 4, 0.8
            snow_thres_start = 0.7
            snow_thres_end = 0.3
            mb_sigma_start = 0.5
            mb_sigma_end = 5.0
            br_coef_start = 0.95
            br_coef_end = 0.7
        elif self.snow_level == 2:
            c = 0.55, 0.3, 2.5, 0.85, 11, 12, 0.55
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 12
            br_coef_start = 0.95
            br_coef_end = 0.55
        elif self.snow_level == 3:
            c = 0.55, 0.3, 2.5, 0.7, 11, 16, 0.4
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 16
            br_coef_start = 0.95
            br_coef_end = 0.4
        elif self.snow_level == 4:
            c = 0.55, 0.3, 2.5, 0.55, 11, 20, 0.3
            snow_thres_start = 1.15
            snow_thres_end = 0.55
            mb_sigma_start = 0.05
            mb_sigma_end = 20
            br_coef_start = 0.95
            br_coef_end = 0.3
        self.snow_thres_list = torch.linspace(snow_thres_start, snow_thres_end, self.num_timesteps).tolist()
        self.mb_sigma_list = torch.linspace(mb_sigma_start, mb_sigma_end, self.num_timesteps).tolist()
        self.br_coef_list = torch.linspace(br_coef_start, br_coef_end, self.num_timesteps).tolist()
        self.snow = []
        self.snow_rot = []
        if self.single_snow:
            sb_list = []
            for _ in range(self.batch_size):
                cs = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
                cs = cs[..., np.newaxis]
                cs = clipped_zoom(cs, c[2])
                sb_list.append(cs)
            snow_layer_base = np.concatenate(sb_list, axis=2)
        else:
            snow_layer_base = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
            snow_layer_base = snow_layer_base[..., np.newaxis]
            snow_layer_base = clipped_zoom(snow_layer_base, c[2])
        vertical_snow = False
        if np.random.uniform() > 0.5:
            vertical_snow = True
        for i in range(self.num_timesteps):
            snow_layer = torch.Tensor(snow_layer_base).clone()
            snow_layer[snow_layer < self.snow_thres_list[i]] = 0
            snow_layer = torch.clip(snow_layer, 0, 1)
            snow_layer = snow_layer.permute((2, 0, 1)).unsqueeze(1)
            kernel_param = tgm.image.get_gaussian_kernel(c[4], self.mb_sigma_list[i])
            motion_kernel = torch.zeros((c[4], c[4]))
            motion_kernel[int(c[4] / 2)] = kernel_param
            horizontal_kernel = motion_kernel[None, None, :]
            horizontal_kernel = horizontal_kernel.repeat(3, 1, 1, 1)
            vertical_kernel = torch.rot90(motion_kernel, k=1, dims=[0, 1])
            vertical_kernel = vertical_kernel[None, None, :]
            vertical_kernel = vertical_kernel.repeat(3, 1, 1, 1)
            vsnow = F.conv2d(snow_layer, vertical_kernel, padding='same', groups=1)
            hsnow = F.conv2d(snow_layer, horizontal_kernel, padding='same', groups=1)
            if self.single_snow:
                vidx = torch.randperm(snow_layer.shape[0])
                vidx = vidx[:int(snow_layer.shape[0] / 2)]
                snow_layer = hsnow
                snow_layer[vidx] = vsnow[vidx]
            elif vertical_snow:
                snow_layer = vsnow
            else:
                snow_layer = hsnow
            self.snow.append(snow_layer)
            self.snow_rot.append(torch.rot90(snow_layer, k=2, dims=[2, 3]))
        if not self.random_snow:
            np.random.set_state(rstate)

    @torch.no_grad()
    def total_forward(self, x_in):
        return self.forward(None, self.num_timesteps - 1, og=x_in)

    @torch.no_grad()
    def forward(self, x, i, og=None):
        og_r = (og + 1.0) / 2.0
        og_gray = rgb_to_grayscale(og_r) * 1.5 + 0.5
        og_gray = torch.maximum(og_r, og_gray)
        br_coef = self.br_coef_list[i]
        scaled_og = br_coef * og_r + (1 - br_coef) * og_gray
        if self.fix_brightness:
            snowy_img = torch.clip(og_r + self.snow[i] + self.snow_rot[i], 0.0, 1.0)
        else:
            snowy_img = torch.clip(scaled_og + self.snow[i] + self.snow_rot[i], 0.0, 1.0)
        return snowy_img * 2.0 - 1.0


class GaussianDiffusion(nn.Module):

    def __init__(self, denoise_fn, *, image_size, device_of_kernel, one_shot_denoise_fn=None, channels=3, timesteps=1000, loss_type='l1', kernel_std=0.1, kernel_size=3, forward_process_type='Decolorization', train_routine='Final', sampling_routine='default', start_kernel_std=0.01, target_kernel_std=1.0, decolor_routine='Constant', decolor_ema_factor=0.9, decolor_total_remove=True, snow_level=1, random_snow=False, to_lab=False, order_seed=-1.0, recon_noise_std=0.0, load_snow_base=False, load_path=None, batch_size=32, single_snow=False, fix_brightness=False, results_folder=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.snow_level = snow_level
        self.random_snow = random_snow
        self.batch_size = batch_size
        self.single_snow = single_snow
        self.to_lab = to_lab
        self.recon_noise_std = recon_noise_std
        if forward_process_type == 'Decolorization':
            self.forward_process = DeColorization(decolor_routine=decolor_routine, decolor_ema_factor=decolor_ema_factor, decolor_total_remove=decolor_total_remove, channels=self.channels, num_timesteps=self.num_timesteps, to_lab=self.to_lab)
        elif forward_process_type == 'Snow':
            if load_path is not None:
                snow_base_path = load_path.replace('model.pt', 'snow_base.npy')
                None
                load_snow_base = True
            else:
                snow_base_path = os.path.join(results_folder, 'snow_base.npy')
                load_snow_base = False
            self.forward_process = Snow(image_size=self.image_size, snow_level=self.snow_level, random_snow=self.random_snow, num_timesteps=self.num_timesteps, snow_base_path=snow_base_path, batch_size=self.batch_size, single_snow=self.single_snow, load_snow_base=load_snow_base, fix_brightness=fix_brightness)

    @torch.no_grad()
    def sample_one_step(self, img, t, init_pred=None):
        orig_mean = torch.mean(img, [2, 3], keepdim=True)
        x = self.prediction_step_t(img, t, init_pred)
        direct_recons = x.clone()
        if self.recon_noise_std > 0.0:
            self.recon_noise_std_array = torch.linspace(0.0, self.recon_noise_std, steps=self.num_timesteps)
        if self.train_routine in ['Final', 'Final_random_mean', 'Final_small_noise', 'Final_random_mean_and_actual']:
            if self.sampling_routine == 'default':
                x_times_sub_1 = x.clone()
                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t - 1)[0]
                for i in range(t.max() - 1):
                    x_times_sub_1[fp_index] = self.forward_process.forward(x_times_sub_1[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t - 1)[0]
                x = x_times_sub_1
            elif self.sampling_routine == 'x0_step_down':
                x_times = x.clone()
                if self.recon_noise_std > 0.0:
                    x_times = x + torch.normal(0.0, self.recon_noise_std, size=x.size())
                x_times_sub_1 = x_times.clone()
                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t)[0]
                for i in range(t.max()):
                    x_times_sub_1 = x_times.clone()
                    x_times[fp_index] = self.forward_process.forward(x_times[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t)[0]
                x = img - x_times + x_times_sub_1
        elif self.train_routine == 'Step':
            img = x
        elif self.train_routine == 'Step_Gradient':
            x = img + x
        return x, direct_recons

    @torch.no_grad()
    def sample_multi_step(self, img, t_start, t_end):
        fp_index = torch.where(t_start > t_end)[0]
        img_new = img.clone()
        while len(fp_index) > 0:
            _, img_new_partial = self.sample_one_step(img_new[fp_index], t_start[fp_index])
            img_new[fp_index] = img_new_partial
            t_start = t_start - 1
            fp_index = torch.where(t_start > t_end)[0]
        return img_new

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None):
        self.forward_process.reset_parameters(batch_size=batch_size)
        if t == None:
            t = self.num_timesteps
        degrade_dict = {}
        og_img = img.clone()
        for i in range(t):
            None
            with torch.no_grad():
                img = self.forward_process.forward(img, i, og=og_img)
        init_pred = None
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long)
            x, cur_direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            if direct_recons is None:
                direct_recons = cur_direct_recons
            img = x
            t = t - 1
        if self.to_lab:
            xt = lab2rgb(xt)
            direct_recons = lab2rgb(direct_recons)
            img = lab2rgb(img)
        return_dict = {'xt': xt, 'direct_recons': direct_recons, 'recon': img}
        return return_dict

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, res_dict=None):
        self.forward_process.reset_parameters(batch_size=batch_size)
        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t
        img_forward_list = []
        img_forward = img
        with torch.no_grad():
            img = self.forward_process.total_forward(img)
        X_0s = []
        X_ts = []
        init_pred = None
        while times:
            step = torch.full((img.shape[0],), times - 1, dtype=torch.long)
            img, direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            X_0s.append(direct_recons.cpu())
            X_ts.append(img.cpu())
            times = times - 1
        init_pred_clone = None
        if init_pred is not None:
            init_pred_clone = init_pred.clone().cpu()
        if self.to_lab:
            for i in range(len(X_0s)):
                X_0s[i] = lab2rgb(X_0s[i])
                X_ts[i] = lab2rgb(X_ts[i])
            if init_pred is not None:
                init_pred_clone = lab2rgb(init_pred_clone)
        return X_0s, X_ts, init_pred_clone, img_forward_list

    def q_sample(self, x_start, t, return_total_blur=False):
        final_sample = x_start.clone()
        noisy_index = torch.where(t == -1)[0]
        max_iters = torch.max(t)
        all_blurs = []
        x = x_start[torch.where(t != -1)]
        blurring_batch_size = x.shape[0]
        if blurring_batch_size == 0:
            return final_sample
        for i in range(max_iters + 1):
            with torch.no_grad():
                x = self.forward_process.forward(x, i, og=final_sample[torch.where(t != -1)])
                all_blurs.append(x)
                if i == max_iters:
                    total_blur = x.clone()
        all_blurs = torch.stack(all_blurs)
        choose_blur = []
        for step in range(blurring_batch_size):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])
        choose_blur = torch.stack(choose_blur)
        final_sample[torch.where(t != -1)] = choose_blur
        if return_total_blur:
            final_sample_total_blur = final_sample.clone()
            final_sample_total_blur[torch.where(t != -1)] = total_blur
            return final_sample, final_sample_total_blur
        return final_sample

    def loss_func(self, pred, true):
        if self.loss_type == 'l1':
            return (pred - true).abs().mean()
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, true)
        elif self.loss_type == 'sqrt':
            return (pred - true).abs().mean().sqrt()
        else:
            raise NotImplementedError()

    def prediction_step_t(self, img, t, init_pred=None):
        return self.denoise_fn(img, t)

    def p_losses(self, x_start, t, t_pred=None):
        b, c, h, w = x_start.shape
        self.forward_process.reset_parameters()
        if self.train_routine == 'Final':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_recon = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_start, x_recon)
        elif self.train_routine == 'Step_Gradient':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t - 1)
            x_blur_diff = x_blur_sub - x_blur
            x_blur_diff_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_diff, x_blur_diff_pred)
        elif self.train_routine == 'Step':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t - 1)
            x_blur_sub_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_sub, x_blur_sub_pred)
        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        if type(img_size) is tuple:
            img_w, img_h = img_size
        else:
            img_h, img_w = img_size, img_size
        assert h == img_h and w == img_w, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t_pred = []
        for i in range(b):
            t_pred.append(torch.randint(0, t[i] + 1, ()).item())
        t_pred = torch.Tensor(t_pred).long() - 1
        t_pred[t_pred < 0] = 0
        return self.p_losses(x, t, t_pred, *args, **kwargs)

    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps
        Forward = []
        Forward.append(img)
        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, t=step)
                Forward.append(n_img)
        Backward = []
        img = n_img
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            Backward.append(img)
            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, t=step)
            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, t=step2)
            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1
        return Forward, Backward, img


class UnetConvNextBlock(nn.Module):

    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3, with_time_emb=True, output_mean_scale=False, residual=False):
        super().__init__()
        self.channels = channels
        self.residual = residual
        None
        self.output_mean_scale = output_mean_scale
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        else:
            time_dim = None
            self.time_mlp = None
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0), ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim), Residual(PreNorm(dim_out, LinearAttention(dim_out))), Downsample(dim_out) if not is_last else nn.Identity()]))
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(nn.ModuleList([ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim), ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim), Residual(PreNorm(dim_in, LinearAttention(dim_in))), Upsample(dim_in) if not is_last else nn.Identity()]))
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(ConvNextBlock(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time=None):
        orig_x = x
        t = None
        if time is not None and exists(self.time_mlp):
            t = self.time_mlp(time)
        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)
        h = []
        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x
        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1, 2, 3], keepdim=True)
            out = out - original_mean + out_mean
        return out


class UnetResNetBlock(nn.Module):

    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, with_time_emb=True, in_channels, resolution):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.with_time_emb = with_time_emb
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([torch.nn.Linear(self.ch, self.temb_ch), torch.nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None):
        assert x.shape[2] == x.shape[3] == self.resolution
        if t is None:
            assert self.with_time_emb == False
            t = torch.full((x.shape[0],), 0, dtype=torch.long)
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class RgbToLab(nn.Module):
    """Convert an image from RGB to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Returns:
        Lab version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> lab = RgbToLab()
        >>> output = lab(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """

    def forward(self, image: torch.Tensor) ->torch.Tensor:
        return rgb_to_lab(image)


class LabToRgb(nn.Module):
    """Convert an image from Lab to RGB.

    Returns:
        RGB version of the image. Range may not be in :math:`[0, 1]`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LabToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """

    def forward(self, image: torch.Tensor, clip: bool=True) ->torch.Tensor:
        return lab_to_rgb(image, clip)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvNextBlock,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
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
    (LayerNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'with_conv': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_arpitbansal297_Cold_Diffusion_Models(_paritybench_base):
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

