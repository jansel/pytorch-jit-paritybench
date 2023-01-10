import sys
_module = sys.modules[__name__]
del sys
data = _module
data_sampler = _module
data_util = _module
ffhq_dataset = _module
paired_image_SR_LR_FullImage_Memory_dataset = _module
paired_image_SR_LR_dataset = _module
paired_image_dataset = _module
prefetch_dataloader = _module
reds_dataset = _module
single_image_dataset = _module
transforms = _module
video_test_dataset = _module
vimeo90k_dataset = _module
demo = _module
demo_ssr = _module
metrics = _module
fid = _module
metric_util = _module
niqe = _module
psnr_ssim = _module
models = _module
HINet_arch = _module
MIRNet_arch = _module
MIRNet_v2_arch = _module
MPRNet_arch = _module
NAFNet_arch = _module
NAFSSR_arch = _module
Restormer_arch = _module
SCUNet_arch = _module
VRT_arch = _module
archs = _module
arch_util = _module
local_arch = _module
base_model = _module
image_restoration_model = _module
losses = _module
loss_util = _module
losses = _module
lr_scheduler = _module
test = _module
train = _module
train_rain = _module
utils = _module
antialias = _module
create_lmdb = _module
dist_util = _module
download_util = _module
face_util = _module
file_client = _module
flow_util = _module
img_util = _module
lmdb_util = _module
logger = _module
matlab_functions = _module
misc = _module
options = _module
version = _module
gopro = _module
rain13k = _module
reds = _module
sidd = _module
make_pickle = _module
setup = _module

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


import numpy as np


import random


import torch


import torch.utils.data


from functools import partial


import math


from torch.utils.data.sampler import Sampler


from torch.nn import functional as F


from torch.utils import data as data


from torchvision.transforms.functional import normalize


from torchvision.transforms.functional import resize


import queue as Queue


from torch.utils.data import DataLoader


import torch.nn as nn


from scipy import linalg


import torch.nn.functional as F


import numbers


import warnings


import torchvision


import torch.utils.checkpoint as checkpoint


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


from functools import reduce


from functools import lru_cache


from torch import nn as nn


from torch.nn import init as init


from torch.nn.modules.batchnorm import _BatchNorm


import time


import logging


from collections import OrderedDict


from copy import deepcopy


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import functools


from collections import Counter


from torch.optim.lr_scheduler import _LRScheduler


import torch.nn.parallel


import torch.distributed as dist


import torch.multiprocessing as mp


from torchvision.utils import make_grid


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias, stride=stride)


class SAM(nn.Module):

    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


class HINet(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(HINet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if i + 1 < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, 2 ** i * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConvBlock(prev_channels, 2 ** i * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            prev_channels = 2 ** i * wf
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, 2 ** i * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, 2 ** i * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d(2 ** i * wf, 2 ** i * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d(2 ** i * wf, 2 ** i * wf, 3, 1, 1))
            prev_channels = 2 ** i * wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x):
        image = x
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if i + 1 < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if i + 1 < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))
        out_2 = self.last(x2)
        out_2 = out_2 + image
        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


class AvgPool2d(nn.Module):

    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) ->str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(self.kernel_size, self.base_size, self.kernel_size, self.fast_imp)

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = self.base_size, self.base_size
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)
        if self.fast_imp:
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = (w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base:

    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class HINetLocal(Local_Base, HINet):

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        HINet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class SKFF(nn.Module):

    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()
        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):

    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class ca_layer(nn.Module):

    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DAU(nn.Module):

    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        self.SA = spatial_attn_layer()
        self.CA = ca_layer(n_feat, reduction, bias=bias)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class ResidualDownSample(nn.Module):

    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()
        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias), nn.PReLU(), nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias), nn.PReLU(), downsamp(channels=in_channels, filt_size=3, stride=2), nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))
        self.bot = nn.Sequential(downsamp(channels=in_channels, filt_size=3, stride=2), nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class DownSample(nn.Module):

    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class ResidualUpSample(nn.Module):

    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()
        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias), nn.PReLU(), nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1, bias=bias), nn.PReLU(), nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))
        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias), nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):

    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class MSRB(nn.Module):

    def __init__(self, n_feat, height, width, stride, bias):
        super(MSRB, self).__init__()
        self.n_feat, self.height, self.width = n_feat, height, width
        self.blocks = nn.ModuleList([nn.ModuleList([DAU(int(n_feat * stride ** i))] * width) for i in range(height)])
        INDEX = np.arange(0, width, 2)
        FEATS = [int(stride ** i * n_feat) for i in range(height)]
        SCALE = [(2 ** i) for i in range(1, height)]
        self.last_up = nn.ModuleDict()
        for i in range(1, height):
            self.last_up.update({f'{i}': UpSample(int(n_feat * stride ** i), 2 ** i, stride)})
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        i = 0
        SCALE.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.down.update({f'{feat}_{scale}': DownSample(feat, scale, stride)})
            i += 1
        i = 0
        FEATS.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.up.update({f'{feat}_{scale}': UpSample(feat, scale, stride)})
            i += 1
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.selective_kernel = nn.ModuleList([SKFF(n_feat * stride ** i, height) for i in range(height)])

    def forward(self, x):
        inp = x.clone()
        blocks_out = []
        for j in range(self.height):
            if j == 0:
                inp = self.blocks[j][0](inp)
            else:
                inp = self.blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            blocks_out.append(inp)
        for i in range(1, self.width):
            if True:
                tmp = []
                for j in range(self.height):
                    TENSOR = []
                    nfeats = 2 ** j * self.n_feat
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(blocks_out[k], j, k))
                    selective_kernel_fusion = self.selective_kernel[j](TENSOR)
                    tmp.append(selective_kernel_fusion)
            else:
                tmp = blocks_out
            for j in range(self.height):
                blocks_out[j] = self.blocks[j][i](tmp[j])
        out = []
        for k in range(self.height):
            out.append(self.select_last_up(blocks_out[k], k))
        out = self.selective_kernel[0](out)
        out = self.conv_out(out)
        out = out + x
        return out

    def select_up_down(self, tensor, j, k):
        if j == k:
            return tensor
        else:
            diff = 2 ** np.abs(j - k)
            if j < k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)

    def select_last_up(self, tensor, k):
        if k == 0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)


class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), nn.LeakyReLU(0.2), nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias))

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        context = self.modeling(x)
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        return x


class RCB(nn.Module):

    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        act = nn.LeakyReLU(0.2)
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups), act, nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups))
        self.act = act
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


class MRB(nn.Module):

    def __init__(self, n_feat, height, width, chan_factor, bias, groups):
        super(MRB, self).__init__()
        self.n_feat, self.height, self.width = n_feat, height, width
        self.dau_top = RCB(int(n_feat * chan_factor ** 0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat * chan_factor ** 1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat * chan_factor ** 2), bias=bias, groups=groups)
        self.down2 = DownSample(int(chan_factor ** 0 * n_feat), 2, chan_factor)
        self.down4 = nn.Sequential(DownSample(int(chan_factor ** 0 * n_feat), 2, chan_factor), DownSample(int(chan_factor ** 1 * n_feat), 2, chan_factor))
        self.up21_1 = UpSample(int(chan_factor ** 1 * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int(chan_factor ** 1 * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int(chan_factor ** 2 * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int(chan_factor ** 2 * n_feat), 2, chan_factor)
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)
        self.skff_top = SKFF(int(n_feat * chan_factor ** 0), 2)
        self.skff_mid = SKFF(int(n_feat * chan_factor ** 1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)
        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)
        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])
        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)
        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])
        out = self.conv_out(x_top)
        out = out + x
        return out


class RRG(nn.Module):

    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()
        modules_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class MIRNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_feat=64, kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False):
        super(MIRNet, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        modules_body = [RRG(n_feat, n_MSRB, height, width, stride, bias) for _ in range(n_RRG)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.body(h)
        h = self.conv_out(h)
        h += x
        return h


class Down(nn.Module):

    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()
        self.bot = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False), nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        return self.bot(x)


class Up(nn.Module):

    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()
        self.bot = nn.Sequential(nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias))

    def forward(self, x):
        return self.bot(x)


class MIRNet_v2(nn.Module):

    def __init__(self, inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, scale=1, bias=False, task=None):
        super(MIRNet_v2, self).__init__()
        kernel_size = 3
        self.task = task
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        modules_body = []
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, inp_img):
        shallow_feats = self.conv_in(inp_img)
        deep_feats = self.body(shallow_feats)
        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)
        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp_img
        return out_img


class CALayer(nn.Module):

    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class Encoder(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)
            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if encoder_outs is not None and decoder_outs is not None:
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        if encoder_outs is not None and decoder_outs is not None:
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        if encoder_outs is not None and decoder_outs is not None:
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        return [enc1, enc2, enc3]


class SkipUpSample(nn.Module):

    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class Decoder(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]


class ORB(nn.Module):

    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ORSNet(nn.Module):

    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()
        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)
        self.up_enc2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat + scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])
        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))
        return x


class MPRNet(nn.Module):

    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(MPRNet, self).__init__()
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        H = x3_img.size(2)
        W = x3_img.size(3)
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]
        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        stage3_img = self.tail(x3_cat)
        return stage3_img + x3_img


class MPRNetLocal(Local_Base, MPRNet):

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MPRNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-06):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFBlock(nn.Module):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class SCAM(nn.Module):
    """
    Stereo Cross Attention Module (SCAM)
    """

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r


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


class NAFBlockSR(nn.Module):
    """
    NAFBlock for Super-Resolution
    """

    def __init__(self, c, fusion=False, drop_out_rate=0.0):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class MySequential(nn.Sequential):

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class NAFNetSR(nn.Module):
    """
    NAFNet for Super-Resolution
    """

    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0.0, drop_out_rate=0.0, fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.body = MySequential(*[DropPath(drop_path_rate, NAFBlockSR(width, fusion=fusion_from <= i and i <= fusion_to, drop_out_rate=drop_out_rate)) for i in range(num_blks)])
        self.up = nn.Sequential(nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1, groups=1, bias=True), nn.PixelShuffle(up_scale))
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = inp,
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out


class NAFSSR(Local_Base, NAFNetSR):

    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class BiasFree_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-05) * self.weight


class WithBias_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-05) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        None
    return PadLayer


class Downsample(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)), int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


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


class Restormer(nn.Module):

    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False):
        super(Restormer, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1


class RestormerLocal(Local_Base, Restormer):

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Restormer.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


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


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1, 2).transpose(0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float('-inf'))
        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)
        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block(nn.Module):

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'
        None
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 4 * input_dim), nn.GELU(), nn.Linear(4 * input_dim, output_dim))

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution
        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False), nn.ReLU(True), nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False))

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SCUNet(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution) for i in range(config[0])] + [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]
        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2) for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]
        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4) for i in range(config[2])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]
        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 8) for i in range(config[3])]
        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False)] + [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4) for i in range(config[4])]
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False)] + [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2) for i in range(config[5])]
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False)] + [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution) for i in range(config[6])]
        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]
        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SCUNetLocal(Local_Base, SCUNet):

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        SCUNet.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = int(H * 1.5), int(W * 1.5)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


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


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.
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
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.
    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)
        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(nn.Conv2d((1 + self.pa_frames // 2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1))
        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)
        mask = torch.sigmoid(mask)
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False), nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False), nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False), nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False), nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
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
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                None
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []
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
            if level in self.return_levels:
                scale = 2 ** (5 - level)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)
        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()
        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)
        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)
        return flow_list[0] if len(flow_list) == 1 else flow_list


class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".
    Args:
        x: (B, D, H, W, C)
    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head mutual attention and self attention.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, mut_attn=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mut_attn = mut_attn
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        self.register_buffer('relative_position_index', self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if self.mut_attn:
            self.register_buffer('position_bias', self.get_sine_position_encoding(window_size[1:], dim // 2, normalize=True))
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)
        if self.mut_attn:
            qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            (q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), torch.chunk(qkv[1], 2, dim=2), torch.chunk(qkv[2], 2, dim=2)
            x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
            x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)
        x = self.proj(x_out)
        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape
        attn = q * self.scale @ k.transpose(-2, -1)
        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x

    def get_position_index(self, window_size):
        """ Get pair-wise relative position index for each token inside the window. """
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """ Get sine position encoding """
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class TMSA(nn.Module):
    """ Temporal Mutual Self Attention (TMSA).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(6, 8, 8), shift_size=(0, 0, 0), mut_attn=True, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn
        assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[2], 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]
        x = self.drop_path(x)
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class TMSAG(nn.Module):
    """ Temporal Mutual Self Attention Group (TMSAG).
    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=[6, 8, 8], shift_size=None, mut_attn=True, mlp_ratio=2.0, qkv_bias=False, qk_scale=None, drop_path=0.0, norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size
        self.blocks = nn.ModuleList([TMSA(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size, mut_attn=mut_attn, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, use_checkpoint_attn=use_checkpoint_attn, use_checkpoint_ffn=use_checkpoint_ffn) for i in range(depth)])

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class RTMSA(nn.Module):
    """ Residual Temporal Mutual Self Attention (RTMSA). Only used in stage 8.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path=0.0, norm_layer=nn.LayerNorm, use_checkpoint_attn=False, use_checkpoint_ffn=None):
        super(RTMSA, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = TMSAG(dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mut_attn=False, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=drop_path, norm_layer=norm_layer, use_checkpoint_attn=use_checkpoint_attn, use_checkpoint_ffn=use_checkpoint_ffn)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.
    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self, in_dim, dim, input_resolution, depth, num_heads, window_size, mul_attn_ratio=0.75, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path=0.0, norm_layer=nn.LayerNorm, pa_frames=2, deformable_groups=16, reshape=None, max_residue_magnitude=10, use_checkpoint_attn=False, use_checkpoint_ffn=False):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'), nn.LayerNorm(dim), Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2), nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim), Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2), nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim), Rearrange('n d h w c -> n c d h w'))
        self.residual_group1 = TMSAG(dim=dim, input_resolution=input_resolution, depth=int(depth * mul_attn_ratio), num_heads=num_heads, window_size=(2, window_size[1], window_size[2]), mut_attn=True, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=drop_path, norm_layer=norm_layer, use_checkpoint_attn=use_checkpoint_attn, use_checkpoint_ffn=use_checkpoint_ffn)
        self.linear1 = nn.Linear(dim, dim)
        self.residual_group2 = TMSAG(dim=dim, input_resolution=input_resolution, depth=depth - int(depth * mul_attn_ratio), num_heads=num_heads, window_size=window_size, mut_attn=False, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=drop_path, norm_layer=norm_layer, use_checkpoint_attn=True, use_checkpoint_ffn=use_checkpoint_ffn)
        self.linear2 = nn.Linear(dim, dim)
        self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups, max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
        self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x
        x = x.transpose(1, 2)
        x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
        x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))
        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 4 frames."""
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))
        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 6 frames."""
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped], x[:, i - 3, ...], [flow1, flow2, flow3]))
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped], x[:, i + 1, ...], [flow1, flow2, flow3]))
        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


class VRT(nn.Module):
    """ Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000
    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self, upscale=4, in_chans=3, img_size=[6, 64, 64], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4], indep_reconsts=[11, 12], embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180], num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], mul_attn_ratio=0.75, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.2, norm_layer=nn.LayerNorm, spynet_path=None, pa_frames=2, deformable_groups=16, recal_all_flows=False, nonblind_denoising=False, use_checkpoint_attn=False, use_checkpoint_ffn=False, no_checkpoint_attn_blocks=[], no_checkpoint_ffn_blocks=[]):
        super().__init__()
        self.in_chans = in_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising
        self.conv_first = nn.Conv3d(in_chans * (1 + 2 * 4) + 1 if self.nonblind_denoising else in_chans * (1 + 2 * 4), embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        scales = [1, 2, 4, 8, 4, 2, 1]
        use_checkpoint_attns = [(False if i in no_checkpoint_attn_blocks else use_checkpoint_attn) for i in range(len(depths))]
        use_checkpoint_ffns = [(False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn) for i in range(len(depths))]
        for i in range(7):
            setattr(self, f'stage{i + 1}', Stage(in_dim=embed_dims[i - 1], dim=embed_dims[i], input_resolution=(img_size[0], img_size[1] // scales[i], img_size[2] // scales[i]), depth=depths[i], num_heads=num_heads[i], mul_attn_ratio=mul_attn_ratio, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, pa_frames=pa_frames, deformable_groups=deformable_groups, reshape=reshapes[i], max_residue_magnitude=10 / scales[i], use_checkpoint_attn=use_checkpoint_attns[i], use_checkpoint_ffn=use_checkpoint_ffns[i]))
        self.stage8 = nn.ModuleList([nn.Sequential(Rearrange('n c d h w ->  n d h w c'), nn.LayerNorm(embed_dims[6]), nn.Linear(embed_dims[6], embed_dims[7]), Rearrange('n d h w c -> n c d h w'))])
        for i in range(7, len(depths)):
            self.stage8.append(RTMSA(dim=embed_dims[i], input_resolution=img_size, depth=depths[i], num_heads=num_heads[i], window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, use_checkpoint_attn=use_checkpoint_attns[i], use_checkpoint_ffn=use_checkpoint_ffns[i]))
        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])
        num_feat = 64
        if self.upscale == 1:
            self.conv_last = nn.Conv3d(embed_dims[0], in_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, in_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        if self.nonblind_denoising:
            x, noise_level_map = x[:, :, :self.in_chans, :, :], x[:, :, self.in_chans:, :, :]
        x_lq = x.clone()
        flows_backward, flows_forward = self.get_flows(x)
        x_backward, x_forward = self.get_aligned_image_2frames(x, flows_backward[0], flows_forward[0])
        x = torch.cat([x, x_backward, x_forward], 2)
        if self.nonblind_denoising:
            x = torch.cat([x, noise_level_map], 2)
        if self.upscale == 1:
            x = self.conv_first(x.transpose(1, 2))
            x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
            x = self.conv_last(x).transpose(1, 2)
            return x + x_lq
        else:
            x = self.conv_first(x.transpose(1, 2))
            x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
            x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)
            _, _, C, H, W = x.shape
            return x + torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)

    def get_flows(self, x):
        """ Get flows for 2 frames, 4 frames or 6 frames."""
        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames
        return flows_backward, flows_forward

    def get_flow_2frames(self, x):
        """Get flow between frames t and t+1 from x."""
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [flow.view(b, n - 1, 2, h // 2 ** i, w // 2 ** i) for flow, i in zip(flows_backward, range(4))]
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n - 1, 2, h // 2 ** i, w // 2 ** i) for flow, i in zip(flows_forward, range(4))]
        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        """Get flow between t and t+2 from (t,t+1) and (t+1,t+2)."""
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]
                flow_n2 = flows[:, i, :, :, :]
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward2.append(torch.stack(flow_list, 1))
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]
                flow_n2 = flows[:, i - 1, :, :, :]
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward2.append(torch.stack(flow_list, 1))
        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        """Get flow between t and t+3 from (t,t+2) and (t+2,t+3)."""
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]
                flow_n2 = flows[:, i + 1, :, :, :]
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_backward3.append(torch.stack(flow_list, 1))
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]
                flow_n2 = flows[:, i - 2, :, :, :]
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))
            flows_forward3.append(torch.stack(flow_list, 1))
        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4'))
        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def forward_features(self, x, flows_backward, flows_forward):
        """Main network for feature extraction."""
        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1
        for layer in self.stage8:
            x = layer(x)
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        return x


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

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

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
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
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
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-08).mean()


def get_pad_layer_1d(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad1d
    else:
        None
    return PadLayer


class Downsample1D(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 8, 64, 64])], {}),
     True),
    (BiasFree_LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextBlock,
     lambda: ([], {'n_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Down,
     lambda: ([], {'in_channels': 4, 'chan_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownSample,
     lambda: ([], {'in_channels': 4, 's_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4, 'ffn_expansion_factor': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp_GEGLU,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MySequential,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (NAFBlock,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NAFBlockSR,
     lambda: ([], {'c': 4}),
     lambda: ([], {}),
     False),
    (NAFNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NAFNetLocal,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (OverlapPatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PSNRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RCB,
     lambda: ([], {'n_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlockNoBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualUpSample,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SCAM,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SkipUpSample,
     lambda: ([], {'in_channels': 4, 's_factor': 4}),
     lambda: ([torch.rand([4, 8, 8, 8]), torch.rand([4, 4, 16, 16])], {}),
     True),
    (Subspace,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNetConvBlock,
     lambda: ([], {'in_size': 4, 'out_size': 4, 'downsample': 4, 'relu_slope': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Up,
     lambda: ([], {'in_channels': 4, 'chan_factor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpSample,
     lambda: ([], {'in_channels': 4, 's_factor': 4}),
     lambda: ([torch.rand([4, 8, 4, 4])], {}),
     True),
    (WindowAttention,
     lambda: ([], {'dim': 4, 'window_size': [4, 4, 4], 'num_heads': 4}),
     lambda: ([torch.rand([4, 32, 4])], {}),
     False),
    (WithBias_LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (skip_blocks,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (spatial_attn_layer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_murufeng_FUIR(_paritybench_base):
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

