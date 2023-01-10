import sys
_module = sys.modules[__name__]
del sys
src = _module
dataset = _module
comp_cars = _module
cub_200 = _module
folder = _module
lsun = _module
p3d_car = _module
shapenet = _module
torch_transforms = _module
kp_eval = _module
model = _module
encoder = _module
field = _module
generator = _module
loss = _module
pytorch3d_monkey = _module
renderer = _module
tools = _module
unicorn = _module
optimizer = _module
reconstruct = _module
scheduler = _module
trainer = _module
utils = _module
chamfer = _module
icp = _module
image = _module
logger = _module
mesh = _module
metrics = _module
path = _module
plot = _module
pytorch = _module

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


from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler


from copy import deepcopy


from functools import lru_cache


import numpy as np


import torch


from torch.utils.data.dataset import Dataset as TorchDataset


from torchvision.transforms import ToTensor


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import RandomHorizontalFlip


from random import random


from torchvision.transforms import functional as Fvision


from torchvision.transforms import InterpolationMode


from scipy.io import loadmat


import string


import pandas as pd


from scipy import io as scio


from torchvision.transforms.functional import to_tensor


from torch.nn import functional as F


from torch import nn


from torchvision import models as tv_models


from math import log2


from math import exp


import torch.nn.functional as F


import torchvision


from collections import OrderedDict


import torch.nn as nn


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import AdamW


from torch.optim import ASGD


from torch.optim import Adamax


from torch.optim import Adadelta


from torch.optim import Adagrad


from torch.optim import RMSprop


import warnings


from collections import Counter


from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import _LRScheduler


import time


import torch.distributed as dist


import torch.multiprocessing as mp


from functools import wraps


from numpy.random import seed as np_seed


from numpy.random import get_state as np_get_state


from numpy.random import set_state as np_set_state


from random import seed as rand_seed


from random import getstate as rand_get_state


from random import setstate as rand_set_state


from torch import manual_seed as torch_seed


from torch import get_rng_state as torch_get_state


from torch import set_rng_state as torch_set_state


from functools import partial


from typing import Optional


from collections import defaultdict


from matplotlib import pyplot as plt


class DDPCust(torch.nn.parallel.DistributedDataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def to(self, device):
        self.module
        return self


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def get_output_size(in_channels, img_size, model):
    x = torch.zeros(1, in_channels, *img_size)
    return np.prod(model(x).shape)


def get_resnet_model(name):
    if name is None:
        name = 'resnet18'
    return {'resnet18': tv_models.resnet18, 'resnet34': tv_models.resnet34, 'resnet50': tv_models.resnet50, 'resnet101': tv_models.resnet101, 'resnet152': tv_models.resnet152, 'resnext50_32x4d': tv_models.resnext50_32x4d, 'resnext101_32x8d': tv_models.resnext101_32x8d, 'wide_resnet50_2': tv_models.wide_resnet50_2, 'wide_resnet101_2': tv_models.wide_resnet101_2}[name]


@torch.no_grad()
def kaiming_weights_init(m, nonlinearity='relu'):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, name='resnet18', **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        pretrained = kwargs.pop('pretrained', False)
        n_features = kwargs.pop('n_features', None)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
            self.encoder = nn.Sequential(*seq)
        out_ch = get_output_size(self.color_channels, img_size, self.encoder)
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features
        self.out_ch = out_ch
        self.fc = fc

    def forward(self, x):
        return self.fc(self.encoder(x).flatten(1))


N_LAYERS = 3


N_UNITS = 128


def create_mlp(in_ch, out_ch, n_units=N_UNITS, n_layers=N_LAYERS, kaiming_init=True, zero_last_init=False, bias_last=True, with_norm=False, dropout=False):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_units)]
        if with_norm:
            seq.append(nn.BatchNorm1d(n_units))
        seq.append(nn.ReLU(True))
        for _ in range(n_layers - 1):
            if dropout:
                seq.append(nn.Dropout(dropout))
            seq.append(nn.Linear(n_units, n_units))
            if with_norm:
                seq.append(nn.BatchNorm1d(n_units))
            seq.append(nn.ReLU(True))
        seq += [nn.Linear(n_units, out_ch, bias=bias_last)]
    else:
        seq = [nn.Linear(in_ch, out_ch, bias=bias_last)]
    mlp = nn.Sequential(*seq)
    if kaiming_init:
        mlp.apply(kaiming_weights_init)
    if zero_last_init:
        with torch.no_grad():
            if isinstance(zero_last_init, bool):
                mlp[-1].weight.zero_()
            else:
                mlp[-1].weight.normal_(mean=0, std=zero_last_init)
            if bias_last:
                mlp[-1].bias.zero_()
    return mlp


class Field(nn.Module):
    """Corresponds to a field modeled by a coordinate-based MLPs"""

    def __init__(self, n_units=N_UNITS, n_layers=N_LAYERS, latent_size=None, in_ch=3, out_ch=3, zero_last_init=True, with_norm=False, dropout=False, bias_last=True):
        super().__init__()
        NU, NL = n_units, n_layers
        if latent_size is not None:
            self.linear_x = nn.Linear(in_ch, NU)
            self.linear_z = nn.Linear(latent_size, NU)
            if with_norm:
                self.act = nn.Sequential(nn.BatchNorm1d(NU), nn.ReLU(True))
            else:
                self.act = nn.ReLU(True)
            self.mlp = create_mlp(NU, out_ch, NU, NL - 1, zero_last_init=zero_last_init, with_norm=with_norm, dropout=dropout, bias_last=bias_last)
            [kaiming_weights_init(m) for m in [self.linear_x, self.linear_z]]
        else:
            self.mlp = create_mlp(in_ch, out_ch, NU, NL, zero_last_init=zero_last_init, with_norm=with_norm, dropout=dropout, bias_last=bias_last)

    def forward(self, x, latent=None):
        if latent is not None:
            N, B = len(x), len(latent)
            x = x[None] if len(x.shape) == 2 else x
            x = self.act((self.linear_x(x) + self.linear_z(latent[:, None])).view(B * N, -1))
            return self.mlp(x).view(B, N, -1)
        else:
            return self.mlp(x)


class Verbose:
    mute = False


class TerminalColors:
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    BOLD = '\x1b[1m'
    UNDERLINE = '\x1b[4m'


def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def print_error(s):
    None


def print_info(s):
    None


def print_warning(s):
    None


def print_log(s, logger=None, level='info'):
    if Verbose.mute:
        return None
    if logger is None:
        logger = logging.getLogger('trainer')
    if level == 'info':
        print_info(s)
        logger.info(s)
    elif level == 'warning':
        print_warning(s)
        logger.warning(s)
    elif level == 'error':
        print_error(s)
        logger.error(s)
    else:
        raise NotImplementedError


class ProgressiveField(nn.Module):

    def __init__(self, inp_dim, name, powers, milestones=None, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.powers = [powers] if isinstance(powers, int) else powers
        self.n_powers = len(self.powers)
        self.latent_size = self.powers[-1]
        assert all([(self.latent_size % p == 0) for p in powers])
        self.repeat_latent = [(self.latent_size // p) for p in powers]
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)
        bias_last = kwargs.pop('bias_last', True)
        self.regressor = create_mlp(inp_dim, self.latent_size, NU, NL, zero_last_init=True)
        NU, NL = kwargs.pop('n_field_units', N_UNITS), kwargs.pop('n_field_layers', N_LAYERS)
        self.field = Field(NU, NL, latent_size=self.latent_size, zero_last_init=True, bias_last=bias_last)
        self.cur_milestone = 0
        self.set_milestones(milestones)
        assert len(kwargs) == 0

    def forward(self, x, features):
        B, C, device = features.size(0), self.latent_size, x.device
        latent_final = self.regressor(features)
        if self.act_idx < self.n_powers:
            p = self.current_code_size
            mask = torch.zeros(B, C, device=device)
            mask[:, :p] = torch.ones(B, p, device=device)
            latent_final = mask * latent_final
        self._latent = latent_final
        return self.field(x, latent_final)

    def step(self):
        self.cur_milestone += 1
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            m, p = self.cur_milestone, self.powers[self.act_idx]
            print_log('Milestone {}, progressive field: {} activated'.format(m, p))
            self.act_idx += 1

    def set_cur_milestone(self, k):
        self.cur_milestone = k
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            self.act_idx += 1
        powers, activations = self.powers, self.activations
        print_log('progressive field activated powers={}'.format([k for k, a in zip(powers, activations) if a]))

    def set_milestones(self, milestones):
        if milestones is not None:
            milestones = [milestones] if isinstance(milestones, int) else milestones
            assert len(milestones) == self.n_powers
            self.act_milestones = milestones
            n_act = (np.asarray(milestones) <= self.cur_milestone).sum()
            self.act_idx = n_act
            self.activations = [True] * n_act + [False] * (self.n_powers - n_act)
            powers, activations = self.powers, self.activations
            print_log('progressive field activated powers={}'.format([k for k, a in zip(powers, activations) if a]))
        else:
            self.act_milestones = [-1] * self.n_powers
            self.act_idx = self.n_powers
            self.activations = [True] * self.n_powers

    @property
    def is_frozen(self):
        return sum(self.activations) == 0

    @property
    def current_code_size(self):
        return self.powers[self.act_idx - 1] if self.act_idx > 0 else 0


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1, groups=1, dilation=1, zero_init=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)
    if zero_init:
        conv.weight.data.zero_()
    return conv


class Blur(nn.Module):

    def __init__(self):
        super().__init__()
        kernel = torch.Tensor([1, 2, 1])
        kernel = kernel[None, None, :] * kernel[None, :, None]
        kernel = kernel / kernel.norm(p=1)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        kernel = self.kernel.unsqueeze(1).expand(-1, C, -1, -1)
        kernel = kernel.reshape(-1, 1, *kernel.shape[2:])
        x = x.view(-1, kernel.size(0), x.size(-2), x.size(-1))
        return F.conv2d(x, kernel, groups=kernel.size(0), padding=0, stride=1).view(B, C, H, W)


def create_upsample_layer(name):
    if name == 'nn':
        return nn.Upsample(scale_factor=2)
    elif name == 'bilinear':
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    elif name == 'bilinear_blur':
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Blur())
    else:
        raise NotImplementedError


class GiraffeGenerator(nn.Module):
    """Neural renderer class from https://github.com/autonomousvision/giraffe"""

    def __init__(self, n_features=128, inp_dim=128, img_size=64, out_dim=3, min_features=16, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.inp_dim = inp_dim
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.n_blocks = int(log2(min(self.img_size)))
        self.feat_w, self.feat_h = [int(s / 2 ** self.n_blocks) for s in self.img_size]
        self.use_rgb_skip = kwargs.pop('use_rgb_skip', True)
        self.use_norm = kwargs.pop('use_norm', False)
        upsample_feat, upsample_rgb = kwargs.pop('upsample_feat', 'nn'), kwargs.pop('upsample_rgb', 'bilinear_blur')
        assert len(kwargs) == 0
        n_ch_fn = lambda i: max(n_features // 2 ** i, min_features)
        n_flat_features = n_features * np.prod([self.feat_w, self.feat_h])
        self.reshape_features = n_features != n_flat_features
        self.conv_in = conv1x1(inp_dim, n_flat_features) if n_flat_features != inp_dim else Identity()
        seq = [conv3x3(n_ch_fn(i + 1), n_ch_fn(i + 2)) for i in range(self.n_blocks - 1)]
        self.conv_layers = nn.ModuleList([conv3x3(n_features, n_features // 2)] + seq)
        self.upsample_feat = create_upsample_layer(upsample_feat)
        self.upsample_rgb = create_upsample_layer(upsample_rgb)
        if self.use_rgb_skip:
            self.conv_rgb = nn.ModuleList([conv3x3(n_ch_fn(i), out_dim) for i in range(self.n_blocks + 1)])
        else:
            self.conv_rgb = conv3x3(n_ch_fn(self.n_blocks), out_dim)
        if self.use_norm:
            self.norms = nn.ModuleList([nn.InstanceNorm2d(n_ch_fn(i + 1)) for i in range(self.n_blocks)])
        self.actvn = nn.ReLU(inplace=True)
        [kaiming_weights_init(m) for m in self.modules()]

    def forward(self, inp):
        inp = inp[..., None, None] if len(inp.shape) < 4 else inp
        net = self.conv_in(inp)
        if self.reshape_features:
            net = net.view(len(net), -1, self.feat_h, self.feat_w)
        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](net))
        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_feat(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)
            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < self.n_blocks - 1:
                    rgb = self.upsample_rgb(rgb)
        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)
        return torch.sigmoid(rgb)


def get_generator(name):
    return {'giraffe': GiraffeGenerator}[name]


class ProgressiveGenerator(nn.Module):

    def __init__(self, inp_dim, powers, milestones, **kwargs):
        super().__init__()
        self.powers = [powers] if isinstance(powers, int) else powers
        self.n_powers = len(self.powers)
        self.latent_size = self.powers[-1]
        assert all([(self.latent_size % p == 0) for p in powers])
        self.repeat_latent = [(self.latent_size // p) for p in powers]
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)
        n_features = kwargs.pop('n_features', self.latent_size)
        self.name = kwargs.pop('name', 'giraffe')
        self.regressor = create_mlp(inp_dim, self.latent_size, NU, NL, zero_last_init=True)
        self.generator = get_generator(self.name)(n_features=n_features, inp_dim=self.latent_size, **kwargs)
        self.cur_milestone = 0
        self.set_milestones(milestones)

    def forward(self, x):
        B, C, device = x.size(0), self.latent_size, x.device
        latent_final = self.regressor(x)
        if self.act_idx < self.n_powers:
            p = self.current_code_size
            mask = torch.zeros(B, C, device=device)
            mask[:, :p] = torch.ones(B, p, device=device)
            latent_final = mask * latent_final
        self._latent = latent_final
        return self.generator(latent_final)

    def step(self):
        self.cur_milestone += 1
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            m, p = self.cur_milestone, self.powers[self.act_idx]
            print_log('Milestone {}, progressive {} generator: power {} activated'.format(m, self.name, p))
            self.act_idx += 1

    def set_cur_milestone(self, k):
        self.cur_milestone = k
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            self.act_idx += 1
        powers, act = self.powers, self.activations
        print_log('progressive {} gen active powers={}'.format(self.name, [k for k, a in zip(powers, act) if a]))

    def set_milestones(self, milestones):
        if milestones is not None:
            milestones = [milestones] if isinstance(milestones, int) else milestones
            assert len(milestones) == self.n_powers
            self.act_milestones = milestones
            n_act = (np.asarray(milestones) <= self.cur_milestone).sum()
            self.act_idx = n_act
            self.activations = [True] * n_act + [False] * (self.n_powers - n_act)
            powers, act = self.powers, self.activations
            print_log('progressive {} gen active powers={}'.format(self.name, [k for k, a in zip(powers, act) if a]))
        else:
            self.act_milestones = [-1] * self.n_powers
            self.act_idx = self.n_powers
            self.activations = [True] * self.n_powers

    @property
    def is_frozen(self):
        return sum(self.activations) == 0

    @property
    def current_code_size(self):
        return self.powers[self.act_idx - 1] if self.act_idx > 0 else 0


class PerceptualLoss(nn.Module):

    def __init__(self, normalize_input=True, normalize_features=True, feature_levels=None, sum_channels=False, requires_grad=False):
        super().__init__()
        self.normalize_input = normalize_input
        self.normalize_features = normalize_features
        self.sum_channels = sum_channels
        self.feature_levels = feature_levels if feature_levels is not None else [3]
        assert isinstance(self.feature_levels, (list, tuple))
        self.max_level = max(self.feature_levels)
        self.register_buffer('mean_rgb', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std_rgb', torch.Tensor([0.229, 0.224, 0.225]))
        layers = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = layers[:4]
        self.slice2 = layers[4:9]
        self.slice3 = layers[9:16]
        self.slice4 = layers[16:23]
        self.slice5 = layers[23:30]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, im1, im2):
        inp = torch.cat([im1, im2], 0)
        if self.normalize_input:
            inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)
        feats = []
        for k in range(1, 6):
            if k > self.max_level:
                break
            inp = getattr(self, f'slice{k}')(inp)
            feats.append(torch.chunk(inp, 2, dim=0))
        losses = []
        for k, (f1, f2) in enumerate(feats, start=1):
            if k in self.feature_levels:
                if self.normalize_features:
                    f1, f2 = map(lambda t: t / (t.norm(dim=1, keepdim=True) + 1e-10), [f1, f2])
                loss = (f1 - f2) ** 2
                if self.sum_channels:
                    losses.append(loss.sum(1).flatten(2).mean(2))
                else:
                    losses.append(loss.flatten(1).mean(1))
        return sum(losses)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class SSIMLoss(torch.nn.Module):

    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def ssim(self, img1, img2):
        window_size, channel = self.window_size, self.channel
        window = self.window
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, inp, target):
        return self.ssim(inp, target).flatten(1).mean(1)


LAYERED_SHADER = True


def layered_rgb_blend(colors, fragments, blend_params, clip_inside=True, debug=False):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_
    mask = fragments.pix_to_face >= 0
    if blend_params.sigma == 0:
        alpha = (fragments.dists <= 0).float() * mask
    elif clip_inside:
        alpha = torch.exp(-fragments.dists.clamp(0) / blend_params.sigma) * mask
    else:
        alpha = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    occ_alpha = torch.cumprod(1.0 - alpha, dim=-1)
    occ_alpha = torch.cat([torch.ones(N, H, W, 1, device=device), occ_alpha], dim=-1)
    colors = torch.cat([colors, background[None, None, None, None].expand(N, H, W, 1, -1)], dim=-2)
    alpha = torch.cat([alpha, torch.ones(N, H, W, 1, device=device)], dim=-1)
    pixel_colors[..., :3] = (occ_alpha[..., None] * alpha[..., None] * colors).sum(-2)
    pixel_colors[..., 3] = 1 - occ_alpha[:, :, :, -1]
    if debug:
        return colors, alpha, occ_alpha, pixel_colors.permute(0, 3, 1, 2)
    else:
        return pixel_colors.permute(0, 3, 1, 2)


class LayeredShader(nn.Module):

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None, clip_inside=True, shading_type='phong', debug=False):
        super().__init__()
        self.lights = lights if lights is not None else DirectionalLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.clip_inside = clip_inside
        if shading_type == 'phong':
            shading_fn = phong_shading
        elif shading_type == 'flat':
            shading_fn = flat_shading
        elif shading_type == 'gouraud':
            shading_fn = gouraud_shading
        elif shading_type == 'raw':
            shading_fn = lambda x: x
        else:
            raise NotImplementedError
        self.shading_fn = shading_fn
        self.shading_type = shading_type
        self.debug = debug

    def to(self, device):
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras
        self.materials = self.materials
        self.lights = self.lights
        return self

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get('blend_params', self.blend_params)
        if self.shading_type == 'raw':
            colors = meshes.sample_textures(fragments)
            if not torch.all(self.lights.ambient_color == 1):
                colors *= self.lights.ambient_color
        else:
            sh_kwargs = {'meshes': meshes, 'fragments': fragments, 'cameras': kwargs.get('cameras', self.cameras), 'lights': kwargs.get('lights', self.lights), 'materials': kwargs.get('materials', self.materials)}
            if self.shading_type != 'gouraud':
                sh_kwargs['texels'] = meshes.sample_textures(fragments)
            colors = self.shading_fn(**sh_kwargs)
        return layered_rgb_blend(colors, fragments, blend_params, clip_inside=self.clip_inside, debug=self.debug)


SHADING_TYPE = 'raw'


VIZ_IMG_SIZE = 256


class Renderer(nn.Module):

    def __init__(self, img_size, **kwargs):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self._init_kwargs = deepcopy(kwargs)
        self.init_cameras(**kwargs.get('cameras', {}))
        self.init_lights(**kwargs.get('lights', {}))
        blend_kwargs = {'sigma': kwargs.get('sigma', 0.0001), 'background_color': kwargs.get('background_color', (1, 1, 1))}
        n_faces = kwargs.get('faces_per_pixel', 25)
        blend_params = BlendParams(**blend_kwargs)
        s_kwargs = {'cameras': self.cameras, 'lights': self.lights, 'blend_params': blend_params, 'debug': kwargs.get('debug', False)}
        if kwargs.get('layered_shader', LAYERED_SHADER):
            shader_cls = LayeredShader
            s_kwargs['clip_inside'] = kwargs.get('clip_inside', True)
            s_kwargs['shading_type'] = kwargs.get('shading_type', SHADING_TYPE)
        else:
            shader_cls = SoftPhongShaderPlus
        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=np.log(1.0 / 0.0001 - 1.0) * blend_params.sigma, faces_per_pixel=n_faces, perspective_correct=False)
        self.renderer = MeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings), shader_cls(**s_kwargs))
        viz_size = kwargs.get('viz_size', (VIZ_IMG_SIZE, VIZ_IMG_SIZE))
        s_kwargs['blend_params'] = BlendParams(background_color=blend_kwargs['background_color'], sigma=0)
        raster_settings = RasterizationSettings(image_size=(viz_size[0] * 2, viz_size[1] * 2), blur_radius=0.0, faces_per_pixel=1, perspective_correct=False)
        self.viz_renderer = VizMeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings), shader_cls(**s_kwargs))

    def init_cameras(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'fov')
        cam_cls = {'fov': FoVPerspectiveCameras, 'perspective': PerspectiveCameras}[name]
        self.cameras = cam_cls(**kwargs)

    def init_lights(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'ambient')
        light_cls = {'ambient': AmbientLights, 'directional': DirectionalLights, 'point': PointLights}[name]
        self.lights = light_cls(**kwargs)
        if name == 'directional':
            self.lights._direction = self.lights.direction
            self.lights._ambient_color = self.lights.ambient_color
            self.lights._diffuse_color = self.lights.diffuse_color
            self.lights._specular_color = self.lights.specular_color

    @property
    def init_kwargs(self):
        return deepcopy(self._init_kwargs)

    def forward(self, meshes, R, T, viz_purpose=False):
        return self.viz_renderer(meshes, R=R, T=T) if viz_purpose else self.renderer(meshes, R=R, T=T)

    def to(self, device):
        super()
        self.renderer = self.renderer
        self.viz_renderer = self.viz_renderer
        return self

    def update_lights(self, direction=None, ka=None, kd=None, ks=None):
        if direction is not None:
            self.lights.direction = direction
        if ka is not None:
            self.lights.ambient_color = ka
        if kd is not None:
            self.lights.diffuse_color = kd
        if ks is not None:
            self.lights.specular_color = ks

    def reset_default_lights(self):
        self.lights.direction = self.lights._direction
        self.lights.ambient_color = self.lights._ambient_color
        self.lights.diffuse_color = self.lights._diffuse_color
        self.lights.specular_color = self.lights._specular_color

    @torch.no_grad()
    def compute_vertex_visibility(self, meshes, R, T):
        fragments = self.viz_renderer.rasterizer(meshes, R=R, T=T)
        pix_to_face = fragments.pix_to_face
        packed_faces = meshes.faces_packed()
        packed_verts = meshes.verts_packed()
        visibility_map = torch.zeros(packed_verts.shape[0])
        visible_faces = pix_to_face.unique()[1:]
        visible_verts_idx = packed_faces[visible_faces]
        unique_visible_verts_idx = torch.unique(visible_verts_idx)
        visibility_map[unique_visible_verts_idx] = 1.0
        return visibility_map.view(len(meshes), -1).bool()


MEMSIZE = 1024


MIN_ANGLE = 20


CHAMFER_FACTOR = 10


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, N=1):
        if isinstance(val, torch.Tensor):
            assert val.numel() == 1
            val = val.item()
        self.val = val
        self.sum += val * N
        self.count += N
        self.avg = self.sum / self.count if self.count != 0 else 0


class Metrics:
    log_data = True

    def __init__(self, *names, log_file=None, append=False):
        self.names = list(names)
        self.meters = defaultdict(AverageMeter)
        if log_file is not None and self.log_data:
            self.log_file = Path(log_file)
            if not self.log_file.exists() or not append:
                with open(self.log_file, mode='w') as f:
                    f.write('iteration\tepoch\tbatch\t' + '\t'.join(self.names) + '\n')
        else:
            self.log_file = None

    def log_and_reset(self, *names, it=None, epoch=None, batch=None):
        self.log(it, epoch, batch)
        self.reset(*names)

    def log(self, it, epoch, batch):
        if self.log_file is not None:
            with open(self.log_file, mode='a') as file:
                file.write(f'{it}\t{epoch}\t{batch}\t' + '\t'.join(map('{:.6f}'.format, self.values)) + '\n')

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self[name].reset()

    def read_log(self):
        if self.log_file is not None:
            return pd.read_csv(self.log_file, sep='\t', index_col=0)
        else:
            return pd.DataFrame()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self[name].avg) for name in self.names])

    def __len__(self):
        return len(self.names)

    @property
    def values(self):
        return [self[name].avg for name in self.names]

    def update(self, *name_val, N=1):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v, N=N)
        else:
            assert len(name_val) == 2
            name, val = name_val
            if name not in self.names:
                raise KeyError(f'{name} not in current metrics')
            if isinstance(val, (tuple, list)):
                self[name].update(val[0], N=val[1])
            else:
                self[name].update(val, N=N)

    def get_named_values(self, filter_fn=None):
        names, values = self.names, self.values
        if filter_fn is not None:
            zip_fn = lambda k_v: filter_fn(k_v[0])
            names, values = map(list, zip(*filter(zip_fn, zip(names, values))))
        return list(zip(names, values))


def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, batch_reduction='mean', point_reduction='mean', return_L1=False, return_mean=False):
    """
    Copy from https://github.com/facebookresearch/pytorch3d repo (see pytorch3d/loss/chamfer.py)
    with following modifications to be comparable to OccNet and DVR results [Niemeyer et al., 2019]
    (https://github.com/autonomousvision/differentiable_volumetric_rendering, see im2mesh/eval.py file for details):
        - support for returning chamfer-L1 instead of chamfer-L2
        - support for mean (cham_x, cham_y) instead of sum

    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    return_normals = x_normals is not None and y_normals is not None
    N, P1, D = x.shape
    P2 = y.shape[1]
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    y_mask = torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError('y does not have the correct shape.')
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError('weights must be of shape (N,).')
        if not (weights >= 0).all():
            raise ValueError('weights cannot be negative.')
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ['mean', 'sum']:
                return (x.sum((1, 2)) * weights).sum() * 0.0, (x.sum((1, 2)) * weights).sum() * 0.0
            return x.sum((1, 2)) * weights * 0.0, x.sum((1, 2)) * weights * 0.0
    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)
    cham_x = x_nn.dists[..., 0]
    cham_y = y_nn.dists[..., 0]
    if return_L1:
        cham_x, cham_y = cham_x.sqrt(), cham_y.sqrt()
    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0
    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)
    if return_normals:
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]
        cham_norm_x = 1 - torch.abs(F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-06))
        cham_norm_y = 1 - torch.abs(F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-06))
        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)
    cham_x = cham_x.sum(1)
    cham_y = cham_y.sum(1)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)
        cham_norm_y = cham_norm_y.sum(1)
    if point_reduction == 'mean':
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths
    if batch_reduction is not None:
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == 'mean':
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div
    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None
    if return_mean:
        cham_dist, cham_normals = 0.5 * cham_dist, 0.5 * cham_normals if return_normals else None
    return cham_dist, cham_normals


def normalize(meshes, center=True, scale_mode='unit_cube', inplace=False, use_center_mass=False):
    if center:
        if use_center_mass:
            offsets = sample_points(meshes, 100000).mean(1)
        else:
            offsets = 0.5 * (meshes.verts_padded().max(1)[0] + meshes.verts_padded().min(1)[0])
        NVs = meshes.num_verts_per_mesh()
        offsets = torch.cat([offset[None].expand(nv, -1) for offset, nv in zip(offsets, NVs)], dim=0)
        meshes = meshes.offset_verts_(-offsets) if inplace else meshes.offset_verts(-offsets)
    if scale_mode == 'none' or scale_mode is None:
        scales = 1.0
    elif scale_mode == 'unit_cube':
        scales = meshes.verts_padded().abs().flatten(1).max(1)[0] * 2
    elif scale_mode == 'unit_sphere':
        scales = meshes.verts_padded().norm(dim=2).max(1)[0]
    else:
        raise NotImplementedError
    return meshes.scale_verts_(1 / scales) if inplace else meshes.scale_verts(1 / scales)


class MeshEvaluator:
    """
    Mesh evaluation class by computing similarity metrics between predicted mesh and GT.
    Code inspired from https://github.com/autonomousvision/differentiable_volumetric_rendering (see im2mesh/eval.py)
    """
    default_names = ['chamfer-L1', 'chamfer-L1-ICP', 'normal-cos', 'normal-cos-ICP']

    def __init__(self, names=None, log_file=None, run_icp=True, estimate_scale=True, anisotropic_scale=True, icp_type='gradient', fast_cpu=False, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)
        self.run_icp = run_icp
        self.estimate_scale = estimate_scale
        self.ani_scale = anisotropic_scale
        self.icp_type = icp_type
        assert icp_type in ['normal', 'gradient']
        self.fast_cpu = fast_cpu
        self.N = 50000 if fast_cpu else 100000
        print_log('MeshEvaluator init: run_icp={}, estimate_scale={}, anisotropic_scale={}, icp_type={}, n_iter={}'.format(run_icp, estimate_scale, anisotropic_scale, icp_type, self.n_iter))

    @property
    def n_iter(self):
        if self.icp_type == 'normal':
            return 10 if self.fast_cpu else 30
        else:
            return 30 if self.fast_cpu else 100

    def update(self, mesh_pred, labels):
        pc_gt, norm_gt = labels['points'], labels['normals']
        vox_gt = labels.get('voxels')
        res = self.evaluate(mesh_pred, pc_gt=pc_gt, norm_gt=norm_gt, vox_gt=vox_gt)
        self.metrics.update(res, N=len(mesh_pred))

    def evaluate(self, mesh_pred, pc_gt, norm_gt, vox_gt=None):
        assert abs(pc_gt.abs().max() - 0.5) < 0.01
        pc_pred, norm_pred = sample_points(mesh_pred, self.N, return_normals=True)
        if self.N < len(pc_gt):
            idxs = torch.randperm(len(pc_gt))[:self.N]
            pc_gt, norm_gt = pc_gt[:, idxs], norm_gt[:, idxs]
        use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
        results = []
        if self.run_icp:
            mesh_pred = normalize(mesh_pred)
            pc_pred2, norm_pred2 = sample_points(mesh_pred, self.N, return_normals=True)
            if self.icp_type == 'normal':
                pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
            else:
                pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)
            pc_preds, norm_preds, tags = [pc_pred, pc_pred_icp], [norm_pred, norm_pred2], ['', '-ICP']
        else:
            pc_preds, norm_preds, tags = [pc_pred], [norm_pred], ['']
        for pc, norm, tag in zip(pc_preds, norm_preds, tags):
            chamfer_L1, normal = chamfer_distance(pc_gt, pc, x_normals=norm_gt, y_normals=norm, return_L1=True, return_mean=True)
            chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
            results += [('chamfer-L1' + tag, chamfer_L1.item()), ('normal-cos' + tag, 1 - normal.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()


N_ELEV_AZIM = [1, 6]


N_POSES = 6


N_VPBINS = 5


PRIOR_TRANSLATION = [0.0, 0.0, 2.732]


class ProxyEvaluator:
    default_names = ['mask_iou']

    def __init__(self, names=None, log_file=None, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)

    def update(self, mask_pred, mask_gt):
        for k in range(len(mask_pred)):
            self.metrics.update(self.evaluate(mask_pred[k], mask_gt[k]))

    def evaluate(self, mask_pred, mask_gt):
        results = []
        miou = (mask_pred * mask_gt).sum() / (mask_pred + mask_gt).clamp(0, 1).sum()
        results += [('mask_iou', miou.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()


SCALE_ELLIPSE = [1, 0.7, 0.7]


def azim_to_rotation_matrix(azim, as_degree=True):
    """Angle with +X in XZ plane"""
    if isinstance(azim, (int, float)):
        azim = torch.Tensor([azim])
    azim_rad = azim * np.pi / 180 if as_degree else azim
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    cos, sin = torch.cos(azim_rad), torch.sin(azim_rad)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R.squeeze()


def convert_3d_to_uv_coordinates(X, eps=1e-05):
    """Resulting UV in [0, 1]"""
    radius = torch.norm(X, dim=-1).clamp(min=eps)
    theta = torch.acos((X[..., 1] / radius).clamp(min=-1 + eps, max=1 - eps))
    phi = torch.atan2(X[..., 0], X[..., 2])
    vv = theta / np.pi
    uu = (phi + np.pi) / (2 * np.pi)
    return torch.stack([uu, vv], dim=-1)


def convert_to_img(arr):
    if isinstance(arr, Image.Image):
        return arr
    if isinstance(arr, torch.Tensor):
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        elif len(arr.shape) == 2:
            arr = arr.unsqueeze(0)
        arr = arr.permute(1, 2, 0).detach().cpu().numpy()
    assert isinstance(arr, np.ndarray)
    if len(arr.shape) == 3:
        if arr.shape[0] <= 3:
            arr = arr.transpose(1, 2, 0)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.clip(0, 1) * 255
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def cpu_angle_between(R1, R2, as_degree=True):
    angle = ((torch.einsum('bii -> b', (R1.transpose(-2, -1) @ R2).view(-1, 3, 3)) - 1) / 2).acos()
    return 180 / np.pi * angle if as_degree else angle


def elev_to_rotation_matrix(elev, as_degree=True):
    """Angle with +Z in YZ plane"""
    if isinstance(elev, (int, float)):
        elev = torch.Tensor([elev])
    elev_rad = elev * np.pi / 180 if as_degree else elev
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    cos, sin = torch.cos(-elev_rad), torch.sin(-elev_rad)
    R[:, 1, 1:] = torch.stack([cos, sin], dim=-1)
    R[:, 2, 1:] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def get_icosphere(level=3, order_verts_by=None, colored=False):
    mesh = ico_sphere(level)
    if order_verts_by is not None:
        assert isinstance(order_verts_by, int)
        verts, faces = mesh.get_mesh_verts_faces(0)
        N = len(verts)
        indices = sorted(range(N), key=lambda i: verts[i][order_verts_by])
        mapping = torch.zeros(N, dtype=torch.long)
        mapping[indices] = torch.arange(N)
        verts.copy_(verts[indices]), faces.copy_(mapping[faces])
    if colored:
        verts = mesh.verts_packed()
        colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        mesh.textures = TexturesVertex(verts_features=colors[None])
    return mesh


def get_loss(name):
    return {'bce': nn.BCEWithLogitsLoss, 'mse': nn.MSELoss, 'l2': nn.MSELoss, 'l1': nn.L1Loss, 'huber': nn.SmoothL1Loss, 'cosine': nn.CosineSimilarity, 'perceptual': PerceptualLoss, 'ssim': SSIMLoss}[name]


def init_rotations(init_type='uniform', N=None, n_elev=None, n_azim=None, elev_range=None, azim_range=None):
    if init_type == 'uniform':
        assert n_elev is not None and n_azim is not None
        assert N == n_elev * n_azim if N is not None else True
        eb, ee = elev_range if elev_range is not None else (-90, 90)
        ab, ae = azim_range if azim_range is not None else (-180, 180)
        er, ar = ee - eb, ae - ab
        elev = torch.Tensor([(k * er / n_elev + eb - er / (2 * n_elev)) for k in range(1, n_elev + 1)])
        if ar == 360 and n_azim > 1:
            azim = torch.Tensor([(k * ar / n_azim + ab) for k in range(n_azim)])
        else:
            azim = torch.Tensor([(k * ar / n_azim + ab - ar / (2 * n_azim)) for k in range(1, n_azim + 1)])
        elev, azim = map(lambda t: t.flatten(), torch.meshgrid(elev, azim, indexing='ij'))
        roll = torch.zeros(elev.shape)
        print_log(f'init_rotations: azim={azim.tolist()}, elev={elev.tolist()}, roll={roll.tolist()}')
        R_init = torch.stack([azim, elev, roll], dim=1)
    elif init_type.startswith('random'):
        R_init = random_rotations(N)
    else:
        raise NotImplementedError
    return R_init


def normal_consistency(meshes, icosphere_topology=True, shared_topology=True):
    """Use a x10 faster routine than the one in PyTorch3D when meshes have an icosphere topology"""
    if not icosphere_topology:
        return mesh_normal_consistency(meshes)
    if meshes.isempty():
        return torch.tensor([0.0], dtype=torch.float32, device=meshes.device, requires_grad=True)
    N = len(meshes)
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    edges_packed = meshes.edges_packed()
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()
    face_to_edge = meshes.faces_packed_to_edges_packed()
    F = faces_packed.shape[0]
    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)
        vert_idx = faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]
        vert_edge_pair_idx = torch.arange(len(edge_idx), device=meshes.device).view(-1, 2)
    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2
    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    loss = 1 - torch.cosine_similarity(n0, n1, dim=1)
    if not shared_topology:
        verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
        verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_edge_pair_idx[:, 0]]
        num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
        weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()
        loss = (loss * weights).sum() / N
    else:
        loss = loss.mean()
    return loss


def path_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repeat(mesh, N):
    """
    Returns N copies using the PyTorch `repeat` convention, compared to the current PyTorch3D function `extend` which
    follows the `repeat_interleave` convention
    """
    assert N >= 1
    if N == 1:
        return mesh
    new_verts_list, new_faces_list = [], []
    for _ in range(N):
        new_verts_list.extend(verts.clone() for verts in mesh.verts_list())
        new_faces_list.extend(faces.clone() for faces in mesh.faces_list())
    textures = mesh.textures
    if isinstance(textures, TexturesVertex):
        new_verts_rgb = textures.verts_features_padded().repeat(N, 1, 1)
        new_textures = TexturesVertex(verts_features=new_verts_rgb)
        new_textures._num_verts_per_mesh = textures._num_verts_per_mesh * N
    elif isinstance(textures, TexturesUV):
        maps = textures.maps_padded().repeat(N, 1, 1, 1)
        uvs = textures.verts_uvs_padded().repeat(N, 1, 1)
        faces = textures.faces_uvs_padded().repeat(N, 1, 1)
        new_textures = TexturesUV(maps, faces, uvs)
        new_textures._num_faces_per_mesh = textures._num_faces_per_mesh * N
    else:
        raise NotImplementedError
    return Meshes(verts=new_verts_list, faces=new_faces_list, textures=new_textures)


def roll_to_rotation_matrix(roll, as_degree=True):
    """Angle with +X in XY plane"""
    if isinstance(roll, (int, float)):
        roll = torch.Tensor([roll])
    roll_rad = roll * np.pi / 180 if as_degree else roll
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    cos, sin = torch.cos(roll_rad), torch.sin(roll_rad)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def safe_model_state_dict(state_dict):
    """Convert a state dict saved from a DataParallel module to normal module state_dict."""
    if not next(iter(state_dict)).startswith('module.'):
        return state_dict
    return keymap(lambda s: s[7:], state_dict, factory=OrderedDict)


def get_torch_device(gpu=None, verbose=False):
    if torch.cuda.is_available():
        device, nb_dev = torch.device(gpu) if gpu is not None else torch.device('cuda:0'), torch.cuda.device_count()
    else:
        device, nb_dev = torch.device('cpu'), None
    if verbose:
        print_log(f'Torch device state: device={device}, nb_dev={nb_dev}')
    return device


@torch.no_grad()
def render_rotated_views(mesh, img_size=256, n_views=50, elev=30, dist=2.5, R=None, T=None, bkg=None, renderer=None, rend_kwargs=None, eye_light=False, device=None):
    device = get_torch_device() if device is None else device
    rend_kwargs = {} if rend_kwargs is None else rend_kwargs
    renderer = Renderer(img_size, **rend_kwargs) if renderer is None else renderer
    if eye_light:
        if R is not None:
            raise NotImplementedError
        if isinstance(renderer.lights, AmbientLights):
            kwargs = renderer.init_kwargs
            kwargs['lights'] = {'name': 'directional', 'direction': [[0, 0, -1]], 'ambient_color': [[0.7, 0.7, 0.7]], 'diffuse_color': [[0.3, 0.3, 0.3]], 'specular_color': [[0.0, 0.0, 0.0]]}
            kwargs['shading_type'] = 'phong'
            kwargs['faces_per_pixel'] = 1
            renderer = Renderer(img_size, **kwargs)
    elev, dist = 0 if R is not None else elev, 0 if T is not None else dist
    R, T = R if R is not None else torch.eye(3), T if T is not None else torch.zeros(3)
    if bkg is not None:
        if bkg.shape[-1] != img_size:
            bkg = F.interpolate(bkg[None], size=(img_size, img_size), mode='bilinear', align_corners=False)[0]
    mesh, renderer = mesh, renderer
    azim = torch.linspace(-180, 180, n_views)
    views, B = [], 10
    for k in range((n_views - 1) // B + 1):
        R_view = look_at_view_transform(dist=1, elev=elev, azim=azim[k * B:(k + 1) * B], device=device)[0]
        T_view = torch.Tensor([[0.0, 0.0, dist]]).expand(len(R_view), -1)
        if eye_light:
            d = torch.Tensor([[0, 0, -1]]) @ R_view.transpose(1, 2)
            renderer.update_lights(direction=d)
        views.append(renderer(mesh.extend(len(R_view)), R_view @ R, T_view + T, viz_purpose=True).clamp(0, 1).cpu())
    rec, alpha = torch.cat(views, dim=0).split([3, 1], dim=1)
    if bkg is not None:
        rec = rec * alpha + (1 - alpha) * bkg.cpu()
    return rec


def path_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def get_files_from(dir_path, valid_extensions=None, recursive=False, sort=False):
    path = path_exists(dir_path)
    if recursive:
        files = [f.absolute() for f in path.glob('**/*') if f.is_file()]
    else:
        files = [f.absolute() for f in path.glob('*') if f.is_file()]
    if valid_extensions is not None:
        valid_extensions = [valid_extensions] if isinstance(valid_extensions, str) else valid_extensions
        valid_extensions = [('.{}'.format(ext) if not ext.startswith('.') else ext) for ext in valid_extensions]
        files = list(filter(lambda f: f.suffix in valid_extensions, files))
    return sorted(files) if sort else files


def save_gif(imgs_or_path, name, in_ext='jpg', size=None, total_sec=10):
    if isinstance(imgs_or_path, (str, Path)):
        path = path_exists(imgs_or_path)
        files = sorted(get_files_from(path, in_ext), key=lambda p: int(p.stem))
        try:
            imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in files]
        except OSError as e:
            print_warning(e)
            return None
    else:
        imgs, path = [convert_to_img(i).convert('P', palette=Image.ADAPTIVE) for i in imgs_or_path], Path('.')
    if len(imgs) > 0:
        if size is not None and size != imgs[0].size:
            imgs = list(map(lambda i: resize(i, size=size), imgs))
        tpf = int(total_sec * 1000 / len(imgs))
        imgs[0].save(path.parent / name, optimize=False, save_all=True, append_images=imgs[1:], duration=tpf, loop=0)


def save_mesh_as_gif(mesh, filename, img_size=256, n_views=50, elev=30, dist=2.732, R=None, T=None, bkg=None, renderer=None, rend_kwargs=None, eye_light=False):
    imgs = render_rotated_views(mesh, img_size, n_views, elev, dist, R=R, T=T, bkg=bkg, renderer=renderer, rend_kwargs=rend_kwargs, eye_light=eye_light)
    save_gif(imgs, filename)


def _save(f, verts, faces, decimal_places: Optional[int]=None, *, verts_rgb: Optional[torch.Tensor]=None, verts_uvs: Optional[torch.Tensor]=None, faces_uvs: Optional[torch.Tensor]=None, save_texture: bool=False) ->None:
    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)
    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return
    if verts_rgb is not None and (verts_rgb.dim() != 2 or verts_rgb.size(1) != 3):
        message = "'verts_rgb' should either be None or of shape (num_verts, 3)."
        raise ValueError(message)
    verts, faces = verts.cpu(), faces.cpu()
    lines = ''
    if len(verts):
        if decimal_places is None:
            float_str = '%f'
        else:
            float_str = '%' + '.%df' % decimal_places
        V, D = verts.shape
        for i in range(V):
            vert = [(float_str % verts[i, j]) for j in range(D)]
            if verts_rgb is not None:
                vert += [(float_str % verts_rgb[i, j]) for j in range(3)]
            lines += 'v %s\n' % ' '.join(vert)
    if save_texture:
        if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)
        if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)
        verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()
        if len(verts_uvs):
            uV, uD = verts_uvs.shape
            for i in range(uV):
                uv = [(float_str % verts_uvs[i, j]) for j in range(uD)]
                lines += 'vt %s\n' % ' '.join(uv)
    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn('Faces have invalid indices')
    if len(faces):
        F, P = faces.shape
        for i in range(F):
            if save_texture:
                face = [('%d/%d' % (faces[i, j] + 1, faces_uvs[i, j] + 1)) for j in range(P)]
            else:
                face = [('%d' % (faces[i, j] + 1)) for j in range(P)]
            if i + 1 < F:
                lines += 'f %s\n' % ' '.join(face)
            elif i + 1 == F:
                lines += 'f %s' % ' '.join(face)
    f.write(lines)


def save_mesh_as_obj(mesh, filename):
    assert len(mesh) == 1
    verts, faces = mesh.get_mesh_verts_faces(0)
    if isinstance(mesh.textures, TexturesUV):
        txt = mesh.textures
        save_obj(filename, verts, faces, verts_uvs=txt.verts_uvs_padded()[0], faces_uvs=txt.faces_uvs_padded()[0], texture_map=txt.maps_padded()[0])
    elif isinstance(mesh.textures, TexturesVertex):
        verts_rgb = mesh.textures.verts_features_list()[0].clamp(0, 1)
        save_obj(filename, verts, faces, verts_rgb=verts_rgb)
    else:
        save_obj(filename, verts, faces)


def torch_to(inp, device, non_blocking=False):
    nb = non_blocking
    if isinstance(inp, torch.Tensor):
        return inp
    elif isinstance(inp, (list, tuple)):
        return type(inp)(map(lambda t: t if isinstance(t, torch.Tensor) else t, inp))
    elif isinstance(inp, dict):
        return valmap(lambda t: t if isinstance(t, torch.Tensor) else t, inp)
    else:
        raise NotImplementedError


class use_seed:

    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            self.torch_cudnn_deterministic = torch.backends.cudnn.deterministic
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)
            torch.backends.cudnn.deterministic = self.torch_cudnn_deterministic

    def __call__(self, f):

        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)
        return wrapper


class Unicorn(nn.Module):
    name = 'unicorn'

    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.init_kwargs = deepcopy(kwargs)
        self.init_kwargs['img_size'] = img_size
        self._init_encoder(img_size, **kwargs.get('encoder', {}))
        self._init_meshes(**kwargs.get('mesh', {}))
        self.renderer = Renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self._debug = False

    @property
    def n_features(self):
        return self.encoder.out_ch if self.shared_encoder else self.encoder_sh.out_ch

    @property
    def tx_code_size(self):
        return self.txt_generator.current_code_size

    @property
    def sh_code_size(self):
        return self.deform_field.current_code_size

    def _init_encoder(self, img_size, **kwargs):
        self.shared_encoder = kwargs.pop('shared', True)
        if self.shared_encoder:
            self.encoder = Encoder(img_size, **kwargs)
        else:
            self.encoder_sh = Encoder(img_size, **kwargs)
            self.encoder_tx = Encoder(img_size, **kwargs)
            self.encoder_pose = Encoder(img_size, **kwargs)
            if len(self.init_kwargs.get('background', {})) > 0:
                self.encoder_bg = Encoder(img_size, **kwargs)

    def _init_meshes(self, **kwargs):
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init', 'sphere')
        scale = kwargs.pop('scale', 1)
        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(4 if 'hr' in mesh_init else 3)
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_src.get_mesh_verts_faces(0)[0])[None])
        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))
        dfield_kwargs = kwargs.pop('deform_fields', {})
        tgen_kwargs = kwargs.pop('texture_uv', {})
        assert len(kwargs) == 0
        self.deform_field = ProgressiveField(inp_dim=self.n_features, name='deformation', **dfield_kwargs)
        self.txt_generator = ProgressiveGenerator(inp_dim=self.n_features, **tgen_kwargs)

    def _init_rend_predictors(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.pose_step = True
        NF, NP = self.n_features, self.n_poses
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')
        azim, elev, roll = [((e[1] - e[0]) / n) for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        if self.n_poses == 1:
            self.register_buffer('R_range', torch.Tensor([azim * 0.5, elev * 0.5, roll * 0.5]))
        else:
            self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range
        self.scale_regressor = create_mlp(NF, 3, NU, NL, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))
        if NP > 1:
            self.proba_regressor = create_mlp(NF, NP, NU, NL)
        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        return 1 if self.hard_select else self.n_poses

    @property
    def hard_select(self):
        if self.alternate_optim and not self._debug:
            return False if self.training and self.pose_step else True
        else:
            return False

    def _init_background_model(self, img_size, **kwargs):
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGenerator(inp_dim=self.n_features, img_size=img_size, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.milestones = {'constant_txt': kwargs.pop('constant_txt', kwargs.pop('contant_text', 0)), 'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0), 'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0), 'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0), 'freeze_shape': kwargs.pop('freeze_shape', 0), 'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt))}
        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        kwargs = deepcopy(kwargs)
        loss_weights = {'rgb': kwargs.pop('rgb_weight', 1.0), 'normal': kwargs.pop('normal_weight', 0), 'laplacian': kwargs.pop('laplacian_weight', 0), 'perceptual': kwargs.pop('perceptual_weight', 0), 'uniform': kwargs.pop('uniform_weight', 0), 'neighbor': kwargs.pop('neighbor_weight', kwargs.pop('swap_weight', 0))}
        name = kwargs.pop('name', 'mse')
        perceptual_kwargs = kwargs.pop('perceptual', {})
        self.nbr_memsize = kwargs.pop('nbr_memsize', kwargs.pop('swap_memsize', MEMSIZE))
        self.nbr_n_vpbins = kwargs.pop('nbr_n_vpbins', kwargs.pop('swap_n_vpbins', N_VPBINS))
        self.nbr_min_angle = kwargs.pop('nbr_min_angle', kwargs.pop('swap_min_angle', MIN_ANGLE))
        self.nbr_memory = {k: torch.empty(0) for k in ['sh', 'tx', 'S', 'R', 'T', 'bg', 'img']}
        kwargs.pop('swap_equal_bins', False)
        assert len(kwargs) == 0, kwargs
        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        super()
        self.mesh_src = self.mesh_src
        self.renderer = self.renderer
        return self

    def forward(self, inp, debug=False):
        self._debug = debug
        imgs, K, B = inp['imgs'], self.n_candidates, len(inp['imgs'])
        perturbed = self.training and np.random.binomial(1, p=0.2)
        average_txt = self.is_live('constant_txt') or perturbed and self.use_mean_txt and self.is_live('mean_txt')
        meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs, average_txt)
        if self.alternate_optim:
            if self.pose_step:
                meshes, bkgs = meshes.detach(), bkgs.detach() if self.pred_background else None
            else:
                R, T = R.detach(), T.detach()
        meshes_to_render = repeat(meshes, len(T) // len(meshes))
        fgs, alpha = self.renderer(meshes_to_render, R, T).split([3, 1], dim=1)
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        losses, select_idx = self.compute_losses(meshes, imgs, rec, R, T, average_txt=average_txt)
        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
            self._debug = False
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec
        return out

    def predict_mesh_pose_bkg(self, imgs, average_txt=False):
        if self.shared_encoder:
            features = self.encoder(imgs)
            meshes = self.predict_meshes(features, average_txt=average_txt)
            R, T = self.predict_poses(features)
            bkgs = self.predict_background(features) if self.pred_background else None
        else:
            features_sh, features_tx = self.encoder_sh(imgs), self.encoder_tx(imgs)
            meshes = self.predict_meshes(features_sh, features_tx, average_txt=average_txt)
            R, T = self.predict_poses(self.encoder_pose(imgs))
            bkgs = self.predict_background(self.encoder_bg(imgs)) if self.pred_background else None
        return meshes, (R, T), bkgs

    def predict_meshes(self, features, features_tx=None, average_txt=False):
        if features_tx is None:
            features_tx = features
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        meshes = self.mesh_src.extend(len(features))
        meshes.offset_verts_(self.predict_disp_verts(verts, features))
        meshes.textures = self.predict_textures(faces, features_tx, average_txt)
        meshes.scale_verts_(self.predict_scales(features))
        return meshes

    def predict_disp_verts(self, verts, features):
        disp_verts = self.deform_field(verts, features)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def predict_textures(self, faces, features, average_txt=False):
        B = len(features)
        maps = self.txt_generator(features)
        if average_txt:
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.2)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    def predict_poses(self, features):
        B = len(features)
        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)
        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0)
        R_pred = R_pred.tanh()[..., [1, 0, 2]]
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)
        if self.n_poses > 1:
            weights = self.proba_regressor(features.view(B, -1)).permute(1, 0)
            self._pose_proba = torch.softmax(weights, dim=0)
            if self.hard_select:
                indices = self._pose_proba.max(0)[1]
                select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
                R, T = map(select_fn, [R, T])
        return R, T

    def predict_background(self, features):
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    def compute_losses(self, meshes, imgs, rec, R, T, average_txt=False):
        K, B = self.n_candidates, len(imgs)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        if self.training:
            update_3d, update_pose = (not self.pose_step, self.pose_step) if self.alternate_optim else (True, True)
        else:
            update_3d, update_pose = False, False
        if 'rgb' in losses:
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(rec, imgs).flatten(1).mean(1)
        if 'perceptual' in losses:
            losses['perceptual'] = self.loss_weights['perceptual'] * self.perceptual_loss(rec, imgs)
        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * normal_consistency(meshes)
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * laplacian_smoothing(meshes, method='uniform')
        if update_3d and 'neighbor' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            verts, faces, textures = meshes.verts_padded(), meshes.faces_padded(), meshes.textures
            scales = self._scales[:, None]
            z_sh, z_tx = [m._latent for m in [self.deform_field, self.txt_generator]]
            z_bg = self.bkg_generator._latent if self.pred_background else torch.empty(B, 1, device=dev)
            for n, t in [('sh', z_sh), ('tx', z_tx), ('bg', z_bg), ('S', scales), ('R', R), ('T', T), ('img', imgs)]:
                self.nbr_memory[n] = torch.cat([self.nbr_memory[n], t.detach()])[-self.nbr_memsize:]
            min_angle, nb_vpbins = self.nbr_min_angle, self.nbr_n_vpbins
            with torch.no_grad():
                sim_sh = (z_sh[None] - self.nbr_memory['sh'][:, None]).pow(2).sum(-1)
                sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).pow(2).sum(-1)
                angles = cpu_angle_between(self.nbr_memory['R'][:, None], R[None]).view(sim_sh.shape)
                angle_bins = torch.randint(0, nb_vpbins, (B,), device=dev).float()
                bin_size = (180.0 - min_angle) / nb_vpbins
                min_angles, max_angles = [((angle_bins + k) * bin_size + min_angle) for k in range(2)]
                invalid_mask = (angles < min_angles).float() + (angles >= max_angles).float()
                idx_sh, idx_tx = map(lambda t: (t + t.max() * invalid_mask).argmin(0), [sim_sh, sim_tx])
            v_src, f_src = self.mesh_src.get_mesh_verts_faces(0)
            nbr_list, select = [], lambda n, indices: self.nbr_memory[n][indices]
            sh_imgs, tx_imgs = select('img', idx_sh), select('img', idx_tx)
            with torch.no_grad():
                if self.shared_encoder:
                    sh_features = self.encoder(sh_imgs)
                    sh_tx = self.predict_textures(f_src, sh_features, average_txt)
                    sh_S = self.predict_scales(sh_features)[:, None]
                    sh_R, sh_T = self.predict_poses(sh_features)
                    sh_bg = self.predict_background(sh_features) if self.pred_background else None
                else:
                    sh_tx = self.predict_textures(f_src, self.encoder_tx(sh_imgs), average_txt)
                    sh_S = self.predict_scales(self.encoder_sh(sh_imgs))[:, None]
                    sh_R, sh_T = self.predict_poses(self.encoder_pose(sh_imgs))
                    sh_bg = self.predict_background(self.encoder_bg(sh_imgs)) if self.pred_background else None
            sh_mesh = Meshes(verts / scales * sh_S, faces, sh_tx)
            nbr_list.append([sh_mesh, sh_R, sh_T, sh_bg, sh_imgs])
            with torch.no_grad():
                if self.shared_encoder:
                    tx_features = self.encoder(tx_imgs)
                    tx_verts = v_src + self.predict_disp_verts(v_src, tx_features).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_features)[:, None]
                    tx_R, tx_T = self.predict_poses(tx_features)
                    tx_bg = self.predict_background(tx_features) if self.pred_background else None
                else:
                    tx_feat_sh = self.encoder_sh(tx_imgs)
                    tx_verts = v_src + self.predict_disp_verts(v_src, tx_feat_sh).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_feat_sh)[:, None]
                    tx_R, tx_T = self.predict_poses(self.encoder_pose(tx_imgs))
                    tx_bg = self.predict_background(self.encoder_bg(tx_imgs)) if self.pred_background else None
            tx_mesh = Meshes(tx_verts * tx_S, faces, textures)
            nbr_list.append([tx_mesh, tx_R, tx_T, tx_bg, tx_imgs])
            loss = 0.0
            for nbr_inp in nbr_list:
                nbr_mesh, R, T, bkgs, imgs = nbr_inp
                rec_sw, alpha_sw = self.renderer(nbr_mesh, R, T).split([3, 1], dim=1)
                rec_sw = rec_sw * alpha_sw + (1 - alpha_sw) * bkgs[:B] if bkgs is not None else rec_sw
                if 'rgb' in losses:
                    loss += self.loss_weights['rgb'] * self.criterion(rec_sw, imgs).flatten(1).mean(1)
                if 'perceptual' in losses:
                    loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw, imgs)
            losses['neighbor'] = self.loss_weights['neighbor'] * loss
        if update_pose and 'uniform' in losses:
            losses['uniform'] = self.loss_weights['uniform'] * (self._pose_proba.mean(1) - 1 / K).abs().mean()
        dist = sum(losses.values())
        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B
        else:
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = v.mean()
        losses['total'] = dist.mean()
        return losses, select_idx

    def iter_step(self):
        self.cur_iter += 1
        if self.alternate_optim and self.cur_iter % self.alternate_optim == 0:
            self.pose_step = not self.pose_step

    def step(self):
        self.cur_epoch += 1
        self.deform_field.step()
        self.txt_generator.step()
        if self.pred_background:
            self.bkg_generator.step()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch
        self.deform_field.set_cur_milestone(epoch)
        self.txt_generator.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state and name != 'T_init':
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def get_synthetic_textures(self, colored=False):
        verts = self.mesh_src.verts_packed()
        if colored:
            colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        else:
            colors = torch.ones(verts.shape, device=verts.device) * 0.8
        return TexturesVertex(verts_features=colors[None])

    def get_prototype(self):
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent = torch.zeros(1, self.n_features, device=verts.device)
        meshes = self.mesh_src.offset_verts(self.deform_field(verts, latent).view(-1, 3))
        return meshes

    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None
        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_init[-1], elev=elev, azim=azim, device=mesh.device)
        return self.renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None
        path = path_mkdir(path or Path('.'))
        d, elev = self.T_init[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.renderer)

    @torch.no_grad()
    def quantitative_eval(self, loader, device, evaluator=None):
        self.eval()
        if loader.dataset.name in ['cub_200']:
            if evaluator is None:
                evaluator = ProxyEvaluator()
            for inp, _ in loader:
                mask_gt = inp['masks']
                meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'])
                mask_pred = self.renderer(meshes, R, T, viz_purpose=True).split([3, 1], dim=1)[1]
                if mask_pred.shape != mask_gt.shape:
                    mask_pred = F.interpolate(mask_pred, mask_gt.shape[-2:], mode='bilinear', align_corners=False)
                mask_pred = (mask_pred > 0.5).float()
                evaluator.update(mask_pred, mask_gt)
        else:
            if loader.dataset.name == 'p3d_car':
                print_warning('make sure that the canonical axes of predicted shapes correspond to the GT shapes axes')
            if evaluator is None:
                evaluator = MeshEvaluator()
            for inp, labels in loader:
                if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                    break
                meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'])
                if not torch.all(inp['poses'] == -1):
                    verts, faces = meshes.verts_padded(), meshes.faces_padded()
                    R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].split([3, 1], dim=2))
                    verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                    meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
                evaluator.update(meshes, torch_to(labels, device))
        return OrderedDict(zip(evaluator.metrics.names, evaluator.metrics.values))

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')
        renderer = self.renderer
        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs']
            meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs)
            rec, alpha = renderer(meshes, R, T).split([3, 1], dim=1)
            if bkgs is not None:
                rec = rec * alpha + (1 - alpha) * bkgs
            B, NV = len(imgs), 50
            d, e = self.T_init[-1], np.mean(self.elev_range)
            for k in range(B):
                i = str(j * B + k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprec_full.png')
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')
                mcenter = normalize(meshes[k])
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter.textures = self.get_synthetic_textures(colored=True)
                save_mesh_as_obj(mcenter, path / f'{i}_meshuv.obj')
                save_mesh_as_gif(mcenter, path / f'{i}_meshuv_raw.gif', dist=d, elev=e, renderer=renderer)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Blur,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GiraffeGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIMLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_monniert_unicorn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

