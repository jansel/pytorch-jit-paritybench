import sys
_module = sys.modules[__name__]
del sys
GAN = _module
blending = _module
sefa = _module
generate_images = _module
icgan = _module
generate = _module
guided = _module
load = _module
compute = _module
extractors = _module
inception = _module
swav = _module
frechet = _module
kernel = _module
prdc = _module
sampling = _module
jacnorm = _module
langevin = _module
multitrunc = _module
polarity = _module
blur = _module
image = _module
normal = _module
losses = _module
cross_entropy = _module
path_length_regularization = _module
r1_penalty = _module
softplus = _module
deepconvolutional = _module
deepinvolutional = _module
equivariant = _module
emerging_conv2d = _module
inverse_op_naive = _module
optimal_transport = _module
stylehypermixerfly = _module
train_v0 = _module
trainer = _module
wrappers = _module
fastgan = _module
ops = _module
stylegan2 = _module
stylegan = _module
stylegan2 = _module
stylegan3 = _module
maua = _module
audiovisual = _module
audioreactive = _module
audio = _module
latent = _module
mir = _module
audio = _module
correlation = _module
efficient_quantile = _module
setup = _module
processing = _module
beat = _module
constantq = _module
convert = _module
helpers = _module
pitch = _module
segment = _module
spectral = _module
video = _module
latent = _module
noise = _module
patch = _module
sample = _module
signal = _module
util = _module
generate = _module
interactive = _module
base = _module
stylegan2 = _module
noise_parameterization = _module
stylegan2 = _module
stylegan3 = _module
primitives = _module
latents = _module
merge = _module
noise = _module
other = _module
render = _module
ffmpeg = _module
gpu2gl = _module
memmap = _module
autoregressive = _module
generate = _module
infinite = _module
min_dalle = _module
generate = _module
rq_dalle = _module
ru_dalle = _module
api = _module
finetune = _module
generate = _module
cli = _module
diffusion = _module
entrypoint = _module
style = _module
laion_clip_retrieval = _module
multicrop = _module
ranker = _module
loop = _module
loop_direct = _module
finetune_stable = _module
image = _module
interp_loop = _module
interpolate = _module
klmc2_animation = _module
load = _module
outpaint = _module
processors = _module
base = _module
glid3xl = _module
glide = _module
guided = _module
latent = _module
stable = _module
video = _module
flow = _module
consistency = _module
lib = _module
mm = _module
sniklaus = _module
utils = _module
grad = _module
loss = _module
nca = _module
generate = _module
train = _module
cutouts = _module
image = _module
io = _module
noise = _module
video = _module
optimizers = _module
parameterizations = _module
biggan = _module
fourier = _module
pixel = _module
rgb = _module
vqgan = _module
perceptors = _module
aesthetic = _module
clip = _module
nima = _module
vgg_kbc = _module
vgg_pgg = _module
prompt = _module
image = _module
image_multires = _module
omnimae = _module
video = _module
video_multires = _module
submodules = _module
bulk = _module
comparison = _module
bsrgan = _module
latent_diffusion = _module
realesrgan = _module
swinir = _module
waifu = _module
single = _module
frame_by_frame = _module
framerate = _module
rife = _module
spatiotemporal = _module
iseebetter = _module
utility = _module
setup = _module
diffusion = _module

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


import random


import re


import matplotlib.pyplot as plt


import numpy as np


import torch


import torchvision


from typing import Generator as PythonGenerator


from typing import List


import torchvision as tv


from numpy import sqrt


from torch.nn.functional import one_hot


import warnings


import torchvision.transforms as transforms


from scipy.stats import truncnorm


from torch import nn


import sklearn.metrics


from copy import deepcopy


from functools import partial


from typing import Callable


from typing import Union


from torch.utils.data import Dataset


from torch.utils.data.dataloader import DataLoader


from torchvision.transforms.functional import to_tensor


import torch.nn as nn


from math import ceil


from math import sqrt


from torch.distributions.multivariate_normal import MultivariateNormal


from torchvision.transforms.functional import resize


from torch.nn.functional import interpolate


from torchvision.transforms import Normalize


import inspect


import torchvision.transforms as tvt


from math import floor


import torch.multiprocessing as mp


from torch.utils.data import Dataset as TorchDataset


from torch.autograd import grad


from torch.nn import GELU


from torch.nn import LayerNorm


from torch.nn import Module


from torch.nn import Sequential


from torch.nn import UpsamplingBilinear2d


from typing import Optional


from typing import Tuple


from torch.nn.functional import conv2d


from torch.nn.functional import l1_loss


from torch.nn.functional import mse_loss


from torch.nn.functional import pad


import torch.nn.functional as F


from random import choice


from torchvision import transforms as vision


from torchvision.transforms.functional import normalize


from typing import Any


from typing import Dict


from torch.nn import Module as TorchModule


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Generator


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.dataset import TensorDataset


from torch import Tensor


import torchaudio


from scipy import signal


from torchaudio.functional import resample


import matplotlib.patches as patches


import scipy


import scipy.stats


import sklearn.cluster


import math


from torch.utils import cpp_extension


from torch.nn.functional import conv1d


from torchaudio.functional import contrast


from torchaudio.functional import highpass_biquad


from torchaudio.functional import lowpass_biquad


import sklearn


from uuid import uuid4


import matplotlib


import scipy.ndimage as ndi


from scipy.interpolate import splev


from scipy.interpolate import splrep


import logging


import time


from torchvision.utils import save_image


import torchvision.transforms as T


from torch.optim.lr_scheduler import OneCycleLR


from torch.nn import ReflectionPad2d


from torchvision.datasets.folder import IMG_EXTENSIONS


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomRotation


from torchvision.transforms.functional import center_crop


from torchvision.transforms.functional import to_pil_image


from torch.nn.functional import grid_sample


from inspect import isfunction


from torch import autocast


from torch import einsum


from functools import reduce


from queue import Empty


from queue import Queue


from torchvision.transforms.functional import gaussian_blur


import matplotlib.pylab as pl


import torchvision.models as models


from torch.nn import functional as F


from torchvision import transforms as T


from torchvision.transforms import functional as TF


from scipy.special import comb


from torchvision.transforms.functional import adjust_sharpness


from time import sleep


from torch import optim


import torchvision.transforms.functional as tvtf


from torchvision import models


from torchvision import transforms


from collections import OrderedDict


from torch.utils.model_zoo import load_url


import torch.utils.checkpoint as checkpoint


from torch.nn.modules.utils import _ntuple


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data import DistributedSampler


from time import time


import functools


URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'


def download(url, filename):
    headers = {'User-Agent': 'Maua', 'From': 'https://github.com/maua-maua-maua/maua'}
    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f'Request to {url} returned status code {r.status_code}')
    file_size = int(r.headers.get('Content-Length', 0))
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = f'Downloading {filename}' + (' (Unknown total file size)' if file_size == 0 else '')
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, 'read', total=file_size, desc=desc) as r_raw:
        with path.open('wb') as f:
            shutil.copyfileobj(r_raw, f)
    return path


class Inception(nn.Module):

    def __init__(self, path='modelzoo/inception-2015-12-05.pt'):
        super().__init__()
        if not os.path.exists(path):
            download(URL, path)
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers

    @torch.inference_mode()
    def forward(self, x):
        """
        Get the inception features without resizing
        x: Image with values in range [-1,1]
        """
        with disable_gpu_fuser_on_pt19():
            bs = x.shape[0]
            assert x.shape[2] == 299 and x.shape[3] == 299
            features = self.layers.forward(x).view((bs, 2048))
            return features


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, widen=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, normalize=False, output_dim=0, hidden_mlp=0, nmb_prototypes=0, eval_mode=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l2norm = normalize
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(nn.Linear(num_out_filters * block.expansion, hidden_mlp), nn.BatchNorm1d(hidden_mlp), nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim))
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.eval_mode:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in inputs]), return_counts=True)[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class InitialBlur(torch.nn.Module):

    def __init__(self, batch_size, blur_init_sigma, blur_fade_kimg, **kwargs) ->None:
        super().__init__()
        self.init_sigma = blur_init_sigma
        self.fade_kimg = batch_size / 32 * blur_fade_kimg
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('InitialBlur')
        parser.add_argument('--blur_init_sigma', type=float, default=10, help='Strength of initial blur at start of training')
        parser.add_argument('--blur_fade_kimg', type=int, default=200, help='How many thousands of images to fade out blurring at start of training')
        return parent_parser

    def forward(self, lightning_module, reals, fakes, **kwargs):
        blur_sigma = max(1 - lightning_module.global_step * self.batch_size / (self.fade_kimg * 1000.0), 0) * self.init_sigma if self.fade_kimg > 0 else 0
        if blur_sigma > 0:
            blur_size = floor(blur_sigma * 3)
            blur_size = blur_size + (1 - blur_size % 2)
            if reals is not None:
                reals = gaussian_blur2d(reals, kernel_size=(blur_size, blur_size), sigma=(blur_sigma, blur_sigma))
            fakes = gaussian_blur2d(fakes, kernel_size=(blur_size, blur_size), sigma=(blur_sigma, blur_sigma))
        return reals, fakes


class NormalLatentDistribution(torch.nn.Module):

    def __init__(self, batch_size, z_dim, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.dev = torch.nn.Parameter(torch.empty(0))

    def forward(self):
        return torch.randn((self.batch_size, self.z_dim), device=self.dev.device)


class Loss(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def pre_G(self, **kwargs):
        pass

    def pre_D(self, **kwargs):
        pass

    def forward(self, **Kwargs):
        return 0


class DiscriminatorCrossEntropy(Loss):

    def __init__(self, logits=True, **kwargs) ->None:
        super().__init__()
        self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none') if logits else torch.nn.BCELoss(reduction='none')

    def forward(self, lightning_module, preds_real, preds_fake, **kwargs):
        loss_D_real = self.cross_entropy(preds_real, torch.ones_like(preds_real))
        loss_D_fake = self.cross_entropy(preds_fake, torch.zeros_like(preds_fake))
        lightning_module.log_dict(dict(loss_D_real=loss_D_real.mean(), loss_D_fake=loss_D_fake.mean(), preds_real=preds_real.sign().mean(), preds_fake=preds_fake.sign().mean()))
        return loss_D_real + loss_D_fake


class GeneratorCrossEntropy(Loss):

    def __init__(self, logits=True, **kwargs) ->None:
        super().__init__()
        self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none') if logits else torch.nn.BCELoss(reduction='none')

    def forward(self, lightning_module, preds_fake, **kwargs):
        loss_G = self.cross_entropy(preds_fake, torch.ones_like(preds_fake))
        lightning_module.log_dict(dict(loss_G=loss_G.mean()))
        return loss_G


class GeneratorPathLengthRegularization(Loss):

    def __init__(self, pl_weight, pl_interval, pl_decay, pl_batch_shrink, **kwargs) ->None:
        super().__init__()
        self.decay = pl_decay
        self.weight = pl_weight
        self.interval = pl_interval
        self.batch_shrink = pl_batch_shrink
        self.c = 0
        pl_mean = torch.ones(())
        self.register_buffer('pl_mean', pl_mean)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('PathLengthRegularization')
        parser.add_argument('--pl_weight', type=float, default=2, help='Strength of path length regularization')
        parser.add_argument('--pl_batch_shrink', type=int, default=2, help='Factor to reduce batch size by for regularization calculation')
        parser.add_argument('--pl_decay', type=float, default=0.01, help='Exponential moving average decay of mean path length')
        parser.add_argument('--pl_interval', type=int, default=4, help='How often to apply regularization (lazy regularization)')
        return parent_parser

    def pre_G(self, latent, **kwargs):
        latent.requires_grad_()

    def forward(self, lightning_module, latent, fakes, preds_fake, **kwargs):
        self.c += 1
        if self.c % self.interval == 0:
            pl_noise = torch.randn_like(fakes) / np.sqrt(fakes.shape[2] * fakes.shape[3])
            pl_grads = grad(outputs=[(fakes * pl_noise).sum()], inputs=[latent], create_graph=True, only_inputs=True)[0]
            path_lengths = pl_grads.square()
            if path_lengths.dim() == 3:
                path_lengths = path_lengths.sum(2)
            path_lengths = path_lengths.mean(1).sqrt()
            pl_mean = torch.lerp(self.pl_mean, path_lengths.mean(), self.decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (path_lengths - pl_mean).square()
            loss_G_pl = pl_penalty * self.weight
            lightning_module.log_dict(dict(pl_penalty=pl_penalty.mean(), loss_G_pl=loss_G_pl.mean()))
            return loss_G_pl
        else:
            return torch.zeros_like(preds_fake)


class DiscriminatorR1Penalty(Loss):

    def __init__(self, batch_size, image_size, r1_gamma, r1_interval, **kwargs) ->None:
        super().__init__()
        self.interval = r1_interval
        self.gamma = r1_gamma
        if self.gamma is None:
            self.gamma = 0.0002 * image_size ** 2 / batch_size
        self.gamma /= 2
        self.c = 0

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('R1Penalty')
        parser.add_argument('--r1_gamma', type=float, default=None, help='Strength of R1 penalty, None uses heuristic introduced in StyleGAN2-ADA')
        parser.add_argument('--r1_interval', type=int, default=16, help='How often to apply penalty (lazy regularization)')
        return parent_parser

    def pre_D(self, reals, **kwargs):
        reals.requires_grad_()

    def forward(self, lightning_module, preds_real, reals, **kwargs):
        self.c += 1
        if self.c % self.interval == 0:
            r1_grads = grad(outputs=[preds_real.sum()], inputs=[reals], create_graph=True, only_inputs=True)[0]
            penalty = r1_grads.square().sum((1, 2, 3))
            loss_D_r1 = penalty * self.gamma
            lightning_module.log_dict(dict(r1_penalty=penalty.mean(), loss_D_r1=loss_D_r1.mean()))
            return loss_D_r1
        else:
            return torch.zeros_like(preds_real)


class DiscriminatorSoftPlus(Loss):

    def forward(self, lightning_module, preds_real, preds_fake, **kwargs):
        loss_D_real = torch.nn.functional.softplus(-preds_real)
        loss_D_fake = torch.nn.functional.softplus(preds_fake)
        lightning_module.log_dict(dict(loss_D_real=loss_D_real.mean(), loss_D_fake=loss_D_fake.mean(), preds_real=preds_real.sign().mean(), preds_fake=preds_fake.sign().mean()))
        return loss_D_real + loss_D_fake


class GeneratorSoftPlus(Loss):

    def forward(self, lightning_module, preds_fake, **kwargs):
        loss_G = torch.nn.functional.softplus(-preds_fake)
        lightning_module.log_dict(dict(loss_G=loss_G.mean()))
        return loss_G


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class DeepConvolutionalGenerator(torch.nn.Module):

    def __init__(self, image_size=64, z_dim=100, ngf=64, img_channels=3, **kwargs):
        super().__init__()
        nb = round(np.log2(image_size)) - 2
        nfs = [z_dim] + list(reversed([min(ngf * 2 ** i, ngf * 8) for i in range(nb)])) + [img_channels]
        blocks = []
        for b, (nf_prev, nf_next) in enumerate(zip(nfs[:-1], nfs[1:])):
            blocks += [torch.nn.ConvTranspose2d(nf_prev, nf_next, kernel_size=4, stride=1 if b == 0 else 2, padding=0 if b == 0 else 1, bias=False)]
            if b < nb:
                blocks += [torch.nn.BatchNorm2d(nf_next), torch.nn.LeakyReLU(0.2, inplace=True)]
            else:
                blocks += [torch.nn.Tanh()]
        self.main = torch.nn.Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DeepConvolutionalGenerator')
        parser.add_argument('--z_dim', type=int, default=100, help='Size of the latent space')
        parser.add_argument('--ngf', type=int, default=64, help='Base number of filters in the generator')
        return parent_parser

    def forward(self, input):
        return self.main(input[..., None, None])


class DeepConvolutionalDiscriminator(torch.nn.Module):
    override_args = {'logits': True}

    def __init__(self, image_size=64, img_channels=3, ndf=64, **kwargs):
        super().__init__()
        nb = round(np.log2(image_size)) - 2
        nfs = [img_channels] + list([min(ndf * 2 ** i, ndf * 8) for i in range(nb)]) + [1]
        blocks = []
        for b, (nf_prev, nf_next) in enumerate(zip(nfs[:-1], nfs[1:])):
            blocks += [torch.nn.Conv2d(nf_prev, nf_next, kernel_size=4, stride=1 if b == nb else 2, padding=0 if b == nb else 1, bias=False)]
            if b < nb:
                blocks += [torch.nn.BatchNorm2d(nf_next), torch.nn.LeakyReLU(0.2, inplace=True)]
        self.main = torch.nn.Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DeepConvolutionalDiscriminator')
        parser.add_argument('--ndf', type=int, default=64, help='Base number of filters in the discriminator')
        return parent_parser

    def forward(self, input):
        return self.main(input).squeeze()


class DeepInvolutionalGenerator(Module):

    def __init__(self, image_size=64, z_dim=100, ngf=64, img_channels=3, **kwargs):
        super().__init__()
        nb = round(np.log2(image_size)) - 1
        nfs = [z_dim] + list(reversed([min(ngf * 2 ** i, ngf * 8) for i in range(nb)])) + [img_channels]
        res = 1
        blocks = []
        for b, (nf_prev, nf_inter, nf_next) in enumerate(zip(nfs[:-1], [nfs[1]] + nfs[1:-1], nfs[1:])):
            blocks += [Involution2d(nf_prev, nf_inter, sigma_mapping=Sequential(LayerNorm([nf_inter, res, res]), GELU())), LayerNorm([nf_inter, res, res]), GELU(), UpsamplingBilinear2d(scale_factor=2), Involution2d(nf_inter, nf_next, sigma_mapping=Sequential(LayerNorm([nf_next, res * 2, res * 2]), GELU()))]
            if b < nb:
                blocks += [LayerNorm([nf_next, res * 2, res * 2]), GELU()]
            res *= 2
        self.main = Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DeepInvolutionalGenerator')
        parser.add_argument('--z_dim', type=int, default=100, help='Size of the latent space')
        parser.add_argument('--ngf', type=int, default=64, help='Base number of filters in the generator')
        return parent_parser

    def forward(self, input):
        return self.main(input[..., None, None])


class DeepInvolutionalDiscriminator(Module):
    override_args = {'logits': True}

    def __init__(self, image_size=64, img_channels=3, ndf=64, **kwargs):
        super().__init__()
        nb = round(np.log2(image_size)) - 1
        nfs = [img_channels] + list([min(ndf * 2 ** i, ndf * 8) for i in range(nb)]) + [1]
        res = image_size
        blocks = []
        for b, (nf_prev, nf_inter, nf_next) in enumerate(zip(nfs[:-1], [nfs[1]] + nfs[1:-1], nfs[1:])):
            blocks += [Involution2d(nf_prev, nf_inter, sigma_mapping=Sequential(LayerNorm([nf_inter, res, res]), GELU())), LayerNorm([nf_inter, res, res]), GELU(), Involution2d(nf_inter, nf_next, stride=2, sigma_mapping=Sequential(LayerNorm([nf_next, res // 2, res // 2]), GELU()))]
            if b < nb:
                blocks += [LayerNorm([nf_next, res // 2, res // 2]), GELU()]
            res //= 2
        self.main = Sequential(*blocks)
        self.main.apply(weights_init)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('DeepInvolutionalDiscriminator')
        parser.add_argument('--ndf', type=int, default=64, help='Base number of filters in the discriminator')
        return parent_parser

    def forward(self, input):
        return self.main(input).squeeze()


class SteerableGenerator(torch.nn.Module):

    def __init__(self, latent_dim=128, n_mlp=4, n_channels=3, n_filters=64, maximum_frequency=6):
        super(SteerableGenerator, self).__init__()
        self.mapping = torch.nn.Sequential(*([torch.nn.Linear(latent_dim, latent_dim), torch.nn.ELU(inplace=True)] * n_mlp))
        self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
        in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * latent_dim)
        self.input_type = in_type
        irreps = [(1, k) for k in range(maximum_frequency + 1)]
        main_type = nn.FieldType(self.gspace, [self.gspace.irrep(*id) for id in irreps])
        blocks = []
        for c, channels in enumerate([n_filters * 3, n_filters * 3, n_filters * 2, n_filters * 2, n_filters, n_filters, n_channels]):
            out_type = nn.FieldType(self.gspace, channels * [main_type.representation])
            blocks.append(nn.SequentialModule(nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False), nn.NormBatchNorm(out_type), nn.NormNonLinearity(out_type)))
            if c % 2 == 1:
                blocks.append(nn.R2Upsampling(out_type, scale_factor=2))
            in_type = out_type
        spectral_type = nn.FieldType(self.gspace, channels * [self.gspace.fibergroup.spectral_regular_representation(*irreps)])
        blocks.append(nn.R2Conv(in_type, spectral_type, kernel_size=3, padding=1))
        self.synthesis = nn.SequentialModule(*blocks)
        self.extract_rotation = ExtractRotation(self.gspace, channels, irreps)

    def forward(self, z: torch.Tensor, r: Optional[float]=None):
        w = self.mapping(z)
        w = w[..., None, None].tile(1, 1, 4, 4)
        x = nn.GeometricTensor(w, self.input_type)
        x = self.synthesis(x)
        x = self.extract_rotation(x, r)
        return x.tensor


class SteerableDiscriminator(torch.nn.Module):

    def __init__(self, image_size=32, n_channels=3, n_filters=64, maximum_frequency=6):
        super(SteerableDiscriminator, self).__init__()
        self.gspace = gspaces.flipRot2dOnR2(N=-1, maximum_frequency=maximum_frequency)
        in_type = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * n_channels)
        self.input_type = in_type
        main_type = nn.FieldType(self.gspace, [self.gspace.irrep(1, k) for k in range(maximum_frequency + 1)])
        blocks = [nn.MaskModule(in_type, image_size, margin=1)]
        for c, channels in enumerate([n_filters, n_filters, n_filters * 2, n_filters * 2, n_filters * 3, n_filters * 3]):
            out_type = nn.FieldType(self.gspace, channels * [main_type.representation])
            blocks.append(nn.SequentialModule(nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False), nn.NormBatchNorm(out_type), nn.NormNonLinearity(out_type)))
            if c % 2 == 1:
                blocks.append(nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=2))
            in_type = out_type
        blocks.append(nn.NormPool(out_type))
        self.main = nn.SequentialModule(*blocks)
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(blocks[-1].out_type.size, n_filters, kernel_size=3, bias=False), torch.nn.BatchNorm2d(n_filters), torch.nn.ELU(inplace=True), torch.nn.Conv2d(n_filters, n_filters, kernel_size=2, bias=False), torch.nn.Flatten(), torch.nn.BatchNorm1d(n_filters), torch.nn.ELU(inplace=True), torch.nn.Linear(n_filters, n_filters), torch.nn.BatchNorm1d(n_filters), torch.nn.ELU(inplace=True), torch.nn.Linear(n_filters, 1))

    def forward(self, img: torch.Tensor):
        x = nn.GeometricTensor(img, self.input_type)
        x = self.main(x)
        x = x.tensor
        x = self.fc(x)
        return x


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, '%d - %d' % (n_in, n_out)
    mask = torch.ones([n_in, n_out])
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k, i:i + 1] = 0
    return mask


def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = torch.ones([h, w, n_in, n_out])
    mask[:l, :, :, :] = 0
    mask[:, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


def get_conv_weight(filter_shape, stable_init=True, unit_testing=False):
    weight = torch.randn(*filter_shape) * 0.002
    center = (filter_shape[0] - 1) // 2
    if stable_init or unit_testing:
        weight[center, center, :, :] += torch.eye(filter_shape[3])
    return weight


def inverse_conv(z, w, is_upper, dilation):
    batchsize, height, width, n_channels = z.shape
    ksize = w.shape[0]
    kcenter = (ksize - 1) // 2
    x = torch.zeros_like(z)

    def filter2image(j, i, m, k):
        m_ = (m - kcenter) * dilation
        k_ = (k - kcenter) * dilation
        return j + k_, i + m_

    def in_bound(idx, lower, upper):
        return idx >= lower and idx < upper

    def reverse_range(n, reverse):
        if reverse:
            return range(n)
        else:
            return reversed(range(n))
    for b in range(batchsize):
        for j in reverse_range(height, is_upper):
            for i in reverse_range(width, is_upper):
                for c_out in reverse_range(n_channels, not is_upper):
                    for c_in in range(n_channels):
                        for k in range(ksize):
                            for m in range(ksize):
                                if k == kcenter and m == kcenter and c_in == c_out:
                                    continue
                                j_, i_ = filter2image(j, i, m, k)
                                if not in_bound(j_, 0, height):
                                    continue
                                if not in_bound(i_, 0, width):
                                    continue
                                x[b, j, i, c_out] -= w[k, m, c_in, c_out] * x[b, j_, i_, c_in]
                    x[b, j, i, c_out] += z[b, j, i, c_out]
                    x[b, j, i, c_out] /= w[kcenter, kcenter, c_out, c_out]
    return x


class EmergingConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        assert stride == 1
        assert (kernel_size - 1) % 2 == 0
        self.kernel_size = kernel_size
        self.center = (kernel_size - 1) // 2
        self.stride = stride
        self.dilation = dilation
        filter_shape = [kernel_size, kernel_size, in_channels, out_channels]
        self.w1 = torch.nn.Parameter(get_conv_weight(filter_shape).permute(3, 2, 0, 1))
        self.w2 = torch.nn.Parameter(get_conv_weight(filter_shape).permute(3, 2, 0, 1))
        self.b = torch.nn.Parameter(torch.zeros((1, out_channels, 1, 1)))
        self.register_buffer('Lmask', get_conv_square_ar_mask(*filter_shape).permute(3, 2, 0, 1))
        self.register_buffer('Umask', self.Lmask.flip((0, 1, 2, 3)))

    def forward(self, z, reverse=False):
        w1, w2 = self.Lmask * self.w1, self.Umask * self.w2
        if reverse:
            x = z - self.b
            x = inverse_conv(x, w2, is_upper=1, dilation=self.dilation)
            x = inverse_conv(x, w1, is_upper=0, dilation=self.dilation)
            return x
        else:
            w1_s = w1[..., self.center:, self.center:]
            w2_s = w2[..., :-self.center, :-self.center]
            padding = self.center * self.dilation
            z = pad(z, (0, padding, 0, padding))
            z = conv2d(z, w1_s, stride=self.stride, dilation=self.dilation)
            z = pad(z, (padding, 0, padding, 0))
            z = conv2d(z, w2_s, stride=self.stride, dilation=self.dilation)
            z = z + self.b
            return z


class InvertibleLeakyReLU(torch.nn.Module):

    def __init__(self, negative_slope=0.1):
        super(InvertibleLeakyReLU, self).__init__()
        self.negative_slope = torch.nn.Parameter(torch.tensor(negative_slope))

    def forward(self, input, reverse=False):
        if reverse:
            return torch.where(input >= 0.0, input, input * (1 / self.negative_slope))
        else:
            return torch.where(input >= 0.0, input, input * self.negative_slope)


class InvertibleSequential(torch.nn.Module):

    def __init__(self, *layers) ->None:
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input, reverse=False):
        x = input
        for lay in self.layers:
            x = lay(x, reverse=reverse)
        return x


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        output = x.div(keep_prob) * random_tensor
        return output


class GLU(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.0, internal=True, lr_mul=1):
        super().__init__()
        self.internal = internal
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % 2 == 0
        self.fc1 = Butterfly(in_dim, hidden_dim) if lr_mul == 1 else EqualButterfly(in_dim, hidden_dim, lr_mul=lr_mul)
        self.act = nn.Sigmoid()
        self.fc2 = Butterfly(hidden_dim // 2, out_dim) if lr_mul == 1 else EqualButterfly(hidden_dim // 2, out_dim, lr_mul=lr_mul)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        if self.internal:
            x = self.drop(x)
        return x


class HyperMixer(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, drop: float=0.0) ->None:
        super().__init__()
        self.mlp_1 = GLU(in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.mlp_2 = GLU(in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape
        """
        w_1 = self.mlp_1(x)
        w_2 = self.mlp_2(x)
        x = self.drop(self.act(w_1.transpose(1, 2) @ x))
        x = self.drop(w_2 @ x)
        return x


class HyperMixerBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, mlp_ratio: Tuple[float, float]=(0.5, 2.0), drop: float=0.1, drop_path: float=0.1) ->None:
        """
        Constructor method
        :param in_dim (int): Input channel dimension
        :param out_dim (int): Output channel dimension
        :param mlp_ratio (Tuple[int, int]): Ratio of hidden dim. of the hyper mixer layer and MLP. Default = (0.5, 2.0)
        :param drop (float): Dropout rate. Default = 0.1
        :param drop_path (float): Dropout path rate. Default = 0.1
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * in_dim) for x in mlp_ratio]
        self.norm1 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_tokens = HyperMixer(in_dim=in_dim, hidden_dim=tokens_dim, drop=drop)
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_channels = GLU(in_dim=in_dim, hidden_dim=channels_dim, drop=drop)
        self.mlp_reduce = GLU(in_dim=in_dim, out_dim=out_dim, drop=drop)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape [batch size, tokens, channels]
        """
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp_tokens(x))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        x = self.mlp_reduce(x)
        return x


class StyleGLU(nn.Module):

    def __init__(self, w_dim, in_dim, hidden_dim=None, out_dim=None, drop=0.0, internal=True):
        super().__init__()
        self.internal = internal
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        assert hidden_dim % 2 == 0
        self.gate_fc = Butterfly(in_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        self.style_fc = Butterfly(w_dim, hidden_dim // 2)
        self.register_buffer('weight', torch.randn(hidden_dim // 2, out_dim))
        self.register_buffer('bias', torch.randn(out_dim))

    def forward(self, x, w):
        x = self.gate_fc(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        s = self.style_fc(w)
        weight = self.weight[None, :, :] * s[:, :, None]
        x = torch.matmul(x, weight) + self.bias
        if self.internal:
            x = self.drop(x)
        else:
            x = torch.tanh(x)
        return x


class StyleHyperMixer(nn.Module):

    def __init__(self, w_dim: int, in_dim: int, hidden_dim: int, drop: float=0.0) ->None:
        super().__init__()
        self.mlp_1 = StyleGLU(w_dim=w_dim, in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.mlp_2 = StyleGLU(w_dim=w_dim, in_dim=in_dim, out_dim=hidden_dim, drop=drop)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, w: torch.Tensor) ->torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape
        """
        w_1 = self.mlp_1(x, w)
        w_2 = self.mlp_2(x, w)
        x = self.drop(self.act(w_1.transpose(1, 2) @ x))
        x = self.drop(w_2 @ x)
        return x


class StyleHyperMixerBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, w_dim: int, mlp_ratio: Tuple[float, float]=(0.5, 2.0), drop: float=0.1, drop_path: float=0.1) ->None:
        """
        Constructor method
        :param in_dim (int): Input channel dimension
        :param out_dim (int): Output channel dimension
        :param mlp_ratio (Tuple[int, int]): Ratio of hidden dim. of the hyper mixer layer and MLP. Default = (0.5, 2.0)
        :param drop (float): Dropout rate. Default = 0.1
        :param drop_path (float): Dropout path rate. Default = 0.1
        """
        super().__init__()
        tokens_dim, channels_dim = [int(x * in_dim) for x in mlp_ratio]
        self.norm1 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_tokens = StyleHyperMixer(w_dim=w_dim, in_dim=in_dim, hidden_dim=tokens_dim, drop=drop)
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm2 = nn.LayerNorm(in_dim, eps=1e-06)
        self.mlp_style = StyleGLU(w_dim=w_dim, in_dim=in_dim, hidden_dim=channels_dim, drop=drop)
        self.mlp_reduce = GLU(in_dim=in_dim, out_dim=out_dim, drop=drop)

    def forward(self, x: torch.Tensor, w: torch.Tensor) ->torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :param w (torch.Tensor): Latent style vector [batch size, 2, latent_dim]
        :return (torch.Tensor): Output tensor of the shape [batch size, tokens, channels]
        """
        x = self.norm1(x)
        x = x + self.drop_path(self.mlp_tokens(x, w[:, 0]))
        x = x + self.drop_path(self.mlp_style(self.norm2(x), w[:, 1]))
        x = self.mlp_reduce(x)
        return x


class Stack(nn.Module):

    def __init__(self, n: int) ->None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x[:, None, :].repeat(1, self.n, 1)


def tensor2bytes(tensor: torch.Tensor, value_range: Tuple[int, int]=(0, 1)) ->np.ndarray:
    """Converts a PyTorch [1,C,H,W] tensor to bytes (e.g. for passing to FFMPEG)

    Args:
        tensor (torch.Tensor): Image tensor to convert to UINT8 bytes

    Returns:
        np.ndarray
    """
    mn, mx = value_range
    return tensor.squeeze(0).permute(1, 2, 0).clamp(mn, mx).sub(mn).div(mx - mn).mul(255).round().byte().detach().cpu().numpy().tobytes()


class VideoWriter:

    def __init__(self, *args, **kwargs):
        self.Q = Queue()
        self.thread = WriteWorker(self.Q, *args, **kwargs)

    def write(self, tensor):
        if self.Q.qsize() > 32:
            tensor = tensor.cpu()
        self.Q.put(tensor)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, type, value, traceback):
        count = 0
        while not self.Q.qsize() == 0:
            sleep(1)
            count += 1
            if count > 30:
                break
        self.thread.stop()


def upscale(video_file, model_name, device, out_dir):
    vr = VideoReader(video_file)
    fps = vr.get_avg_fps()
    h, w, _ = vr[0].shape
    out_file = f'{out_dir}/{Path(video_file).stem}_{model_name}.mp4'
    with VideoWriter(output_file=out_file, output_size=(4 * w, 4 * h), fps=fps) as video:
        frames = (vr[i].permute(2, 0, 1).unsqueeze(0).div(255) for i in range(len(vr)))
        for frame in tqdm(upscale_images(frames, model_name, device), total=len(vr)):
            video.write(frame)
    return VideoReader(out_file)


class StyleHyperMixerFlyGenerator(nn.Module):

    def __init__(self, z_dim=512, w_dim=512, n_map=8, image_size=1024, img_channels=3, ngf=512, drop=0.1, lr_map=0.01, **kwargs) ->None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.n_map = n_map
        self.image_size = image_size
        self.img_channels = img_channels
        self.ngf = ngf
        self.mapping = nn.Sequential(*([PixelNorm(), GLU(in_dim=z_dim, out_dim=w_dim, drop=0, lr_mul=lr_map)] + [GLU(in_dim=w_dim, out_dim=w_dim, drop=0, lr_mul=lr_map) for _ in range(n_map - 1)] + [Stack(3 * int(np.log2(image_size) - 1))]))
        self.register_buffer('constant_input', torch.randn((1, ngf, 4, 4)))
        block_resolutions = 2 ** np.arange(2, np.log2(image_size) + 1).astype(int)
        log_n_channels = np.arange(np.log2(ngf), 4, -1)
        n_channels = np.concatenate((ngf * np.ones(len(block_resolutions) - len(log_n_channels) + 1), 2 ** log_n_channels)).astype(int)
        self.synthesis = nn.ModuleList([StyleHyperMixerBlock(in_dim, out_dim, w_dim, drop=drop, drop_path=drop) for in_dim, out_dim in zip(n_channels[:-1], n_channels[1:])])
        self.to_rgbs = nn.ModuleList([StyleGLU(w_dim, out_dim, out_dim, img_channels, drop, internal=False) for out_dim in n_channels[1:]])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('StyleHyperMixerFlyGenerator')
        parser.add_argument('--z_dim', type=int, default=512, help='Size of the input latent vector')
        parser.add_argument('--w_dim', type=int, default=512, help='Size of mapped latent vector')
        parser.add_argument('--n_map', type=int, default=8, help='Number of mapping layers')
        parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
        parser.add_argument('--ngf', type=int, default=512, help='Base number of filters')
        parser.add_argument('--drop', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--lr_map', type=float, default=0.01, help='Equalized learning rate for mapping network')
        return parent_parser

    def forward(self, z: torch.Tensor) ->torch.Tensor:
        """
        Forward pass
        :param z (torch.Tensor): Latent style vector [batch size, latent_dim]
        :return (torch.Tensor): Output tensor of the shape [batch size, channels, resolution, resolution]
        """
        ws = self.mapping(z)
        x = self.constant_input.repeat((len(ws), 1, 1, 1))
        img = None
        for i, (block, to_rgb) in enumerate(zip(self.synthesis, self.to_rgbs)):
            w = ws[:, 3 * i:3 * (i + 1)]
            w12 = w[:, :2]
            w3 = w[:, -1]
            B, _, H, W = x.shape
            x = x.flatten(2).transpose(2, 1)
            x = block(x, w12)
            y = to_rgb(x, w3).transpose(2, 1).reshape(B, -1, H, W)
            img = img + y if img is not None else y
            if W != self.image_size:
                x = x.transpose(2, 1).reshape(B, -1, H, W)
                x_img = torch.cat((x, img), dim=1)
                x_img = upscale(x_img)
                x, img = x_img[:, :-3], x_img[:, -3:]
        return img


def downscale(x):
    return interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)


class HyperMixerFlyDiscriminator(nn.Module):
    override_args = {'logits': True}

    def __init__(self, image_size=1024, img_channels=3, ndf=512, drop=0.1, **kwargs) ->None:
        super().__init__()
        self.image_size = image_size
        self.img_channels = img_channels
        self.ndf = ndf
        block_resolutions = 2 ** np.arange(np.log2(image_size), 1, -1).astype(int)
        log_n_channels = np.arange(4, np.log2(ndf))
        n_channels = np.concatenate((2 ** log_n_channels, ndf * np.ones(len(block_resolutions) - len(log_n_channels)))).astype(int)
        self.encode = nn.ModuleList([nn.Sequential(Butterfly(img_channels, n_channels[0]), nn.GELU())] + [HyperMixerBlock(in_dim, out_dim, drop=drop, drop_path=drop) for in_dim, out_dim in zip(n_channels[:-1], n_channels[1:])])
        self.predict = GLU(n_channels[-1] * 4 * 4, ndf, 1, drop=drop, internal=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HyperMixerFlyDiscriminator')
        parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
        parser.add_argument('--ndf', type=int, default=512, help='Base number of filters')
        parser.add_argument('--drop', type=float, default=0.1, help='Dropout rate')
        return parent_parser

    def forward(self, img: torch.Tensor) ->torch.Tensor:
        x = img
        for block in self.encode:
            B, _, H, W = x.shape
            x = x.flatten(2).transpose(2, 1)
            x = block(x)
            if W != 4:
                x = x.transpose(2, 1).reshape(B, -1, H, W)
                x = downscale(x)
        logits = self.predict(x.flatten(1))
        return logits


def activate(x: Tensor, act: str, alpha: float):
    if act == 'relu':
        return torch.nn.functional.relu(x)
    elif act == 'lrelu':
        return torch.nn.functional.leaky_relu(x, alpha)
    elif act == 'tanh':
        return torch.tanh(x)
    elif act == 'sigmoid':
        return torch.sigmoid(x)
    elif act == 'elu':
        return torch.nn.functional.elu(x)
    elif act == 'selu':
        return torch.nn.functional.selu(x)
    elif act == 'softplus':
        return torch.nn.functional.softplus(x)
    elif act == 'swish':
        return torch.sigmoid(x) * x
    else:
        return x


def get_activation_defaults(activation: str):
    if activation == 'relu':
        return torch.tensor(0.0), torch.tensor(sqrt(2))
    elif activation == 'lrelu':
        return torch.tensor(0.2), torch.tensor(sqrt(2))
    elif activation == 'tanh':
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == 'sigmoid':
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == 'elu':
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == 'selu':
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == 'softplus':
        return torch.tensor(0.0), torch.tensor(1.0)
    elif activation == 'swish':
        return torch.tensor(0.0), torch.tensor(sqrt(2))
    else:
        return torch.tensor(0.0), torch.tensor(1.0)


def bias_act(x: Tensor, b: Optional[Tensor]=None, act: str='linear', alpha: Optional[Tensor]=None, gain: Optional[Tensor]=None, clamp: Optional[Tensor]=None):
    def_alpha, def_gain = get_activation_defaults(act)
    alpha = alpha if alpha is not None else def_alpha
    gain = gain if gain is not None else def_gain
    clamp = clamp if clamp is not None else torch.tensor(-1.0)
    if b is not None:
        x = x + b.reshape([(-1 if i == 1 else 1) for i in range(x.ndim)])
    x = activate(x, act, alpha)
    if gain != 1:
        x = x * gain
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)
    return x


class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, activation: str='linear', lr_multiplier: float=1.0, bias_init: float=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: Tensor):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and not self.bias_gain == 1.0:
            b = b * self.bias_gain
        if self.activation == 'linear':
            x = torch.nn.functional.linear(x, w, b)
        else:
            x = bias_act(torch.nn.functional.linear(x, w.T, None), b, act=self.activation)
        return x


def normalize_2nd_moment(x, dim: Tensor=1, eps: float=1e-08):
    return x / ((x * x).mean(dim=dim, keepdim=True) + eps).sqrt()


class MappingNetwork(torch.nn.Module):

    def __init__(self, z_dim: int, c_dim: int, w_dim: int, num_ws: int, num_layers: int=8, embed_features: Optional[int]=None, layer_features: Optional[int]=None, activation: str='lrelu', lr_multiplier: float=0.01, w_avg_beta: float=0.998):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        self.embed = None
        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        fcs = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            fcs.append(layer)
        self.fcs = torch.nn.ModuleList(fcs)
        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z: Tensor, c: Optional[Tensor], truncation_psi: float=1.0, truncation_cutoff: Optional[int]=None):
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z)
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c))
            x = torch.cat([x, y], dim=1) if x is not None else y
        for layer in self.fcs:
            x = layer(x)
        if self.num_ws is not None:
            x = x.unsqueeze(1)
            x = torch.cat([x for _ in range(self.num_ws)], dim=1)
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


activation_funcs = {'linear': dict(func=lambda x, **_: x, def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=False), 'relu': dict(func=lambda x, **_: torch.nn.functional.relu(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(sqrt(2)), has_2nd_grad=False), 'lrelu': dict(func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha), def_alpha=torch.tensor(0.2), def_gain=torch.tensor(sqrt(2)), has_2nd_grad=False), 'tanh': dict(func=lambda x, **_: torch.tanh(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=True), 'sigmoid': dict(func=lambda x, **_: torch.sigmoid(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=True), 'elu': dict(func=lambda x, **_: torch.nn.functional.elu(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=True), 'selu': dict(func=lambda x, **_: torch.nn.functional.selu(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=True), 'softplus': dict(func=lambda x, **_: torch.nn.functional.softplus(x), def_alpha=torch.tensor(0.0), def_gain=torch.tensor(1.0), has_2nd_grad=True), 'swish': dict(func=lambda x, **_: torch.sigmoid(x) * x, def_alpha=torch.tensor(0.0), def_gain=torch.tensor(sqrt(2)), has_2nd_grad=True)}


def _get_filter_size(f: Optional[Tensor]):
    if f is None:
        return torch.ones(2)
    return f.shape[-1], f.shape[0]


def upfirdn2d(x: Tensor, f: Optional[Tensor], up: Tensor=torch.tensor(1), down: Tensor=torch.tensor(1), padding: Tensor=torch.zeros(4), gain: Tensor=torch.tensor(1)):
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = up.repeat(2)
    downx, downy = down.repeat(2)
    padx0, padx1, pady0, pady1 = padding
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0):x.shape[2] - max(-pady1, 0), max(-padx0, 0):x.shape[3] - max(-padx1, 0)]
    f = f * gain ** (f.ndim / 2)
    f = f[None, None].repeat(num_channels, 1, 1, 1)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = torch.nn.functional.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def conv2d_resample(x: Tensor, w: Tensor, f: Optional[Tensor]=None, up: Tensor=torch.tensor(1), down: Tensor=torch.tensor(1), padding: Tensor=torch.tensor(0), groups: Tensor=torch.tensor(1)):
    out_channels, in_channels_per_group, kh, kw = w.shape
    fw, fh = _get_filter_size(f)
    px0, px1, py0, py1 = padding.repeat(4)
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2
    if up > 1:
        if groups == 1:
            w = w.permute(1, 0, 2, 3, 4)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.permute(0, 2, 1, 3, 4)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = torch.max(torch.min(-px0, -px1), 0)
        pyt = torch.max(torch.min(-py0, -py1), 0)
        x = torch.nn.functional.conv_transpose2d(x, w, stride=up, padding=(pyt, pxt), groups=groups)
        x = upfirdn2d(x=x, f=f, padding=(px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt), gain=up ** 2)
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=down)
        return x
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return torch.nn.functional.conv2d(x, w, padding=(py0, px0), groups=groups)
    assert False, 'Something weird is going on...'
    return x


def setup_filter(f: List[int], device: torch.device=torch.device('cpu'), normalize: bool=True, gain: Tensor=torch.tensor(1), separable: Optional[bool]=None):
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    if f.ndim == 0:
        f = f[None]
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    if normalize:
        f /= f.sum()
    f = f * gain ** (f.ndim / 2)
    f = f
    return f


class Conv2dLayer(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool=True, activation: str='linear', up: int=1, down: int=1, resample_filter: List[int]=[1, 3, 3, 1], conv_clamp: Optional[float]=None, trainable: bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / sqrt(in_channels * kernel_size ** 2)
        self.act_gain = activation_funcs[activation]['def_gain']
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x: Tensor, gain: float=1.0):
        w = self.weight * self.weight_gain
        b = self.bias if self.bias is not None else None
        x = conv2d_resample(x=x, w=w, f=self.resample_filter, up=self.up, down=self.down, padding=self.padding)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


def modulated_conv2d(x: Tensor, weight: Tensor, styles: Tensor, noise: Optional[Tensor]=None, up: Tensor=torch.tensor(1), down: Tensor=torch.tensor(1), padding: Tensor=torch.tensor(0), resample_filter: Optional[Tensor]=None, demodulate: bool=True):
    B, xc, xh, xw = x.shape
    wco, wci, kh, kw = weight.shape
    if x.dtype == torch.float16 and demodulate:
        weight = weight / (torch.amax(torch.abs(weight), dim=(1, 2, 3)).reshape(weight.shape[0], 1, 1, 1) * sqrt(wci * kh * kw))
        styles = styles / torch.max(torch.abs(styles), dim=1).values.unsqueeze(1)
    w = weight.unsqueeze(0) * styles.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if demodulate:
        denom = ((w * w).sum((2, 3, 4)) + 1e-08).sqrt()
        w = w / denom.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x = conv2d_resample(x=x.reshape(1, B * xc, xh, xw), w=w.reshape(B * wco, wci, kh, kw), f=resample_filter, up=up, down=down, padding=padding, groups=B)
    x = x.reshape(B, wco, xh * up, xw * up)
    if noise is not None:
        x = x + noise
    return x


class SynthesisLayer(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int, kernel_size: int=3, up: int=1, use_noise: bool=True, activation: str='lrelu', resample_filter: List[int]=[1, 3, 3, 1], conv_clamp: Optional[float]=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = activation_funcs[activation]['def_gain']
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
        self.noise_adjusted = False
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x: Tensor, w: Tensor, noise_mode: str='const', gain: float=1.0):
        styles = self.affine(w)
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device)
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, padding=self.padding, resample_filter=self.resample_filter)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ToRGBLayer(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, kernel_size: int=1, conv_clamp: Optional[float]=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / sqrt(in_channels * kernel_size ** 2)

    def forward(self, x: Tensor, w: Tensor):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act(x, self.bias, clamp=self.conv_clamp)
        return x


def upsample2d(x: Tensor, f: Tensor, up: Tensor=torch.tensor(2), padding: Tensor=torch.tensor(0), gain: Tensor=torch.tensor(1)):
    upx, upy = up.repeat(2)
    padx0, padx1, pady0, pady1 = padding.repeat(4)
    fw, fh = _get_filter_size(f)
    p = padx0 + (fw + upx - 1) // 2, padx1 + (fw - upx) // 2, pady0 + (fh + upy - 1) // 2, pady1 + (fh - upy) // 2
    return upfirdn2d(x, f, up=up, padding=p, gain=gain * upx * upy)


class SynthesisBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int, img_channels: int, is_last: bool, architecture: str='skip', resample_filter: List[int]=[1, 3, 3, 1], conv_clamp: int=256.0, use_fp16: bool=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.const = None
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
        self.conv0 = None
        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2, resample_filter=resample_filter, conv_clamp=conv_clamp, **layer_kwargs)
            self.num_conv += 1
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, conv_clamp=conv_clamp, **layer_kwargs)
        self.num_conv += 1
        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp)
            self.num_torgb += 1
        self.skip = None
        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2, resample_filter=resample_filter)

    def forward(self, x: Optional[Tensor], img: Optional[Tensor], ws: Tensor, noise_mode: str='const') ->Tuple[Tensor, Tensor]:
        w_idx = 0
        if self.in_channels == 0:
            x = self.const
            x = torch.stack([x for _ in range(ws.shape[0])])
        if self.in_channels == 0:
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=sqrt(0.5))
            x = self.conv0(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=sqrt(0.5))
            w_idx += 1
            x = y + x
        else:
            x = self.conv0(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
            x = self.conv1(x, ws[:, w_idx], noise_mode, gain=1.0)
            w_idx += 1
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, ws[:, w_idx])
            w_idx += 1
            y = y.contiguous()
            img = img + y if img is not None else y
        if img is None:
            img = torch.empty(0)
        return x, img


class SynthesisNetwork(torch.nn.Module):

    def __init__(self, w_dim: int, img_resolution: int, img_channels: int, channel_base: int=32768, channel_max: int=512, num_fp16_res: int=0, **block_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [(2 ** i) for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.num_ws = 0
        bs = []
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            bs.append(block)
        self.bs = torch.nn.ModuleList(bs)

    def forward(self, ws: Tensor, noise_mode: str='const'):
        w_idx = 0
        x = img = None
        for block in self.bs:
            block_ws = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
            x, img = block(x, img, block_ws, noise_mode)
            w_idx += block.num_conv
        return img


class Generator(torch.nn.Module):

    def __init__(self, z_dim: int, c_dim: int, w_dim: int, img_resolution: int, img_channels: int, mapping_kwargs={}, **synthesis_kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z: Tensor, c: Optional[Tensor]=None, truncation_psi: float=1.0, truncation_cutoff: Optional[float]=None, noise_mode='const'):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, noise_mode)
        return img


img_channels = 3


ndf = 64


class MauaMapper(torch.nn.Module):

    def forward(self):
        raise NotImplementedError()


class MauaSynthesizer(torch.nn.Module):
    _hook_handles = []

    def forward(self):
        raise NotImplementedError()

    def change_output_resolution(self):
        raise NotImplementedError()

    def refresh_model_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


def load_nvidia(path, for_inference=None):
    with dnnlib.util.open_url(path) as f:
        G_persistence = legacy.load_network_pkl(f)['G_ema']
    try:
        G = stylegan3.Generator(G_persistence.mapping.z_dim, G_persistence.mapping.c_dim, G_persistence.mapping.w_dim, G_persistence.img_resolution, G_persistence.img_channels, mapping_kwargs=dict(num_layers=G_persistence.mapping.num_layers))
        G.load_state_dict(G_persistence.state_dict())
        try_stylegan2 = False
    except:
        try_stylegan2 = True
    if try_stylegan2:
        try:
            G = (stylegan2_inference if for_inference else stylegan2_train).Generator(G_persistence.mapping.z_dim, G_persistence.mapping.c_dim, G_persistence.mapping.w_dim, G_persistence.img_resolution, G_persistence.img_channels, mapping_kwargs=dict(num_layers=G_persistence.mapping.num_layers))
            G.load_state_dict(G_persistence.state_dict())
        except:
            G = deepcopy(G_persistence)
    del G_persistence
    return G


def load_nvidia_pt(path, z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3, map_layers=8, for_inference=False):
    state_dict = torch.load(path)['G_ema']
    try:
        G = stylegan3.Generator(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=dict(num_layers=map_layers))
        G.load_state_dict(state_dict)
        try_stylegan2 = False
    except:
        try_stylegan2 = True
    if try_stylegan2:
        G = (stylegan2_inference if for_inference else stylegan2_train).Generator(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=dict(num_layers=map_layers))
        G.load_state_dict(state_dict)
    del state_dict
    return G


def load_rosinality2ada(path, blur_scale=4.0, for_inference=False):
    state_dict = torch.load(path)
    state_ros = state_dict['g_ema']
    state_nv = {}
    nv_key = 'bs.0' if for_inference else 'b4'
    if tuple(state_ros[f'input.input'].shape) != (1,):
        state_nv[f'synthesis.{nv_key}.const'] = state_ros[f'input.input'].squeeze(0)
        use_const = True
    else:
        state_nv[f'synthesis.{nv_key}.const.affine.weight'] = state_ros[f'input.linear.weight'].squeeze(0)
        state_nv[f'synthesis.{nv_key}.const.affine.bias'] = state_ros[f'input.linear.bias'].squeeze(0)
        use_const = False
    state_nv[f'synthesis.{nv_key}.conv1.noise_const'] = state_ros[f'noises.noise_0'].squeeze(0).squeeze(0)
    state_nv[f'synthesis.{nv_key}.conv1.weight'] = state_ros[f'conv1.conv.weight'].squeeze(0)
    state_nv[f'synthesis.{nv_key}.conv1.bias'] = state_ros[f'conv1.activate.bias']
    state_nv[f'synthesis.{nv_key}.conv1.affine.weight'] = state_ros[f'conv1.conv.modulation.weight']
    state_nv[f'synthesis.{nv_key}.conv1.affine.bias'] = state_ros[f'conv1.conv.modulation.bias']
    if not for_inference:
        state_nv[f'synthesis.{nv_key}.conv1.noise_strength'] = state_ros[f'conv1.noise.weight'].squeeze(0)
    state_nv[f'synthesis.{nv_key}.torgb.weight'] = state_ros[f'to_rgb1.conv.weight'].squeeze(0)
    state_nv[f'synthesis.{nv_key}.torgb.bias'] = state_ros[f'to_rgb1.bias'].squeeze(-1).squeeze(-1).squeeze(0)
    state_nv[f'synthesis.{nv_key}.torgb.affine.weight'] = state_ros[f'to_rgb1.conv.modulation.weight']
    state_nv[f'synthesis.{nv_key}.torgb.affine.bias'] = state_ros[f'to_rgb1.conv.modulation.bias']
    state_nv[f'synthesis.{nv_key}.resample_filter'] = state_ros[f'convs.0.conv.blur.kernel'] / blur_scale
    state_nv[f'synthesis.{nv_key}.conv1.resample_filter'] = state_ros[f'convs.0.conv.blur.kernel'] / blur_scale
    max_res, num_map = 4, 1
    for key, val in state_ros.items():
        if key.startswith('style'):
            _, num, weight_or_bias = key.split('.')
            nv_key = f'mapping.fcs.{int(num) - 1}.{weight_or_bias}' if for_inference else f'mapping.fc{int(num) - 1}.{weight_or_bias}'
            state_nv[nv_key] = val
            if int(num) > num_map:
                num_map = int(num)
        if key.startswith('noises'):
            n = int(key.split('_')[1])
            r = 2 ** (3 + (n - 1) // 2)
            nv_block = f'synthesis.bs.{(n - 1) // 2 + 1}' if for_inference else f'synthesis.b{r}'
            state_nv[f'{nv_block}.conv{(n - 1) % 2}.noise_const'] = state_ros[f'noises.noise_{n}'].squeeze(0).squeeze(0)
        if key.startswith('convs'):
            n = int(key.split('.')[1])
            r = 2 ** (3 + n // 2)
            nv_block = f'synthesis.bs.{n // 2 + 1}' if for_inference else f'synthesis.b{r}'
            ros_name = '.'.join(key.split('.')[2:])
            if ros_name == 'conv.weight':
                state_nv[f'{nv_block}.conv{n % 2}.weight'] = state_ros[f'convs.{n}.conv.weight'].squeeze(0)
            elif ros_name == 'activate.bias':
                state_nv[f'{nv_block}.conv{n % 2}.bias'] = state_ros[f'convs.{n}.activate.bias']
            elif ros_name == 'conv.modulation.weight':
                state_nv[f'{nv_block}.conv{n % 2}.affine.weight'] = state_ros[f'convs.{n}.conv.modulation.weight']
            elif ros_name == 'conv.modulation.bias':
                state_nv[f'{nv_block}.conv{n % 2}.affine.bias'] = state_ros[f'convs.{n}.conv.modulation.bias']
            elif ros_name == 'noise.weight' and not for_inference:
                state_nv[f'{nv_block}.conv{n % 2}.noise_strength'] = state_ros[f'convs.{n}.noise.weight'].squeeze(0)
            elif ros_name == 'conv.blur.kernel':
                state_nv[f'{nv_block}.conv0.resample_filter'] = state_ros[f'convs.{n}.conv.blur.kernel'] / blur_scale
                state_nv[f'{nv_block}.conv1.resample_filter'] = state_ros[f'convs.{n}.conv.blur.kernel'] / blur_scale
            else:
                raise Exception(f'Key {key} not recognized!')
            if r > max_res:
                max_res = r
        if key.startswith('to_rgbs'):
            n = int(key.split('.')[1])
            r = 2 ** (3 + n)
            nv_block = f'synthesis.bs.{n + 1}' if for_inference else f'synthesis.b{r}'
            ros_name = '.'.join(key.split('.')[2:])
            if ros_name == 'conv.weight':
                state_nv[f'{nv_block}.torgb.weight'] = state_ros[f'to_rgbs.{n}.conv.weight'].squeeze(0)
            elif ros_name == 'bias':
                state_nv[f'{nv_block}.torgb.bias'] = state_ros[f'to_rgbs.{n}.bias'].squeeze(-1).squeeze(-1).squeeze(0)
            elif ros_name == 'conv.modulation.weight':
                state_nv[f'{nv_block}.torgb.affine.weight'] = state_ros[f'to_rgbs.{n}.conv.modulation.weight']
            elif ros_name == 'conv.modulation.bias':
                state_nv[f'{nv_block}.torgb.affine.bias'] = state_ros[f'to_rgbs.{n}.conv.modulation.bias']
            elif ros_name == 'upsample.kernel':
                state_nv[f'{nv_block}.resample_filter'] = state_ros[f'to_rgbs.{n}.upsample.kernel'] / blur_scale
            else:
                raise Exception(f'Key {key} not recognized!')
    if 'latent_avg' in state_dict:
        state_nv['mapping.w_avg'] = state_dict['latent_avg']
    else:
        state_nv['mapping.w_avg'] = torch.zeros(512)
    z_dim = 512
    w_dim = 512
    c_dim = 0
    chnls = 3
    G = (stylegan2_inference if for_inference else stylegan2_train).Generator(z_dim, c_dim, w_dim, max_res, chnls, mapping_kwargs=dict(num_layers=num_map))
    G.load_state_dict(state_nv)
    return G


def load_network(path, for_inference=False):
    errors = {}
    for name, loader in [('NVIDIA StyleGAN3 loader', load_nvidia), ('NVIDIA non-persistence loader', load_nvidia_pt), ('Rosinality StyleGAN2 to ADA-PT converter', load_rosinality2ada), ('Rosinality StyleGAN2 to Inference converter', partial(load_rosinality2ada, for_inference=True))]:
        try:
            return loader(path, for_inference=for_inference)
        except:
            errors[name] = traceback.format_exc()
    error_str = '\n'.join([f'\n{k}:\n{e}\n' for k, e in errors.items()])
    raise Exception(f'Error loading checkpoint! None of the converters succeeded:\n{error_str}')


class StyleGANMapper(MauaMapper):
    MapperClsFn = lambda : None

    def __init__(self, model_file: str, inference: bool) ->None:
        super().__init__()
        if model_file is None or model_file == 'None':
            self.G_map = self.__class__.MapperClsFn(inference)(z_dim=512, c_dim=0, w_dim=512, num_ws=18)
        else:
            self.G_map = load_network(model_file, inference).mapping
        self.z_dim, self.c_dim = self.G_map.z_dim, self.G_map.c_dim
        self.modulation_targets = {'latent_z': (self.z_dim,), 'truncation': (1,)}
        if self.c_dim > 0:
            self.modulation_targets['class_conditioning'] = self.c_dim,

    def forward(self, latent_z: Tensor, class_conditioning: Optional[Tensor]=None, truncation: float=1.0):
        return self.G_map.forward(latent_z, class_conditioning, truncation_psi=truncation)


class StyleGANSynthesizer(MauaSynthesizer):
    pass


class StyleGAN2Mapper(StyleGANMapper):
    MapperClsFn = lambda inference: (stylegan2_inference if inference else stylegan2_train).MappingNetwork


def get_hook(G_synth, layer, size, strategy):
    size = np.flip(size)
    if strategy == 'stretch':

        def hook(module, input, output):
            return interpolate(output, tuple(size), mode='bicubic', align_corners=False)
        return hook
    elif strategy == 'pad-zero':
        original_size = getattr(G_synth, G_synth.layer_names[max(layer - 1, 0)]).out_size
        pad_h, pad_w = (size - original_size).astype(int) // 2
        padding = pad_w, pad_w, pad_h, pad_h

        def hook(module, input, output):
            return pad(output, padding, mode='constant', value=0)
        return hook
    else:
        raise Exception(f'Resize strategy not found: {strategy}')


class StyleGAN2Synthesizer(StyleGANSynthesizer):
    __constants__ = ['w_dim', 'num_ws', 'layer_names']

    def __init__(self, model_file: str, inference: bool, output_size: Optional[Tuple[int, int]], strategy: str, layer: int) ->None:
        super().__init__()
        if model_file is None or model_file == 'None':
            self.G_synth = (stylegan2_inference if inference else stylegan2_train).SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
        else:
            self.G_synth: (stylegan2_inference if inference else stylegan2_train).SynthesisNetwork = load_network(model_file, inference).synthesis
        if not hasattr(self.G_synth, 'bs'):
            self.G_synth.bs = []
            for res in self.G_synth.block_resolutions:
                self.G_synth.bs.append(getattr(self.G_synth, f'b{res}'))
        if output_size is None:
            output_size = self.G_synth.img_resolution, self.G_synth.img_resolution
        self.w_dim, self.num_ws = self.G_synth.w_dim, self.G_synth.num_ws
        self.layer_names = [f'bs.{c // 2}.conv{1 if block_size == 4 else c % 2}' for c, block_size in enumerate(sorted(self.G_synth.block_resolutions * 2))]
        self.modulation_targets = {'latent_w': (self.w_dim,), 'latent_w_plus': (self.num_ws, self.w_dim), 'translation': (2,), 'rotation': (1,)}
        self.translate_hook, self.rotate_hook, self.zoom_hook = None, None, None
        self.change_output_resolution(output_size, strategy, layer)

    def forward(self, latents: Tensor, translation: Optional[Tensor]=None, translation_layer: int=7, zoom: Optional[Tensor]=None, zoom_layer: int=7, zoom_center: Optional[int]=None, rotation: Optional[Tensor]=None, rotation_layer: int=7, rotation_center: Optional[int]=None, **noise) ->Tensor:
        if translation is not None:
            self.apply_translation(translation_layer, translation)
        if zoom is not None:
            self.apply_zoom(zoom_layer, zoom, zoom_center)
        if rotation is not None:
            self.apply_rotation(rotation_layer, rotation, rotation_center)
        if noise is not None:
            noises, l = list(noise.values()), 0
            for block in self.G_synth.bs:
                if l >= len(noises):
                    continue
                for c in (([block.conv0] if hasattr(block, 'conv0') else []) + [block.conv1]):
                    noise_l = noises[l]
                    if (noise_l.shape[-2], noise_l.shape[-1]) != (c.noise_const.shape[-2], c.noise_const.shape[-1]):
                        warnings.warn(f'Supplied noise for SynthesisLayer {l} has shape {noise_l.shape} while the expected shape is {c.noise_const.shape}. Resizing the supplied noise to match...')
                        h, w = c.noise_const.shape[-2], c.noise_const.shape[-1]
                        noise_l = torch.nn.functional.interpolate(noise_l, (h, w), mode='bicubic', align_corners=False)
                    setattr(c, 'noise_const', noise_l)
                    l += 1
        return self.G_synth.forward(latents, noise_mode='const')

    def change_output_resolution(self, output_size: Tuple[int, int], strategy: str, layer: int):
        self.refresh_model_hooks()
        if output_size != (self.G_synth.img_resolution, self.G_synth.img_resolution):
            _, block, conv = self.layer_names[layer].split('.')
            synth_block = self.G_synth.bs[int(block)]
            synth_layer = getattr(synth_block, conv)
            torgb_layer = getattr(synth_block, 'torgb')
            layer_size = synth_layer.resolution
            lay_mult = self.G_synth.img_resolution // layer_size
            unrounded_size = np.array(output_size) / lay_mult
            target_size = np.round(unrounded_size).astype(int)
            if sum(abs(unrounded_size - target_size)) > 1e-10:
                warnings.warn(f'Layer {layer} resizes to multiples of {lay_mult}. --output-size rounded to {lay_mult * target_size}')
            use_pre_hook = layer == 0
            feat_hook, prev_img_hook, rgb_hook = get_hook(layer_size, target_size, strategy, pre=use_pre_hook)
            self._hook_handles.append(synth_layer.register_forward_pre_hook(feat_hook) if layer == 0 else synth_layer.register_forward_hook(feat_hook))
            if not use_pre_hook:
                self._hook_handles.append(synth_block.register_forward_hook(prev_img_hook))
                self._hook_handles.append(torgb_layer.register_forward_hook(rgb_hook))
            for l in range(layer + 1, len(self.layer_names)):
                _, b, c = self.layer_names[l].split('.')
                noise_layer = getattr(self.G_synth.bs[int(b)], c)

                def noise_adjust(mod, input: Tuple[Tensor, Tensor, str, bool, float]) ->None:
                    if not hasattr(mod, 'noise_adjusted') or not mod.noise_adjusted:
                        _, _, h, w = input[0].shape
                        dev, dtype = mod.noise_const.device, mod.noise_const.dtype
                        mod.noise_const = torch.randn((1, 1, h * mod.up, w * mod.up), device=dev, dtype=dtype)
                        mod.noise_adjusted = True
                self._hook_handles.append(noise_layer.register_forward_pre_hook(noise_adjust))
            self.forward(torch.randn((1, 18, 512)))
        self.output_size = output_size

    def apply_translation(self, layer, translation):
        _, block, conv = self.layer_names[layer].split('.')
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def translate_hook(module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]) ->Tuple[Tensor]:
            _, _, h, w = output.shape
            output = kT.translate(output, translation * torch.tensor([[h, w]], device=translation.device), padding_mode='reflection')
            return output
        if self.translate_hook:
            self.translate_hook.remove()
        self.translate_hook = synth_layer.register_forward_hook(translate_hook)

    def apply_rotation(self, layer, angle, center):
        _, block, conv = self.layer_names[layer].split('.')
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def rotation_hook(module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]) ->Tuple[Tensor]:
            output = kT.rotate(output, angle.squeeze(), center, padding_mode='reflection')
            return output
        if self.rotate_hook:
            self.rotate_hook.remove()
        self.rotate_hook = synth_layer.register_forward_hook(rotation_hook)

    def apply_zoom(self, layer, zoom, center):
        _, block, conv = self.layer_names[layer].split('.')
        synth_layer = getattr(self.G_synth.bs[int(block)], conv)

        def zoom_hook(module, input: Tuple[Tensor, Tensor, str, bool, float], output: Tuple[Tensor]) ->Tuple[Tensor]:
            output = kT.scale(output, zoom.squeeze(), center, padding_mode='reflection')
            return output
        if self.zoom_hook:
            self.zoom_hook.remove()
        self.zoom_hook = synth_layer.register_forward_hook(zoom_hook)

    def make_noise_pyramid(self, noise, layer_limit=8):
        noises = {}
        for l, layer in enumerate(self.layer_names[1:]):
            if l > layer_limit:
                continue
            _, block, conv = layer.split('.')
            synth_layer = getattr(self.G_synth.bs[int(block)], conv)
            h, w = synth_layer.noise_const.shape[-2], synth_layer.noise_const.shape[-1]
            try:
                noises[f'noise{l}'] = torch.nn.functional.interpolate(noise, (h, w), mode='bicubic', align_corners=False).cpu()
            except RuntimeError:
                noises[f'noise{l}'] = torch.nn.functional.interpolate(noise.cpu(), (h, w), mode='bicubic', align_corners=False)
            noises[f'noise{l}'] /= noises[f'noise{l}'].std((1, 2, 3), keepdim=True)
        return noises


class StyleGAN3Mapper(StyleGANMapper):
    MapperClsFn = lambda inference: stylegan3.MappingNetwork


layer_multipliers = {(1024): {(0): 64, (1): 64, (2): 64, (3): 32, (4): 32, (5): 16, (6): 8, (7): 8, (8): 4, (9): 4, (10): 2, (11): 1, (12): 1, (13): 1, (14): 1, (15): 1}, (512): {(0): 32, (1): 32, (2): 32, (3): 16, (4): 16, (5): 8, (6): 8, (7): 4, (8): 4, (9): 2, (10): 2, (11): 1, (12): 1, (13): 1, (14): 1}, (256): {(0): 16, (1): 16, (2): 16, (3): 16, (4): 8, (5): 8, (6): 4, (7): 4, (8): 2, (9): 2, (10): 2, (11): 1, (12): 1, (13): 1, (14): 1}}


def make_transform_mat(translate: Tuple[float, float], angle: float) ->torch.Tensor:
    s = np.sin(angle.squeeze().cpu() / 360.0 * np.pi * 2)
    c = np.cos(angle.squeeze().cpu() / 360.0 * np.pi * 2)
    m = np.array([[c, s, translate.squeeze().cpu()[0]], [-s, c, translate.squeeze().cpu()[1]], [0, 0, 0]])
    try:
        m = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        warnings.warn(f'Singular transform matrix, continuing with pseudo-inverse of transform matrix which might not give expected results! (If you want no translation or rotation, set them to None rather than 0)')
        m = np.linalg.pinv(m)
    return torch.from_numpy(m)


class StyleGAN3Synthesizer(StyleGANSynthesizer):

    def __init__(self, model_file: str, inference: bool, output_size: Optional[Tuple[int, int]], strategy: str, layer: int) ->None:
        super().__init__()
        if model_file is None or model_file == 'None':
            self.G_synth = stylegan3.SynthesisNetwork(w_dim=512, img_resolution=1024, img_channels=3)
        else:
            self.G_synth: stylegan3.SynthesisNetwork = load_network(model_file).synthesis
        if output_size is None:
            output_size = self.G_synth.img_resolution, self.G_synth.img_resolution
        self.w_dim, self.num_ws = self.G_synth.w_dim, self.G_synth.num_ws
        self.modulation_targets = {'latent_w': (self.w_dim,), 'latent_w_plus': (self.num_ws, self.w_dim), 'translation': (2,), 'rotation': (1,)}
        self.change_output_resolution(output_size, strategy, layer)

    def forward(self, latents: torch.Tensor=None, translation: torch.Tensor=None, rotation: torch.Tensor=None) ->torch.Tensor:
        if translation == 0 and rotation == 0:
            self.G_synth.input.affine.bias.data.add_(self.avg_shift)
            self.G_synth.input.affine.weight.data.zero_()
        elif not (translation is None or rotation is None):
            self.G_synth.input.transform.copy_(make_transform_mat(translation, rotation))
        return self.G_synth.forward(latents)

    def change_output_resolution(self, output_size: Tuple[int, int], strategy: str, layer: int):
        self.refresh_model_hooks()
        if output_size != (self.G_synth.img_resolution, self.G_synth.img_resolution):
            lay_mult = layer_multipliers[self.G_synth.img_resolution][layer]
            unrounded_size = np.array(output_size) / lay_mult + 20
            size = np.round(unrounded_size).astype(int)
            if sum(abs(unrounded_size - size)) > 1e-10:
                warnings.warn(f'Layer {layer} resizes to multiples of {lay_mult}. --output-size rounded to {lay_mult * (size - 20)}')
            synth_layer = getattr(self.G_synth, 'input' if layer == 0 else self.G_synth.layer_names[layer - 1])
            hook = synth_layer.register_forward_hook(get_hook(self.G_synth, layer, size, strategy))
            self._hook_handles.append(hook)
        self.output_size = output_size


class Noise(torch.nn.Module):

    def __init__(self, length, size):
        super().__init__()
        self.length = length
        self.size = size


class Blend(Noise):

    def __init__(self, rng, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer('noise', torch.randn((2, modulator.shape[1], size[0], size[1]), generator=rng, device=rng.device))
        self.register_buffer('modulator', modulator)

    def forward(self, i, b):
        mod = self.modulator[i:i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum('MHW,BM->BHW', self.noise[0], mod)
        right = torch.einsum('MHW,BM->BHW', self.noise[1], 1 - mod)
        return left + right


class Multiply(Noise):

    def __init__(self, rng, length, size, modulator):
        super().__init__(length, size)
        self.register_buffer('noise', torch.randn((modulator.shape[1], size[0], size[1]), generator=rng, device=rng.device))
        self.register_buffer('modulator', modulator)

    def forward(self, i, b):
        mod = self.modulator[i:i + b]
        mod = mod.reshape(len(mod), -1)
        left = torch.einsum('MHW,BM->BHW', self.noise, mod)
        return left


class Loop(Noise):

    def __init__(self, rng, length, size, n_loops=1, sigma=5):
        super().__init__(length, size)
        self.sigma = sigma
        self.register_buffer('noise', torch.randn((3, size[0], size[1]), generator=rng, device=rng.device))
        self.register_buffer('idx', torch.linspace(0, n_loops * 2 * torch.pi, length))

    def forward(self, i, b):
        freqs = torch.cos(self.idx[i:i + b, None, None] + self.noise[[0]]).div(self.sigma / 50)
        out = torch.sin(freqs + self.noise[[1]]) * self.noise[[2]]
        out = out / (out.square().mean(dim=(1, 2), keepdim=True).sqrt() + torch.finfo(out.dtype).eps)
        return out


class Average(Noise):

    def __init__(self, left, right):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right

    def forward(self, i, b):
        return (self.left(i, b) + self.right(i, b)) / 2


class Modulate(Noise):

    def __init__(self, left, right, modulator):
        super().__init__(left.length, left.size)
        self.left = left
        self.right = right
        self.register_buffer('modulator', modulator.mean(1))

    def forward(self, i, b):
        mod = self.modulator[i:i + b, None, None]
        return self.left(i, b) * mod + self.right(i, b) * (1 - mod)


class ScaleBias(Noise):

    def __init__(self, base, scale, bias):
        super().__init__(base.length, base.size)
        self.base = base
        self.scale = scale
        self.bias = bias

    def forward(self, i, b):
        return self.scale * self.base(i, b) + self.bias


UNITFEATS = ['rms', 'drop_strength', 'onsets', 'spectral_flatness']


ALLFEATS = ['chromagram', 'tonnetz', 'mfcc', 'spectral_contrast'] + UNITFEATS


def gaussian_filter(x, sigma, causal=None, mode='circular'):
    """Smooth tensors along time (first) axis with gaussian kernel.

    Args:
        x (torch.tensor): Tensor to be smoothed
        sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
        causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.

    Returns:
        torch.tensor: Smoothed tensor
    """
    dim = len(x.shape)
    n_frames = x.shape[0]
    while len(x.shape) < 3:
        x = x[:, None]
    radius = min(int(sigma * 4), 3 * len(x))
    channels = x.shape[1]
    kernel = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[radius + 1:] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)
    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)
    if radius > n_frames:
        x = F.pad(x, (n_frames, n_frames), mode=mode)
        None
        x = F.pad(x, (radius - n_frames, radius - n_frames), mode='replicate')
    else:
        x = F.pad(x, (radius, radius), mode=mode)
    x = F.conv1d(x, weight=kernel, groups=channels)
    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)
    if len(x.shape) > dim:
        x = x.squeeze()
    return x


def spline_loop_latents(y, size, n_loops=1):
    y = torch.cat((y, y[[0]]))
    t_in = torch.linspace(0, 1, len(y))
    t_out = torch.linspace(0, n_loops, size) % 1
    coeffs = natural_cubic_spline_coeffs(t_in, y.permute(1, 0, 2))
    out = NaturalCubicSpline(coeffs).evaluate(t_out)
    return out.permute(1, 0, 2)


def latent_patch(rng, latents, palette, segmentations, features, tempo, fps, patch_type, segments, loop_bars, seq_feat, seq_feat_weight, mod_feat, mod_feat_weight, merge_type, merge_depth):
    feature = seq_feat_weight * features[seq_feat]
    segmentation = segmentations[seq_feat, segments]
    permutation = torch.randperm(len(palette), generator=rng, device=rng.device)
    if patch_type == 'segmentation':
        selection = permutation[:segments]
        selectseq = selection[segmentation.cpu().numpy()]
        sequence = palette[selectseq]
        sequence = gaussian_filter(sequence, 5)
    elif patch_type == 'feature':
        n_select = feature.shape[1]
        if n_select == 1:
            selection = permutation[:2]
            sequence = feature[..., None] * palette[selection][[0]] + (1 - feature[..., None]) * palette[selection][[1]]
        else:
            selection = permutation[:n_select]
            sequence = torch.einsum('TN,NWL->TWL', feature, palette[selection])
    elif patch_type == 'loop':
        selection = permutation[:segments]
        n_loops = len(latents) / fps / 60 / tempo / 4 / loop_bars
        sequence = spline_loop_latents(palette[selection], len(latents), n_loops=n_loops)
    sequence = gaussian_filter(sequence, 1)
    if merge_depth == 'low':
        lays = slice(0, 6)
    elif merge_depth == 'mid':
        lays = slice(6, 12)
    elif merge_depth == 'high':
        lays = slice(12, 18)
    elif merge_depth == 'lowmid':
        lays = slice(0, 12)
    elif merge_depth == 'midhigh':
        lays = slice(6, 18)
    elif merge_depth == 'all':
        lays = slice(0, 18)
    if merge_type == 'average':
        latents[:, lays] += sequence[:, lays]
        latents[:, lays] /= 2
    elif merge_type == 'modulate':
        modulation = mod_feat_weight * features[mod_feat][..., None]
        latents[:, lays] *= 1 - modulation
        latents[:, lays] += modulation * sequence[:, lays]
    else:
        latents[:, lays] = sequence[:, lays]
    return latents


def noise_patch(rng, noise, features, tempo, fps, patch_type, loop_bars, seq_feat, seq_feat_weight, mod_feat, mod_feat_weight, merge_type, merge_depth, noise_mean, noise_std):
    if merge_depth == 'low':
        lays = range(0, 6)
    elif merge_depth == 'mid':
        lays = range(6, 12)
    elif merge_depth == 'high':
        lays = range(12, 17)
    elif merge_depth == 'lowmid':
        lays = range(0, 12)
    elif merge_depth == 'midhigh':
        lays = range(6, 17)
    elif merge_depth == 'all':
        lays = range(0, 17)
    feature = seq_feat_weight * features[seq_feat]
    for n in lays:
        if patch_type == 'blend':
            new_noise = Blend(rng=rng, length=len(feature), size=noise[n].size, modulator=feature)
        elif patch_type == 'multiply':
            new_noise = Multiply(rng=rng, length=len(feature), size=noise[n].size, modulator=feature)
        elif patch_type == 'loop':
            n_loops = len(feature) / fps / 60 / tempo / 4 / loop_bars
            new_noise = Loop(rng=rng, length=len(feature), size=noise[n].size, n_loops=n_loops)
        if merge_type == 'average':
            noise[n] = Average(left=noise[n], right=new_noise)
        elif merge_type == 'modulate':
            noise[n] = Modulate(left=noise[n], right=new_noise, modulator=mod_feat_weight * features[mod_feat])
        else:
            noise[n] = new_noise
        noise[n] = ScaleBias(noise[n], scale=noise_std, bias=noise_mean)
    return noise


def random_choice(rng, options, weights=None, n=1, replacement=False):
    if weights is None:
        probabilities = torch.ones(len(options), device=rng.device) / len(options)
    else:
        probabilities = torch.tensor(weights, device=rng.device) / np.sum(weights)
    idx = probabilities.multinomial(num_samples=n, replacement=replacement, generator=rng)
    return options[idx]


def skewnorm(rng, a, loc, scale, size=()):
    """
    skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)
    From https://github.com/scipy/scipy/blob/main/scipy/stats/_continuous_distns.py#L7608-L7682
    """
    u0 = torch.randn(size, generator=rng, device=rng.device)
    v = torch.randn(size, generator=rng, device=rng.device)
    d = a / np.sqrt(1 + a ** 2)
    u1 = d * u0 + v * np.sqrt(1 - d ** 2)
    return loc + scale * torch.where(u0 >= 0, u1, -u1)


class Patch(torch.nn.Module):

    def __init__(self, features, segmentations, tempo, fps=24, seed=42, min_subpatches=2, max_subpatches=20, device='cuda'):
        super().__init__()
        rng = torch.Generator(device)
        rng.manual_seed(seed)
        self.seed = seed
        self.rng = rng
        self.fps = fps
        self.tempo = tempo
        self.length = features[list(features.keys())[0]].shape[0]
        self.features = features
        self.segmentations = segmentations
        self.n_base_latents = torch.randint(3, 15, size=(), generator=rng, device=rng.device).item()
        self.sigma_base_noise = 1 + 9 * torch.rand((), generator=rng, device=rng.device).item()
        self.loops_base_noise = random_choice(rng, [1, 2, 4, 8, 16, 32, 64])
        self.ks = np.unique([k for _, k in segmentations]).tolist()
        self.min_subpatches, self.max_subpatches = min_subpatches, max_subpatches
        self.randomize_latent_patches()
        self.randomize_noise_patches()

    def __getstate__(self):
        return {k: v for k, v in list(self.__dict__.items()) + [('device', self.rng.device)] if k != 'rng'}

    def __setstate__(self, d):
        self.__dict__ = d
        self.rng = torch.Generator(d['device']).manual_seed(d['seed'])

    def randomize_latent_patches(self):
        self.latent_patches = [self.random_latent_patch() for _ in range(torch.randint(self.min_subpatches, self.max_subpatches, size=(), generator=self.rng, device=self.rng.device))]

    def randomize_noise_patches(self):
        self.noise_patches = [self.random_noise_patch() for _ in range(torch.randint(self.min_subpatches, self.max_subpatches, size=(), generator=self.rng, device=self.rng.device))]

    def update_intensity(self, val):
        for p in range(len(self.latent_patches)):
            self.latent_patches[p]['seq_feat_weight'] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.latent_patches[p]['mod_feat_weight'] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
        for p in range(len(self.noise_patches)):
            self.noise_patches[p]['seq_feat_weight'] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.noise_patches[p]['mod_feat_weight'] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()
            self.noise_patches[p]['noise_std'] = skewnorm(self.rng, a=5, loc=val, scale=0.5).item()

    def random_latent_patch(self):
        return dict(patch_type=random_choice(self.rng, ['segmentation', 'feature', 'loop']), segments=random_choice(self.rng, self.ks), loop_bars=random_choice(self.rng, [4, 8, 16, 32], weights=[2, 2, 2, 1]), seq_feat=random_choice(self.rng, ALLFEATS), seq_feat_weight=1, mod_feat=random_choice(self.rng, UNITFEATS), mod_feat_weight=1, merge_type=random_choice(self.rng, ['average', 'modulate'], weights=[1, 3]), merge_depth=random_choice(self.rng, ['low', 'mid', 'high', 'lowmid', 'midhigh', 'all'], weights=[3, 3, 3, 2, 2, 1]))

    def random_noise_patch(self):
        return dict(patch_type=random_choice(self.rng, ['blend', 'multiply', 'loop']), loop_bars=random_choice(self.rng, [4, 8, 16, 32], weights=[2, 2, 2, 1]), seq_feat=random_choice(self.rng, ALLFEATS), seq_feat_weight=1, mod_feat=random_choice(self.rng, UNITFEATS), mod_feat_weight=1, merge_type=random_choice(self.rng, ['average', 'modulate'], weights=[1, 3]), merge_depth=random_choice(self.rng, ['low', 'mid', 'high', 'lowmid', 'midhigh', 'all'], weights=[3, 3, 3, 2, 2, 1]), noise_mean=0, noise_std=1)

    def forward(self, latent_palette, downscale_factor=1, aspect_ratio=1):
        self.rng.manual_seed(self.seed)
        base_selection = torch.randperm(len(latent_palette), generator=self.rng, device=self.rng.device)[:self.n_base_latents]
        latents = spline_loop_latents(latent_palette[base_selection], self.length)
        for subpatch in self.latent_patches:
            latents = latent_patch(self.rng, latents, latent_palette, self.segmentations, self.features, self.tempo, self.fps, **subpatch)
        noise = [Loop(rng=self.rng, length=self.length, size=(round(aspect_ratio * size / downscale_factor), round(size / downscale_factor)), n_loops=self.loops_base_noise, sigma=self.sigma_base_noise) for size in [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]]
        for subpatch in self.noise_patches:
            noise = noise_patch(self.rng, noise, self.features, self.tempo, self.fps, **subpatch)
        return latents, [n for n in noise]

    def __repr__(self):
        reprs = []
        for patches in [self.latent_patches, self.noise_patches]:
            header = [''] + [k for k in patches[0]]
            values = [([str(i + 1)] + [(f'{v:.4f}' if isinstance(v, float) else f'{v}').replace('spectral_', '') for v in p.values()]) for i, p in enumerate(patches)]
            widths = [max([len(row[n]) for row in [header] + values]) for n in range(len(header))]
            seps = [('-' * w) for w in widths]
            strs = [' | '.join([row[c].ljust(widths[c]) for c in range(len(row))]) for row in [header, seps] + values]
            reprs.append(strs)
        return 'Patch(\n  Latent(\n    ' + '\n    '.join(reprs[0]) + '\n  ),\n  Noise(\n    ' + '\n    '.join(reprs[1]) + '\n  )\n)'

    def save(self, path):
        with open(path, mode='w') as f:
            state = dict(seed=self.seed, latent_patches=self.latent_patches, noise_patches=self.noise_patches, n_base_latents=self.n_base_latents, sigma_base_noise=self.sigma_base_noise, loops_base_noise=self.loops_base_noise)
            f.write(json.dumps(state))

    @staticmethod
    def load(path, features, segmentations, tempo, fps, device):
        patch = Patch(features, segmentations, tempo, fps, device)
        with open(path, mode='r') as f:
            patch_info = json.loads(f.read())
        for key, val in patch_info.items():
            setattr(patch, key, val)
        return patch


class EMAFade(torch.nn.Module):

    def __init__(self, fade_frames) ->None:
        super().__init__()
        self.fade_frames = fade_frames
        self.smooth_schedule = torch.cat((torch.linspace(1, 0, fade_frames), torch.linspace(0, 1, fade_frames)))
        self.avg = None

    def forward(self, x, i, total_length):
        batch_size = x.shape[0]
        fade_start = total_length - self.fade_frames
        if i < self.fade_frames or i + batch_size >= fade_start:
            for batch_idx, frame_idx in enumerate(range(i, i + batch_size)):
                if frame_idx == fade_start:
                    self.avg = x[batch_idx]
                if self.fade_frames < frame_idx < fade_start or self.avg is None:
                    continue
                else:
                    smooth_idx = frame_idx - fade_start if frame_idx - fade_start >= 0 else self.fade_frames + frame_idx
                    self.avg *= 1 - self.smooth_schedule[smooth_idx]
                    self.avg += x[batch_idx] * self.smooth_schedule[smooth_idx]
                    x[batch_idx] = self.avg.clone()
        return x


def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    p = p.permute(2, 0, 1)[..., None]
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a[None] * torch.cos(p) + c[None] * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


class LoopLatents(torch.nn.Module):

    def __init__(self, latent_selection, loop_len, type='spline', smooth=10):
        super().__init__()
        latent_selection = latent_selection.cpu()
        if loop_len == 1 or type == 'constant':
            latents = latent_selection[[0]]
            loop_len = 1
        elif type == 'spline':
            latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
            x = np.linspace(0, 1, loop_len)
            latents = np.zeros((loop_len, *latent_selection.shape[1:]))
            for lay in range(latent_selection.shape[1]):
                for lat in range(latent_selection.shape[2]):
                    tck = splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
                    latents[:, lay, lat] = splev(x, tck)
            latents = torch.from_numpy(latents)
        elif type == 'slerp':
            latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])
            latents = []
            for n in range(len(latent_selection)):
                for val in np.linspace(0.0, 1.0, int(ceil(loop_len / len(latent_selection)))):
                    latents.append(torch.from_numpy(slerp(latent_selection[n % len(latent_selection)][0], latent_selection[(n + 1) % len(latent_selection)][0], val)))
            latents = torch.stack(latents).unsqueeze(1).tile(1, 18, 1)
            latents = F.interpolate(latents.permute(2, 1, 0), loop_len, mode='linear', align_corners=False).permute(2, 1, 0)
            latents = gaussian_filter(latents, 1)
        elif type == 'gaussian':
            latents = torch.cat([lat.tile(round(loop_len / len(latent_selection)), 1, 1) for lat in latent_selection])
            latents = F.interpolate(latents.permute(2, 1, 0), loop_len, mode='linear', align_corners=False).permute(2, 1, 0)
            latents = gaussian_filter(latents, smooth)
        self.register_buffer('latents', latents)
        self.index, self.length = 0, loop_len

    def forward(self):
        latents = self.latents[[self.index % self.length]]
        self.index += 1
        return latents


class TempoLoopLatents(LoopLatents):

    def __init__(self, tempo, latent_selection, n_bars, fps, **loop_latents_kwargs):
        if len(latent_selection) == 1:
            loop_len = 1
        else:
            loop_len = round(n_bars * fps * 60 / (tempo / 4))
        super().__init__(latent_selection, loop_len, **loop_latents_kwargs)


class PitchTrackLatents(torch.nn.Module):

    def __init__(self, pitch_track, latent_selection):
        super().__init__()
        low, high = np.percentile(pitch_track, 25), np.percentile(pitch_track, 75)
        pitch_track -= low
        pitch_track /= high
        pitch_track *= len(latent_selection)
        None
        pitch_track = pitch_track.round().long().numpy()
        pitch_track = pitch_track % len(latent_selection)
        latents = torch.from_numpy(latent_selection.numpy()[pitch_track])
        self.register_buffer('latents', latents)
        self.index = 0

    def forward(self):
        latent = self.latents[[self.index]]
        self.index += 1
        return latent


class TonalLatents(torch.nn.Module):

    def __init__(self, chroma_or_tonnetz, latent_selection):
        super().__init__()
        chroma_or_tonnetz /= chroma_or_tonnetz.sum(1)[:, None]
        None
        latents = torch.einsum('Atwl,Atwl->twl', chroma_or_tonnetz.permute(1, 0)[..., None, None], latent_selection[torch.arange(chroma_or_tonnetz.shape[1]) % len(latent_selection), None])
        self.register_buffer('latents', latents)
        self.index = 0

    def forward(self):
        latents = self.latents[[self.index]]
        self.index += 1
        return latents


class ModulatedLatents(torch.nn.Module):

    def __init__(self, modulation, base_latents):
        super().__init__()
        self.register_buffer('latents', modulation[:, None, None] * base_latents[[0]])
        self.index = 0

    def forward(self):
        latents = self.latents[[self.index]]
        self.index += 1
        return latents


class LucidSonicDreamLatents(torch.nn.Module):
    pass


class LazyModulatedLatents(torch.nn.Module):
    pass


class ModulationSum(torch.nn.Module):

    def __init__(self, modulated_modules):
        super().__init__()
        self.modulated_modules = modulated_modules

    def forward(self):
        average, weight = None, torch.zeros([1])
        for mod in self.modulated_modules:
            try:
                weight += mod.modulation[mod.index % len(mod.modulation)]
                if average is None:
                    average = mod.forward().squeeze()
                else:
                    average += mod.forward().squeeze()
            except Exception as e:
                None
                None
                traceback.print_exc()
                exit()
        try:
            return (average / weight).float().unsqueeze(0)
        except:
            return None


class LoopNoise(torch.nn.Module):

    def __init__(self, loop_len, size, smooth):
        super().__init__()
        self.register_buffer('noise', gaussian_filter(torch.randn((loop_len, 1, size, size)), smooth))
        self.noise /= gaussian_filter(self.noise.std((1, 2, 3)), smooth).reshape(-1, 1, 1, 1)
        self.index, self.length = 0, loop_len

    def forward(self):
        noise = self.noise[[self.index % self.length]]
        self.index += 1
        return noise


class TempoLoopNoise(LoopNoise):

    def __init__(self, tempo, n_bars, fps, **loop_noise_kwargs):
        loop_len = round(n_bars * fps * 60 / (tempo / 4))
        super().__init__(loop_len, **loop_noise_kwargs)


class TonalNoise(torch.nn.Module):

    def __init__(self, chroma_or_tonnetz, size):
        super().__init__()
        chroma_or_tonnetz /= chroma_or_tonnetz.sum(1)[:, None]
        noises = torch.randn(chroma_or_tonnetz.shape[1], 1, 1, size, size)
        noise = torch.einsum('Atchw,Atchw->tchw', chroma_or_tonnetz.permute(1, 0)[..., None, None, None], noises)
        self.register_buffer('noise', noise)
        self.noise /= gaussian_filter(self.noise.std((1, 2, 3)), 10).reshape(-1, 1, 1, 1)
        self.index = 0

    def forward(self):
        noise = self.noise[[self.index]]
        self.index += 1
        return noise


class ModulatedNoise(torch.nn.Module):

    def __init__(self, modulation, base_noise=None, size=None):
        super().__init__()
        self.modulation = modulation
        if base_noise is not None:
            self.base_noise = base_noise
        else:
            self.base_noise = LoopNoise(len(modulation), size, 1)
        self.index = 0

    def forward(self):
        noise = self.modulation[self.index] * self.base_noise.forward()
        self.index += 1
        return noise


class CosSinNoise(torch.nn.Module):

    def __init__(self, n_frames):
        pass


class Truncation(torch.nn.Module):
    pass


class Translation(torch.nn.Module):
    pass


class Rotation(torch.nn.Module):
    pass


class Zoom(torch.nn.Module):
    pass


class RealtimeModule(torch.nn.Module):
    __constants__ = ['motion_react', 'motion_randomness', 'motion_smooth', 'truncation']

    def __init__(self, model_file, w, h, motion_react, motion_randomness, motion_smooth, truncation, resize_strategy, resize_layer, device, dtype, inference_network=False):
        super().__init__()
        self.motion_react, self.motion_randomness, self.motion_smooth = motion_react, motion_randomness, motion_smooth
        self.truncation = truncation
        self.G_map = StyleGAN2Mapper(model_file, inference_network)
        self.G_synth = StyleGAN2Synthesizer(model_file, inference_network, (w, h), strategy=resize_strategy, layer=resize_layer)
        rand_factors = torch.ones(B, 512, dtype=dtype, device=device)
        rand_factors[torch.rand_like(rand_factors) > 0.5] -= 0.5
        self.register_buffer('rand_factors', rand_factors)
        self.register_buffer('motion_signs', torch.sign(torch.randn(B, 512, dtype=dtype, device=device)))
        self.i = torch.tensor(0)

    def forward(self, latent: torch.Tensor):
        self.motion_signs[latent - self.motion_react < -2 * self.truncation] = 1
        self.motion_signs[latent + self.motion_react >= 2 * self.truncation] = -1
        if self.i % (24 * 4) == 0:
            new_factors = torch.ones_like(self.rand_factors)
            new_factors[torch.rand_like(self.rand_factors) > 0.5] -= 0.5
            self.rand_factors.set_(new_factors.data)
        motion_noise = self.motion_react * self.motion_signs * self.rand_factors
        latent = latent * self.motion_smooth + (latent + motion_noise) * (1 - self.motion_smooth)
        self.i.set_(self.i + 1)
        mapped_latent = self.G_map(latent)
        raw_img = self.G_synth(mapped_latent)
        img = raw_img.add(1).div(2).clamp(0, 1)
        return latent, img


class Layer(torch.nn.Module):

    def __init__(self, x, f, *args, **kwargs):
        super(Layer, self).__init__()
        self.x = x
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.f(self.x(x, *self.args, **self.kwargs))


class ImageRanker(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()


class BaseDiffusionProcessor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        pass


def fetch(path_or_url):
    if not (path_or_url.startswith('http://') or path_or_url.startswith('https://')):
        return open(path_or_url, 'rb')
    return requests.get(path_or_url, stream=True).raw


class ImagePrompt(torch.nn.Module):

    def __init__(self, img=None, path=None, size=None, weight=1.0):
        super().__init__()
        self.weight = weight
        if path is not None:
            allowed_types = str, Path
            assert isinstance(path, allowed_types), f'path must be one of {allowed_types}'
            img = Image.open(fetch(path)).convert('RGB')
            img = to_tensor(img).unsqueeze(0)
        elif img is not None:
            allowed_types = Image.Image, torch.Tensor, np.ndarray
            assert isinstance(img, allowed_types), f'img must be one of {allowed_types}'
            if isinstance(img, (Image.Image, np.ndarray)):
                img = to_tensor(img).unsqueeze(0)
            else:
                assert img.dim() == 4, 'img must be of shape (B, C, H, W)'
        else:
            raise Exception('path or img must be specified')
        if size is not None:
            img = resize(img, out_shape=size)
        self.register_buffer('img', img.mul(2).sub(1).clamp(-1, 1))

    def forward(self):
        return self.img, self.weight


class ContentPrompt(ImagePrompt):
    pass


class StylePrompt(ImagePrompt):
    pass


class TextPrompt(torch.nn.Module):

    def __init__(self, text, weight=1.0):
        super().__init__()
        self.text = text
        self.weight = weight

    def forward(self):
        return self.text, self.weight


def destitch(img, tile_size, overtile=1):
    _, _, H, W = img.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    tiled = []
    for y in torch.linspace(0, H - tile_size, n_rows).round().long():
        for x in torch.linspace(0, W - tile_size, n_cols).round().long():
            tiled.append(img[..., y:y + tile_size, x:x + tile_size])
    return torch.cat(tiled, dim=0)


def interp(t):
    return 3 * t ** 2 - 2 * t ** 3


def perlin(width, height, scale=10, device='cuda'):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None]
    ys = torch.linspace(0, 1, scale + 1)[None, :-1]
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = to_pil_image(out.clamp(0, 1).squeeze())
    out = ImageOps.autocontrast(out)
    return to_tensor(out)


def initialize_image(init, shape):
    if init == 'random':
        img = torch.randn((1, 3, *shape))
    elif init == 'perlin':
        img = (resize(create_perlin_noise([(1.5 ** -i * 0.5) for i in range(12)], 1, 1, False), out_shape=shape) + resize(create_perlin_noise([(1.5 ** -i * 0.5) for i in range(8)], 4, 4, True), out_shape=shape) - 1).unsqueeze(0)
    elif init is not None:
        img = resize(to_tensor(Image.open(init).convert('RGB')).unsqueeze(0).mul(2).sub(1), out_shape=shape)
    else:
        raise Exception('init strategy not recognized!')
    return img


def smoothstep(x, N=2):
    result = torch.zeros_like(x)
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
    result *= x ** (N + 1)
    return result


def blend_weight1d(total_size, fade_in, fade_out):
    return torch.cat((smoothstep(torch.linspace(0, 1, fade_in)), torch.ones(total_size - fade_in - fade_out), smoothstep(torch.linspace(1, 0, fade_out))))


def restitch(tiled, H, W, overtile=1):
    _, C, _, tile_size = tiled.shape
    n_rows = round(np.floor(H / tile_size) + overtile)
    n_cols = round(np.floor(W / tile_size) + overtile)
    out = torch.zeros((1, C, H, W), device=tiled.device)
    rescale = torch.zeros_like(out)
    i = 0
    ys = torch.linspace(0, H - tile_size, n_rows).round().long()
    xs = torch.linspace(0, W - tile_size, n_cols).round().long()
    fade = tile_size - ys[1]
    for y in ys:
        wy = blend_weight1d(tile_size, fade_in=0 if y == 0 else fade, fade_out=0 if y == ys[-1] else fade)
        for x in xs:
            wx = blend_weight1d(tile_size, fade_in=0 if x == 0 else fade, fade_out=0 if x == xs[-1] else fade)
            weight = wy.reshape(1, 1, -1, 1) * wx.reshape(1, 1, 1, -1)
            out[..., y:y + tile_size, x:x + tile_size] += tiled[i] * weight
            rescale[..., y:y + tile_size, x:x + tile_size] += weight
            i += 1
    return out / rescale


def round64(x):
    return round(x / 64) * 64


MODEL_MODULES = {'RIFE-1.0': rife, 'RIFE-1.1': rife, 'RIFE-2.0': rife, 'RIFE-2.1': rife, 'RIFE-2.2': rife, 'RIFE-2.3': rife, 'RIFE-2.4': rife, 'RIFE-3.0': rife, 'RIFE-3.1': rife, 'RIFE-3.2': rife, 'RIFE-3.4': rife, 'RIFE-3.5': rife, 'RIFE-3.6': rife, 'RIFE-3.8': rife, 'RIFE-3.9': rife, 'RIFE-4.0': rife}


class MultiResolutionDiffusionProcessor(torch.nn.Module):

    def forward(self, diffusion: BaseDiffusionProcessor, init: str, text: Optional[str]=None, image: Optional[str]=None, content: Optional[str]=None, style: Optional[str]=None, schedule: Dict[Tuple[int, int], float]={(512, 512), 0.5}, pre_hook: Optional[Callable]=None, post_hook: Optional[Callable]=None, super_res_model: Optional[str]=None, tile_size: Optional[int]=None, stitch: bool=True, max_batch: int=4, verbose: bool=True):
        shapes = [(round64(h), round64(w)) for h, w in list(schedule.keys())]
        t_starts = list(schedule.values())
        if tile_size is None:
            tile_size = diffusion.image_size
        img = initialize_image(init, shapes[0])
        if content is None:
            content = dict(img=img.clone())
        else:
            content = dict(path=content)
        for scale, t_start in enumerate(t_starts):
            if verbose:
                None
            if scale != 0:
                if super_res_model:
                    try:
                        img = upscale_image(img.add(1).div(2), model_name=super_res_model).mul(2).sub(1)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise
                img = resize(img, out_shape=shapes[scale], interp_method=lanczos3).cpu()
            if pre_hook:
                img = pre_hook(img)
            needs_stitching = stitch and min(shapes[scale]) > tile_size
            if needs_stitching:
                img = destitch(img, tile_size=tile_size)
            prompts = [ContentPrompt(**content)] if not needs_stitching else []
            if style is not None:
                prompts.append(StylePrompt(path=style, size=shapes[scale]))
            if text is not None:
                prompts.append(TextPrompt(text))
            if image is not None:
                prompts.append(ImagePrompt(path=image))
            dev = diffusion.device
            if img.shape[0] > max_batch:
                tiles = tqdm(img.split(max_batch)) if verbose else img.split(max_batch)
                img = torch.cat([diffusion(ims, prompts, t_start, verbose=False) for ims in tiles])
            else:
                img = diffusion(img, prompts, t_start, verbose=verbose)
            if needs_stitching:
                img = restitch(img, *shapes[scale])
            if post_hook:
                img = post_hook(img)
        return img


class CFGDenoiser(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class LatentGradientGuidedConditioning(torch.nn.Module):

    def __init__(self, diffusion, model, bert, ldm, grad_modules, device):
        super().__init__()
        self.diffusion, self.model, self.bert, self.ldm = diffusion, model, bert, ldm
        self.grad_modules = torch.nn.ModuleList(grad_modules)
        self.timestep_map = diffusion.timestep_map
        sqrt_one_minus_alphas_cumprod = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).float()
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

    def set_targets(self, prompts, noise):
        for grad_module in self.grad_modules:
            grad_module.set_targets(prompts)

    def forward(self, x, t, kw={}):
        ot = t.clone()
        t = torch.tensor([self.timestep_map.index(t) for t in t.long()], device=x.device, dtype=torch.long)
        with torch.enable_grad():
            half = x.shape[0] // 2
            x = x[:half].detach().requires_grad_()
            out = self.diffusion.p_mean_variance(self.model, x, t[:half], clip_denoised=False, model_kwargs={k: (v[:half] if v is not None else None) for k, v in kw.items()})['pred_xstart']
            sigma = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            out = out * sigma + x * (1 - sigma)
            img = self.ldm.decode(out / 0.18215)
            img_grad = torch.zeros_like(img)
            for grad_mod in self.grad_modules:
                sub_grad = grad_mod(img, ot)
                if torch.isnan(sub_grad).any():
                    None
                    sub_grad = torch.zeros_like(img)
                img_grad += sub_grad
            grad = -torch.autograd.grad(img, x, img_grad)[0]
        return grad


class ConvBlock(torch.nn.Sequential):

    def __init__(self, c_in, c_out):
        super().__init__(torch.nn.Conv2d(c_in, c_out, 3, padding=1), torch.nn.ReLU(inplace=True))


class FourierFeatures(torch.nn.Module):

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = torch.nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * torch.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SkipBlock(torch.nn.Module):

    def __init__(self, main, skip=None):
        super().__init__()
        self.main = torch.nn.Sequential(*main)
        self.skip = skip if skip else torch.nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def t_to_alpha_sigma(t):
    return torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)


class SecondaryDiffusionImageNet2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        c = 64
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = torch.nn.AvgPool2d(2)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.net = torch.nn.Sequential(ConvBlock(3 + 16, cs[0]), ConvBlock(cs[0], cs[0]), SkipBlock([self.down, ConvBlock(cs[0], cs[1]), ConvBlock(cs[1], cs[1]), SkipBlock([self.down, ConvBlock(cs[1], cs[2]), ConvBlock(cs[2], cs[2]), SkipBlock([self.down, ConvBlock(cs[2], cs[3]), ConvBlock(cs[3], cs[3]), SkipBlock([self.down, ConvBlock(cs[3], cs[4]), ConvBlock(cs[4], cs[4]), SkipBlock([self.down, ConvBlock(cs[4], cs[5]), ConvBlock(cs[5], cs[5]), ConvBlock(cs[5], cs[5]), ConvBlock(cs[5], cs[4]), self.up]), ConvBlock(cs[4] * 2, cs[4]), ConvBlock(cs[4], cs[3]), self.up]), ConvBlock(cs[3] * 2, cs[3]), ConvBlock(cs[3], cs[2]), self.up]), ConvBlock(cs[2] * 2, cs[2]), ConvBlock(cs[2], cs[1]), self.up]), ConvBlock(cs[1] * 2, cs[1]), ConvBlock(cs[1], cs[0]), self.up]), ConvBlock(cs[0] * 2, cs[0]), torch.nn.Conv2d(cs[0], 3, 3, padding=1))

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


def get_checkpoint(checkpoint_name):
    if checkpoint_name == 'uncondImageNet512':
        checkpoint_path = 'modelzoo/512x512_diffusion_uncond_finetune_008100.pt'
        if not os.path.exists(checkpoint_path):
            download('https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt', checkpoint_path)
        checkpoint_config = {'image_size': 512}
    elif checkpoint_name == 'uncondImageNet256':
        checkpoint_path = 'modelzoo/256x256_diffusion_uncond.pt'
        if not os.path.exists(checkpoint_path):
            download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', checkpoint_path)
        checkpoint_config = {'image_size': 256}
    return checkpoint_path, checkpoint_config


def create_models(checkpoint='uncondImageNet512', timestep_respacing='100', diffusion_steps=1000, use_secondary=False):
    checkpoint_path, checkpoint_config = get_checkpoint(checkpoint)
    model_config = model_and_diffusion_defaults()
    model_config.update({'attention_resolutions': '32, 16, 8', 'class_cond': False, 'diffusion_steps': diffusion_steps, 'rescale_timesteps': True, 'timestep_respacing': timestep_respacing, 'learn_sigma': True, 'noise_schedule': 'linear', 'num_channels': 256, 'num_head_channels': 64, 'num_res_blocks': 2, 'resblock_updown': True, 'use_fp16': True, 'use_scale_shift_norm': True})
    model_config.update(checkpoint_config)
    diffusion_model, diffusion = create_model_and_diffusion(**model_config)
    diffusion_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    diffusion_model.requires_grad_(False).eval()
    for name, param in diffusion_model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        diffusion_model.convert_to_fp16()
    if use_secondary:
        checkpoint_path = 'modelzoo/secondary_model_imagenet_2.pth'
        if not os.path.exists(checkpoint_path):
            download('https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', checkpoint_path)
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        secondary_model.eval().requires_grad_(False)
    else:
        secondary_model = None
    return diffusion_model, diffusion, secondary_model


class GLID3XL(BaseDiffusionProcessor):

    def __init__(self, grad_modules=[], cfg_scale=3, sampler='ddim', timesteps=100, model_checkpoint='finetune', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ddim_eta=0):
        super().__init__()
        self.use_backward_guidance = any(gm.scale > 0 for gm in grad_modules)
        self.model, self.diffusion, self.ldm, self.bert = create_models(checkpoint=model_checkpoint, timestep_respacing=f'ddim{timesteps}' if sampler == 'ddim' else str(timesteps), use_backward_guidance=self.use_backward_guidance, device=device)
        if self.model.clip_conditioned:
            self.clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
            self.clip_model.eval().requires_grad_(False)
        self.conditioning = LatentGradientGuidedConditioning(self.diffusion, self.model, self.bert, self.ldm, [gm for gm in grad_modules if gm.scale != 0], device) if self.use_backward_guidance else None
        self.cfg_scale = cfg_scale

        def model_fn(x_t, ts, scale, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        if sampler == 'p':
            self.sample_fn = lambda _, scale: partial(self.diffusion.p_sample, model=partial(model_fn, scale=scale), clip_denoised=False)
        elif sampler == 'ddim':
            self.sample_fn = lambda _, scale: partial(self.diffusion.ddim_sample, model=partial(model_fn, scale=scale), eta=ddim_eta, clip_denoised=False)
        elif sampler == 'plms':
            self.sample_fn = lambda old_eps, scale: partial(self.diffusion.prk_sample if len(old_eps) < 3 else partial(self.diffusion.plms_sample, old_eps=old_eps), model=partial(model_fn, scale=scale), clip_denoised=False)
        else:
            raise NotImplementedError()
        self.device = device
        self.model = self.model
        self.original_num_steps = self.diffusion.original_num_steps
        self.timestep_map = self.diffusion.timestep_map
        self.image_size = self.model.image_size * 8

    @torch.no_grad()
    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        start_step = round(t_start * (len(self.timestep_map) - 1))
        n_steps = round((t_end - t_start) * (len(self.timestep_map) - 1))
        t = torch.tensor([start_step] * B, device=self.device, dtype=torch.long)
        B = img.shape[0]
        img = self.ldm.encode(img).sample() * 0.18215
        noise = torch.randn_like(img)
        if self.use_backward_guidance:
            self.conditioning.set_targets([p for p in prompts], noise)
        img = self.diffusion.q_sample(img, t, noise)
        img = torch.cat([img, img], dim=0)
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                text, _ = prompt()
                break
        neg = ''
        text_emb = self.bert.encode([text] * B)
        text_blank = self.bert.encode([neg] * B)
        context = torch.cat([text_emb, text_blank], dim=0)
        if self.model.clip_conditioned:
            text_emb_clip = self.clip_model.encode_text(clip.tokenize([text] * B, truncate=True))
            text_emb_clip_blank = self.clip_model.encode_text(clip.tokenize([neg] * B, truncate=True))
            clip_context = torch.cat([text_emb_clip, text_emb_clip_blank], dim=0)
        kw = {'context': context, 'clip_embed': clip_context if self.model.clip_conditioned else None, 'image_embed': image_embed if self.model.image_conditioned else None}
        t = torch.tensor([start_step] * img.shape[0], device=self.device, dtype=torch.long)
        old_eps = []
        for _ in (trange if verbose else range)(n_steps):
            out = self.sample_fn(old_eps, self.cfg_scale)(x=img, t=t, cond_fn=self.conditioning, model_kwargs=kw)
            img = out['sample']
            if 'eps' in out:
                if len(old_eps) >= 3:
                    old_eps.pop(0)
                old_eps.append(out['eps'])
            t -= 1
        return self.ldm.decode(out['pred_xstart'][:B] / 0.18215)


class GLIDE(BaseDiffusionProcessor):

    def __init__(self, cfg_scale=3, sampler='plms', timesteps=25, model_checkpoint='base', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ddim_eta=0, temp=1.0):
        super().__init__()
        self.temp, self.device = temp, device
        self.cfg_scale = cfg_scale
        using_gpu = device.type == 'cuda'
        options = model_and_diffusion_defaults()
        options['use_fp16'] = using_gpu
        options['timestep_respacing'] = str(timesteps + 2)
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if using_gpu:
            model.convert_to_fp16()
        model
        model.load_state_dict(load_checkpoint(model_checkpoint, device, cache_dir='modelzoo/'))
        self.model, self.diffusion = model, diffusion
        self.ctx, self.original_num_steps = options['text_ctx'], options['diffusion_steps']
        self.timestep_map = np.linspace(0, self.original_num_steps, timesteps + 1).round().astype(int)
        self.image_size = 256
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = using_gpu
        options_up['timestep_respacing'] = str(round(0.6 * timesteps) + 2)
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        if using_gpu:
            model_up.convert_to_fp16()
        model_up
        model_up.load_state_dict(load_checkpoint('upsample', device, cache_dir='modelzoo/'))
        self.model_up, self.diffusion_up = model_up, diffusion_up
        self.scale_factor = options_up['image_size'] / options['image_size']

        def model_fn(x_t, ts, scale, **kwargs):
            half = x_t[:len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        if sampler == 'p':
            self.sample_fn = lambda _, scale: partial(self.diffusion.p_sample, model=partial(model_fn, scale=scale))
            self.upsample_fn = self.diffusion_up.p_sample_loop
        elif sampler == 'ddim':
            self.sample_fn = lambda _, scale: partial(self.diffusion.ddim_sample, eta=ddim_eta, model=partial(model_fn, scale=scale))
            self.upsample_fn = self.diffusion_up.ddim_sample_loop
        elif sampler == 'plms':
            self.sample_fn = lambda old_eps, scale: partial(self.diffusion.prk_sample if len(old_eps) < 3 else partial(self.diffusion.plms_sample, old_eps=old_eps), model=partial(model_fn, scale=scale))
            self.upsample_fn = self.diffusion_up.plms_sample_loop
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        start_step = round(t_start * (len(self.timestep_map) - 1))
        n_steps = round((t_end - t_start) * (len(self.timestep_map) - 1))
        B, C, H, W = img.shape
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, _ = prompt()
                tokens = self.model.tokenizer.encode(txt)
                tokens, mask = self.model.tokenizer.padded_tokens_and_mask(tokens, self.ctx)
                uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask([], self.ctx)
                model_kwargs = dict(tokens=torch.tensor([tokens] * B + [uncond_tokens] * B, device=self.device), mask=torch.tensor([mask] * B + [uncond_mask] * B, dtype=torch.bool, device=self.device))
                break
        if n_steps is None:
            n_steps = start_step
        img = resize(img, scale_factors=1 / self.scale_factor)
        noise = torch.randn_like(img)
        img = self.diffusion.q_sample(img, torch.tensor([start_step] * B, device=self.device, dtype=torch.long), noise)
        img = torch.cat((img, noise))
        t = torch.tensor([start_step] * img.shape[0], device=self.device, dtype=torch.long)
        self.model.del_cache()
        old_eps = []
        for _ in (trange if verbose else range)(n_steps):
            out = self.sample_fn(old_eps, self.cfg_scale)(x=img, t=t, model_kwargs=model_kwargs)
            img = out['sample']
            if 'eps' in out:
                if len(old_eps) >= 3:
                    old_eps.pop(0)
                old_eps.append(out['eps'])
            t -= 1
        self.model.del_cache()
        if t.sum() == 0:
            tokens = self.model_up.tokenizer.encode(txt)
            tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(tokens, self.ctx)
            model_kwargs = dict(low_res=((out['sample'][:B] + 1) * 127.5).round() / 127.5 - 1, tokens=torch.tensor([tokens] * B, device=self.device), mask=torch.tensor([mask] * B, dtype=torch.bool, device=self.device))
            self.model_up.del_cache()
            up_shape = B, C, H, W
            final = self.upsample_fn(self.model_up, up_shape, noise=torch.randn(up_shape, device=self.device) * self.temp, device=self.device, model_kwargs=model_kwargs, progress=verbose)[:B]
            self.model_up.del_cache()
        else:
            final = resize(out['pred_xstart'][:B], scale_factors=self.scale_factor)
        return final


class GradientGuidedConditioning(torch.nn.Module):

    def __init__(self, diffusion, model, grad_modules, speed='fast'):
        super().__init__()
        self.speed = speed
        if speed == 'hyper':
            pass
        elif speed == 'fast':
            self.model = model
        else:
            self.model = partial(diffusion.p_mean_variance, model=model, clip_denoised=False)
        self.grad_modules = torch.nn.ModuleList(grad_modules)
        self.timestep_map = diffusion.timestep_map
        sqrt_alphas_cumprod = torch.from_numpy(diffusion.sqrt_alphas_cumprod).float()
        sqrt_one_minus_alphas_cumprod = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).float()
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

    def set_targets(self, prompts, noise):
        self.noise = noise
        for grad_module in self.grad_modules:
            grad_module.set_targets(prompts)

    def forward(self, x, t, kw={}):
        ot = t.clone()
        t = torch.tensor([self.timestep_map.index(t) for t in t.long()], device=x.device, dtype=torch.long)
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            alpha = self.sqrt_alphas_cumprod[t]
            sigma = self.sqrt_one_minus_alphas_cumprod[t]
            if torch.isnan(x).any():
                None
            if self.speed == 'hyper':
                img = (x - sigma.reshape(-1, 1, 1, 1) * self.noise).div(alpha.reshape(-1, 1, 1, 1))
            elif self.speed == 'fast':
                cosine_t = torch.atan2(sigma, alpha) * 2 / torch.pi
                out = self.model(x, cosine_t).pred
                img = out * sigma.reshape(-1, 1, 1, 1) + x * (1 - sigma.reshape(-1, 1, 1, 1))
            else:
                out = self.model(x=x, t=t, model_kwargs=kw)['pred_xstart']
                img = out * sigma.reshape(-1, 1, 1, 1) + x * (1 - sigma.reshape(-1, 1, 1, 1))
            if torch.isnan(img).any():
                None
            img_grad = torch.zeros_like(img)
            for grad_mod in self.grad_modules:
                sub_grad = grad_mod(img, ot)
                if torch.isnan(sub_grad).any():
                    None
                    sub_grad = torch.zeros_like(img)
                img_grad += sub_grad
            grad = -torch.autograd.grad(img, x, img_grad)[0]
        return grad


class GuidedDiffusion(BaseDiffusionProcessor):

    def __init__(self, grad_modules, sampler='ddim', timesteps=100, model_checkpoint='uncondImageNet512', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ddim_eta=0, plms_order=2, speed='fast'):
        super().__init__()
        self.model, self.diffusion, secondary_model = create_models(checkpoint=model_checkpoint, timestep_respacing=f'ddim{timesteps}' if sampler == 'ddim' else str(timesteps), use_secondary=speed == 'fast')
        self.conditioning = GradientGuidedConditioning(self.diffusion, secondary_model if speed == 'fast' else self.model, [gm for gm in grad_modules if gm.scale != 0], speed=speed)
        if sampler == 'p':
            self.sample_fn = lambda _: partial(self.diffusion.p_sample, clip_denoised=False, model_kwargs={})
        elif sampler == 'ddim':
            self.sample_fn = lambda _: partial(self.diffusion.ddim_sample, eta=ddim_eta, clip_denoised=False, model_kwargs={})
        elif sampler == 'plms':
            self.sample_fn = lambda old_out: partial(self.diffusion.plms_sample, order=plms_order, old_out=old_out, clip_denoised=False, model_kwargs={})
        else:
            raise NotImplementedError()
        self.device = device
        self.model = self.model
        self.conditioning = self.conditioning
        self.original_num_steps = self.diffusion.original_num_steps
        self.timestep_map = self.diffusion.timestep_map
        self.image_size = self.model.image_size

    @torch.no_grad()
    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        start_step = round(t_start * (len(self.timestep_map) - 1))
        n_steps = round((t_end - t_start) * (len(self.timestep_map) - 1))
        t = torch.tensor([start_step] * img.shape[0], device=self.device, dtype=torch.long)
        noise = torch.randn_like(img)
        self.conditioning.set_targets([p for p in prompts], noise)
        img = self.diffusion.q_sample(img, t, noise)
        out = None
        for _ in (trange if verbose else range)(n_steps):
            out = self.sample_fn(out)(model=self.model, x=img, t=t, cond_fn=self.conditioning)
            img = out['sample']
            t -= 1
        return out['pred_xstart']


class LatentConditioning(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, _ = prompt()
                conditioning = self.model.get_learned_conditioning([txt])
                unconditional = self.model.get_learned_conditioning([''])
        return conditioning, unconditional


class Silence:

    def __enter__(self):
        logging.set_verbosity_error()
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        logging.set_verbosity_warning()


def load_model_from_config(config, ckpt):
    with Silence():
        sd = torch.load(ckpt, map_location='cpu')['state_dict']
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model
        model.eval()
    return model


def get_model(checkpoint):
    config = os.path.abspath(os.path.dirname(__file__)) + '/../../submodules/stable_diffusion/configs/stable-diffusion/v1-inference.yaml'
    version = checkpoint.replace('.', '-')
    ckpt = f'modelzoo/stable-diffusion-v{version}.ckpt'
    if checkpoint in ['1.1', '1.2', '1.3']:
        if not os.path.exists(ckpt):
            hf_hub_download(repo_id=f'CompVis/stable-diffusion-v-{version}-original', filename=f'sd-v{version}.ckpt', cache_dir='modelzoo/', force_filename=f'stable-diffusion-v{version}.ckpt', use_auth_token=True)
    elif checkpoint == '1.4':
        if not os.path.exists(ckpt):
            download('https://bearsharktopus.b-cdn.net/drilbot_pics/sd-v1-4.ckpt', ckpt)
    elif checkpoint == 'pinkney':
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/../../submodules/stable_diffusion_image_conditioned')
        config = os.path.abspath(os.path.dirname(__file__)) + '/../../submodules/stable_diffusion_image_conditioned/configs/stable-diffusion/sd-image-condition-finetune.yaml'
        ckpt = f'modelzoo/stable-diffusion-image-conditioned.ckpt'
        if not os.path.exists(ckpt):
            download('https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned/resolve/main/sd-clip-vit-l14-img-embed_ema_only.ckpt', ckpt)
    else:
        ckpt = checkpoint
    return load_model_from_config(OmegaConf.load(config), ckpt)


class LatentDiffusion(BaseDiffusionProcessor):

    def __init__(self, cfg_scale=3, sampler='ddim', timesteps=100, model_checkpoint='large', ddim_eta=0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.model = get_model(model_checkpoint)
        self.conditioning = LatentConditioning(self.model)
        self.cfg_scale = cfg_scale
        if sampler == 'plms':
            sampler = PLMSSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.plms_sampling
        else:
            sampler = DDIMSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.ddim_sampling
        self.device = device
        self.model = self.model
        self.original_num_steps = sampler.ddpm_num_timesteps
        self.timestep_map = np.linspace(0, sampler.ddpm_num_timesteps, timesteps + 1).round().astype(int)
        self.image_size = self.model.image_size * 8

    @torch.no_grad()
    def forward(self, img, prompts, t_start, t_end=1, verbose=True):
        conditioning, unconditional = self.conditioning([p for p in prompts])
        start_step = round(t_start * (len(self.timestep_map) - 1))
        n_steps = round((t_end - t_start) * (len(self.timestep_map) - 1))
        with self.model.ema_scope():
            x_T = self.model.get_first_stage_encoding(self.model.encode_first_stage(img))
            t = torch.ones([x_T.shape[0]], device=self.device, dtype=torch.long) * self.timestep_map[start_step]
            x_T = self.model.q_sample(x_T, t - 1, torch.randn_like(x_T))
            samples, _ = self.sample_fn(x_T=x_T, shape=x_T.shape, timesteps=n_steps, cond=conditioning.tile(x_T.shape[0], 1, 1), unconditional_guidance_scale=self.cfg_scale, unconditional_conditioning=unconditional.tile(x_T.shape[0], 1, 1) if self.cfg_scale != 1 else None)
            samples_out = self.model.decode_first_stage(samples)
        return samples_out


class StableConditioning(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, _ = prompt()
                conditioning = self.model.get_learned_conditioning([txt])
                unconditional = self.model.get_learned_conditioning([''])
            elif type(prompt) == ImagePrompt:
                img, _ = prompt()
                conditioning = self.model.get_learned_conditioning(img)
                unconditional = self.model.get_learned_conditioning(torch.rand_like(img).mul(2).sub(1))
        return conditioning, unconditional


def cfg_forward(x, sigma, uncond, cond, cond_scale, model):
    x_in = torch.cat([x] * 2)
    sigma_in = torch.cat([sigma] * 2)
    cond_in = torch.cat([uncond, cond])
    uncond, cond = model(x_in, sigma_in, cond=cond_in).chunk(2)
    return uncond + (cond - uncond) * cond_scale


def conditioning_wrapper(model, cond_fn):

    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * k_diffusion.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised
    return model_fn


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def sliced_cross_attention(x, context=None, mask=None, self=None):
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)
    k_in = self.to_k(context)
    v_in = self.to_v(context)
    del context, x
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in
    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1
    if mem_required > mem_free_total:
        steps = 2 ** math.ceil(math.log(mem_required / mem_free_total, 2))
    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')
    slice_size = q.shape[1] // steps if q.shape[1] % steps == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
        s2 = s1.softmax(dim=-1, dtype=q.dtype)
        del s1
        r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
        del s2
    del q, k, v
    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1
    return self.to_out(r2)


def use_sliced_attention(module):
    if module.__class__.__name__ == 'CrossAttention':
        module.forward = partial(sliced_cross_attention, self=module)


class StableDiffusion(LatentDiffusion):

    def __init__(self, grad_modules=[], cfg_scale=7.5, sampler='euler_ancestral', timesteps=50, model_checkpoint='1.4', ddim_eta=0, device='cuda' if torch.cuda.is_available() else 'cpu', sliced_attention=True):
        super(BaseDiffusionProcessor, self).__init__()
        self.model = get_model(model_checkpoint)
        self.image_size = self.model.image_size * 8
        if sliced_attention:
            self.model.apply(use_sliced_attention)
        self.conditioning = StableConditioning(self.model)
        self.cfg_scale = cfg_scale
        if sampler == 'plms':
            sampler = PLMSSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.plms_sampling
            self.original_num_steps = sampler.ddpm_num_timesteps
        elif sampler == 'ddim':
            sampler = DDIMSampler(self.model)
            sampler.make_schedule(ddim_num_steps=timesteps, ddim_eta=ddim_eta, verbose=False)
            self.sample_fn = sampler.ddim_sampling
            self.original_num_steps = sampler.ddpm_num_timesteps
        else:
            self.model_wrap = k_diffusion.external.CompVisDenoiser(self.model)
            self.sigmas = self.model_wrap.get_sigmas(timesteps)
            sample_fn = getattr(k_diffusion.sampling, f'sample_{sampler}')
            if 'sigma_min' in inspect.signature(sample_fn).parameters:

                def dpm_wrap(model_fn, x, sigmas, extra_args, disable, **kwargs):
                    sigma_min, sigma_max = sigmas[-1].item(), sigmas[0].item()
                    if sigma_min == 0:
                        sigma_min = sigmas[-2].item()
                    if sigma_max == 0:
                        sigma_max = sigmas[1].item()
                    dpm_kwargs = dict(sigma_min=sigma_min, sigma_max=sigma_max)
                    if 'n' in inspect.signature(sample_fn).parameters:
                        dpm_kwargs['n'] = len(sigmas)
                    return sample_fn(model_fn, x, **dpm_kwargs, extra_args=extra_args, disable=disable, **kwargs)
                self.sample_fn = dpm_wrap
            else:
                self.sample_fn = sample_fn
            self.original_num_steps = len(self.model.alphas_cumprod)
            self.model_fn = partial(cfg_forward, model=self.model_wrap)
            self.grad_modules = [gm for gm in grad_modules if gm.scale != 0]
            if len(self.grad_modules) > 0:

                def cond_fn(x, t, denoised, **kwargs):
                    img = self.model.differentiable_decode_first_stage(denoised)
                    img_grad = torch.zeros_like(img)
                    for grad_mod in self.grad_modules:
                        img_grad += grad_mod(img, t)
                    grad = -torch.autograd.grad(img, x, img_grad)[0]
                    return grad
                self.model_fn = conditioning_wrapper(self.model_fn, cond_fn)
        self.device = device
        self.model = self.model
        self.timestep_map = np.linspace(0, self.original_num_steps, timesteps + 1).round().astype(int)

    def encode(self, img):
        return self.model.get_first_stage_encoding(self.model.encode_first_stage(img))

    def decode(self, x):
        return self.model.decode_first_stage(x)

    def get_sigmas(self, t_s, t_e=None):
        step_start = round(t_s * (len(self.sigmas) - 1))
        if t_e is None:
            return self.sigmas[step_start]
        else:
            step_end = round(t_e * (len(self.sigmas) - 1)) + 1
            return self.sigmas[step_start:step_end]

    @torch.inference_mode()
    def forward(self, img, prompts, t_start, t_end=1, verbose=True, reverse=False, latent=False):
        if not hasattr(self, 'sigmas'):
            return super().forward(img, prompts, t_start, t_end, verbose)
        sigmas = self.get_sigmas(t_start, t_end)
        if reverse:
            sigmas = sigmas.flip(0)
        prompts = [p for p in prompts]
        [gm.set_targets(prompts) for gm in self.grad_modules]
        cond, uncond = self.conditioning(prompts)
        cond_shape = img.shape[0], cond.shape[1], cond.shape[2]
        cond_info = {'cond': cond.expand(cond_shape), 'uncond': uncond.expand(cond_shape), 'cond_scale': self.cfg_scale}
        with autocast(self.device), self.model.ema_scope():
            if t_start > 0 or reverse:
                x = img if latent else self.encode(img)
                x += torch.randn_like(x) * sigmas[0]
            else:
                b, _, h, w = img.shape
                if not latent:
                    h, w = h // 8, w // 8
                x = torch.randn([b, 4, h, w], device=img.device, dtype=img.dtype) * sigmas[0]
            out = self.sample_fn(self.model_fn, x, sigmas, extra_args=cond_info, disable=not verbose)
            out = out if latent else self.decode(out)
        return out.float()


class VideoFrames(Dataset):

    def __init__(self, filename, height, width, device):
        super().__init__()
        self.reader = decord.VideoReader(filename, width=width, height=height)
        self.prepare = lambda x: x.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return torch.stack([self.prepare(self.reader[i]) for i in idx])
        return self.prepare(self.reader[idx])


def build_output_name(init=None, style=None, text=None, image=None, unique=True):
    out_name = str(uuid4())[:6] if unique else 'video'
    if text is not None:
        out_name = f"{text.replace(' ', '_')}_{out_name}"
    if image is not None:
        out_name = f'{Path(image).stem}_{out_name}'
    if style is not None:
        out_name = f'{Path(style).stem}_{out_name}'
    if init is not None:
        out_name = f'{Path(init).stem}_{out_name}'
    return out_name


def flow_warp_map(flow: torch.Tensor) ->torch.Tensor:
    b, h, w, two = flow.shape
    flow[..., 0] /= w
    flow[..., 1] /= h
    global NEUTRAL
    if NEUTRAL is None or (NEUTRAL.shape[1], NEUTRAL.shape[2]) != (h, w):
        NEUTRAL = torch.stack(torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), indexing='xy'), axis=2).unsqueeze(0)
    warp_map = NEUTRAL + flow
    return warp_map


def encode_mflo(flow):
    """Optical flow encoding which can be saved as JPEG"""
    absmax = np.max(np.abs(flow))
    one, two, three, four = struct.pack('!f', absmax)
    h, w, _ = flow.shape
    absmax_channel = np.zeros((h, w, 1), dtype=np.uint8)
    absmax_channel[:h // 2, :w // 2] = one
    absmax_channel[:h // 2, w // 2:] = two
    absmax_channel[h // 2:, :w // 2] = three
    absmax_channel[h // 2:, w // 2:] = four
    mflo = np.round((flow / absmax + 1) * 127.5).astype(np.uint8)
    mflo = np.concatenate((mflo, absmax_channel), axis=2)
    return mflo


def decode_mflo(mflo):
    h, w, _ = mflo.shape
    absmax_channel = mflo[..., 2]
    one = np.mean(absmax_channel[:h // 2, :w // 2].astype(np.float32)).round().astype(np.uint8)
    two = np.mean(absmax_channel[:h // 2, w // 2:].astype(np.float32)).round().astype(np.uint8)
    three = np.mean(absmax_channel[h // 2:, :w // 2].astype(np.float32)).round().astype(np.uint8)
    four = np.mean(absmax_channel[h // 2:, w // 2:].astype(np.float32)).round().astype(np.uint8)
    absmax, = struct.unpack('!f', bytearray([one, two, three, four]))
    flow = mflo[..., :2]
    flow = (flow.astype(np.float32) / 127.5 - 1) * absmax
    return flow


class FramesOnDisk(Dataset):

    def __init__(self, basename, device):
        super().__init__()
        self.basename = basename
        self.device = device
        self.write_queue = Queue()
        self.length = len(glob(f'{basename}*'))
        self.write_thread = WriteThread(self.write_queue, self.basename, daemon=True)
        self.write_thread.start()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]
        tensors = []
        for i in idx:
            file = f'{self.basename}{i}.jpg'
            if os.path.exists(file.replace('.jpg', '.mflo')):
                mflo = np.asarray(Image.open(file.replace('.jpg', '.mflo')))
                tensor = torch.tensor(decode_mflo(mflo))
            else:
                image = Image.open(file)
                tensor = to_tensor(image)
                if image.mode == 'RGB':
                    tensor = tensor.mul(2).sub(1)
            tensors.append(tensor)
        return torch.stack(tensors)

    def insert(self, item, idx=None):
        self.write_queue.put((item, idx if idx is not None else len(self)))
        self.length += 1

    def finalize(self):
        self.length -= 1
        self.write_thread.join()
        return self


def initialize_cache_files(names, out_name, device):
    os.makedirs(f'workspace/{out_name}', exist_ok=True)
    return easydict.EasyDict({name: FramesOnDisk(f'workspace/{out_name}/{name}', device) for name in names})


def sample(tensor, uv):
    height, width = tensor.shape[-2:]
    max_pos = torch.tensor([width - 1, height - 1], device=tensor.device).view(2, 1, 1)
    grid = uv.div(max_pos / 2).sub(1).movedim(0, -1).unsqueeze(0)
    return grid_sample(tensor.unsqueeze(0), grid, align_corners=True).squeeze(0)


@torch.no_grad()
def check_consistency(flow_forward, flow_backward):
    dev = flow_forward.device
    batch, height, width, two = flow_forward.shape
    flow_forward, flow_backward = flow_forward.permute(0, 3, 1, 2), flow_backward.permute(0, 3, 1, 2)
    dx_ker = torch.tensor([[[[0, 0, 0], [1, 0, -1], [0, 0, 0]]]], device=dev).float().div(2).repeat(2, 2, 1, 1)
    dy_ker = torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, -1, 0]]]], device=dev).float().div(2).repeat(2, 2, 1, 1)
    f_x = conv2d(flow_backward, dx_ker, padding='same')
    f_y = conv2d(flow_backward, dy_ker, padding='same')
    motionedge = torch.cat([f_x, f_y]).square().sum(dim=(0, 1))
    y, x = torch.meshgrid([torch.arange(0, height, device=dev), torch.arange(0, width, device=dev)], indexing='ij')
    p1 = torch.stack([x, y])
    v1 = flow_forward.squeeze(0)
    p0 = p1 + flow_backward.squeeze()
    v0 = sample(v1, p0)
    p1_back = p0 + v0
    v1_back = flow_backward.squeeze(0)
    r1 = torch.floor(p0)
    r2 = r1 + 1
    max_pos = torch.tensor([width - 1, height - 1], device=dev).view(2, 1, 1)
    min_pos = torch.tensor([0, 0], device=dev).view(2, 1, 1)
    overshoot = torch.logical_or(r1.lt(min_pos), r2.gt(max_pos))
    overshoot = torch.logical_or(overshoot[0], overshoot[1])
    missed = (p1_back - p1).square().sum(dim=0).ge(torch.stack([v1_back, v0]).square().sum(dim=(0, 1)).mul(0.01).add(0.5))
    motion_boundary = motionedge.ge(v1_back.square().sum(dim=0).mul(0.01).add(0.002))
    reliable = torch.ones((height, width), device=dev)
    reliable[motion_boundary] = 0
    reliable[missed] = -0.75
    reliable[overshoot] = 0
    mask = gaussian_blur(reliable.unsqueeze(0), 3).clip(0, 1)
    return mask


def check_consistency_np(flow1, flow2, edges_unreliable=True):
    flow1 = np.flip(flow1, axis=2)
    flow2 = np.flip(flow2, axis=2)
    h, w, _ = flow1.shape
    orig_coord = np.flip(np.mgrid[:w, :h], 0).T
    warp_coord = orig_coord + flow1
    warp_coord_inbound = np.zeros_like(warp_coord)
    warp_coord_inbound[..., 0] = np.clip(warp_coord[..., 0], 0, h - 2)
    warp_coord_inbound[..., 1] = np.clip(warp_coord[..., 1], 0, w - 2)
    warp_coord_floor = np.floor(warp_coord_inbound).astype(np.int)
    alpha = warp_coord_inbound - warp_coord_floor
    flow2_00 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1]]
    flow2_01 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1] + 1]
    flow2_10 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1]]
    flow2_11 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1] + 1]
    flow2_0_blend = (1 - alpha[..., 1, None]) * flow2_00 + alpha[..., 1, None] * flow2_01
    flow2_1_blend = (1 - alpha[..., 1, None]) * flow2_10 + alpha[..., 1, None] * flow2_11
    warp_coord_flow2 = (1 - alpha[..., 0, None]) * flow2_0_blend + alpha[..., 0, None] * flow2_1_blend
    rewarp_coord = warp_coord + warp_coord_flow2
    squared_diff = np.sum((rewarp_coord - orig_coord) ** 2, axis=2)
    threshold = 0.01 * np.sum(warp_coord_flow2 ** 2 + flow1 ** 2, axis=2) + 0.5
    reliable_flow = np.where(squared_diff >= threshold, -0.75, 1)
    if edges_unreliable:
        reliable_flow = np.where(np.logical_or.reduce((warp_coord[..., 0] < 0, warp_coord[..., 1] < 0, warp_coord[..., 0] >= h - 1, warp_coord[..., 1] >= w - 1)), 0, reliable_flow)
    dx = np.diff(flow1, axis=1, append=0)
    dy = np.diff(flow1, axis=0, append=0)
    motion_edge = np.sum(dx ** 2 + dy ** 2, axis=2)
    motion_threshold = 0.01 * np.sum(flow1 ** 2, axis=2) + 0.002
    reliable_flow = np.where(np.logical_and(motion_edge > motion_threshold, reliable_flow != -0.75), 0, reliable_flow)
    reliable_flow = scipy.ndimage.gaussian_filter(reliable_flow, [3, 3])
    reliable_flow = reliable_flow.clip(0, 1)
    return reliable_flow


def get_consistency_map(forward_flow, backward_flow, consistency='full'):
    if consistency == 'magnitude':
        reliable_flow = torch.sqrt(forward_flow[..., 0] ** 2 + forward_flow[..., 1] ** 2)
    elif consistency == 'full':
        reliable_flow = check_consistency(forward_flow, backward_flow)
    elif consistency == 'numpy':
        reliable_flow = torch.from_numpy(check_consistency_np(forward_flow.detach().cpu().numpy(), backward_flow.detach().cpu().numpy()))
    else:
        reliable_flow = torch.ones((forward_flow.shape[0], forward_flow.shape[1]))
    return reliable_flow


def luminance(tensor):
    return 0.2126 * tensor[:, :, 0] + 0.7152 * tensor[:, :, 1] + 0.0722 * tensor[:, :, 2]


def get_flow_model(which: List[str]=['farneback']):
    pred_fns = []
    if 'unflow' in which:
        pred_fns.append(sniklaus.get_prediction_fn('unflow'))
    if 'pwc' in which:
        pred_fns.append(sniklaus.get_prediction_fn('pwc'))
    if 'spynet' in which:
        pred_fns.append(sniklaus.get_prediction_fn('spynet'))
    if 'liteflownet' in which:
        pred_fns.append(sniklaus.get_prediction_fn('liteflownet'))
    for w in which:
        if w in mm.AVAILABLE_MODELS:
            pred_fns.append(mm.get_prediction_fn(w))
    if 'farneback' in which:
        pred_fns.append(lambda im1, im2: torch.from_numpy(cv2.calcOpticalFlowFarneback(luminance(im1.detach().squeeze().permute(1, 2, 0)).mul(255).byte().cpu().numpy(), luminance(im2.detach().squeeze().permute(1, 2, 0)).mul(255).byte().cpu().numpy(), flow=None, pyr_scale=0.8, levels=15, winsize=15, iterations=15, poly_n=7, poly_sigma=1.5, flags=10)).unsqueeze(0))
    if 'deepflow2' in which:
        raise Exception('deepflow2 not working quite yet...')
        models.append(lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2)))
    return lambda im1, im2: torch.mean(torch.stack([pred(im1, im2) for pred in pred_fns]), dim=0).float()


@torch.inference_mode()
def initialize_optical_flow(cache, init, consistency_trust, width, height, device):
    flow_model = get_flow_model()
    frames = VideoFrames(init, height=min(height, 512), width=min(width, 512), device=device)
    N = len(frames)
    if len(cache.flow) == N and tuple(cache.flow[0].shape[1:3]) == (height, width):
        None
        return
    else:
        cache.flow.length = cache.consistency.length = 0
    for f_n in trange(N, desc='Calculating optical flow...'):
        prev = frames[(f_n - 1) % N].add(1).div(2)
        curr = frames[f_n].add(1).div(2)
        forward = flow_model(curr, prev)
        backward = flow_model(prev, curr)
        maxflow = max(forward.shape[0], forward.shape[1])
        forward, backward = forward.clamp(-maxflow, maxflow), backward.clamp(-maxflow, maxflow)
        if consistency_trust > 0:
            consistency = get_consistency_map(forward, backward)
            consistency = interpolate(consistency.unsqueeze(1), (height, width), mode='bilinear')
            cache.consistency.insert(consistency)
        forward *= np.mean((width / forward.shape[1], height / forward.shape[2]))
        forward = interpolate(forward.permute(0, 3, 1, 2), (height, width), mode='bilinear').permute(0, 2, 3, 1)
        cache.flow.insert(forward)


def get_histogram(tensor):
    mu_h = tensor.mean(list(range(len(tensor.shape) - 1)))
    h = tensor - mu_h
    h = h.permute(0, 3, 1, 2).reshape(tensor.size(3), -1)
    Ch = torch.mm(h, h.T) / h.shape[1] + torch.finfo(tensor.dtype).eps * torch.eye(h.shape[0], device=tensor.device)
    return mu_h, h, Ch


def match_histogram(target_tensor, source_tensor, mode='avg'):
    if mode == 'False':
        return target_tensor
    backup = target_tensor.clone()
    try:
        if mode == 'avg':
            elementwise = True
            random_frame = False
        else:
            elementwise = False
            random_frame = True
        if not isinstance(source_tensor, list):
            source_tensor = [source_tensor]
        output_tensor = torch.zeros_like(target_tensor)
        for source in source_tensor:
            target = target_tensor.permute(0, 3, 2, 1)
            source = source.permute(0, 3, 2, 1)
            if elementwise:
                source = source.mean(0).unsqueeze(0)
            if random_frame:
                source = source[np.random.randint(0, source.shape[0])].unsqueeze(0)
            matched_tensor = torch.zeros_like(target)
            for idx in range(target.shape[0] if elementwise else 1):
                frame = target[idx].unsqueeze(0) if elementwise else target
                _, t, Ct = get_histogram(frame + 0.001 * torch.randn(size=frame.shape, device=frame.device))
                mu_s, _, Cs = get_histogram(source + 0.001 * torch.randn(size=source.shape, device=frame.device))
                eva_t, eve_t = torch.linalg.eigh(Ct, UPLO='U')
                Et = torch.sqrt(torch.diagflat(eva_t))
                Et[Et != Et] = 0
                Qt = torch.mm(torch.mm(eve_t, Et), eve_t.T)
                eva_s, eve_s = torch.linalg.eigh(Cs, UPLO='U')
                Es = torch.sqrt(torch.diagflat(eva_s))
                Es[Es != Es] = 0
                Qs = torch.mm(torch.mm(eve_s, Es), eve_s.T)
                ts = torch.mm(torch.mm(Qs, torch.inverse(Qt)), t)
                match = ts.reshape(*frame.permute(0, 3, 1, 2).shape).permute(0, 2, 3, 1)
                match += mu_s
                if elementwise:
                    matched_tensor[idx] = match
                else:
                    matched_tensor = match
            output_tensor += matched_tensor.permute(0, 3, 2, 1) / len(source_tensor)
    except RuntimeError:
        traceback.print_exc()
        None
        output_tensor = backup
    return output_tensor.clamp(min([s.min().item() for s in source_tensor]), max([s.max().item() for s in source_tensor]))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warp(x, f):
    return grid_sample(x, f, padding_mode='reflection', align_corners=False)


class VideoFlowDiffusionProcessor(torch.nn.Module):

    def forward(self, diffusion: BaseDiffusionProcessor, init: str, text: Optional[str]=None, image: Optional[str]=None, style: Optional[str]=None, size: Tuple[int]=(256, 256), first_skip: float=0.4, first_frame_init: Optional[str]=None, skip: float=0.7, blend: float=2, consistency_trust: float=0.75, wrap_around: int=0, turbo: int=1, noise_injection: float=0.02, flow_exaggeration: float=1.0, pre_hook: Optional[Callable]=None, post_hook: Optional[Callable]=None, hist_persist: bool=False, constant_seed: Optional[int]=None, device: str='cuda', preview: bool=False):
        height, width = [round64(s) for s in size]
        frames = VideoFrames(init, height, width, device)
        N = len(frames)
        cache = initialize_cache_files(names=['frame', 'flow', 'consistency'], out_name=build_output_name(init, unique=False), device=device)
        cache.frame.length = 0
        initialize_optical_flow(cache, init, consistency_trust, width, height, device)
        if first_frame_init is not None:
            out_img = ImagePrompt(path=first_frame_init, size=(height, width)).img
            cache.frame.insert(out_img)
            hist_img = out_img.clone()
        else:
            out_img = None
        loop_fade = torch.sqrt(torch.linspace(1, 0, wrap_around, device=device)).reshape(-1, 1, 1, 1)
        turbo_blend = torch.linspace(0, 1, turbo + 1, device=device)[1:]
        turbo_prev_img = turbo_next_img = None
        try:
            for f_n in trange(0, N + wrap_around + turbo, turbo, unit_scale=turbo):
                if constant_seed:
                    seed_everything(constant_seed)
                if f_n >= N + wrap_around:
                    turbo_next_img = cache.frame[f_n % N]
                if f_n > 0:
                    for t, f_t in enumerate(range(f_n - turbo, f_n)):
                        flow_map = flow_warp_map(cache.flow[f_t % N] * flow_exaggeration)
                        if turbo_prev_img is not None:
                            turbo_prev_img = warp(turbo_prev_img, flow_map)
                        if t != 0 and f_n < N + wrap_around:
                            turbo_next_img = warp(turbo_next_img, flow_map)
                        if turbo_prev_img is not None:
                            img = turbo_prev_img * (1.0 - turbo_blend[t]) + turbo_next_img * turbo_blend[t]
                        else:
                            img = turbo_next_img
                        cache.frame.insert(img, f_t % N)
                    out_img = turbo_next_img
                prompts = [ContentPrompt(frames[f_n % N])]
                if style is not None:
                    prompts.append(StylePrompt(path=style, size=(height, width)))
                if text is not None:
                    prompts.append(TextPrompt(text))
                if image is not None:
                    prompts.append(ImagePrompt(path=image))
                init_img = frames[f_n % N]
                if blend > 0:
                    if consistency_trust > 0:
                        flow_mask = cache.consistency[f_n % N]
                        flow_mask *= consistency_trust
                        flow_mask += 1 - consistency_trust
                    else:
                        flow_mask = torch.ones_like(init_img)
                    flow_mask *= blend
                    flow = flow_warp_map(cache.flow[f_n % N] * flow_exaggeration)
                    prev_img = frames[(f_n - 1) % N] if f_n == 0 else out_img
                    prev_warp = warp(prev_img, flow)
                    init_img += flow_mask * prev_warp
                    init_img /= 1 + flow_mask
                if f_n / N >= 1:
                    init_img = loop_fade[[f_n % N]] * init_img + (1 - loop_fade[[f_n % N]]) * cache.frame[f_n % N]
                if pre_hook:
                    init_img = pre_hook(init_img)
                if hist_persist and f_n > 0:
                    init_img = match_histogram(init_img, hist_img)
                init_img += noise_injection * torch.randn_like(init_img)
                out_img = diffusion.forward(init_img, prompts, first_skip if f_n == 0 else skip, verbose=False)
                if hist_persist and f_n == 0:
                    hist_img = out_img.clone()
                if post_hook:
                    out_img = post_hook(out_img)
                if preview:
                    plt.imshow(to_pil_image(out_img.squeeze().add(1).div(2).clamp(0, 1)))
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.5)
                cache.frame.insert(out_img, f_n % N)
                turbo_prev_img = turbo_next_img
                turbo_next_img = out_img
        except KeyboardInterrupt:
            None
        return cache.frame.finalize()


class GradModule(torch.nn.Module):

    def __init__(self, scale) ->None:
        super().__init__()
        self.scale = scale

    def set_targets(self, prompts):
        pass

    def forward(self, img, t):
        return torch.zeros_like(img)


def differentiable_histogram(x, weighting=None, nbins=255):
    B = x.shape[0]
    hist = torch.zeros(B, nbins, device=x.device)
    delta = 1 / (nbins - 1)
    bins = torch.arange(nbins + 1) * delta
    if weighting is None:
        weighting = torch.ones_like(x)
    for dim in range(nbins):
        bin_val_prev, bin_val, bin_val_next = bins[dim - 1], bins[dim] if dim > 0 else 0, bins[dim + 1]
        mask_sub = ((bin_val > x) & (x >= bin_val_prev)).float()
        mask_plus = ((bin_val_next > x) & (x >= bin_val)).float()
        hist[:, dim] += torch.sum(((x - bin_val_prev) * weighting * mask_sub).view(B, -1), dim=-1)
        hist[:, dim] += torch.sum(((bin_val_next - x) * weighting * mask_plus).view(B, -1), dim=-1)
    hist /= hist.sum(axis=-1, keepdim=True)
    return hist


class ColorMatchGrads(GradModule):

    def __init__(self, scale, saturation_weighting=True, bins=255) ->None:
        super().__init__(scale)
        self.bins = bins
        self.saturation_weighting = saturation_weighting

    def histogram(self, img):
        hue, sat, val = rgb_to_hsv(img.add(1).div(2).clamp(1e-08, 1 - 1e-08)).clamp(0, 1).unbind(1)
        if self.saturation_weighting:
            weighting = (sat * val).sqrt()
        else:
            weighting = None
        hist = differentiable_histogram(hue, weighting, self.bins)
        return hist

    def set_targets(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, StylePrompt):
                img, _ = prompt()
                self.register_buffer('target', self.histogram(img))

    def forward(self, img, t):
        loss = self.scale * mse_loss(self.histogram(img), self.target)
        grad = torch.autograd.grad(loss, img)[0]
        return grad


class NormalizeGradients(torch.autograd.Function):

    @staticmethod
    def forward(self, input_tensor):
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + torch.finfo(grad_input.dtype).eps)
        return grad_input, None


normalize_gradients = NormalizeGradients.apply


def normalize_weights(tensor, strategy: str):
    if strategy == 'elements':
        return tensor.numel()
    elif strategy == 'channels':
        return tensor.size(1)
    elif strategy == 'area':
        return tensor.size(2) * tensor.size(3)
    return 1


def scaled_mse_loss(input: torch.Tensor, target: torch.Tensor, eps: float=1e-08):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1."""
    diff = input - target
    return diff.pow(2).sum() / diff.abs().sum().add(eps)


def feature_loss(input: torch.Tensor, target: torch.Tensor, norm_weights: Optional[str]='elements', scaled: bool=True):
    if scaled:
        loss = scaled_mse_loss(input, target)
        loss /= normalize_weights(input, norm_weights)
    else:
        loss = mse_loss(input, target)
        loss /= normalize_weights(input, norm_weights)
        loss = normalize_gradients(loss)
    return loss


def gram_matrix(x, shift_x=0, shift_y=0, shift_t=0, flip_h=False, flip_v=False, use_covariance=False):
    B, C, H, W = x.size()
    if not (shift_x == 0 and shift_y == 0):
        x = x[:, :, shift_y:, shift_x:]
        y = x[:, :, :H - shift_y, :W - shift_x]
        B, C, H, W = x.size()
    if flip_h:
        y = x[:, :, :, ::-1]
    if flip_v:
        y = x[:, :, ::-1, :]
    else:
        y = x
    x_flat = x.reshape(B * C, H * W)
    y_flat = y.reshape(B * C, H * W)
    if use_covariance:
        x_flat = x_flat - x_flat.mean(1).unsqueeze(1)
        y_flat = y_flat - y_flat.mean(1).unsqueeze(1)
    return x_flat @ y_flat.T


class Perceptor(nn.Module):

    def __init__(self, content_strength, content_layers, style_strength, style_layers) ->None:
        super().__init__()
        self.content_layers, self.style_layers = content_layers, style_layers
        self.content_strength, self.style_strength = content_strength, style_strength
        self.embeddings = [None for _ in content_layers + style_layers]
        self.targets = None
        self.loss = 0

    def register_layer_hooks(self):
        c = -1
        for c, layer in enumerate(self.content_layers):

            def content_hook(module, input, output, l=c):
                embedding = output.squeeze().flatten(1)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += self.content_strength * feature_loss(embedding, self.targets[l])
            getattr(self.net, str(layer)).register_forward_hook(content_hook)
        for s, layer in enumerate(self.style_layers):

            def style_hook(module, input, output, l=c + 1 + s):
                embedding = gram_matrix(output)
                if self.targets is None:
                    self.embeddings[l] = embedding
                else:
                    self.loss += self.style_strength * feature_loss(embedding, self.targets[l])
            getattr(self.net, str(layer)).register_forward_hook(style_hook)

    def get_target_embeddings(self, contents=None, styles=None, content_weights=None, style_weights=None):
        if isinstance(contents, torch.Tensor):
            contents = [contents]
        content_embeddings = None
        if contents is not None:
            if content_weights is None:
                content_weights = torch.ones(len(contents))
            content_weights /= content_weights.sum()
            for content, content_weight in zip(contents, content_weights):
                if content_embeddings is None:
                    content_embeddings = content_weight * self.forward(content)[:len(self.content_layers)]
                else:
                    content_embeddings += content_weight * self.forward(content)[:len(self.content_layers)]
        style_embeddings = None
        if styles is not None:
            if style_weights is None:
                style_weights = torch.ones(len(styles))
            style_weights /= style_weights.sum()
            for style, style_weight in zip(styles, style_weights):
                if style_embeddings is None:
                    style_embeddings = style_weight * self.forward(style)[len(self.content_layers):]
                else:
                    style_embeddings += style_weight * self.forward(style)[len(self.content_layers):]
        if content_embeddings is None:
            return style_embeddings
        if style_embeddings is None:
            return content_embeddings
        return torch.cat((content_embeddings, style_embeddings))

    def forward(self, x):
        self.net(self.preprocess(x))
        return torch.nested_tensor(self.embeddings, device=x.device)

    def get_loss(self, x, targets):
        assert len(targets) == len(self.embeddings), f"The target embeddings don't match this perceptor's embeddings: {len(targets)}. Expected: {len(self.embeddings)}"
        self.loss = 0
        self.targets = targets
        self.forward(x)
        self.targets = None
        return self.loss


class Scale(nn.Module):

    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, input: Dict[str, torch.Tensor]):
        return self.module(input) * self.scale


class KBCPerceptor(Perceptor):
    """VGG network by Katherine Crowson"""
    poolings = {'max': nn.MaxPool2d, 'avg': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1.0, 'avg': 2.0, 'l2': 0.78}

    def __init__(self, content_layers=None, style_layers=None, content_strength=1, style_strength=1, pooling='max'):
        if content_layers is None:
            content_layers = [22]
        if style_layers is None:
            style_layers = [1, 6, 11, 20, 29]
        super().__init__(content_strength, content_layers, style_strength, style_layers)
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        net = models.vgg19(pretrained=True).features
        self.net = nn.Sequential(*list(net.children())[:max(content_layers + style_layers) + 1])
        self.net[0] = self._change_padding_mode(self.net[0], 'replicate')
        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.net):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                self.net[i] = Scale(self.poolings[pooling](2), pool_scale)
        self.net.eval()
        self.net.requires_grad_(False)
        self.register_layer_hooks()

    @staticmethod
    def _change_padding_mode(conv: nn.Conv2d, padding_mode: str):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding, padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv


default_layers = {'prune': {'content': [22], 'style': [3, 8, 15, 22, 29]}, 'nyud': {'content': [22], 'style': [3, 8, 15, 22, 29]}, 'fcn32s': {'content': [22], 'style': [3, 8, 15, 22, 29]}, 'sod': {'content': [22], 'style': [3, 8, 15, 22, 29]}, 'vgg16': {'content': [22], 'style': [3, 8, 15, 22, 29]}, 'vgg19': {'content': [26], 'style': [3, 8, 17, 26, 35]}, 'nin': {'content': [19], 'style': [5, 12, 19, 27]}}


class NIN(nn.Module):

    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        self.features = nn.Sequential(nn.Conv2d(3, 96, (11, 11), (4, 4)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(96, 96, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(256, 256, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(384, 384, (1, 1)), nn.ReLU(inplace=True), pool2d, nn.Dropout(0.5), nn.Conv2d(384, 1024, (3, 3), (1, 1), (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, (1, 1)), nn.ReLU(inplace=True), nn.Conv2d(1024, 1000, (1, 1)), nn.ReLU(inplace=True), nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True), nn.Softmax())


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))


class VGG_FCN32S(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, (7, 7)), nn.ReLU(True), nn.Dropout(0.5), nn.Conv2d(4096, 4096, (1, 1)), nn.ReLU(True), nn.Dropout(0.5))


class VGG_PRUNED(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5))


class VGG_SOD(nn.Module):

    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 100))


def build_sequential(channel_list, pooling):
    layers = []
    in_channels = 3
    for c in channel_list:
        if c == 'P':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2) if pooling == 'max' else nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, c, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


channel_list = {'VGG-16p': [24, 22, 'P', 41, 51, 'P', 108, 89, 111, 'P', 184, 276, 228, 'P', 512, 512, 512, 'P'], 'VGG-16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'], 'VGG-19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']}


def select_model(model_name, pooling):
    if 'prun' in model_name:
        model_file = 'modelzoo/vgg16-prune.pth'
        if not path.exists(model_file):
            gdown.download('https://drive.google.com/uc?id=1aaNqJ5D2A-vev3IZFv6dSkovuA3XwYsq', model_file)
        cnn = VGG_PRUNED(build_sequential(channel_list['VGG-16p'], pooling))
    elif 'nyud' in model_name:
        model_file = 'modelzoo/nyud-fcn32s-color-heavy.pth'
        if not path.exists(model_file):
            gdown.download('https://drive.google.com/uc?id=1MKj6Dntzh7t45PxM4I0ixWaQtisAg9hy', model_file)
        cnn = VGG_FCN32S(build_sequential(channel_list['VGG-16'], pooling))
    elif 'fcn32s' in model_name:
        model_file = 'modelzoo/fcn32s-heavy-pascal.pth'
        if not path.exists(model_file):
            gdown.download('https://drive.google.com/uc?id=1bcAnvfMuuEbJqjaVWIUCD9HUgD1fvxI_', model_file)
        cnn = VGG_FCN32S(build_sequential(channel_list['VGG-16'], pooling))
    elif 'sod' in model_name:
        model_file = 'modelzoo/vgg16-sod.pth'
        if not path.exists(model_file):
            gdown.download('https://drive.google.com/uc?id=1EU-F9ugeIeTO9ay4PinzsBXgEuCYBu0Z', model_file)
        cnn = VGG_SOD(build_sequential(channel_list['VGG-16'], pooling))
    elif 'vgg16' in model_name:
        model_file = 'modelzoo/vgg16.pth'
        if not path.exists(model_file):
            sd = load_url('https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth')
            map = {'classifier.1.weight': 'classifier.0.weight', 'classifier.1.bias': 'classifier.0.bias', 'classifier.4.weight': 'classifier.3.weight', 'classifier.4.bias': 'classifier.3.bias'}
            sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
            torch.save(sd, model_file)
        cnn = VGG(build_sequential(channel_list['VGG-16'], pooling))
    elif 'vgg19' in model_name:
        model_file = 'modelzoo/vgg19.pth'
        if not path.exists(model_file):
            sd = load_url('https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth')
            map = {'classifier.1.weight': 'classifier.0.weight', 'classifier.1.bias': 'classifier.0.bias', 'classifier.4.weight': 'classifier.3.weight', 'classifier.4.bias': 'classifier.3.bias'}
            sd = OrderedDict([(map[k] if k in map else k, v) for k, v in sd.items()])
            torch.save(sd, model_file)
        cnn = VGG(build_sequential(channel_list['VGG-19'], pooling))
    elif 'nin' in model_name:
        model_file = 'modelzoo/nin.pth'
        if not path.exists(model_file):
            download('https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth', model_file)
        cnn = NIN(pooling)
    else:
        raise ValueError('Model architecture not recognized.')
    state_dict = torch.load(model_file)
    cnn.load_state_dict(state_dict)
    for param in cnn.parameters():
        param.requires_grad = False
    return cnn.features


class PGGPerceptor(Perceptor):
    """VGG networks ported by ProGamerGov from Caffe for his amazing style transfer implementation neural-style-pt"""

    def __init__(self, content_layers=None, style_layers=None, content_strength=1, style_strength=1, model_name='vgg19', pooling='max'):
        if content_layers is None:
            content_layers = default_layers[model_name]['content']
        if style_layers is None:
            style_layers = default_layers[model_name]['style']
        super().__init__(content_strength, content_layers, style_strength, style_layers)
        net = select_model(model_name.lower(), pooling)
        self.net = nn.Sequential(*list(net.children())[:max(content_layers + style_layers) + 1])
        mean_pixel = torch.tensor([103.939, 116.779, 123.68]) / 255
        self.preprocess = lambda x: 255 * (x[:, [2, 1, 0]] - mean_pixel.reshape(1, 3, 1, 1))
        self.register_layer_hooks()


def load_perceptor(name: str) ->Perceptor:
    if name.startswith('pgg'):
        return partial(PGGPerceptor, model_name=name.replace('pgg-', ''))
    if name.startswith('kbc'):
        return KBCPerceptor


class VGGGrads(GradModule):

    def __init__(self, scale=1, perceptor='kbc') ->None:
        super().__init__(scale)
        self.perceptor = load_perceptor(perceptor)(content_strength=0, style_strength=scale, **dict(content_layers=[]))

    def set_targets(self, prompts):
        device = next(self.perceptor.parameters()).device
        for prompt in prompts:
            if isinstance(prompt, StylePrompt):
                img, _ = prompt()
                img = img
                self.register_buffer('target_embeddings', self.perceptor.get_target_embeddings(None, [img.add(1).div(2)]))

    def forward(self, img, t):
        loss = self.perceptor.get_loss(img.add(1).div(2), self.target_embeddings)
        grad = torch.autograd.grad(loss, img)[0]
        return grad


class Cutouts(nn.Module):

    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomAffine(degrees=15, translate=(0.1, 0.1)), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomPerspective(distortion_scale=0.4, p=0.7), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomGrayscale(p=0.15), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01)])

    def forward(self, input, t):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1).normal_(mean=0.8, std=0.3).clip(float(self.cut_size / max_size), 1.0))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resize(cutout, (self.cut_size, self.cut_size)))
            del cutout
        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class DangoCutouts(nn.Module):

    def __init__(self, cut_size, cut_overview=[12] * 400 + [4] * 600, cut_innercut=[4] * 400 + [12] * 600, cut_pow=1, cut_icgray_p=[0.2] * 400 + [0] * 600, animation_mode='Video Input', skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cut_overview = cut_overview
        self.cut_innercut = cut_innercut
        self.cut_pow = cut_pow
        self.cut_icgray_p = cut_icgray_p
        self.skip_augs = skip_augs
        if animation_mode == 'None':
            self.augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomGrayscale(p=0.1), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
        elif animation_mode == 'Video Input':
            self.augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomAffine(degrees=15, translate=(0.1, 0.1)), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomPerspective(distortion_scale=0.4, p=0.7), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomGrayscale(p=0.15), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01)])
        elif animation_mode == '2D' or animation_mode == '3D':
            self.augs = T.Compose([T.RandomHorizontalFlip(p=0.4), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.RandomGrayscale(p=0.1), T.Lambda(lambda x: x + torch.randn_like(x) * 0.01), T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3)])

    def forward(self, input, t):
        overview = self.cut_overview[999 - t]
        inner_crop = self.cut_innercut[999 - t]
        ic_grey_p = self.cut_icgray_p[999 - t]
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(input, ((sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2), mode='reflect')
        cutout = resize(pad_input, out_shape=output_shape)
        if overview > 0:
            if overview <= 4:
                if overview >= 1:
                    cutouts.append(cutout)
                if overview >= 2:
                    cutouts.append(gray(cutout))
                if overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(overview):
                    cutouts.append(cutout)
        if inner_crop > 0:
            for i in range(inner_crop):
                size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(ic_grey_p * inner_crop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        if not self.skip_augs:
            cutouts = self.augs(cutouts)
        return cutouts


def random_cutouts(input, cut_size=224, cutn=32, cut_pow=1.0):
    sideY, sideX = input.shape[-2:]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, cut_size)
    if sideY < sideX:
        size = sideY
        tops = torch.zeros(cutn // 4, dtype=int)
        lefts = torch.linspace(0, sideX - size, cutn // 4, dtype=int)
    else:
        size = sideX
        tops = torch.linspace(0, sideY - size, cutn // 4, dtype=int)
        lefts = torch.zeros(cutn // 4, dtype=int)
    cutouts = []
    for offsety, offsetx in zip(tops, lefts):
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutouts.append(resize(cutout, out_shape=(cut_size, cut_size)))
    for _ in range(cutn - len(cutouts)):
        size = (torch.rand([]) ** cut_pow * max_size).clamp(min_size, max_size).round().long().item()
        loc = torch.randint(0, (sideX - size + 1) * (sideY - size + 1), ())
        offsety, offsetx = torch.div(loc, sideX - size + 1, rounding_mode='floor'), loc % (sideX - size + 1)
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutouts.append(resize(cutout, out_shape=(cut_size, cut_size)))
    return torch.cat(cutouts)


class MauaCutouts(nn.Module):

    def __init__(self, cut_size, cutn, pow_gain=16.0):
        super().__init__()
        self.cut_size, self.cutn, self.pow_gain = cut_size, cutn, pow_gain

    def forward(self, input, t):
        pow = self.pow_gain ** ((500 - t) / 500)
        return random_cutouts(input, self.cut_size, self.cutn, pow)


def make_cutouts(cutouts, cut_size, cutn, **cutout_kwargs):
    if cutouts == 'normal':
        return Cutouts(cut_size, cutn, **cutout_kwargs)
    elif cutouts == 'maua':
        return MauaCutouts(cut_size, cutn, **cutout_kwargs)
    elif cutouts == 'dango':
        return DangoCutouts(cut_size, **cutout_kwargs)
    else:
        raise Exception(f'Cutouts {cutouts} not recognized!')


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class CLIPGrads(GradModule):

    def __init__(self, scale=1, perceptors=['ViT-B/16'], cutouts='maua', cutout_kwargs=dict(cutn=32), cutout_batches=8, clamp_gradient=None):
        super().__init__(scale)
        self.clip_models = torch.nn.ModuleList([clip.load(name, jit=False)[0].eval().requires_grad_(False) for name in perceptors])
        self.normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.cutouts = torch.nn.ModuleList([make_cutouts(cutouts, cut_size=clip_model.visual.input_resolution, **cutout_kwargs) for clip_model in self.clip_models])
        self.cutout_batches = cutout_batches
        self.clamp_gradient = clamp_gradient

    def set_targets(self, prompts):
        target_embeds, weights = [[] for _ in range(len(self.clip_models))], []
        device = next(self.clip_models[0].parameters()).device
        for prompt in prompts:
            if isinstance(prompt, TextPrompt):
                txt, weight = prompt()
                tokens = clip.tokenize(txt, truncate=True)
                weights.append(weight)
                for c, clip_model in enumerate(self.clip_models):
                    target_embeds[c].append(clip_model.encode_text(tokens).float())
            elif isinstance(prompt, StylePrompt):
                img, weight = prompt()
                img = img
                for _ in range(self.cutout_batches):
                    for c, clip_model in enumerate(self.clip_models):
                        im_cuts = clip_model.encode_image(self.normalize(self.cutouts[c](img, t=0))).float()
                        target_embeds[c].append(im_cuts)
                    weights.extend([0.5 * weight / im_cuts.shape[0]] * im_cuts.shape[0])
        for c, target_embed in enumerate(target_embeds):
            self.clip_models[c].register_buffer('target', torch.cat(target_embed).unsqueeze(0))
        weights = torch.tensor(weights, device=device, dtype=torch.float)
        if weights.sum().abs() < 0.001:
            raise RuntimeError('The weights must not sum to 0.')
        weights /= weights.sum().abs()
        self.register_buffer('weights', weights)

    def forward(self, img, t):
        grad = torch.zeros_like(img)
        for c, clip_model in enumerate(self.clip_models):
            for _ in range(self.cutout_batches):
                image_embeds = clip_model.encode_image(self.normalize(self.cutouts[c](img.add(1).div(2), t[[0]].long()))).float()
                dists = spherical_dist_loss(image_embeds.unsqueeze(1), clip_model.target)
                loss = dists.view((-1, img.shape[0], dists.shape[-1])).mul(self.weights).sum(2).mean(0)
                grad += torch.autograd.grad(loss.sum() * self.scale, img)[0] / self.cutout_batches
        if self.clamp_gradient:
            magnitude = grad.square().mean().sqrt()
            grad *= magnitude.clamp(max=self.clamp_gradient) / magnitude
        return grad


class LossGrads(GradModule):

    def __init__(self, loss_fn, scale=1):
        super().__init__(scale)
        self.loss_fn = loss_fn

    def forward(self, img, t):
        loss = self.loss_fn(img).sum() * self.scale
        grad = torch.autograd.grad(loss, img)[0]
        return grad


class LPIPSGrads(GradModule):

    def __init__(self, scale=1):
        super().__init__(scale)
        self.lpips_model = lpips.LPIPS(net='vgg', verbose=False)

    def set_targets(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, ContentPrompt):
                img, _ = prompt()
                self.register_buffer('target', img)

    def forward(self, img, t):
        if hasattr(self, 'target'):
            loss = self.lpips_model(resample(img, 256), resample(self.target, 256)).sum() * self.scale
            grad = torch.autograd.grad(loss, img)[0]
        else:
            grad = torch.zeros_like(img)
        return grad


class LatentSSIMGrads(GradModule):

    def __init__(self, scale, model) ->None:
        super().__init__(scale)
        self.ssim = SSIM(data_range=10, channel=4)
        self.model = model

    def set_targets(self, prompts):
        for prompt in prompts:
            if isinstance(prompt, ContentPrompt):
                img, _ = prompt()
                latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(img))
                self.register_buffer('target', latent)

    def forward(self, x, t):
        loss = 1 - self.ssim(x, self.target)
        grad = torch.autograd.grad(loss, x)[0]
        return grad


ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])


def perchannel_conv(x, filters):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def perception(x):
    filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
    return perchannel_conv(x, filters)


class CA(torch.nn.Module):

    def __init__(self, chn=12, hidden_n=96):
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn * 4, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5):
        y = perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        udpate_mask = (torch.rand(b, 1, h, w) + update_rate).floor()
        return x + y * udpate_mask

    def seed(self, n, sz=128):
        return torch.zeros(n, self.chn, sz, sz)


class Parameterization(nn.Module):

    def __init__(self, height, width, tensor, ema=False, decay=0.99):
        super().__init__()
        self.h, self.w = height, width
        self.tensor = nn.Parameter(tensor)
        self.ema = ema
        if ema:
            self.decay = decay
            self.register_buffer('biased', torch.zeros_like(tensor))
            self.register_buffer('average', torch.zeros_like(tensor))
            self.register_buffer('accum', torch.tensor(1.0))
            self.update_ema()

    def encode(self, tensor: torch.Tensor) ->None:
        raise NotImplementedError()

    def decode(self, tensor: torch.Tensor=None) ->torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def update_ema(self):
        if self.ema:
            self.accum.mul_(self.decay)
            self.biased.mul_(self.decay)
            self.biased.add_((1 - self.decay) * self.tensor)
            self.average.copy_(self.biased)
            self.average.div_(1 - self.accum)

    @torch.no_grad()
    def reset_ema(self):
        if self.ema:
            self.biased.set_(torch.zeros_like(self.biased))
            self.average.set_(torch.zeros_like(self.average))
            self.accum.set_(torch.ones_like(self.accum))
            self.update_ema()

    def decode_average(self):
        if self.ema:
            return self.decode(self.average)
        return self.decode()


class HdrLoss(nn.Module):

    def __init__(self, pallet_size, n_pallets, gamma=2.5, weight=0.15, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.register_buffer('comp', torch.linspace(0, 1, pallet_size).pow(gamma).view(pallet_size, 1).repeat(1, n_pallets))
        self.register_buffer('weight', torch.as_tensor(weight))

    def forward(self, input):
        if isinstance(input, Pixel):
            pallet = input.sort_pallet()
            magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
            color_norms = torch.linalg.vector_norm(pallet * magic_color.sqrt(), dim=-1)
            loss_raw = F.mse_loss(color_norms, self.comp)
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return 'HDR normalization'


def break_tensor(tensor):
    floors = tensor.floor().long()
    ceils = tensor.ceil().long()
    rounds = tensor.round().long()
    fracs = tensor - floors
    return floors, ceils, rounds, fracs


class ReplaceGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class Pixel(Parameterization):
    """
    Differentiable image format for pixel art images
    Implementation adapted from PyTTI 5 https://github.com/sportsracer48/pytti/blob/p5/Image/PixelImage.py
    """

    def __init__(self, width, height, scale, pallet_size, n_pallets, gamma=1, hdr_weight=0.5, norm_weight=0.1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__(width * scale, height * scale)
        self.pallet_inertia = 2
        pallet = torch.linspace(0, self.pallet_inertia, pallet_size).pow(gamma).view(pallet_size, 1, 1).repeat(1, n_pallets, 3)
        self.pallet = nn.Parameter(pallet)
        self.pallet_size = pallet_size
        self.n_pallets = n_pallets
        self.value = nn.Parameter(torch.zeros(height, width))
        self.tensor = nn.Parameter(torch.zeros(n_pallets, height, width))
        self.output_axes = 'n', 's', 'y', 'x'
        self.latent_strength = 0.1
        self.scale = scale
        self.hdr_loss = HdrLoss(pallet_size, n_pallets, gamma, hdr_weight) if hdr_weight != 0 else None
        self.loss = PalletLoss(n_pallets, norm_weight)
        self.register_buffer('pallet_target', torch.empty_like(self.pallet))
        self.use_pallet_target = False

    def clone(self):
        width, height = self.image_shape
        dummy = Pixel(width // self.scale, height // self.scale, self.scale, self.pallet_size, self.n_pallets, hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight), norm_weight=float(self.loss.weight))
        with torch.no_grad():
            dummy.value.set_(self.value.clone())
            dummy.tensor.set_(self.tensor.clone())
            dummy.pallet.set_(self.pallet.clone())
            dummy.pallet_target.set_(self.pallet_target.clone())
            dummy.use_pallet_target = self.use_pallet_target
        return dummy

    def set_pallet_target(self, pil_image):
        if pil_image is None:
            self.use_pallet_target = False
            return
        dummy = self.clone()
        dummy.use_pallet_target = False
        dummy.encode_image(pil_image)
        with torch.no_grad():
            self.pallet_target.set_(dummy.sort_pallet())
            self.pallet.set_(self.pallet_target.clone())
            self.use_pallet_target = True

    @torch.no_grad()
    def lock_pallet(self, lock=True):
        if lock:
            self.pallet_target.set_(self.sort_pallet().clone())
        self.use_pallet_target = lock

    def image_loss(self):
        return [x for x in [self.hdr_loss, self.loss] if x is not None]

    def sort_pallet(self):
        if self.use_pallet_target:
            return self.pallet_target
        pallet = (self.pallet / self.pallet_inertia).clamp_(0, 1)
        magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
        color_norms = pallet.square().mul_(magic_color).sum(dim=-1)
        pallet_indices = color_norms.argsort(dim=0).T
        pallet = torch.stack([pallet[i][:, j] for j, i in enumerate(pallet_indices)], dim=1)
        return pallet

    def get_image_tensor(self):
        return torch.cat([self.value.unsqueeze(0), self.tensor])

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        self.value.set_(tensor[0])
        self.tensor.set_(tensor[1:])

    def decode_tensor(self):
        width, height = self.image_shape
        pallet = self.sort_pallet()
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)
        pallet_weights = self.tensor.movedim(0, 2)
        pallets = F.one_hot(pallet_weights.argmax(dim=2), num_classes=self.n_pallets)
        pallet_weights = pallet_weights.softmax(dim=2).unsqueeze(-1)
        pallets = pallets.unsqueeze(-1)
        colors_disc = pallet[value_rounds]
        colors_disc = (colors_disc * pallets).sum(dim=2)
        colors_disc = F.interpolate(colors_disc.movedim(2, 0).unsqueeze(0), (height, width), mode='nearest')
        colors_cont = pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(colors_cont.movedim(2, 0).unsqueeze(0), (height, width), mode='nearest')
        return replace_grad(colors_disc, colors_cont * 0.5 + colors_disc * 0.5)

    @torch.no_grad()
    def render_value_image(self):
        width, height = self.image_shape
        values = self.value.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3)
        array = np.array(values.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_pallet(self):
        pallet = self.sort_pallet()
        width, height = self.n_pallets * 16, self.pallet_size * 32
        array = np.array(pallet.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_channel(self, pallet_i):
        width, height = self.image_shape
        pallet = self.sort_pallet()
        pallet[:, :pallet_i, :] = 0.5
        pallet[:, pallet_i + 1:, :] = 0.5
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)
        pallet_weights = self.tensor.movedim(0, 2)
        pallets = F.one_hot(pallet_weights.argmax(dim=2), num_classes=self.n_pallets)
        pallet_weights = pallet_weights.softmax(dim=2).unsqueeze(-1)
        colors_cont = pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(colors_cont.movedim(2, 0).unsqueeze(0), (height, width), mode='nearest')
        tensor = named_rearrange(colors_cont, self.output_axes, ('y', 'x', 's'))
        array = np.array(tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8))[:, :, :]
        return Image.fromarray(array)

    @torch.no_grad()
    def update(self):
        self.pallet.clamp_(0, self.pallet_inertia)
        self.value.clamp_(0, 1)
        self.tensor.clamp_(0, float('inf'))

    def encode_image(self, pil_image, smart_encode=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        width, height = self.image_shape
        scale = self.scale
        color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref)
        with torch.no_grad():
            magic_color = self.pallet.new_tensor([[[0.299]], [[0.587]], [[0.114]]])
            value_ref = torch.linalg.vector_norm(color_ref * magic_color.sqrt(), dim=0)
            self.value.set_(value_ref)
        if smart_encode:
            mse = HSVLoss.TargetImage('HSV loss', self.image_shape, pil_image)
            if self.hdr_loss is not None:
                before_weight = self.hdr_loss.weight.detach()
                self.hdr_loss.set_weight(0.01)
            guide = DirectImageGuide(self, None, optimizer=optim.Adam([self.pallet, self.tensor], lr=0.1))
            guide.run_steps(201, [], [], [mse])
            if self.hdr_loss is not None:
                self.hdr_loss.set_weight(before_weight)

    @torch.no_grad()
    def encode_random(self, random_pallet=False):
        self.value.uniform_()
        self.tensor.uniform_()
        if random_pallet:
            self.pallet.uniform_(to=self.pallet_inertia)


class PalletLoss(nn.Module):

    def __init__(self, n_pallets, weight=0.15, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.n_pallets = n_pallets
        self.register_buffer('weight', torch.as_tensor(weight))

    def forward(self, input):
        if isinstance(input, Pixel):
            tensor = input.tensor.movedim(0, -1).contiguous().view(-1, self.n_pallets).softmax(dim=-1)
            N, n = tensor.shape
            mu = tensor.mean(dim=0, keepdim=True)
            sigma = tensor.std(dim=0, keepdim=True)
            tensor = tensor.sub(mu)
            S = (tensor.transpose(0, 1) @ tensor).div(sigma * sigma.transpose(0, 1) * N)
            S.sub_(torch.diag(S.diagonal()))
            loss_raw = S.mean()
            loss_raw.add_(sigma.mul(N).pow(-1).mean())
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return 'Palette normalization'


class ClampWithGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def to_colorspace(tensor, colorspace):
    if colorspace == 'rgb':
        return tensor
    else:
        raise NotImplementedError()


class RGB(Parameterization):

    def __init__(self, height, width, tensor=None, colorspace='rgb', ema=False):
        if tensor is None:
            tensor = torch.empty(1, 3, height, width).uniform_().mul(0.1)
        Parameterization.__init__(self, height, width, tensor, ema)
        self.colorspace = colorspace

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        return clamp_with_grad(tensor, 0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(to_colorspace(tensor.clamp(0, 1), self.colorspace).data)

    def forward(self):
        return self.decode()


def maybe_download_vqgan(model_dir):
    if model_dir == 'imagenet_1024':
        config_path, checkpoint_path = 'modelzoo/vqgan_imagenet_f16_1024.yaml', 'modelzoo/vqgan_imagenet_f16_1024.ckpt'
        if not os.path.exists(checkpoint_path):
            download('http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml', config_path)
            download('http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt', checkpoint_path)
    elif model_dir == 'imagenet_16384':
        config_path, checkpoint_path = 'modelzoo/vqgan_imagenet_f16_16384.yaml', 'modelzoo/vqgan_imagenet_f16_16384.ckpt'
        if not os.path.exists(checkpoint_path):
            download('http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml', config_path)
            download('http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt', checkpoint_path)
    elif model_dir == 'coco':
        config_path, checkpoint_path = 'modelzoo/coco.yaml', 'modelzoo/coco.ckpt'
        if not os.path.exists(checkpoint_path):
            download('https://dl.nmkd.de/ai/clip/coco/coco.yaml', config_path)
            download('https://dl.nmkd.de/ai/clip/coco/coco.ckpt', checkpoint_path)
    elif model_dir == 'faceshq':
        config_path, checkpoint_path = 'modelzoo/faceshq.yaml', 'modelzoo/faceshq.ckpt'
        if not os.path.exists(checkpoint_path):
            download('https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT', config_path)
            download('https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt', checkpoint_path)
    elif model_dir == 'wikiart_1024':
        config_path, checkpoint_path = 'modelzoo/wikiart_1024.yaml', 'modelzoo/wikiart_1024.ckpt'
        if not os.path.exists(checkpoint_path):
            download('http://mirror.io.community/blob/vqgan/wikiart.yaml', config_path)
            download('http://mirror.io.community/blob/vqgan/wikiart.ckpt', checkpoint_path)
    elif model_dir == 'wikiart_16384':
        config_path, checkpoint_path = 'modelzoo/wikiart_16384.yaml', 'modelzoo/wikiart_16384.ckpt'
        if not os.path.exists(checkpoint_path):
            download('http://mirror.io.community/blob/vqgan/wikiart_16384.yaml', config_path)
            download('http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt', checkpoint_path)
    elif model_dir == 'sflckr':
        config_path, checkpoint_path = 'modelzoo/sflckr.yaml', 'modelzoo/sflckr.ckpt'
        if not os.path.exists(checkpoint_path):
            download('https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1', config_path)
            download('https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1', checkpoint_path)
    else:
        config_path = sorted(glob(model_dir + '/*.yaml'), reverse=True)[0]
        checkpoint_path = sorted(glob(model_dir + '/*.ckpt'), reverse=True)[0]
    return config_path, checkpoint_path


def load_vqgan_model(model_dir):
    config_path, checkpoint_path = maybe_download_vqgan(model_dir)
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown vqgan type: {config.model.target}')
    del model.loss
    return model


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]) @ codebook
    return replace_grad(x_q, x)


class VQGAN(Parameterization):

    def __init__(self, height, width, tensor=None, vqgan_model='imagenet_16384', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ema=False):
        if tensor is None:
            tensor = torch.empty(1, 3, height, width, device=device).uniform_()
        model = load_vqgan_model(vqgan_model)
        tensor = model.encode(tensor.clamp(0, 1) * 2 - 1)[0]
        Parameterization.__init__(self, tensor.shape[2], tensor.shape[3], tensor, ema)
        self.model = model
        self.codebook = self.model.quantize.embedding.weight

    def decode(self, tensor=None):
        if tensor is None:
            tensor = self.tensor
        tens_q = vector_quantize(self.tensor.movedim(1, 3), self.codebook).movedim(3, 1)
        return clamp_with_grad(self.model.decode(tens_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def encode(self, tensor):
        self.tensor.set_(self.model.encode(tensor.clamp(0, 1) * 2 - 1)[0].data)

    def forward(self):
        return self.decode()


class PadIm2Video(torch.nn.Module):

    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__()
        self.time_dim = time_dim
        assert ntimes > 0
        assert pad_type in ['zero', 'repeat']
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(self.time_dim)
        if x.shape[self.time_dim] == 1:
            if self.pad_type == 'repeat':
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == 'zero':
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = torch.nn.functional.pad(x, padarg)
        return x


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


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = nn.functional.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, attn_target, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, non_skip_wt=1.0, non_skip_wt_learnable=False, layer_scale_type=None, layer_scale_init_value=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_type = layer_scale_type
        if non_skip_wt_learnable is False:
            self.non_skip_wt = non_skip_wt
        else:
            self.non_skip_wt = nn.Parameter((torch.ones(1) * non_skip_wt).squeeze())
        if self.layer_scale_type is not None:
            assert non_skip_wt == 1.0 and non_skip_wt_learnable is False
            assert self.layer_scale_type in ['per_channel', 'scalar'], f'Found Layer scale type {self.layer_scale_type}'
            if self.layer_scale_type == 'per_channel':
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == 'scalar':
                gamma_shape = [1, 1, 1]
            self.layer_scale_gamma1 = nn.Parameter(torch.ones(size=gamma_shape) * layer_scale_init_value, requires_grad=True)
            self.layer_scale_gamma2 = nn.Parameter(torch.ones(size=gamma_shape) * layer_scale_init_value, requires_grad=True)

    def forward(self, x):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.non_skip_wt)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.non_skip_wt)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale_gamma2)
        return x

    def extra_repr(self) ->str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)
        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(p[1].requires_grad) + ')\n'
        return string_repr


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_layout = 1, img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = np.prod(self.patches_layout)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedConv(nn.Module):

    def __init__(self, conv_param_list, img_size=224, patch_size=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_layout = 1, img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = np.prod(self.patches_layout)
        layers = []
        for idx, k in enumerate(conv_param_list):
            conv = nn.Conv2d(k['input_channels'], k['output_channels'], kernel_size=k['kernel_size'], stride=k['stride'], padding=k['padding'], bias=k['bias'])
            layers.append(conv)
            if idx != len(conv_param_list) - 1:
                if k['norm'] == 'bn':
                    norm = nn.BatchNorm2d(k['output_channels'])
                    layers.append(norm)
                elif k['norm'] == 'lnfp32':
                    norm = Fp32GroupNorm(1, k['output_channels'])
                    layers.append(norm)
                elif k['norm'] == 'ln':
                    norm = nn.GroupNorm(1, k['output_channels'])
                    layers.append(norm)
                if k['act'] == 'relu':
                    act = nn.ReLU(inplace=True)
                    layers.append(act)
                elif k['act'] == 'gelu':
                    act = nn.GELU()
                    layers.append(act)
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem, img_size):
        super().__init__()
        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            self.proj = proj_stem[0]
        assert isinstance(img_size, list) and len(img_size) >= 3, 'Need the full C[xT]xHxW in generic'
        with torch.no_grad():
            dummy_img = torch.zeros([1] + img_size)
            self.patches_layout = tuple(self.proj(dummy_img).shape[2:])
            self.num_patches = np.prod(self.patches_layout)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, mlp_ratio, attn_target, drop_rate, drop_path_rate, drop_path_type, force_cast_ln_fp32, classifier_feature, use_cls_token, learnable_pos_embed, non_skip_wt, non_skip_wt_learnable, layer_scale_type, layer_scale_init_value, patch_embed_type, patch_embed_params_list, layer_norm_eps=1e-06, masked_image_modeling=False, patch_drop_min_patches=-1, patch_drop_max_patches=-1, patch_drop_at_eval=False, add_pos_same_dtype=False, patch_dropping=False, post_encoder_params=None, decoder=None, mask_token_embed_dim=None):
        super().__init__()
        assert use_cls_token or classifier_feature == 'global_pool'
        self.masked_image_modeling = masked_image_modeling
        self.patch_drop_min_patches = patch_drop_min_patches
        self.patch_drop_max_patches = patch_drop_max_patches
        self.patch_drop_at_eval = patch_drop_at_eval
        self.add_pos_same_dtype = add_pos_same_dtype
        self.patch_dropping = patch_dropping
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        if force_cast_ln_fp32:
            norm_layer = partial(Fp32LayerNorm, eps=layer_norm_eps)
        self.num_features = self.embed_dim = embed_dim
        assert classifier_feature in ['cls_token', 'global_pool']
        self.classifier_feature = classifier_feature
        assert in_chans == 3, 'Only 3 channels supported'
        if patch_embed_type == 'linear':
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        elif patch_embed_type == 'conv':
            self.patch_embed = PatchEmbedConv(conv_param_list=patch_embed_params_list, img_size=img_size, patch_size=patch_size)
        elif patch_embed_type == 'generic':
            self.patch_embed = PatchEmbedGeneric(patch_embed_params_list, img_size=img_size)
        num_patches = self.patch_embed.num_patches
        assert self.patch_embed.patches_layout[-1] == self.patch_embed.patches_layout[-2], 'Interpolation of pos embed not supported for non-square layouts'
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.first_patch_idx = 1
            total_num_patches = num_patches + 1
        else:
            self.cls_token = None
            self.first_patch_idx = 0
            total_num_patches = num_patches
        if learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, total_num_patches, embed_dim))
        else:
            self.register_buffer('pos_embed', get_sinusoid_encoding_table(total_num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        assert drop_path_type in ['progressive', 'uniform'], f'Drop path types are: [progressive, uniform]. Got {drop_path_type}.'
        if drop_path_type == 'progressive':
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == 'uniform':
            dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, attn_target=attn_target, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, non_skip_wt=non_skip_wt, non_skip_wt_learnable=non_skip_wt_learnable, layer_scale_type=layer_scale_type, layer_scale_init_value=layer_scale_init_value) for i in range(depth)])
        self.post_encoder = None
        if post_encoder_params is not None:
            self.post_encoder = hydra.utils.instantiate(post_encoder_params, _convert_='all')
        if self.patch_dropping and decoder is None:
            self.decoder = None
        if mask_token_embed_dim is None:
            mask_token_embed_dim = embed_dim
        if decoder is not None:
            self.decoder = decoder(first_patch_idx=self.first_patch_idx, patches_layout=self.patch_embed.patches_layout, embed_dim=mask_token_embed_dim)
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        if learnable_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
        if self.patch_dropping and patch_embed_type == 'linear':
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)
        if self.masked_image_modeling:
            assert self.patch_drop_max_patches == -1
            self.mask_token = nn.Parameter(torch.zeros(1, mask_token_embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.patch_dropping:
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, Fp32LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patch_drop(self, x, npatch_per_img, patch_start_idx=1, npatch_to_keep=None):
        """
        Randomly drop patches from the input
        Input:
            - x: B x N x C
        Returns:
            - y: B x N' x C where N' is sampled from [self.patch_drop_min_patches, self.patch_drop_max_patches]
        """
        if self.patch_drop_min_patches < 0 or self.patch_drop_min_patches == npatch_per_img or npatch_to_keep is not None and npatch_to_keep < 0:
            return x
        if self.training is False and self.patch_drop_at_eval is False:
            return x
        rnd_inds = [torch.randperm(npatch_per_img, device=x.device) for _ in range(x.shape[0])]
        if npatch_to_keep is None:
            npatch_to_keep = torch.randint(low=self.patch_drop_min_patches, high=self.patch_drop_max_patches, size=(1,)).item()
        class_tokens = x[:, :patch_start_idx, ...]
        patch_tokens = x[:, patch_start_idx:, ...]
        patch_tokens = [patch_tokens[i, rnd_inds[i][:npatch_to_keep]] for i in range(x.shape[0])]
        patch_tokens = torch.stack(patch_tokens)
        x = torch.cat([class_tokens, patch_tokens], dim=1)
        return x

    def masked_patch_drop(self, x, mask):
        mask = mask.view(x.shape[0], -1)
        cls_token, patches = x[:, :self.first_patch_idx], x[:, self.first_patch_idx:]
        patches = patches[~mask].reshape(x.shape[0], -1, patches.shape[-1])
        x = torch.cat([cls_token, patches], dim=1)
        return x

    def apply_mask(self, x, mask):
        mask = mask.view(x.shape[0], -1)
        x[mask, :] = self.mask_token
        return x

    def insert_masks(self, x, mask):
        embed_dim = x.shape[-1]
        mask = mask.view(x.shape[0], -1)
        B, N = mask.shape
        tmp = torch.empty(B, N, embed_dim)
        tmp[mask] = self.mask_token
        tmp[~mask] = x[:, self.first_patch_idx:].reshape(-1, x.shape[-1])
        x = torch.cat([x[:, :self.first_patch_idx], tmp], dim=1)
        return x

    def prepare_tokens(self, x, npatch_to_keep, mask):
        B = x.shape[0]
        input_shape = x.shape
        x = self.patch_embed(x)
        npatch_per_img = x.shape[1]
        if self.patch_dropping is False and mask is not None:
            x = self.apply_mask(x, mask)
        if self.cls_token is not None:
            class_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((class_tokens, x), dim=1)
        pos_embed = self.get_pos_embedding(npatch_per_img, self.pos_embed, self.patch_embed.patches_layout, input_shape, first_patch_idx=self.first_patch_idx)
        if self.add_pos_same_dtype:
            pos_embed = pos_embed.type_as(x)
        x = x + pos_embed
        if self.patch_dropping and mask is not None:
            x = self.masked_patch_drop(x, mask)
        else:
            x = self.patch_drop(x, npatch_per_img, patch_start_idx=self.first_patch_idx, npatch_to_keep=npatch_to_keep)
        x = self.pos_drop(x)
        return x

    @classmethod
    def get_pos_embedding(cls, npatch_per_img, pos_embed, patches_layout, input_shape, first_patch_idx=1):
        pos_embed = cls.interpolate_pos_encoding(npatch_per_img, pos_embed, patches_layout, input_shape=input_shape, first_patch_idx=first_patch_idx)
        return pos_embed

    def forward_features(self, x, npatch_to_keep, mask=None, use_checkpoint=False):
        assert npatch_to_keep is None
        if mask is not None and isinstance(mask, list) and not all(mask):
            mask = None
        orig_input_shape = x.shape
        x = self.prepare_tokens(x, npatch_to_keep, mask=mask)
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.classifier_feature == 'cls_token' and (mask is None or self.decoder is None):
            assert self.first_patch_idx == 1, 'Must have a CLS token at 0'
            x = x[:, 0]
        elif self.classifier_feature == 'global_pool' and (mask is None or self.decoder is None):
            x = x[:, self.first_patch_idx:, ...].mean(dim=1)
        elif self.patch_dropping and mask is not None and self.decoder is not None:
            x = self.norm(x)
            if self.post_encoder:
                x_dtype = x.dtype
                x = self.post_encoder(x)
            x = self.insert_masks(x, mask)
            if self.first_patch_idx == 0:
                cls_token = None
            else:
                cls_token = x[:, self.first_patch_idx]
            x = self.decoder(x, orig_input_shape, self.pos_embed, use_checkpoint=use_checkpoint)
            if isinstance(x, list):
                decoder_patch_features = [el[:, self.first_patch_idx:] for el in x]
            else:
                decoder_patch_features = x[:, self.first_patch_idx:]
            return cls_token, decoder_patch_features
        elif mask is not None:
            pass
        x = self.norm(x)
        return self.pre_logits(x)

    def get_intermediate_features(self, x, names, npatch_to_keep, mask, use_checkpoint=False):
        interms = []
        x = self.prepare_tokens(x, npatch_to_keep, mask=mask)
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            interms.append(self.norm(x))
        if self.post_encoder:
            assert len(names) == 1 and names[0] in ['last_all']
            interms.append(self.post_encoder(interms[-1]))
        output = []
        for name in names:
            if name.startswith('blkCLS'):
                assert self.first_patch_idx == 1, 'Must have CLS token at 0'
                v = int(name.replace('blkCLS', ''))
                output.append(interms[v][:, 0])
            elif name.startswith('concatCLS'):
                assert self.first_patch_idx == 1, 'Must have CLS token at 0'
                v = int(name.replace('concatCLS', ''))
                feat = torch.cat([x[:, 0] for x in interms[-v:]], dim=-1)
                output.append(feat)
            elif name == 'lastCLS':
                assert self.first_patch_idx == 1, 'Must have CLS token at 0'
                output.append(interms[-1][:, 0])
            elif name == 'last_all':
                output.append(interms[-1])
            elif name == 'last_patch_avg':
                output.append(interms[-1][:, self.first_patch_idx:, ...].mean(dim=1))
        return output

    def forward(self, x: torch.Tensor, out_feat_keys: List[str]=None, npatch_to_keep: int=None, mask: torch.Tensor=None, use_checkpoint: bool=False) ->List[torch.Tensor]:
        if out_feat_keys is None or len(out_feat_keys) == 0:
            x = self.forward_features(x, npatch_to_keep, mask=mask, use_checkpoint=use_checkpoint)
            if not isinstance(x, tuple):
                x = x.unsqueeze(0)
            else:
                x = [x]
        else:
            x = self.get_intermediate_features(x, out_feat_keys, npatch_to_keep, mask=mask, use_checkpoint=use_checkpoint)
        return x

    @staticmethod
    def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
        N = pos_embed.shape[1]
        if N == target_spatial_size:
            return pos_embed
        dim = pos_embed.shape[-1]
        pos_embed = nn.functional.interpolate(pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=math.sqrt(target_spatial_size / N), mode='bicubic')
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    @classmethod
    def interpolate_pos_encoding(cls, npatch_per_img, pos_embed, patches_layout, input_shape=None, first_patch_idx=1):
        assert first_patch_idx == 0 or first_patch_idx == 1, 'there is 1 CLS token or none'
        N = pos_embed.shape[1] - first_patch_idx
        if npatch_per_img == N:
            return pos_embed
        class_emb = pos_embed[:, :first_patch_idx]
        pos_embed = pos_embed[:, first_patch_idx:]
        if input_shape is None or patches_layout[0] == 1:
            pos_embed = cls.interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
        elif patches_layout[0] > 1:
            assert len(input_shape) == 4, 'temporal interpolation not supported'
            num_frames = patches_layout[0]
            num_spatial_tokens = patches_layout[1] * patches_layout[2]
            pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
            pos_embed = cls.interpolate_pos_encoding_2d(npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0))
        else:
            raise ValueError("This type of interpolation isn't implemented")
        return torch.cat((class_emb, pos_embed), dim=1)

    def get_layer_id(self, layer_name):
        num_layers = self.get_num_layers()
        if layer_name in ['cls_token', 'pos_embed']:
            return 0
        elif layer_name.find('patch_embed') != -1:
            return 0
        elif layer_name.find('blocks') != -1:
            return int(layer_name.split('blocks')[1].split('.')[1]) + 1
        else:
            return num_layers

    def get_num_layers(self):
        return len(self.blocks) + 1


class Decoder(nn.Module):

    def __init__(self, first_patch_idx, patches_layout, attn_target, embed_dim, decoder_embed_dim=512, decoder_depth=8, drop_path_rate=0.0, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, layer_norm_eps=1e-06, non_skip_wt=1.0, return_interim_layers=False, share_pos_embed=False, learnable_pos_embed=True, init_pos_embed_random=False, non_skip_wt_learnable=False, layer_scale_type=None, layer_scale_init_value=0.0001, final_projection=None, pos_sum_embed_only=False, **kwargs):
        super().__init__()
        self.patches_layout = patches_layout
        self.first_patch_idx = first_patch_idx
        assert first_patch_idx == 0 or first_patch_idx == 1
        self.share_pos_embed = share_pos_embed
        self.build_pos_embedding(share_pos_embed=share_pos_embed, learnable_pos_embed=learnable_pos_embed, patches_layout=patches_layout, first_patch_idx=first_patch_idx, embed_dim=embed_dim, init_pos_embed_random=init_pos_embed_random)
        self.pos_sum_embed_only = pos_sum_embed_only
        if pos_sum_embed_only:
            assert decoder_depth == -1, 'Do not specify decoder_depth'
            return
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(decoder_embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([Block(dim=decoder_embed_dim, attn_target=attn_target, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, non_skip_wt=non_skip_wt, non_skip_wt_learnable=non_skip_wt_learnable, layer_scale_type=layer_scale_type, layer_scale_init_value=layer_scale_init_value) for i in range(decoder_depth)])
        self.return_interim_layers = return_interim_layers
        self.final_projection = None
        if final_projection is not None:
            self.final_projection = hydra.utils.instantiate(final_projection, _convert_='all', _recursive_=False)

    def build_pos_embedding(self, share_pos_embed, learnable_pos_embed, patches_layout, first_patch_idx, embed_dim, init_pos_embed_random):
        if share_pos_embed is True:
            self.pos_embed = None
        elif learnable_pos_embed is True:
            self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(patches_layout) + first_patch_idx, embed_dim))
            if init_pos_embed_random:
                trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer('pos_embed', get_sinusoid_encoding_table(np.prod(patches_layout) + first_patch_idx, embed_dim))

    def forward(self, x, orig_input_shape, input_pos_embed=None, use_checkpoint=False):
        curr_pos_embed = input_pos_embed if self.share_pos_embed else self.pos_embed
        pos_embed = VisionTransformer.get_pos_embedding(x.size(1) - self.first_patch_idx, curr_pos_embed, self.patches_layout, input_shape=orig_input_shape, first_patch_idx=self.first_patch_idx)
        x = x + pos_embed
        if self.pos_sum_embed_only:
            return x
        x = self.decoder_embed(x)
        interim = []
        for i, blk in enumerate(self.decoder_blocks):
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if self.return_interim_layers or i == len(self.decoder_blocks) - 1:
                interim.append(x)
        interim = [self.norm(el) for el in interim]
        if self.final_projection is not None:
            interim = [self.final_projection(el) for el in interim]
        if self.return_interim_layers:
            return interim
        return interim[-1]


class PosEmbedSumDecoder(Decoder):

    def __init__(self, first_patch_idx, patches_layout, embed_dim, share_pos_embed=False, learnable_pos_embed=True, init_pos_embed_random=False):
        super().__init__(pos_sum_embed_only=True, decoder_depth=-1, attn_target=None, first_patch_idx=first_patch_idx, patches_layout=patches_layout, embed_dim=embed_dim, share_pos_embed=share_pos_embed, learnable_pos_embed=learnable_pos_embed, init_pos_embed_random=init_pos_embed_random)


class TransformerBlocks(nn.Module):

    def __init__(self, attn_target, embed_dim, num_blocks, drop_path_rate=0.0, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, non_skip_wt=1.0, attn_drop_rate=0.0, layer_norm_eps=1e-06, non_skip_wt_learnable=False, layer_scale_type=None, layer_scale_init_value=0.0001):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, attn_target=attn_target, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, non_skip_wt=non_skip_wt, non_skip_wt_learnable=non_skip_wt_learnable, layer_scale_type=layer_scale_type, layer_scale_init_value=layer_scale_init_value) for i in range(num_blocks)])

    def forward(self, x, use_checkpoint=False):
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x


class OmniMAE(nn.Module):

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, imgOrVideo, mask=None):
        outputs = self.trunk(imgOrVideo, mask=mask)
        if mask is not None:
            return self.head(outputs[0][1])
        else:
            return self.head(outputs[0])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseDiffusionProcessor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cutouts,
     lambda: ([], {'cut_size': 4, 'cutn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Decoder,
     lambda: ([], {'first_patch_idx': 0, 'patches_layout': 4, 'attn_target': _mock_layer, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeepConvolutionalDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DeepConvolutionalGenerator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 100])], {}),
     True),
    (DiscriminatorR1Penalty,
     lambda: ([], {'batch_size': 4, 'image_size': 4, 'r1_gamma': 4, 'r1_interval': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EMAFade,
     lambda: ([], {'fade_frames': 4}),
     lambda: ([torch.rand([4, 4]), 0, 0], {}),
     False),
    (EmergingConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FourierFeatures,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fp32GroupNorm,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FullyConnectedLayer,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneratorPathLengthRegularization,
     lambda: ([], {'pl_weight': 4, 'pl_interval': 4, 'pl_decay': 4, 'pl_batch_shrink': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GradModule,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HdrLoss,
     lambda: ([], {'pallet_size': 4, 'n_pallets': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleLeakyReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InvertibleSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Layer,
     lambda: ([], {'x': _mock_layer(), 'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LoopNoise,
     lambda: ([], {'loop_len': 4, 'size': 4, 'smooth': 4}),
     lambda: ([], {}),
     True),
    (Loss,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiPrototypes,
     lambda: ([], {'output_dim': 4, 'nmb_prototypes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormalLatentDistribution,
     lambda: ([], {'batch_size': 4, 'z_dim': 4}),
     lambda: ([], {}),
     True),
    (PadIm2Video,
     lambda: ([], {'ntimes': 4, 'pad_type': 'zero'}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PalletLoss,
     lambda: ([], {'n_pallets': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PosEmbedSumDecoder,
     lambda: ([], {'first_patch_idx': 0, 'patches_layout': 4, 'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RGB,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([], {}),
     False),
    (Scale,
     lambda: ([], {'module': _mock_layer(), 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Stack,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (TempoLoopNoise,
     lambda: ([], {'tempo': 4, 'n_bars': 4, 'fps': 4, 'size': 4, 'smooth': 4}),
     lambda: ([], {}),
     True),
    (TextPrompt,
     lambda: ([], {'text': 4}),
     lambda: ([], {}),
     True),
    (ToRGBLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'w_dim': 4}),
     lambda: ([torch.rand([16, 4, 1, 1]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_maua_maua_maua_maua(_paritybench_base):
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

