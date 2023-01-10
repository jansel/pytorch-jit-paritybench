import sys
_module = sys.modules[__name__]
del sys
avg_checkpoints = _module
benchmark = _module
bulk_runner = _module
clean_checkpoint = _module
convert_from_mxnet = _module
convert_nest_flax = _module
generate_readmes = _module
hubconf = _module
inference = _module
generate_csv_results = _module
setup = _module
tests = _module
test_layers = _module
test_models = _module
test_optim = _module
test_utils = _module
timm = _module
data = _module
auto_augment = _module
config = _module
constants = _module
dataset = _module
dataset_factory = _module
distributed_sampler = _module
loader = _module
mixup = _module
random_erasing = _module
readers = _module
class_map = _module
img_extensions = _module
reader = _module
reader_factory = _module
reader_hfds = _module
reader_image_folder = _module
reader_image_in_tar = _module
reader_image_tar = _module
reader_tfds = _module
reader_wds = _module
shared_count = _module
real_labels = _module
tf_preprocessing = _module
transforms = _module
transforms_factory = _module
layers = _module
activations = _module
activations_jit = _module
activations_me = _module
adaptive_avgmax_pool = _module
attention_pool2d = _module
blur_pool = _module
bottleneck_attn = _module
cbam = _module
classifier = _module
cond_conv2d = _module
config = _module
conv2d_same = _module
conv_bn_act = _module
create_act = _module
create_attn = _module
create_conv2d = _module
create_norm = _module
create_norm_act = _module
drop = _module
eca = _module
evo_norm = _module
fast_norm = _module
filter_response_norm = _module
gather_excite = _module
global_context = _module
halo_attn = _module
helpers = _module
inplace_abn = _module
lambda_layer = _module
linear = _module
median_pool = _module
mixed_conv2d = _module
ml_decoder = _module
mlp = _module
non_local_attn = _module
norm = _module
norm_act = _module
padding = _module
patch_embed = _module
pool2d_same = _module
pos_embed = _module
pos_embed_rel = _module
pos_embed_sincos = _module
selective_kernel = _module
separable_conv = _module
space_to_depth = _module
split_attn = _module
split_batchnorm = _module
squeeze_excite = _module
std_conv = _module
test_time_pool = _module
trace_utils = _module
weight_init = _module
loss = _module
asymmetric_loss = _module
binary_cross_entropy = _module
cross_entropy = _module
jsd = _module
models = _module
_builder = _module
_efficientnet_blocks = _module
_efficientnet_builder = _module
_factory = _module
_features = _module
_features_fx = _module
_helpers = _module
_hub = _module
_manipulate = _module
_pretrained = _module
_prune = _module
_registry = _module
beit = _module
byoanet = _module
byobnet = _module
cait = _module
coat = _module
convit = _module
convmixer = _module
convnext = _module
crossvit = _module
cspnet = _module
deit = _module
densenet = _module
dla = _module
dpn = _module
edgenext = _module
efficientformer = _module
efficientnet = _module
factory = _module
features = _module
fx_features = _module
gcvit = _module
ghostnet = _module
gluon_resnet = _module
gluon_xception = _module
hardcorenas = _module
hrnet = _module
hub = _module
inception_resnet_v2 = _module
inception_v3 = _module
inception_v4 = _module
levit = _module
maxxvit = _module
mlp_mixer = _module
mobilenetv3 = _module
mobilevit = _module
mvitv2 = _module
nasnet = _module
nest = _module
nfnet = _module
pit = _module
pnasnet = _module
poolformer = _module
pvt_v2 = _module
registry = _module
regnet = _module
res2net = _module
resnest = _module
resnet = _module
resnetv2 = _module
rexnet = _module
selecsls = _module
senet = _module
sequencer = _module
sknet = _module
swin_transformer = _module
swin_transformer_v2 = _module
swin_transformer_v2_cr = _module
tnt = _module
tresnet = _module
twins = _module
vgg = _module
visformer = _module
vision_transformer = _module
vision_transformer_hybrid = _module
vision_transformer_relpos = _module
volo = _module
vovnet = _module
xception = _module
xception_aligned = _module
xcit = _module
optim = _module
adabelief = _module
adafactor = _module
adahessian = _module
adamp = _module
adamw = _module
adan = _module
lamb = _module
lars = _module
lookahead = _module
madgrad = _module
nadam = _module
nvnovograd = _module
optim_factory = _module
radam = _module
rmsprop_tf = _module
sgdp = _module
scheduler = _module
cosine_lr = _module
multistep_lr = _module
plateau_lr = _module
poly_lr = _module
scheduler = _module
scheduler_factory = _module
step_lr = _module
tanh_lr = _module
utils = _module
agc = _module
checkpoint_saver = _module
clip_grad = _module
cuda = _module
decay_batch = _module
distributed = _module
jit = _module
log = _module
metrics = _module
misc = _module
model = _module
model_ema = _module
random = _module
summary = _module
version = _module
train = _module
validate = _module

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


import logging


import time


from collections import OrderedDict


from functools import partial


import torch.nn as nn


import torch.nn.parallel


import numpy as np


import pandas as pd


import math


import functools


from copy import deepcopy


from torch.testing._internal.common_utils import TestCase


from torch.autograd import Variable


from torch.nn.modules.batchnorm import BatchNorm2d


from torchvision.ops.misc import FrozenBatchNorm2d


from typing import Optional


import torch.utils.data as data


from torchvision.datasets import CIFAR100


from torchvision.datasets import CIFAR10


from torchvision.datasets import MNIST


from torchvision.datasets import KMNIST


from torchvision.datasets import FashionMNIST


from torchvision.datasets import ImageFolder


from torch.utils.data import Sampler


import torch.distributed as dist


import random


from itertools import repeat


from typing import Callable


import torch.utils.data


from itertools import islice


from typing import Any


from typing import Dict


from typing import List


from typing import Tuple


from torch.utils.data import Dataset


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


import numbers


import warnings


from typing import Sequence


import torchvision.transforms.functional as F


from torchvision import transforms


from torch import nn as nn


from torch.nn import functional as F


import torch.nn.functional as F


from typing import Union


from typing import Type


import types


from torch import nn


from torch import Tensor


from torch.nn.modules.transformer import _get_activation_fn


from torch.nn.init import _calculate_fan_in_and_fan_out


from torch.hub import load_state_dict_from_url


import re


from collections import defaultdict


from torch.hub import HASH_REGEX


from torch.hub import download_url_to_file


from torch.hub import urlparse


import collections.abc


from itertools import chain


from torch.utils.checkpoint import checkpoint


import torch.hub


import torch.utils.checkpoint as cp


from torch.jit.annotations import List


import torch.utils.checkpoint as checkpoint


from functools import reduce


from math import ceil


from typing import cast


import torch.utils.checkpoint


import torch.jit


from torch.optim.optimizer import Optimizer


from torch.optim import Optimizer


from typing import TYPE_CHECKING


import torch.optim


import torch.optim as optim


from torch.optim.optimizer import required


import abc


from abc import ABC


from torch import distributed as dist


import torchvision.utils


from torch.nn.parallel import DistributedDataParallel as NativeDDP


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """

    def __init__(self, inplace: bool=False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return F.gelu(input)


class GELUTanh(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """

    def __init__(self, inplace: bool=False):
        super(GELUTanh, self).__init__()

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return F.gelu(input, approximate='tanh')


def hard_mish(x, inplace: bool=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


def hard_sigmoid(x, inplace: bool=False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_swish(x, inplace: bool=False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def mish(x, inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """

    def __init__(self, inplace: bool=False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """

    def __init__(self, num_parameters: int=1, init: float=0.25, inplace: bool=False) ->None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        return F.prelu(input, self.weight)


class Sigmoid(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def swish(x, inplace: bool=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class Tanh(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


_has_hardsigmoid = 'hardsigmoid' in dir(torch.nn.functional)


_has_hardswish = 'hardswish' in dir(torch.nn.functional)


_has_mish = 'mish' in dir(torch.nn.functional)


_has_silu = 'silu' in dir(torch.nn.functional)


_ACT_LAYER_DEFAULT = dict(silu=nn.SiLU if _has_silu else Swish, swish=nn.SiLU if _has_silu else Swish, mish=nn.Mish if _has_mish else Mish, relu=nn.ReLU, relu6=nn.ReLU6, leaky_relu=nn.LeakyReLU, elu=nn.ELU, prelu=PReLU, celu=nn.CELU, selu=nn.SELU, gelu=GELU, gelu_tanh=GELUTanh, sigmoid=Sigmoid, tanh=Tanh, hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoid, hard_swish=nn.Hardswish if _has_hardswish else HardSwish, hard_mish=HardMish)


@torch.jit.script
def hard_mish_jit(x, inplace: bool=False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


class HardSigmoidJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool=False):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


class HardSwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


@torch.jit.script
def mish_jit(x, _inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class MishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def swish_jit(x, inplace: bool=False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


class SwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


_ACT_LAYER_JIT = dict(silu=nn.SiLU if _has_silu else SwishJit, swish=nn.SiLU if _has_silu else SwishJit, mish=nn.Mish if _has_mish else MishJit, hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoidJit, hard_swish=nn.Hardswish if _has_hardswish else HardSwishJit, hard_mish=HardMishJit)


@torch.jit.script
def hard_mish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.0)
    m = torch.where((x >= -2.0) & (x <= 0.0), x + 1.0, m)
    return grad_output * m


@torch.jit.script
def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJitAutoFn(torch.autograd.Function):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_jit_bwd(x, grad_output)


class HardMishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishJitAutoFn.apply(x)


@torch.jit.script
def hard_sigmoid_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.0) & (x <= 3.0)) / 6.0
    return grad_output * m


@torch.jit.script
def hard_sigmoid_jit_fwd(x, inplace: bool=False):
    return (x + 3).clamp(min=0, max=6).div(6.0)


class HardSigmoidJitAutoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(x, grad_output)


class HardSigmoidMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidJitAutoFn.apply(x)


@torch.jit.script
def hard_swish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.0)
    m = torch.where((x >= -3.0) & (x <= 3.0), x / 3.0 + 0.5, m)
    return grad_output * m


@torch.jit.script
def hard_swish_jit_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(x, grad_output)

    @staticmethod
    def symbolic(g, self):
        input = g.op('Add', self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
        hardtanh_ = g.op('Clip', input, g.op('Constant', value_t=torch.tensor(0, dtype=torch.float)), g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        hardtanh_ = g.op('Div', hardtanh_, g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        return g.op('Mul', self, hardtanh_)


class HardSwishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


class MishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    def symbolic(g, x):
        return g.op('Mul', x, g.op('Sigmoid', x))

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


class SwishMe(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


_ACT_LAYER_ME = dict(silu=nn.SiLU if _has_silu else SwishMe, swish=nn.SiLU if _has_silu else SwishMe, mish=nn.Mish if _has_mish else MishMe, hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoidMe, hard_swish=nn.Hardswish if _has_hardswish else HardSwishMe, hard_mish=HardMishMe)


_EXPORTABLE = False


def is_exportable():
    return _EXPORTABLE


_NO_JIT = False


def is_no_jit():
    return _NO_JIT


_SCRIPTABLE = False


def is_scriptable():
    return _SCRIPTABLE


def get_act_layer(name: Union[Type[nn.Module], str]='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if not isinstance(name, str):
        return name
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_LAYER_JIT:
            return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name: Union[nn.Module, str], inplace=None, **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        return act_layer(**kwargs)


class MLP(nn.Module):

    def __init__(self, act_layer='relu', inplace=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.act = create_act_layer(act_layer, inplace=inplace)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    if len(size) != 2:
        raise ValueError(error_msg)
    return size


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) ->torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = int(output_size), int(output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = output_size[0], output_size[0]
    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size
    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [(crop_width - image_width) // 2 if crop_width > image_width else 0, (crop_height - image_height) // 2 if crop_height > image_height else 0, (crop_width - image_width + 1) // 2 if crop_width > image_width else 0, (crop_height - image_height + 1) // 2 if crop_height > image_height else 0]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg='Please provide only two dimensions (h, w) for size.')
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}(size={self.size})'


class FastAdaptiveAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3), keepdim=not self.flatten)


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'pool_type=' + self.pool_type + ', flatten=' + str(self.flatten) + ')'


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    return x * cos_emb + rot(x) * sin_emb


def inv_freq_bands(num_bands: int, temperature: float=100000.0, step: int=2, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None) ->torch.Tensor:
    inv_freq = 1.0 / temperature ** (torch.arange(0, num_bands, step, dtype=dtype, device=device) / num_bands)
    return inv_freq


def pixel_freq_bands(num_bands: int, max_freq: float=224.0, linear_bands: bool=True, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=dtype, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=dtype, device=device)
    return bands * torch.pi


def build_fourier_pos_embed(feat_shape: List[int], bands: Optional[torch.Tensor]=None, num_bands: int=64, max_res: int=224, linear_bands: bool=False, include_grid: bool=False, concat_out: bool=True, in_pixels: bool=True, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None) ->List[torch.Tensor]:
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands, dtype=dtype, device=device)
        else:
            bands = inv_freq_bands(num_bands, step=1, dtype=dtype, device=device)
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype
    if in_pixels:
        grid = torch.stack(torch.meshgrid([torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype) for s in feat_shape]), dim=-1)
    else:
        grid = torch.stack(torch.meshgrid([torch.arange(s, device=device, dtype=dtype) for s in feat_shape]), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands
    pos_sin, pos_cos = pos.sin(), pos.cos()
    out = (grid, pos_sin, pos_cos) if include_grid else (pos_sin, pos_cos)
    if concat_out:
        out = torch.cat(out, dim=-1)
    return out


def build_rotary_pos_embed(feat_shape: List[int], bands: Optional[torch.Tensor]=None, dim: int=64, max_freq: float=224, linear_bands: bool=False, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None):
    """
    NOTE: shape arg should include spatial dim only
    """
    feat_shape = torch.Size(feat_shape)
    sin_emb, cos_emb = build_fourier_pos_embed(feat_shape, bands=bands, num_bands=dim // 4, max_res=max_freq, linear_bands=linear_bands, concat_out=False, device=device, dtype=dtype)
    N = feat_shape.numel()
    sin_emb = sin_emb.reshape(N, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(N, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self, dim, max_res=224, linear_bands: bool=False):
        super().__init__()
        self.dim = dim
        self.register_buffer('bands', pixel_freq_bands(dim // 4, max_res, linear_bands=linear_bands), persistent=False)

    def get_embed(self, shape: List[int]):
        return build_rotary_pos_embed(shape, self.bands)

    def forward(self, x):
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


def _trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

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
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class RotAttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """

    def __init__(self, in_features: int, out_features: int=None, embed_dim: int=None, num_heads: int=4, qkv_bias: bool=True):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed = RotaryEmbedding(self.head_dim)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        qc, q = q[:, :, :1], q[:, :, 1:]
        sin_emb, cos_emb = self.pos_embed.get_embed((H, W))
        q = apply_rot_embed(q, sin_emb, cos_emb)
        q = torch.cat([qc, q], dim=2)
        kc, k = k[:, :, :1], k[:, :, 1:]
        k = apply_rot_embed(k, sin_emb, cos_emb)
        k = torch.cat([kc, k], dim=2)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class AttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """

    def __init__(self, in_features: int, feat_size: Union[int, Tuple[int, int]], out_features: int=None, embed_dim: int=None, num_heads: int=4, qkv_bias: bool=True):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        spatial_dim = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


def get_padding(kernel_size, stride, dilation=1):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling

    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride

    Returns:
        torch.Tensor: the transformed tensor.
    """

    def __init__(self, channels, filt_size=3, stride=2) ->None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4
        coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = F.pad(x, self.padding, 'reflect')
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, height, width, dim)
        rel_k: (2 * window - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size + 1) // 2
    x = q @ rel_k.transpose(-1, -2)
    x = x.reshape(-1, W, rel_size)
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, rel_size - W])
    x_pad = x_pad.reshape(-1, W + 1, rel_size)
    x = x_pad[:, :W, win_size - 1:]
    x = x.reshape(B, H, 1, W, win_size).expand(-1, -1, win_size, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    """

    def __init__(self, block_size, win_size, dim_head, scale):
        """
        Args:
            block_size (int): block size
            win_size (int): neighbourhood window size
            dim_head (int): attention head dim
            scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        self.height_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)
        self.width_rel = nn.Parameter(torch.randn(win_size * 2 - 1, dim_head) * scale)

    def forward(self, q):
        B, BB, HW, _ = q.shape
        q = q.reshape(-1, self.block_size, self.block_size, self.dim_head)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))
        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, BB, HW, -1)
        return rel_logits


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class BottleneckAttn(nn.Module):
    """ Bottleneck Attention
    Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        stride (int): output stride of the module, avg pool used if stride == 2 (default: 1).
        num_heads (int): parallel attention heads (default: 4)
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=None, qk_ratio=1.0, qkv_bias=False, scale_pos_embed=False):
        super().__init__()
        assert feat_size is not None, 'A concrete feature size matching expected input (H, W) is required'
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.qkv = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.pos_embed.height, '')
        _assert(W == self.pos_embed.width, '')
        x = self.qkv(x)
        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale
        else:
            attn = q @ k * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)
        out = self.pool(out)
        return out


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(ChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.amax((2, 3), keepdim=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightChannelAttn, self).__init__(channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.amax((2, 3), keepdim=True)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * F.sigmoid(x_attn)


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: List[int], s: List[int], d: List[int]=(1, 1), value: float=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor]=None, stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), dilation: Tuple[int, int]=(1, 1), groups: int=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def get_condconv_initializer(initializer, num_experts, expert_shape):

    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if len(weight.shape) != 2 or weight.shape[0] != num_experts or weight.shape[1] != num_params:
            raise ValueError('CondConv variables must have shape [num_experts, num_params]')
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


def is_static_pad(kernel_size: int, stride: int=1, dilation: int=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def get_padding_value(padding, kernel_size, **kwargs) ->Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))
        if bias:
            self.bias_shape = self.out_channels,
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.reshape(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            self.add_module(str(idx), create_conv2d_pad(in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, groups=conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs
        if 'groups' in kwargs:
            groups = kwargs.pop('groups')
            if groups == in_channels:
                kwargs['depthwise'] = True
            else:
                assert groups == 1
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None, device=None, dtype=None):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, **factory_kwargs)
        except TypeError:
            super(BatchNormAct2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(x, self.running_mean if not self.training or self.track_running_stats else None, self.running_var if not self.training or self.track_running_stats else None, self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


def _num_groups(num_channels, num_groups, group_size):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


def fast_group_norm(x: torch.Tensor, num_groups: int, weight: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None, eps: float=1e-05) ->torch.Tensor:
    if torch.jit.is_scripting():
        return F.group_norm(x, num_groups, weight, bias, eps)
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_gpu_dtype()
        x, weight, bias = x, weight, bias
    with torch.amp.autocast(enabled=False):
        return F.group_norm(x, num_groups, weight, bias, eps)


_USE_FAST_NORM = False


def is_fast_norm():
    return _USE_FAST_NORM


class GroupNormAct(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=32, eps=1e-05, affine=True, group_size=None, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(GroupNormAct, self).__init__(_num_groups(num_channels, num_groups, group_size), num_channels, eps=eps, affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


has_apex = False


def fast_layer_norm(x: torch.Tensor, normalized_shape: List[int], weight: Optional[torch.Tensor]=None, bias: Optional[torch.Tensor]=None, eps: float=1e-05) ->torch.Tensor:
    if torch.jit.is_scripting():
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    if has_apex:
        return fused_layer_norm_affine(x, weight, bias, normalized_shape, eps)
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_gpu_dtype()
        x, weight, bias = x, weight, bias
    with torch.amp.autocast(enabled=False):
        return F.layer_norm(x, normalized_shape, weight, bias, eps)


class LayerNormAct(nn.LayerNorm):

    def __init__(self, normalization_shape: Union[int, List[int], torch.Size], eps=1e-05, affine=True, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct, self).__init__(normalization_shape, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):

    def __init__(self, num_channels, eps=1e-05, affine=True, apply_act=True, act_layer=nn.ReLU, inplace=True, drop_layer=None):
        super(LayerNormAct2d, self).__init__(num_channels, eps=eps, elementwise_affine=affine)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        act_layer = get_act_layer(act_layer)
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)
        x = self.act(x)
        return x


def instance_std(x, eps: float=1e-05):
    std = x.float().var(dim=(2, 3), unbiased=False, keepdim=True).add(eps).sqrt()
    return std.expand(x.shape)


class EvoNorm2dB0(nn.Module):

    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=0.001, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.v is not None:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(self.running_var * (1 - self.momentum) + var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            left = var.add(self.eps).sqrt_().view(v_shape).expand_as(x)
            v = self.v.view(v_shape)
            right = x * v + instance_std(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


def instance_rms(x, eps: float=1e-05):
    rms = x.float().square().mean(dim=(2, 3), keepdim=True).add(eps).sqrt()
    return rms.expand(x.shape)


class EvoNorm2dB1(nn.Module):

    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-05, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(self.running_var * (1 - self.momentum) + var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = (x + 1) * instance_rms(x, self.eps)
            x = x / left.max(right)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


class EvoNorm2dB2(nn.Module):

    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-05, **_):
        super().__init__()
        self.apply_act = apply_act
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.apply_act:
            if self.training:
                var = x.float().var(dim=(0, 2, 3), unbiased=False)
                n = x.numel() / x.shape[1]
                self.running_var.copy_(self.running_var * (1 - self.momentum) + var.detach() * self.momentum * (n / (n - 1)))
            else:
                var = self.running_var
            var = var.view(v_shape)
            left = var.add(self.eps).sqrt_()
            right = instance_rms(x, self.eps) - x
            x = x / left.max(right)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


def group_std(x, groups: int=32, eps: float=1e-05, flatten: bool=False):
    B, C, H, W = x.shape
    x_dtype = x.dtype
    _assert(C % groups == 0, '')
    if flatten:
        x = x.reshape(B, groups, -1)
        std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt()
    else:
        x = x.reshape(B, groups, C // groups, H, W)
        std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt()
    return std.expand(x.shape).reshape(B, C, H, W)


class EvoNorm2dS0(nn.Module):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-05, **_):
        super().__init__()
        self.apply_act = apply_act
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.v is not None:
            nn.init.ones_(self.v)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.v is not None:
            v = self.v.view(v_shape)
            x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


class EvoNorm2dS0a(EvoNorm2dS0):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=0.001, **_):
        super().__init__(num_features, groups=groups, group_size=group_size, apply_act=apply_act, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        d = group_std(x, self.groups, self.eps)
        if self.v is not None:
            v = self.v.view(v_shape)
            x = x * (x * v).sigmoid()
        x = x / d
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


class EvoNorm2dS1(nn.Module):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, act_layer=None, eps=1e-05, **_):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.pre_act_norm = False
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.apply_act:
            x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


class EvoNorm2dS1a(EvoNorm2dS1):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, act_layer=None, eps=0.001, **_):
        super().__init__(num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = self.act(x) / group_std(x, self.groups, self.eps)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


def group_rms(x, groups: int=32, eps: float=1e-05):
    B, C, H, W = x.shape
    _assert(C % groups == 0, '')
    x_dtype = x.dtype
    x = x.reshape(B, groups, C // groups, H, W)
    rms = x.float().square().mean(dim=(2, 3, 4), keepdim=True).add(eps).sqrt_()
    return rms.expand(x.shape).reshape(B, C, H, W)


class EvoNorm2dS2(nn.Module):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, act_layer=None, eps=1e-05, **_):
        super().__init__()
        act_layer = act_layer or nn.SiLU
        self.apply_act = apply_act
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer)
        else:
            self.act = nn.Identity()
        if group_size:
            assert num_features % group_size == 0
            self.groups = num_features // group_size
        else:
            self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        if self.apply_act:
            x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


class EvoNorm2dS2a(EvoNorm2dS2):

    def __init__(self, num_features, groups=32, group_size=None, apply_act=True, act_layer=None, eps=0.001, **_):
        super().__init__(num_features, groups=groups, group_size=group_size, apply_act=apply_act, act_layer=act_layer, eps=eps)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = self.act(x) / group_rms(x, self.groups, self.eps)
        return x * self.weight.view(v_shape) + self.bias.view(v_shape)


def inv_instance_rms(x, eps: float=1e-05):
    rms = x.square().float().mean(dim=(2, 3), keepdim=True).add(eps).rsqrt()
    return rms.expand(x.shape)


class FilterResponseNormAct2d(nn.Module):

    def __init__(self, num_features, apply_act=True, act_layer=nn.ReLU, inplace=None, rms=True, eps=1e-05, **_):
        super(FilterResponseNormAct2d, self).__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape) + self.bias.view(v_shape)
        return self.act(x)


class FilterResponseNormTlu2d(nn.Module):

    def __init__(self, num_features, apply_act=True, eps=1e-05, rms=True, **_):
        super(FilterResponseNormTlu2d, self).__init__()
        self.apply_act = apply_act
        self.rms = rms
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.tau = nn.Parameter(torch.zeros(num_features)) if apply_act else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.tau is not None:
            nn.init.zeros_(self.tau)

    def forward(self, x):
        _assert(x.dim() == 4, 'expected 4D input')
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape) + self.bias.view(v_shape)
        return torch.maximum(x, self.tau.reshape(v_shape)) if self.tau is not None else x


class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, apply_act=True, act_layer='leaky_relu', act_param=0.01, drop_layer=None):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            elif act_layer == nn.ELU:
                self.act_name = 'elu'
            elif act_layer == nn.LeakyReLU:
                self.act_name = 'leaky_relu'
            elif act_layer is None or act_layer == nn.Identity:
                self.act_name = 'identity'
            else:
                assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        output = inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps, self.act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output


_NORM_ACT_MAP = dict(batchnorm=BatchNormAct2d, batchnorm2d=BatchNormAct2d, groupnorm=GroupNormAct, groupnorm1=functools.partial(GroupNormAct, num_groups=1), layernorm=LayerNormAct, layernorm2d=LayerNormAct2d, evonormb0=EvoNorm2dB0, evonormb1=EvoNorm2dB1, evonormb2=EvoNorm2dB2, evonorms0=EvoNorm2dS0, evonorms0a=EvoNorm2dS0a, evonorms1=EvoNorm2dS1, evonorms1a=EvoNorm2dS1a, evonorms2=EvoNorm2dS2, evonorms2a=EvoNorm2dS2a, frn=FilterResponseNormAct2d, frntlu=FilterResponseNormTlu2d, inplaceabn=InplaceAbn, iabn=InplaceAbn)


_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct, LayerNormAct, LayerNormAct2d, FilterResponseNormAct2d, InplaceAbn}


_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}


def get_norm_act_layer(norm_layer, act_layer=None):
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP.get(layer_name, None)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer, types.FunctionType):
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        elif type_name.startswith('groupnorm1'):
            norm_act_layer = functools.partial(GroupNormAct, num_groups=1)
        elif type_name.startswith('layernorm2d'):
            norm_act_layer = LayerNormAct2d
        elif type_name.startswith('layernorm'):
            norm_act_layer = LayerNormAct
        else:
            assert False, f'No equivalent norm_act layer for {type_name}'
    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer


class ConvNormAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_layer=None):
        super(ConvNormAct, self).__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """

    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(SpatialAttn, self).__init__()
        self.conv = ConvNormAct(2, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """

    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvNormAct(1, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.amax(dim=1, keepdim=True)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class CbamModule(nn.Module):

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=1, spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(channels, rd_ratio=rd_ratio, rd_channels=rd_channels, rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=1, spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels, rd_ratio=rd_ratio, rd_channels=rd_channels, rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv
    if not pool_type:
        assert num_classes == 0 or use_conv, 'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(self, in_chs, num_classes, pool_type='avg', drop_rate=0.0, use_conv=False):
        super(ClassifierHead, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(in_chs, num_classes, pool_type, use_conv=use_conv)
        self.fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def forward(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.fc(x)
            return self.flatten(x)


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


class ConvNormActAa(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, apply_act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, aa_layer=None, drop_layer=None):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None and stride == 2
        self.conv = create_conv2d(in_channels, out_channels, kernel_size, stride=1 if use_aa else stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)
        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=use_aa)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.aa(x)
        return x


def drop_block_2d(x, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-06)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False, fast: bool=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float=0.0, training: bool=False, scale_by_keep: bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep: bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class EcaModule(nn.Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid', rd_ratio=1 / 8, rd_channels=None, rd_divisor=8, use_mlp=False):
        super(EcaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        if use_mlp:
            assert channels is not None
            if rd_channels is None:
                rd_channels = make_divisible(channels * rd_ratio, divisor=rd_divisor)
            act_layer = act_layer or nn.ReLU
            self.conv = nn.Conv1d(1, rd_channels, kernel_size=1, padding=0, bias=True)
            self.act = create_act_layer(act_layer)
            self.conv2 = nn.Conv1d(rd_channels, 1, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.act = None
            self.conv2 = None
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = self.conv(y)
        if self.conv2 is not None:
            y = self.act(y)
            y = self.conv2(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


class CecaModule(nn.Module):
    """Constructs a circular ECA module.

    ECA module where the conv uses circular padding rather than zero padding.
    Unlike the spatial dimension, the channels do not have inherent ordering nor
    locality. Although this module in essence, applies such an assumption, it is unnecessary
    to limit the channels on either "edge" from being circularly adapted to each other.
    This will fundamentally increase connectivity and possibly increase performance metrics
    (accuracy, robustness), without significantly impacting resource metrics
    (parameter size, throughput,latency, etc)

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
        gamm: used in kernel_size calc, see above
        beta: used in kernel_size calc, see above
        act_layer: optional non-linearity after conv, enables conv bias, this is an experiment
        gate_layer: gating non-linearity to use
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1, act_layer=None, gate_layer='sigmoid'):
        super(CecaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        has_act = act_layer is not None
        assert kernel_size % 2 == 1
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=has_act)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = F.pad(y, (self.padding, self.padding), mode='circular')
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


class BilinearAttnTransform(nn.Module):

    def __init__(self, in_channels, block_size, groups, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BilinearAttnTransform, self).__init__()
        self.conv1 = ConvNormAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1))
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size))
        self.conv2 = ConvNormAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, '')
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(B * C, block_size, block_size, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def forward(self, x):
        _assert(x.shape[-1] % self.block_size == 0, '')
        _assert(x.shape[-2] % self.block_size == 0, '')
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.block_size, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.block_size))
        p = self.conv_p(rp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)
        y = self.conv2(y)
        return y


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float('inf'))
        return F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-06, gain_init=1.0):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(self.weight.reshape(1, self.out_channels, -1), None, None, weight=(self.gain * self.scale).view(-1), training=True, momentum=0.0, eps=self.eps).reshape_as(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding='SAME', dilation=1, groups=1, bias=False, eps=1e-06):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


_leaf_modules = {BilinearAttnTransform, Conv2dSame, MaxPool2dSame, ScaledStdConv2dSame, StdConv2dSame, AvgPool2dSame, CondConv2d}


def register_notrace_module(module: Type[nn.Module]):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


class ConvMlp(nn.Module):

    def __init__(self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0, drop_rate: float=0.2, act_layer: nn.Module=None, conv_layer: nn.Module=None):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            output_size = max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1])
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class GatherExcite(nn.Module):
    """ Gather-Excite Attention Module
    """

    def __init__(self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=1, add_maxpool=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module('conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(f'conv{i + 1}', create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        elif self.extent == 0:
            x_ge = x.mean(dim=(2, 3), keepdims=True)
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
        else:
            x_ge = F.avg_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-06, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False, rd_ratio=1.0 / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)
        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x
        return x


class HaloAttn(nn.Module):
    """ Halo Attention

    Paper: `Scaling Local Self-Attention for Parameter Efficient Visual Backbones`
        - https://arxiv.org/abs/2103.12731

    The internal dimensions of the attention module are controlled by the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query and key (qk) dimensions are determined by
        * num_heads * dim_head if dim_head is not None
        * num_heads * (dim_out * attn_ratio // num_heads) if dim_head is None
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not used

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """

    def __init__(self, dim, dim_out=None, feat_size=None, stride=1, num_heads=8, dim_head=None, block_size=8, halo_size=3, qk_ratio=1.0, qkv_bias=False, avg_down=False, scale_pos_embed=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        assert stride in (1, 2)
        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2
        self.block_stride = 1
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = 1 if use_avg_pool else stride
            self.block_size_ds = self.block_size // self.block_stride
        self.q = nn.Conv2d(dim, self.dim_out_qk, 1, stride=self.block_stride, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, bias=qkv_bias)
        self.pos_embed = PosEmbedRel(block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk, scale=self.scale)
        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = self.q.weight.shape[1] ** -0.5
        trunc_normal_(self.q.weight, std=std)
        trunc_normal_(self.kv.weight, std=std)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H % self.block_size == 0, '')
        _assert(W % self.block_size == 0, '')
        num_h_blocks = H // self.block_size
        num_w_blocks = W // self.block_size
        num_blocks = num_h_blocks * num_w_blocks
        q = self.q(x)
        q = q.reshape(-1, self.dim_head_qk, num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds).permute(0, 1, 3, 5, 2, 4)
        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1, num_blocks).transpose(1, 3)
        kv = self.kv(x)
        kv = F.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size])
        kv = kv.unfold(2, self.win_size, self.block_size).unfold(3, self.win_size, self.block_size).reshape(B * self.num_heads, self.dim_head_qk + self.dim_head_v, num_blocks, -1).permute(0, 2, 3, 1)
        k, v = torch.split(kv, [self.dim_head_qk, self.dim_head_v], dim=-1)
        if self.scale_pos_embed:
            attn = (q @ k.transpose(-1, -2) + self.pos_embed(q)) * self.scale
        else:
            attn = q @ k.transpose(-1, -2) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 3)
        out = out.reshape(-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks)
        out = out.permute(0, 3, 1, 4, 2).contiguous().view(B, self.dim_out_v, H // self.block_stride, W // self.block_stride)
        out = self.pool(out)
        return out


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = torch.stack(torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos


class LambdaLayer(nn.Module):
    """Lambda Layer

    Paper: `LambdaNetworks: Modeling Long-Range Interactions Without Attention`
        - https://arxiv.org/abs/2102.08602

    NOTE: intra-depth parameter 'u' is fixed at 1. It did not appear worth the complexity to add.

    The internal dimensions of the lambda module are controlled via the interaction of several arguments.
      * the output dimension of the module is specified by dim_out, which falls back to input dim if not set
      * the value (v) dimension is set to dim_out // num_heads, the v projection determines the output dim
      * the query (q) and key (k) dimension are determined by
        * dim_head = (dim_out * attn_ratio // num_heads) if dim_head is None
        * q = num_heads * dim_head, k = dim_head
      * as seen above, attn_ratio determines the ratio of q and k relative to the output if dim_head not set

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map for relative pos variant H, W
        stride (int): output stride of the module, avg pool used if stride == 2
        num_heads (int): parallel attention heads.
        dim_head (int): dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        r (int): local lambda convolution radius. Use lambda conv if set, else relative pos if not. (default: 9)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool): add bias to q, k, and v projections
    """

    def __init__(self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=16, r=9, qk_ratio=1.0, qkv_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, ' should be divided by num_heads'
        self.dim_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads
        self.qkv = nn.Conv2d(dim, num_heads * self.dim_qk + self.dim_qk + self.dim_v, kernel_size=1, bias=qkv_bias)
        self.norm_q = nn.BatchNorm2d(num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm2d(self.dim_v)
        if r is not None:
            self.conv_lambda = nn.Conv3d(1, self.dim_qk, (r, r, 1), padding=(r // 2, r // 2, 0))
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [(2 * s - 1) for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(torch.zeros(rel_size[0], rel_size[1], self.dim_qk))
            self.register_buffer('rel_pos_indices', rel_pos_indices(feat_size), persistent=False)
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_qk ** -0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.num_heads * self.dim_qk, self.dim_qk, self.dim_v], dim=1)
        q = self.norm_q(q).reshape(B, self.num_heads, self.dim_qk, M).transpose(-1, -2)
        v = self.norm_v(v).reshape(B, self.dim_v, M).transpose(-1, -2)
        k = F.softmax(k.reshape(B, self.dim_qk, M), dim=-1)
        content_lam = k @ v
        content_out = q @ content_lam.unsqueeze(1)
        if self.pos_emb is None:
            position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))
            position_lam = position_lam.reshape(B, 1, self.dim_qk, H * W, self.dim_v).transpose(2, 3)
        else:
            pos_emb = self.pos_emb[self.rel_pos_indices[0], self.rel_pos_indices[1]].expand(B, -1, -1, -1)
            position_lam = (pos_emb.transpose(-1, -2) @ v.unsqueeze(1)).unsqueeze(1)
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(-2)
        out = (content_out + position_out).transpose(-1, -2).reshape(B, C, H, W)
        out = self.pool(out)
        return out


class Linear(nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias if self.bias is not None else None
            return F.linear(input, self.weight, bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


to_4tuple = _ntuple(4)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_4tuple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - ih % self.stride[0], 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - iw % self.stride[1], 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = pl, pr, pt, pb
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class TransformerDecoderLayerOptimal(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05) ->None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor]=None, memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None, memory_key_padding_mask: Optional[Tensor]=None) ->Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLDecoder(nn.Module):

    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768, initial_num_features=2048):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding, dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
        self.query_embed.requires_grad_(False)
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
        self.duplicate_pooling = torch.nn.Parameter(torch.Tensor(embed_len_decoder, decoder_embedding, self.duplicate_factor))
        self.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)
        self.group_fc = GroupFC(embed_len_decoder)

    def forward(self, x):
        if len(x.shape) == 4:
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:
            embedding_spatial = x
        embedding_spatial_786 = self.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)
        bs = embedding_spatial_786.shape[0]
        query_embed = self.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))
        h = h.transpose(0, 1)
        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.duplicate_factor, device=h.device, dtype=h.dtype)
        self.group_fc(h, self.duplicate_pooling, out_extrap)
        h_out = out_extrap.flatten(1)[:, :self.num_classes]
        h_out += self.duplicate_pooling_bias
        logits = h_out
        return logits


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-06)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, gate_layer=None, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class NonLocalAttn(nn.Module):
    """Spatial NL block for image classification.

    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(self, in_channels, use_scale=True, rd_ratio=1 / 8, rd_channels=None, rd_divisor=8, **kwargs):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        B, C, H, W = t.size()
        t = t.view(B, C, -1).permute(0, 2, 1)
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).permute(0, 2, 1)
        att = torch.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, g)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut
        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


class BatNonLocalAttn(nn.Module):
    """ BAT
    Adapted from: https://github.com/BA-Transform/BAT-Image-Classification
    """

    def __init__(self, in_channels, block_size=7, groups=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8, drop_rate=0.2, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, **_):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.conv1 = ConvNormAct(in_channels, rd_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.ba = BilinearAttnTransform(rd_channels, block_size, groups, act_layer=act_layer, norm_layer=norm_layer)
        self.conv2 = ConvNormAct(rd_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.dropout = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.ba(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=32, eps=1e-05, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.fast_norm = is_fast_norm()

    def forward(self, x):
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class GroupNorm1(nn.GroupNorm):
    """ Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """

    def __init__(self, num_channels, eps=1e-06, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


def _is_contiguous(tensor: torch.Tensor) ->bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@torch.jit.script
def _layer_norm_cf(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
    x = (x - u) * torch.rsqrt(s + eps)
    x = x * weight[:, None, None] + bias[:, None, None]
    return x


class LayerNormExp2d(nn.LayerNorm):
    """ LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

    Experimental implementation w/ manual norm for tensors non-contiguous tensors.

    This improves throughput in some scenarios (tested on Ampere GPU), esp w/ channels_last
    layout. However, benefits are not always clear and can perform worse on other GPUs.
    """

    def __init__(self, num_channels, eps=1e-06):
        super().__init__(num_channels, eps=eps)

    def forward(self, x) ->torch.Tensor:
        if _is_contiguous(x):
            x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            x = _layer_norm_cf(x, self.weight, self.bias, self.eps)
        return x


class SyncBatchNormAct(nn.SyncBatchNorm):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = super().forward(x)
        if hasattr(self, 'drop'):
            x = self.drop(x)
        if hasattr(self, 'act'):
            x = self.act(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1, patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        if stem_conv:
            self.conv = nn.Sequential(nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        else:
            self.conv = None
        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride)
        self.num_patches = img_size // patch_size * (img_size // patch_size)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.proj(x)
        return x


def gen_relative_position_index(window_size: Tuple[int, int]) ->torch.Tensor:
    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
    window_area = window_size[0] * window_size[1]
    coords = torch.stack(torch.meshgrid([torch.arange(window_size[0]), torch.arange(window_size[1])]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = torch.zeros(size=(window_area + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


class RelPosBias(nn.Module):
    """ Relative Position Bias
    Adapted from Swin-V1 relative position bias impl, modularized.
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.bias_shape = (self.window_area + prefix_tokens,) * 2 + (num_heads,)
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3 * prefix_tokens
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        self.register_buffer('relative_position_index', gen_relative_position_index(self.window_size, class_token=prefix_tokens > 0), persistent=False)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_bias(self) ->torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.bias_shape).permute(2, 0, 1)
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor]=None):
        return attn + self.get_bias()


def gen_relative_log_coords(win_size: Tuple[int, int], pretrained_win_size: Tuple[int, int]=(0, 0), mode='swin'):
    assert mode in ('swin', 'cr', 'rw')
    relative_coords_h = torch.arange(-(win_size[0] - 1), win_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(win_size[1] - 1), win_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
    relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()
    if mode == 'swin':
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= pretrained_win_size[0] - 1
            relative_coords_table[:, :, 1] /= pretrained_win_size[1] - 1
        else:
            relative_coords_table[:, :, 0] /= win_size[0] - 1
            relative_coords_table[:, :, 1] /= win_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(1.0 + relative_coords_table.abs()) / math.log2(8)
    elif mode == 'rw':
        relative_coords_table[:, :, 0] /= win_size[0] - 1
        relative_coords_table[:, :, 1] /= win_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(1.0 + relative_coords_table.abs())
        relative_coords_table /= math.log2(9)
    else:
        relative_coords_table = torch.sign(relative_coords_table) * torch.log(1.0 + relative_coords_table.abs())
    return relative_coords_table


class RelPosMlp(nn.Module):
    """ Log-Coordinate Relative Position MLP
    Based on ideas presented in Swin-V2 paper (https://arxiv.org/abs/2111.09883)

    This impl covers the 'swin' implementation as well as two timm specific modes ('cr', and 'rw')
    """

    def __init__(self, window_size, num_heads=8, hidden_dim=128, prefix_tokens=0, mode='cr', pretrained_window_size=(0, 0)):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == 'swin':
            self.bias_act = nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = True, False
        elif mode == 'rw':
            self.bias_act = nn.Tanh()
            self.bias_gain = 4
            mlp_bias = True
        else:
            self.bias_act = nn.Identity()
            self.bias_gain = None
            mlp_bias = True
        self.mlp = Mlp(2, hidden_features=hidden_dim, out_features=num_heads, act_layer=nn.ReLU, bias=mlp_bias, drop=(0.125, 0.0))
        self.register_buffer('relative_position_index', gen_relative_position_index(window_size), persistent=False)
        self.register_buffer('rel_coords_log', gen_relative_log_coords(window_size, pretrained_window_size, mode=mode), persistent=False)

    def get_bias(self) ->torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[self.relative_position_index.view(-1)]
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = F.pad(relative_position_bias, [self.prefix_tokens, 0, self.prefix_tokens, 0])
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor]=None):
        return attn + self.get_bias()


def generate_lookup_tensor(length: int, max_relative_position: Optional[int]=None):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1
    vocab_size = 2 * max_relative_position + 1
    ret = torch.zeros(length, length, vocab_size)
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret


def reindex_2d_einsum_lookup(relative_position_tensor, height: int, width: int, height_lookup: torch.Tensor, width_lookup: torch.Tensor) ->torch.Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    reindexed_tensor = torch.einsum('nhw,ixh->nixw', relative_position_tensor, height_lookup)
    reindexed_tensor = torch.einsum('nixw,jyw->nijxy', reindexed_tensor, width_lookup)
    area = height * width
    return reindexed_tensor.reshape(relative_position_tensor.shape[0], area, area)


class RelPosBiasTf(nn.Module):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads
        vocab_height = 2 * window_size[0] - 1
        vocab_width = 2 * window_size[1] - 1
        self.bias_shape = self.num_heads, vocab_height, vocab_width
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.bias_shape))
        self.register_buffer('height_lookup', generate_lookup_tensor(window_size[0]), persistent=False)
        self.register_buffer('width_lookup', generate_lookup_tensor(window_size[1]), persistent=False)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.relative_position_bias_table, std=0.02)

    def get_bias(self) ->torch.Tensor:
        return reindex_2d_einsum_lookup(self.relative_position_bias_table, self.window_size[0], self.window_size[1], self.height_lookup, self.width_lookup)

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor]=None):
        return attn + self.get_bias()


class FourierEmbed(nn.Module):

    def __init__(self, max_res: int=224, num_bands: int=64, concat_grid=True, keep_spatial=False):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer('bands', pixel_freq_bands(max_res, num_bands), persistent=False)

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(feat_shape, self.bands, include_grid=self.concat_grid, dtype=x.dtype, device=x.device)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)
        if self.keep_spatial:
            x = torch.cat([x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1)
        else:
            x = torch.cat([x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1)
            x = x.reshape(B, feat_shape.numel(), -1)
        return x


class SelectiveKernelAttn(nn.Module):

    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x):
        _assert(x.shape[1] == self.num_paths, '')
        x = x.sum(1).mean((2, 3), keepdim=True)
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = torch.softmax(x, dim=1)
        return x


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernel(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, dilation=1, groups=1, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=8, keep_3x3=True, split_input=True, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_layer=None):
        """ Selective Kernel Convolution Module

        As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications.

        Largest change is the input split, which divides the input channels across each convolution path, this can
        be viewed as a grouping of sorts, but the output channel counts expand to the module level value. This keeps
        the parameter count from ballooning when the convolutions themselves don't have groups, but still provides
        a noteworthy increase in performance over similar param count models without this attention layer. -Ross W

        Args:
            in_channels (int):  module input (feature) channel count
            out_channels (int):  module output (feature) channel count
            kernel_size (int, list): kernel size for each convolution branch
            stride (int): stride for convolutions
            dilation (int): dilation for module as a whole, impacts dilation of each branch
            groups (int): number of groups for each branch
            rd_ratio (int, float): reduction factor for attention features
            keep_3x3 (bool): keep all branch convolution kernels as 3x3, changing larger kernels for dilations
            split_input (bool): split input channels evenly across each convolution branch, keeps param count lower,
                can be viewed as grouping by path, output expands to module out_channels count
            act_layer (nn.Module): activation layer to use
            norm_layer (nn.Module): batchnorm/norm layer to use
            aa_layer (nn.Module): anti-aliasing module
            drop_layer (nn.Module): spatial drop module in convs (drop block, etc)
        """
        super(SelectiveKernel, self).__init__()
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [(dilation * (k - 1) // 2) for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)
        conv_kwargs = dict(stride=stride, groups=groups, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_layer=drop_layer)
        self.paths = nn.ModuleList([ConvNormActAa(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs) for k, d in zip(kernel_size, dilation)])
        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1)
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)]
        else:
            x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = torch.sum(x, dim=1)
        return x


class SeparableConvNormAct(nn.Module):
    """ Separable Conv w/ trailing Norm and Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False, channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, apply_act=True, drop_layer=None):
        super(SeparableConvNormAct, self).__init__()
        self.conv_dw = create_conv2d(in_channels, int(in_channels * channel_multiplier), kernel_size, stride=stride, dilation=dilation, padding=padding, depthwise=True)
        self.conv_pw = create_conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.bn(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1, padding='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv_dw = create_conv2d(in_chs, in_chs, kernel_size, stride=stride, padding=padding, dilation=dilation, depthwise=True)
        self.bn_dw = norm_layer(in_chs)
        self.act_dw = act_layer(inplace=True) if act_layer is not None else nn.Identity()
        self.conv_pw = create_conv2d(in_chs, out_chs, kernel_size=1)
        self.bn_pw = norm_layer(out_chs)
        self.act_pw = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class SpaceToDepthModule(nn.Module):

    def __init__(self, no_jit=False):
        super().__init__()
        if not no_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class RadixSoftmax(nn.Module):

    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8, act_layer=nn.ReLU, norm_layer=None, drop_layer=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, mid_chs, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)
        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, num_splits=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])

    def forward(self, input: torch.Tensor):
        if self.training:
            split_size = input.shape[0] // self.num_splits
            assert input.shape[0] == split_size * self.num_splits, 'batch size must be evenly divisible by num_splits'
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid', **_):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class SqueezeExciteCl(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=8, bias=True, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Linear(channels, rd_channels, bias=bias)
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((1, 2), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(self, in_channel, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False, eps=1e-06):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-06, gain_init=1.0):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(self.weight.reshape(1, self.out_channels, -1), None, None, weight=(self.gain * self.scale).view(-1), training=True, momentum=0.0, eps=self.eps).reshape_as(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TestTimePoolHead(nn.Module):

    def __init__(self, base, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.base = base
        self.original_pool = original_pool
        base_fc = self.base.get_classifier()
        if isinstance(base_fc, nn.Conv2d):
            self.fc = base_fc
        else:
            self.fc = nn.Conv2d(self.base.num_features, self.base.num_classes, kernel_size=1, bias=True)
            self.fc.weight.data.copy_(base_fc.weight.data.view(self.fc.weight.size()))
            self.fc.bias.data.copy_(base_fc.bias.data.view(self.fc.bias.size()))
        self.base.reset_classifier(0)

    def forward(self, x):
        x = self.base.forward_features(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)


class AsymmetricLossMultiLabel(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossMultiLabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()


class AsymmetricLossSingleLabel(nn.Module):

    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float=0.1, reduction='mean'):
        super(AsymmetricLossSingleLabel, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w
        if self.eps > 0:
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)
        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class BinaryCrossEntropy(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """

    def __init__(self, smoothing=0.1, target_threshold: Optional[float]=None, weight: Optional[torch.Tensor]=None, reduction: str='mean', pos_weight: Optional[torch.Tensor]=None):
        super(BinaryCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            num_classes = x.shape[-1]
            off_value = self.smoothing / num_classes
            on_value = 1.0 - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full((target.size()[0], num_classes), off_value, device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold)
        return F.binary_cross_entropy_with_logits(x, target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction)


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-07, 1).log()
        loss += self.alpha * sum([F.kl_div(logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU, gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=1, group_size=0, pad_type='', skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0.0):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='bn1', hook_type='forward', num_chs=self.conv.out_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='', noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.conv_dw = create_conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='', noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.0):
        super(InvertedResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)
        self.conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type, **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, inplace=True)
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='', noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, num_experts=0, drop_path_rate=0.0):
        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, group_size=group_size, pad_type=pad_type, act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size, pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, conv_kwargs=conv_kwargs, drop_path_rate=drop_path_rate)
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, group_size=0, pad_type='', force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class FeatureInfo:

    def __init__(self, feature_info: List[Dict], out_indices: Tuple[int]):
        prev_reduction = 1
        for fi in feature_info:
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: Tuple[int]):
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key, idx=None):
        """ Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tupple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys=None, idx=None):
        """ return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [(self.info[i] if keys is None else {k: self.info[i][k] for k in keys}) for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx=None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx=None):
        """ feature module name accessor
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


def _get_feature_info(net, out_indices):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, 'Provided feature_info is not valid'


def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


def _module_list(module, flatten_sequential=False):
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml


class FeatureDictNet(nn.ModuleDict):
    """ Feature extractor with OrderedDict return

    Wrap a model and extract features as specified by the out indices, the network is
    partially re-built from contained modules.

    There is a strong assumption that the modules have been registered into the model in the same
    order as they are used. There should be no reuse of the same nn.Module more than once, including
    trivial modules like `self.relu = nn.ReLU`.

    Only submodules that are directly assigned to the model class (`model.feature1`) or at most
    one Sequential container deep (`model.features.1`, with flatten_sequent=True) can be captured.
    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`

    Arguments:
        model (nn.Module): model from which we will extract the features
        out_indices (tuple[int]): model output indices to extract features for
        out_map (sequence): list or tuple specifying desired return id for each out index,
            otherwise str(index) is used
        feature_concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.concat = feature_concat
        self.return_layers = {}
        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def _collect(self, x) ->Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) ->Dict[str, torch.Tensor]:
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """ Feature extractor with list return

    See docstring for FeatureDictNet above, this class exists only to appease Torchscript typing constraints.
    In eager Python we could have returned List[Tensor] vs Dict[id, Tensor] based on a member bool.
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureListNet, self).__init__(model, out_indices=out_indices, out_map=out_map, feature_concat=feature_concat, flatten_sequential=flatten_sequential)

    def forward(self, x) ->List[torch.Tensor]:
        return list(self._collect(x).values())


class FeatureHooks:
    """ Feature Hook Helper

    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torchscript.
    """

    def __init__(self, hooks, named_modules, out_map=None, default_hook_type='forward'):
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = h.get('hook_type', default_hook_type)
            if hook_type == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, 'Unsupported hook type'
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, hook_id, *args):
        x = args[-1]
        if isinstance(x, tuple):
            x = x[0]
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) ->Dict[str, torch.tensor]:
        output = self._feature_outputs[device]
        self._feature_outputs[device] = OrderedDict()
        return output


class FeatureHookNet(nn.ModuleDict):
    """ FeatureHookNet

    Wrap a model and extract features specified by the out indices using forward/forward-pre hooks.

    If `no_rewrite` is True, features are extracted via hooks without modifying the underlying
    network in any way.

    If `no_rewrite` is False, the model will be re-written as in the
    FeatureList/FeatureDict case by folding first to second (Sequential only) level modules into this one.

    FIXME this does not currently work with Torchscript, see FeatureHooks class
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, no_rewrite=False, feature_concat=False, flatten_sequential=False, default_hook_type='forward'):
        super(FeatureHookNet, self).__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_as_dict = out_as_dict
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {f['module']: (f['hook_type'] if 'hook_type' in f else default_hook_type) for f in self.feature_info.get_dicts()}
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.out_as_dict else list(out.values())


_autowrap_functions = set()


def create_feature_extractor(model: nn.Module, return_nodes: Union[Dict[str, str], List[str]]):
    assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
    return _create_feature_extractor(model, return_nodes, tracer_kwargs={'leaf_modules': list(_leaf_modules), 'autowrap_functions': list(_autowrap_functions)})


class FeatureGraphNet(nn.Module):
    """ A FX Graph based feature extractor that works with the model feature_info metadata
    """

    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        assert has_fx_feature_extraction, 'Please update to PyTorch 1.10+, torchvision 0.11+ for FX feature extraction'
        self.feature_info = _get_feature_info(model, out_indices)
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        return_nodes = {info['module']: (out_map[i] if out_map is not None else info['module']) for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = create_feature_extractor(model, return_nodes)

    def forward(self, x):
        return list(self.graph_module(x).values())


class GraphExtractNet(nn.Module):
    """ A standalone feature extraction wrapper that maps dict -> list or single tensor
    NOTE:
      * one can use feature_extractor directly if dictionary output is desired
      * unlike FeatureGraphNet, this is intended to be used standalone and not with model feature_info
      metadata for builtin feature extraction mode
      * create_feature_extractor can be used directly if dictionary output is acceptable

    Args:
        model: model to extract features from
        return_nodes: node names to return features from (dict or list)
        squeeze_out: if only one output, and output in list format, flatten to single tensor
    """

    def __init__(self, model, return_nodes: Union[Dict[str, str], List[str]], squeeze_out: bool=True):
        super().__init__()
        self.squeeze_out = squeeze_out
        self.graph_module = create_feature_extractor(model, return_nodes)

    def forward(self, x) ->Union[List[torch.Tensor], torch.Tensor]:
        out = list(self.graph_module(x).values())
        if self.squeeze_out and len(out) == 1:
            return out[0]
        return out


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < reps - 1 else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_distance, num_heads))
        self.register_buffer('relative_position_index', gen_relative_position_index(window_size))

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_area + 1, self.window_area + 1, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class Beit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_values=None, use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, head_init_scale=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None) for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

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
        nwd = {'pos_embed', 'cls_token'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^cls_token|pos_embed|patch_embed|rel_pos_bias', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.fc_norm is not None:
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class DownsampleAvg(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True, conv_layer=None, norm_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer='leaky_relu', act_param=0.01):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False), InplaceAbn(nf, act_layer=act_layer, act_param=act_param))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=0.001)
        elif aa_layer is None:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=0.001)
        else:
            self.conv1 = nn.Sequential(conv2d_iabn(inplanes, planes, stride=1, act_param=0.001), aa_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        rd_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, rd_channels=rd_chs) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = out + shortcut
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """ ResNe(X)t Bottleneck Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.25, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_last=False, attn_layer=None, drop_block=None, drop_path=0.0):
        super(BottleneckBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        attn_last = attn_layer is not None and attn_last
        attn_first = attn_layer is not None and not attn_last
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvNormAct(mid_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.attn2 = attn_layer(mid_chs, act_layer=act_layer) if attn_first else nn.Identity()
        self.conv3 = ConvNormAct(mid_chs, out_chs, kernel_size=1, apply_act=False, **ckwargs)
        self.attn3 = attn_layer(out_chs, act_layer=act_layer) if attn_last else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.act3 = create_act_layer(act_layer)

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        x = self.attn3(x)
        x = self.drop_path(x) + shortcut
        x = self.act3(x)
        return x


class DarkBlock(nn.Module):
    """ DarkNet Block
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, drop_block=None, drop_path=0.0):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(mid_chs, out_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x


class EdgeBlock(nn.Module):
    """ EdgeResidual / Fused-MBConv / MobileNetV1-like 3x3 + 1x1 block (w/ activated output)
    """

    def __init__(self, in_chs, out_chs, dilation=1, bottle_ratio=0.5, groups=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, drop_block=None, drop_path=0.0):
        super(EdgeBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, mid_chs, kernel_size=3, dilation=dilation, groups=groups, drop_layer=drop_block, **ckwargs)
        self.attn = attn_layer(mid_chs, act_layer=act_layer) if attn_layer is not None else nn.Identity()
        self.conv2 = ConvNormAct(mid_chs, out_chs, kernel_size=1, **ckwargs)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x) + shortcut
        return x


def downsample_avg(in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    return nn.Sequential(*[pool, nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), norm_layer(out_channels)])


def downsample_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = first_dilation or dilation if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)
    return nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False), norm_layer(out_channels)])


def create_shortcut(downsample_type, in_chs, out_chs, kernel_size, stride, dilation=(1, 1), norm_layer=None, preact=False):
    assert downsample_type in ('avg', 'conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0], norm_layer=norm_layer, preact=preact)
        if not downsample_type:
            return None
        elif downsample_type == 'avg':
            return downsample_avg(in_chs, out_chs, **dargs)
        else:
            return downsample_conv(in_chs, out_chs, kernel_size=kernel_size, **dargs)
    else:
        return nn.Identity()


def trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.

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
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) ->nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class Stem(nn.Module):

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int=3, padding: str='', bias: bool=False, act_layer: str='gelu', norm_layer: str='batchnorm2d', norm_eps: float=1e-05):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2
        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.norm1 = norm_act_layer(out_chs[0])
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == 'truncated_normal':
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.8796256610342398)
    elif distribution == 'normal':
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == 'uniform':
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f'invalid distribution {distribution}')


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


def _init_weights(module: nn.Module, name: str, head_bias: float=0.0, flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif flax:
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-06)
                else:
                    nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False, preserve_rng_state=True):
    """A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """

    def run_function(start, end, functions):

        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward
    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)
    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def create_block(block: Union[str, nn.Module], **kwargs):
    if isinstance(block, (nn.Module, partial)):
        return block(**kwargs)
    assert block in _block_registry, f'Unknown block type ({block}'
    return _block_registry[block](**kwargs)


def reduce_feat_size(feat_size, stride=2):
    return None if feat_size is None else tuple([(s // stride) for s in feat_size])


def get_attn(attn_type):
    if isinstance(attn_type, torch.nn.Module):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ecam':
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'ge':
                module_cls = GatherExcite
            elif attn_type == 'gc':
                module_cls = GlobalContext
            elif attn_type == 'gca':
                module_cls = partial(GlobalContext, fuse_add=True, fuse_scale=False)
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            elif attn_type == 'sk':
                module_cls = SelectiveKernel
            elif attn_type == 'splat':
                module_cls = SplitAttn
            elif attn_type == 'lambda':
                return LambdaLayer
            elif attn_type == 'bottleneck':
                return BottleneckAttn
            elif attn_type == 'halo':
                return HaloAttn
            elif attn_type == 'nl':
                module_cls = NonLocalAttn
            elif attn_type == 'bat':
                module_cls = BatNonLocalAttn
            else:
                assert False, 'Invalid attn module (%s)' % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def override_kwargs(block_kwargs, model_kwargs):
    """ Override model level attn/self-attn/block kwargs w/ block level

    NOTE: kwargs are NOT merged across levels, block_kwargs will fully replace model_kwargs
    for the block if set to anything that isn't None.

    i.e. an empty block_kwargs dict will remove kwargs set at model level for that block
    """
    out_kwargs = block_kwargs if block_kwargs is not None else model_kwargs
    return out_kwargs or {}


class ClassAttn(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class LayerScaleBlockClassAttn(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn, mlp_block=Mlp, init_values=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class TalkingHeadAttn(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=TalkingHeadAttn, mlp_block=Mlp, init_values=0.0001):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Cait(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, block_layers=LayerScaleBlock, block_layers_token=LayerScaleBlockClassAttn, patch_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, attn_block=TalkingHeadAttn, mlp_block=Mlp, init_values=0.0001, attn_block_token_only=ClassAttn, mlp_block_token_only=Mlp, depth_token_only=2, mlp_ratio_token_only=4.0):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.patch_embed = patch_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.Sequential(*[block_layers(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attn_block=attn_block, mlp_block=mlp_block, init_values=init_values) for i in range(depth)])
        self.blocks_token_only = nn.ModuleList([block_layers_token(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_token_only, qkv_bias=qkv_bias, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer, act_layer=act_layer, attn_block=attn_block_token_only, mlp_block=mlp_block_token_only, init_values=init_values) for i in range(depth_token_only)])
        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
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
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):

        def _matcher(name):
            if any([name.startswith(n) for n in ('cls_token', 'pos_embed', 'patch_embed')]):
                return 0
            elif name.startswith('blocks.'):
                return int(name.split('.')[1]) + 1
            elif name.startswith('blocks_token_only.'):
                to_offset = len(self.blocks) - len(self.blocks_token_only) + 1
                return int(name.split('.')[1]) + to_offset
            elif name.startswith('norm.'):
                return len(self.blocks)
            else:
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                1. An integer of window size, which assigns all attention heads with the same window s
                    size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits (
                    e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()
        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch, kernel_size=(cur_window, cur_window), padding=(padding_size, padding_size), dilation=(dilation, dilation), groups=cur_head_split * Ch)
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [(x * Ch) for x in self.head_splits]

    def forward(self, q, v, size: Tuple[int, int]):
        B, h, N, Ch = q.shape
        H, W = size
        _assert(N == 1 + H * W, '')
        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]
        v_img = v_img.transpose(-1, -2).reshape(B, h * Ch, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, h, Ch, H * W).transpose(-1, -2)
        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))
        return EV_hat


class FactorAttnConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.crpe = shared_crpe

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att
        crpe = self.crpe(q, v, size=size)
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding. 
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        _assert(N == 1 + H * W, '')
        cls_token, img_tokens = x[:, :1], x[:, 1:]
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x


class SerialBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, shared_cpe=None, shared_crpe=None):
        super().__init__()
        self.cpe = shared_cpe
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Tuple[int, int]):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ParallelBlock(nn.Module):

    def __init__(self, dim, num_heads, num_parallel=2, mlp_ratio=4.0, qkv_bias=False, init_values=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([('norm', norm_layer(dim)), ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)), ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()), ('drop_path', DropPath(drop_path) if drop_path > 0.0 else nn.Identity())])))
            self.ffns.append(nn.Sequential(OrderedDict([('norm', norm_layer(dim)), ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)), ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()), ('drop_path', DropPath(drop_path) if drop_path > 0.0 else nn.Identity())])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


def insert_cls(x, cls_token):
    """ Insert CLS token. """
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    return x


def remove_cls(x):
    """ Remove CLS token. """
    return x[:, 1:, :]


class CoaT(nn.Module):
    """ CoaT class. """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=(0, 0, 0, 0), serial_depths=(0, 0, 0, 0), parallel_depth=0, num_heads=0, mlp_ratios=(0, 0, 0, 0), qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), return_interm_layers=False, out_features=None, crpe_window=None, global_pool='token'):
        super().__init__()
        assert global_pool in ('token', 'avg')
        crpe_window = crpe_window or {(3): 2, (5): 3, (7): 3}
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool
        img_size = to_2tuple(img_size)
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=nn.LayerNorm)
        self.patch_embed2 = PatchEmbed(img_size=[(x // 4) for x in img_size], patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], norm_layer=nn.LayerNorm)
        self.patch_embed3 = PatchEmbed(img_size=[(x // 8) for x in img_size], patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], norm_layer=nn.LayerNorm)
        self.patch_embed4 = PatchEmbed(img_size=[(x // 16) for x in img_size], patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3], norm_layer=nn.LayerNorm)
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        dpr = drop_path_rate
        assert dpr == 0.0
        self.serial_blocks1 = nn.ModuleList([SerialBlock(dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe1, shared_crpe=self.crpe1) for _ in range(serial_depths[0])])
        self.serial_blocks2 = nn.ModuleList([SerialBlock(dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe2, shared_crpe=self.crpe2) for _ in range(serial_depths[1])])
        self.serial_blocks3 = nn.ModuleList([SerialBlock(dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe3, shared_crpe=self.crpe3) for _ in range(serial_depths[2])])
        self.serial_blocks4 = nn.ModuleList([SerialBlock(dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_cpe=self.cpe4, shared_crpe=self.crpe4) for _ in range(serial_depths[3])])
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([ParallelBlock(dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer, shared_crpes=(self.crpe1, self.crpe2, self.crpe3, self.crpe4)) for _ in range(parallel_depth)])
        else:
            self.parallel_blocks = None
        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = norm_layer(embed_dims[1])
                self.norm3 = norm_layer(embed_dims[2])
            else:
                self.norm2 = self.norm3 = None
            self.norm4 = norm_layer(embed_dims[3])
            if self.parallel_depth > 0:
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
                self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            else:
                self.aggregate = None
                self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token1, std=0.02)
        trunc_normal_(self.cls_token2, std=0.02)
        trunc_normal_(self.cls_token3, std=0.02)
        trunc_normal_(self.cls_token4, std=0.02)
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
        return {'cls_token1', 'cls_token2', 'cls_token3', 'cls_token4'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem1='^cls_token1|patch_embed1|crpe1|cpe1', serial_blocks1='^serial_blocks1\\.(\\d+)', stem2='^cls_token2|patch_embed2|crpe2|cpe2', serial_blocks2='^serial_blocks2\\.(\\d+)', stem3='^cls_token3|patch_embed3|crpe3|cpe3', serial_blocks3='^serial_blocks3\\.(\\d+)', stem4='^cls_token4|patch_embed4|crpe4|cpe4', serial_blocks4='^serial_blocks4\\.(\\d+)', parallel_blocks=[('^parallel_blocks\\.(\\d+)', None), ('^norm|aggregate', (99999,))])
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x0):
        B = x0.shape[0]
        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.grid_size
        x1 = insert_cls(x1, self.cls_token1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.grid_size
        x2 = insert_cls(x2, self.cls_token2)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.grid_size
        x3 = insert_cls(x3, self.cls_token3)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.grid_size
        x4 = insert_cls(x4, self.cls_token4)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        if self.parallel_blocks is None:
            if not torch.jit.is_scripting() and self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                return x4
        for blk in self.parallel_blocks:
            x2, x3, x4 = self.cpe2(x2, (H2, W2)), self.cpe3(x3, (H3, W3)), self.cpe4(x4, (H4, W4))
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])
        if not torch.jit.is_scripting() and self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            return [x2, x3, x4]

    def forward_head(self, x_feat: Union[torch.Tensor, List[torch.Tensor]], pre_logits: bool=False):
        if isinstance(x_feat, list):
            assert self.aggregate is not None
            if self.global_pool == 'avg':
                x = torch.cat([xl[:, 1:].mean(dim=1, keepdim=True) for xl in x_feat], dim=1)
            else:
                x = torch.stack([xl[:, 0] for xl in x_feat], dim=1)
            x = self.aggregate(x).squeeze(dim=1)
        else:
            x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x) ->torch.Tensor:
        if not torch.jit.is_scripting() and self.return_interm_layers:
            return self.forward_features(x)
        else:
            x_feat = self.forward_features(x)
            x = self.forward_head(x_feat)
            return x


class GPSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, locality_strength=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.locality_strength = locality_strength
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.rel_indices: torch.Tensor = torch.zeros(1, 1, 1, 3)

    def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = q @ k.transpose(-2, -1) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)
        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)
        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1
        kernel_size = int(self.num_heads ** 0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= self.locality_strength

    def get_rel_indices(self, num_patches: int) ->torch.Tensor:
        img_size = int(num_patches ** 0.5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices


class MHSA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = q @ k.transpose(-2, -1) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)
        img_size = int(N ** 0.5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** 0.5
        distances = distances
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / N
        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = feature_size[0] // patch_size[0], feature_size[1] // patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ConViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=nn.LayerNorm, local_up_to_layer=3, locality_strength=1.0, use_pos_embed=True):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        embed_dim *= num_heads
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_gpsa=True, locality_strength=locality_strength) if i < local_up_to_layer else Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_gpsa=False)) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        for n, m in self.named_modules():
            if hasattr(m, 'local_init'):
                m.local_init()

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
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for u, blk in enumerate(self.blocks):
            if u == self.local_up_to_layer:
                x = torch.cat((cls_tokens, x), dim=1)
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class Residual(nn.Module):

    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class ConvMixer(nn.Module):

    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, global_pool='avg', act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.grad_checkpointing = False
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size), act_layer(), nn.BatchNorm2d(dim))
        self.blocks = nn.Sequential(*[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'), act_layer(), nn.BatchNorm2d(dim))), nn.Conv2d(dim, dim, kernel_size=1), act_layer(), nn.BatchNorm2d(dim)) for i in range(depth)])
        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^stem', blocks='^blocks\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.pooling(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    elif pool_type == 'avg':
        return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
    elif pool_type == 'max':
        return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
    else:
        assert False, f'Unsupported pool type {pool_type}'


class Downsample2d(nn.Module):
    """ A downsample pooling module supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(self, dim: int, dim_out: int, pool_type: str='avg2', padding: str='', bias: bool=True):
        super().__init__()
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        if pool_type == 'max':
            self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=padding or 1)
        elif pool_type == 'max2':
            self.pool = create_pool2d('max', 2, padding=padding or 0)
        elif pool_type == 'avg':
            self.pool = create_pool2d('avg', kernel_size=3, stride=2, count_include_pad=False, padding=padding or 1)
        else:
            self.pool = create_pool2d('avg', 2, padding=padding or 0)
        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.expand(x)
        return x


class LayerScale2d(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


_NORM_MAP = dict(batchnorm=nn.BatchNorm2d, batchnorm2d=nn.BatchNorm2d, batchnorm1d=nn.BatchNorm1d, groupnorm=GroupNorm, groupnorm1=GroupNorm1, layernorm=LayerNorm, layernorm2d=LayerNorm2d)


_NORM_TYPES = {m for n, m in _NORM_MAP.items()}


def get_norm_layer(norm_layer):
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools.partial))
    norm_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace('_', '')
        norm_layer = _NORM_MAP.get(layer_name, None)
    elif norm_layer in _NORM_TYPES:
        norm_layer = norm_layer
    elif isinstance(norm_layer, types.FunctionType):
        norm_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower().replace('_', '')
        norm_layer = _NORM_MAP.get(type_name, None)
        assert norm_layer is not None, f'No equivalent norm layer for {type_name}'
    if norm_kwargs:
        norm_layer = functools.partial(norm_layer, **norm_kwargs)
    return norm_layer


class ConvNeXtStage(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=7, stride=2, depth=2, dilation=(1, 1), drop_path_rates=None, ls_init_value=1.0, conv_mlp=False, conv_bias=True, act_layer='gelu', norm_layer=None, norm_layer_cl=None):
        super().__init__()
        self.grad_checkpointing = False
        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0
            self.downsample = nn.Sequential(norm_layer(in_chs), create_conv2d(in_chs, out_chs, kernel_size=ds_ks, stride=stride, dilation=dilation[0], padding=pad, bias=conv_bias))
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(in_chs=in_chs, out_chs=out_chs, kernel_size=kernel_size, dilation=dilation[1], drop_path=drop_path_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp, conv_bias=conv_bias, act_layer=act_layer, norm_layer=norm_layer if conv_mlp else norm_layer_cl))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


to_ntuple = _ntuple


class ConvNeXt(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), kernel_sizes=7, ls_init_value=1e-06, stem_type='patch', patch_size=4, head_init_scale=1.0, head_norm_first=False, conv_mlp=False, conv_bias=True, act_layer='gelu', norm_layer=None, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
        else:
            assert conv_mlp, 'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []
        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias), norm_layer(dims[0]))
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias), nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias), norm_layer(dims[0]))
            stem_stride = 4
        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(prev_chs, out_chs, kernel_size=kernel_sizes[i], stride=stride, dilation=(first_dilation, dilation), depth=depths[i], drop_path_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp, conv_bias=conv_bias, act_layer=act_layer, norm_layer=norm_layer, norm_layer_cl=norm_layer_cl))
            prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([('global_pool', SelectAdaptivePool2d(pool_type=global_pool)), ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)), ('flatten', nn.Flatten(1) if global_pool else nn.Identity()), ('drop', nn.Dropout(self.drop_rate)), ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks='^stages\\.(\\d+)' if coarse else [('^stages\\.(\\d+)\\.downsample', (0,)), ('^stages\\.(\\d+)\\.blocks\\.(\\d+)', None), ('^norm_pre', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        return x


def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func


@register_notrace_function
def cal_rel_pos_type(attn: torch.Tensor, q: torch.Tensor, has_cls_token: bool, q_size: List[int], k_size: List[int], rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    B, n_head, q_N, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum('byhwc,hkc->byhwk', r_q, Rh)
    rel_w = torch.einsum('byhwc,wkc->byhwk', r_q, Rw)
    attn[:, :, sp_idx:, sp_idx:] = (attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]).view(B, -1, q_h * q_w, k_h * k_w)
    return attn


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


@register_notrace_function
def reshape_post_pool(x, num_heads: int, cls_tok: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, List[int]]:
    feat_size = [x.shape[2], x.shape[3]]
    L_pooled = x.shape[2] * x.shape[3]
    x = x.reshape(-1, num_heads, x.shape[1], L_pooled).transpose(2, 3)
    if cls_tok is not None:
        x = torch.cat((cls_tok, x), dim=2)
    return x, feat_size


@register_notrace_function
def reshape_pre_pool(x, feat_size: List[int], has_cls_token: bool=True) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    H, W = feat_size
    if has_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    else:
        cls_tok = None
    x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return x, cls_tok


class MultiScaleAttention(nn.Module):

    def __init__(self, dim, dim_out, feat_size, num_heads=8, qkv_bias=True, mode='conv', kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1), has_cls_token=True, rel_pos_type='spatial', residual_pooling=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])
        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        if mode in ('avg', 'max'):
            pool_op = nn.MaxPool2d if mode == 'max' else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == 'conv' or mode == 'conv_unshared':
            dim_conv = dim_out // num_heads if mode == 'conv' else dim_out
            if kernel_q:
                self.pool_q = nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q, padding=padding_q, groups=dim_conv, bias=False)
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f'Unsupported model {mode}')
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)
        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)
        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, _ = reshape_post_pool(v, self.num_heads, v_tok)
        if self.norm_v is not None:
            v = self.norm_v(v)
        attn = q * self.scale @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(attn, q, self.has_cls_token, q_size, k_size, self.rel_pos_h, self.rel_pos_w)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        if self.residual_pooling:
            x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x, q_size


class MultiScaleAttentionPoolFirst(nn.Module):

    def __init__(self, dim, dim_out, feat_size, num_heads=8, qkv_bias=True, mode='conv', kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1), has_cls_token=True, rel_pos_type='spatial', residual_pooling=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])
        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        if mode in ('avg', 'max'):
            pool_op = nn.MaxPool2d if mode == 'max' else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == 'conv' or mode == 'conv_unshared':
            dim_conv = dim // num_heads if mode == 'conv' else dim
            if kernel_q:
                self.pool_q = nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q, padding=padding_q, groups=dim_conv, bias=False)
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False)
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f'Unsupported model {mode}')
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)
        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape
        fold_dim = 1 if self.unshared else self.num_heads
        x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
        q = k = v = x
        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)
        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, v_size = reshape_post_pool(v, self.num_heads, v_tok)
        else:
            v_size = feat_size
        if self.norm_v is not None:
            v = self.norm_v(v)
        q_N = q_size[0] * q_size[1] + int(self.has_cls_token)
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
        q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)
        k_N = k_size[0] * k_size[1] + int(self.has_cls_token)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
        k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)
        v_N = v_size[0] * v_size[1] + int(self.has_cls_token)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
        v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = q * self.scale @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(attn, q, self.has_cls_token, q_size, k_size, self.rel_pos_h, self.rel_pos_w)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        if self.residual_pooling:
            x = x + q
        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x, q_size


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, dim_out, num_heads, feat_size, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0, norm_layer=nn.LayerNorm, kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1), mode='conv', has_cls_token=True, expand_attn=False, pool_first=False, rel_pos_type='spatial', residual_pooling=True):
        super().__init__()
        proj_needed = dim != dim_out
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_token = has_cls_token
        self.norm1 = norm_layer(dim)
        self.shortcut_proj_attn = nn.Linear(dim, dim_out) if proj_needed and expand_attn else None
        if stride_q and prod(stride_q) > 1:
            kernel_skip = [(s + 1 if s > 1 else s) for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.shortcut_pool_attn = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip)
        else:
            self.shortcut_pool_attn = None
        att_dim = dim_out if expand_attn else dim
        attn_layer = MultiScaleAttentionPoolFirst if pool_first else MultiScaleAttention
        self.attn = attn_layer(dim, att_dim, num_heads=num_heads, feat_size=feat_size, qkv_bias=qkv_bias, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv, norm_layer=norm_layer, has_cls_token=has_cls_token, mode=mode, rel_pos_type=rel_pos_type, residual_pooling=residual_pooling)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_dim_out = dim_out
        self.shortcut_proj_mlp = nn.Linear(dim, dim_out) if proj_needed and not expand_attn else None
        self.mlp = Mlp(in_features=att_dim, hidden_features=int(att_dim * mlp_ratio), out_features=mlp_dim_out)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _shortcut_pool(self, x, feat_size: List[int]):
        if self.shortcut_pool_attn is None:
            return x
        if self.has_cls_token:
            cls_tok, x = x[:, :1, :], x[:, 1:, :]
        else:
            cls_tok = None
        B, L, C = x.shape
        H, W = feat_size
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.shortcut_pool_attn(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        if cls_tok is not None:
            x = torch.cat((cls_tok, x), dim=1)
        return x

    def forward(self, x, feat_size: List[int]):
        x_norm = self.norm1(x)
        x_shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        x, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x)
        x_norm = self.norm2(x)
        x_shortcut = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        x = x_shortcut + self.drop_path2(self.mlp(x_norm))
        return x, feat_size_new


def _compute_num_patches(img_size, patches):
    return [(i[0] // p * i[1] // p) for i, p in zip(img_size, patches)]


@register_notrace_function
def scale_image(x, ss: Tuple[int, int], crop_scale: bool=False):
    """
    Pulled out of CrossViT.forward_features to bury conditional logic in a leaf node for FX tracing.
    Args:
        x (Tensor): input image
        ss (tuple[int, int]): height and width to scale to
        crop_scale (bool): whether to crop instead of interpolate to achieve the desired scale. Defaults to False
    Returns:
        Tensor: the "scaled" image batch tensor
    """
    H, W = x.shape[-2:]
    if H != ss[0] or W != ss[1]:
        if crop_scale and ss[0] <= H and ss[1] <= W:
            cu, cl = int(round((H - ss[0]) / 2.0)), int(round((W - ss[1]) / 2.0))
            x = x[:, :, cu:cu + ss[0], cl:cl + ss[1]]
        else:
            x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    return x


class CrossViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, img_scale=(1.0, 1.0), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=((1, 3, 1), (1, 3, 1), (1, 3, 1)), num_heads=(6, 12), mlp_ratio=(2.0, 2.0, 4.0), multi_conv=False, crop_scale=False, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), global_pool='token'):
        super().__init__()
        assert global_pool in ('token', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.img_size = to_2tuple(img_size)
        img_scale = to_2tuple(img_scale)
        self.img_size_scaled = [tuple([int(sj * si) for sj in self.img_size]) for si in img_scale]
        self.crop_scale = crop_scale
        num_patches = _compute_num_patches(self.img_size_scaled, patch_size)
        self.num_branches = len(patch_size)
        self.embed_dim = embed_dim
        self.num_features = sum(embed_dim)
        self.patch_embed = nn.ModuleList()
        for i in range(self.num_branches):
            setattr(self, f'pos_embed_{i}', nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])))
            setattr(self, f'cls_token_{i}', nn.Parameter(torch.zeros(1, 1, embed_dim[i])))
        for im_s, p, d in zip(self.img_size_scaled, patch_size, embed_dim):
            self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        self.pos_drop = nn.Dropout(p=drop_rate)
        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([(nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()) for i in range(self.num_branches)])
        for i in range(self.num_branches):
            trunc_normal_(getattr(self, f'pos_embed_{i}'), std=0.02)
            trunc_normal_(getattr(self, f'cls_token_{i}'), std=0.02)
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
        out = set()
        for i in range(self.num_branches):
            out.add(f'cls_token_{i}')
            pe = getattr(self, f'pos_embed_{i}', None)
            if pe is not None and pe.requires_grad:
                out.add(f'pos_embed_{i}')
        return out

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('token', 'avg')
            self.global_pool = global_pool
        self.head = nn.ModuleList([(nn.Linear(self.embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()) for i in range(self.num_branches)])

    def forward_features(self, x) ->List[torch.Tensor]:
        B = x.shape[0]
        xs = []
        for i, patch_embed in enumerate(self.patch_embed):
            x_ = x
            ss = self.img_size_scaled[i]
            x_ = scale_image(x_, ss, self.crop_scale)
            x_ = patch_embed(x_)
            cls_tokens = self.cls_token_0 if i == 0 else self.cls_token_1
            cls_tokens = cls_tokens.expand(B, -1, -1)
            x_ = torch.cat((cls_tokens, x_), dim=1)
            pos_embed = self.pos_embed_0 if i == 0 else self.pos_embed_1
            x_ = x_ + pos_embed
            x_ = self.pos_drop(x_)
            xs.append(x_)
        for i, blk in enumerate(self.blocks):
            xs = blk(xs)
        xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
        return xs

    def forward_head(self, xs: List[torch.Tensor], pre_logits: bool=False) ->torch.Tensor:
        xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
        if pre_logits or isinstance(self.head[0], nn.Identity):
            return torch.cat([x for x in xs], dim=1)
        return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)

    def forward(self, x):
        xs = self.forward_features(x)
        x = self.forward_head(xs)
        return x


class CrossStage(nn.Module):
    """Cross Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, expand_ratio=1.0, groups=1, first_dilation=None, avg_down=False, down_growth=False, cross_linear=False, block_dpr=None, block_fn=BottleneckBlock, **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
            else:
                self.conv_down = ConvNormActAa(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = nn.Identity()
            prev_chs = in_chs
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition_b = ConvNormAct(prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = x.split(self.expand_chs // 2, dim=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out


class CrossStage3(nn.Module):
    """Cross Stage 3.
    Similar to CrossStage, but with only one transition conv for the output.
    """

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, expand_ratio=1.0, groups=1, first_dilation=None, avg_down=False, down_growth=False, cross_linear=False, block_dpr=None, block_fn=BottleneckBlock, **block_kwargs):
        super(CrossStage3, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if stride != 1 or first_dilation != dilation:
            if avg_down:
                self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
            else:
                self.conv_down = ConvNormActAa(in_chs, down_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs
        self.conv_exp = ConvNormAct(prev_chs, exp_chs, kernel_size=1, apply_act=not cross_linear, **conv_kwargs)
        prev_chs = exp_chs // 2
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs
        self.conv_transition = ConvNormAct(exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        x1, x2 = x.split(self.expand_chs // 2, dim=1)
        x1 = self.blocks(x1)
        out = self.conv_transition(torch.cat([x1, x2], dim=1))
        return out


class DarkStage(nn.Module):
    """DarkNet stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, block_ratio=1.0, bottle_ratio=1.0, groups=1, first_dilation=None, avg_down=False, block_fn=BottleneckBlock, block_dpr=None, **block_kwargs):
        super(DarkStage, self).__init__()
        first_dilation = first_dilation or dilation
        conv_kwargs = dict(act_layer=block_kwargs.get('act_layer'), norm_layer=block_kwargs.get('norm_layer'))
        aa_layer = block_kwargs.pop('aa_layer', None)
        if avg_down:
            self.conv_down = nn.Sequential(nn.AvgPool2d(2) if stride == 2 else nn.Identity(), ConvNormActAa(in_chs, out_chs, kernel_size=1, stride=1, groups=groups, **conv_kwargs))
        else:
            self.conv_down = ConvNormActAa(in_chs, out_chs, kernel_size=3, stride=stride, dilation=first_dilation, groups=groups, aa_layer=aa_layer, **conv_kwargs)
        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(in_chs=prev_chs, out_chs=block_out_chs, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, drop_path=block_dpr[i] if block_dpr is not None else 0.0, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _pad_arg(x, n):
    if not isinstance(x, (tuple, list)):
        x = x,
    curr_n = len(x)
    pad_n = n - curr_n
    if pad_n <= 0:
        return x[:n]
    return tuple(x + (x[-1],) * pad_n)


MATCH_PREV_GROUP = 99999,


def _get_attn_fn(stage_args):
    attn_layer = stage_args.pop('attn_layer')
    attn_kwargs = stage_args.pop('attn_kwargs', None) or {}
    if attn_layer is not None:
        attn_layer = get_attn(attn_layer)
        if attn_kwargs:
            attn_layer = partial(attn_layer, **attn_kwargs)
    return attn_layer, stage_args


def _get_block_fn(stage_args):
    block_type = stage_args.pop('block_type')
    assert block_type in ('dark', 'edge', 'bottle')
    if block_type == 'dark':
        return DarkBlock, stage_args
    elif block_type == 'edge':
        return EdgeBlock, stage_args
    else:
        return BottleneckBlock, stage_args


def _get_stage_fn(stage_args):
    stage_type = stage_args.pop('stage_type')
    assert stage_type in ('dark', 'csp', 'cs3')
    if stage_type == 'dark':
        stage_args.pop('expand_ratio', None)
        stage_args.pop('cross_linear', None)
        stage_args.pop('down_growth', None)
        stage_fn = DarkStage
    elif stage_type == 'csp':
        stage_fn = CrossStage
    else:
        stage_fn = CrossStage3
    return stage_fn, stage_args


def create_csp_stem(in_chans=3, out_chs=32, kernel_size=3, stride=2, pool='', padding='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None):
    stem = nn.Sequential()
    feature_info = []
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    stem_depth = len(out_chs)
    assert stem_depth
    assert stride in (1, 2, 4)
    prev_feat = None
    prev_chs = in_chans
    last_idx = stem_depth - 1
    stem_stride = 1
    for i, chs in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        conv_stride = 2 if i == 0 and stride > 1 or i == last_idx and stride > 2 and not pool else 1
        if conv_stride > 1 and prev_feat is not None:
            feature_info.append(prev_feat)
        stem.add_module(conv_name, ConvNormAct(prev_chs, chs, kernel_size, stride=conv_stride, padding=padding if i == 0 else '', act_layer=act_layer, norm_layer=norm_layer))
        stem_stride *= conv_stride
        prev_chs = chs
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', conv_name]))
    if pool:
        assert stride > 2
        if prev_feat is not None:
            feature_info.append(prev_feat)
        if aa_layer is not None:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            stem.add_module('aa', aa_layer(channels=prev_chs, stride=2))
            pool_name = 'aa'
        else:
            stem.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            pool_name = 'pool'
        stem_stride *= 2
        prev_feat = dict(num_chs=prev_chs, reduction=stem_stride, module='.'.join(['stem', pool_name]))
    feature_info.append(prev_feat)
    return stem, feature_info


class DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d, drop_rate=0.0, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bottleneck_fn(self, xs):
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))
        return bottleneck_output

    def any_requires_grad(self, x):
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, x):

        def closure(*xs):
            return self.bottleneck_fn(xs)
        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception('Memory Efficient not supported in JIT')
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)
        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=BatchNormAct2d, drop_rate=0.0, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, norm_layer=norm_layer, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm_layer=BatchNormAct2d, aa_layer=None):
        super(DenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    return global_pool, fc


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000, in_chans=3, global_pool='avg', bn_size=4, stem_type='', norm_layer=BatchNormAct2d, aa_layer=None, drop_rate=0, memory_efficient=False, aa_stem_only=True):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(DenseNet, self).__init__()
        deep_stem = 'deep' in stem_type
        num_init_features = growth_rate * 2
        if aa_layer is None:
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            stem_pool = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_layer(channels=num_init_features, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = growth_rate
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (growth_rate // 4)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (growth_rate // 4)
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)), ('norm0', norm_layer(stem_chs_1)), ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)), ('norm1', norm_layer(stem_chs_2)), ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)), ('norm2', norm_layer(num_init_features)), ('pool0', stem_pool)]))
        else:
            self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', norm_layer(num_init_features)), ('pool0', stem_pool)]))
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module=f'features.norm{2 if deep_stem else 0}')]
        current_stride = 4
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, norm_layer=norm_layer, drop_rate=drop_rate, memory_efficient=memory_efficient)
            module_name = f'denseblock{i + 1}'
            self.features.add_module(module_name, block)
            num_features = num_features + num_layers * growth_rate
            transition_aa_layer = None if aa_stem_only else aa_layer
            if i != len(block_config) - 1:
                self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.' + module_name)]
                current_stride *= 2
                trans = DenseTransition(num_input_features=num_features, num_output_features=num_features // 2, norm_layer=norm_layer, aa_layer=transition_aa_layer)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2
        self.features.add_module('norm5', norm_layer(num_features))
        self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.norm5')]
        self.num_features = num_features
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^features\\.conv[012]|features\\.norm[012]|features\\.pool[012]', blocks='^features\\.(?:denseblock|transition)(\\d+)' if coarse else [('^features\\.denseblock(\\d+)\\.denselayer(\\d+)', None), ('^features\\.transition(\\d+)', MATCH_PREV_GROUP)])
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class DlaBasic(nn.Module):
    """DLA Basic"""

    def __init__(self, inplanes, planes, stride=1, dilation=1, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, shortcut=None, children: Optional[List[torch.Tensor]]=None):
        if shortcut is None:
            shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu(out)
        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut: Optional[torch.Tensor]=None, children: Optional[List[torch.Tensor]]=None):
        if shortcut is None:
            shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += shortcut
        out = self.relu(out)
        return out


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4, cardinality=8, base_width=4):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes
        self.conv1 = nn.Conv2d(inplanes, mid_planes * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes * scale)
        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=cardinality, bias=False))
            bns.append(nn.BatchNorm2d(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) if self.is_first else None
        self.conv3 = nn.Conv2d(mid_planes * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, shortcut: Optional[torch.Tensor]=None, children: Optional[List[torch.Tensor]]=None):
        if shortcut is None:
            shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        out += shortcut
        out = self.relu(out)
        return out


class DlaRoot(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, shortcut):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x_children: List[torch.Tensor]):
        x = self.conv(torch.cat(x_children, 1))
        x = self.bn(x)
        if self.shortcut:
            x += x_children[0]
        x = self.relu(x)
        return x


class DlaTree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, dilation=1, cardinality=1, base_width=64, level_root=False, root_dim=0, root_kernel_size=1, root_shortcut=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(out_channels))
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_shortcut)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_shortcut=root_shortcut))
            self.tree1 = DlaTree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
            self.root = None
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, shortcut: Optional[torch.Tensor]=None, children: Optional[List[torch.Tensor]]=None):
        if children is None:
            children = []
        bottom = self.downsample(x)
        shortcut = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x


class DLA(nn.Module):

    def __init__(self, levels, channels, output_stride=32, num_classes=1000, in_chans=3, global_pool='avg', cardinality=1, base_width=64, block=DlaBottle2neck, shortcut_root=False, drop_rate=0.0):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        assert output_stride == 32
        self.base_layer = nn.Sequential(nn.Conv2d(in_chans, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        cargs = dict(cardinality=cardinality, base_width=base_width, root_shortcut=shortcut_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False, **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True, **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True, **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True, **cargs)
        self.feature_info = [dict(num_chs=channels[0], reduction=1, module='level0'), dict(num_chs=channels[1], reduction=2, module='level1'), dict(num_chs=channels[2], reduction=4, module='level2'), dict(num_chs=channels[3], reduction=8, module='level3'), dict(num_chs=channels[4], reduction=16, module='level4'), dict(num_chs=channels[5], reduction=32, module='level5')]
        self.num_features = channels[-1]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^base_layer', blocks='^level(\\d+)' if coarse else [('^level(\\d+)\\.tree(\\d+)', None), ('^level(\\d+)\\.root', (2,)), ('^level(\\d+)', (1,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    def forward_features(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.fc(x)
            return self.flatten(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class CatBnAct(nn.Module):

    def __init__(self, in_chs, norm_layer=BatchNormAct2d):
        super(CatBnAct, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self.bn(x)


class BnActConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride, groups=1, norm_layer=BatchNormAct2d):
        super(BnActConv2d, self).__init__()
        self.bn = norm_layer(in_chs, eps=0.001)
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        return self.conv(self.bn(x))


class DualPathBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False
        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=self.key_stride, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc, kernel_size=1)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            if self.c1x1_w_s1 is not None:
                x_s = self.c1x1_w_s1(x_in)
            else:
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):

    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32, global_pool='avg', b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), output_stride=32, num_classes=1000, in_chans=3, drop_rate=0.0, fc_act_layer=nn.ELU):
        super(DPN, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.b = b
        assert output_stride == 32
        norm_layer = partial(BatchNormAct2d, eps=0.001)
        fc_norm_layer = partial(BatchNormAct2d, eps=0.001, act_layer=fc_act_layer, inplace=False)
        bw_factor = 1 if small else 4
        blocks = OrderedDict()
        blocks['conv1_1'] = ConvNormAct(in_chans, num_init_features, kernel_size=3 if small else 7, stride=2, norm_layer=norm_layer)
        blocks['conv1_pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_info = [dict(num_chs=num_init_features, reduction=2, module='features.conv1_1')]
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=4, module=f'features.conv2_{k_sec[0]}')]
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=8, module=f'features.conv3_{k_sec[1]}')]
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=16, module=f'features.conv4_{k_sec[2]}')]
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = k_r * bw // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        self.feature_info += [dict(num_chs=in_chs, reduction=32, module=f'features.conv5_{k_sec[3]}')]
        blocks['conv5_bn_ac'] = CatBnAct(in_chs, norm_layer=fc_norm_layer)
        self.num_features = in_chs
        self.features = nn.Sequential(blocks)
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^features\\.conv1', blocks=[('^features\\.conv(\\d+)' if coarse else '^features\\.conv(\\d+)_(\\d+)', None), ('^features\\.conv5_bn_ac', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.classifier(x)
            return self.flatten(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all you Need" paper.
    Based on the official XCiT code
        - https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-06

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)


class ConvBlock(nn.Module):

    def __init__(self, dim, dim_out=None, kernel_size=7, stride=1, conv_bias=True, expand_ratio=4, ls_init_value=1e-06, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, drop_path=0.0):
        super().__init__()
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out
        self.conv_dw = create_conv2d(dim, dim_out, kernel_size=kernel_size, stride=stride, depthwise=True, bias=conv_bias)
        self.norm = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class CrossCovarianceAttn(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class SplitTransposeBlock(nn.Module):

    def __init__(self, dim, num_scales=1, num_heads=8, expand_ratio=4, use_pos_emb=True, conv_bias=True, qkv_bias=True, ls_init_value=1e-06, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, drop_path=0.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)
        convs = []
        for i in range(self.num_scales):
            convs.append(create_conv2d(width, width, kernel_size=3, depthwise=True, bias=conv_bias))
        self.convs = nn.ModuleList(convs)
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = norm_layer(dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm = norm_layer(dim, eps=1e-06)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((B, H, W)).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class EdgeNeXtStage(nn.Module):

    def __init__(self, in_chs, out_chs, stride=2, depth=2, num_global_blocks=1, num_heads=4, scales=2, kernel_size=7, expand_ratio=4, use_pos_emb=False, downsample_block=False, conv_bias=True, ls_init_value=1.0, drop_path_rates=None, norm_layer=LayerNorm2d, norm_layer_cl=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU):
        super().__init__()
        self.grad_checkpointing = False
        if downsample_block or stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(norm_layer(in_chs), nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2, bias=conv_bias))
            in_chs = out_chs
        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                stage_blocks.append(ConvBlock(dim=in_chs, dim_out=out_chs, stride=stride if downsample_block and i == 0 else 1, conv_bias=conv_bias, kernel_size=kernel_size, expand_ratio=expand_ratio, ls_init_value=ls_init_value, drop_path=drop_path_rates[i], norm_layer=norm_layer_cl, act_layer=act_layer))
            else:
                stage_blocks.append(SplitTransposeBlock(dim=in_chs, num_scales=scales, num_heads=num_heads, expand_ratio=expand_ratio, use_pos_emb=use_pos_emb, conv_bias=conv_bias, ls_init_value=ls_init_value, drop_path=drop_path_rates[i], norm_layer=norm_layer_cl, act_layer=act_layer))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class EdgeNeXt(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', dims=(24, 48, 88, 168), depths=(3, 3, 9, 3), global_block_counts=(0, 1, 1, 1), kernel_sizes=(3, 5, 7, 9), heads=(8, 8, 8, 8), d2_scales=(2, 2, 3, 4), use_pos_emb=(False, True, False, False), ls_init_value=1e-06, head_init_scale=1.0, expand_ratio=4, downsample_block=False, conv_bias=True, stem_type='patch', head_norm_first=False, act_layer=nn.GELU, drop_path_rate=0.0, drop_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        norm_layer = partial(LayerNorm2d, eps=1e-06)
        norm_layer_cl = partial(nn.LayerNorm, eps=1e-06)
        self.feature_info = []
        assert stem_type in ('patch', 'overlap')
        if stem_type == 'patch':
            self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=conv_bias), norm_layer(dims[0]))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=9, stride=4, padding=9 // 2, bias=conv_bias), norm_layer(dims[0]))
        curr_stride = 4
        stages = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        in_chs = dims[0]
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            stages.append(EdgeNeXtStage(in_chs=in_chs, out_chs=dims[i], stride=stride, depth=depths[i], num_global_blocks=global_block_counts[i], num_heads=heads[i], drop_path_rates=dp_rates[i], scales=d2_scales[i], expand_ratio=expand_ratio, kernel_size=kernel_sizes[i], use_pos_emb=use_pos_emb[i], ls_init_value=ls_init_value, downsample_block=downsample_block, conv_bias=conv_bias, norm_layer=norm_layer, norm_layer_cl=norm_layer_cl, act_layer=act_layer))
            in_chs = dims[i]
            self.feature_info += [dict(num_chs=in_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = dims[-1]
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([('global_pool', SelectAdaptivePool2d(pool_type=global_pool)), ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)), ('flatten', nn.Flatten(1) if global_pool else nn.Identity()), ('drop', nn.Dropout(self.drop_rate)), ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks='^stages\\.(\\d+)' if coarse else [('^stages\\.(\\d+)\\.downsample', (0,)), ('^stages\\.(\\d+)\\.blocks\\.(\\d+)', None), ('^norm_pre', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class Stem4(nn.Sequential):

    def __init__(self, in_chs, out_chs, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = 4
        self.add_module('conv1', nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1))
        self.add_module('norm1', norm_layer(out_chs // 2))
        self.add_module('act1', act_layer())
        self.add_module('conv2', nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1))
        self.add_module('norm2', norm_layer(out_chs))
        self.add_module('act2', act_layer())


class Downsample(nn.Module):
    """ Image to Patch Embedding, downsampling between stage1 and stage2
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Flat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ConvMlpWithNorm(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class MetaBlock1d(nn.Module):

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, layer_scale_init_value=1e-05):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class MetaBlock2d(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, drop=0.0, drop_path=0.0, layer_scale_init_value=1e-05):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        self.mlp = ConvMlpWithNorm(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale2d(dim, layer_scale_init_value)
        self.ls2 = LayerScale2d(dim, layer_scale_init_value)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path(self.ls2(self.mlp(x)))
        return x


class EfficientFormerStage(nn.Module):

    def __init__(self, dim, dim_out, depth, downsample=True, num_vit=1, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, norm_layer_cl=nn.LayerNorm, drop=0.0, drop_path=0.0, layer_scale_init_value=1e-05):
        super().__init__()
        self.grad_checkpointing = False
        if downsample:
            self.downsample = Downsample(in_chs=dim, out_chs=dim_out, norm_layer=norm_layer)
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()
        blocks = []
        if num_vit and num_vit >= depth:
            blocks.append(Flat())
        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit and num_vit > remain_idx:
                blocks.append(MetaBlock1d(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer_cl, drop=drop, drop_path=drop_path[block_idx], layer_scale_init_value=layer_scale_init_value))
            else:
                blocks.append(MetaBlock2d(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop, drop_path=drop_path[block_idx], layer_scale_init_value=layer_scale_init_value))
                if num_vit and num_vit == remain_idx:
                    blocks.append(Flat())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class EfficientFormer(nn.Module):

    def __init__(self, depths, embed_dims=None, in_chans=3, num_classes=1000, global_pool='avg', downsamples=None, num_vit=0, mlp_ratios=4, pool_size=3, layer_scale_init_value=1e-05, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, norm_layer_cl=nn.LayerNorm, drop_rate=0.0, drop_path_rate=0.0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.stem = Stem4(in_chans, embed_dims[0], norm_layer=norm_layer)
        prev_dim = embed_dims[0]
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        stages = []
        for i in range(len(depths)):
            stage = EfficientFormerStage(prev_dim, embed_dims[i], depths[i], downsample=downsamples[i], num_vit=num_vit if i == 3 else 0, pool_size=pool_size, mlp_ratio=mlp_ratios, act_layer=act_layer, norm_layer_cl=norm_layer_cl, norm_layer=norm_layer, drop=drop_rate, drop_path=dpr[i], layer_scale_init_value=layer_scale_init_value)
            prev_dim = embed_dims[i]
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        self.num_features = embed_dims[-1]
        self.norm = norm_layer_cl(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters() if 'attention_biases' in k}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^stem', blocks=[('^stages\\.(\\d+)', None), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            return x, x_dist
        else:
            return (x + x_dist) / 2

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


_DEBUG_BUILDER = False


_logger = logging.getLogger('validate')


def _log_info_if(msg, condition):
    if condition:
        _logger.info(msg)


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, channel_min, round_limit=round_limit)


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, output_stride=32, pad_type='', round_chs_fn=round_channels, se_from_exp=False, act_layer=None, norm_layer=None, se_layer=None, drop_path_rate=0.0, feature_location=''):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = get_attn(se_layer)
        try:
            self.se_layer(8, rd_ratio=1.0)
            self.se_has_ratio = True
        except TypeError:
            self.se_has_ratio = False
        self.drop_path_rate = drop_path_rate
        if feature_location == 'depthwise':
            _logger.warning("feature_location=='depthwise' is deprecated, using 'expansion'")
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = _DEBUG_BUILDER
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        if 'force_in_chs' in ba and ba['force_in_chs']:
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    se_ratio /= ba.get('exp_ratio', 1.0)
                if self.se_has_ratio:
                    ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
                else:
                    ba['se_layer'] = self.se_layer
        if bt == 'ir':
            _log_info_if('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = CondConvResidual(**ba) if ba.get('num_experts', 0) else InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            _log_info_if('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        _log_info_if('Building model trunk with %d stages...' % len(model_block_args), self.verbose)
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            feature_info = dict(module='act1', num_chs=in_chs, stage=0, reduction=current_stride, hook_type='forward' if self.feature_location != 'bottleneck' else '')
            self.features.append(feature_info)
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            _log_info_if('Stack: {}'.format(stack_idx), self.verbose)
            assert isinstance(stack_args, list)
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                _log_info_if(' Block: {}'.format(block_idx), self.verbose)
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    block_args['stride'] = 1
                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args) or model_block_args[next_stack_idx][0]['stride'] > 1
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        _log_info_if('  Converting stride to dilation to maintain output_stride=={}'.format(self.output_stride), self.verbose)
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)
                if extract_features:
                    feature_info = dict(stage=stack_idx + 1, reduction=current_stride, **block.feature_info(self.feature_location))
                    module_name = f'blocks.{stack_idx}.{block_idx}'
                    leaf_name = feature_info.get('module', '')
                    feature_info['module'] = '.'.join([module_name, leaf_name]) if leaf_name else module_name
                    self.features.append(feature_info)
                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages


def _init_weight_goog(m, n='', fix_group_fanout=True):
    """ Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(lambda w: nn.init.normal_(w, 0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)


class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None, se_layer=None, drop_rate=0.0, drop_path_rate=0.0, global_pool='avg'):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)
        builder = EfficientNetBuilder(output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_act_layer(self.num_features, inplace=True)
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^conv_stem|bn1', blocks=[('^blocks\\.(\\d+)' if coarse else '^blocks\\.(\\d+)\\.(\\d+)', None), ('conv_head|bn2', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class EfficientNetFeatures(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3, stem_size=32, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None, se_layer=None, drop_rate=0.0, drop_path_rate=0.0):
        super(EfficientNetFeatures, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)
        builder = EfficientNetBuilder(output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate, feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}
        efficientnet_init_weights(self)
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def create_attn(attn_type, channels, **kwargs):
    module_cls = get_attn(attn_type)
    if module_cls is not None:
        return module_cls(channels, **kwargs)
    return None


class FeatureBlock(nn.Module):

    def __init__(self, dim, levels=0, reduction='max', act_layer=nn.GELU):
        super().__init__()
        reductions = levels
        levels = max(1, levels)
        if reduction == 'avg':
            pool_fn = partial(nn.AvgPool2d, kernel_size=2)
        else:
            pool_fn = partial(nn.MaxPool2d, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential()
        for i in range(levels):
            self.blocks.add_module(f'conv{i + 1}', MbConvBlock(dim, act_layer=act_layer))
            if reductions:
                self.blocks.add_module(f'pool{i + 1}', pool_fn())
                reductions -= 1

    def forward(self, x):
        return self.blocks(x)


class WindowAttentionGlobal(nn.Module):

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int], use_global: bool=True, qkv_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_global = use_global
        self.rel_pos = RelPosBias(window_size=window_size, num_heads=num_heads)
        if self.use_global:
            self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q_global: Optional[torch.Tensor]=None):
        B, N, C = x.shape
        if self.use_global and q_global is not None:
            _assert(x.shape[-1] == q_global.shape[-1], 'x and q_global seq lengths should be equal')
            kv = self.qkv(x)
            kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q = q_global.repeat(B // q_global.shape[0], 1, 1, 1)
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.rel_pos(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class GlobalContextVitBlock(nn.Module):

    def __init__(self, dim: int, feat_size: Tuple[int, int], num_heads: int, window_size: int=7, mlp_ratio: float=4.0, use_global: bool=True, qkv_bias: bool=True, layer_scale: Optional[float]=None, proj_drop: float=0.0, attn_drop: float=0.0, drop_path: float=0.0, attn_layer: Callable=WindowAttentionGlobal, act_layer: Callable=nn.GELU, norm_layer: Callable=nn.LayerNorm):
        super().__init__()
        feat_size = to_2tuple(feat_size)
        window_size = to_2tuple(window_size)
        self.window_size = window_size
        self.num_windows = int(feat_size[0] // window_size[0] * (feat_size[1] // window_size[1]))
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, window_size=window_size, use_global=use_global, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _window_attn(self, x, q_global: Optional[torch.Tensor]=None):
        B, H, W, C = x.shape
        x_win = window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_win = self.attn(x_win, q_global)
        x = window_reverse(attn_win, self.window_size, (H, W))
        return x

    def forward(self, x, q_global: Optional[torch.Tensor]=None):
        x = x + self.drop_path1(self.ls1(self._window_attn(self.norm1(x), q_global)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class GlobalContextVitStage(nn.Module):

    def __init__(self, dim, depth: int, num_heads: int, feat_size: Tuple[int, int], window_size: Tuple[int, int], downsample: bool=True, global_norm: bool=False, stage_norm: bool=False, mlp_ratio: float=4.0, qkv_bias: bool=True, layer_scale: Optional[float]=None, proj_drop: float=0.0, attn_drop: float=0.0, drop_path: Union[List[float], float]=0.0, act_layer: Callable=nn.GELU, norm_layer: Callable=nn.LayerNorm, norm_layer_cl: Callable=LayerNorm2d):
        super().__init__()
        if downsample:
            self.downsample = Downsample2d(dim=dim, dim_out=dim * 2, norm_layer=norm_layer)
            dim = dim * 2
            feat_size = feat_size[0] // 2, feat_size[1] // 2
        else:
            self.downsample = nn.Identity()
        self.feat_size = feat_size
        window_size = to_2tuple(window_size)
        feat_levels = int(math.log2(min(feat_size) / min(window_size)))
        self.global_block = FeatureBlock(dim, feat_levels)
        self.global_norm = norm_layer_cl(dim) if global_norm else nn.Identity()
        self.blocks = nn.ModuleList([GlobalContextVitBlock(dim=dim, num_heads=num_heads, feat_size=feat_size, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, use_global=i % 2 != 0, layer_scale=layer_scale, proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer=act_layer, norm_layer=norm_layer_cl) for i in range(depth)])
        self.norm = norm_layer_cl(dim) if stage_norm else nn.Identity()
        self.dim = dim
        self.feat_size = feat_size
        self.grad_checkpointing = False

    def forward(self, x):
        x = self.downsample(x)
        global_query = self.global_block(x)
        x = x.permute(0, 2, 3, 1)
        global_query = self.global_norm(global_query.permute(0, 2, 3, 1))
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, global_query)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GlobalContextVit(nn.Module):

    def __init__(self, in_chans: int=3, num_classes: int=1000, global_pool: str='avg', img_size: Tuple[int, int]=224, window_ratio: Tuple[int, ...]=(32, 32, 16, 32), window_size: Tuple[int, ...]=None, embed_dim: int=64, depths: Tuple[int, ...]=(3, 4, 19, 5), num_heads: Tuple[int, ...]=(2, 4, 8, 16), mlp_ratio: float=3.0, qkv_bias: bool=True, layer_scale: Optional[float]=None, drop_rate: float=0.0, proj_drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, weight_init='', act_layer: str='gelu', norm_layer: str='layernorm2d', norm_layer_cl: str='layernorm', norm_eps: float=1e-05):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        norm_layer = partial(get_norm_layer(norm_layer), eps=norm_eps)
        norm_layer_cl = partial(get_norm_layer(norm_layer_cl), eps=norm_eps)
        img_size = to_2tuple(img_size)
        feat_size = tuple(d // 4 for d in img_size)
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        num_stages = len(depths)
        self.num_features = int(embed_dim * 2 ** (num_stages - 1))
        if window_size is not None:
            window_size = to_ntuple(num_stages)(window_size)
        else:
            assert window_ratio is not None
            window_size = tuple([(img_size[0] // r, img_size[1] // r) for r in to_ntuple(num_stages)(window_ratio)])
        self.stem = Stem(in_chs=in_chans, out_chs=embed_dim, act_layer=act_layer, norm_layer=norm_layer)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        for i in range(num_stages):
            last_stage = i == num_stages - 1
            stage_scale = 2 ** max(i - 1, 0)
            stages.append(GlobalContextVitStage(dim=embed_dim * stage_scale, depth=depths[i], num_heads=num_heads[i], feat_size=(feat_size[0] // stage_scale, feat_size[1] // stage_scale), window_size=window_size[i], downsample=i != 0, stage_norm=last_stage, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, layer_scale=layer_scale, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer, norm_layer_cl=norm_layer_cl))
        self.stages = nn.Sequential(*stages)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        if weight_init:
            named_apply(partial(self._init_weights, scheme=weight_init), self)

    def _init_weights(self, module, name, scheme='vit'):
        if scheme == 'vit':
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-06)
                    else:
                        nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters() if any(n in k for n in ['relative_position_bias_table', 'rel_pos.mlp'])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^stem', blocks='^stages\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is None:
            global_pool = self.head.global_pool.pool_type
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: torch.Tensor) ->torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class GhostModule(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False), nn.BatchNorm2d(init_channels), nn.ReLU(inplace=True) if relu else nn.Sequential())
        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False), nn.BatchNorm2d(new_channels), nn.ReLU(inplace=True) if relu else nn.Sequential())

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.0):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False), nn.BatchNorm2d(in_chs), nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chs))

    def forward(self, x):
        shortcut = x
        x = self.ghost1(x)
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(shortcut)
        return x


class GhostNet(nn.Module):

    def __init__(self, cfgs, num_classes=1000, width=1.0, in_chans=3, output_stride=32, global_pool='avg', drop_rate=0.2):
        super(GhostNet, self).__init__()
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, 2, 1, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        prev_chs = stem_chs
        stages = nn.ModuleList([])
        block = GhostBottleneck
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layers.append(block(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio))
                prev_chs = out_chs
            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1
        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.pool_dim = prev_chs = out_chs
        self.blocks = nn.Sequential(*stages)
        self.num_features = out_chs = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^conv_stem|bn1', blocks=[('^blocks\\.(\\d+)' if coarse else '^blocks\\.(\\d+)\\.(\\d+)', None), ('conv_head', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.pool_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class Xception65(nn.Module):
    """Modified Aligned Xception.

    NOTE: only the 65 layer version is included here, the 71 layer variant
    was not correct and had no pretrained weights
    """

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg'):
        super(Xception65, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_dilation = 1
            exit_dilation = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_dilation = 1
            exit_dilation = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_dilation = 2
            exit_dilation = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(num_features=32)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(num_features=64)
        self.act2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block1_act = nn.ReLU(inplace=True)
        self.block2 = Block(128, 256, stride=2, start_with_relu=False, norm_layer=norm_layer)
        self.block3 = Block(256, 728, stride=entry_block3_stride, norm_layer=norm_layer)
        self.mid = nn.Sequential(OrderedDict([('block%d' % i, Block(728, 728, stride=1, dilation=middle_dilation, norm_layer=norm_layer)) for i in range(4, 20)]))
        self.block20 = Block(728, (728, 1024, 1024), stride=exit_block20_stride, dilation=exit_dilation[0], norm_layer=norm_layer)
        self.block20_act = nn.ReLU(inplace=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(num_features=1536)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(num_features=1536)
        self.act4 = nn.ReLU(inplace=True)
        self.num_features = 2048
        self.conv5 = SeparableConv2d(1536, self.num_features, 3, stride=1, dilation=exit_dilation[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(num_features=self.num_features)
        self.act5 = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=64, reduction=2, module='act2'), dict(num_chs=128, reduction=4, module='block1_act'), dict(num_chs=256, reduction=8, module='block3.rep.act1'), dict(num_chs=728, reduction=16, module='block20.rep.act1'), dict(num_chs=2048, reduction=32, module='act5')]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^conv[12]|bn[12]', blocks=[('^mid\\.block(\\d+)', None), ('^block(\\d+)', None), ('^conv[345]|bn[345]', (99,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.block1_act(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.mid(x)
        x = self.block20(x)
        x = self.block20_act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


_BN_MOMENTUM = 0.1


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_in_chs, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_in_chs, num_channels)
        self.num_in_chs = num_in_chs
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_in_chs, num_channels):
        error_msg = ''
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
        elif num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
        elif num_branches != len(num_in_chs):
            error_msg = 'NUM_BRANCHES({}) <> num_in_chs({})'.format(num_branches, len(num_in_chs))
        if error_msg:
            _logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_in_chs[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_in_chs[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=_BN_MOMENTUM))
        layers = [block(self.num_in_chs[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_in_chs[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_in_chs[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()
        num_branches = self.num_branches
        num_in_chs = self.num_in_chs
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_in_chs[j], num_in_chs[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_in_chs[i], momentum=_BN_MOMENTUM), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_in_chs[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_in_chs[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_in_chs[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_in_chs[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3, momentum=_BN_MOMENTUM), nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_in_chs(self):
        return self.num_in_chs

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])
        x_fuse = []
        for i, fuse_outer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_outer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + fuse_outer[j](x[j])
            x_fuse.append(self.fuse_act(y))
        return x_fuse


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, act_layer='leaky_relu', aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=0.001)
        if stride == 1:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=0.001)
        elif aa_layer is None:
            self.conv2 = conv2d_iabn(planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=0.001)
        else:
            self.conv2 = nn.Sequential(conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=0.001), aa_layer(channels=planes, filt_size=3, stride=2))
        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, rd_channels=reduction_chs) if use_se else None
        self.conv3 = conv2d_iabn(planes, planes * self.expansion, kernel_size=1, stride=1, act_layer='identity')
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + shortcut
        out = self.act(out)
        return out


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, head='classification'):
        super(HighResolutionNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        stem_width = cfg['STEM_WIDTH']
        self.conv1 = nn.Conv2d(in_chans, stem_width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_width, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.head = head
        self.head_channels = None
        if head == 'classification':
            self.num_features = 2048
            self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
            self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        elif head == 'incre':
            self.num_features = 2048
            self.incre_modules, _, _ = self._make_head(pre_stage_channels, True)
        else:
            self.incre_modules = None
            self.num_features = 256
        curr_stride = 2
        self.feature_info = [dict(num_chs=64, reduction=curr_stride, module='stem')]
        for i, c in enumerate(self.head_channels if self.head_channels else num_channels):
            curr_stride *= 2
            c = c * 4 if self.head_channels else c
            self.feature_info += [dict(num_chs=c, reduction=curr_stride, module=f'stage{i + 1}')]
        self.init_weights()

    def _make_head(self, pre_stage_channels, incre_only=False):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_modules.append(self._make_layer(head_block, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)
        if incre_only:
            return incre_modules, None, None
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=self.head_channels[3] * head_block.expansion, out_channels=self.num_features, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.num_features, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=_BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_in_chs, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_in_chs, num_channels, fuse_method, reset_multi_scale_output))
            num_in_chs = modules[-1].get_num_in_chs()
        return nn.Sequential(*modules), num_in_chs

    @torch.jit.ignore
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^conv[12]|bn[12]', blocks='^(?:layer|stage|transition)(\\d+)' if coarse else [('^layer(\\d+)\\.(\\d+)', None), ('^stage(\\d+)\\.(\\d+)', None), ('^transition(\\d+)', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def stages(self, x) ->List[torch.Tensor]:
        x = self.layer1(x)
        xl = [t(x) for i, t in enumerate(self.transition1)]
        yl = self.stage2(xl)
        xl = [(t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]) for i, t in enumerate(self.transition2)]
        yl = self.stage3(xl)
        xl = [(t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]) for i, t in enumerate(self.transition3)]
        yl = self.stage4(xl)
        return yl

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        yl = self.stages(x)
        if self.incre_modules is None or self.downsamp_modules is None:
            return yl
        y = self.incre_modules[0](yl[0])
        for i, down in enumerate(self.downsamp_modules):
            y = self.incre_modules[i + 1](yl[i + 1]) + down(y)
        y = self.final_layer(y)
        return y

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        y = self.forward_features(x)
        x = self.forward_head(y)
        return x


class HighResolutionNetFeatures(HighResolutionNet):
    """HighResolutionNet feature extraction

    The design of HRNet makes it easy to grab feature maps, this class provides a simple wrapper to do so.
    It would be more complicated to use the FeatureNet helpers.

    The `feature_location=incre` allows grabbing increased channel count features using part of the
    classification head. If `feature_location=''` the default HRNet features are returned. First stem
    conv is used for stride 2 features.
    """

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, feature_location='incre', out_indices=(0, 1, 2, 3, 4)):
        assert feature_location in ('incre', '')
        super(HighResolutionNetFeatures, self).__init__(cfg, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool, drop_rate=drop_rate, head=feature_location)
        self.feature_info = FeatureInfo(self.feature_info, out_indices)
        self._out_idx = {i for i in out_indices}

    def forward_features(self, x):
        assert False, 'Not supported'

    def forward(self, x) ->List[torch.tensor]:
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if 0 in self._out_idx:
            out.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.stages(x)
        if self.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.incre_modules)]
        for i, f in enumerate(x):
            if i + 1 in self._out_idx:
                out.append(f)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, no_relu=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.relu = None if no_relu else nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.relu is not None:
            out = self.relu(out)
        return out


def flatten_modules(named_modules, depth=1, prefix='', module_types='sequential'):
    prefix_is_tuple = isinstance(prefix, tuple)
    if isinstance(module_types, str):
        if module_types == 'container':
            module_types = nn.Sequential, nn.ModuleList, nn.ModuleDict
        else:
            module_types = nn.Sequential,
    for name, module in named_modules:
        if depth and isinstance(module, module_types):
            yield from flatten_modules(module.named_children(), depth - 1, prefix=(name,) if prefix_is_tuple else name, module_types=module_types)
        elif prefix_is_tuple:
            name = prefix + (name,)
            yield name, module
        else:
            if prefix:
                name = '.'.join([prefix, name])
            yield name, module


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, output_stride=32, global_pool='avg'):
        super(InceptionResnetV2, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        assert output_stride == 32
        self.conv2d_1a = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.feature_info = [dict(num_chs=64, reduction=2, module='conv2d_2b')]
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.feature_info += [dict(num_chs=192, reduction=4, module='conv2d_4a')]
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.feature_info += [dict(num_chs=320, reduction=8, module='repeat')]
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.feature_info += [dict(num_chs=1088, reduction=16, module='repeat_1')]
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(no_relu=True)
        self.conv2d_7b = BasicConv2d(2080, self.num_features, kernel_size=1, stride=1)
        self.feature_info += [dict(num_chs=self.num_features, reduction=32, module='conv2d_7b')]
        self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        module_map = {k: i for i, (k, _) in enumerate(flatten_modules(self.named_children(), prefix=()))}
        module_map.pop(('classif',))

        def _matcher(name):
            if any([name.startswith(n) for n in ('conv2d_1', 'conv2d_2')]):
                return 0
            elif any([name.startswith(n) for n in ('conv2d_3', 'conv2d_4')]):
                return 1
            elif any([name.startswith(n) for n in ('block8', 'conv2d_7')]):
                return len(module_map) + 1
            else:
                for k in module_map.keys():
                    if k == tuple(name.split('.')[:len(k)]):
                        return module_map[k]
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.classif

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classif = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classif(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class InceptionA(nn.Module):

    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionB(nn.Module):

    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionC(nn.Module):

    def __init__(self):
        super(InceptionC, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    """Inception-V3 with no AuxLogits
    FIXME two class defs are redundant, but less screwing around with torchsript fussyness and inconsistent returns
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg', aux_logits=False):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(in_chans, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        else:
            self.AuxLogits = None
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.feature_info = [dict(num_chs=64, reduction=2, module='Conv2d_2b_3x3'), dict(num_chs=192, reduction=4, module='Conv2d_4a_3x3'), dict(num_chs=288, reduction=8, module='Mixed_5d'), dict(num_chs=768, reduction=16, module='Mixed_6e'), dict(num_chs=2048, reduction=32, module='Mixed_7c')]
        self.num_features = 2048
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                trunc_normal_(m.weight, std=stddev)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        module_map = {k: i for i, (k, _) in enumerate(flatten_modules(self.named_children(), prefix=()))}
        module_map.pop(('fc',))

        def _matcher(name):
            if any([name.startswith(n) for n in ('Conv2d_1', 'Conv2d_2')]):
                return 0
            elif any([name.startswith(n) for n in ('Conv2d_3', 'Conv2d_4')]):
                return 1
            else:
                for k in module_map.keys():
                    if k == tuple(name.split('.')[:len(k)]):
                        return module_map[k]
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_preaux(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Pool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.Pool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x

    def forward_postaux(self, x):
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def forward_features(self, x):
        x = self.forward_preaux(x)
        x = self.forward_postaux(x)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class InceptionV3Aux(InceptionV3):
    """InceptionV3 with AuxLogits
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg', aux_logits=True):
        super(InceptionV3Aux, self).__init__(num_classes, in_chans, drop_rate, global_pool, aux_logits)

    def forward_features(self, x):
        x = self.forward_preaux(x)
        aux = self.AuxLogits(x) if self.training else None
        x = self.forward_postaux(x)
        return x, aux

    def forward(self, x):
        x, aux = self.forward_features(x)
        x = self.forward_head(x)
        return x, aux


class Mixed3a(nn.Module):

    def __init__(self):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed4a(nn.Module):

    def __init__(self):
        super(Mixed4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed5a(nn.Module):

    def __init__(self):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):

    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, drop_rate=0.0, global_pool='avg'):
        super(InceptionV4, self).__init__()
        assert output_stride == 32
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 1536
        self.features = nn.Sequential(BasicConv2d(in_chans, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed3a(), Mixed4a(), Mixed5a(), InceptionA(), InceptionA(), InceptionA(), InceptionA(), ReductionA(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), InceptionB(), ReductionB(), InceptionC(), InceptionC(), InceptionC())
        self.feature_info = [dict(num_chs=64, reduction=2, module='features.2'), dict(num_chs=160, reduction=4, module='features.3'), dict(num_chs=384, reduction=8, module='features.9'), dict(num_chs=1024, reduction=16, module='features.17'), dict(num_chs=1536, reduction=32, module='features.21')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^features\\.[012]\\.', blocks='^features\\.(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ConvNorm(nn.Sequential):

    def __init__(self, in_chs, out_chs, kernel_size=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_chs, out_chs, kernel_size, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chs))
        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1), w.size(0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LinearNorm(nn.Sequential):

    def __init__(self, in_features, out_features, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', nn.Linear(in_features, out_features, bias=False))
        self.add_module('bn', nn.BatchNorm1d(out_features))
        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        x = self.c(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class NormLinear(nn.Sequential):

    def __init__(self, in_features, out_features, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_features))
        self.add_module('l', nn.Linear(in_features, out_features, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if self.l.bias is not None:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Subsample(nn.Module):

    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[:, ::self.stride, ::self.stride]
        return x.reshape(B, -1, C)


class AttentionSubsample(nn.Module):
    ab: Dict[str, torch.Tensor]

    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2, act_layer=None, stride=2, resolution=14, resolution_out=7, use_conv=False):
        super().__init__()
        self.stride = stride
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * self.num_heads
        self.resolution = resolution
        self.resolution_out_area = resolution_out ** 2
        self.use_conv = use_conv
        if self.use_conv:
            ln_layer = ConvNorm
            sub_layer = partial(nn.AvgPool2d, kernel_size=1, padding=0)
        else:
            ln_layer = LinearNorm
            sub_layer = partial(Subsample, resolution=resolution)
        self.kv = ln_layer(in_dim, self.val_attn_dim + self.key_attn_dim, resolution=resolution)
        self.q = nn.Sequential(sub_layer(stride=stride), ln_layer(in_dim, self.key_attn_dim, resolution=resolution_out))
        self.proj = nn.Sequential(act_layer(), ln_layer(self.val_attn_dim, out_dim, resolution=resolution_out))
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, self.resolution ** 2))
        k_pos = torch.stack(torch.meshgrid(torch.arange(resolution), torch.arange(resolution))).flatten(1)
        q_pos = torch.stack(torch.meshgrid(torch.arange(0, resolution, step=stride), torch.arange(0, resolution, step=stride))).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = rel_pos[0] * resolution + rel_pos[1]
        self.register_buffer('attention_bias_idxs', rel_pos)
        self.ab = {}

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.ab:
            self.ab = {}

    def get_attention_biases(self, device: torch.device) ->torch.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.ab:
                self.ab[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.ab[device_key]

    def forward(self, x):
        if self.use_conv:
            B, C, H, W = x.shape
            k, v = self.kv(x).view(B, self.num_heads, -1, H * W).split([self.key_dim, self.val_dim], dim=2)
            q = self.q(x).view(B, self.num_heads, self.key_dim, self.resolution_out_area)
            attn = q.transpose(-2, -1) @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).reshape(B, -1, self.resolution, self.resolution)
        else:
            B, N, C = x.shape
            k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 1, 3)
            q = self.q(x).view(B, self.resolution_out_area, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
            attn = q @ k * self.scale + self.get_attention_biases(x.device)
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
        x = self.proj(x)
        return x


def stem_b16(in_chs, out_chs, activation, resolution=224):
    return nn.Sequential(ConvNorm(in_chs, out_chs // 8, 3, 2, 1, resolution=resolution), activation(), ConvNorm(out_chs // 8, out_chs // 4, 3, 2, 1, resolution=resolution // 2), activation(), ConvNorm(out_chs // 4, out_chs // 2, 3, 2, 1, resolution=resolution // 4), activation(), ConvNorm(out_chs // 2, out_chs, 3, 2, 1, resolution=resolution // 8))


class Levit(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage

    NOTE: distillation is defaulted to True since pretrained weights use it, will cause problems
    w/ train scripts that don't take tuple outputs,
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=(192,), key_dim=64, depth=(12,), num_heads=(3,), attn_ratio=2, mlp_ratio=2, hybrid_backbone=None, down_ops=None, act_layer='hard_swish', attn_act_layer='hard_swish', use_conv=False, global_pool='avg', drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        act_layer = get_act_layer(act_layer)
        attn_act_layer = get_act_layer(attn_act_layer)
        ln_layer = ConvNorm if use_conv else LinearNorm
        self.use_conv = use_conv
        if isinstance(img_size, tuple):
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.grad_checkpointing = False
        num_stages = len(embed_dim)
        assert len(depth) == len(num_heads) == num_stages
        key_dim = to_ntuple(num_stages)(key_dim)
        attn_ratio = to_ntuple(num_stages)(attn_ratio)
        mlp_ratio = to_ntuple(num_stages)(mlp_ratio)
        down_ops = down_ops or (('Subsample', key_dim[0], embed_dim[0] // key_dim[0], 4, 2, 2), ('Subsample', key_dim[0], embed_dim[1] // key_dim[1], 4, 2, 2), ('',))
        self.patch_embed = hybrid_backbone or stem_b16(in_chans, embed_dim[0], activation=act_layer)
        self.blocks = []
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(Residual(Attention(ed, kd, nh, attn_ratio=ar, act_layer=attn_act_layer, resolution=resolution, use_conv=use_conv), drop_path_rate))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(Residual(nn.Sequential(ln_layer(ed, h, resolution=resolution), act_layer(), ln_layer(h, ed, bn_weight_init=0, resolution=resolution)), drop_path_rate))
            if do[0] == 'Subsample':
                resolution_out = (resolution - 1) // do[5] + 1
                self.blocks.append(AttentionSubsample(*embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2], attn_ratio=do[3], act_layer=attn_act_layer, stride=do[5], resolution=resolution, resolution_out=resolution_out, use_conv=use_conv))
                resolution = resolution_out
                if do[4] > 0:
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(Residual(nn.Sequential(ln_layer(embed_dim[i + 1], h, resolution=resolution), act_layer(), ln_layer(h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution)), drop_path_rate))
        self.blocks = nn.Sequential(*self.blocks)
        self.head = NormLinear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None, distillation=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if not self.use_conv:
            x = x.flatten(2).transpose(1, 2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class LevitDistilled(Levit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dist = NormLinear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False

    @torch.jit.ignore
    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=None, distillation=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = NormLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = NormLinear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_head(self, x):
        if self.global_pool == 'avg':
            x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            return x, x_dist
        else:
            return (x + x_dist) / 2


class Attention2d(nn.Module):
    """ multi-head attention for 2D NCHW tensors"""

    def __init__(self, dim: int, dim_out: Optional[int]=None, dim_head: int=32, bias: bool=True, expand_first: bool=True, head_first: bool=True, rel_pos_cls: Callable=None, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor]=None):
        B, C, H, W = x.shape
        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1).chunk(3, dim=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1).unbind(1)
        attn = q.transpose(-2, -1) @ k * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionCl(nn.Module):
    """ Channels-last multi-head attention (B, ..., C) """

    def __init__(self, dim: int, dim_out: Optional[int]=None, dim_head: int=32, bias: bool=True, expand_first: bool=True, head_first: bool=True, rel_pos_cls: Callable=None, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, dim_attn * 3, bias=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor]=None):
        B = x.shape[0]
        restore_shape = x.shape[:-1]
        if self.head_first:
            q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)
        else:
            q, k, v = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(1, 3).unbind(2)
        attn = q @ k.transpose(-2, -1) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _init_transformer(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                if 'mlp' in name:
                    nn.init.normal_(module.bias, std=1e-06)
                else:
                    nn.init.zeros_(module.bias)


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


@register_notrace_function
def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition_nchw(x, grid_size: List[int]):
    B, C, H, W = x.shape
    _assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    _assert(W % grid_size[1] == 0, '')
    x = x.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    windows = x.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, C, grid_size[0], grid_size[1])
    return windows


@register_notrace_function
def grid_reverse_nchw(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], C, grid_size[0], grid_size[1])
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous().view(-1, C, H, W)
    return x


def window_partition_nchw(x, window_size: List[int]):
    B, C, H, W = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, '')
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


@register_notrace_function
def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def extend_tuple(x, n):
    if not isinstance(x, (tuple, list)):
        x = x,
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n


class NormMlpHead(nn.Module):

    def __init__(self, in_features, num_classes, hidden_size=None, pool_type='avg', drop_rate=0.0, norm_layer=nn.LayerNorm, act_layer=nn.Tanh):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_features = in_features
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(in_features, hidden_size)), ('act', act_layer())]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(self.drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, drop=0.0, drop_path=0.0):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """

    def __init__(self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=Affine, act_layer=nn.GELU, init_values=0.0001, drop=0.0, drop_path=0.0):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        nn.init.normal_(self.proj.weight, std=1e-06)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, drop=0.0, drop_path=0.0):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(self, num_classes=1000, img_size=224, in_chans=3, patch_size=16, num_blocks=8, embed_dim=512, mlp_ratio=(0.5, 4.0), block_layer=MixerBlock, mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, drop_rate=0.0, drop_path_rate=0.0, nlhb=False, stem_norm=False, global_pool='avg'):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.stem = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        self.blocks = nn.Sequential(*[block_layer(embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate) for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.0
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head(x)
        return x


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280, head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True, round_chs_fn=round_channels, drop_rate=0.0, drop_path_rate=0.0, global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)
        builder = EfficientNetBuilder(output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp, act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^conv_stem|bn1', blocks='^blocks\\.(\\d+)' if coarse else '^blocks\\.(\\d+)\\.(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.flatten(x)
            if self.drop_rate > 0.0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3, stem_size=16, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels, se_from_exp=True, act_layer=None, norm_layer=None, se_layer=None, drop_rate=0.0, drop_path_rate=0.0):
        super(MobileNetV3Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp, act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate, feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}
        efficientnet_init_weights(self)
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) ->List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `https://arxiv.org/abs/2206.02680`
    This layer can be used for self- as well as cross-attention.
    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_drop (float): Dropout value for context scores. Default: 0.0
        bias (bool): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(self, embed_dim: int, attn_drop: float=0.0, proj_drop: float=0.0, bias: bool=True) ->None:
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(in_channels=embed_dim, out_channels=1 + 2 * embed_dim, bias=bias, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, bias=bias, kernel_size=1)
        self.out_drop = nn.Dropout(proj_drop)

    def _forward_self_attn(self, x: torch.Tensor) ->torch.Tensor:
        qkv = self.qkv_proj(x)
        query, key, value = qkv.split([1, self.embed_dim, self.embed_dim], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    @torch.jit.ignore()
    def _forward_cross_attn(self, x: torch.Tensor, x_prev: Optional[torch.Tensor]=None) ->torch.Tensor:
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert kv_patch_area == q_patch_area, 'The number of pixels in a patch for query and key_value should be the same'
        qk = F.conv2d(x_prev, weight=self.qkv_proj.weight[:self.embed_dim + 1], bias=self.qkv_proj.bias[:self.embed_dim + 1])
        query, key = qk.split([1, self.embed_dim], dim=1)
        value = F.conv2d(x, weight=self.qkv_proj.weight[self.embed_dim + 1], bias=self.qkv_proj.bias[self.embed_dim + 1] if self.qkv_proj.bias is not None else None)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor]=None) ->torch.Tensor:
        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)


class LinearTransformerBlock(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        mlp_ratio (float): Inner dimension ratio of the FFN relative to embed_dim
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout rate for attention in multi-head attention. Default: 0.0
        drop_path (float): Stochastic depth rate Default: 0.0
        norm_layer (Callable): Normalization layer. Default: layer_norm_2d
    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(self, embed_dim: int, mlp_ratio: float=2.0, drop: float=0.0, attn_drop: float=0.0, drop_path: float=0.0, act_layer=None, norm_layer=None) ->None:
        super().__init__()
        act_layer = act_layer or nn.SiLU
        norm_layer = norm_layer or GroupNorm1
        self.norm1 = norm_layer(embed_dim)
        self.attn = LinearSelfAttention(embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = ConvMlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor]=None) ->torch.Tensor:
        if x_prev is None:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
        else:
            res = x
            x = self.norm1(x)
            x = self.attn(x, x_prev)
            x = self.drop_path1(x) + res
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class MultiScaleVitStage(nn.Module):

    def __init__(self, dim, dim_out, depth, num_heads, feat_size, mlp_ratio=4.0, qkv_bias=True, mode='conv', kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1), has_cls_token=True, expand_attn=False, pool_first=False, rel_pos_type='spatial', residual_pooling=True, norm_layer=nn.LayerNorm, drop_path=0.0):
        super().__init__()
        self.grad_checkpointing = False
        self.blocks = nn.ModuleList()
        if expand_attn:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)
        for i in range(depth):
            attention_block = MultiScaleBlock(dim=dim, dim_out=out_dims[i], num_heads=num_heads, feat_size=feat_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q if i == 0 else (1, 1), stride_kv=stride_kv, mode=mode, has_cls_token=has_cls_token, pool_first=pool_first, rel_pos_type=rel_pos_type, residual_pooling=residual_pooling, expand_attn=expand_attn, norm_layer=norm_layer, drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path)
            dim = out_dims[i]
            self.blocks.append(attention_block)
            if i == 0:
                feat_size = tuple([(size // stride) for size, stride in zip(feat_size, stride_q)])
        self.feat_size = feat_size

    def forward(self, x, feat_size: List[int]):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, feat_size = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x, feat_size = blk(x, feat_size)
        return x, feat_size


class ActConvBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=''):
        super(ActConvBn, self).__init__()
        self.act = nn.ReLU()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, stem_cell=False, padding=''):
        super(BranchSeparables, self).__init__()
        middle_channels = out_channels if stem_cell else in_channels
        self.act_1 = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, middle_channels, kernel_size, stride=stride, padding=padding)
        self.bn_sep_1 = nn.BatchNorm2d(middle_channels, eps=0.001)
        self.act_2 = nn.ReLU()
        self.separable_2 = SeparableConv2d(middle_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act_1(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.act_2(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellBase(nn.Module):

    def cell_forward(self, x_left, x_right):
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_2)
        x_comb_iter_3_right = self.comb_iter_3_right(x_right)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_left)
        if self.comb_iter_4_right is not None:
            x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        else:
            x_comb_iter_4_right = x_right
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem0(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(CellStem0, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(in_chs_left, out_chs_left, kernel_size=5, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_0_right = nn.Sequential(OrderedDict([('max_pool', create_pool2d('max', 3, stride=2, padding=pad_type)), ('conv', create_conv2d(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)), ('bn', nn.BatchNorm2d(out_chs_left, eps=0.001))]))
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=7, stride=2, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=2, padding=pad_type)
        self.comb_iter_2_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=5, stride=2, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, stride=2, padding=pad_type)
        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=2, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(in_chs_right, out_chs_right, kernel_size=3, stride=2, stem_cell=True, padding=pad_type)
        self.comb_iter_4_right = ActConvBn(out_chs_right, out_chs_right, kernel_size=1, stride=2, padding=pad_type)

    def forward(self, x_left):
        x_right = self.conv_1x1(x_left)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_size, num_channels, pad_type=''):
        super(CellStem1, self).__init__()
        self.num_channels = num_channels
        self.stem_size = stem_size
        self.conv_1x1 = ActConvBn(2 * self.num_channels, self.num_channels, 1, stride=1)
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_size, self.num_channels // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_channels, eps=0.001, momentum=0.1)
        self.comb_iter_0_left = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(self.num_channels, self.num_channels, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(self.num_channels, self.num_channels, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(self.num_channels, self.num_channels, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.act(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2(x_relu)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(FirstCell, self).__init__()
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1)
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))
        self.path_2 = nn.Sequential()
        self.path_2.add_module('pad', nn.ZeroPad2d((-1, 1, -1, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_chs_left, out_chs_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_chs_left * 2, eps=0.001, momentum=0.1)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

    def forward(self, x, x_prev):
        x_relu = self.act(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2(x_relu)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 1, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_left, out_chs_left, 5, 1, pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_left, out_chs_left, 3, 1, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_left = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type=''):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, 1, stride=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, 1, stride=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_0_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_1_left = create_pool2d('max', 3, 2, padding=pad_type)
        self.comb_iter_1_right = BranchSeparables(out_chs_right, out_chs_right, 7, 2, pad_type)
        self.comb_iter_2_left = create_pool2d('avg', 3, 2, count_include_pad=False, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, 5, 2, pad_type)
        self.comb_iter_3_right = create_pool2d('avg', 3, 1, count_include_pad=False, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_right, out_chs_right, 3, 1, pad_type)
        self.comb_iter_4_right = create_pool2d('max', 3, 2, padding=pad_type)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    """NASNetALarge (6 @ 4032) """

    def __init__(self, num_classes=1000, in_chans=3, stem_size=96, channel_multiplier=2, num_features=4032, output_stride=32, drop_rate=0.0, global_pool='avg', pad_type='same'):
        super(NASNetALarge, self).__init__()
        self.num_classes = num_classes
        self.stem_size = stem_size
        self.num_features = num_features
        self.channel_multiplier = channel_multiplier
        self.drop_rate = drop_rate
        assert output_stride == 32
        channels = self.num_features // 24
        self.conv0 = ConvNormAct(in_channels=in_chans, out_channels=self.stem_size, kernel_size=3, padding=0, stride=2, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)
        self.cell_stem_0 = CellStem0(self.stem_size, num_channels=channels // channel_multiplier ** 2, pad_type=pad_type)
        self.cell_stem_1 = CellStem1(self.stem_size, num_channels=channels // channel_multiplier, pad_type=pad_type)
        self.cell_0 = FirstCell(in_chs_left=channels, out_chs_left=channels // 2, in_chs_right=2 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_1 = NormalCell(in_chs_left=2 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_2 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_3 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_4 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.cell_5 = NormalCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=6 * channels, out_chs_right=channels, pad_type=pad_type)
        self.reduction_cell_0 = ReductionCell0(in_chs_left=6 * channels, out_chs_left=2 * channels, in_chs_right=6 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_6 = FirstCell(in_chs_left=6 * channels, out_chs_left=channels, in_chs_right=8 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_7 = NormalCell(in_chs_left=8 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_8 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_9 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_10 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.cell_11 = NormalCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=12 * channels, out_chs_right=2 * channels, pad_type=pad_type)
        self.reduction_cell_1 = ReductionCell1(in_chs_left=12 * channels, out_chs_left=4 * channels, in_chs_right=12 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_12 = FirstCell(in_chs_left=12 * channels, out_chs_left=2 * channels, in_chs_right=16 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_13 = NormalCell(in_chs_left=16 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_14 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_15 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_16 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.cell_17 = NormalCell(in_chs_left=24 * channels, out_chs_left=4 * channels, in_chs_right=24 * channels, out_chs_right=4 * channels, pad_type=pad_type)
        self.act = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=96, reduction=2, module='conv0'), dict(num_chs=168, reduction=4, module='cell_stem_1.conv_1x1.act'), dict(num_chs=1008, reduction=8, module='reduction_cell_0.conv_1x1.act'), dict(num_chs=2016, reduction=16, module='reduction_cell_1.conv_1x1.act'), dict(num_chs=4032, reduction=32, module='act')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^conv0|cell_stem_[01]', blocks=[('^cell_(\\d+)', None), ('^reduction_cell_0', (6,)), ('^reduction_cell_1', (12,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
        x = self.act(x_cell_17)
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvPool(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, pad_type=''):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        _assert(x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        _assert(x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x


def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C = x.shape
    _assert(H % block_size == 0, '`block_size` must divide input height evenly')
    _assert(W % block_size == 0, '`block_size` must divide input width evenly')
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    return x


@register_notrace_function
def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = x.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)
    return x


class NestLevel(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """

    def __init__(self, num_blocks, block_size, seq_length, num_heads, depth, embed_dim, prev_embed_dim=None, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rates=[], norm_layer=None, act_layer=None, pad_type=''):
        super().__init__()
        self.block_size = block_size
        self.grad_checkpointing = False
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))
        if prev_embed_dim is not None:
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type)
        else:
            self.pool = nn.Identity()
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_encoder = nn.Sequential(*[TransformerLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)
        x = blockify(x, self.block_size)
        x = x + self.pos_embed
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.transformer_encoder, x)
        else:
            x = self.transformer_encoder(x)
        x = deblockify(x, self.block_size)
        return x.permute(0, 3, 1, 2)


def _init_nest_weights(module: nn.Module, name: str='', head_bias: float=0.0):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Nest(nn.Module):
    """ Nested Transformer (NesT)

    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(self, img_size=224, in_chans=3, patch_size=4, num_levels=3, embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 20), num_classes=1000, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.5, norm_layer=None, act_layer=None, pad_type='', weight_init='', global_pool='avg'):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            num_heads (int, tuple): number of attention heads for each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map

        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()
        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'
        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], 'Model only handles square inputs'
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert img_size // patch_size % math.sqrt(self.num_blocks[0]) == 0, "First level blocks don't fit evenly. Check `img_size`, `patch_size`, and `num_levels`"
        self.block_size = int(img_size // patch_size // math.sqrt(self.num_blocks[0]))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], flatten=False)
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(NestLevel(self.num_blocks[i], self.block_size, self.seq_length, num_heads[i], depths[i], dim, prev_dim, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dp_rates[i], norm_layer, act_layer, pad_type=pad_type))
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = dim
            curr_stride *= 2
        self.levels = nn.Sequential(*levels)
        self.norm = norm_layer(embed_dims[-1])
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=0.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {f'level.{i}.pos_embed' for i in range(len(self.levels))}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^patch_embed', blocks=[('^levels\\.(\\d+)' if coarse else '^levels\\.(\\d+)\\.transformer_encoder\\.(\\d+)', None), ('^levels\\.(\\d+)\\.(?:pool|pos_embed)', (0,)), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.levels:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.levels(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def gelu(x: torch.Tensor, inplace: bool=False) ->torch.Tensor:
    return F.gelu(x)


def gelu_tanh(x: torch.Tensor, inplace: bool=False) ->torch.Tensor:
    return F.gelu(x, approximate='tanh')


def sigmoid(x, inplace: bool=False):
    return x.sigmoid_() if inplace else x.sigmoid()


def tanh(x, inplace: bool=False):
    return x.tanh_() if inplace else x.tanh()


_ACT_FN_DEFAULT = dict(silu=F.silu if _has_silu else swish, swish=F.silu if _has_silu else swish, mish=F.mish if _has_mish else mish, relu=F.relu, relu6=F.relu6, leaky_relu=F.leaky_relu, elu=F.elu, celu=F.celu, selu=F.selu, gelu=gelu, gelu_tanh=gelu_tanh, sigmoid=sigmoid, tanh=tanh, hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid, hard_swish=F.hardswish if _has_hardswish else hard_swish, hard_mish=hard_mish)


_ACT_FN_JIT = dict(silu=F.silu if _has_silu else swish_jit, swish=F.silu if _has_silu else swish_jit, mish=F.mish if _has_mish else mish_jit, hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid_jit, hard_swish=F.hardswish if _has_hardswish else hard_swish_jit, hard_mish=hard_mish_jit)


def hard_mish_me(x, inplace: bool=False):
    return HardMishJitAutoFn.apply(x)


def hard_sigmoid_me(x, inplace: bool=False):
    return HardSigmoidJitAutoFn.apply(x)


def hard_swish_me(x, inplace=False):
    return HardSwishJitAutoFn.apply(x)


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


def swish_me(x, inplace=False):
    return SwishJitAutoFn.apply(x)


_ACT_FN_ME = dict(silu=F.silu if _has_silu else swish_me, swish=F.silu if _has_silu else swish_me, mish=F.mish if _has_mish else mish_me, hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid_me, hard_swish=F.hardswish if _has_hardswish else hard_swish_me, hard_mish=hard_mish_me)


def get_act_fn(name: Union[Callable, str]='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if isinstance(name, Callable):
        return name
    if not (is_no_jit() or is_exportable() or is_scriptable()):
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    if not (is_no_jit() or is_exportable()):
        if name in _ACT_FN_JIT:
            return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


class GammaAct(nn.Module):

    def __init__(self, act_type='relu', gamma: float=1.0, inplace=False):
        super().__init__()
        self.act_fn = get_act_fn(act_type)
        self.gamma = gamma
        self.inplace = inplace

    def forward(self, x):
        return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)


class NormFreeBlock(nn.Module):
    """Normalization-Free pre-activation block.
    """

    def __init__(self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None, alpha=1.0, beta=1.0, bottle_ratio=0.25, group_size=None, ch_div=1, reg=True, extra_conv=False, skipinit=False, attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0.0):
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * groups
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain
        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, conv_layer=conv_layer)
        else:
            self.downsample = None
        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if extra_conv:
            self.act2b = act_layer(inplace=True)
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, groups=groups)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1, gain_init=1.0 if skipinit else 0.0)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)
        else:
            self.attn_last = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.0)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)
        if self.skipinit_gain is not None:
            out.mul_(self.skipinit_gain)
        out = out * self.alpha + shortcut
        return out


_nonlin_gamma = dict(identity=1.0, celu=1.270926833152771, elu=1.2716004848480225, gelu=1.7015043497085571, leaky_relu=1.70590341091156, log_sigmoid=1.9193484783172607, log_softmax=1.0002083778381348, relu=1.7139588594436646, relu6=1.7131484746932983, selu=1.0008515119552612, sigmoid=4.803835391998291, silu=1.7881293296813965, softsign=2.338853120803833, softplus=1.9203323125839233, tanh=1.5939117670059204)


def act_with_gamma(act_type, gamma: float=1.0):

    def _create(inplace=False):
        return GammaAct(act_type, gamma=gamma, inplace=inplace)
    return _create


def create_stem(in_chs, out_chs, stem_type='', conv_layer=None, act_layer=None, preact_feature=True):
    stem_stride = 2
    stem_feature = dict(num_chs=out_chs, reduction=2, module='stem.conv')
    stem = OrderedDict()
    assert stem_type in ('', 'deep', 'deep_tiered', 'deep_quad', '3x3', '7x7', 'deep_pool', '3x3_pool', '7x7_pool')
    if 'deep' in stem_type:
        if 'quad' in stem_type:
            assert not 'pool' in stem_type
            stem_chs = out_chs // 8, out_chs // 4, out_chs // 2, out_chs
            strides = 2, 1, 1, 2
            stem_stride = 4
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv3')
        else:
            if 'tiered' in stem_type:
                stem_chs = 3 * out_chs // 8, out_chs // 2, out_chs
            else:
                stem_chs = out_chs // 2, out_chs // 2, out_chs
            strides = 2, 1, 1
            stem_feature = dict(num_chs=out_chs // 2, reduction=2, module='stem.conv2')
        last_idx = len(stem_chs) - 1
        for i, (c, s) in enumerate(zip(stem_chs, strides)):
            stem[f'conv{i + 1}'] = conv_layer(in_chs, c, kernel_size=3, stride=s)
            if i != last_idx:
                stem[f'act{i + 2}'] = act_layer(inplace=True)
            in_chs = c
    elif '3x3' in stem_type:
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=3, stride=2)
    else:
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
    if 'pool' in stem_type:
        stem['pool'] = nn.MaxPool2d(3, stride=2, padding=1)
        stem_stride = 4
    return nn.Sequential(stem), stem_stride, stem_feature


class SequentialTuple(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialTuple, self).__init__(*args)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) ->Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class Transformer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvHeadPooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(ConvHeadPooling, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1, padding=stride // 2, stride=stride, padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token) ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        cls_token = self.fc(cls_token)
        return x, cls_token


class ConvEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingVisionTransformer(nn.Module):
    """ Pooling-based Vision Transformer

    A PyTorch implement of 'Rethinking Spatial Dimensions of Vision Transformers'
        - https://arxiv.org/abs/2103.16302
    """

    def __init__(self, img_size, patch_size, stride, base_dims, depth, heads, mlp_ratio, num_classes=1000, in_chans=3, global_pool='token', distilled=False, attn_drop_rate=0.0, drop_rate=0.0, drop_path_rate=0.0):
        super(PoolingVisionTransformer, self).__init__()
        assert global_pool in ('token',)
        padding = 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        height = math.floor((img_size[0] + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)
        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_tokens = 2 if distilled else 1
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        transformers = []
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage])]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-06)
        self.num_features = self.embed_dim = base_dims[-1] * heads[-1]
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.distilled_training = False
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    def get_classifier(self):
        if self.head_dist is not None:
            return self.head, self.head_dist
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x, cls_tokens = self.transformers((x, cls_tokens))
        cls_tokens = self.norm(cls_tokens)
        return cls_tokens

    def forward_head(self, x, pre_logits: bool=False) ->torch.Tensor:
        if self.head_dist is not None:
            assert self.global_pool == 'token'
            x, x_dist = x[:, 0], x[:, 1]
            if not pre_logits:
                x = self.head(x)
                x_dist = self.head_dist(x_dist)
            if self.distilled_training and self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            if self.global_pool == 'token':
                x = x[:, 0]
            if not pre_logits:
                x = self.head(x)
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class FactorizedReduction(nn.Module):

    def __init__(self, in_channels, out_channels, padding=''):
        super(FactorizedReduction, self).__init__()
        self.act = nn.ReLU()
        self.path_1 = nn.Sequential(OrderedDict([('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding))]))
        self.path_2 = nn.Sequential(OrderedDict([('pad', nn.ZeroPad2d((-1, 1, -1, 1))), ('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False)), ('conv', create_conv2d(in_channels, out_channels // 2, kernel_size=1, padding=padding))]))
        self.final_path_bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.act(x)
        x_path1 = self.path_1(x)
        x_path2 = self.path_2(x)
        out = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        return out


class Cell(CellBase):

    def __init__(self, in_chs_left, out_chs_left, in_chs_right, out_chs_right, pad_type='', is_reduction=False, match_prev_layer_dims=False):
        super(Cell, self).__init__()
        stride = 2 if is_reduction else 1
        self.match_prev_layer_dimensions = match_prev_layer_dims
        if match_prev_layer_dims:
            self.conv_prev_1x1 = FactorizedReduction(in_chs_left, out_chs_left, padding=pad_type)
        else:
            self.conv_prev_1x1 = ActConvBn(in_chs_left, out_chs_left, kernel_size=1, padding=pad_type)
        self.conv_1x1 = ActConvBn(in_chs_right, out_chs_right, kernel_size=1, padding=pad_type)
        self.comb_iter_0_left = BranchSeparables(out_chs_left, out_chs_left, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_0_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_1_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=7, stride=stride, padding=pad_type)
        self.comb_iter_1_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_2_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=5, stride=stride, padding=pad_type)
        self.comb_iter_2_right = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3, stride=stride, padding=pad_type)
        self.comb_iter_3_left = BranchSeparables(out_chs_right, out_chs_right, kernel_size=3)
        self.comb_iter_3_right = create_pool2d('max', 3, stride=stride, padding=pad_type)
        self.comb_iter_4_left = BranchSeparables(out_chs_left, out_chs_left, kernel_size=3, stride=stride, padding=pad_type)
        if is_reduction:
            self.comb_iter_4_right = ActConvBn(out_chs_right, out_chs_right, kernel_size=1, stride=stride, padding=pad_type)
        else:
            self.comb_iter_4_right = None

    def forward(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        x_out = self.cell_forward(x_left, x_right)
        return x_out


class PNASNet5Large(nn.Module):

    def __init__(self, num_classes=1000, in_chans=3, output_stride=32, drop_rate=0.0, global_pool='avg', pad_type=''):
        super(PNASNet5Large, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.num_features = 4320
        assert output_stride == 32
        self.conv_0 = ConvNormAct(in_chans, 96, kernel_size=3, stride=2, padding=0, norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.1), apply_act=False)
        self.cell_stem_0 = CellStem0(in_chs_left=96, out_chs_left=54, in_chs_right=96, out_chs_right=54, pad_type=pad_type)
        self.cell_stem_1 = Cell(in_chs_left=96, out_chs_left=108, in_chs_right=270, out_chs_right=108, pad_type=pad_type, match_prev_layer_dims=True, is_reduction=True)
        self.cell_0 = Cell(in_chs_left=270, out_chs_left=216, in_chs_right=540, out_chs_right=216, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_1 = Cell(in_chs_left=540, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_2 = Cell(in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_3 = Cell(in_chs_left=1080, out_chs_left=216, in_chs_right=1080, out_chs_right=216, pad_type=pad_type)
        self.cell_4 = Cell(in_chs_left=1080, out_chs_left=432, in_chs_right=1080, out_chs_right=432, pad_type=pad_type, is_reduction=True)
        self.cell_5 = Cell(in_chs_left=1080, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_6 = Cell(in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)
        self.cell_7 = Cell(in_chs_left=2160, out_chs_left=432, in_chs_right=2160, out_chs_right=432, pad_type=pad_type)
        self.cell_8 = Cell(in_chs_left=2160, out_chs_left=864, in_chs_right=2160, out_chs_right=864, pad_type=pad_type, is_reduction=True)
        self.cell_9 = Cell(in_chs_left=2160, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type, match_prev_layer_dims=True)
        self.cell_10 = Cell(in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.cell_11 = Cell(in_chs_left=4320, out_chs_left=864, in_chs_right=4320, out_chs_right=864, pad_type=pad_type)
        self.act = nn.ReLU()
        self.feature_info = [dict(num_chs=96, reduction=2, module='conv_0'), dict(num_chs=270, reduction=4, module='cell_stem_1.conv_1x1.act'), dict(num_chs=1080, reduction=8, module='cell_4.conv_1x1.act'), dict(num_chs=2160, reduction=16, module='cell_8.conv_1x1.act'), dict(num_chs=4320, reduction=32, module='act')]
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^conv_0|cell_stem_[01]', blocks='^cell_(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.act(x_cell_11)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Args:
        dim: embedding dim
        pool_size: pooling size
        mlp_ratio: mlp expansion ratio
        act_layer: activation
        norm_layer: normalization
        drop: dropout rate
        drop path: Stochastic Depth, refer to https://arxiv.org/abs/1603.09382
        use_layer_scale, --layer_scale_init_value: LayerScale, refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=GroupNorm1, drop=0.0, drop_path=0.0, layer_scale_init_value=1e-05):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ConvMlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if layer_scale_init_value:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def forward(self, x):
        if self.layer_scale_1 is not None:
            x = x + self.drop_path1(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path2(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, pool_size=3, mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=GroupNorm1, drop_rate=0.0, drop_path_rate=0.0, layer_scale_init_value=1e-05):
    """ generate PoolFormer blocks for a stage """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate, drop_path=block_dpr, layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)
    return blocks


class PoolFormer(nn.Module):
    """ PoolFormer
    """

    def __init__(self, layers, embed_dims=(64, 128, 320, 512), mlp_ratios=(4, 4, 4, 4), downsamples=(True, True, True, True), pool_size=3, in_chans=3, num_classes=1000, global_pool='avg', norm_layer=GroupNorm1, act_layer=nn.GELU, in_patch_size=7, in_stride=4, in_pad=2, down_patch_size=3, down_stride=2, down_pad=1, drop_rate=0.0, drop_path_rate=0.0, layer_scale_init_value=1e-05, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False
        self.patch_embed = PatchEmbed(patch_size=in_patch_size, stride=in_stride, padding=in_pad, in_chs=in_chans, embed_dim=embed_dims[0])
        network = []
        for i in range(len(layers)):
            network.append(basic_blocks(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratios[i], act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate, drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value))
            if i < len(layers) - 1 and (downsamples[i] or embed_dims[i] != embed_dims[i + 1]):
                network.append(PatchEmbed(in_chs=embed_dims[i], embed_dim=embed_dims[i + 1], patch_size=down_patch_size, stride=down_stride, padding=down_pad))
        self.network = nn.Sequential(*network)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^patch_embed', blocks=[('^network\\.(\\d+).*\\.proj', (99999,)), ('^network\\.(\\d+)', None) if coarse else ('^network\\.(\\d+)\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.network(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean([-2, -1])
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MlpWithDepthwiseConv(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, extra_relu=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU() if extra_relu else nn.Identity()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_size: List[int]):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, feat_size[0], feat_size[1])
        x = self.relu(x)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, 'Set larger patch_size than stride'
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        feat_size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, feat_size


class PyramidVisionTransformerStage(nn.Module):

    def __init__(self, dim: int, dim_out: int, depth: int, downsample: bool=True, num_heads: int=8, sr_ratio: int=1, linear_attn: bool=False, mlp_ratio: float=4.0, qkv_bias: bool=True, drop: float=0.0, attn_drop: float=0.0, drop_path: Union[List[float], float]=0.0, norm_layer: Callable=nn.LayerNorm):
        super().__init__()
        self.grad_checkpointing = False
        if downsample:
            self.downsample = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dim, embed_dim=dim_out)
        else:
            assert dim == dim_out
            self.downsample = None
        self.blocks = nn.ModuleList([Block(dim=dim_out, num_heads=num_heads, sr_ratio=sr_ratio, linear_attn=linear_attn, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(dim_out)

    def forward(self, x, feat_size: List[int]) ->Tuple[torch.Tensor, List[int]]:
        if self.downsample is not None:
            x, feat_size = self.downsample(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x = blk(x, feat_size)
        x = self.norm(x)
        x = x.reshape(x.shape[0], feat_size[0], feat_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return x, feat_size


class PyramidVisionTransformerV2(nn.Module):

    def __init__(self, img_size=None, in_chans=3, num_classes=1000, global_pool='avg', depths=(3, 4, 6, 3), embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), sr_ratios=(8, 4, 2, 1), mlp_ratios=(8.0, 8.0, 4.0, 4.0), qkv_bias=True, linear=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        assert global_pool in ('avg', '')
        self.global_pool = global_pool
        self.depths = depths
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)
        assert len(embed_dims) == num_stages
        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        cur = 0
        prev_dim = embed_dims[0]
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(PyramidVisionTransformerStage(dim=prev_dim, dim_out=embed_dims[i], depth=depths[i], downsample=i > 0, num_heads=num_heads[i], sr_ratio=sr_ratios[i], mlp_ratio=mlp_ratios[i], linear_attn=linear, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))
            prev_dim = embed_dims[i]
            cur += depths[i]
        self.num_features = embed_dims[-1]
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^patch_embed', blocks='^stages\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('avg', '')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x, feat_size = self.patch_embed(x)
        for stage in self.stages:
            x, feat_size = stage(x, feat_size=feat_size)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x.mean(dim=(-1, -2))
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PreBottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=(1, 1), bottle_ratio=1, group_size=1, se_ratio=0.25, downsample='conv1x1', linear_out=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_block=None, drop_path_rate=0.0):
        super(PreBottleneck, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size
        self.norm1 = norm_act_layer(in_chs)
        self.conv1 = create_conv2d(in_chs, bottleneck_chs, kernel_size=1)
        self.norm2 = norm_act_layer(bottleneck_chs)
        self.conv2 = create_conv2d(bottleneck_chs, bottleneck_chs, kernel_size=3, stride=stride, dilation=dilation[0], groups=groups)
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.norm3 = norm_act_layer(bottleneck_chs)
        self.conv3 = create_conv2d(bottleneck_chs, out_chs, kernel_size=1)
        self.downsample = create_shortcut(downsample, in_chs, out_chs, 1, stride, dilation, preact=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        pass

    def forward(self, x):
        x = self.norm1(x)
        shortcut = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.norm3(x)
        x = self.conv3(x)
        if self.downsample is not None:
            x = self.drop_path(x) + self.downsample(shortcut)
        return x


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, depth, in_chs, out_chs, stride, dilation, drop_path_rates=None, block_fn=Bottleneck, **block_kwargs):
        super(RegStage, self).__init__()
        self.grad_checkpointing = False
        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = first_dilation, dilation
            dpr = drop_path_rates[i] if drop_path_rates is not None else 0.0
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fn(block_in_chs, out_chs, stride=block_stride, dilation=block_dilation, drop_path_rate=dpr, **block_kwargs))
            first_dilation = dilation

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.children(), x)
        else:
            for block in self.children():
                x = block(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, group_size, q=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % q == 0
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = width_initial * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    groups = np.array([group_size for _ in range(num_stages)])
    return widths.astype(int).tolist(), num_stages, groups.astype(int).tolist()


class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=26, scale=4, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=None, attn_layer=None, **_):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)
        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None
        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None
        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, cardinality=1, base_width=64, avd=False, avd_first=False, is_first=False, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1
        assert attn_layer is None
        assert aa_layer is None
        assert drop_path is None
        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None
        if self.radix >= 1:
            self.conv2 = SplitAttn(group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_layer=drop_block)
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        if self.avd_first is not None:
            out = self.avd_first(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)
        if self.avd_last is not None:
            out = self.avd_last(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.act3(out)
        return out


def drop_blocks(drop_prob=0.0):
    return [None, None, partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None, partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.0) if drop_prob else None]


def make_blocks(block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32, down_kernel_size=1, avg_down=False, drop_block_rate=0.0, drop_path_rate=0.0, **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride
        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size, stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)
        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)
            blocks.append(block_fn(inplanes, planes, stride, downsample, first_dilation=prev_dilation, drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))
    return stages, feature_info


_model_entrypoints = {}


_module_to_models = defaultdict(set)


def split_model_name_tag(model_name: str, no_tag=''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def get_arch_name(model_name: str) ->Tuple[str, Optional[str]]:
    return split_model_name_tag(model_name)[0]


def model_entrypoint(model_name, module_filter: Optional[str]=None):
    """Fetch a model entrypoint for specified model name
    """
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg', cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1, down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=0.0, zero_init_last=True, block_args=None):
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = stem_width, stem_width
            if 'tiered' in stem_type:
                stem_chs = 3 * (stem_width // 4), stem_width
            self.conv1 = nn.Sequential(*[nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False), norm_layer(stem_chs[0]), act_layer(inplace=True), nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False), norm_layer(stem_chs[1]), act_layer(inplace=True), nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False), create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None, norm_layer(inplanes), act_layer(inplace=True)]))
        elif aa_layer is not None:
            if issubclass(aa_layer, nn.AvgPool2d):
                self.maxpool = aa_layer(2)
            else:
                self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_layer(channels=inplanes, stride=2)])
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width, output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down, down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer, drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)
        self.feature_info.extend(stage_feature_info)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.init_weights(zero_init_last=zero_init_last)

    @staticmethod
    def from_pretrained(model_name: str, load_weights=True, **kwargs) ->'ResNet':
        entry_fn = model_entrypoint(model_name, 'resnet')
        return entry_fn(pretrained=not load_weights, **kwargs)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^conv1|bn1|maxpool', blocks='^layer(\\d+)' if coarse else '^layer(\\d+)\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1, act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.0):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)
        if proj_layer is not None:
            self.downsample = proj_layer(in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True, conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        x_preact = self.norm1(x)
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class DownsampleConv(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True, conv_layer=None, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(x))


class ResNetStage(nn.Module):
    """ResNet Stage."""

    def __init__(self, in_chs, out_chs, stride, dilation, depth, bottle_ratio=0.25, groups=1, avg_down=False, block_dpr=None, block_fn=PreActBottleneck, act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.0
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups, first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate, **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self, x):
        x = self.blocks(x)
        return x


def init_weights_vit_jax(module: nn.Module, name: str='', head_bias: float=0.0):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-06) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str=''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_timm(module: nn.Module, name: str=''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float=0.0):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=None, class_token=True, no_embed_class=False, pre_norm=False, fc_norm=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not pre_norm)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-06)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight
    return conv_weight


def resample_abs_pos_embed(posemb, new_size: List[int], old_size: Optional[List[int]]=None, num_prefix_tokens: int=1, interpolation: str='bicubic', antialias: bool=True, verbose: bool=False):
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:
        return posemb
    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)
    if verbose:
        _logger.info(f'Resized position embedding: {old_size} to {new_size}.')
    if posemb_prefix is not None:
        None
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb


def resample_patch_embed(patch_embed, new_size: List[int], interpolation: str='bicubic', antialias: bool=True, verbose: bool=False):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    assert len(patch_embed.shape) == 4, 'Four dimensions expected'
    assert len(new_size) == 2, 'New shape should only be hw'
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed
    if verbose:
        _logger.info(f'Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.')

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.0
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T
    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)
    v_resample_kernel = torch.vmap(torch.vmap(resample_kernel, 0, 0), 1, 1)
    return v_resample_kernel(patch_embed)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str=''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)
    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
    if hasattr(model.patch_embed, 'backbone'):
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(embed_conv_w, model.patch_embed.proj.weight.shape[-2:], interpolation=interpolation, antialias=antialias, verbose=True)
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(pos_embed_w, new_size=model.patch_embed.grid_size, num_prefix_tokens=num_prefix_tokens, interpolation=interpolation, antialias=antialias, verbose=True)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([_n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([_n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))


def is_stem_deep(stem_type):
    return any([(s in stem_type) for s in ('deep', 'tiered')])


def create_resnetv2_stem(in_chs, out_chs=64, stem_type='', preact=True, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32)):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same', 'tiered')
    if is_stem_deep(stem_type):
        if 'tiered' in stem_type:
            stem_chs = 3 * out_chs // 8, out_chs // 2
        else:
            stem_chs = out_chs // 2, out_chs // 2
        stem['conv1'] = conv_layer(in_chs, stem_chs[0], kernel_size=3, stride=2)
        stem['norm1'] = norm_layer(stem_chs[0])
        stem['conv2'] = conv_layer(stem_chs[0], stem_chs[1], kernel_size=3, stride=1)
        stem['norm2'] = norm_layer(stem_chs[1])
        stem['conv3'] = conv_layer(stem_chs[1], out_chs, kernel_size=3, stride=1)
        if not preact:
            stem['norm3'] = norm_layer(out_chs)
    else:
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
        if not preact:
            stem['norm'] = norm_layer(out_chs)
    if 'fixed' in stem_type:
        stem['pad'] = nn.ConstantPad2d(1, 0.0)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        stem['pool'] = create_pool2d('max', kernel_size=3, stride=2, padding='same')
    else:
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    return nn.Sequential(stem)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(self, layers, channels=(256, 512, 1024, 2048), num_classes=1000, in_chans=3, global_pool='avg', output_stride=32, width_factor=1, stem_chs=64, stem_type='', avg_down=False, preact=True, act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32), drop_rate=0.0, drop_path_rate=0.0, zero_init_last=False):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor
        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))
        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(prev_chs, out_chs, stride=stride, dilation=dilation, depth=d, avg_down=avg_down, act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)
        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else nn.Identity()
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)
        self.init_weights(zero_init_last=zero_init_last)
        self.grad_checkpointing = False

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^stem', blocks='^stages\\.(\\d+)' if coarse else [('^stages\\.(\\d+)\\.blocks\\.(\\d+)', None), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x, flatten=True)
        else:
            x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


SEWithNorm = partial(SEModule, norm_layer=nn.BatchNorm2d)


class LinearBottleneck(nn.Module):

    def __init__(self, in_chs, out_chs, stride, exp_ratio=1.0, se_ratio=0.0, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_path=None):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_chs <= out_chs
        self.in_channels = in_chs
        self.out_channels = out_chs
        if exp_ratio != 1.0:
            dw_chs = make_divisible(round(in_chs * exp_ratio), divisor=ch_div)
            self.conv_exp = ConvNormAct(in_chs, dw_chs, act_layer=act_layer)
        else:
            dw_chs = in_chs
            self.conv_exp = None
        self.conv_dw = ConvNormAct(dw_chs, dw_chs, 3, stride=stride, groups=dw_chs, apply_act=False)
        if se_ratio > 0:
            self.se = SEWithNorm(dw_chs, rd_channels=make_divisible(int(dw_chs * se_ratio), ch_div))
        else:
            self.se = None
        self.act_dw = create_act_layer(dw_act_layer)
        self.conv_pwl = ConvNormAct(dw_chs, out_chs, 1, apply_act=False)
        self.drop_path = drop_path

    def feat_channels(self, exp=False):
        return self.conv_dw.out_channels if exp else self.out_channels

    def forward(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        return x


def _block_cfg(width_mult=1.0, depth_mult=1.0, initial_chs=16, final_chs=180, se_ratio=0.0, ch_div=1):
    layers = [1, 2, 2, 3, 3, 5]
    strides = [1, 2, 2, 2, 1, 2]
    layers = [ceil(element * depth_mult) for element in layers]
    strides = sum([([element] + [1] * (layers[idx] - 1)) for idx, element in enumerate(strides)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    depth = sum(layers[:]) * 3
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs
    out_chs_list = []
    for i in range(depth // 3):
        out_chs_list.append(make_divisible(round(base_chs * width_mult), divisor=ch_div))
        base_chs += final_chs / (depth // 3 * 1.0)
    se_ratios = [0.0] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])
    return list(zip(out_chs_list, exp_ratios, strides, se_ratios))


def _build_blocks(block_cfg, prev_chs, width_mult, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_path_rate=0.0):
    feat_chs = [prev_chs]
    feature_info = []
    curr_stride = 2
    features = []
    num_blocks = len(block_cfg)
    for block_idx, (chs, exp_ratio, stride, se_ratio) in enumerate(block_cfg):
        if stride > 1:
            fname = 'stem' if block_idx == 0 else f'features.{block_idx - 1}'
            feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=fname)]
            curr_stride *= stride
        block_dpr = drop_path_rate * block_idx / (num_blocks - 1)
        drop_path = DropPath(block_dpr) if block_dpr > 0.0 else None
        features.append(LinearBottleneck(in_chs=prev_chs, out_chs=chs, exp_ratio=exp_ratio, stride=stride, se_ratio=se_ratio, ch_div=ch_div, act_layer=act_layer, dw_act_layer=dw_act_layer, drop_path=drop_path))
        prev_chs = chs
        feat_chs += [features[-1].feat_channels()]
    pen_chs = make_divisible(1280 * width_mult, divisor=ch_div)
    feature_info += [dict(num_chs=feat_chs[-1], reduction=curr_stride, module=f'features.{len(features) - 1}')]
    features.append(ConvNormAct(prev_chs, pen_chs, act_layer=act_layer))
    return features, feature_info


class ReXNetV1(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, initial_chs=16, final_chs=180, width_mult=1.0, depth_mult=1.0, se_ratio=1 / 12.0, ch_div=1, act_layer='swish', dw_act_layer='relu6', drop_rate=0.2, drop_path_rate=0.0):
        super(ReXNetV1, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        assert output_stride == 32
        stem_base_chs = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base_chs * width_mult), divisor=ch_div)
        self.stem = ConvNormAct(in_chans, stem_chs, 3, stride=2, act_layer=act_layer)
        block_cfg = _block_cfg(width_mult, depth_mult, initial_chs, final_chs, se_ratio, ch_div)
        features, self.feature_info = _build_blocks(block_cfg, stem_chs, width_mult, ch_div, act_layer, dw_act_layer, drop_path_rate)
        self.num_features = features[-1].out_channels
        self.features = nn.Sequential(*features)
        self.head = ClassifierHead(self.num_features, num_classes, global_pool, drop_rate)
        efficientnet_init_weights(self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^stem', blocks='^features\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.features, x, flatten=True)
        else:
            x = self.features(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SequentialList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class SelectSeq(nn.Module):

    def __init__(self, mode='index', index=0):
        super(SelectSeq, self).__init__()
        self.mode = mode
        self.index = index

    @torch.jit._overload_method
    def forward(self, x):
        pass

    @torch.jit._overload_method
    def forward(self, x):
        pass

    def forward(self, x) ->torch.Tensor:
        if self.mode == 'index':
            return x[self.index]
        else:
            return torch.cat(x, dim=1)


def conv_bn(in_chs, out_chs, k=3, stride=1, padding=None, dilation=1):
    if padding is None:
        padding = (stride - 1 + dilation * (k - 1)) // 2
    return nn.Sequential(nn.Conv2d(in_chs, out_chs, k, stride, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_chs), nn.ReLU(inplace=True))


class SelecSLSBlock(nn.Module):

    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride, dilation=1):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]
        self.conv1 = conv_bn(in_chs, mid_chs, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(mid_chs, mid_chs, 1)
        self.conv3 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_bn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_bn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_bn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1)

    def forward(self, x: List[torch.Tensor]) ->List[torch.Tensor]:
        if not isinstance(x, list):
            x = [x]
        assert len(x) in [1, 2]
        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLS(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    def __init__(self, cfg, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(SelecSLS, self).__init__()
        self.stem = conv_bn(in_chans, 32, stride=2)
        self.features = SequentialList(*[cfg['block'](*block_args) for block_args in cfg['features']])
        self.from_seq = SelectSeq()
        self.head = nn.Sequential(*[conv_bn(*conv_args) for conv_args in cfg['head']])
        self.num_features = cfg['num_features']
        self.feature_info = cfg['feature_info']
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks='^features\\.(\\d+)', blocks_head='^head')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(self.from_seq(x))
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = self.se_module(out) + shortcut
        out = self.relu(out)
        return out


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2, in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.pool0 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='layer0')]
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.feature_info += [dict(num_chs=64 * block.expansion, reduction=4, module='layer1')]
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=128 * block.expansion, reduction=8, module='layer2')]
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=256 * block.expansion, reduction=16, module='layer3')]
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.feature_info += [dict(num_chs=512 * block.expansion, reduction=32, module='layer4')]
        self.num_features = 512 * block.expansion
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^layer0', blocks='^layer(\\d+)' if coarse else '^layer(\\d+)\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.last_linear = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class RNNIdentity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, None]:
        return x, None


class RNN2DBase(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, bidirectional: bool=True, union='cat', with_fc=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union
        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc
        self.fc = None
        if with_fc:
            if union == 'cat':
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == 'add':
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == 'vertical':
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == 'horizontal':
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError('Unrecognized union: ' + union)
        elif union == 'cat':
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f'The output channel {2 * self.output_size} is different from the input channel {input_size}.')
        elif union == 'add':
            pass
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
        elif union == 'vertical':
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
            self.with_horizontal = False
        elif union == 'horizontal':
            if self.output_size != input_size:
                raise ValueError(f'The output channel {self.output_size} is different from the input channel {input_size}.')
            self.with_vertical = False
        else:
            raise ValueError('Unrecognized union: ' + union)
        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)
        else:
            v = None
        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)
        else:
            h = None
        if v is not None and h is not None:
            if self.union == 'cat':
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif v is not None:
            x = v
        elif h is not None:
            x = h
        if self.fc is not None:
            x = self.fc(x)
        return x


class LSTM2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, bidirectional: bool=True, union='cat', with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class Sequencer2DBlock(nn.Module):

    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM2D, mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, num_layers=1, bidirectional=True, union='cat', with_fc=True, drop=0.0, drop_path=0.0):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional, union=union, with_fc=with_fc)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Shuffle(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2D(nn.Module):

    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_stage(index, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer, rnn_layer, mlp_layer, norm_layer, act_layer, num_layers, bidirectional, union, with_fc, drop=0.0, drop_path_rate=0.0, **kwargs):
    assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
    blocks = []
    for block_idx in range(layers[index]):
        drop_path = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_layer(embed_dims[index], hidden_sizes[index], mlp_ratio=mlp_ratios[index], rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer, num_layers=num_layers, bidirectional=bidirectional, union=union, with_fc=with_fc, drop=drop, drop_path=drop_path))
    if index < len(embed_dims) - 1:
        blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))
    blocks = nn.Sequential(*blocks)
    return blocks


class Sequencer2D(nn.Module):

    def __init__(self, num_classes=1000, img_size=224, in_chans=3, global_pool='avg', layers=[4, 3, 8, 3], patch_sizes=[7, 2, 1, 1], embed_dims=[192, 384, 384, 384], hidden_sizes=[48, 96, 96, 96], mlp_ratios=[3.0, 3.0, 3.0, 3.0], block_layer=Sequencer2DBlock, rnn_layer=LSTM2D, mlp_layer=Mlp, norm_layer=partial(nn.LayerNorm, eps=1e-06), act_layer=nn.GELU, num_rnn_layers=1, bidirectional=True, union='cat', with_fc=True, drop_rate=0.0, drop_path_rate=0.0, nlhb=False, stem_norm=False):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = embed_dims[-1]
        self.feature_dim = -1
        self.embed_dims = embed_dims
        self.stem = PatchEmbed(img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if stem_norm else None, flatten=False)
        self.blocks = nn.Sequential(*[get_stage(i, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer=block_layer, rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer, num_layers=num_rnn_layers, bidirectional=bidirectional, union=union, with_fc=with_fc, drop=drop_rate, drop_path_rate=drop_path_rate) for i, _ in enumerate(embed_dims)])
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], self.num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.0
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks=[('^blocks\\.(\\d+)\\..*\\.down', (99999,)), ('^blocks\\.(\\d+)', None) if coarse else ('^blocks\\.(\\d+)\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(1, 2))
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(SelectiveKernelBasic, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = SelectiveKernel(inplanes, first_planes, stride=stride, dilation=first_dilation, aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv2 = ConvNormAct(first_planes, outplanes, kernel_size=3, dilation=dilation, apply_act=False, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(SelectiveKernelBottleneck, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        self.conv1 = ConvNormAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernel(first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality, aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv3 = ConvNormAct(width, outplanes, kernel_size=1, apply_act=False, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table, persistent=False)
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
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowMultiHeadAttention(nn.Module):
    """This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int], drop_attn: float=0.0, drop_proj: float=0.0, meta_hidden_dim: int=384, sequential_attn: bool=False) ->None:
        super(WindowMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, 'The number of input features (in_features) are not divisible by the number of heads (num_heads).'
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        self.meta_mlp = Mlp(2, hidden_features=meta_hidden_dim, out_features=num_heads, act_layer=nn.ReLU, drop=(0.125, 0.0))
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) ->None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(torch.meshgrid([torch.arange(self.window_size[0], device=device), torch.arange(self.window_size[1], device=device)]), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(1.0 + relative_coordinates.abs())
        self.register_buffer('relative_coordinates_log', relative_coordinates_log, persistent=False)

    def update_input_size(self, new_window_size: int, **kwargs: Any) ->None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        self.window_size: int = new_window_size
        self._make_pair_wise_relative_positions()

    def _relative_positional_encodings(self) ->torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(self.num_heads, window_area, window_area)
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def _forward_sequential(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        """
        assert False, 'not implemented'

    def _forward_batch(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """This function performs standard (non-sequential) scaled cosine self-attention.
        """
        Bw, L, C = x.shape
        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attn = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale
        attn = attn + self._relative_positional_encodings()
        if mask is not None:
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        if self.sequential_attn:
            return self._forward_sequential(x, mask)
        else:
            return self._forward_batch(x, mask)


class SwinTransformerBlock(nn.Module):
    """This class implements the Swin transformer block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized
    """

    def __init__(self, dim: int, num_heads: int, feat_size: Tuple[int, int], window_size: Tuple[int, int], shift_size: Tuple[int, int]=(0, 0), mlp_ratio: float=4.0, init_values: Optional[float]=0, drop: float=0.0, drop_attn: float=0.0, drop_path: float=0.0, extra_norm: bool=False, sequential_attn: bool=False, norm_layer: Type[nn.Module]=nn.LayerNorm) ->None:
        super(SwinTransformerBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values
        self.attn = WindowMultiHeadAttention(dim=dim, num_heads=num_heads, window_size=self.window_size, drop_attn=drop_attn, drop_proj=drop, sequential_attn=sequential_attn)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop, out_features=dim)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim) if extra_norm else nn.Identity()
        self._make_attention_mask()
        self.init_weights()

    def _calc_window_shift(self, target_window_size):
        window_size = [(f if f <= w else w) for f, w in zip(self.feat_size, target_window_size)]
        shift_size = [(0 if f <= w else s) for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) ->None:
        """Method generates the attention mask used in shift case."""
        if any(self.shift_size):
            H, W = self.feat_size
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for h in (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)):
                for w in (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def update_input_size(self, new_window_size: Tuple[int, int], new_feat_size: Tuple[int, int]) ->None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = new_feat_size
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(new_window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.update_input_size(new_window_size=self.window_size)
        self._make_attention_mask()

    def _shifted_window_attn(self, x):
        H, W = self.feat_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, self.feat_size)
        if do_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        x = x.view(B, L, C)
        return x

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(x)
        return x


def bhwc_to_bchw(x: torch.Tensor) ->torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


class PatchMerging(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized.
    """

    def __init__(self, dim: int, norm_layer: Type[nn.Module]=nn.LayerNorm) ->None:
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 5, 3, 1).flatten(3)
        x = self.norm(x)
        x = bhwc_to_bchw(self.reduction(x))
        return x


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
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, pretrained_window_size=pretrained_window_size) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformer(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None, window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, weight_init='', **kwargs):
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i in range(self.num_layers):
            layers += [BasicLayer(dim=embed_dim[i], out_dim=embed_out_dim[i], input_resolution=(self.patch_grid[0] // 2 ** i, self.patch_grid[1] // 2 ** i), depth=depths[i], num_heads=num_heads[i], head_dim=head_dim[i], window_size=window_size[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer, downsample=PatchMerging if i < self.num_layers - 1 else None)]
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.0
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^absolute_pos_embed|patch_embed', blocks='^layers\\.(\\d+)' if coarse else [('^layers\\.(\\d+).downsample', (0,)), ('^layers\\.(\\d+)\\.\\w+\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SwinTransformerV2(nn.Module):
    """ Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), input_resolution=(self.patch_embed.grid_size[0] // 2 ** i_layer, self.patch_embed.grid_size[1] // 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = {'absolute_pos_embed'}
        for n, m in self.named_modules():
            if any([(kw in n) for kw in ('cpb_mlp', 'logit_scale', 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^absolute_pos_embed|patch_embed', blocks='^layers\\.(\\d+)' if coarse else [('^layers\\.(\\d+).downsample', (0,)), ('^layers\\.(\\d+)\\.\\w+\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def bchw_to_bhwc(x: torch.Tensor) ->torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


class SwinTransformerStage(nn.Module):
    """This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(self, embed_dim: int, depth: int, downscale: bool, num_heads: int, feat_size: Tuple[int, int], window_size: Tuple[int, int], mlp_ratio: float=4.0, init_values: Optional[float]=0.0, drop: float=0.0, drop_attn: float=0.0, drop_path: Union[List[float], float]=0.0, norm_layer: Type[nn.Module]=nn.LayerNorm, extra_norm_period: int=0, extra_norm_stage: bool=False, sequential_attn: bool=False) ->None:
        super(SwinTransformerStage, self).__init__()
        self.downscale: bool = downscale
        self.grad_checkpointing: bool = False
        self.feat_size: Tuple[int, int] = (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size
        self.downsample = PatchMerging(embed_dim, norm_layer=norm_layer) if downscale else nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False
        embed_dim = embed_dim * 2 if downscale else embed_dim
        self.blocks = nn.Sequential(*[SwinTransformerBlock(dim=embed_dim, num_heads=num_heads, feat_size=self.feat_size, window_size=window_size, shift_size=tuple([(0 if index % 2 == 0 else w // 2) for w in window_size]), mlp_ratio=mlp_ratio, init_values=init_values, drop=drop, drop_attn=drop_attn, drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path, extra_norm=_extra_norm(index), sequential_attn=sequential_attn, norm_layer=norm_layer) for index in range(depth)])

    def update_input_size(self, new_window_size: int, new_feat_size: Tuple[int, int]) ->None:
        """Method updates the resolution to utilize and the window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = (new_feat_size[0] // 2, new_feat_size[1] // 2) if self.downscale else new_feat_size
        for block in self.blocks:
            block.update_input_size(new_window_size=new_window_size, new_feat_size=self.feat_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        x = self.downsample(x)
        B, C, H, W = x.shape
        L = H * W
        x = bchw_to_bhwc(x).reshape(B, L, C)
        for block in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = bhwc_to_bchw(x.reshape(B, H, W, -1))
        return x


def init_weights(module: nn.Module, name: str=''):
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        elif 'head' in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class SwinTransformerV2Cr(nn.Module):
    """ Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size (Tuple[int, int]): Input resolution.
        window_size (Optional[int]): Window size. If None, img_size // window_div. Default: None
        img_window_ratio (int): Window size to image size ratio. Default: 32
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input channels.
        depths (int): Depth of the stage (number of layers).
        num_heads (int): Number of attention heads to be utilized.
        embed_dim (int): Patch embedding dimension. Default: 96
        num_classes (int): Number of output classes. Default: 1000
        mlp_ratio (int):  Ratio of the hidden dimension in the FFN to the input channels. Default: 4
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Dropout rate of attention map. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.0
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed. Default: False
    """

    def __init__(self, img_size: Tuple[int, int]=(224, 224), patch_size: int=4, window_size: Optional[int]=None, img_window_ratio: int=32, in_chans: int=3, num_classes: int=1000, embed_dim: int=96, depths: Tuple[int, ...]=(2, 2, 6, 2), num_heads: Tuple[int, ...]=(3, 6, 12, 24), mlp_ratio: float=4.0, init_values: Optional[float]=0.0, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer: Type[nn.Module]=nn.LayerNorm, extra_norm_period: int=0, extra_norm_stage: bool=False, sequential_attn: bool=False, global_pool: str='avg', weight_init='skip', **kwargs: Any) ->None:
        super(SwinTransformerV2Cr, self).__init__()
        img_size = to_2tuple(img_size)
        window_size = tuple([(s // img_window_ratio) for s in img_size]) if window_size is None else to_2tuple(window_size)
        self.num_classes: int = num_classes
        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patch_grid_size: Tuple[int, int] = self.patch_embed.grid_size
        drop_path_rate = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        stages = []
        for index, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stage_scale = 2 ** max(index - 1, 0)
            stages.append(SwinTransformerStage(embed_dim=embed_dim * stage_scale, depth=depth, downscale=index != 0, feat_size=(patch_grid_size[0] // stage_scale, patch_grid_size[1] // stage_scale), num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, init_values=init_values, drop=drop_rate, drop_attn=attn_drop_rate, drop_path=drop_path_rate[sum(depths[:index]):sum(depths[:index + 1])], extra_norm_period=extra_norm_period, extra_norm_stage=extra_norm_stage or index + 1 == len(depths), sequential_attn=sequential_attn, norm_layer=norm_layer))
        self.stages = nn.Sequential(*stages)
        self.global_pool: str = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes else nn.Identity()
        if weight_init != 'skip':
            named_apply(init_weights, self)

    def update_input_size(self, new_img_size: Optional[Tuple[int, int]]=None, new_window_size: Optional[int]=None, img_window_ratio: int=32) ->None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            new_img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            img_window_ratio (int): divisor for calculating window size from image size
        """
        if new_img_size is None:
            new_img_size = self.img_size
        else:
            new_img_size = to_2tuple(new_img_size)
        if new_window_size is None:
            new_window_size = tuple([(s // img_window_ratio) for s in new_img_size])
        new_patch_grid_size = new_img_size[0] // self.patch_size, new_img_size[1] // self.patch_size
        for index, stage in enumerate(self.stages):
            stage_scale = 2 ** max(index - 1, 0)
            stage.update_input_size(new_window_size=new_window_size, new_img_size=(new_patch_grid_size[0] // stage_scale, new_patch_grid_size[1] // stage_scale))

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^patch_embed', blocks='^stages\\.(\\d+)' if coarse else [('^stages\\.(\\d+).downsample', (0,)), ('^stages\\.(\\d+)\\.\\w+\\.(\\d+)', None)])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self) ->nn.Module:
        """Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str]=None) ->None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
        """
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) ->torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=(2, 3))
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PixelEmbed(nn.Module):
    """ Image to Pixel Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size
        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        _assert(W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x


class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, in_dim=48, depth=12, num_heads=12, in_num_head=4, mlp_ratio=4.0, qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, first_stride=4):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.pixel_embed = PixelEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, in_dim=in_dim, stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]
        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim, new_patch_size[0], new_patch_size[1]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(Block(dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.patch_pos, std=0.02)
        trunc_normal_(self.pixel_pos, std=0.02)
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
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^cls_token|patch_pos|pixel_pos|pixel_embed|norm[12]_proj|proj', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.blocks:
                pixel_embed, patch_embed = checkpoint(blk, pixel_embed, patch_embed)
        else:
            for blk in self.blocks:
                pixel_embed, patch_embed = blk(pixel_embed, patch_embed)
        patch_embed = self.norm(patch_embed)
        return patch_embed

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class TResNet(nn.Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, v2=False, global_pool='fast', drop_rate=0.0):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TResNet, self).__init__()
        aa_layer = BlurPool2d
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        if v2:
            self.inplanes = self.inplanes // 8 * 8
            self.planes = self.planes // 8 * 8
        conv1 = conv2d_iabn(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(Bottleneck if v2 else BasicBlock, self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer)
        layer2 = self._make_layer(Bottleneck if v2 else BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer)
        self.body = nn.Sequential(OrderedDict([('SpaceToDepth', SpaceToDepthModule()), ('conv1', conv1), ('layer1', layer1), ('layer2', layer2), ('layer3', layer3), ('layer4', layer4)]))
        self.feature_info = [dict(num_chs=self.planes, reduction=2, module=''), dict(num_chs=self.planes * (Bottleneck.expansion if v2 else 1), reduction=4, module='body.layer1'), dict(num_chs=self.planes * 2 * (Bottleneck.expansion if v2 else 1), reduction=8, module='body.layer2'), dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'), dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4')]
        self.num_features = self.planes * 8 * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer='identity')]
            downsample = nn.Sequential(*layers)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^body\\.conv1', blocks='^body\\.layer(\\d+)' if coarse else '^body\\.layer(\\d+)\\.(\\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='fast'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        return self.body(x)

    def forward_head(self, x, pre_logits: bool=False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


Size_ = Tuple[int, int]


class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PosConv(nn.Module):

    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim))
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return [('proj.%d.weight' % i) for i in range(4)]


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)

    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg', embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), wss=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-06), block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False
        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[k], ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set([('pos_block.' + n) for n, p in self.pos_block.named_parameters()])

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem='^patch_embeds.0', blocks=[('^(?:blocks|patch_embeds|pos_block)\\.(\\d+)', None), ('^norm', (99999,))] if coarse else [('^blocks\\.(\\d+)\\.(\\d+)', None), ('^(?:patch_embeds|pos_block)\\.(\\d+)', (0,)), ('^norm', (99999,))])
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class VGG(nn.Module):

    def __init__(self, cfg: List[Any], num_classes: int=1000, in_chans: int=3, output_stride: int=32, mlp_ratio: float=1.0, act_layer: nn.Module=nn.ReLU, conv_layer: nn.Module=nn.Conv2d, norm_layer: nn.Module=None, global_pool: str='avg', drop_rate: float=0.0) ->None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                else:
                    layers += [conv2d, act_layer(inplace=True)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))
        self.pre_logits = ConvMlp(prev_chs, self.num_features, 7, mlp_ratio=mlp_ratio, drop_rate=drop_rate, act_layer=act_layer, conv_layer=conv_layer)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        self._initialize_weights()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^features\\.0', blocks='^features\\.(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(self.num_features, self.num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: torch.Tensor) ->torch.Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool=False):
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def _initialize_weights(self) ->None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SpatialMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2:
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1, groups=self.group, bias=False)
            self.act2 = act_layer()
        else:
            self.conv2 = None
            self.act2 = None
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)
        self.drop3 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)
        if self.conv2 is not None:
            x = self.conv2(x)
            x = self.act2(x)
        x = self.conv3(x)
        x = self.drop3(x)
        return x


class Visformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=LayerNorm2d, attn_stage='111', pos_embed=True, spatial_conv='111', vit_stem=False, group=8, global_pool='avg', conv_init=False, embed_norm=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_stem = vit_stem
        self.conv_init = conv_init
        if isinstance(depth, (list, tuple)):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.pos_embed = pos_embed
        self.grad_checkpointing = False
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if self.vit_stem:
            self.stem = None
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=embed_norm, flatten=False)
            img_size = [(x // patch_size) for x in img_size]
        elif self.init_channels is None:
            self.stem = None
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size // 2, in_chans=in_chans, embed_dim=embed_dim // 2, norm_layer=embed_norm, flatten=False)
            img_size = [(x // (patch_size // 2)) for x in img_size]
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_chans, self.init_channels, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(self.init_channels), nn.ReLU(inplace=True))
            img_size = [(x // 2) for x in img_size]
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size // 4, in_chans=self.init_channels, embed_dim=embed_dim // 2, norm_layer=embed_norm, flatten=False)
            img_size = [(x // (patch_size // 4)) for x in img_size]
        if self.pos_embed:
            if self.vit_stem:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim // 2, *img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage1 = nn.Sequential(*[Block(dim=embed_dim // 2, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group, attn_disabled=attn_stage[0] == '0', spatial_conv=spatial_conv[0] == '1') for i in range(self.stage_num1)])
        if not self.vit_stem:
            self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=patch_size // 8, in_chans=embed_dim // 2, embed_dim=embed_dim, norm_layer=embed_norm, flatten=False)
            img_size = [(x // (patch_size // 8)) for x in img_size]
            if self.pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, *img_size))
        self.stage2 = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group, attn_disabled=attn_stage[1] == '0', spatial_conv=spatial_conv[1] == '1') for i in range(self.stage_num1, self.stage_num1 + self.stage_num2)])
        if not self.vit_stem:
            self.patch_embed3 = PatchEmbed(img_size=img_size, patch_size=patch_size // 8, in_chans=embed_dim, embed_dim=embed_dim * 2, norm_layer=embed_norm, flatten=False)
            img_size = [(x // (patch_size // 8)) for x in img_size]
            if self.pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim * 2, *img_size))
        self.stage3 = nn.Sequential(*[Block(dim=embed_dim * 2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group=group, attn_disabled=attn_stage[2] == '0', spatial_conv=spatial_conv[2] == '1') for i in range(self.stage_num1 + self.stage_num2, depth)])
        self.num_features = embed_dim if self.vit_stem else embed_dim * 2
        self.norm = norm_layer(self.num_features)
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        if self.pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_stem:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^patch_embed1|pos_embed1|stem', blocks=[('^stage(\\d+)\\.(\\d+)' if coarse else '^stage(\\d+)\\.(\\d+)', None), ('^(?:patch_embed|pos_embed)(\\d+)', (0,)), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = self.pos_drop(x + self.pos_embed1)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage1, x)
        else:
            x = self.stage1(x)
        if not self.vit_stem:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = self.pos_drop(x + self.pos_embed2)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage2, x)
        else:
            x = self.stage2(x)
        if not self.vit_stem:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = self.pos_drop(x + self.pos_embed3)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stage3, x)
        else:
            x = self.stage3(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ResPostBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class RelPosAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, rel_pos_cls=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rel_pos = rel_pos_cls(num_heads=num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor]=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelPosBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, rel_pos_cls=None, init_values=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelPosAttention(dim, num_heads, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor]=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostRelPosBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, rel_pos_cls=None, init_values=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values
        self.attn = RelPosAttention(dim, num_heads, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor]=None):
        x = x + self.drop_path1(self.norm1(self.attn(x, shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VisionTransformerRelPos(nn.Module):
    """ Vision Transformer w/ Relative Position Bias

    Differing from classic vit, this impl
      * uses relative position index (swin v1 / beit) or relative log coord + mlp (swin v2) pos embed
      * defaults to no class token (can be enabled)
      * defaults to global avg pool for head (can be changed)
      * layer-scale (residual branch gain) enabled
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='avg', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=1e-06, class_token=False, fc_norm=False, rel_pos_type='mlp', rel_pos_dim=None, shared_rel_pos=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, weight_init='skip', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=RelPosBlock):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'avg')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token (default: False)
            fc_norm (bool): use pre classifier norm instead of pre-pool
            rel_pos_ty pe (str): type of relative position
            shared_rel_pos (bool): share relative pos across all blocks
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        feat_size = self.patch_embed.grid_size
        rel_pos_args = dict(window_size=feat_size, prefix_tokens=self.num_prefix_tokens)
        if rel_pos_type.startswith('mlp'):
            if rel_pos_dim:
                rel_pos_args['hidden_dim'] = rel_pos_dim
            if 'swin' in rel_pos_type:
                rel_pos_args['mode'] = 'swin'
            elif 'rw' in rel_pos_type:
                rel_pos_args['mode'] = 'rw'
            rel_pos_cls = partial(RelPosMlp, **rel_pos_args)
        else:
            rel_pos_cls = partial(RelPosBias, **rel_pos_args)
        self.shared_rel_pos = None
        if shared_rel_pos:
            self.shared_rel_pos = rel_pos_cls(num_heads=num_heads)
            rel_pos_cls = None
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim)) if class_token else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, rel_pos_cls=rel_pos_cls, init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'moco', '')
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-06)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|patch_embed', blocks=[('^blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
            else:
                x = blk(x, shared_rel_pos=shared_rel_pos)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = head_dim ** -0.5
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads, self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x


class Outlooker(nn.Module):

    def __init__(self, dim, kernel_size, padding, stride=1, num_heads=1, mlp_ratio=3.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, qkv_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size, padding=padding, stride=stride, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = q * self.scale @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    if block_type == 'ca':
        return ClassBlock(**kargs)


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3, padding=1, stride=2, mlp_ratio=3.0, qkv_bias=False, attn_drop=0, drop_path_rate=0.0, **kwargs):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding, stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=block_dpr))
    blocks = nn.Sequential(*blocks)
    return blocks


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.0, qkv_bias=False, attn_drop=0, drop_path_rate=0.0, **kwargs):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=block_dpr))
    blocks = nn.Sequential(*blocks)
    return blocks


class VOLO(nn.Module):
    """
    Vision Outlooker, the main class of our model
    """

    def __init__(self, layers, img_size=224, in_chans=3, num_classes=1000, global_pool='token', patch_size=8, stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=(True, False, False, False), outlook_attention=(True, False, False, False), mlp_ratio=3.0, qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, post_layers=('ca', 'ca'), use_aux_head=True, use_mix_token=False, pooling_scale=2):
        super().__init__()
        num_layers = len(layers)
        mlp_ratio = to_ntuple(num_layers)(mlp_ratio)
        img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.mix_token = use_mix_token
        self.pooling_scale = pooling_scale
        self.num_features = embed_dims[-1]
        if use_mix_token:
            self.beta = 1.0
            assert global_pool == 'token', 'return all tokens if mix_token is enabled'
        self.grad_checkpointing = False
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size, in_chans=in_chans, hidden_dim=stem_hidden_dim, embed_dim=embed_dims[0])
        patch_grid = img_size[0] // patch_size // pooling_scale, img_size[1] // patch_size // pooling_scale
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_grid[0], patch_grid[1], embed_dims[-1]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers, num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            else:
                stage = transformer_blocks(Transformer, i, embed_dims[i], layers, num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias, drop_path_rate=drop_path_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                network.append(stage)
            if downsamples[i]:
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))
        self.network = nn.ModuleList(network)
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList([get_block(post_layers[i], dim=embed_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratio[-1], qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer) for i in range(len(post_layers))])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=0.02)
        if use_aux_head:
            self.aux_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.aux_head = None
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[('^network\\.(\\d+)\\.(\\d+)', None), ('^network\\.(\\d+)', (0,))], blocks2=[('^cls_token', (0,)), ('^post_network\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.aux_head is not None:
            self.aux_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            if idx == 2:
                x = x + self.pos_embed
                x = self.pos_drop(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.post_network:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)
        return x

    def forward_train(self, x):
        """ A separate forward fn for training with mix_token (if a train script supports).
        Combining multiple modes in as single forward with different return types is torchscript hell.
        """
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1, sbby1 = self.pooling_scale * bbx1, self.pooling_scale * bby1
            sbbx2, sbby2 = self.pooling_scale * bbx2, self.pooling_scale * bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        if self.global_pool == 'avg':
            x_cls = x.mean(dim=1)
        elif self.global_pool == 'token':
            x_cls = x[:, 0]
        else:
            x_cls = x
        if self.aux_head is None:
            return x_cls
        x_aux = self.aux_head(x[:, 1:])
        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]
        if self.mix_token and self.training:
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = self.forward_tokens(x)
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool == 'avg':
            out = x.mean(dim=1)
        elif self.global_pool == 'token':
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out
        out = self.head(out)
        if self.aux_head is not None:
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]
        return out

    def forward(self, x):
        """ simplified forward (without mix token training) """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class SequentialAppendList(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) ->torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x


class OsaBlock(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, layer_per_block, residual=False, depthwise=False, attn='', norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path=None):
        super(OsaBlock, self).__init__()
        self.residual = residual
        self.depthwise = depthwise
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)
        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = ConvNormAct(next_in_chs, mid_chs, 1, **conv_kwargs)
        else:
            self.conv_reduction = None
        mid_convs = []
        for i in range(layer_per_block):
            if self.depthwise:
                conv = SeparableConvNormAct(mid_chs, mid_chs, **conv_kwargs)
            else:
                conv = ConvNormAct(next_in_chs, mid_chs, 3, **conv_kwargs)
            next_in_chs = mid_chs
            mid_convs.append(conv)
        self.conv_mid = SequentialAppendList(*mid_convs)
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = ConvNormAct(next_in_chs, out_chs, **conv_kwargs)
        self.attn = create_attn(attn, out_chs) if attn else None
        self.drop_path = drop_path

    def forward(self, x):
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.residual:
            x = x + output[0]
        return x


class OsaStage(nn.Module):

    def __init__(self, in_chs, mid_chs, out_chs, block_per_stage, layer_per_block, downsample=True, residual=True, depthwise=False, attn='ese', norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path_rates=None):
        super(OsaStage, self).__init__()
        self.grad_checkpointing = False
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None
        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            if drop_path_rates is not None and drop_path_rates[i] > 0.0:
                drop_path = DropPath(drop_path_rates[i])
            else:
                drop_path = None
            blocks += [OsaBlock(in_chs, mid_chs, out_chs, layer_per_block, residual=residual and i > 0, depthwise=depthwise, attn=attn if last_block else '', norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class VovNet(nn.Module):

    def __init__(self, cfg, in_chans=3, num_classes=1000, global_pool='avg', drop_rate=0.0, stem_stride=4, output_stride=32, norm_layer=BatchNormAct2d, act_layer=nn.ReLU, drop_path_rate=0.0):
        """ VovNet (v2)
        """
        super(VovNet, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert stem_stride in (4, 2)
        assert output_stride == 32
        stem_chs = cfg['stem_chs']
        stage_conv_chs = cfg['stage_conv_chs']
        stage_out_chs = cfg['stage_out_chs']
        block_per_stage = cfg['block_per_stage']
        layer_per_block = cfg['layer_per_block']
        conv_kwargs = dict(norm_layer=norm_layer, act_layer=act_layer)
        last_stem_stride = stem_stride // 2
        conv_type = SeparableConvNormAct if cfg['depthwise'] else ConvNormAct
        self.stem = nn.Sequential(*[ConvNormAct(in_chans, stem_chs[0], 3, stride=2, **conv_kwargs), conv_type(stem_chs[0], stem_chs[1], 3, stride=1, **conv_kwargs), conv_type(stem_chs[1], stem_chs[2], 3, stride=last_stem_stride, **conv_kwargs)])
        self.feature_info = [dict(num_chs=stem_chs[1], reduction=2, module=f'stem.{1 if stem_stride == 4 else 2}')]
        current_stride = stem_stride
        stage_dpr = torch.split(torch.linspace(0, drop_path_rate, sum(block_per_stage)), block_per_stage)
        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
        stage_args = dict(residual=cfg['residual'], depthwise=cfg['depthwise'], attn=cfg['attn'], **conv_kwargs)
        stages = []
        for i in range(4):
            downsample = stem_stride == 2 or i > 0
            stages += [OsaStage(in_ch_list[i], stage_conv_chs[i], stage_out_chs[i], block_per_stage[i], layer_per_block, downsample=downsample, drop_path_rates=stage_dpr[i], **stage_args)]
            self.num_features = stage_out_chs[i]
            current_stride *= 2 if downsample else 1
            self.feature_info += [dict(num_chs=self.num_features, reduction=current_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks='^stages\\.(\\d+)' if coarse else '^stages\\.(\\d+).blocks\\.(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward_head(self, x, pre_logits: bool=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0.0, global_pool='avg'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_features = 2048
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        self.block4 = Block(728, 728, 3, 1)
        self.block5 = Block(728, 728, 3, 1)
        self.block6 = Block(728, 728, 3, 1)
        self.block7 = Block(728, 728, 3, 1)
        self.block8 = Block(728, 728, 3, 1)
        self.block9 = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)
        self.block12 = Block(728, 1024, 2, 2, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536, self.num_features, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.num_features)
        self.act4 = nn.ReLU(inplace=True)
        self.feature_info = [dict(num_chs=64, reduction=2, module='act2'), dict(num_chs=128, reduction=4, module='block2.rep.0'), dict(num_chs=256, reduction=8, module='block3.rep.0'), dict(num_chs=728, reduction=16, module='block12.rep.0'), dict(num_chs=2048, reduction=32, module='act4')]
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^conv[12]|bn[12]', blocks=[('^block(\\d+)', None), ('^conv[34]|bn[34]', (99,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PreSeparableConv2d(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1, padding='', act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, first_act=True):
        super(PreSeparableConv2d, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer=act_layer)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = norm_act_layer(in_chs, inplace=True) if first_act else nn.Identity()
        self.conv_dw = create_conv2d(in_chs, in_chs, kernel_size, stride=stride, padding=padding, dilation=dilation, depthwise=True)
        self.conv_pw = create_conv2d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


to_3tuple = _ntuple(3)


class XceptionModule(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, pad_type='', start_with_relu=True, no_skip=False, act_layer=nn.ReLU, norm_layer=None):
        super(XceptionModule, self).__init__()
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = ConvNormAct(in_chs, self.out_channels, 1, stride=stride, norm_layer=norm_layer, apply_act=False)
        else:
            self.shortcut = None
        separable_act_layer = None if start_with_relu else act_layer
        self.stack = nn.Sequential()
        for i in range(3):
            if start_with_relu:
                self.stack.add_module(f'act{i + 1}', act_layer(inplace=i > 0))
            self.stack.add_module(f'conv{i + 1}', SeparableConv2d(in_chs, out_chs[i], 3, stride=stride if i == 2 else 1, dilation=dilation, padding=pad_type, act_layer=separable_act_layer, norm_layer=norm_layer))
            in_chs = out_chs[i]

    def forward(self, x):
        skip = x
        x = self.stack(x)
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        if not self.no_skip:
            x = x + skip
        return x


class PreXceptionModule(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1, pad_type='', no_skip=False, act_layer=nn.ReLU, norm_layer=None):
        super(PreXceptionModule, self).__init__()
        out_chs = to_3tuple(out_chs)
        self.in_channels = in_chs
        self.out_channels = out_chs[-1]
        self.no_skip = no_skip
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.shortcut = create_conv2d(in_chs, self.out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.norm = get_norm_act_layer(norm_layer, act_layer=act_layer)(in_chs, inplace=True)
        self.stack = nn.Sequential()
        for i in range(3):
            self.stack.add_module(f'conv{i + 1}', PreSeparableConv2d(in_chs, out_chs[i], 3, stride=stride if i == 2 else 1, dilation=dilation, padding=pad_type, act_layer=act_layer, norm_layer=norm_layer, first_act=i > 0))
            in_chs = out_chs[i]

    def forward(self, x):
        x = self.norm(x)
        skip = x
        x = self.stack(x)
        if not self.no_skip:
            x = x + self.shortcut(skip)
        return x


class XceptionAligned(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, block_cfg, num_classes=1000, in_chans=3, output_stride=32, preact=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_rate=0.0, global_pool='avg'):
        super(XceptionAligned, self).__init__()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.stem = nn.Sequential(*[ConvNormAct(in_chans, 32, kernel_size=3, stride=2, **layer_args), create_conv2d(32, 64, kernel_size=3, stride=1) if preact else ConvNormAct(32, 64, kernel_size=3, stride=1, **layer_args)])
        curr_dilation = 1
        curr_stride = 2
        self.feature_info = []
        self.blocks = nn.Sequential()
        module_fn = PreXceptionModule if preact else XceptionModule
        for i, b in enumerate(block_cfg):
            b['dilation'] = curr_dilation
            if b['stride'] > 1:
                name = f'blocks.{i}.stack.conv2' if preact else f'blocks.{i}.stack.act3'
                self.feature_info += [dict(num_chs=to_3tuple(b['out_chs'])[-2], reduction=curr_stride, module=name)]
                next_stride = curr_stride * b['stride']
                if next_stride > output_stride:
                    curr_dilation *= b['stride']
                    b['stride'] = 1
                else:
                    curr_stride = next_stride
            self.blocks.add_module(str(i), module_fn(**b, **layer_args))
            self.num_features = self.blocks[-1].out_channels
        self.feature_info += [dict(num_chs=self.num_features, reduction=curr_stride, module='blocks.' + str(len(self.blocks) - 1))]
        self.act = act_layer(inplace=True) if preact else nn.Identity()
        self.head = ClassifierHead(in_chs=self.num_features, num_classes=num_classes, pool_type=global_pool, drop_rate=drop_rate)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^stem', blocks='^blocks\\.(\\d+)')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.act(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_planes))


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = img_size[1] // patch_size * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if patch_size == 16:
            self.proj = torch.nn.Sequential(conv3x3(in_chans, embed_dim // 8, 2), act_layer(), conv3x3(embed_dim // 8, embed_dim // 4, 2), act_layer(), conv3x3(embed_dim // 4, embed_dim // 2, 2), act_layer(), conv3x3(embed_dim // 2, embed_dim, 2))
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(conv3x3(in_chans, embed_dim // 4, 2), act_layer(), conv3x3(embed_dim // 4, embed_dim // 2, 2), act_layer(), conv3x3(embed_dim // 2, embed_dim, 2))
        else:
            raise 'For convolutional projection, patch size has to be in [8, 16]'

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows to augment the
    implicit communication performed by the block diagonal scatter attention. Implemented using 2 layers of separable
    3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features)

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1.0, tokens_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        if eta is not None:
            self.gamma1 = nn.Parameter(eta * torch.ones(dim))
            self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class XCABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiT(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token', embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1.0, tokens_norm=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate after positional embedding, and in XCA/CA projection + MLP
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate (constant across all layers)
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos_embed: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA

        Notes:
            - Although `layer_norm` is user specifiable, there are hard-coded `BatchNorm2d`s in the local patch
              interaction (class LPI) and the patch embedding (class ConvPatchEmbed)
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        img_size = to_2tuple(img_size)
        assert img_size[0] % patch_size == 0 and img_size[0] % patch_size == 0, '`patch_size` should divide image dimensions evenly'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.grad_checkpointing = False
        self.patch_embed = ConvPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, act_layer=act_layer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([XCABlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta) for _ in range(depth)])
        self.cls_attn_blocks = nn.ModuleList([ClassAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm) for _ in range(cls_attn_layers)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(stem='^cls_token|pos_embed|patch_embed', blocks='^blocks\\.(\\d+)', cls_attn_blocks=[('^cls_attn_blocks\\.(\\d+)', None), ('^norm', (99999,))])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        for blk in self.cls_attn_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AdaptiveCatAvgMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Affine,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AsymmetricLossMultiLabel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionPool2d,
     lambda: ([], {'in_features': 4, 'feat_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinaryCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (BlurPool2d,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CecaModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelAttn,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClassAttentionBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ClassBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ClassifierHead,
     lambda: ([], {'in_chs': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CondConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (Conv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvEmbedding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'patch_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvHeadPooling,
     lambda: ([], {'in_feature': 4, 'out_feature': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvMlpWithNorm,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNorm,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvPatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CrossAttentionBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (DlaBasic,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DlaBottleneck,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample,
     lambda: ([], {'in_embed_dim': 4, 'out_embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample2D,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Downsample2d,
     lambda: ([], {'dim': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EcaModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EffectiveSEModule,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastAdaptiveAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FourierEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELUTanh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GammaAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBottleneck,
     lambda: ([], {'in_chs': 4, 'mid_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostModule,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GluMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupNorm1,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardMishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardMishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSigmoidJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSigmoidMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (InceptionB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (InceptionC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {}),
     True),
    (InceptionResnetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (InceptionV4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (LSTM2D,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormAct,
     lambda: ([], {'normalization_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormAct2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNormExp2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScale2d,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerScaleBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LayerScaleBlockClassAttn,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (LightChannelAttn,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearNorm,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearSelfAttention,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MedianPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MetaBlock2d,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mixed3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Mixed4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {}),
     True),
    (Mixed5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (MixerBlock,
     lambda: ([], {'dim': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NonLocalAttn,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (OutlookAttention,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OverlapPatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ParallelBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (PatchMerging,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PosEmbedRel,
     lambda: ([], {'block_size': 4, 'win_size': 4, 'dim_head': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNNIdentity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RadixSoftmax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (ReductionB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (RelPosBiasTf,
     lambda: ([], {'window_size': [4, 4], 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 16, 16])], {}),
     True),
    (RelPosBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RelativePositionBias,
     lambda: ([], {'window_size': [4, 4], 'num_heads': 4}),
     lambda: ([], {}),
     True),
    (ResBlock,
     lambda: ([], {'dim': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResPostBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResPostRelPosBlock,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'m': _mock_layer(), 'drop': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEResNetBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'groups': 1, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledStdConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledStdConv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelecSLSBlock,
     lambda: ([], {'in_chs': 4, 'skip_chs': 4, 'mid_chs': 4, 'out_chs': 4, 'is_first': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelectAdaptivePool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelectSeq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sequencer2DBlock,
     lambda: ([], {'dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequentialList,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequentialTuple,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Shuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftTargetCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpaceToDepth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialGatingBlock,
     lambda: ([], {'dim': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGatingUnit,
     lambda: ([], {'dim': 4, 'seq_len': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialMlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SplitAttn,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SplitBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SqueezeExcite,
     lambda: ([], {'in_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExciteCl,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StdConv2d,
     lambda: ([], {'in_channel': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StdConv2dSame,
     lambda: ([], {'in_channel': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Stem4,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwishJit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SwishMe,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SyncBatchNormAct,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Tanh,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerLayer,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG,
     lambda: ([], {'cfg': _mock_config()}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (XCiT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_rwightman_pytorch_image_models(_paritybench_base):
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

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

    def test_069(self):
        self._check(*TESTCASES[69])

    def test_070(self):
        self._check(*TESTCASES[70])

    def test_071(self):
        self._check(*TESTCASES[71])

    def test_072(self):
        self._check(*TESTCASES[72])

    def test_073(self):
        self._check(*TESTCASES[73])

    def test_074(self):
        self._check(*TESTCASES[74])

    def test_075(self):
        self._check(*TESTCASES[75])

    def test_076(self):
        self._check(*TESTCASES[76])

    def test_077(self):
        self._check(*TESTCASES[77])

    def test_078(self):
        self._check(*TESTCASES[78])

    def test_079(self):
        self._check(*TESTCASES[79])

    def test_080(self):
        self._check(*TESTCASES[80])

    def test_081(self):
        self._check(*TESTCASES[81])

    def test_082(self):
        self._check(*TESTCASES[82])

    def test_083(self):
        self._check(*TESTCASES[83])

    def test_084(self):
        self._check(*TESTCASES[84])

    def test_085(self):
        self._check(*TESTCASES[85])

    def test_086(self):
        self._check(*TESTCASES[86])

    def test_087(self):
        self._check(*TESTCASES[87])

    def test_088(self):
        self._check(*TESTCASES[88])

    def test_089(self):
        self._check(*TESTCASES[89])

    def test_090(self):
        self._check(*TESTCASES[90])

    def test_091(self):
        self._check(*TESTCASES[91])

    def test_092(self):
        self._check(*TESTCASES[92])

    def test_093(self):
        self._check(*TESTCASES[93])

    def test_094(self):
        self._check(*TESTCASES[94])

    def test_095(self):
        self._check(*TESTCASES[95])

    def test_096(self):
        self._check(*TESTCASES[96])

    def test_097(self):
        self._check(*TESTCASES[97])

    def test_098(self):
        self._check(*TESTCASES[98])

    def test_099(self):
        self._check(*TESTCASES[99])

    def test_100(self):
        self._check(*TESTCASES[100])

    def test_101(self):
        self._check(*TESTCASES[101])

    def test_102(self):
        self._check(*TESTCASES[102])

    def test_103(self):
        self._check(*TESTCASES[103])

    def test_104(self):
        self._check(*TESTCASES[104])

    def test_105(self):
        self._check(*TESTCASES[105])

    def test_106(self):
        self._check(*TESTCASES[106])

    def test_107(self):
        self._check(*TESTCASES[107])

    def test_108(self):
        self._check(*TESTCASES[108])

    def test_109(self):
        self._check(*TESTCASES[109])

    def test_110(self):
        self._check(*TESTCASES[110])

    def test_111(self):
        self._check(*TESTCASES[111])

    def test_112(self):
        self._check(*TESTCASES[112])

    def test_113(self):
        self._check(*TESTCASES[113])

    def test_114(self):
        self._check(*TESTCASES[114])

    def test_115(self):
        self._check(*TESTCASES[115])

    def test_116(self):
        self._check(*TESTCASES[116])

    def test_117(self):
        self._check(*TESTCASES[117])

    def test_118(self):
        self._check(*TESTCASES[118])

    def test_119(self):
        self._check(*TESTCASES[119])

    def test_120(self):
        self._check(*TESTCASES[120])

    def test_121(self):
        self._check(*TESTCASES[121])

    def test_122(self):
        self._check(*TESTCASES[122])

    def test_123(self):
        self._check(*TESTCASES[123])

    def test_124(self):
        self._check(*TESTCASES[124])

    def test_125(self):
        self._check(*TESTCASES[125])

    def test_126(self):
        self._check(*TESTCASES[126])

    def test_127(self):
        self._check(*TESTCASES[127])

    def test_128(self):
        self._check(*TESTCASES[128])

    def test_129(self):
        self._check(*TESTCASES[129])

    def test_130(self):
        self._check(*TESTCASES[130])

    def test_131(self):
        self._check(*TESTCASES[131])

    def test_132(self):
        self._check(*TESTCASES[132])

    def test_133(self):
        self._check(*TESTCASES[133])

    def test_134(self):
        self._check(*TESTCASES[134])

    def test_135(self):
        self._check(*TESTCASES[135])

    def test_136(self):
        self._check(*TESTCASES[136])

    def test_137(self):
        self._check(*TESTCASES[137])

    def test_138(self):
        self._check(*TESTCASES[138])

    def test_139(self):
        self._check(*TESTCASES[139])

