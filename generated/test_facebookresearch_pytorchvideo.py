import sys
_module = sys.modules[__name__]
del sys
conf = _module
hubconf = _module
dataset = _module
dataset_utils = _module
download_objectron_data = _module
pytorchvideo = _module
accelerator = _module
deployment = _module
common = _module
model_transmuter = _module
mobile_cpu = _module
transmuter = _module
transmuter_mobile_cpu = _module
utils = _module
model_conversion = _module
efficient_blocks = _module
efficient_block_base = _module
no_op_convert_block = _module
data = _module
ava = _module
charades = _module
clip_sampling = _module
dataset_manifest_utils = _module
decoder = _module
domsev = _module
ego4d = _module
ego4d_dataset = _module
encoded_video = _module
encoded_video_decord = _module
encoded_video_pyav = _module
encoded_video_torchvision = _module
epic_kitchen = _module
epic_kitchen_dataset = _module
epic_kitchen_forecasting = _module
epic_kitchen_recognition = _module
frame_video = _module
hmdb51 = _module
json_dataset = _module
kinetics = _module
labeled_video_dataset = _module
labeled_video_paths = _module
ssv2 = _module
ucf101 = _module
utils = _module
video = _module
layers = _module
activation_functions = _module
attention = _module
conv_helper = _module
convolutions = _module
fully_connected = _module
pool = _module
attention = _module
attention_torchscript = _module
batch_norm = _module
convolutions = _module
distributed = _module
drop_path = _module
fusion = _module
mlp = _module
nonlocal_net = _module
positional_encoding = _module
positional_encoding_torchscript = _module
squeeze_excitation = _module
swish = _module
losses = _module
soft_target_cross_entropy = _module
models = _module
efficient_x3d = _module
residual_blocks = _module
audio_visual_slowfast = _module
byol = _module
csn = _module
head = _module
hub = _module
csn = _module
efficient_x3d_mobile_cpu = _module
r2plus1d = _module
resnet = _module
slowfast = _module
utils = _module
vision_transformers = _module
x3d = _module
masked_multistream = _module
memory_bank = _module
net = _module
r2plus1d = _module
resnet = _module
simclr = _module
slowfast = _module
stem = _module
vision_transformers = _module
weight_init = _module
x3d = _module
detection_hook = _module
engine = _module
hook = _module
transforms = _module
augmentations = _module
augmix = _module
functional = _module
mix = _module
rand_augment = _module
transforms = _module
transforms_factory = _module
pytorchvideo_trainer = _module
callbacks = _module
precise_batchnorm = _module
datamodule = _module
collators = _module
datamodule = _module
rand_erase_transform = _module
transforms = _module
module = _module
byol = _module
distributed_utils = _module
losses = _module
lr_policy = _module
moco_v2 = _module
optimizer = _module
simclr = _module
ssl_helper = _module
video_classification = _module
train_app = _module
setup = _module
tests = _module
test_conf_datamodule = _module
test_conf_module = _module
test_task_byol = _module
test_task_moco_v2 = _module
test_task_module_all = _module
test_task_simclr = _module
test_task_video_classification = _module
util = _module
benchmark_accelerator_efficient_blocks = _module
benchmark_transforms = _module
test_accelerator_deployment_mobile_cpu_model_conversion = _module
test_accelerator_deployment_model_transmuter = _module
test_accelerator_efficient_blocks_mobile_cpu_activation_attention = _module
test_accelerator_efficient_blocks_mobile_cpu_conv3d = _module
test_accelerator_efficient_blocks_mobile_cpu_head_layer = _module
test_accelerator_efficient_blocks_mobile_cpu_residual_block = _module
test_accelerator_models_efficient_x3d = _module
test_data_ava_dataset = _module
test_data_charades_dataset = _module
test_data_dataset_manifest_utils = _module
test_data_domsev_dataset = _module
test_data_encoded_video = _module
test_data_epic_kitchen_dataset = _module
test_data_epic_kitchen_forecasting = _module
test_data_epic_kitchen_recognition = _module
test_data_epic_kitchen_utils = _module
test_data_frame_video = _module
test_data_json_dataset = _module
test_data_labeled_video_dataset = _module
test_data_ssv2_dataset = _module
test_data_utils = _module
test_fuse_bn = _module
test_layers_attention = _module
test_layers_convolutions = _module
test_layers_drop_path = _module
test_layers_fusion = _module
test_layers_mlp = _module
test_layers_nonlocal_net = _module
test_layers_positional_encoding = _module
test_layers_squeeze_excitation = _module
test_losses_soft_target_cross_entropy = _module
test_models_audio_visual_slowfast = _module
test_models_byol = _module
test_models_csn = _module
test_models_head = _module
test_models_hub_vision_transformers = _module
test_models_masked_multistream = _module
test_models_memory_bank = _module
test_models_r2plus1d = _module
test_models_resnet = _module
test_models_slowfast = _module
test_models_stem = _module
test_models_vision_transformers = _module
test_models_x3d = _module
test_simclr = _module
test_transforms = _module
test_uniform_clip_sampler = _module
utils = _module
slurm = _module
train = _module
visualization = _module

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


import numpy as np


import torch


from torch.utils.data import Dataset


from typing import List


import logging


import torch.nn as nn


from copy import deepcopy


from typing import Dict


from abc import abstractmethod


from collections import defaultdict


from typing import Any


from typing import Callable


from typing import Optional


from typing import Set


from typing import Type


import functools


import itertools


import torch.utils.data


import math


import random


import time


from enum import Enum


import torch.autograd.profiler as profiler


import torchaudio


from torchvision.transforms import CenterCrop


from torchvision.transforms import Compose


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from typing import BinaryIO


from typing import TypeVar


from typing import Union


import re


from typing import Iterable


from abc import ABC


from collections import OrderedDict


import numpy


from torch.nn.common_types import _size_3_t


import torch.fx


import torch.distributed as dist


from torch import nn


from torch._C._distributed_c10d import ProcessGroup


from torch.autograd.function import Function


import torch.nn.functional as F


import copy


from torchvision.ops import RoIAlign


from torch.hub import load_state_dict_from_url


from torch.nn.utils.rnn import pack_padded_sequence


from functools import partial


import warnings


from torch.nn.common_types import _size_2_t


from torchvision.transforms import Lambda


from torchvision.transforms._transforms_video import CenterCropVideo


from torchvision.transforms._transforms_video import NormalizeVideo


import torchvision


import torchvision.transforms.functional_tensor as F_t


from torchvision.transforms.functional import InterpolationMode


import torchvision.transforms


from typing import Generator


from torch.utils.data import DataLoader


from torch.utils.data._utils.collate import default_collate


from torch.utils.data import RandomSampler


from torch.utils.data.distributed import DistributedSampler


from typing import Mapping


from typing import Sequence


from typing import Literal


from typing import TypedDict


from torch.optim.lr_scheduler import _LRScheduler


from functools import wraps


import torchvision.io as io


from torch.utils.mobile_optimizer import optimize_for_mobile


from torch.utils.data import SequentialSampler


import collections


from torch.multiprocessing import Process


from torch.utils.data import DistributedSampler


from torch.utils.data import TensorDataset


import torch.nn


from collections import Counter


from itertools import permutations


from torchvision.transforms._transforms_video import RandomCropVideo


from torchvision.transforms._transforms_video import RandomHorizontalFlipVideo


import torchvision.transforms as transforms


from torchaudio.transforms import MelSpectrogram


from torchaudio.transforms import Resample


from types import SimpleNamespace


import matplotlib.pyplot as plt


class EfficientBlockBase(nn.Module):
    """
    PyTorchVideo/accelerator provides a set of efficient blocks
    that have optimal efficiency for each target hardware device.

    Each efficient block has two forms:
    - original form: this form is for training. When efficient block is instantiated,
        it is in this original form.
    - deployable form: this form is for deployment. Once the network is ready for
        deploy, it can be converted into deployable form for efficient execution
        on target hardware. One block is transformed into deployable form by calling
        convert() method. By conversion to deployable form,
        various optimization (operator fuse, kernel optimization, etc.) are applied.

    EfficientBlockBase is the base class for efficient blocks.
    All efficient blocks should inherit this base class
    and implement following methods:
    - forward(): same as required by nn.Module
    - convert(): called to convert block into deployable form
    """

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class NoOpConvertBlock(EfficientBlockBase):
    """
    This class provides an interface with EfficientBlockBase for modules that do not
    need convert.
    Args:
        model (nn.Module): NoOpConvertBlock takes model as input and generate a wrapper
            instance of EfficientBlockBase with same functionality as model, with no change
            applied when convert() is called.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def convert(self, *args, **kwargs):
        pass

    def forward(self, x):
        return self.model(x)


class _NaiveSwish(nn.Module):
    """
    Helper class to implement naive swish for deploy. It is not intended to be used to
    build network.
    """

    def __init__(self):
        super().__init__()
        self.mul_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.mul_func.mul(x, torch.sigmoid(x))


class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).

    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    """

    def forward(self, x):
        return SwishFunction.apply(x)


class HardSwish(EfficientBlockBase):
    """
    Hardswish activation function. It is natively supported by Pytorch Mobile, and has
    better latency than Swish in int8 mode.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


class ReLU(EfficientBlockBase):
    """
    ReLU activation function for EfficientBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


class Identity(EfficientBlockBase):
    """
    Identity operation for EfficientBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


class _Reshape(nn.Module):
    """
    Helper class to implement data reshape as a module.
    Args:
        reshape_size (tuple): size of data after reshape.
    """

    def __init__(self, reshape_size: Tuple):
        super().__init__()
        self.reshape_size = reshape_size

    def forward(self, x):
        return torch.reshape(x, self.reshape_size)


class _SkipConnectMul(nn.Module):
    """
    Helper class to implement skip multiplication.
    Args:
        layer (nn.Module): layer for skip multiplication. With input x, _SkipConnectMul
            implements layer(x)*x.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.mul_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.mul_func.mul(x, self.layer(x))


class SqueezeExcitation(EfficientBlockBase):
    """
    Efficient Squeeze-Excitation (SE). The Squeeze-Excitation block is described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    This implementation has the same instantiation interface as SE implementation in
    fvcore, and in original mode for training it is just a wrapped version of SE in
    fvcore. Since conv3d in original SE implementation of fvcore is not well supported
    by QNNPACK, here convert() method is implemented which converts class instance into
    a equivalent efficient deployable form.

    convert_flag variable is to record whether the SqueezeExcitation instance
    has been converted; SqueezeExcitation is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.
    """

    def __init__(self, num_channels: int, num_channels_reduced: Optional[int]=None, reduction_ratio: float=2.0, is_3d: bool=False, activation: Optional[nn.Module]=None) ->None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()
        self.se = SqueezeExcitationFVCore(num_channels, num_channels_reduced=num_channels_reduced, reduction_ratio=reduction_ratio, is_3d=is_3d, activation=activation)
        self.is_3d = is_3d
        self.convert_flag = False

    def convert(self, input_blob_size, **kwargs):
        """
        Converts into efficient version of squeeze-excite (SE) for CPU.
        It changes conv in original SE into linear layer (better supported by CPU).
        """
        if self.is_3d:
            avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            avg_pool = nn.AdaptiveAvgPool2d(1)
        """
        Reshape tensor size to (B, C) for linear layer.
        """
        reshape0 = _Reshape((input_blob_size[0], input_blob_size[1]))
        fc0 = nn.Linear(self.se.block[0].in_channels, self.se.block[0].out_channels, bias=not self.se.block[0].bias is None)
        state_dict_fc0 = deepcopy(self.se.block[0].state_dict())
        state_dict_fc0['weight'] = state_dict_fc0['weight'].squeeze()
        fc0.load_state_dict(state_dict_fc0)
        activation = deepcopy(self.se.block[1])
        fc1 = nn.Linear(self.se.block[2].in_channels, self.se.block[2].out_channels, bias=not self.se.block[2].bias is None)
        state_dict_fc1 = deepcopy(self.se.block[2].state_dict())
        state_dict_fc1['weight'] = state_dict_fc1['weight'].squeeze()
        fc1.load_state_dict(state_dict_fc1)
        sigmoid = deepcopy(self.se.block[3])
        """
        Output of linear layer has output shape of (B, C). Need to reshape to proper
        shape before multiplying with input tensor.
        """
        reshape_size_after_sigmoid = (input_blob_size[0], input_blob_size[1], 1, 1) + ((1,) if self.is_3d else ())
        reshape1 = _Reshape(reshape_size_after_sigmoid)
        se_layers = nn.Sequential(avg_pool, reshape0, fc0, activation, fc1, sigmoid, reshape1)
        self.se = _SkipConnectMul(se_layers)
        self.convert_flag = True

    def forward(self, x) ->torch.Tensor:
        out = self.se(x)
        return out


class _Conv3dTemporalKernel3Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with temporal kernel of 3 into equivalent conv2ds.
    In conv3d with temporal kernel 3 and input I, for output temporal index of t (O[:,:,t,:,:]),
    the conv can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t:t+3,:,:])
                 = conv2d_0(I[:,:,t,:,:]) + conv2d_1(I[:,:,t+1,:,:]) + conv2d_2(I[:,:,t+2,:,:])
    If bias is considered:
    O[:,:,t,:,:] = conv3d_w_bias(I[:,:,t:t+3,:,:])
                 = conv2d_0_wo_bias(I[:,:,t,:,:])
                   + conv2d_1_w_bias(I[:,:,t+1,:,:]) + conv2d_2_wo_bias(I[:,:,t+2,:,:])
    The input Conv3d also needs zero padding of size 1 in temporal dimension.
    """

    def __init__(self, conv3d_in: nn.Conv3d, input_THW_tuple: Tuple):
        """
        Args:
            conv3d_in (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            input_THW_tuple (tuple): input THW size for conv3d_in during forward.
        """
        super().__init__()
        assert conv3d_in.padding[0] == 1, f'_Conv3dTemporalKernel3Eq only support temporal padding of 1, but got {conv3d_in.padding[0]}'
        assert conv3d_in.padding_mode == 'zeros', f'_Conv3dTemporalKernel3Eq only support zero padding, but got {conv3d_in.padding_mode}'
        self._input_THW_tuple = input_THW_tuple
        padding_2d = conv3d_in.padding[1:]
        in_channels = conv3d_in.in_channels
        out_channels = conv3d_in.out_channels
        kernel_size = conv3d_in.kernel_size[1:]
        groups = conv3d_in.groups
        stride_2d = conv3d_in.stride[1:]
        if self._input_THW_tuple[0] > 1:
            self._conv2d_3_3_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_2d, stride=stride_2d, groups=groups, bias=False)
            self._conv2d_3_3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_2d, stride=stride_2d, groups=groups, bias=False)
        self._conv2d_3_3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_2d, stride=stride_2d, groups=groups, bias=conv3d_in.bias is not None)
        state_dict = conv3d_in.state_dict()
        state_dict_1 = deepcopy(state_dict)
        state_dict_1['weight'] = state_dict['weight'][:, :, 1]
        self._conv2d_3_3_1.load_state_dict(state_dict_1)
        if self._input_THW_tuple[0] > 1:
            state_dict_0 = deepcopy(state_dict)
            state_dict_0['weight'] = state_dict['weight'][:, :, 0]
            if conv3d_in.bias is not None:
                """
                Don't need bias for other conv2d instances to avoid duplicated addition of bias.
                """
                state_dict_0.pop('bias')
            self._conv2d_3_3_0.load_state_dict(state_dict_0)
            state_dict_2 = deepcopy(state_dict)
            state_dict_2['weight'] = state_dict['weight'][:, :, 2]
            if conv3d_in.bias is not None:
                state_dict_2.pop('bias')
            self._conv2d_3_3_2.load_state_dict(state_dict_2)
            self._add_funcs = nn.ModuleList([nn.quantized.FloatFunctional() for _ in range(2 * (self._input_THW_tuple[0] - 1))])
            self._cat_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Use three conv2d to emulate conv3d.
        This forward assumes zero padding of size 1 in temporal dimension.
        """
        if self._input_THW_tuple[0] > 1:
            out_tensor_list = []
            """
            First output plane in temporal dimension,
            conv2d_3_3_0 is skipped due to zero padding.
            """
            cur_tensor = self._add_funcs[0].add(self._conv2d_3_3_1(x[:, :, 0]), self._conv2d_3_3_2(x[:, :, 1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            for idx in range(2, self._input_THW_tuple[0]):
                cur_tensor = self._add_funcs[2 * idx - 3].add(self._add_funcs[2 * idx - 2].add(self._conv2d_3_3_0(x[:, :, idx - 2]), self._conv2d_3_3_1(x[:, :, idx - 1])), self._conv2d_3_3_2(x[:, :, idx])).unsqueeze(2)
                out_tensor_list.append(cur_tensor)
            """
            Last output plane in temporal domain, conv2d_3_3_2 is skipped due to zero padding.
            """
            cur_tensor = self._add_funcs[-1].add(self._conv2d_3_3_0(x[:, :, -2]), self._conv2d_3_3_1(x[:, :, -1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            return self._cat_func.cat(out_tensor_list, 2)
        else:
            return self._conv2d_3_3_1(x[:, :, 0]).unsqueeze(2)


class _Conv3dTemporalKernel5Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with kernel size of (5, k, k) into equivalent conv2ds.
    In such conv3d and input I, for output temporal index of t (O[:,:,t,:,:]), the conv
    can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t:t+5,:,:])
                 = conv2d_0(I[:,:,t,:,:]) + conv2d_1(I[:,:,t+1,:,:]) + conv2d_2(I[:,:,t+2,:,:])
                   + conv2d_3(I[:,:,t+3,:,:]) + conv2d_4(I[:,:,t+4,:,:])
    If bias is considered:
    O[:,:,t,:,:] = conv3d_w_bias(I[:,:,t:t+3,:,:])
                 = conv2d_0_wo_bias(I[:,:,t,:,:])
                   + conv2d_1_wo_bias(I[:,:,t+1,:,:]) + conv2d_2_w_bias(I[:,:,t+2,:,:])
                   + conv2d_3_wo_bias(I[:,:,t+1,:,:]) + conv2d_4_wo_bias(I[:,:,t+2,:,:])
    The input Conv3d also needs zero padding of size 2 in temporal dimension at begin and end.
    """

    def __init__(self, conv3d_in: nn.Conv3d, thw_shape: Tuple[int, int, int]):
        """
        Args:
            conv3d_in (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            thw_shape (tuple): input THW size for conv3d_in during forward.
        """
        super().__init__()
        assert conv3d_in.padding[0] == 2, f'_Conv3dTemporalKernel5Eq only support temporal padding of 2, but got {conv3d_in.padding[0]}'
        assert conv3d_in.padding_mode == 'zeros', f'_Conv3dTemporalKernel5Eq only support zero padding, but got {conv3d_in.padding_mode}'
        self._thw_shape = thw_shape
        padding_2d = conv3d_in.padding[1:]
        in_channels = conv3d_in.in_channels
        out_channels = conv3d_in.out_channels
        kernel_size = conv3d_in.kernel_size[1:]
        groups = conv3d_in.groups
        stride_2d = conv3d_in.stride[1:]
        t, h, w = self._thw_shape
        args_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'padding': padding_2d, 'stride': stride_2d, 'groups': groups}
        for iter_idx in range(5):
            if iter_idx != 2:
                if t > 1:
                    self.add_module(f'_conv2d_{iter_idx}', nn.Conv2d(**args_dict, bias=False))
            else:
                self.add_module(f'_conv2d_{iter_idx}', nn.Conv2d(**args_dict, bias=conv3d_in.bias is not None))
        original_state_dict = conv3d_in.state_dict()
        state_dict_to_load = deepcopy(original_state_dict)
        state_dict_to_load['weight'] = original_state_dict['weight'][:, :, 2]
        self._conv2d_2.load_state_dict(state_dict_to_load)
        if t > 1:
            if conv3d_in.bias is not None:
                state_dict_to_load.pop('bias')
            state_dict_to_load['weight'] = original_state_dict['weight'][:, :, 0]
            self._conv2d_0.load_state_dict(state_dict_to_load)
            state_dict_to_load['weight'] = original_state_dict['weight'][:, :, 1]
            self._conv2d_1.load_state_dict(state_dict_to_load)
            state_dict_to_load['weight'] = original_state_dict['weight'][:, :, 3]
            self._conv2d_3.load_state_dict(state_dict_to_load)
            state_dict_to_load['weight'] = original_state_dict['weight'][:, :, 4]
            self._conv2d_4.load_state_dict(state_dict_to_load)
            self._add_funcs = nn.ModuleList([nn.quantized.FloatFunctional() for _ in range(4 * t - 6)])
            self._cat_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Use three conv2d to emulate conv3d.
        Args:
           x (torch.Tensor): 5D tensor of (B, C, T, H, W)
        """
        t, h, w = self._thw_shape
        out_tensor_list = []
        if t == 1:
            return self._conv2d_2(x[:, :, 0]).unsqueeze(2)
        elif t == 2:
            cur_tensor = self._add_funcs[0].add(self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[1].add(self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        elif t == 3:
            cur_tensor = self._add_funcs[0].add(self._add_funcs[1].add(self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])), self._conv2d_4(x[:, :, 2])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[2].add(self._add_funcs[3].add(self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1])), self._conv2d_3(x[:, :, 2])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[4].add(self._add_funcs[5].add(self._conv2d_0(x[:, :, 0]), self._conv2d_1(x[:, :, 1])), self._conv2d_2(x[:, :, 2])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        elif t == 4:
            cur_tensor = self._add_funcs[0].add(self._add_funcs[1].add(self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])), self._conv2d_4(x[:, :, 2])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[2].add(self._add_funcs[3].add(self._add_funcs[4].add(self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1])), self._conv2d_3(x[:, :, 2])), self._conv2d_4(x[:, :, 3])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[5].add(self._add_funcs[6].add(self._add_funcs[7].add(self._conv2d_0(x[:, :, 0]), self._conv2d_1(x[:, :, 1])), self._conv2d_2(x[:, :, 2])), self._conv2d_3(x[:, :, 3])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            cur_tensor = self._add_funcs[8].add(self._add_funcs[9].add(self._conv2d_0(x[:, :, 1]), self._conv2d_1(x[:, :, 2])), self._conv2d_2(x[:, :, 3])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        else:
            add_func_idx_base = 0
            cur_tensor = self._add_funcs[add_func_idx_base].add(self._add_funcs[add_func_idx_base + 1].add(self._conv2d_2(x[:, :, 0]), self._conv2d_3(x[:, :, 1])), self._conv2d_4(x[:, :, 2])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 2
            cur_tensor = self._add_funcs[add_func_idx_base].add(self._add_funcs[add_func_idx_base + 1].add(self._add_funcs[add_func_idx_base + 2].add(self._conv2d_1(x[:, :, 0]), self._conv2d_2(x[:, :, 1])), self._conv2d_3(x[:, :, 2])), self._conv2d_4(x[:, :, 3])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 3
            for idx in range(4, t):
                cur_tensor = self._add_funcs[add_func_idx_base].add(self._add_funcs[add_func_idx_base + 1].add(self._add_funcs[add_func_idx_base + 2].add(self._add_funcs[add_func_idx_base + 3].add(self._conv2d_0(x[:, :, idx - 4]), self._conv2d_1(x[:, :, idx - 3])), self._conv2d_2(x[:, :, idx - 2])), self._conv2d_3(x[:, :, idx - 1])), self._conv2d_4(x[:, :, idx])).unsqueeze(2)
                out_tensor_list.append(cur_tensor)
                add_func_idx_base += 4
            cur_tensor = self._add_funcs[add_func_idx_base].add(self._add_funcs[add_func_idx_base + 1].add(self._add_funcs[add_func_idx_base + 2].add(self._conv2d_0(x[:, :, -4]), self._conv2d_1(x[:, :, -3])), self._conv2d_2(x[:, :, -2])), self._conv2d_3(x[:, :, -1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
            add_func_idx_base += 3
            cur_tensor = self._add_funcs[add_func_idx_base].add(self._add_funcs[add_func_idx_base + 1].add(self._conv2d_0(x[:, :, -3]), self._conv2d_1(x[:, :, -2])), self._conv2d_2(x[:, :, -1])).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        return self._cat_func.cat(out_tensor_list, 2)


class _Conv3dTemporalKernel1Decomposed(nn.Module):
    """
    Helper class for decomposing conv3d with temporal kernel of 1 into conv2d on
    multiple temporal planes.
    In conv3d with temporal kernel 1 and input I, for output temporal index of t (O[:,:,t,:,:]),
    the conv can be expressed as:
    O[:,:,t,:,:] = conv3d(I[:,:,t,:,:])
                 = conv2d(I[:,:,t,:,:])
    The full output can be obtained by concat O[:,:,t,:,:] for t in 0...T,
    where T is the length of I in temporal dimension.
    """

    def __init__(self, conv3d_eq: nn.Conv3d, input_THW_tuple: Tuple):
        """
        Args:
            conv3d_eq (nn.Module): input nn.Conv3d module to be converted
                into equivalent conv2d.
            input_THW_tuple (tuple): input THW size for conv3d_eq during forward.
        """
        super().__init__()
        in_channels = conv3d_eq.in_channels
        out_channels = conv3d_eq.out_channels
        bias_flag = conv3d_eq.bias is not None
        self.conv2d_eq = nn.Conv2d(in_channels, out_channels, kernel_size=(conv3d_eq.kernel_size[1], conv3d_eq.kernel_size[2]), stride=(conv3d_eq.stride[1], conv3d_eq.stride[2]), groups=conv3d_eq.groups, bias=bias_flag, padding=(conv3d_eq.padding[1], conv3d_eq.padding[2]), dilation=(conv3d_eq.dilation[1], conv3d_eq.dilation[2]))
        state_dict = conv3d_eq.state_dict()
        state_dict['weight'] = state_dict['weight'].squeeze(2)
        self.conv2d_eq.load_state_dict(state_dict)
        self.input_THW_tuple = input_THW_tuple

    def forward(self, x):
        out_tensor_list = []
        for idx in range(self.input_THW_tuple[0]):
            cur_tensor = self.conv2d_eq(x[:, :, idx]).unsqueeze(2)
            out_tensor_list.append(cur_tensor)
        return torch.cat(out_tensor_list, 2)


supported_act_functions = {'relu': ReLU, 'swish': Swish, 'hswish': HardSwish, 'identity': Identity}


class Conv3dPwBnAct(EfficientBlockBase):
    """
    Implements Conv3d + Bn + Activation for pointwise layers.
    The conv layer has fixed kernel_size = (1,1,1),
    groups = 1, padding = 0, stride = 1, dilation = 1.

                          Input
                            |
                            ↓
                        conv3d (1x1x1)
                            ↓
                        BatchNorm (optional)
                            ↓
                        Activation

    Conv3dPwBnAct is in original form (for training) once instantiated. User can
    call convert() method to convert it into deployable form for deployment.

    convert_flag variable is to record whether the Conv3dPwBnAct instance
    has been converted; Conv3dPwBnAct is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Current implementation of this layer in QNNPACK is very efficient.
    Args:
        in_channels (int): number of input channels for conv3d 1x1x1.
        out_channels (int): number of output channels for conv3d 1x1x1.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(self, in_channels: int, out_channels: int, bias=False, activation: str='relu', use_bn=True, norm_eps: float=1e-05, norm_momentum: float=0.1):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.act = activation
        kernel = OrderedDict()
        kernel['conv'] = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
        if use_bn:
            kernel['bn'] = nn.BatchNorm3d(out_channels, eps=norm_eps, momentum=norm_momentum)
        assert activation in supported_act_functions, f'Conv3dPwBnAct: {activation} is not in supported_act_functions.'
        kernel['act'] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, convert_for_quantize: bool=False, native_conv3d_op_qnnpack: bool=False, **kwargs):
        """
        Converts the block into efficient form.
        For fp32 operation, or quantized but with older version of QNNPACK w/o native int8
        Conv3d support, this function converts Conv3d into equivalent Conv2d for Pytorch
        Mobile deployment.
        The Conv3d -> Conv2d conversion is done by first fuse conv3d with bn,
        convert conv3d into equivalent conv2d, and optionally fuse conv2d with relu.
        After conversion, the forwarding of this module becomes:
        Input (5d tensor) --> reshape (4d tensor) --> conv2d (4d tensor)
            --> reshape (5d tensor) --> output (5d tensor)

        For quantized operation on new version of QNNPACK with native int8 Conv3d, this
        function will only apply operator fusion.
        Args:
            input_blob_size (tuple): blob size at the input of Conv3dPwBnAct instance.
            convert_for_quantize (bool): whether this module is intended to be quantized.
            native_conv3d_op_qnnpack (bool): whether the QNNPACK version has native int8
                Conv3d.
            kwargs (any): any extra keyword arguments from upstream unused by convert().
        """
        assert self.convert_flag is False, 'Conv3dPwBnAct: already converted, cannot be converted again'
        self.kernel.eval()
        if hasattr(self.kernel, 'bn'):
            self.kernel = fuse_modules(self.kernel, ['conv', 'bn'])
        if convert_for_quantize and native_conv3d_op_qnnpack:
            if self.act == 'relu':
                self.kernel = fuse_modules(self.kernel, ['conv', 'act.act'])
            self.kernel.eval()
        else:
            batch_size = input_blob_size[0]
            input_THW_tuple = input_blob_size[2:]
            self._input_tensor_reshape_size = batch_size, self._in_channels, input_THW_tuple[0] * input_THW_tuple[1], input_THW_tuple[2]
            self._output_tensor_size = batch_size, self._out_channels, input_THW_tuple[0], input_THW_tuple[1], input_THW_tuple[2]
            conv2d_eq = nn.Conv2d(self._in_channels, self._out_channels, kernel_size=1, bias=self.kernel.conv.bias is not None)
            conv_state_dict = self.kernel.conv.state_dict()
            conv_state_dict['weight'] = conv_state_dict['weight'].squeeze(2)
            conv2d_eq.load_state_dict(conv_state_dict)
            self.kernel.conv = conv2d_eq
            self.kernel.act.convert(input_blob_size, **kwargs)
            if self.act == 'relu':
                self.kernel = fuse_modules(self.kernel, ['conv', 'act.act'])
            self.kernel = nn.Sequential(_Reshape(self._input_tensor_reshape_size), self.kernel, _Reshape(self._output_tensor_size))
            self.kernel.eval()
        self.convert_flag = True

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d3x3x3DwBnAct(EfficientBlockBase):
    """
    Implements Conv3d (3x3x3 dw) + (optional) Bn + Activation layers.
    The conv layer has fixed kernel_size = (3,3,3), depthwise, zero padding size of
    (1,1,1), temporal stride = 1, dilation = 1

                      Input
                        |
                        ↓
                    conv3d (3x3x3 dw)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    Current implementation of this layer in QNNPACK is reasonably efficient.

    convert_flag variable is to record whether the Conv3d3x3x3DwBnAct instance
    has been converted; Conv3d3x3x3DwBnAct is in original form if convert_flag is false,
    while it is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of channels for conv3d 3x3x3 dw.
        spatial_stride (tuple length of 2): spatial stride for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    Current implementation of this layer in Pytorch Mobile is efficient.
    Sidenote: QNNPACK has best support for dw with 3x3 spatial kernel.
    For other spatial kernels like 7x7 dw, the efficiency may be lower.
    """

    def __init__(self, in_channels: int, spatial_stride: int=1, bias=False, activation: str='relu', use_bn=True, norm_eps: float=1e-05, norm_momentum: float=0.1):
        super().__init__()
        kernel = OrderedDict()
        conv_stride = 1, spatial_stride, spatial_stride
        kernel['conv'] = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=conv_stride, groups=in_channels, padding=1, bias=bias)
        if use_bn:
            kernel['bn'] = nn.BatchNorm3d(in_channels, eps=norm_eps, momentum=norm_momentum)
        assert activation in supported_act_functions, f'Conv3d3x3x3DwBnAct: {activation} is not in supported_act_functions.'
        kernel['act'] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, convert_for_quantize: bool=False, native_conv3d_op_qnnpack: bool=False, **kwargs):
        """
        Converts the block into efficient form.
        For fp32 operation, or quantized but with older version of QNNPACK w/o native int8
        Conv3d support, this function converts Conv3d into equivalent Conv2d for Pytorch
        Mobile deployment.
        For quantized operation on new version of QNNPACK with native int8 Conv3d, this
        function will only apply operator fusion.
        Args:
            input_blob_size (tuple): blob size at the input of Conv3d3x3x3DwBnAct
                instance during forward.
            convert_for_quantize (bool): whether this module is intended to be quantized.
            native_conv3d_op_qnnpack (bool): whether the QNNPACK version has native int8
                Conv3d.
            kwargs (any): any keyword argument (unused).
        """
        assert self.convert_flag is False, 'Conv3d3x3x3DwBnAct: already converted, cannot be converted twice.'
        self.kernel.eval()
        if hasattr(self.kernel, 'bn'):
            self.kernel = fuse_modules(self.kernel, ['conv', 'bn'])
        if convert_for_quantize is False or native_conv3d_op_qnnpack is False:
            self.kernel.conv = _Conv3dTemporalKernel3Decomposed(self.kernel.conv, input_blob_size[2:])
        self.kernel.act.convert(input_blob_size, **kwargs)
        """
        Since conv3d is converted into multiple conv2d,
        will not fuse conv with act to keep arithmetic equivalency.
        """
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3dTemporalKernel1BnAct(EfficientBlockBase):
    """
    Implements Conv3d + Bn + Activation where Conv3d has temporal kernel of 1.
    The conv layer has padding[0] = 0, stride[0] = 1, dilation[0] = 1.

                                  Input
                                    |
                                    ↓
                                conv3d (1xkxk)
                                    ↓
                                BatchNorm (optional)
                                    ↓
                                Activation

    Current implementation of this layer in QNNPACK is reasonably efficient
    (not as efficient as Conv3dPwBnAct for 1x1x1 kernel).
    Args:
        in_channels (int): number of input channels for conv3d 1x1x1.
        out_channels (int): number of output channels for conv3d 1x1x1.
        bias (bool): if true, use bias for conv.
        groups (int): number of groups for conv.
        spstial_kernel (int): spatial kernel for conv3d.
        spstial_stride (int): spatial stride for conv3d.
        spatial_padding (int): spatial padding for conv3d.
        spatial_dilation (int): spatial dilation for conv3d.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(self, in_channels: int, out_channels: int, bias=False, groups: int=1, spatial_kernel: int=1, spatial_stride: int=1, spatial_padding: int=0, spatial_dilation: int=1, activation: str='relu', use_bn=True, norm_eps: float=1e-05, norm_momentum: float=0.1):
        super().__init__()
        kernel_size = 1, spatial_kernel, spatial_kernel
        stride = 1, spatial_stride, spatial_stride
        padding = 0, spatial_padding, spatial_padding
        dilation = 1, spatial_dilation, spatial_dilation
        kernel = OrderedDict()
        kernel['conv'] = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            kernel['bn'] = nn.BatchNorm3d(out_channels, eps=norm_eps, momentum=norm_momentum)
        assert activation in supported_act_functions, f'Conv3dTemporalKernel1BnAct: {activation} is not in supported_act_functions.'
        kernel['act'] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, **kwargs):
        """
        Converts Conv3d into equivalent Conv2d for QNNPACK deployment.
        This conversion is done by first fuse conv3d with bn,
        convert conv3d into equivalent conv2d,
        and optionally fuse conv2d with relu.
        Args:
            input_blob_size (tuple): blob size at the input of
                Conv3dTemporalKernel1BnAct instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert self.convert_flag is False, 'Conv3dTemporalKernel1BnAct: already converted, cannot be converted again'
        self.kernel.eval()
        if hasattr(self.kernel, 'bn'):
            self.kernel = fuse_modules(self.kernel, ['conv', 'bn'])
        self.kernel.conv = _Conv3dTemporalKernel1Decomposed(self.kernel.conv, input_blob_size[2:])
        self.kernel.act.convert(input_blob_size, **kwargs)
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d3x1x1BnAct(EfficientBlockBase):
    """
    Implements Conv3d (3x1x1) + (optional) Bn + Activation for pointwise layers.
    The conv layer has fixed kernel of (3, 1, 1), zero padding size of
    (1, 0, 0), stride = (1, 1, 1), dilation = 1.

                      Input
                        |
                        ↓
                    conv3d (3x1x1)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    For regular convolution (i.e., groups=1), current implementation of this layer in
    QNNPACK is reasonably efficient.
    For depthwise convolution (i.e., groups=out_channels), current implementation of this
    layer in QNNPACK is not efficient as Conv3d3x3x3DwBnRelu, as QNNPACK does not have
    optimization for 1x1 depthwise convolution. The latencies of fp32 operation are similar
    for Conv3d3x1x1BnAct and Conv3d3x3x3DwBnRelu, while with int8 operation Conv3d3x1x1BnAct
    is 1.5X slower than Conv3d3x3x3DwBnRelu.

    self.convert_flag property records whether the Conv3d3x1x1BnAct instance has been
    converted; Conv3d3x1x1BnAct is in original form if convert_flag is false, while it
    is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of input channels for conv3d 3x1x1.
        out_channels (int): number of output channels for conv3d 3x1x1.
        groups (int): number of groups for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(self, in_channels: int, out_channels: int, groups: int=1, bias=False, activation: str='relu', use_bn=True, norm_eps=1e-05, norm_momentum=0.1):
        super().__init__()
        kernel = OrderedDict()
        kernel['conv'] = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), groups=groups, padding=(1, 0, 0), bias=bias)
        if groups == out_channels:
            logging.warn('Conv3d3x1x1BnAct has low efficiency for depthwise conv. Consider using Conv3d3x3x3DwBnRelu instead.')
        if use_bn:
            kernel['bn'] = nn.BatchNorm3d(out_channels, eps=norm_eps, momentum=norm_momentum)
        assert activation in supported_act_functions, f'Conv3d3x1x1BnAct: {activation} is not in supported_act_functions.'
        kernel['act'] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size, **kwargs):
        """
        Converts Conv3d into equivalent Conv2d for Pytorch Mobile deployment

        """
        assert self.convert_flag is False, 'Conv3d3x1x1BnAct: already converted, cannot be converted twice'
        self.kernel.eval()
        if hasattr(self.kernel, 'bn'):
            self.kernel = fuse_modules(self.kernel, ['conv', 'bn'])
        self.kernel.conv = _Conv3dTemporalKernel3Decomposed(self.kernel.conv, input_blob_size[2:])
        self.kernel.act.convert(input_blob_size, **kwargs)
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class Conv3d5x1x1BnAct(EfficientBlockBase):
    """
    Implements Conv3d (5x1x1) + (optional) Bn + Activation for pointwise layers.
    The conv layer has fixed kernel of (5, 1, 1), zero padding size of
    (2, 0, 0), stride = (1, 1, 1), dilation = 1.

                      Input
                        |
                        ↓
                    conv3d (5x1x1)
                        ↓
                    BatchNorm (optional)
                        ↓
                    Activation

    For regular convolution (i.e., groups=1), current implementation of this layer in
    QNNPACK is reasonably efficient.

    self.convert_flag property records whether the Conv3d5x1x1BnAct instance has been
    converted; Conv3d5x1x1BnAct is in original form if convert_flag is false, while it
    is in deployable form if convert_flag is true.

    Args:
        in_channels (int): number of input channels for conv3d 3x1x1.
        out_channels (int): number of output channels for conv3d 3x1x1.
        groups (int): number of groups for conv.
        bias (bool): if true, use bias for conv.
        activation (str): applies selected activation from supported_act_functions.
            See activation_functions.py for more info about supported activations.
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (bool): if true, use batchnorm.
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(self, in_channels: int, out_channels: int, groups: int=1, bias=False, activation: str='relu', use_bn=True, norm_eps=1e-05, norm_momentum=0.1):
        super().__init__()
        kernel = OrderedDict()
        kernel['conv'] = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 1, 1), groups=groups, padding=(2, 0, 0), bias=bias)
        if use_bn:
            kernel['bn'] = nn.BatchNorm3d(out_channels, eps=norm_eps, momentum=norm_momentum)
        assert activation in supported_act_functions, f'Conv3d5x1x1BnAct: {activation} is not in supported_act_functions.'
        kernel['act'] = supported_act_functions[activation]()
        self.kernel = nn.Sequential(kernel)
        self.convert_flag = False

    def convert(self, input_blob_size, **kwargs):
        """
        Converts Conv3d into equivalent Conv2d for Pytorch Mobile deployment

        """
        assert self.convert_flag is False, 'Conv3d5x1x1BnAct: already converted, cannot be converted twice'
        self.kernel.eval()
        if hasattr(self.kernel, 'bn'):
            self.kernel = fuse_modules(self.kernel, ['conv', 'bn'])
        self.kernel.conv = _Conv3dTemporalKernel5Decomposed(self.kernel.conv, input_blob_size[2:])
        self.kernel.act.convert(input_blob_size, **kwargs)
        self.convert_flag = True
        self.kernel.eval()

    def forward(self, x):
        x = self.kernel(x)
        return x


class FullyConnected(NoOpConvertBlock):
    """
    Implements fully connected layer. This operator is natively supported by QNNPACK for
    mobile CPU with good efficiency, and no change is made upon convert().
    Args:
        in_features (int): input channels for FC layer.
        out_features (int): output channels for FC layer.
        bias (bool): if True, bias is applied
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__(model=nn.Linear(in_features, out_features, bias=bias))


class AdaptiveAvgPool3dOutSize1(EfficientBlockBase):
    """
    Implements AdaptiveAvgPool3d with output (T, H, W) = (1, 1, 1). This operator has
    better efficiency than AdaptiveAvgPool for mobile CPU.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, **kwargs):
        """
        Converts AdaptiveAvgPool into AvgPool with constant kernel size for better
        efficiency.
        Args:
            input_blob_size (tuple): blob size at the input of
                AdaptiveAvgPool3dOutSize1 instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert self.convert_flag is False, 'AdaptiveAvgPool3dOutSize1: already converted, cannot be converted again'
        kernel_size = input_blob_size[2:]
        self.pool = nn.AvgPool3d(kernel_size)
        self.convert_flag = True

    def forward(self, x):
        return self.pool(x)


class AdaptiveAvgPool2dOutSize1(EfficientBlockBase):
    """
    Implements AdaptiveAvgPool2d with output (H, W) = (1, 1). This operator has
    better efficiency than AdaptiveAvgPool for mobile CPU.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.convert_flag = False

    def convert(self, input_blob_size: Tuple, **kwargs):
        """
        Converts AdaptiveAvgPool into AvgPool with constant kernel size for better
        efficiency.
        Args:
            input_blob_size (tuple): blob size at the input of
                AdaptiveAvgPool2dOutSize1 instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert self.convert_flag is False, 'AdaptiveAvgPool2dOutSize1: already converted, cannot be converted again'
        kernel_size = input_blob_size[2:]
        self.pool = nn.AvgPool2d(kernel_size)
        self.convert_flag = True

    def forward(self, x):
        return self.pool(x)


class AdaptiveAvgPool3d(NoOpConvertBlock):
    """
    Implements AdaptiveAvgPool3d with any output (T, H, W) size. This operator is
    supported by QNNPACK for mobile CPU with resonable efficiency, and no change is
    made upon convert(). If the output (T, H, W) = (1, 1, 1), use AdaptiveAvgPool3dOutSize1
    for better efficiency.
    Args:
        output_size (int or tuple): when it is a tuple, the output (T, H, W) of pool
            will be equal to output_size. When it is an int, the output (T, H, W)
            will be equal to (output_size, output_size, output_size).
    """

    def __init__(self, output_size: Union[int, Tuple]):
        super().__init__(model=nn.AdaptiveAvgPool3d(output_size))


class AdaptiveAvgPool2d(NoOpConvertBlock):
    """
    Implements AdaptiveAvgPool2d with any output (H, W) size. This operator is
    supported by QNNPACK for mobile CPU with resonable efficiency, and no change is
    made upon convert(). If the output (H, W) = (1, 1), use AdaptiveAvgPool2dOutSize1
    for better efficiency.
    Args:
        output_size (int or tuple): when it is a tuple, the output (H, W) of pool
            will be equal to output_size. When it is an int, the output (H, W)
            will be equal to (output_size, output_size).
    """

    def __init__(self, output_size: Union[int, Tuple]):
        super().__init__(model=nn.AdaptiveAvgPool2d(output_size))


class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.

    ::

                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer=nn.GELU, dropout_rate: float=0.0, bias_on: bool=True) ->None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_on)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_on)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@torch.fx.wrap
def _squeeze_dims_fx(tensor: torch.Tensor, tensor_dim: int) ->torch.Tensor:
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.squeeze(1)
    else:
        raise NotImplementedError(f'Unsupported input dimension {tensor.shape}')
    return tensor


@torch.jit.script
def _squeeze_dims_jit(tensor: torch.Tensor, tensor_dim: int) ->torch.Tensor:
    return _squeeze_dims_fx(tensor, tensor_dim)


@torch.fx.wrap
def _unsqueeze_dims_fx(tensor: torch.Tensor) ->Tuple[torch.Tensor, int]:
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f'Unsupported input dimension {tensor.shape}')
    return tensor, tensor_dim


@torch.jit.script
def _unsqueeze_dims_jit(tensor: torch.Tensor) ->Tuple[torch.Tensor, int]:
    return _unsqueeze_dims_fx(tensor)


class _AttentionPool(torch.nn.Module):

    def __init__(self, pool: Optional[torch.nn.Module], has_cls_embed: bool, norm: Optional[torch.nn.Module]) ->None:
        """Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


        Params:
            pool (Optional[Callable]): Pool operation that is applied to the input tensor.
                If pool is none, return the input tensor.
            has_cls_embed (bool): Whether the input tensor contains cls token. Pool
                operation excludes cls token.
            norm: (Optional[Callable]): Optional normalization operation applied to
            tensor after pool.
        """
        super().__init__()
        self.has_pool = pool is not None
        self.pool = pool if pool is not None else torch.nn.Identity()
        self.has_cls_embed = has_cls_embed
        if norm is not None:
            self.norm_before_pool = isinstance(norm, (torch.nn.BatchNorm3d, torch.nn.Identity))
            self.has_norm = True
            self.norm = norm
        else:
            self.norm_before_pool = False
            self.has_norm = False
            self.norm = torch.nn.Identity()

    def forward(self, tensor: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int]]:
        """
        Args:
            tensor (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).

        Returns:
            tensor (torch.Tensor): Input tensor after pool.
            thw_shape (List[int]): Output tensor shape (before flattening).
        """
        if not self.has_pool:
            return tensor, thw_shape
        tensor_dim = tensor.ndim
        if torch.jit.is_scripting():
            tensor, tensor_dim = _unsqueeze_dims_jit(tensor)
        else:
            tensor, tensor_dim = _unsqueeze_dims_fx(tensor)
        cls_tok: torch.Tensor = torch.tensor(0)
        if self.has_cls_embed:
            cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]
        B, N, L, C = tensor.shape
        T, H, W = thw_shape
        tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        if self.norm_before_pool:
            tensor = self.norm(tensor)
            tensor = torch.nn.functional.gelu(tensor)
        tensor = self.pool(tensor)
        thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
        L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
        tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
        if self.has_cls_embed:
            tensor = torch.cat((cls_tok, tensor), dim=2)
        if self.has_norm and not self.norm_before_pool:
            tensor = self.norm(tensor)
        if torch.jit.is_scripting():
            tensor = _squeeze_dims_jit(tensor, tensor_dim)
        else:
            tensor = _squeeze_dims_fx(tensor, tensor_dim)
        return tensor, thw_shape


def _post_attention_pool(tensor: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int]]:
    B, N, L, C, T, H, W, tensor_dim = thw_shape
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if torch.jit.is_scripting():
        tensor = _squeeze_dims_jit(tensor, tensor_dim)
    else:
        tensor = _squeeze_dims_fx(tensor, tensor_dim)
    return tensor, thw_shape


def _pre_attention_pool(tensor: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, Tuple[int, int, int, int, int, int, int, int]]:
    """
    Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


    Args:
        tensor (torch.Tensor): Input tensor.
        pool (Optional[Callable]): Pool operation that is applied to the input tensor.
            If pool is none, return the input tensor.
        thw_shape (List): The shape of the input tensor (before flattening).
        has_cls_embed (bool): Whether the input tensor contains cls token. Pool
            operation excludes cls token.
        norm: (Optional[Callable]): Optional normalization operation applied to
         tensor after pool.

    Returns:
        tensor (torch.Tensor): Input tensor after pool.
        thw_shape (List[int]): Output tensor shape (before flattening).
    """
    if torch.jit.is_scripting():
        tensor, tensor_dim = _unsqueeze_dims_jit(tensor)
    else:
        tensor, tensor_dim = _unsqueeze_dims_fx(tensor)
    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return tensor, (B, N, L, C, T, H, W, tensor_dim)


class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.

    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=False, dropout_rate: float=0.0, kernel_q: _size_3_t=(1, 1, 1), kernel_kv: _size_3_t=(1, 1, 1), stride_q: _size_3_t=(1, 1, 1), stride_kv: _size_3_t=(1, 1, 1), norm_layer=nn.LayerNorm, has_cls_embed: bool=True, pool_mode: str='conv', pool_first: bool=False, residual_pool: bool=True, depthwise_conv: bool=True, bias_on: bool=True, separate_qkv: bool=True) ->None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        assert pool_mode in ['conv', 'avg', 'max']
        assert not pool_first
        assert not has_cls_embed
        assert not separate_qkv
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True if bias_on else False)
        if kernel_q is not None and numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = None
        if kernel_kv is not None and numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = None
        if pool_mode in ('avg', 'max'):
            pool_op = nn.MaxPool3d if pool_mode == 'max' else nn.AvgPool3d
            self.pool_q = pool_op(kernel_q, stride_q, padding_q, ceil_mode=False) if kernel_q is not None else None
            self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if kernel_kv is not None else None
            self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if kernel_kv is not None else None
        elif pool_mode == 'conv':
            self.pool_q = nn.Conv3d(head_dim, head_dim, kernel_q, stride=stride_q, padding=padding_q, groups=head_dim if depthwise_conv else 1, bias=False) if kernel_q is not None else None
            self.pool_k = nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim if depthwise_conv else 1, bias=False) if kernel_kv is not None else None
            self.pool_v = nn.Conv3d(head_dim, head_dim, kernel_kv, stride=stride_kv, padding=padding_kv, groups=head_dim if depthwise_conv else 1, bias=False) if kernel_kv is not None else None
        else:
            raise NotImplementedError(f'Unsupported model {pool_mode}')

    def _qkv_proj(self, q: torch.Tensor, q_size: List[int], k: torch.Tensor, k_size: List[int], v: torch.Tensor, v_size: List[int], batch_size: List[int], chan_size: List[int]) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(q).reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads).permute(0, 2, 1, 3)
        return q, k, v

    def _qkv_pool(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]]:
        if self.pool_q is None:
            q_shape = thw_shape
        else:
            q, q_shape = _pre_attention_pool(q, [thw_shape[0], thw_shape[1], thw_shape[2]])
            q = nn.functional.gelu(q)
            q = self.pool_q(q)
            q, q_shape = _post_attention_pool(q, q_shape)
        if self.pool_k is None:
            k_shape = thw_shape
        else:
            k, k_shape = _pre_attention_pool(k, [thw_shape[0], thw_shape[1], thw_shape[2]])
            k = nn.functional.gelu(k)
            k = self.pool_k(k)
            k, k_shape = _post_attention_pool(k, k_shape)
        if self.pool_v is None:
            v_shape = thw_shape
        else:
            v, v_shape = _pre_attention_pool(v, [thw_shape[0], thw_shape[1], thw_shape[2]])
            v = nn.functional.gelu(v)
            v = self.pool_v(v)
            v, v_shape = _post_attention_pool(v, v_shape)
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(self, q_shape: List[int], k_shape: List[int], v_shape: List[int]) ->Tuple[int]:
        q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
        k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
        v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
        return q_N, k_N, v_N

    def _reshape_qkv_to_seq(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_N: int, v_N: int, k_N: int, B: int, C: int) ->Tuple[int]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def forward(self, x: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, q_shape, k, k_shape, v, v_shape = self._qkv_pool(q, k, v, thw_shape)
        attn = q * self.scale @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        N = q.shape[2]
        if self.residual_pool:
            x = (attn @ v + q).transpose(1, 2).reshape(B, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, q_shape


def drop_path(x: torch.Tensor, drop_prob: float=0.0, training: bool=False) ->torch.Tensor:
    """
    Stochastic Depth per sample.

    Args:
        x (tensor): Input tensor.
        drop_prob (float): Probability to apply drop path.
        training (bool): If True, apply drop path to input. Otherwise (tesing), return input.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float=0.0) ->None:
        """
        Args:
            drop_prob (float): Probability to apply drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        return drop_path(x, self.drop_prob, self.training)


class MultiScaleBlock(nn.Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, dropout_rate: float=0.0, droppath_rate: float=0.0, act_layer: nn.Module=nn.GELU, norm_layer: nn.Module=nn.LayerNorm, attn_norm_layer: nn.Module=nn.LayerNorm, kernel_q: _size_3_t=(1, 1, 1), kernel_kv: _size_3_t=(1, 1, 1), stride_q: _size_3_t=(1, 1, 1), stride_kv: _size_3_t=(1, 1, 1), pool_mode: str='conv', has_cls_embed: bool=True, pool_first: bool=False, residual_pool: bool=False, depthwise_conv: bool=True, bias_on: bool=True, separate_qkv: bool=True) ->None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.norm1_is_batchnorm_1d = isinstance(self.norm1, nn.BatchNorm1d)
        kernel_skip = [(s + 1 if s > 1 else s) for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=dropout_rate, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv, norm_layer=attn_norm_layer, has_cls_embed=has_cls_embed, pool_mode=pool_mode, pool_first=pool_first, residual_pool=residual_pool, bias_on=bias_on, depthwise_conv=depthwise_conv, separate_qkv=separate_qkv)
        self.drop_path = DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_is_batchnorm_1d = isinstance(self.norm2, nn.BatchNorm1d)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim_out, act_layer=act_layer, dropout_rate=dropout_rate, bias_on=bias_on)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out, bias=bias_on)
        else:
            self.proj = nn.Identity()
        self.pool_skip = nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False) if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1 else None
        self._attention_pool = _AttentionPool(self.pool_skip, has_cls_embed=self.has_cls_embed, norm=None)

    def forward(self, x: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """
        x_block, thw_shape_new = self.attn(self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1) if self.norm1_is_batchnorm_1d else self.norm1(x), thw_shape)
        x_res, _ = self._attention_pool(x, thw_shape)
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1) if self.norm2_is_batchnorm_1d else self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class ScriptableMultiScaleBlock(nn.Module):
    """
    Implementation of a multiscale vision transformer block. Each block contains a
    multiscale attention layer and a Mlp layer.

    ::


                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    """

    def __init__(self, dim: int, dim_out: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, dropout_rate: float=0.0, droppath_rate: float=0.0, act_layer: nn.Module=nn.GELU, norm_layer: nn.Module=nn.LayerNorm, attn_norm_layer: nn.Module=nn.LayerNorm, kernel_q: _size_3_t=(1, 1, 1), kernel_kv: _size_3_t=(1, 1, 1), stride_q: _size_3_t=(1, 1, 1), stride_kv: _size_3_t=(1, 1, 1), pool_mode: str='conv', has_cls_embed: bool=True, pool_first: bool=False, residual_pool: bool=False, depthwise_conv: bool=True, bias_on: bool=True, separate_qkv: bool=True) ->None:
        """
        Args:
            dim (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            num_heads (int): Number of heads in the attention layer.
            mlp_ratio (float): Mlp ratio which controls the feature dimension in the
                hidden layer of the Mlp block.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): DropOut rate. If set to 0, DropOut is disabled.
            droppath_rate (float): DropPath rate. If set to 0, DropPath is disabled.
            act_layer (nn.Module): Activation layer used in the Mlp layer.
            norm_layer (nn.Module): Normalization layer.
            attn_norm_layer (nn.Module): Normalization layer in the attention module.
            kernel_q (_size_3_t): Pooling kernel size for q. If pooling kernel size is
                1 for all the dimensions, pooling is not used (by default).
            kernel_kv (_size_3_t): Pooling kernel size for kv. If pooling kernel size
                is 1 for all the dimensions, pooling is not used. By default, pooling
                is disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """
        super().__init__()
        assert not pool_first
        assert not separate_qkv
        self.dim = dim
        self.dim_out = dim_out
        kernel_skip = [(s + 1 if s > 1 else s) for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=dropout_rate, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv, norm_layer=attn_norm_layer, has_cls_embed=has_cls_embed, pool_mode=pool_mode, pool_first=pool_first, residual_pool=residual_pool, bias_on=bias_on, depthwise_conv=depthwise_conv, separate_qkv=separate_qkv)
        self.drop_path = DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim_out, act_layer=act_layer, dropout_rate=dropout_rate, bias_on=bias_on)
        self.proj = nn.Linear(dim, dim_out, bias=bias_on) if dim != dim_out else nn.Identity()
        self.pool_skip = nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False) if len(stride_skip) > 0 and numpy.prod(stride_skip) > 1 else None

    def forward(self, x: torch.Tensor, thw_shape: List[int]) ->Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """
        x_block, thw_shape_new = self.attn(x, thw_shape)
        if self.pool_skip is None:
            x_res = x
        else:
            x_res, res_shape = _pre_attention_pool(x, [thw_shape[0], thw_shape[1], thw_shape[2]])
            x_res = self.pool_skip(x_res)
            x_res, _ = _post_attention_pool(x_res, res_shape)
        x = x_res + self.drop_path(x_block)
        x_norm = x
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    """
    An implementation of 1D naive sync batch normalization. See details in
    NaiveSyncBatchNorm2d below.

    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):
        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(f'Cannot set num_sync_devices separately when global_sync = {self.global_sync}')
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(f'num_sync_devices cannot be None when global_sync = {self.global_sync}')
        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (du.get_local_size(), self.num_sync_devices)
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm1d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)
        B, C = input.shape[0], input.shape[1]
        assert B > 0, 'SyncBatchNorm does not support zero batch size.'
        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])
        vec = torch.cat([mean, meansqr], dim=0)
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (1.0 / self.num_sync_devices)
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        return input * scale + bias


class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """
    An implementation of 2D naive sync batch normalization.
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.

    Note:
        This module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):
        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(f'Cannot set num_sync_devices separately when global_sync = {self.global_sync}')
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(f'num_sync_devices cannot be None when global_sync = {self.global_sync}')
        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (du.get_local_size(), self.num_sync_devices)
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm2d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)
        B, C = input.shape[0], input.shape[1]
        assert B > 0, 'SyncBatchNorm does not support zero batch size.'
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])
        vec = torch.cat([mean, meansqr], dim=0)
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (1.0 / self.num_sync_devices)
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        return input * scale + bias


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):
    """
    Naive version of Synchronized 3D BatchNorm. See details in
    NaiveSyncBatchNorm2d above.
    Args:
        num_sync_devices (int): number of (local) devices to sync.
        global_sync (bool): sync across all devices (on all machines).
        args (list): other arguments.
    """

    def __init__(self, num_sync_devices=None, global_sync=True, **args):
        self.global_sync = global_sync
        if self.global_sync and num_sync_devices is not None:
            raise ValueError(f'Cannot set num_sync_devices separately when global_sync = {self.global_sync}')
        if not self.global_sync and num_sync_devices is None:
            raise ValueError(f'num_sync_devices cannot be None when global_sync = {self.global_sync}')
        if not self.global_sync:
            self.num_sync_devices = num_sync_devices
            if self.num_sync_devices > 0:
                assert du.get_local_size() % self.num_sync_devices == 0, (du.get_local_size(), self.num_sync_devices)
                self.num_groups = du.get_local_size() // self.num_sync_devices
            else:
                self.num_sync_devices = du.get_local_size()
                self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        if du.get_world_size() == 1 or not self.training:
            return super().forward(input)
        B, C = input.shape[0], input.shape[1]
        assert B > 0, 'SyncBatchNorm does not support zero batch size.'
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])
        vec = torch.cat([mean, meansqr], dim=0)
        if self.global_sync:
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
        else:
            vec = du.GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (1.0 / self.num_sync_devices)
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        return input * scale + bias


class ConvReduce3D(nn.Module):
    """
    Builds a list of convolutional operators and performs summation on the outputs.

    ::

                            Conv3d, Conv3d, ...,  Conv3d
                                           ↓
                                          Sum
    """

    def __init__(self, *, in_channels: int, out_channels: int, kernel_size: Tuple[_size_3_t], stride: Optional[Tuple[_size_3_t]]=None, padding: Optional[Tuple[_size_3_t]]=None, padding_mode: Optional[Tuple[str]]=None, dilation: Optional[Tuple[_size_3_t]]=None, groups: Optional[Tuple[int]]=None, bias: Optional[Tuple[bool]]=None, reduction_method: str='sum') ->None:
        """
        Args:
            in_channels int: number of input channels.
            out_channels int: number of output channels produced by the convolution(s).
            kernel_size tuple(_size_3_t): Tuple of sizes of the convolutionaling kernels.
            stride tuple(_size_3_t): Tuple of strides of the convolutions.
            padding tuple(_size_3_t): Tuple of paddings added to all three sides of the
                input.
            padding_mode tuple(string): Tuple of padding modes for each convs.
                Options include `zeros`, `reflect`, `replicate` or `circular`.
            dilation tuple(_size_3_t): Tuple of spacings between kernel elements.
            groups tuple(_size_3_t): Tuple of numbers of blocked connections from input
                channels to output channels.
            bias tuple(bool): If `True`, adds a learnable bias to the output.
            reduction_method str: Options include `sum` and `cat`.
        """
        super().__init__()
        assert reduction_method in ('sum', 'cat')
        self.reduction_method = reduction_method
        conv_list = []
        for ind in range(len(kernel_size)):
            conv_param = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size[ind]}
            if stride is not None and stride[ind] is not None:
                conv_param['stride'] = stride[ind]
            if padding is not None and padding[ind] is not None:
                conv_param['padding'] = padding[ind]
            if dilation is not None and dilation[ind] is not None:
                conv_param['dilation'] = dilation[ind]
            if groups is not None and groups[ind] is not None:
                conv_param['groups'] = groups[ind]
            if bias is not None and bias[ind] is not None:
                conv_param['bias'] = bias[ind]
            if padding_mode is not None and padding_mode[ind] is not None:
                conv_param['padding_mode'] = padding_mode[ind]
            conv_list.append(nn.Conv3d(**conv_param))
        self.convs = nn.ModuleList(conv_list)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        output = []
        for ind in range(len(self.convs)):
            output.append(self.convs[ind](x))
        if self.reduction_method == 'sum':
            output = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduction_method == 'cat':
            output = torch.cat(output, dim=1)
        return output


def set_attributes(self, params: List[object]=None) ->None:
    """
    An utility function used in classes to set attributes from the input list of parameters.
    Args:
        params (list): list of parameters.
    """
    if params:
        for k, v in params.items():
            if k != 'self':
                setattr(self, k, v)


class Conv2plus1d(nn.Module):
    """
    Implementation of 2+1d Convolution by factorizing 3D Convolution into an 1D temporal
    Convolution and a 2D spatial Convolution with Normalization and Activation module
    in between:

    ::

                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)

    The 2+1d Convolution is used to build the R(2+1)D network.
    """

    def __init__(self, *, conv_t: nn.Module=None, norm: nn.Module=None, activation: nn.Module=None, conv_xy: nn.Module=None, conv_xy_first: bool=False) ->None:
        """
        Args:
            conv_t (torch.nn.modules): temporal convolution module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            conv_xy (torch.nn.modules): spatial convolution module.
            conv_xy_first (bool): If True, spatial convolution comes before temporal conv
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv_t is not None
        assert self.conv_xy is not None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
        x = self.norm(x) if self.norm else x
        x = self.activation(x) if self.activation else x
        x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
        return x


def _verify_feature_dim(feature_dims: List[int]):
    assert isinstance(feature_dims, list)
    assert all(x > 0 for x in feature_dims)


class ConcatFusion(nn.Module):
    """
    Concatenates all inputs by their last dimension. The resulting tensor last dim will be
    the sum of the last dimension of all input tensors.
    """

    def __init__(self, feature_dims: List[int]):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self._output_dim = sum(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) ->torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, sum(feature_dims)) where sum(feature_dims)
                is the sum of all input feature_dims.
        """
        return torch.cat(input_list, dim=-1)


class TemporalConcatFusion(nn.Module):
    """
    Concatenates all inputs by their temporal dimension which is assumed to be dim=1.
    """

    def __init__(self, feature_dims: List[int]):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self._output_dim = max(feature_dims)
        assert self._output_dim == min(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) ->torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim)

        Returns:
            Tensor of shape (batch_size, sum(seq_len), feature_dim) where sum(seq_len) is
                the sum of all input tensors.
        """
        return torch.cat(input_list, dim=1)


class ReduceFusion(nn.Module):
    """
    Generic fusion method which takes a callable which takes the list of input tensors
    and expects a single tensor to be used. This class can be used to implement fusion
    methods like "sum", "max" and "prod".
    """

    def __init__(self, feature_dims: List[int], reduce_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        _verify_feature_dim(feature_dims)
        self.reduce_fn = reduce_fn
        self._output_dim = max(feature_dims)
        assert self._output_dim == min(feature_dims)

    @property
    def output_dim(self):
        """
        Last dimension size of forward(..) tensor output.
        """
        return self._output_dim

    def forward(self, input_list: List[torch.Tensor]) ->torch.Tensor:
        """
        Args:
            input_list (List[torch.Tensor]): a list of tensors of shape
                (batch_size, seq_len, feature_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, feature_dim).
        """
        return self.reduce_fn(torch.stack(input_list))


class NonLocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper:
    Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    In Proceedings of the IEEE conference on CVPR, 2018.
    """

    def __init__(self, *, conv_theta: nn.Module, conv_phi: nn.Module, conv_g: nn.Module, conv_out: nn.Module, pool: Optional[nn.Module]=None, norm: Optional[nn.Module]=None, instantiation: str='dot_product') ->None:
        super().__init__()
        set_attributes(self, locals())
        assert None not in (conv_theta, conv_phi, conv_g, conv_out)
        assert instantiation in ('dot_product', 'softmax'), 'Unknown norm type {}'.format(instantiation)
        assert len({self.conv_theta.out_channels, self.conv_phi.out_channels, self.conv_g.out_channels, self.conv_out.in_channels}) == 1, "Nonlocal convolution's input/ output dimension mismatch."

    def forward(self, x) ->torch.Tensor:
        dim_inner = self.conv_theta.out_channels
        x_identity = x
        N, C, T, H, W = x.size()
        theta = self.conv_theta(x)
        if self.pool is not None:
            x = self.pool(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)
        theta = theta.view(N, dim_inner, -1)
        phi = phi.view(N, dim_inner, -1)
        g = g.view(N, dim_inner, -1)
        theta_phi = torch.einsum('nct,ncp->ntp', (theta, phi))
        if self.instantiation == 'softmax':
            theta_phi = theta_phi * dim_inner ** -0.5
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == 'dot_product':
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        theta_phi_g = torch.einsum('ntg,ncg->nct', (theta_phi, g))
        theta_phi_g = theta_phi_g.view(N, dim_inner, T, H, W)
        p = self.conv_out(theta_phi_g)
        if self.norm is not None:
            p = self.norm(p)
        return x_identity + p


class PositionalEncoding(nn.Module):
    """
    Applies a positional encoding to a tensor with shape (batch_size x seq_len x embed_dim).

    The positional encoding is computed as follows:
        PE(pos,2i) = sin(pos/10000^(2i/dmodel))
        PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

        where pos = position, pos in [0, seq_len)
        dmodel = data embedding dimension = embed_dim
        i = dimension index, i in [0, embed_dim)

    Reference: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
    Implementation Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embed_dim: int, seq_len: int=1024) ->None:
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim, dtype=torch.float)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        assert self.pe.size(1) >= x.size(1), 'Cannot apply position encoding of size ' + f'{self.pe.size()} when input has size {x.size()}'
        return x + self.pe[:, :x.size(1), :]


class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(self, embed_dim: int, patch_embed_shape: Tuple[int, int, int], sep_pos_embed: bool=False, has_cls: bool=True) ->None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert len(patch_embed_shape) == 3, 'Patch_embed_shape should be in the form of (T, H, W).'
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = tuple(patch_embed_shape)
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            self.cls_token = torch.tensor(0)
            num_patches = self.num_spatial_patch * self.num_temporal_patch
        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_spatial_patch, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patch, embed_dim))
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed_class = torch.tensor([])
            self.pos_embed = torch.tensor([])
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.pos_embed_spatial = torch.tensor([])
            self.pos_embed_temporal = torch.tensor([])
            self.pos_embed_class = torch.tensor([])

    @torch.jit.export
    def patch_embed_shape(self) ->Tuple[int, int, int]:
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(1, self.num_temporal_patch, 1) + torch.repeat_interleave(self.pos_embed_temporal, self.num_spatial_patch, dim=1)
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        return x


class ScriptableSpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(self, embed_dim: int, patch_embed_shape: Tuple[int, int, int], sep_pos_embed: bool=False, has_cls: bool=True) ->None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert len(patch_embed_shape) == 3, 'Patch_embed_shape should be in the form of (T, H, W).'
        assert not has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_spatial_patch, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.num_temporal_patch, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        assert self.sep_pos_embed
        pos_embed = self.pos_embed_spatial.repeat(1, self.num_temporal_patch, 1) + torch.repeat_interleave(self.pos_embed_temporal, self.num_spatial_patch, dim=1)
        x = x + pos_embed
        return x


class SqueezeAndExcitationLayer2D(nn.Module):
    """2D Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, in_planes: int, reduction_ratio: Optional[int]=16, reduced_planes: Optional[int]=None):
        """
        Args:
            in_planes (int): input channel dimension.
            reduction_ratio (int): factor by which in_planes should be reduced to
                get the output channel dimension.
            reduced_planes (int): Output channel dimension. Only one of reduction_ratio
                or reduced_planes should be defined.
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        assert bool(reduction_ratio) != bool(reduced_planes), 'Only of reduction_ratio or reduced_planes should be defined for SE layer'
        reduced_planes = in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        self.excitation = nn.Sequential(nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True), nn.ReLU(), nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (tensor): 2D image of format C * H * W
        """
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


def convert_to_one_hot(targets: torch.Tensor, num_class: int, label_smooth: float=0.0) ->torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    Args:
        targets (torch.Tensor): Index labels to be converted.
        num_class (int): Total number of classes.
        label_smooth (float): Label smooth value for non-target classes. Label smooth
            is disabled by default (0).
    """
    assert torch.max(targets).item() < num_class, 'Class Index must be less than number of classes'
    assert 0 <= label_smooth < 1.0, 'Label smooth value needs to be between 0 and 1.'
    non_target_value = label_smooth / num_class
    target_value = 1.0 - label_smooth + non_target_value
    one_hot_targets = torch.full((targets.shape[0], num_class), non_target_value, dtype=torch.long if label_smooth == 0.0 else None, device=targets.device)
    one_hot_targets.scatter_(1, targets.long().view(-1, 1), target_value)
    return one_hot_targets


class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Adapted from Classy Vision: ./classy_vision/losses/soft_target_cross_entropy_loss.py.
    This allows the targets for the cross entropy loss to be multi-label.
    """

    def __init__(self, ignore_index: int=-100, reduction: str='mean', normalize_targets: bool=True) ->None:
        """
        Args:
            ignore_index (int): sample should be ignored for loss if the class is this value.
            reduction (str): specifies reduction to apply to the output.
            normalize_targets (bool): whether the targets should be normalized to a sum of 1
                based on the total count of positive targets for a given sample.
        """
        super().__init__()
        set_attributes(self, locals())
        assert isinstance(self.normalize_targets, bool)
        if self.reduction not in ['mean', 'none']:
            raise NotImplementedError('reduction type "{}" not implemented'.format(self.reduction))
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        """
        Args:
            input (torch.Tensor): the shape of the tensor is N x C, where N is the number of
                samples and C is the number of classes. The tensor is raw input without
                softmax/sigmoid.
            target (torch.Tensor): the shape of the tensor is N x C or N. If the shape is N, we
                will convert the target to one hot vectors.
        """
        if target.ndim == 1:
            assert input.shape[0] == target.shape[0], 'SoftTargetCrossEntropyLoss requires input and target to have same batch size!'
            target = convert_to_one_hot(target.view(-1, 1), input.shape[1])
        assert input.shape == target.shape, f'SoftTargetCrossEntropyLoss requires input and target to be same shape: {input.shape} != {target.shape}'
        N, C = target.shape
        valid_mask = torch.ones((N, 1), dtype=torch.float)
        if 0 <= self.ignore_index <= C - 1:
            drop_idx = target[:, self.ignore_idx] > 0
            valid_mask[drop_idx] = 0
        valid_targets = target.float() * valid_mask
        if self.normalize_targets:
            valid_targets /= self.eps + valid_targets.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -valid_targets * F.log_softmax(input, -1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        if self.reduction == 'mean':
            loss = per_sample_loss.sum() / torch.sum(torch.sum(valid_mask, -1) > 0).clamp(min=1)
        elif self.reduction == 'none':
            loss = per_sample_loss
        return loss


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
        ceil (bool): If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class X3dBottleneckBlock(EfficientBlockBase):
    """
    Implements a X3D style residual block with optional squeeze-excite (SE)
    using efficient blocks.

                    Input +----------------------+
                    |                            |
                    v                            |
                    conv3d[0] (1x1x1)            |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    activation[0]                |
                    |                            |
                    v                            |
                    conv3d[1] (3x3x3 dw)         |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    Squeeze-Excite (optional)    |
                    |                            |
                    v                            |
                    activation[1]                |
                    |                            |
                    v                            |
                    conv3d[2] (1x1x1)            |
                    |                            |
                    v                            |
                    batchNorm (optional)         |
                    |                            |
                    v                            |
                    sum  <-----------------------+
                    |
                    v
                    activation[2]

    Args:
        in_channels (int): input channels for for 1x1x1 conv3d[0].
        mid_channels (int): channels for 3x3x3 dw conv3d[1].
        out_channels (int): output channels for 1x1x1 conv3d[2].
        spatial_stride (int): spatial stride for 3x3x3 dw conv3d[1].
        se_ratio (float): if > 0, apply SE to the 3x3x3 dw conv3d[1], with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        bias (tuple of bool): if bias[i] is true, use bias for conv3d[i].
        act_functions (tuple of str): act_functions[i] is the activation function after
            conv3d[i]. act_functions[i] should be a key in dict supported_act_functions
            (see activation_functions.py for more info about supported activations).
            Currently ReLU ('relu'), Swish ('swish'), Hardswish ('hswish'), Identity
            ('identity') are supported.
        use_bn (tuple of bool): if use_bn[i] is true, use batchnorm after conv3d[i].
        norm_eps (float): epsilon for batchnorm.
        norm_momentum (float): momentum for batchnorm.

    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, use_residual: bool=True, spatial_stride: int=1, se_ratio: float=0.0625, act_functions: Optional[Tuple[str]]=('relu', 'relu', 'relu'), bias: Optional[Tuple[bool]]=(False, False, False), use_bn: Optional[Tuple[bool]]=(True, True, True), norm_eps: float=1e-05, norm_momentum: float=0.1):
        super().__init__()
        self._use_residual = use_residual
        self._res_proj = None
        if self._use_residual:
            self._residual_add_func = nn.quantized.FloatFunctional()
            if spatial_stride != 1 or in_channels != out_channels:
                self._res_proj = Conv3dTemporalKernel1BnAct(in_channels, out_channels, bias=False, groups=1, spatial_kernel=1, spatial_stride=spatial_stride, spatial_padding=0, spatial_dilation=1, activation='identity', use_bn=True)
        layers = OrderedDict()
        assert act_functions[0] in supported_act_functions, f'{act_functions[0]} is not supported.'
        layers['conv_0'] = Conv3dPwBnAct(in_channels, mid_channels, bias=bias[0], activation=act_functions[0], use_bn=use_bn[0], norm_eps=norm_eps, norm_momentum=norm_momentum)
        self._spatial_stride = spatial_stride
        self._mid_channels = mid_channels
        assert act_functions[1] in supported_act_functions, f'{act_functions[1]} is not supported.'
        layers['conv_1'] = Conv3d3x3x3DwBnAct(mid_channels, spatial_stride=self._spatial_stride, bias=bias[1], activation='identity', use_bn=use_bn[1], norm_eps=norm_eps, norm_momentum=norm_momentum)
        if se_ratio > 0:
            layers['se'] = SqueezeExcitation(num_channels=mid_channels, num_channels_reduced=round_width(mid_channels, se_ratio), is_3d=True)
        layers['act_func_1'] = supported_act_functions[act_functions[1]]()
        self._out_channels = out_channels
        assert act_functions[2] in supported_act_functions, f'{act_functions[2]} is not supported.'
        layers['conv_2'] = Conv3dPwBnAct(mid_channels, out_channels, bias=bias[2], activation='identity', use_bn=use_bn[2], norm_eps=norm_eps, norm_momentum=norm_momentum)
        self.final_act = supported_act_functions[act_functions[2]]()
        self.layers = nn.Sequential(layers)
        self.convert_flag = False

    def forward(self, x):
        out = self.layers(x)
        if self._use_residual:
            if self._res_proj is not None:
                x = self._res_proj(x)
            out = self._residual_add_func.add(x, out)
        out = self.final_act(out)
        return out

    def convert(self, input_blob_size, *args, convert_for_quantize=False, native_conv3d_op_qnnpack=False, **kwargs):
        assert self.convert_flag is False, 'X3dBottleneckBlock: already converted, cannot be converted twice'
        batch_size = input_blob_size[0]
        THW_size = tuple(input_blob_size[2:])
        if self._res_proj is not None:
            self._res_proj.convert(input_blob_size, convert_for_quantize=convert_for_quantize, native_conv3d_op_qnnpack=native_conv3d_op_qnnpack)
        self.layers.conv_0.convert(input_blob_size, convert_for_quantize=convert_for_quantize, native_conv3d_op_qnnpack=native_conv3d_op_qnnpack)
        input_blob_size = (batch_size, self._mid_channels) + THW_size
        self.layers.conv_1.convert(input_blob_size, convert_for_quantize=convert_for_quantize, native_conv3d_op_qnnpack=native_conv3d_op_qnnpack)
        THW_size = THW_size[0], THW_size[1] // self._spatial_stride, THW_size[2] // self._spatial_stride
        input_blob_size = (batch_size, self._mid_channels) + THW_size
        if hasattr(self.layers, 'se'):
            self.layers.se.convert(input_blob_size)
        self.layers.act_func_1.convert(input_blob_size)
        self.layers.conv_2.convert(input_blob_size, convert_for_quantize=convert_for_quantize, native_conv3d_op_qnnpack=native_conv3d_op_qnnpack)
        input_blob_size = (batch_size, self._out_channels) + THW_size
        self.final_act.convert(input_blob_size)
        self.convert_flag = True


class EfficientX3d(nn.Module):
    """
    This class implements an X3D network for classification with efficient blocks.
    Args:
        num_classes (int): Number of classes in classification.
        dropout (float): Dropout rate used for training the network.
        expansion (str): Expansion for X3D. Possible options: 'XS', 'S', 'M', 'L'.
        head_act (str): The activation function to be applied in head, should be a key
            in dict supported_act_functions (see activation_functions.py for more info
            about supported activations).
        enable_head (bool): Whether X3D model provides head.
    """

    def __init__(self, num_classes: int=400, dropout: float=0.5, expansion: str='XS', head_act: str='identity', enable_head: bool=True):
        super().__init__()
        assert expansion in ('XS', 'S', 'M', 'L'), f'Expansion {expansion} not supported.'
        s1 = OrderedDict()
        s1['pathway0_stem_conv_xy'] = Conv3dTemporalKernel1BnAct(3, 24, bias=False, groups=1, spatial_kernel=3, spatial_stride=2, spatial_padding=1, activation='identity', use_bn=False)
        s1['pathway0_stem_conv'] = Conv3d5x1x1BnAct(24, 24, bias=False, groups=24, use_bn=True)
        self.s1 = nn.Sequential(s1)
        s2 = OrderedDict()
        depth_s2 = 5 if expansion == 'L' else 3
        for i_block in range(depth_s2):
            cur_block = X3dBottleneckBlock(in_channels=24, mid_channels=54, out_channels=24, use_residual=True, spatial_stride=2 if i_block == 0 else 1, se_ratio=0.0625 if i_block % 2 == 0 else 0, act_functions=('relu', 'swish', 'relu'), use_bn=(True, True, True))
            s2[f'pathway0_res{i_block}'] = cur_block
        self.s2 = nn.Sequential(s2)
        s3 = OrderedDict()
        depth_s3 = 10 if expansion == 'L' else 5
        for i_block in range(depth_s3):
            cur_block = X3dBottleneckBlock(in_channels=24 if i_block == 0 else 48, mid_channels=108, out_channels=48, use_residual=True, spatial_stride=2 if i_block == 0 else 1, se_ratio=0.0625 if i_block % 2 == 0 else 0, act_functions=('relu', 'swish', 'relu'), use_bn=(True, True, True))
            s3[f'pathway0_res{i_block}'] = cur_block
        self.s3 = nn.Sequential(s3)
        s4 = OrderedDict()
        depth_s4 = 25 if expansion == 'L' else 11
        for i_block in range(depth_s4):
            cur_block = X3dBottleneckBlock(in_channels=48 if i_block == 0 else 96, mid_channels=216, out_channels=96, use_residual=True, spatial_stride=2 if i_block == 0 else 1, se_ratio=0.0625 if i_block % 2 == 0 else 0, act_functions=('relu', 'swish', 'relu'), use_bn=(True, True, True))
            s4[f'pathway0_res{i_block}'] = cur_block
        self.s4 = nn.Sequential(s4)
        s5 = OrderedDict()
        depth_s5 = 15 if expansion == 'L' else 7
        for i_block in range(depth_s5):
            cur_block = X3dBottleneckBlock(in_channels=96 if i_block == 0 else 192, mid_channels=432, out_channels=192, use_residual=True, spatial_stride=2 if i_block == 0 else 1, se_ratio=0.0625 if i_block % 2 == 0 else 0, act_functions=('relu', 'swish', 'relu'), use_bn=(True, True, True))
            s5[f'pathway0_res{i_block}'] = cur_block
        self.s5 = nn.Sequential(s5)
        self.enable_head = enable_head
        if enable_head:
            head = OrderedDict()
            head['conv_5'] = Conv3dPwBnAct(in_channels=192, out_channels=432, bias=False, use_bn=True)
            head['avg_pool'] = AdaptiveAvgPool3dOutSize1()
            head['lin_5'] = Conv3dPwBnAct(in_channels=432, out_channels=2048, bias=False, use_bn=False)
            self.head = nn.Sequential(head)
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            self.projection = FullyConnected(2048, num_classes, bias=True)
            assert head_act in supported_act_functions, f'{head_act} is not supported.'
            self.act = supported_act_functions[head_act]()

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_head:
            x = self.head(x)
            x = x.permute((0, 2, 3, 4, 1))
            if hasattr(self, 'dropout'):
                x = self.dropout(x)
            x = self.projection(x)
            if not self.training:
                x = self.act(x)
                x = x.mean([1, 2, 3])
            x = x.view(x.shape[0], -1)
        return x


class FuseAudioToFastSlow(nn.Module):
    """
    Given a list of two tensors from Slow pathway and Fast pathway, fusion information
    from the Fast pathway to the Slow on through a convolution followed by a
    concatenation, then return the fused list of tensors from Slow and Fast pathway in
    order.
    """

    def __init__(self, block_fast_to_slow: nn.Module, block_audio_to_fastslow: nn.Module) ->None:
        """
        Args:
            conv_fast_to_slow (nn.module): convolution to perform fusion.
            norm (nn.module): normalization module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        x_a = x[2]
        fuse = self.block_fast_to_slow(x_f)
        average_a = torch.mean(x_a, dim=-1, keepdim=True)
        fuse_a = self.block_audio_to_fastslow(average_a)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        None
        return [fuse_a + x_s_fuse, x_f, x_a]


def _init_resnet_weights(model: nn.Module, fc_init_std: float=0.01) ->None:
    """
    Performs ResNet style weight initialization. That is, recursively initialize the
    given model in the following way for each type:
        Conv - Follow the initialization of kaiming_normal:
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
        BatchNorm - Set weight and bias of last BatchNorm at every residual bottleneck
            to 0.
        Linear - Set weight to 0 mean Gaussian with std deviation fc_init_std and bias
            to 0.
    Args:
        model (nn.Module): Model to be initialized.
        fc_init_std (float): the expected standard deviation for fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            if m.weight is not None:
                if hasattr(m, 'block_final_bn') and m.block_final_bn:
                    m.weight.data.fill_(0.0)
                else:
                    m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            if hasattr(m, 'xavier_init') and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def _init_vit_weights(model: nn.Module, trunc_normal_std: float=0.02) ->None:
    """
    Weight initialization for vision transformers.

    Args:
        model (nn.Module): Model to be initialized.
        trunc_normal_std (float): the expected standard deviation for fully-connected
            layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)


def init_net_weights(model: nn.Module, init_std: float=0.01, style: str='resnet') ->None:
    """
    Performs weight initialization. Options include ResNet style weight initialization
    and transformer style weight initialization.

    Args:
        model (nn.Module): Model to be initialized.
        init_std (float): The expected standard deviation for initialization.
        style (str): Options include "resnet" and "vit".
    """
    assert style in ['resnet', 'vit']
    if style == 'resnet':
        return _init_resnet_weights(model, init_std)
    elif style == 'vit':
        return _init_vit_weights(model, init_std)
    else:
        raise NotImplementedError


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
    Details can be found in:
    https://arxiv.org/pdf/2006.07733.pdf
    """

    def __init__(self, mmt: float, backbone: nn.Module, predictor: nn.Module, backbone_mmt: nn.Module, projector: Optional[nn.Module]=None, projector_mmt: Optional[nn.Module]=None) ->None:
        """
        Args:
            backbone (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): An mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            backbone_mmt (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector_mmt (nn.Module): Am mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            predictor (nn.Module): predictor MLP of BYOL of similar structure as the
                projector MLP.
            mmt (float): momentum update ratio for the momentum backbone.
        """
        super().__init__()
        self.mmt: float = mmt
        if projector is not None:
            backbone = nn.Sequential(backbone, projector)
        init_net_weights(backbone)
        self.backbone = backbone
        if projector_mmt is not None:
            backbone_mmt = nn.Sequential(backbone_mmt, projector_mmt)
        init_net_weights(backbone_mmt)
        self.backbone_mmt = backbone_mmt
        for p in self.backbone_mmt.parameters():
            p.requires_grad = False
        init_net_weights(predictor)
        self.predictor = predictor
        self._copy_weights_to_backbone_mmt()

    def _copy_weights_to_backbone_mmt(self) ->None:
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            p.data.copy_(dist[name].data)

    @torch.no_grad()
    def momentum_update_backbone(self) ->None:
        """
        Momentum update on the backbone.
        """
        m = self.mmt
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            p.data = dist[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def forward_backbone_mmt(self, x: torch.Tensor) ->torch.Tensor:
        """
        Forward momentum backbone.
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        with torch.no_grad():
            proj = self.backbone_mmt(x)
        return F.normalize(proj, dim=1)

    def forward(self, x: torch.Tensor) ->Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        if not self.training:
            x = self.backbone(x)
            x = F.normalize(x, dim=1)
            return x
        proj = self.backbone(x)
        pred = self.predictor(proj)
        pred = F.normalize(pred, dim=1)
        out_proj = F.normalize(proj, dim=1)
        return out_proj, pred


class SequencePool(nn.Module):
    """
    Sequence pool produces a single embedding from a sequence of embeddings. Currently
    it supports "mean" and "cls".

    """

    def __init__(self, mode: str) ->None:
        """
        Args:
            mode (str): Optionals include "cls" and "mean". If set to "cls", it assumes
                the first element in the input is the cls token and returns it. If set
                to "mean", it returns the mean of the entire sequence.
        """
        super().__init__()
        assert mode in ['cls', 'mean'], 'Unsupported mode for SequencePool.'
        self.mode = mode

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.mode == 'cls':
            x = x[:, 0]
        elif self.mode == 'mean':
            x = x.mean(1)
        else:
            raise NotImplementedError
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNet basic head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_basic_head`.
    """

    def __init__(self, pool: nn.Module=None, dropout: nn.Module=None, proj: nn.Module=None, activation: nn.Module=None, output_pool: nn.Module=None) ->None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.proj is not None:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        if self.activation is not None:
            x = self.activation(x)
        if self.output_pool is not None:
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x


class ResNetRoIHead(nn.Module):
    """
    ResNet RoI head. This layer performs an optional pooling operation
    followed by an RoI projection, an optional 2D spatial pool, an optional dropout,
    a fully-connected projection, an activation layer
    and a global spatiotemporal averaging.
                                        Pool3d
                                           ↓
                                       RoI Align
                                           ↓
                                        Pool2d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_roi_pooling_head`.
    """

    def __init__(self, pool: nn.Module=None, pool_spatial: nn.Module=None, roi_layer: nn.Module=None, dropout: nn.Module=None, proj: nn.Module=None, activation: nn.Module=None, output_pool: nn.Module=None) ->None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            pool_spatial (torch.nn.modules): pooling module.
            roi_spatial (torch.nn.modules): RoI (Ex: Align, pool) module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor, bboxes: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.tensor): input tensor
            bboxes (torch.tensor): Accociated bounding boxes.
                The format is N*5 (Index, X_1,Y_1,X_2,Y_2) if using RoIAlign
                and N*6 (Index, x_ctr, y_ctr, width, height, angle_degrees) if
                using RoIAlignRotated.
        """
        if self.pool is not None:
            x = self.pool(x)
        if self.roi_layer is not None:
            temporal_dim = x.shape[-3]
            if temporal_dim != 1:
                raise Exception('Temporal dimension should be 1. Consider modifying the pool layer.')
            x = torch.squeeze(x, -3)
            x = self.roi_layer(x, bboxes)
            if self.pool_spatial is not None:
                x = self.pool_spatial(x)
            x = x.unsqueeze(-3)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.proj is not None:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        if self.activation is not None:
            x = self.activation(x)
        if self.output_pool is not None:
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x


class VisionTransformerBasicHead(nn.Module):
    """
    Vision transformer basic head.

    ::

                                      SequencePool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation


    The builder can be found in `create_vit_basic_head`.
    """

    def __init__(self, sequence_pool: nn.Module=None, dropout: nn.Module=None, proj: nn.Module=None, activation: nn.Module=None) ->None:
        """
        Args:
            sequence_pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.proj is not None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.sequence_pool is not None:
            x = self.sequence_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MaskedTemporalPooling(torch.nn.Module):
    """
    Applies temporal pooling operations on masked inputs. For each pooling operation
    all masked values are ignored.
    """

    def __init__(self, method: str):
        """
        method (str): the method of pooling to use. Options:
            'max': reduces temporal dimension to each valid max value.
            'avg': averages valid values in the temporal dimension.
            'sum': sums valid values in the temporal dimension.
            Note if all batch row elements are invalid, the temporal dimension is
            pooled to 0 values.
        """
        super().__init__()
        assert method in ('max', 'avg', 'sum')
        self._method = method

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        assert x.dim() == 3, 'Requires x shape (batch_size x seq_len x feature_dim)'
        b, t = x.shape[0], x.shape[1]
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool)
        if self._method == 'max':
            x[~mask, :] = float('-inf')
            invalid_first_dim = ~mask.view(b, -1).any(dim=-1)
            x[invalid_first_dim, :] = 0
            x = torch.max(x, dim=1)[0]
        elif self._method == 'avg':
            x = x * mask.unsqueeze(-1).float()
            mask = mask.view(b, t, -1).any(dim=-1)
            valid_lengths = mask.float().sum(dim=-1).int()
            x = x.sum(dim=1)
            x = x.div(valid_lengths.clamp(min=1).unsqueeze(-1).expand(x.size()).float())
        elif self._method == 'sum':
            x = x * mask.unsqueeze(-1).float()
            x = x.sum(dim=1)
        else:
            raise NotImplementedError(f"{self._method} not available options are: 'max', 'avg', 'sum'")
        return x


class TransposeMultiheadAttention(nn.Module):
    """
    Wrapper for nn.MultiheadAttention which first transposes the input tensor
    from (batch_size, seq_len, feature_dim) to (seq_length, batch_size, feature_dim),
    then applies the attention and transposes the attention outputs back to the input
    shape.
    """

    def __init__(self, feature_dim: int, num_heads: int=1):
        """
        Args:
            feature_dim (int): attention embedding dimension
            num_heads (int): number of attention heads
        """
        super().__init__()
        self._attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self._attention_weights = None

    @property
    def attention_weights(self) ->Optional[torch.Tensor]:
        """
        Contains attention weights from last forward call.
        """
        return self._attention_weights

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, seq_len, feature_dim)
        """
        assert x.dim() == 3, 'Requires x shape (batch_size x seq_len x feature_dim)'
        if mask is not None:
            mask[:, 0] = True
            mask = ~mask
        x = x.transpose(0, 1)
        attn_output, self._attention_weights = self._attention(x, x, x, key_padding_mask=mask)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class LearnMaskedDefault(nn.Module):
    """
    Learns default values to fill invalid entries within input tensors. The
    invalid entries are represented by a mask which is passed into forward alongside
    the input tensor. Note the default value is only used if all entries in the batch row are
    invalid rather than just a portion of invalid entries within each batch row.
    """

    def __init__(self, feature_dim: int, init_method: str='gaussian', freeze: bool=False):
        """
        Args:
            feature_dim (int): the size of the default value parameter, this must match the
                input tensor size.
            init_method (str): the initial default value parameter. Options:
                'guassian'
                'zeros'
            freeze (bool): If True, the learned default parameter weights are frozen.
        """
        super().__init__()
        if init_method == 'zeros':
            self._learned_defaults = nn.Parameter(torch.zeros(feature_dim), requires_grad=not freeze)
        elif init_method == 'gaussian':
            self._learned_defaults = nn.Parameter(torch.Tensor(feature_dim), requires_grad=not freeze)
            nn.init.normal_(self._learned_defaults)
        else:
            raise NotImplementedError(f"{init_method} not available. Options are: 'zeros' or 'gaussian'")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): tensor of shape (batch_size, feature_dim).
            mask (torch.Tensor): bool tensor of shape (batch_size, seq_len) If all elements
                in the batch dimension are False the learned default parameter is used for
                that batch element.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        mask = mask.view(mask.shape[0], -1).any(dim=-1)
        for i in range(1, x.dim()):
            mask = mask.unsqueeze(i)
        x = x * mask.float() + self._learned_defaults * (1 - mask.float())
        return x


class LSTM(nn.Module):
    """
    Wrapper for torch.nn.LSTM that handles masked inputs.
    """

    def __init__(self, dim_in: int, hidden_dim: int, dropout: float=0.0, bidirectional: bool=False):
        """
        Args:
          dim_in (int): input feature dimension
          hidden_dim (int): hidden dimesion of lstm layer
          dropout (float): dropout rate - 0.0 if no dropout
          bidirectional (bool): bidirectional or forward only
        """
        super().__init__()
        self.lstm = nn.LSTM(dim_in, hidden_dim, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.lstm.flatten_parameters()
        self.output_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.bidirectional = bidirectional

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            data (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, output_dim) - outoput_dim is determined by
                hidden_dim and whether bidirectional or not
        """
        assert data.dim() == 3
        b, t = data.shape[0], data.shape[1]
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool)
        lengths = mask.sum(axis=1)
        x_packed = pack_padded_sequence(data, lengths.clamp(1, data.size(1)), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x_packed)
        if self.bidirectional:
            out = torch.cat([h[0, :, :], h[1, :, :]], dim=-1)
        else:
            out = h[-1, :, :]
        return out


class TransposeTransformerEncoder(nn.Module):
    """
    Wrapper for torch.nn.TransformerEncoder that handles masked inputs.
    """

    def __init__(self, dim_in: int, num_heads: int=1, num_layers: int=1):
        """
        Args:
          dim_in (int): input feature dimension
          num_heads (int): number of heads in the nn.MultiHeadAttention layers
          num_layers (int): the number of sub-encoder-layers in the encoder
        """
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_in, num_heads), num_layers)

    def forward(self, data: torch.Tensor, mask: Optional[torch.Tensor]=None) ->torch.Tensor:
        """
        Args:
            data (torch.Tensor): tensor with shape (batch_size, seq_len, feature_dim)
            mask (torch.Tensor): bool tensor with shape (batch_size, seq_len).
                Sequence elements that are False are invalid.

        Returns:
            Tensor with shape (batch_size, feature_dim)
        """
        if mask is not None:
            mask[:, 0] = True
            mask = ~mask
        out = self.encoder(src=data.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        return out[:, 0, :]


class MaskedSequential(nn.Sequential):
    """
    A sequential container that overrides forward to take a mask as well as the usual
    input tensor. This mask is only applied to modules in _MASK_MODULES (which take
    the mask argument).
    """
    _MASK_MODULES = [MaskedTemporalPooling, LearnMaskedDefault, TransposeMultiheadAttention, LSTM, TransposeTransformerEncoder]

    def forward(self, input: torch.Tensor, mask: torch.Tensor) ->torch.Tensor:
        for module in self:
            if any(isinstance(module, mask_type) for mask_type in self._MASK_MODULES):
                input = module(input, mask=mask)
            else:
                input = module(input)
        return input


class MaskedMultiPathWay(nn.Module):
    """
    Masked multi-pathway is composed of a list of stream nn.Modules followed by a
    fusion nn.Module that reduces these streams. Each stream module takes a mask
    and input tensor.

    ::

                            Pathway 1  ... Pathway N
                                ↓              ↓
                             Block 1        Block N
                                ↓⭠ --Fusion----↓
    """

    def __init__(self, *, multipathway_blocks: nn.ModuleList, multipathway_fusion: Optional[nn.Module]) ->None:
        """
        Args:
            multipathway_blocks (nn.module_list): list of models from all pathways.
            multipathway_fusion (nn.module): fusion model.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x_and_mask: List[Tuple[torch.Tensor, torch.Tensor]]) ->torch.Tensor:
        out = []
        for pathway_idx in range(len(self.multipathway_blocks)):
            out.append(self.multipathway_blocks[pathway_idx](*x_and_mask[pathway_idx]))
        if self.multipathway_fusion is not None:
            x = self.multipathway_fusion(out)
        return x


class MemoryBank(nn.Module):
    """
    Performs Non-Parametric Instance Discrimination for self supervised learning on
    video. A memory bank is built to keep and update the historical feature embedding
    and use them for contrastive learning.

    The original paper is:
    Unsupervised Feature Learning via Non-Parametric Instance Discrimination
    https://arxiv.org/pdf/1805.01978.pdf

    More details can be found from the memory bank part in the following paper:
    Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/pdf/1911.05722.pdf
    """

    def __init__(self, backbone: nn.Module, mlp: Optional[nn.Module]=None, neg_size: int=4096, temperature: float=0.07, bank_size: int=1280000, dim: int=2048, mmt: float=0.999) ->None:
        """
        Args:
            backbone (nn.Module): backbone used to forward the input.
            mlp (nn.Module): multi-layer perception used in memory bank instance
                discrimination model.
            neg_size (int): size of negative samples per instance.
            temperature (float): temperature to use for contrastive learning.
            bank_size (int): size of the memory bank, expected to be the same size as
                the training set.
            dim (int): dimension of the channel.
            mmt (float): momentum to use.
        """
        super().__init__()
        set_attributes(self, locals())
        self._init_mem_bank(bank_size, dim)

    def _init_mem_bank(self, bank_size: int, dim: int) ->None:
        """
        Given the memory bank size and the channel dimension, initialize the memory
            bank.
        Args:
            bank_size (int): size of the memory bank, expected to be the same size as
                 the training set.
            dim (int): dimension of the channel.
        """
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer('memory', torch.rand(bank_size, dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x: torch.Tensor, x_ind: torch.Tensor) ->torch.Tensor:
        """
        Perform contrastive learning with random sampled negative instance from the
            memory bank. During training, update the memory bank with latest feature
            embedding.
        Args:
            x (torch.tensor): a batch of image with augmentation. The input tensor
                shape should able to be feed into the backbone.
            x_ind (torch.tensor): the index of the image x from the dataset. Expected
                shape is B.
        """
        batch_size = x.shape[0]
        x = self.backbone(x)
        if self.mlp is not None:
            x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        idx = torch.randint(0, self.bank_size, size=(batch_size, self.neg_size + 1))
        idx.select(1, 0).copy_(x_ind.data)
        weight = torch.index_select(self.memory, 0, idx.view(-1)).detach()
        weight = weight.view(batch_size, self.neg_size + 1, self.dim)
        out = torch.einsum('bkc,bc->bk', weight, x)
        out = torch.div(out, self.temperature)
        gt = torch.zeros((batch_size,), device=x.device, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(out, gt)
        if self.training:
            with torch.no_grad():
                pos = torch.index_select(self.memory, 0, x_ind.view(-1))
                pos.mul_(self.mmt)
                pos.add_(torch.mul(x, 1 - self.mmt))
                norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated = pos.div(norm)
                self.memory.index_copy_(0, x_ind, updated)
        return loss


class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.

    ::

                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓

    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(self, *, blocks: nn.ModuleList) ->None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        init_net_weights(self)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x


class DetectionBBoxNetwork(nn.Module):
    """
    A general purpose model that handles bounding boxes as part of input.
    """

    def __init__(self, model: nn.Module, detection_head: nn.Module):
        """
        Args:
            model (nn.Module): a model that preceeds the head. Ex: stem + stages.
            detection_head (nn.Module): a network head. that can take in input bounding boxes
                and the outputs from the model.
        """
        super().__init__()
        self.model = model
        self.detection_head = detection_head

    def forward(self, x: torch.Tensor, bboxes: torch.Tensor):
        """
        Args:
            x (torch.tensor): input tensor
            bboxes (torch.tensor): accociated bounding boxes.
                The format is N*5 (Index, X_1,Y_1,X_2,Y_2) if using RoIAlign
                and N*6 (Index, x_ctr, y_ctr, width, height, angle_degrees) if
                using RoIAlignRotated.
        """
        features = self.model(x)
        out = self.detection_head(features, bboxes)
        return out.view(out.shape[0], -1)


class MultiPathWayWithFuse(nn.Module):
    """
    Build multi-pathway block with fusion for video recognition, each of the pathway
    contains its own Blocks and Fusion layers across different pathways.

    ::

                            Pathway 1  ... Pathway N
                                ↓              ↓
                             Block 1        Block N
                                ↓⭠ --Fusion----↓
    """

    def __init__(self, *, multipathway_blocks: nn.ModuleList, multipathway_fusion: Optional[nn.Module], inplace: Optional[bool]=True) ->None:
        """
        Args:
            multipathway_blocks (nn.module_list): list of models from all pathways.
            multipathway_fusion (nn.module): fusion model.
            inplace (bool): If inplace, directly update the input list without making
                a copy.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x: List[torch.Tensor]) ->torch.Tensor:
        assert isinstance(x, list), 'input for MultiPathWayWithFuse needs to be a list of tensors'
        if self.inplace:
            x_out = x
        else:
            x_out = [None] * len(x)
        for pathway_idx in range(len(self.multipathway_blocks)):
            if self.multipathway_blocks[pathway_idx] is not None:
                x_out[pathway_idx] = self.multipathway_blocks[pathway_idx](x[pathway_idx])
        if self.multipathway_fusion is not None:
            x_out = self.multipathway_fusion(x_out)
        return x_out


class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The builder can be found in `create_res_block`.
    """

    def __init__(self, branch1_conv: nn.Module=None, branch1_norm: nn.Module=None, branch2: nn.Module=None, activation: nn.Module=None, branch_fusion: Callable=None) ->nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
            branch_fusion: (Callable): A callable or layer that combines branch1
                and branch2.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) ->torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            x = self.branch_fusion(shortcut, self.branch2(x))
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableBottleneckBlock(nn.Module):
    """
    Separable Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order. Requires a tuple of models to be
    provided to conv_b, norm_b, act_b to perform Convolution, Normalization, and
    Activations in parallel Separably.

    ::


                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                 Conv3d(s) (conv_b), ...
                                         ↓ (↓)
                              Normalization(s) (norm_b), ...
                                         ↓ (↓)
                                 Activation(s) (act_b), ...
                                         ↓ (↓)
                                  Reduce (sum or cat)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)
    """

    def __init__(self, *, conv_a: nn.Module, norm_a: nn.Module, act_a: nn.Module, conv_b: nn.ModuleList, norm_b: nn.ModuleList, act_b: nn.ModuleList, conv_c: nn.Module, norm_c: nn.Module, reduce_method: str='sum') ->None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules_list): convolutional module(s).
            norm_b (torch.nn.modules_list): normalization module(s).
            act_b (torch.nn.modules_list): activation module(s).
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
            reduce_method (str): if multiple conv_b is used, reduce the output with
                `sum`, or `cat`.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_b, self.conv_c)), f'{self.conv_a}, {self.conv_b}, {self.conv_c} has None'
        assert reduce_method in ['sum', 'cat']
        if self.norm_c is not None:
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.conv_a is not None:
            x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)
        output = []
        for ind in range(len(self.conv_b)):
            x_ = self.conv_b[ind](x)
            if self.norm_b[ind] is not None:
                x_ = self.norm_b[ind](x_)
            if self.act_b[ind] is not None:
                x_ = self.act_b[ind](x_)
            output.append(x_)
        if self.reduce_method == 'sum':
            x = torch.stack(output, dim=0).sum(dim=0, keepdim=False)
        elif self.reduce_method == 'cat':
            x = torch.cat(output, dim=1)
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

    ::


                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    The builder can be found in `create_bottleneck_block`.
    """

    def __init__(self, *, conv_a: nn.Module=None, norm_a: nn.Module=None, act_a: nn.Module=None, conv_b: nn.Module=None, norm_b: nn.Module=None, act_b: nn.Module=None, conv_c: nn.Module=None, norm_c: nn.Module=None) ->None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules): convolutional module.
            norm_b (torch.nn.modules): normalization module.
            act_b (torch.nn.modules): activation module.
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b, self.conv_c))
        if self.norm_c is not None:
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        if self.act_b is not None:
            x = self.act_b(x)
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class ResStage(nn.Module):
    """
    ResStage composes sequential blocks that make up a ResNet. These blocks could be,
    for example, Residual blocks, Non-Local layers, or Squeeze-Excitation layers.

    ::


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    The builder can be found in `create_res_stage`.
    """

    def __init__(self, res_blocks: nn.ModuleList) ->nn.Module:
        """
        Args:
            res_blocks (torch.nn.module_list): ResBlock module(s).
        """
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for _, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        return x


class SimCLR(nn.Module):
    """
    Skeletal NN.Module for the SimCLR model that supports
    arbitrary bacbone and projector models.
    """

    def __init__(self, backbone: nn.Module, projector: Optional[nn.Module]=None) ->None:
        """
        Args:
            backbone (nn.Module): backbone for simclr, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): An mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
        """
        super().__init__()
        if projector is not None:
            backbone = nn.Sequential(backbone, projector)
        init_net_weights(backbone)
        self.backbone = backbone

    def forward(self, x_list: Union[torch.Tensor, List[torch.Tensor]]) ->Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x_list (list(tensor) or tensor): Expects a list of 2 tensors
                for trainin phase and single tensor for the train and val
                phases. Here all tensors are expected to be of the shape,
                N x C x T x H x W.
        """
        if not self.training:
            assert isinstance(x_list, torch.Tensor), 'Expected tensor for test/val phase in SimCLR'
            if self.backbone is not None:
                x_list = self.backbone(x_list)
            x_list = F.normalize(x_list, p=2, dim=1)
            return x_list
        assert isinstance(x_list, list) and len(x_list) == 2, f'Invalid list input to SimCLR. Expected len 2 but received {len(x_list)}'
        for i, x in enumerate(x_list):
            if self.backbone is not None:
                x = self.backbone(x)
            x = F.normalize(x, p=2, dim=1)
            x_list[i] = x
        return x_list


class PoolConcatPathway(nn.Module):
    """
    Given a list of tensors, perform optional spatio-temporal pool and concatenate the
        tensors along the channel dimension.
    """

    def __init__(self, retain_list: bool=False, pool: Optional[nn.ModuleList]=None, dim: int=1) ->None:
        """
        Args:
            retain_list (bool): if True, return the concatenated tensor in a list.
            pool (nn.module_list): if not None, list of pooling models for different
                pathway before performing concatenation.
            dim (int): dimension to performance concatenation.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x: List[torch.Tensor]) ->torch.Tensor:
        if self.pool is not None:
            assert len(x) == len(self.pool)
        output = []
        for ind in range(len(x)):
            if x[ind] is not None:
                if self.pool is not None and self.pool[ind] is not None:
                    x[ind] = self.pool[ind](x[ind])
                output.append(x[ind])
        if self.retain_list:
            return [torch.cat(output, 1)]
        else:
            return torch.cat(output, 1)


class FuseFastToSlow(nn.Module):
    """
    Given a list of two tensors from Slow pathway and Fast pathway, fusion information
    from the Fast pathway to the Slow on through a convolution followed by a
    concatenation, then return the fused list of tensors from Slow and Fast pathway in
    order.
    """

    def __init__(self, conv_fast_to_slow: nn.Module, norm: Optional[nn.Module]=None, activation: Optional[nn.Module]=None) ->None:
        """
        Args:
            conv_fast_to_slow (nn.module): convolution to perform fusion.
            norm (nn.module): normalization module.
            activation (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_fast_to_slow(x_f)
        if self.norm is not None:
            fuse = self.norm(fuse)
        if self.activation is not None:
            fuse = self.activation(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class ResNetBasicStem(nn.Module):
    """
    ResNet basic 3D stem module. Performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.

    ::

                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    The builder can be found in `create_res_basic_stem`.
    """

    def __init__(self, *, conv: nn.Module=None, norm: nn.Module=None, activation: nn.Module=None, pool: nn.Module=None) ->None:
        """
        Args:
            conv (torch.nn.modules): convolutional module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv is not None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class PatchEmbed(nn.Module):
    """
    Transformer basic patch embedding module. Performs patchifying input, flatten and
    and transpose.

    ::

                                       PatchModel
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    The builder can be found in `create_patch_embed`.

    """

    def __init__(self, *, patch_model: nn.Module=None) ->None:
        super().__init__()
        set_attributes(self, locals())
        assert self.patch_model is not None

    def forward(self, x) ->torch.Tensor:
        x = self.patch_model(x)
        return x.flatten(2).transpose(1, 2)


class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik,
    Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227

    ::

                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head


    The builder can be found in `create_mvit`.
    """

    def __init__(self, *, patch_embed: Optional[nn.Module], cls_positional_encoding: nn.Module, pos_drop: Optional[nn.Module], blocks: nn.ModuleList, norm_embed: Optional[nn.Module], head: Optional[nn.Module]) ->None:
        """
        Args:
            patch_embed (nn.Module): Patch embed module.
            cls_positional_encoding (nn.Module): Positional encoding module.
            pos_drop (Optional[nn.Module]): Dropout module after patch embed.
            blocks (nn.ModuleList): Stack of multi-scale transformer blocks.
            norm_layer (nn.Module): Normalization layer before head.
            head (Optional[nn.Module]): Head module.
        """
        super().__init__()
        assert hasattr(cls_positional_encoding, 'patch_embed_shape'), 'cls_positional_encoding should have method patch_embed_shape.'
        self.patch_embed = patch_embed or torch.nn.Identity()
        self.cls_positional_encoding = cls_positional_encoding
        self.pos_drop = pos_drop or torch.nn.Identity()
        self.blocks = blocks
        self.norm_embed = norm_embed or torch.nn.Identity()
        self.head = head or torch.nn.Identity()
        init_net_weights(self, init_std=0.02, style='vit')

    def _get_bn_w_b(self, bn, repeat=1):
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)).repeat(repeat))
        b_bn = (bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))).repeat(repeat)
        return w_bn, b_bn

    def fuse_norm_before_linear(self, bn, linear):
        if bn is None:
            return linear
        w_bn, b_bn = self._get_bn_w_b(bn)
        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(linear.weight, w_bn)
        fused_linear.bias.data[:] = torch.matmul(linear.weight, b_bn) + linear.bias if linear.bias is not None else torch.matmul(linear.weight, b_bn)
        return fused_linear

    def fuse_norm_after_linear(self, linear, bn):
        if bn is None:
            return linear
        assert linear.in_features % bn.bias.shape[0] == 0
        num_heads = linear.in_features // bn.bias.shape[0]
        w_bn, b_bn = self._get_bn_w_b(bn, repeat=num_heads)
        fused_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
        fused_linear.weight.data[:] = torch.mm(w_bn, linear.weight)
        fused_linear.bias.data[:] = torch.matmul(w_bn, linear.bias) + b_bn if linear.bias is not None else b_bn
        return fused_linear

    def fuse_bn(self):
        assert not self.training
        for blk in self.blocks:
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_before_linear(blk.norm1, blk.attn.q)
                blk.attn.k = self.fuse_norm_before_linear(blk.norm1, blk.attn.k)
                blk.attn.v = self.fuse_norm_before_linear(blk.norm1, blk.attn.v)
            else:
                blk.attn.qkv = self.fuse_norm_before_linear(blk.norm1, blk.attn.qkv)
            blk.norm1 = nn.Identity()
            if blk.attn.separate_qkv:
                blk.attn.q = self.fuse_norm_after_linear(blk.attn.q, blk.attn.norm_q)
                blk.attn.k = self.fuse_norm_after_linear(blk.attn.k, blk.attn.norm_k)
                blk.attn.v = self.fuse_norm_after_linear(blk.attn.v, blk.attn.norm_v)
            else:
                w_q, w_k, w_v = blk.attn.qkv.weight.chunk(3)
                b_q, b_k, b_v = blk.attn.qkv.bias.chunk(3)
                tmp_q = nn.Linear(w_q.shape[1], w_q.shape[0], bias=True)
                tmp_k = nn.Linear(w_k.shape[1], w_k.shape[0], bias=True)
                tmp_v = nn.Linear(w_v.shape[1], w_v.shape[0], bias=True)
                tmp_q.weight.data[:] = w_q
                tmp_k.weight.data[:] = w_k
                tmp_v.weight.data[:] = w_v
                tmp_q.bias.data[:] = b_q
                tmp_k.bias.data[:] = b_k
                tmp_v.bias.data[:] = b_v
                tmp_q = self.fuse_norm_after_linear(tmp_q, blk.attn.norm_q)
                tmp_k = self.fuse_norm_after_linear(tmp_k, blk.attn.norm_k)
                tmp_v = self.fuse_norm_after_linear(tmp_v, blk.attn.norm_v)
                blk.attn.qkv.weight.data[:] = torch.cat([tmp_q.weight.data, tmp_k.weight.data, tmp_v.weight.data], dim=0)
                blk.attn.qkv.bias.data[:] = torch.cat([tmp_q.bias.data, tmp_k.bias.data, tmp_v.bias.data], dim=0)
            blk.attn.norm_q = nn.Identity()
            blk.attn.norm_k = nn.Identity()
            blk.attn.norm_v = nn.Identity()
            blk.mlp.fc1 = self.fuse_norm_before_linear(blk.norm2, blk.mlp.fc1)
            if blk.dim != blk.dim_out:
                blk.proj = self.fuse_norm_before_linear(blk.norm2, blk.proj)
            blk.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)
        x = self.pos_drop(x)
        thw = self.cls_positional_encoding.patch_embed_shape()
        for blk in self.blocks:
            x, thw = blk(x, thw)
        x = self.norm_embed(x)
        x = self.head(x)
        return x


class ProjectedPool(nn.Module):
    """
    A pooling module augmented with Conv, Normalization and Activation both
    before and after pooling for the head layer of X3D.

    ::

                                    Conv3d (pre_conv)
                                           ↓
                                 Normalization (pre_norm)
                                           ↓
                                   Activation (pre_act)
                                           ↓
                                        Pool3d
                                           ↓
                                    Conv3d (post_conv)
                                           ↓
                                 Normalization (post_norm)
                                           ↓
                                   Activation (post_act)
    """

    def __init__(self, *, pre_conv: nn.Module=None, pre_norm: nn.Module=None, pre_act: nn.Module=None, pool: nn.Module=None, post_conv: nn.Module=None, post_norm: nn.Module=None, post_act: nn.Module=None) ->None:
        """
        Args:
            pre_conv (torch.nn.modules): convolutional module.
            pre_norm (torch.nn.modules): normalization module.
            pre_act (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
            post_conv (torch.nn.modules): convolutional module.
            post_norm (torch.nn.modules): normalization module.
            post_act (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.pre_conv is not None
        assert self.pool is not None
        assert self.post_conv is not None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.pre_conv(x)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.pre_act is not None:
            x = self.pre_act(x)
        x = self.pool(x)
        x = self.post_conv(x)
        if self.post_norm is not None:
            x = self.post_norm(x)
        if self.post_act is not None:
            x = self.post_act(x)
        return x


def _mix_labels(labels: torch.Tensor, num_classes: int, lam: float=1.0, label_smoothing: float=0.0, one_hot: bool=False):
    """
    This function converts class indices to one-hot vectors and mix labels, given the
    number of classes.

    Args:
        labels (torch.Tensor): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixing labels.
        label_smoothing (float): Label smoothing value.
    """
    if one_hot:
        labels1 = labels
        labels2 = labels.flip(0)
    else:
        labels1 = convert_to_one_hot(labels, num_classes, label_smoothing)
        labels2 = convert_to_one_hot(labels.flip(0), num_classes, label_smoothing)
    return labels1 * lam + labels2 * (1.0 - lam)


class MixUp(torch.nn.Module):
    """
    Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float=1.0, label_smoothing: float=0.0, num_classes: int=400, one_hot: bool=False) ->None:
        """
        This implements MixUp for videos.

        Args:
            alpha (float): Mixup alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.mixup_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.one_hot = one_hot

    def forward(self, x_video: torch.Tensor, labels: torch.Tensor, **args: Any) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
            Optional: x_audio: Audio input tensor.
        """
        assert x_video.size(0) > 1, 'MixUp cannot be applied to a single instance.'
        mixup_lambda = self.mixup_beta_sampler.sample()
        x_video_flipped = x_video.flip(0).mul_(1.0 - mixup_lambda)
        x_video.mul_(mixup_lambda).add_(x_video_flipped)
        new_labels = _mix_labels(labels, self.num_classes, mixup_lambda, self.label_smoothing, one_hot=self.one_hot)
        if args.get('x_audio', None) is not None:
            x_audio = args['x_audio']
            assert x_audio.size(0) > 1, 'MixUp cannot be applied to a single instance.'
            x_audio_flipped = x_audio.flip(0).mul_(1.0 - mixup_lambda)
            x_audio.mul_(mixup_lambda).add_(x_audio_flipped)
            return x_video, x_audio, new_labels
        else:
            return x_video, new_labels


class CutMix(torch.nn.Module):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    (https://arxiv.org/abs/1905.04899)
    """

    def __init__(self, alpha: float=1.0, label_smoothing: float=0.0, num_classes: int=400, one_hot: bool=False) ->None:
        """
        This implements CutMix for videos.

        Args:
            alpha (float): CutMix alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        super().__init__()
        self.one_hot = one_hot
        self.cutmix_beta_sampler = torch.distributions.beta.Beta(alpha, alpha)
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _clip(self, value: int, min_value: int, max_value: int) ->int:
        """
        Clip value based on minimum value and maximum value.
        """
        return min(max(value, min_value), max_value)

    def _get_rand_box(self, input_shape: Tuple[int], cutmix_lamda: float) ->Tuple[int]:
        """
        Get a random square box given a lambda value.
        """
        ratio = (1 - cutmix_lamda) ** 0.5
        input_h, input_w = input_shape[-2:]
        cut_h, cut_w = int(input_h * ratio), int(input_w * ratio)
        cy = torch.randint(input_h, (1,)).item()
        cx = torch.randint(input_w, (1,)).item()
        yl = self._clip(cy - cut_h // 2, 0, input_h)
        yh = self._clip(cy + cut_h // 2, 0, input_h)
        xl = self._clip(cx - cut_w // 2, 0, input_w)
        xh = self._clip(cx + cut_w // 2, 0, input_w)
        return yl, yh, xl, xh

    def _cutmix(self, x: torch.Tensor, cutmix_lamda: float) ->Tuple[torch.Tensor, float]:
        """
        Perform CutMix and return corrected lambda value.
        """
        yl, yh, xl, xh = self._get_rand_box(x.size(), cutmix_lamda)
        box_area = float((yh - yl) * (xh - xl))
        cutmix_lamda_corrected = 1.0 - box_area / (x.size(-2) * x.size(-1))
        x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        return x, cutmix_lamda_corrected

    def forward(self, x_video: torch.Tensor, labels: torch.Tensor, **args: Any) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        assert x_video.size(0) > 1, 'Cutmix cannot be applied to a single instance.'
        assert x_video.dim() == 4 or x_video.dim() == 5, 'Please correct input shape.'
        cutmix_lamda = self.cutmix_beta_sampler.sample()
        x_video, cutmix_lamda_corrected = self._cutmix(x_video, cutmix_lamda)
        new_labels = _mix_labels(labels, self.num_classes, cutmix_lamda_corrected, self.label_smoothing, one_hot=self.one_hot)
        if args.get('x_audio', None) is not None:
            x_audio = args['x_audio']
            assert x_audio.size(0) > 1, 'Cutmix cannot be applied to a single instance.'
            assert x_audio.dim() == 4 or x_audio.dim() == 5, 'Please correct input shape.'
            x_audio, _ = self._cutmix(x_audio, cutmix_lamda)
            return x_video, x_audio, new_labels
        else:
            return x_video, new_labels


class MixVideo(torch.nn.Module):
    """
    Stochastically applies either MixUp or CutMix to the input video.
    """

    def __init__(self, cutmix_prob: float=0.5, mixup_alpha: float=1.0, cutmix_alpha: float=1.0, label_smoothing: float=0.0, num_classes: int=400, one_hot: bool=False):
        """
        Args:
            cutmix_prob (float): Probability of using CutMix. MixUp will be used with
                probability 1 - cutmix_prob. If cutmix_prob is 0, then MixUp is always
                used. If cutmix_prob is 1, then CutMix is always used.
            mixup_alpha (float): MixUp alpha value.
            cutmix_alpha (float): CutMix alpha value.
            label_smoothing (float): Label smoothing value.
            num_classes (int): Number of total classes.
        """
        assert 0.0 <= cutmix_prob <= 1.0, 'cutmix_prob should be between 0.0 and 1.0'
        super().__init__()
        self.cutmix_prob = cutmix_prob
        self.mixup = MixUp(alpha=mixup_alpha, label_smoothing=label_smoothing, num_classes=num_classes, one_hot=one_hot)
        self.cutmix = CutMix(alpha=cutmix_alpha, label_smoothing=label_smoothing, num_classes=num_classes)

    def forward(self, x_video: torch.Tensor, labels: torch.Tensor, **args: Any) ->Dict[str, Any]:
        """
        The input is a batch of samples and their corresponding labels.

        Args:
            x (torch.Tensor): Input tensor. The input should be a batch of videos with
                shape (B, C, T, H, W).
            labels (torch.Tensor): Labels for input with shape (B).
        """
        if args.get('x_audio', None) is None:
            if torch.rand(1).item() < self.cutmix_prob:
                x_video, new_labels = self.cutmix(x_video, labels)
            else:
                x_video, new_labels = self.mixup(x_video, labels)
            return x_video, new_labels
        else:
            x_audio = args['x_audio']
            if torch.rand(1).item() < self.cutmix_prob:
                x_video, new_labels, x_audio = self.cutmix(x_video, labels, x_audio)
            else:
                x_video, new_labels, x_audio = self.mixup(x_video, labels, x_audio)
            return x_video, x_audio, new_labels


class RemoveKey(torch.nn.Module):
    """
    Removes the given key from the input dict. Useful for removing modalities from a
    video clip that aren't needed.
    """

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def __call__(self, x: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        if self._key in x:
            del x[self._key]
        return x


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int=-3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.uniform_temporal_subsample(x, self._num_samples, self._temporal_dim)


class UniformTemporalSubsampleRepeated(torch.nn.Module):
    """
    ``nn.Module`` wrapper for
    ``pytorchvideo.transforms.functional.uniform_temporal_subsample_repeated``.
    """

    def __init__(self, frame_ratios: Tuple[int], temporal_dim: int=-3):
        super().__init__()
        self._frame_ratios = frame_ratios
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.uniform_temporal_subsample_repeated(x, self._frame_ratios, self._temporal_dim)


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(self, size: int, interpolation: str='bilinear', backend: str='pytorch'):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.short_side_scale(x, self._size, self._interpolation, self._backend)


class RandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(self, min_size: int, max_size: int, interpolation: str='bilinear', backend: str='pytorch'):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return pytorchvideo.transforms.functional.short_side_scale(x, size, self._interpolation, self._backend)


class UniformCropVideo(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_crop``.
    """

    def __init__(self, size: int, video_key: str='video', aug_index_key: str='aug_index'):
        super().__init__()
        self._size = size
        self._video_key = video_key
        self._aug_index_key = aug_index_key

    def __call__(self, x: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        x[self._video_key] = pytorchvideo.transforms.functional.uniform_crop(x[self._video_key], self._size, x[self._aug_index_key])
        return x


class ConvertFloatToUint8(torch.nn.Module):
    """
    Converts a video from dtype float32 to dtype uint8.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.uint8)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert x.dtype == torch.float or x.dtype == torch.half, 'image must have dtype torch.uint8'
        return self.convert_func(x)


class ConvertUint8ToFloat(torch.nn.Module):
    """
    Converts a video from dtype uint8 to dtype float32.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.float32)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert x.dtype == torch.uint8, 'image must have dtype torch.uint8'
        return self.convert_func(x)


class MoveChannelRear(torch.nn.Module):
    """
    A Scriptable version to perform C X Y Z -> X Y Z C.
    """

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        x = x.permute([1, 2, 3, 0])
        return x


class MoveChannelFront(torch.nn.Module):
    """
    A Scriptable version to perform X Y Z C -> C X Y Z.
    """

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        x = x.permute([3, 0, 1, 2])
        return x


class RandomResizedCrop(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.random_resized_crop``.
    """

    def __init__(self, target_height: int, target_width: int, scale: Tuple[float, float], aspect_ratio: Tuple[float, float], shift: bool=False, log_uniform_ratio: bool=True, interpolation: str='bilinear', num_tries: int=10) ->None:
        super().__init__()
        self._target_height = target_height
        self._target_width = target_width
        self._scale = scale
        self._aspect_ratio = aspect_ratio
        self._shift = shift
        self._log_uniform_ratio = log_uniform_ratio
        self._interpolation = interpolation
        self._num_tries = num_tries

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input video tensor with shape (C, T, H, W).
        """
        return pytorchvideo.transforms.functional.random_resized_crop(x, self._target_height, self._target_width, self._scale, self._aspect_ratio, self._shift, self._log_uniform_ratio, self._interpolation, self._num_tries)


class Permute(torch.nn.Module):
    """
    Permutes the dimensions of a video.
    """

    def __init__(self, dims: Tuple[int]):
        """
        Args:
            dims (Tuple[int]): The desired ordering of dimensions.
        """
        assert (d in dims for d in range(len(dims))), 'dims must contain every dimension (0, 1, 2, ...)'
        super().__init__()
        self._dims = dims

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        return x.permute(*self._dims)


class OpSampler(torch.nn.Module):
    """
    Given a list of transforms with weights, OpSampler applies weighted sampling to
    select n transforms, which are then applied sequentially to the input.
    """

    def __init__(self, transforms_list: List[Callable], transforms_prob: Optional[List[float]]=None, num_sample_op: int=1, randomly_sample_depth: bool=False, replacement: bool=False):
        """
        Args:
            transforms_list (List[Callable]): A list of tuples of all available transforms
                to sample from.
            transforms_prob (Optional[List[float]]): The probabilities associated with
                each transform in transforms_list. If not provided, the sampler assumes a
                uniform distribution over all transforms. They do not need to sum up to one
                but weights need to be positive.
            num_sample_op (int): Number of transforms to sample and apply to input.
            randomly_sample_depth (bool): If randomly_sample_depth is True, then uniformly
                sample the number of transforms to apply, between 1 and num_sample_op.
            replacement (bool): If replacement is True, transforms are drawn with replacement.
        """
        super().__init__()
        assert len(transforms_list) > 0, 'Argument transforms_list cannot be empty.'
        assert num_sample_op > 0, 'Need to sample at least one transform.'
        assert num_sample_op <= len(transforms_list), 'Argument num_sample_op cannot be greater than number of available transforms.'
        if transforms_prob is not None:
            assert len(transforms_list) == len(transforms_prob), 'Argument transforms_prob needs to have the same length as transforms_list.'
            assert min(transforms_prob) > 0, 'Argument transforms_prob needs to be greater than 0.'
        self.transforms_list = transforms_list
        self.transforms_prob = torch.FloatTensor(transforms_prob if transforms_prob is not None else [1] * len(transforms_list))
        self.num_sample_op = num_sample_op
        self.randomly_sample_depth = randomly_sample_depth
        self.replacement = replacement

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        depth = torch.randint(1, self.num_sample_op + 1, (1,)).item() if self.randomly_sample_depth else self.num_sample_op
        index_list = torch.multinomial(self.transforms_prob, depth, replacement=self.replacement)
        for index in index_list:
            x = self.transforms_list[index](x)
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return torchvision.transforms.Lambda(pytorchvideo.transforms.functional.div_255)(x)


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction: str='mean') ->None:
        """
        Args:
            reduction (str): specifies reduction to apply to the output.
                It can be "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class NtxentLoss(nn.Module):
    """
    NT-Xent loss for SimCLR Self-Supervised learning approach -
    https://arxiv.org/abs/2002.05709

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(self, temperature: float) ->None:
        super().__init__()
        set_attributes(self, locals())

    def forward(self, x_list: List[torch.Tensor]) ->torch.Tensor:
        """
        Args:
            x_list (list[torch.tensor]): A list of two tensors of shape N x C.
                Where, N is the batch size and C is the SSL model's embedding size.
        """
        assert len(x_list) == 2, f'Invalid list input to SimCLR. Expected dimention 2 but received {len(x_list)}'
        out_1, out_2 = x_list
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1 = du.AllGatherWithGradient.apply(out_1)
            out_2 = du.AllGatherWithGradient.apply(out_2)
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(out.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(out.shape[0], -1)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class SimilarityLoss(nn.Module):
    """
    Temperature-scaled Similarity loss for BYOL Self-Supervised learning
    approach - https://arxiv.org/abs/2006.07733

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(self, temperature: float) ->None:
        super().__init__()
        self.temperature = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor) ->torch.Tensor:
        """
        Args:
            q and k (nn.tensor): inputs to calculate the similarity, expected to have
                the same shape of `N x C`. Where N is the batch size and C
                is the SSL model's embedding size.
        """
        similarity = torch.einsum('nc,nc->n', [q, k])
        similarity /= self.temperature
        loss = -similarity.mean()
        return loss


class ContrastiveLoss(nn.Module):
    """
    Temperature-scaled Contrastive loss for MoCo and other Self-Supervised learning
    approaches - https://arxiv.org/abs/1911.05722

    Args:
        temperature (float): scalar value to scale the loss by.
    """

    def __init__(self, reduction: str='mean', temperature: float=0.1) ->None:
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, inputs: torch.Tensor) ->torch.Tensor:
        """
        Args:
            inputs (nn.tensor):  Expected to have the same shape of `N x C`.
                Where, N is the batch size and C is the SSL model's embedding size.
        """
        inputs = torch.div(inputs, self.temperature)
        targets = torch.zeros(inputs.shape[0], dtype=torch.long)
        loss = nn.CrossEntropyLoss(reduction=self.reduction)(inputs, targets)
        return loss


class MOCO(nn.Module):
    """
    Momentum Contrast for unsupervised Visual Representation Learning
    Details can be found in:
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, mmt: float, backbone: nn.Module, backbone_mmt: nn.Module, projector: Optional[nn.Module]=None, projector_mmt: Optional[nn.Module]=None) ->None:
        """
        Args:
            backbone (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector (nn.Module): An mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            backbone_mmt (nn.Module): backbone for byol, input shape depends on the forward
                input size. Standard inputs include `B x C`, `B x C x H x W`, and
                `B x C x T x H x W`.
            projector_mmt (nn.Module): Am mlp with 2 to 3 hidden layers,
                with (synchronized) BatchNorm and ReLU activation.
            mmt (float): momentum update ratio for the momentum backbone.
        """
        super().__init__()
        self.mmt: float = mmt
        if projector is not None:
            backbone = nn.Sequential(backbone, projector)
        init_net_weights(backbone)
        self.backbone = backbone
        if projector_mmt is not None:
            backbone_mmt = nn.Sequential(backbone_mmt, projector_mmt)
        init_net_weights(backbone_mmt)
        self.backbone_mmt = backbone_mmt
        for p in self.backbone_mmt.parameters():
            p.requires_grad = False
        self._copy_weights_to_backbone_mmt()

    def _copy_weights_to_backbone_mmt(self) ->None:
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            p.data.copy_(dist[name].data)

    @torch.no_grad()
    def momentum_update_backbone(self) ->None:
        """
        Momentum update on the backbone.
        """
        m = self.mmt
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p
        for name, p in self.backbone_mmt.named_parameters():
            p.data = dist[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def forward_backbone_mmt(self, x: torch.Tensor) ->torch.Tensor:
        """
        Forward momentum backbone.
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        with torch.no_grad():
            proj = self.backbone_mmt(x)
            out_proj = F.normalize(proj, dim=1)
        return out_proj

    def forward(self, x: torch.Tensor) ->Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x (tensor): input to be forwarded of shape N x C x T x H x W
        """
        proj = self.backbone(x)
        out_proj = F.normalize(proj, dim=1)
        return out_proj


class SSLFineTuningModel(nn.Module):
    """
    Model consisting of a backbone sequentially followed by an an MLP.
    Used for supervised finetuning of the SSL pre-trained models.

    Args:
        backbone (nn.Module): A model whole weights are conditionally
            updated based on the betach_backbone parameter.
        mlp (nn.Module): If specified, the MLP module to attach to the bacbone
            for the supervised finetuning phase.
        detach_bacbone: If true, detaches bacbone and no gradient are tracked and
            updated for the bacbone. Only updates the MLP weights during finetuning.
    """

    def __init__(self, backbone: nn.Module, mlp: nn.Module, detach_backbone: bool) ->None:
        super().__init__()
        self.backbone = backbone
        self.mlp = mlp
        self.detach_backbone = detach_backbone
        for p in self.backbone.parameters():
            p.requires_grad = False if detach_backbone else True

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.backbone(x)
        if self.detach_backbone:
            x = x.detach()
        if self.mlp is not None:
            x = self.mlp(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveAvgPool2d,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AdaptiveAvgPool2dOutSize1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AdaptiveAvgPool3d,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AdaptiveAvgPool3dOutSize1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Conv3d3x1x1BnAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3d3x3x3DwBnAct,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3d5x1x1BnAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3dPwBnAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3dTemporalKernel1BnAct,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvReduce3D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvertFloatToUint8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EfficientBlockBase,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (FullyConnected,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FuseAudioToFastSlow,
     lambda: ([], {'block_fast_to_slow': _mock_layer(), 'block_audio_to_fastslow': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (FuseFastToSlow,
     lambda: ([], {'conv_fast_to_slow': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSTM,
     lambda: ([], {'dim_in': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LearnMaskedDefault,
     lambda: ([], {'feature_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedTemporalPooling,
     lambda: ([], {'method': 'max'}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoOpConvertBlock,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoolConcatPathway,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequencePool,
     lambda: ([], {'mode': 'cls'}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SimilarityLoss,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (SoftTargetCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftTargetCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransposeMultiheadAttention,
     lambda: ([], {'feature_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TransposeTransformerEncoder,
     lambda: ([], {'dim_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (_NaiveSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_SkipConnectMul,
     lambda: ([], {'layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_facebookresearch_pytorchvideo(_paritybench_base):
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

