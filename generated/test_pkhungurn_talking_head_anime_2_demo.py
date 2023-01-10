import sys
_module = sys.modules[__name__]
del sys
main = _module
tha2 = _module
app = _module
ifacialmocap_puppeteer = _module
manual_poser = _module
compute = _module
cached_computation_func = _module
cached_computation_protocol = _module
mocap = _module
ifacialmocap_constants = _module
ifacialmocap_pose_converter = _module
nn = _module
backbone = _module
poser_args = _module
poser_encoder_decoder_00 = _module
backcomp = _module
conv = _module
encoder_decoder_module = _module
init_function = _module
resnet_block = _module
u_net_module = _module
tha = _module
combiner = _module
face_morpher = _module
two_algo_face_rotator = _module
base = _module
conv = _module
init_function = _module
module_factory = _module
nonlinearity_factory = _module
normalization = _module
pass_through = _module
resnet_block = _module
spectral_norm = _module
util = _module
view_change = _module
batch_module = _module
batch_input_model_factory = _module
batch_input_module = _module
eyebrow = _module
eyebrow_decomposer_00 = _module
eyebrow_morphing_combiner_00 = _module
face = _module
face_morpher_08 = _module
util = _module
poser = _module
general_poser_02 = _module
modes = _module
mode_20 = _module
mode_20_wx = _module
poser = _module
util = _module

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


import queue


import time


from typing import Optional


import numpy


import torch


import logging


from typing import List


from typing import Callable


from typing import Dict


from torch import Tensor


from abc import ABC


from abc import abstractmethod


from torch.nn import Sigmoid


from torch.nn import Sequential


from torch.nn import Tanh


import math


from torch.nn import ModuleList


from torch.nn import Module


from torch.nn import Conv2d


from torch.nn import InstanceNorm2d


from torch.nn import ReLU


from torch.nn import ConvTranspose2d


from torch.nn.init import kaiming_normal_


from torch.nn.init import xavier_normal_


from torch import relu


from torch.nn.functional import affine_grid


from torch.nn.functional import grid_sample


from torch import zero_


from torch.nn.init import normal_


from torch.nn import LeakyReLU


from torch.nn import ELU


from torch.nn import BatchNorm2d


from torch.nn import Parameter


from torch.nn.init import constant_


from torch.nn.utils import spectral_norm


from typing import Tuple


from enum import Enum


from matplotlib import cm


class ModuleFactory(ABC):

    @abstractmethod
    def create(self) ->Module:
        pass


class NormalizationLayerFactory(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, num_features: int, affine: bool=True) ->Module:
        pass

    @staticmethod
    def resolve_2d(factory: Optional['NormalizationLayerFactory']) ->'NormalizationLayerFactory':
        if factory is None:
            return InstanceNorm2dFactory()
        else:
            return factory


class ReLUFactory(ModuleFactory):

    def __init__(self, inplace: bool=False):
        self.inplace = inplace

    def create(self) ->Module:
        return ReLU(self.inplace)


def resolve_nonlinearity_factory(nonlinearity_fatory: Optional[ModuleFactory]) ->ModuleFactory:
    if nonlinearity_fatory is None:
        return ReLUFactory(inplace=True)
    else:
        return nonlinearity_fatory


def apply_spectral_norm(module: Module, use_spectrial_norm: bool=False) ->Module:
    if use_spectrial_norm:
        return spectral_norm(module)
    else:
        return module


def create_init_function(method: str='none'):

    def init(module: Module):
        if method == 'none':
            return module
        elif method == 'he':
            kaiming_normal_(module.weight)
            return module
        elif method == 'xavier':
            xavier_normal_(module.weight)
            return module
        elif method == 'dcgan':
            normal_(module.weight, 0.0, 0.02)
            return module
        elif method == 'dcgan_001':
            normal_(module.weight, 0.0, 0.01)
            return module
        elif method == 'zero':
            with torch.no_grad():
                zero_(module.weight)
            return module
        else:
            raise ('Invalid initialization method %s' % method)
    return init


def wrap_conv_or_linear_module(module: Module, initialization_method: str, use_spectral_norm: bool):
    init = create_init_function(initialization_method)
    return apply_spectral_norm(init(module), use_spectral_norm)


class BlockArgs:

    def __init__(self, initialization_method: str='he', use_spectral_norm: bool=False, normalization_layer_factory: Optional[NormalizationLayerFactory]=None, nonlinearity_factory: Optional[ModuleFactory]=None):
        self.nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        self.normalization_layer_factory = normalization_layer_factory
        self.use_spectral_norm = use_spectral_norm
        self.initialization_method = initialization_method

    def wrap_module(self, module: Module) ->Module:
        return wrap_conv_or_linear_module(module, self.initialization_method, self.use_spectral_norm)


def create_conv3(in_channels: int, out_channels: int, bias: bool=False, initialization_method='he', use_spectral_norm: bool=False) ->Module:
    return wrap_conv_or_linear_module(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias), initialization_method, use_spectral_norm)


def create_conv3_from_block_args(in_channels: int, out_channels: int, bias: bool=False, block_args: Optional[BlockArgs]=None):
    if block_args is None:
        block_args = BlockArgs()
    return create_conv3(in_channels, out_channels, bias, block_args.initialization_method, block_args.use_spectral_norm)


class PoserArgs00:

    def __init__(self, image_size: int, input_image_channels: int, output_image_channels: int, start_channels: int, num_pose_params: int, block_args: Optional[BlockArgs]=None):
        self.num_pose_params = num_pose_params
        self.start_channels = start_channels
        self.output_image_channels = output_image_channels
        self.input_image_channels = input_image_channels
        self.image_size = image_size
        if block_args is None:
            self.block_args = BlockArgs(normalization_layer_factory=InstanceNorm2dFactory(), nonlinearity_factory=ReLUFactory(inplace=True))
        else:
            self.block_args = block_args

    def create_alpha_block(self):
        from torch.nn import Sequential
        return Sequential(create_conv3(in_channels=self.start_channels, out_channels=1, bias=True, initialization_method=self.block_args.initialization_method, use_spectral_norm=False), Sigmoid())

    def create_all_channel_alpha_block(self):
        from torch.nn import Sequential
        return Sequential(create_conv3(in_channels=self.start_channels, out_channels=self.output_image_channels, bias=True, initialization_method=self.block_args.initialization_method, use_spectral_norm=False), Sigmoid())

    def create_color_change_block(self):
        return Sequential(create_conv3_from_block_args(in_channels=self.start_channels, out_channels=self.output_image_channels, bias=True, block_args=self.block_args), Tanh())

    def create_grid_change_block(self):
        return create_conv3(in_channels=self.start_channels, out_channels=2, bias=False, initialization_method='zero', use_spectral_norm=False)


class PoserEncoderDecoder00Args(PoserArgs00):

    def __init__(self, image_size: int, input_image_channels: int, output_image_channels: int, num_pose_params: int, start_channels: int, bottleneck_image_size, num_bottleneck_blocks, max_channels: int, block_args: Optional[BlockArgs]=None):
        super().__init__(image_size, input_image_channels, output_image_channels, start_channels, num_pose_params, block_args)
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        assert bottleneck_image_size > 1
        if block_args is None:
            self.block_args = BlockArgs(normalization_layer_factory=InstanceNorm2dFactory(), nonlinearity_factory=ReLUFactory(inplace=True))
        else:
            self.block_args = block_args


def create_conv1(in_channels: int, out_channels: int, initialization_method='he', bias: bool=False, use_spectral_norm: bool=False) ->Module:
    return wrap_conv_or_linear_module(Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias), initialization_method, use_spectral_norm)


class ResnetBlock(Module):

    @staticmethod
    def create(num_channels: int, is1x1: bool=False, use_scale_parameters: bool=False, block_args: Optional[BlockArgs]=None):
        if block_args is None:
            block_args = BlockArgs()
        return ResnetBlock(num_channels, is1x1, block_args.initialization_method, block_args.nonlinearity_factory, block_args.normalization_layer_factory, block_args.use_spectral_norm, use_scale_parameters)

    def __init__(self, num_channels: int, is1x1: bool=False, initialization_method: str='he', nonlinearity_factory: ModuleFactory=None, normalization_layer_factory: Optional[NormalizationLayerFactory]=None, use_spectral_norm: bool=False, use_scale_parameter: bool=False):
        super().__init__()
        self.use_scale_parameter = use_scale_parameter
        if self.use_scale_parameter:
            self.scale = Parameter(torch.zeros(1))
        nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        if is1x1:
            self.resnet_path = Sequential(create_conv1(num_channels, num_channels, initialization_method, bias=True, use_spectral_norm=use_spectral_norm), nonlinearity_factory.create(), create_conv1(num_channels, num_channels, initialization_method, bias=True, use_spectral_norm=use_spectral_norm))
        else:
            self.resnet_path = Sequential(create_conv3(num_channels, num_channels, bias=False, initialization_method=initialization_method, use_spectral_norm=use_spectral_norm), NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(num_channels, affine=True), nonlinearity_factory.create(), create_conv3(num_channels, num_channels, bias=False, initialization_method=initialization_method, use_spectral_norm=use_spectral_norm), NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(num_channels, affine=True))

    def forward(self, x):
        if self.use_scale_parameter:
            return x + self.scale * self.resnet_path(x)
        else:
            return x + self.resnet_path(x)


def create_conv3_block(in_channels: int, out_channels: int, initialization_method='he', nonlinearity_factory: Optional[ModuleFactory]=None, normalization_layer_factory: Optional[NormalizationLayerFactory]=None, use_spectral_norm: bool=False) ->Module:
    nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
    return Sequential(create_conv3(in_channels, out_channels, bias=False, initialization_method=initialization_method, use_spectral_norm=use_spectral_norm), NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True), resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_conv3_block_from_block_args(in_channels: int, out_channels: int, block_args: Optional[BlockArgs]=None):
    if block_args is None:
        block_args = BlockArgs()
    return create_conv3_block(in_channels, out_channels, block_args.initialization_method, block_args.nonlinearity_factory, block_args.normalization_layer_factory, block_args.use_spectral_norm)


def create_downsample_block(in_channels: int, out_channels: int, is_output_1x1: bool=False, initialization_method='he', nonlinearity_factory: Optional[ModuleFactory]=None, normalization_layer_factory: Optional[NormalizationLayerFactory]=None, use_spectral_norm: bool=False) ->Module:
    if is_output_1x1:
        return Sequential(wrap_conv_or_linear_module(Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), initialization_method, use_spectral_norm), resolve_nonlinearity_factory(nonlinearity_factory).create())
    else:
        return Sequential(wrap_conv_or_linear_module(Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), initialization_method, use_spectral_norm), NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True), resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_downsample_block_from_block_args(in_channels: int, out_channels: int, is_output_1x1: bool=False, block_args: Optional[BlockArgs]=None):
    if block_args is None:
        block_args = BlockArgs()
    return create_downsample_block(in_channels, out_channels, is_output_1x1, block_args.initialization_method, block_args.nonlinearity_factory, block_args.normalization_layer_factory, block_args.use_spectral_norm)


def create_upsample_block(in_channels: int, out_channels: int, initialization_method='he', nonlinearity_factory: Optional[ModuleFactory]=None, normalization_layer_factory: Optional[NormalizationLayerFactory]=None, use_spectral_norm: bool=False) ->Module:
    nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
    return Sequential(wrap_conv_or_linear_module(ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), initialization_method, use_spectral_norm), NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True), resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_upsample_block_from_block_args(in_channels: int, out_channels: int, block_args: Optional[BlockArgs]=None) ->Module:
    if block_args is None:
        block_args = BlockArgs()
    return create_upsample_block(in_channels, out_channels, block_args.initialization_method, block_args.nonlinearity_factory, block_args.normalization_layer_factory, block_args.use_spectral_norm)


class PoserEncoderDecoder00(Module):

    def __init__(self, args: PoserEncoderDecoder00Args):
        super().__init__()
        self.args = args
        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1
        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(create_conv3_block_from_block_args(args.input_image_channels, args.start_channels, args.block_args))
        current_image_size = args.image_size
        current_num_channels = args.start_channels
        while current_image_size > args.bottleneck_image_size:
            next_image_size = current_image_size // 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.downsample_blocks.append(create_downsample_block_from_block_args(in_channels=current_num_channels, out_channels=next_num_channels, is_output_1x1=False, block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        assert len(self.downsample_blocks) == self.num_levels
        self.bottleneck_blocks = ModuleList()
        self.bottleneck_blocks.append(create_conv3_block_from_block_args(in_channels=current_num_channels + args.num_pose_params, out_channels=current_num_channels, block_args=args.block_args))
        for i in range(1, args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(ResnetBlock.create(num_channels=current_num_channels, is1x1=False, block_args=args.block_args))
        self.upsample_blocks = ModuleList()
        while current_image_size < args.image_size:
            next_image_size = current_image_size * 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.upsample_blocks.append(create_upsample_block_from_block_args(in_channels=current_num_channels, out_channels=next_num_channels, block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // 2 ** level)

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Optional[Tensor]=None) ->List[Tensor]:
        if self.args.num_pose_params != 0:
            assert pose is not None
        else:
            assert pose is None
        outputs = []
        feature = image
        outputs.append(feature)
        for block in self.downsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        if pose is not None:
            n, c = pose.shape
            pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.bottleneck_image_size, self.args.bottleneck_image_size)
            feature = torch.cat([feature, pose], dim=1)
        for block in self.bottleneck_blocks:
            feature = block(feature)
            outputs.append(feature)
        for block in self.upsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        outputs.reverse()
        return outputs


def Conv7(in_channels: int, out_channels: int, initialization_method='he') ->Module:
    init = create_init_function(initialization_method)
    return init(Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False))


def Conv7Block(in_channels: int, out_channels: int, initialization_method='he') ->Module:
    return Sequential(Conv7(in_channels, out_channels, initialization_method), InstanceNorm2d(out_channels, affine=True), ReLU(inplace=True))


def DownsampleBlock(in_channels: int, initialization_method='he') ->Module:
    init = create_init_function(initialization_method)
    return Sequential(init(Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)), InstanceNorm2d(in_channels * 2, affine=True), ReLU(inplace=True))


def Conv3(in_channels: int, out_channels: int, initialization_method='he') ->Module:
    init = create_init_function(initialization_method)
    return init(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))


class ResNetBlock(Module):

    def __init__(self, num_channels: int, initialization_method: str='he'):
        super().__init__()
        self.conv1 = Conv3(num_channels, num_channels, initialization_method)
        self.norm1 = InstanceNorm2d(num_features=num_channels, affine=True)
        self.conv2 = Conv3(num_channels, num_channels, initialization_method)
        self.norm2 = InstanceNorm2d(num_features=num_channels, affine=True)

    def forward(self, x):
        return x + self.norm2(self.conv2(relu(self.norm1(self.conv1(x)))))


def UpsampleBlock(in_channels: int, out_channels: int, initialization_method='he') ->Module:
    init = create_init_function(initialization_method)
    return Sequential(init(ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)), InstanceNorm2d(out_channels, affine=True), ReLU(inplace=True))


class EncoderDecoderModule(Module):

    def __init__(self, image_size: int, image_channels: int, output_channels: int, bottleneck_image_size: int, bottleneck_block_count: int, initialization_method: str='he'):
        super().__init__()
        self.module_list = ModuleList()
        self.module_list.append(Conv7Block(image_channels, output_channels))
        current_size = image_size
        current_channels = output_channels
        while current_size > bottleneck_image_size:
            self.module_list.append(DownsampleBlock(current_channels, initialization_method))
            current_size //= 2
            current_channels *= 2
        for i in range(bottleneck_block_count):
            self.module_list.append(ResNetBlock(current_channels, initialization_method))
        while current_size < image_size:
            self.module_list.append(UpsampleBlock(current_channels, current_channels // 2, initialization_method))
            current_size *= 2
            current_channels //= 2

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class UNetModule(Module):

    def __init__(self, image_size: int, image_channels: int, output_channels: int, bottleneck_image_size: int, bottleneck_block_count: int, initialization_method: str='he'):
        super().__init__()
        self.downward_modules = ModuleList()
        self.downward_module_channel_count = {}
        self.downward_modules.append(Conv7Block(image_channels, output_channels, initialization_method))
        self.downward_module_channel_count[image_size] = output_channels
        current_channels = output_channels
        current_image_size = image_size
        while current_image_size > bottleneck_image_size:
            self.downward_modules.append(DownsampleBlock(current_channels, initialization_method))
            current_channels = current_channels * 2
            current_image_size = current_image_size // 2
            self.downward_module_channel_count[current_image_size] = current_channels
        self.bottleneck_modules = ModuleList()
        for i in range(bottleneck_block_count):
            self.bottleneck_modules.append(ResNetBlock(current_channels, initialization_method))
        self.upsampling_modules = ModuleList()
        while current_image_size < image_size:
            if current_image_size == bottleneck_image_size:
                input_channels = current_channels
            else:
                input_channels = current_channels + self.downward_module_channel_count[current_image_size]
            self.upsampling_modules.insert(0, UpsampleBlock(input_channels, current_channels // 2, initialization_method))
            current_channels = current_channels // 2
            current_image_size = current_image_size * 2
        self.upsampling_modules.insert(0, Conv7Block(current_channels + output_channels, output_channels, initialization_method))

    def forward(self, x):
        downward_outputs = []
        for module in self.downward_modules:
            x = module(x)
            downward_outputs.append(x)
        for module in self.bottleneck_modules:
            x = module(x)
        x = self.upsampling_modules[-1](x)
        for i in range(len(self.upsampling_modules) - 2, -1, -1):
            y = torch.cat([x, downward_outputs[i]], dim=1)
            x = self.upsampling_modules[i](y)
        return x


class PixelNormalization(Module):

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + self.epsilon)


class Bias2d(Module):

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return x + self.bias


class PassThrough(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ViewChange(Module):

    def __init__(self, new_size: List[int]):
        super().__init__()
        self.new_size = new_size

    def forward(self, x: Tensor):
        n = x.shape[0]
        return x.view([n] + self.new_size)


class ViewImageAsVector(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        assert x.dim() == 4
        n, c, w, h = x.shape
        return x.view(n, c * w * h)


class ViewVectorAsMultiChannelImage(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        assert x.dim() == 2
        n, c = x.shape
        return x.view(n, c, 1, 1)


class ViewVectorAsOneChannelImage(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        assert x.dim() == 2
        n, c = x.shape
        return x.view(n, 1, c, 1)


class BatchInputModule(Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_from_batch(self, batch: List[Tensor]):
        pass


class EyebrowDecomposer00Args(PoserEncoderDecoder00Args):

    def __init__(self, image_size: int=128, image_channels: int=4, start_channels: int=64, bottleneck_image_size=16, num_bottleneck_blocks=6, max_channels: int=512, block_args: Optional[BlockArgs]=None):
        super().__init__(image_size, image_channels, image_channels, 0, start_channels, bottleneck_image_size, num_bottleneck_blocks, max_channels, block_args)


def apply_color_change(alpha, color_change, image: Tensor) ->Tensor:
    return color_change * alpha + image * (1 - alpha)


class EyebrowDecomposer00(BatchInputModule):

    def __init__(self, args: EyebrowDecomposer00Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00(args)
        self.background_layer_alpha = self.args.create_alpha_block()
        self.background_layer_color_change = self.args.create_color_change_block()
        self.eyebrow_layer_alpha = self.args.create_alpha_block()
        self.eyebrow_layer_color_change = self.args.create_color_change_block()

    def forward(self, image: Tensor) ->List[Tensor]:
        feature = self.body(image)[0]
        background_layer_alpha = self.background_layer_alpha(feature)
        background_layer_color_change = self.background_layer_color_change(feature)
        background_layer_1 = apply_color_change(background_layer_alpha, background_layer_color_change, image)
        eyebrow_layer_alpha = self.eyebrow_layer_alpha(feature)
        eyebrow_layer_color_change = self.eyebrow_layer_color_change(feature)
        eyebrow_layer = apply_color_change(eyebrow_layer_alpha, image, eyebrow_layer_color_change)
        return [eyebrow_layer, eyebrow_layer_alpha, eyebrow_layer_color_change, background_layer_1, background_layer_alpha, background_layer_color_change]
    EYEBROW_LAYER_INDEX = 0
    EYEBROW_LAYER_ALPHA_INDEX = 1
    EYEBROW_LAYER_COLOR_CHANGE_INDEX = 2
    BACKGROUND_LAYER_INDEX = 3
    BACKGROUND_LAYER_ALPHA_INDEX = 4
    BACKGROUND_LAYER_COLOR_CHANGE_INDEX = 5
    OUTPUT_LENGTH = 6

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0])


class EyebrowMorphingCombiner00Args(PoserEncoderDecoder00Args):

    def __init__(self, image_size: int=128, image_channels: int=4, num_pose_params: int=12, start_channels: int=64, bottleneck_image_size=16, num_bottleneck_blocks=6, max_channels: int=512, block_args: Optional[BlockArgs]=None):
        super().__init__(image_size, 2 * image_channels, image_channels, num_pose_params, start_channels, bottleneck_image_size, num_bottleneck_blocks, max_channels, block_args)


def apply_grid_change(grid_change, image: Tensor) ->Tensor:
    n, c, h, w = image.shape
    device = grid_change.device
    grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).repeat(n, 1, 1)
    base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
    grid = base_grid + grid_change
    resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return resampled_image


def apply_rgb_change(alpha: Tensor, color_change: Tensor, image: Tensor):
    image_rgb = image[:, 0:3, :, :]
    color_change_rgb = color_change[:, 0:3, :, :]
    output_rgb = color_change_rgb * alpha + image_rgb * (1 - alpha)
    return torch.cat([output_rgb, image[:, 3:4, :, :]], dim=1)


class EyebrowMorphingCombiner00(BatchInputModule):

    def __init__(self, args: EyebrowMorphingCombiner00Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00(args)
        self.morphed_eyebrow_layer_grid_change = self.args.create_grid_change_block()
        self.morphed_eyebrow_layer_alpha = self.args.create_alpha_block()
        self.morphed_eyebrow_layer_color_change = self.args.create_color_change_block()
        self.combine_alpha = self.args.create_alpha_block()

    def forward(self, background_layer: Tensor, eyebrow_layer: Tensor, pose: Tensor) ->List[Tensor]:
        combined_image = torch.cat([background_layer, eyebrow_layer], dim=1)
        feature = self.body(combined_image, pose)[0]
        morphed_eyebrow_layer_grid_change = self.morphed_eyebrow_layer_grid_change(feature)
        morphed_eyebrow_layer_alpha = self.morphed_eyebrow_layer_alpha(feature)
        morphed_eyebrow_layer_color_change = self.morphed_eyebrow_layer_color_change(feature)
        warped_eyebrow_layer = apply_grid_change(morphed_eyebrow_layer_grid_change, eyebrow_layer)
        morphed_eyebrow_layer = apply_color_change(morphed_eyebrow_layer_alpha, morphed_eyebrow_layer_color_change, warped_eyebrow_layer)
        combine_alpha = self.combine_alpha(feature)
        eyebrow_image = apply_rgb_change(combine_alpha, morphed_eyebrow_layer, background_layer)
        eyebrow_image_no_combine_alpha = apply_rgb_change((morphed_eyebrow_layer[:, 3:4, :, :] + 1.0) / 2.0, morphed_eyebrow_layer, background_layer)
        return [eyebrow_image, combine_alpha, eyebrow_image_no_combine_alpha, morphed_eyebrow_layer, morphed_eyebrow_layer_alpha, morphed_eyebrow_layer_color_change, warped_eyebrow_layer, morphed_eyebrow_layer_grid_change]
    EYEBROW_IMAGE_INDEX = 0
    COMBINE_ALPHA_INDEX = 1
    EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX = 2
    MORPHED_EYEBROW_LAYER_INDEX = 3
    MORPHED_EYEBROW_LAYER_ALPHA_INDEX = 4
    MORPHED_EYEBROW_LAYER_COLOR_CHANGE_INDEX = 5
    WARPED_EYEBROW_LAYER_INDEX = 6
    MORPHED_EYEBROW_LAYER_GRID_CHANGE_INDEX = 7
    OUTPUT_LENGTH = 8

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0], batch[1], batch[2])


class LeakyReLUFactory(ModuleFactory):

    def __init__(self, inplace: bool=False, negative_slope: float=0.01):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def create(self) ->Module:
        return LeakyReLU(inplace=self.inplace, negative_slope=self.negative_slope)


class FaceMorpher08Args:

    def __init__(self, image_size: int=256, image_channels: int=4, num_expression_params: int=67, start_channels: int=16, bottleneck_image_size=4, num_bottleneck_blocks=3, max_channels: int=512, block_args: Optional[BlockArgs]=None):
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        assert bottleneck_image_size > 1
        self.bottleneck_image_size = bottleneck_image_size
        self.start_channels = start_channels
        self.image_channels = image_channels
        self.num_expression_params = num_expression_params
        self.image_size = image_size
        if block_args is None:
            self.block_args = BlockArgs(normalization_layer_factory=InstanceNorm2dFactory(), nonlinearity_factory=LeakyReLUFactory(negative_slope=0.2, inplace=True))
        else:
            self.block_args = block_args


class FaceMorpher08(BatchInputModule):

    def __init__(self, args: FaceMorpher08Args):
        super().__init__()
        self.args = args
        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1
        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(create_conv3_block_from_block_args(args.image_channels, args.start_channels, args.block_args))
        current_image_size = args.image_size
        current_num_channels = args.start_channels
        while current_image_size > args.bottleneck_image_size:
            next_image_size = current_image_size // 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.downsample_blocks.append(create_downsample_block_from_block_args(in_channels=current_num_channels, out_channels=next_num_channels, is_output_1x1=False, block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        assert len(self.downsample_blocks) == self.num_levels
        self.bottleneck_blocks = ModuleList()
        self.bottleneck_blocks.append(create_conv3_block_from_block_args(in_channels=current_num_channels + args.num_expression_params, out_channels=current_num_channels, block_args=args.block_args))
        for i in range(1, args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(ResnetBlock.create(num_channels=current_num_channels, is1x1=False, block_args=args.block_args))
        self.upsample_blocks = ModuleList()
        while current_image_size < args.image_size:
            next_image_size = current_image_size * 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.upsample_blocks.append(create_upsample_block_from_block_args(in_channels=current_num_channels, out_channels=next_num_channels, block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        self.iris_mouth_grid_change = self.create_grid_change_block()
        self.iris_mouth_color_change = self.create_color_change_block()
        self.iris_mouth_alpha = self.create_alpha_block()
        self.eye_color_change = self.create_color_change_block()
        self.eye_alpha = self.create_alpha_block()

    def create_alpha_block(self):
        return Sequential(create_conv3(in_channels=self.args.start_channels, out_channels=1, bias=True, initialization_method=self.args.block_args.initialization_method, use_spectral_norm=False), Sigmoid())

    def create_color_change_block(self):
        return Sequential(create_conv3_from_block_args(in_channels=self.args.start_channels, out_channels=self.args.image_channels, bias=True, block_args=self.args.block_args), Tanh())

    def create_grid_change_block(self):
        return create_conv3(in_channels=self.args.start_channels, out_channels=2, bias=False, initialization_method='zero', use_spectral_norm=False)

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // 2 ** level)

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Tensor) ->List[Tensor]:
        feature = image
        for block in self.downsample_blocks:
            feature = block(feature)
        n, c = pose.shape
        pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.bottleneck_image_size, self.args.bottleneck_image_size)
        feature = torch.cat([feature, pose], dim=1)
        for block in self.bottleneck_blocks:
            feature = block(feature)
        for block in self.upsample_blocks:
            feature = block(feature)
        iris_mouth_grid_change = self.iris_mouth_grid_change(feature)
        iris_mouth_image_0 = self.apply_grid_change(iris_mouth_grid_change, image)
        iris_mouth_color_change = self.iris_mouth_color_change(feature)
        iris_mouth_alpha = self.iris_mouth_alpha(feature)
        iris_mouth_image_1 = self.apply_color_change(iris_mouth_alpha, iris_mouth_color_change, iris_mouth_image_0)
        eye_color_change = self.eye_color_change(feature)
        eye_alpha = self.eye_alpha(feature)
        output_image = self.apply_color_change(eye_alpha, eye_color_change, iris_mouth_image_1.detach())
        return [output_image, eye_alpha, eye_color_change, iris_mouth_image_1, iris_mouth_alpha, iris_mouth_color_change, iris_mouth_image_0]

    def merge_down(self, top_layer: Tensor, bottom_layer: Tensor):
        top_layer_rgb = top_layer[:, 0:3, :, :]
        top_layer_a = top_layer[:, 3:4, :, :]
        return bottom_layer * (1 - top_layer_a) + torch.cat([top_layer_rgb * top_layer_a, top_layer_a], dim=1)

    def apply_grid_change(self, grid_change, image: Tensor) ->Tensor:
        n, c, h, w = image.shape
        device = grid_change.device
        grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
        grid = base_grid + grid_change
        resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return resampled_image

    def apply_color_change(self, alpha, color_change, image: Tensor) ->Tensor:
        return color_change * alpha + image * (1 - alpha)

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0], batch[1])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bias2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderDecoderModule,
     lambda: ([], {'image_size': 4, 'image_channels': 4, 'output_channels': 4, 'bottleneck_image_size': 4, 'bottleneck_block_count': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PassThrough,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNormalization,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBlock,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ViewImageAsVector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ViewVectorAsMultiChannelImage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ViewVectorAsOneChannelImage,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_pkhungurn_talking_head_anime_2_demo(_paritybench_base):
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

