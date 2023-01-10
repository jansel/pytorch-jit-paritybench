import sys
_module = sys.modules[__name__]
del sys
app = _module
manual_poser = _module
puppeteer = _module
nn = _module
conv = _module
encoder_decoder_module = _module
init_function = _module
resnet_block = _module
u_net_module = _module
poser = _module
morph_rotate_combine_poser = _module
poser = _module
puppet = _module
head_pose_solver = _module
util = _module
tha = _module
batch_input_module = _module
combiner = _module
face_morpher = _module
two_algo_face_rotator = _module
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


import numpy


import torch


import numpy as np


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Sequential


from torch.nn import InstanceNorm2d


from torch.nn import ReLU


from torch.nn import ConvTranspose2d


from torch.nn import ModuleList


from torch.nn.init import kaiming_normal_


from torch.nn.init import xavier_normal_


from torch import relu


from typing import List


from torch import Tensor


import abc


from abc import ABC


from torch.nn import Sigmoid


from torch.nn import Tanh


from torch.nn.functional import affine_grid


from torch.nn.functional import grid_sample


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
        else:
            raise ('Invalid initialization method %s' % method)
    return init


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


class BatchInputModule(Module, ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward_from_batch(self, batch):
        pass


class Combiner(BatchInputModule):

    def __init__(self, image_size: int=256, image_channels: int=4, pose_size: int=3, intermediate_channels: int=64, bottleneck_image_size: int=32, bottleneck_block_count: int=6, initialization_method: str='he'):
        super().__init__()
        self.main_body = UNetModule(image_size=image_size, image_channels=2 * image_channels + pose_size, output_channels=intermediate_channels, bottleneck_image_size=bottleneck_image_size, bottleneck_block_count=bottleneck_block_count, initialization_method=initialization_method)
        self.combine_alpha_mask = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Sigmoid())
        self.retouch_alpha_mask = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Sigmoid())
        self.retouch_color_change = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Tanh())

    def forward(self, first_image: Tensor, second_image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), first_image.size(2), first_image.size(3))
        x = torch.cat([first_image, second_image, pose], dim=1)
        y = self.main_body(x)
        combine_alpha_mask = self.combine_alpha_mask(y)
        combined_image = combine_alpha_mask * first_image + (1 - combine_alpha_mask) * second_image
        retouch_alpha_mask = self.retouch_alpha_mask(y)
        retouch_color_change = self.retouch_color_change(y)
        final_image = retouch_alpha_mask * combined_image + (1 - retouch_alpha_mask) * retouch_color_change
        return [final_image, combined_image, combine_alpha_mask, retouch_alpha_mask, retouch_color_change]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2])


class FaceMorpher(BatchInputModule):

    def __init__(self, image_size: int=256, image_channels: int=4, pose_size: int=3, intermediate_channels: int=64, bottleneck_image_size: int=32, bottleneck_block_count: int=6, initialization_method: str='he'):
        super().__init__()
        self.main_body = EncoderDecoderModule(image_size=image_size, image_channels=image_channels + pose_size, output_channels=intermediate_channels, bottleneck_image_size=bottleneck_image_size, bottleneck_block_count=bottleneck_block_count, initialization_method=initialization_method)
        self.color_change = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Tanh())
        self.alpha_mask = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Sigmoid())

    def forward(self, image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        y = self.main_body(x)
        color = self.color_change(y)
        alpha = self.alpha_mask(y)
        output_image = alpha * image + (1 - alpha) * color
        return [output_image, alpha, color]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class TwoAlgoFaceRotator(BatchInputModule):

    def __init__(self, image_size: int=256, image_channels: int=4, pose_size: int=3, intermediate_channels: int=64, bottleneck_image_size: int=32, bottleneck_block_count: int=6, initialization_method: str='he', align_corners: bool=True):
        super().__init__()
        self.align_corners = align_corners
        self.main_body = EncoderDecoderModule(image_size=image_size, image_channels=image_channels + pose_size, output_channels=intermediate_channels, bottleneck_image_size=bottleneck_image_size, bottleneck_block_count=bottleneck_block_count, initialization_method=initialization_method)
        self.pumarola_color_change = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Tanh())
        self.pumarola_alpha_mask = Sequential(Conv7(intermediate_channels, image_channels, initialization_method), Sigmoid())
        self.zhou_grid_change = Conv7(intermediate_channels, 2, initialization_method)

    def forward(self, image: Tensor, pose: Tensor):
        n = image.size(0)
        c = image.size(1)
        h = image.size(2)
        w = image.size(3)
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        y = self.main_body(x)
        color_change = self.pumarola_color_change(y)
        alpha_mask = self.pumarola_alpha_mask(y)
        color_changed = alpha_mask * image + (1 - alpha_mask) * color_change
        grid_change = torch.transpose(self.zhou_grid_change(y).view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        device = self.zhou_grid_change.weight.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=self.align_corners)
        grid = base_grid + grid_change
        resampled = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=self.align_corners)
        return [color_changed, resampled, color_change, alpha_mask, grid_change, grid]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderDecoderModule,
     lambda: ([], {'image_size': 4, 'image_channels': 4, 'output_channels': 4, 'bottleneck_image_size': 4, 'bottleneck_block_count': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBlock,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pkhungurn_talking_head_anime_demo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

