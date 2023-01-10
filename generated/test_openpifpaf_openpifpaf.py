import sys
_module = sys.modules[__name__]
del sys
artwork = _module
download_data = _module
nb_cell_tags = _module
setup = _module
openpifpaf = _module
_version = _module
annotation = _module
benchmark = _module
configurable = _module
conftest = _module
count_ops = _module
cpp_extension = _module
datasets = _module
collate = _module
factory = _module
image_list = _module
loader_with_reset = _module
module = _module
multiloader = _module
multimodule = _module
torch_dataset = _module
decoder = _module
cifcaf = _module
cifdet = _module
decoder = _module
multi = _module
pose_distance = _module
crafted = _module
euclidean = _module
oks = _module
pose_similarity = _module
track_annotation = _module
track_base = _module
tracking_pose = _module
utils = _module
nms = _module
encoder = _module
annrescaler = _module
caf = _module
cif = _module
cifdet = _module
single_image = _module
tcaf = _module
eval = _module
export_coreml = _module
export_onnx = _module
export_torchscript = _module
headmeta = _module
logger = _module
logs = _module
metric = _module
base = _module
classification = _module
coco = _module
migrate = _module
network = _module
basenetworks = _module
factory = _module
heads = _module
losses = _module
components = _module
composite = _module
multi_head = _module
model_migration = _module
nets = _module
running_cache = _module
tracking_base = _module
tracking_heads = _module
trainer = _module
optimize = _module
plugin = _module
plugins = _module
animalpose = _module
animal_kp = _module
constants = _module
voc_to_coco = _module
apollocar3d = _module
apollo_kp = _module
apollo_to_coco = _module
metrics = _module
transforms = _module
cifar10 = _module
basenet = _module
datamodule = _module
cocodet = _module
cocokp = _module
dataset = _module
crowdpose = _module
module = _module
nuscenes = _module
nuscenes = _module
posetrack = _module
cocokpst = _module
datasets = _module
draw_poses = _module
image_to_tracking = _module
normalize_transform = _module
posetrack2017 = _module
posetrack2018 = _module
wholebody = _module
Get_annotations_from_coco_wholebody = _module
Visualize_annotations = _module
wholebody = _module
wholebody_metric = _module
predict = _module
predictor = _module
profiler = _module
show = _module
animation_frame = _module
annotation_painter = _module
canvas = _module
cli = _module
fields = _module
painters = _module
signal = _module
stream = _module
train = _module
annotations = _module
assertion = _module
compose = _module
crop = _module
deinterlace = _module
encoders = _module
hflip = _module
image = _module
impute = _module
minsize = _module
multi_scale = _module
pad = _module
pair = _module
blank_past = _module
camera_shift = _module
crop = _module
pad = _module
sample_pairing = _module
single_image = _module
preprocess = _module
random = _module
rotate = _module
scale = _module
toannotations = _module
unclipped = _module
video = _module
visualizer = _module
cifhr = _module
multi_tracking = _module
occupancy = _module
seeds = _module
openpifpaf_testplugin = _module
test_clis = _module
test_cmake = _module
test_coreml_export = _module
test_forward = _module
test_help = _module
test_image_scale = _module
test_input_processing = _module
test_localization = _module
test_multiprocessing = _module
test_network = _module
test_onnx_export = _module
test_plugin = _module
test_scale_loss = _module
test_torchscript = _module
test_train = _module
test_transforms = _module
test_weighted_cif = _module
versioneer = _module

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


import torch.utils.cpp_extension


import torch


import logging


from typing import List


from typing import Optional


import torch.utils.data


import numpy as np


import time


from collections import defaultdict


import torchvision


from typing import ClassVar


from typing import Tuple


import torchvision.models


from typing import Callable


from typing import Dict


from typing import Set


from typing import Type


import warnings


import functools


import math


import typing as t


import copy


import random


import torchvision.transforms.functional


import itertools


class DecoderModule(torch.nn.Module):

    def __init__(self, cif_meta, caf_meta):
        super().__init__()
        self.cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(len(cif_meta.keypoints), torch.LongTensor(caf_meta.skeleton) - 1)
        self.cif_stride = cif_meta.stride
        self.caf_stride = caf_meta.stride

    def forward(self, cif_field, caf_field):
        return self.cpp_decoder.call(cif_field, self.cif_stride, caf_field, self.caf_stride)


class EncoderDecoder(torch.nn.Module):

    def __init__(self, traced_encoder, decoder):
        super().__init__()
        self.traced_encoder = traced_encoder
        self.decoder = decoder

    def forward(self, x):
        cif_head_batch, caf_head_batch = self.traced_encoder(x)
        o = [self.decoder(cif_head, caf_head) for cif_head, caf_head in zip(cif_head_batch, caf_head_batch)]
        return o


LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network.

    :param name: a short name for the base network, e.g. resnet50
    :param stride: total stride from input to output
    :param out_features: number of output features
    """

    def __init__(self, name: str, *, stride: int, out_features: int):
        super().__init__()
        self.name = name
        self.stride = stride
        self.out_features = out_features
        LOG.info('%s: stride = %d, output features = %d', name, stride, out_features)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, x):
        raise NotImplementedError


class ShuffleNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_shufflenetv2, out_features=2048):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_shufflenetv2(self.pretrained)
        self.conv1 = base_vision.conv1
        self.stage2 = base_vision.stage2
        self.stage3 = base_vision.stage3
        self.stage4 = base_vision.stage4
        self.conv5 = base_vision.conv5

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('ShuffleNetv2')
        assert cls.pretrained
        group.add_argument('--shufflenetv2-no-pretrain', dest='shufflenetv2_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.shufflenetv2_pretrained


class Resnet(BaseNetwork):
    pretrained = True
    pool0_stride = 0
    input_conv_stride = 2
    input_conv2_stride = 0
    remove_last_block = False
    block5_dilation = 1

    def __init__(self, name, torchvision_resnet, out_features=2048):
        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32
        input_modules = modules[:4]
        if self.pool0_stride:
            if self.pool0_stride != 2:
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2
        if self.input_conv_stride != 2:
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)
        if self.input_conv2_stride:
            assert not self.pool0_stride
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False), torch.nn.BatchNorm2d(channels), torch.nn.ReLU(inplace=True))
            input_modules.append(conv2)
            stride *= 2
            LOG.debug('replaced max pool with [3x3 conv, bn, relu] with %d channels', channels)
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2
        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue
                m.stride = torch.nn.modules.utils._pair(1)
                if m.kernel_size[0] == 1:
                    continue
                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation
                m.padding = torch.nn.modules.utils._pair(padding)
        super().__init__(name, stride=stride, out_features=out_features)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, x):
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('ResNet')
        assert cls.pretrained
        group.add_argument('--resnet-no-pretrain', dest='resnet_pretrained', default=True, action='store_false', help='use randomly initialized models')
        group.add_argument('--resnet-pool0-stride', default=cls.pool0_stride, type=int, help='stride of zero removes the pooling op')
        group.add_argument('--resnet-input-conv-stride', default=cls.input_conv_stride, type=int, help='stride of the input convolution')
        group.add_argument('--resnet-input-conv2-stride', default=cls.input_conv2_stride, type=int, help='stride of the optional 2nd input convolution')
        group.add_argument('--resnet-block5-dilation', default=cls.block5_dilation, type=int, help='use dilated convs in block5')
        assert not cls.remove_last_block
        group.add_argument('--resnet-remove-last-block', default=False, action='store_true', help='create a network without the last block')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.resnet_pretrained
        cls.pool0_stride = args.resnet_pool0_stride
        cls.input_conv_stride = args.resnet_input_conv_stride
        cls.input_conv2_stride = args.resnet_input_conv2_stride
        cls.block5_dilation = args.resnet_block5_dilation
        cls.remove_last_block = args.resnet_remove_last_block


class InvertedResidualK(torch.nn.Module):
    """Based on torchvision.models.shufflenet.InvertedResidual."""

    def __init__(self, inp, oup, first_in_stage, *, stride=1, layer_norm, non_linearity, dilation=1, kernel_size=3):
        super().__init__()
        assert (stride != 1 or dilation != 1 or inp != oup) or not first_in_stage
        LOG.debug('InvResK: %d %d %s, stride=%d, dilation=%d', inp, oup, first_in_stage, stride, dilation)
        self.first_in_stage = first_in_stage
        branch_features = oup // 2
        padding = (kernel_size - 1) // 2 * dilation
        self.branch1 = None
        if self.first_in_stage:
            self.branch1 = torch.nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), layer_norm(inp), torch.nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity())
        self.branch2 = torch.nn.Sequential(torch.nn.Conv2d(inp if first_in_stage else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity(), self.depthwise_conv(branch_features, branch_features, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), layer_norm(branch_features), torch.nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), non_linearity())

    @staticmethod
    def depthwise_conv(in_f, out_f, kernel_size, stride=1, padding=0, bias=False, dilation=1):
        return torch.nn.Conv2d(in_f, out_f, kernel_size, stride, padding, bias=bias, groups=in_f, dilation=dilation)

    def forward(self, x):
        if self.branch1 is None:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = torchvision.models.shufflenetv2.channel_shuffle(out, 2)
        return out


class ShuffleNetV2K(BaseNetwork):
    """Based on torchvision.models.ShuffleNetV2 where
    the kernel size in stages 2,3,4 is 5 instead of 3."""
    input_conv2_stride = 0
    input_conv2_outchannels = None
    layer_norm = None
    stage4_dilation = 1
    kernel_width = 5
    conv5_as_stage = False
    non_linearity = None

    def __init__(self, name, stages_repeats, stages_out_channels):
        layer_norm = ShuffleNetV2K.layer_norm
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d
        non_linearity = ShuffleNetV2K.non_linearity
        if non_linearity is None:
            non_linearity = lambda : torch.nn.ReLU(inplace=True)
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        _stage_out_channels = stages_out_channels
        stride = 16
        input_modules = []
        input_channels = 3
        output_channels = _stage_out_channels[0]
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), layer_norm(output_channels), non_linearity())
        input_modules.append(conv1)
        input_channels = output_channels
        if self.input_conv2_stride:
            output_channels = self.input_conv2_outchannels or input_channels
            conv2 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), layer_norm(output_channels), non_linearity())
            input_modules.append(conv2)
            stride *= 2
            input_channels = output_channels
            LOG.debug('replaced max pool with [3x3 conv, bn, relu]')
        stages = []
        for repeats, output_channels, dilation in zip(stages_repeats, _stage_out_channels[1:], [1, 1, self.stage4_dilation]):
            stage_stride = 2 if dilation == 1 else 1
            stride = int(stride * stage_stride / 2)
            seq = [InvertedResidualK(input_channels, output_channels, True, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=dilation, stride=stage_stride)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, False, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=dilation))
            stages.append(torch.nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = _stage_out_channels[-1]
        if self.conv5_as_stage:
            use_first_in_stage = input_channels != output_channels
            conv5 = torch.nn.Sequential(InvertedResidualK(input_channels, output_channels, use_first_in_stage, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=self.stage4_dilation), InvertedResidualK(output_channels, output_channels, False, kernel_size=self.kernel_width, layer_norm=layer_norm, non_linearity=non_linearity, dilation=self.stage4_dilation))
        else:
            conv5 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), layer_norm(output_channels), non_linearity())
        super().__init__(name, stride=stride, out_features=output_channels)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]
        self.conv5 = conv5

    def forward(self, x):
        x = self.input_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('shufflenetv2k')
        group.add_argument('--shufflenetv2k-input-conv2-stride', default=cls.input_conv2_stride, type=int, help='stride of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-input-conv2-outchannels', default=cls.input_conv2_outchannels, type=int, help='out channels of the optional 2nd input convolution')
        group.add_argument('--shufflenetv2k-stage4-dilation', default=cls.stage4_dilation, type=int, help='dilation factor of stage 4')
        group.add_argument('--shufflenetv2k-kernel', default=cls.kernel_width, type=int, help='kernel width')
        assert not cls.conv5_as_stage
        group.add_argument('--shufflenetv2k-conv5-as-stage', default=False, action='store_true')
        layer_norm_group = group.add_mutually_exclusive_group()
        layer_norm_group.add_argument('--shufflenetv2k-instance-norm', default=False, action='store_true')
        layer_norm_group.add_argument('--shufflenetv2k-group-norm', default=False, action='store_true')
        non_linearity_group = group.add_mutually_exclusive_group()
        non_linearity_group.add_argument('--shufflenetv2k-leaky-relu', default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.input_conv2_stride = args.shufflenetv2k_input_conv2_stride
        cls.input_conv2_outchannels = args.shufflenetv2k_input_conv2_outchannels
        cls.stage4_dilation = args.shufflenetv2k_stage4_dilation
        cls.kernel_width = args.shufflenetv2k_kernel
        cls.conv5_as_stage = args.shufflenetv2k_conv5_as_stage
        if args.shufflenetv2k_instance_norm:
            cls.layer_norm = lambda x: torch.nn.InstanceNorm2d(x, affine=True, track_running_stats=True)
        if args.shufflenetv2k_group_norm:
            cls.layer_norm = lambda x: torch.nn.GroupNorm((32 if x % 32 == 0 else 29) if x > 100 else 4, x)
        if args.shufflenetv2k_leaky_relu:
            cls.non_linearity = lambda : torch.nn.LeakyReLU(inplace=True)


class MobileNetV2(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_mobilenetv2, out_features=1280):
        super().__init__(name, stride=32, out_features=out_features)
        base_vision = torchvision_mobilenetv2(self.pretrained)
        self.backbone = list(base_vision.children())[0]

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('MobileNetV2')
        assert cls.pretrained
        group.add_argument('--mobilenetv2-no-pretrain', dest='mobilenetv2_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.mobilenetv2_pretrained


class MobileNetV3(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_mobilenetv3, out_features=960):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_mobilenetv3(self.pretrained)
        self.backbone = list(base_vision.children())[0]
        input_conv = list(self.backbone)[0][0]
        input_conv.stride = torch.nn.modules.utils._pair(1)

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('MobileNetV3')
        assert cls.pretrained
        group.add_argument('--mobilenetv3-no-pretrain', dest='mobilenetv3_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.mobilenetv3_pretrained


class SqueezeNet(BaseNetwork):
    pretrained = True

    def __init__(self, name, torchvision_squeezenet, out_features=512):
        super().__init__(name, stride=16, out_features=out_features)
        base_vision = torchvision_squeezenet(self.pretrained)
        for m in base_vision.modules():
            if isinstance(m, (torch.nn.MaxPool2d,)) and m.padding != 1:
                LOG.debug('adjusting maxpool2d padding to 1 from padding=%d, kernel=%d, stride=%d', m.padding, m.kernel_size, m.stride)
                m.padding = 1
            if isinstance(m, (torch.nn.Conv2d,)):
                target_padding = (m.kernel_size[0] - 1) // 2
                if m.padding[0] != target_padding:
                    LOG.debug('adjusting conv2d padding to %d (kernel=%d, padding=%d)', target_padding, m.kernel_size, m.padding)
                    m.padding = torch.nn.modules.utils._pair(target_padding)
        self.backbone = list(base_vision.children())[0]

    def forward(self, x):
        x = self.backbone(x)
        return x

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('SqueezeNet')
        assert cls.pretrained
        group.add_argument('--squeezenet-no-pretrain', dest='squeezenet_pretrained', default=True, action='store_false', help='use randomly initialized models')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pretrained = args.squeezenet_pretrained


class PifHFlip(torch.nn.Module):

    def __init__(self, keypoints, hflip):
        super().__init__()
        flip_indices = torch.LongTensor([(keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i) for kp_i, kp_name in enumerate(keypoints)])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, 0, :, :] *= -1.0
        return out


class PafHFlip(torch.nn.Module):

    def __init__(self, keypoints, skeleton, hflip):
        super().__init__()
        skeleton_names = [(keypoints[j1 - 1], keypoints[j2 - 1]) for j1, j2 in skeleton]
        flipped_skeleton_names = [(hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2) for j1, j2 in skeleton_names]
        LOG.debug('skeleton = %s, flipped_skeleton = %s', skeleton_names, flipped_skeleton_names)
        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)
        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, 0, :, :] *= -1.0
        out[2][:, :, 0, :, :] *= -1.0
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, paf_i])
            out[1][:, paf_i] = out[2][:, paf_i]
            out[2][:, paf_i] = cc
        return out


@functools.lru_cache(16)
def index_field_torch(shape: t.Tuple[int, int], device: torch.device, unsqueeze: t.Tuple[int, int]=(0, 0)) ->torch.Tensor:
    assert len(shape) == 2
    xy = torch.empty((2, shape[0], shape[1]), device=device)
    xy[0] = torch.arange(shape[1], device=device)
    xy[1] = torch.arange(shape[0], device=device).unsqueeze(1)
    for dim in unsqueeze:
        xy = torch.unsqueeze(xy, dim)
    return xy


class SoftClamp(torch.nn.Module):

    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        above_max = x > self.max_value
        x[above_max] = self.max_value + torch.log(1 - self.max_value + x[above_max])
        return x


class Bce(torch.nn.Module):
    background_weight = 1.0
    focal_alpha = 0.5
    focal_gamma = 1.0
    focal_detach = False
    focal_clamp = True
    soft_clamp_value = 5.0
    background_clamp = -15.0
    min_bce = 0.0

    def __init__(self, **kwargs):
        super().__init__()
        for n, v in kwargs.items():
            assert hasattr(self, n)
            setattr(self, n, v)
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Bce Loss')
        group.add_argument('--background-weight', default=cls.background_weight, type=float, help='BCE weight where ground truth is background')
        group.add_argument('--focal-alpha', default=cls.focal_alpha, type=float, help='scale parameter of focal loss')
        group.add_argument('--focal-gamma', default=cls.focal_gamma, type=float, help='use focal loss with the given gamma')
        assert not cls.focal_detach
        group.add_argument('--focal-detach', default=False, action='store_true')
        assert cls.focal_clamp
        group.add_argument('--no-focal-clamp', dest='focal_clamp', default=True, action='store_false')
        group.add_argument('--bce-min', default=cls.min_bce, type=float, help='gradient clipped below')
        group.add_argument('--bce-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for BCE')
        group.add_argument('--bce-background-clamp', default=cls.background_clamp, type=float, help='background clamp for BCE')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.background_weight = args.background_weight
        cls.focal_alpha = args.focal_alpha
        cls.focal_gamma = args.focal_gamma
        cls.focal_detach = args.focal_detach
        cls.focal_clamp = args.focal_clamp
        cls.min_bce = args.bce_min
        cls.soft_clamp_value = args.bce_soft_clamp
        cls.background_clamp = args.bce_background_clamp

    def forward(self, x, t):
        t_zeroone = t.clone()
        t_zeroone[t_zeroone > 0.0] = 1.0
        if self.background_clamp is not None:
            bg_clamp_mask = (t_zeroone == 0.0) & (x < self.background_clamp)
            x[bg_clamp_mask] = self.background_clamp
        bce = torch.nn.functional.binary_cross_entropy_with_logits(x, t_zeroone, reduction='none')
        if self.soft_clamp is not None:
            bce = self.soft_clamp(bce)
        if self.min_bce > 0.0:
            torch.clamp_min_(bce, self.min_bce)
        if self.focal_gamma != 0.0:
            p = torch.sigmoid(x)
            pt = p * t_zeroone + (1 - p) * (1 - t_zeroone)
            if self.focal_clamp and self.min_bce > 0.0:
                pt_threshold = math.exp(-self.min_bce)
                torch.clamp_max_(pt, pt_threshold)
            focal = 1.0 - pt
            if self.focal_gamma != 1.0:
                focal = (focal + 0.0001) ** self.focal_gamma
            if self.focal_detach:
                focal = focal.detach()
            bce = focal * bce
        if self.focal_alpha == 0.5:
            bce = 0.5 * bce
        elif self.focal_alpha >= 0.0:
            alphat = self.focal_alpha * t_zeroone + (1 - self.focal_alpha) * (1 - t_zeroone)
            bce = alphat * bce
        weight_mask = t_zeroone != t
        bce[weight_mask] = bce[weight_mask] * t[weight_mask]
        if self.background_weight != 1.0:
            bg_weight = torch.ones_like(t, requires_grad=False)
            bg_weight[t == 0] *= self.background_weight
            bce = bce * bg_weight
        return bce


class BceDistance(Bce):

    def forward(self, x, t):
        t_sign = t.clone()
        t_sign[t > 0.0] = 1.0
        t_sign[t <= 0.0] = -1.0
        x_detached = x.detach()
        focal_loss_modification = 1.0
        p_bar = 1.0 / (1.0 + torch.exp(t_sign * x_detached))
        if self.focal_alpha:
            focal_loss_modification *= self.focal_alpha
        if self.focal_gamma == 1.0:
            p = 1.0 - p_bar
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)
            focal_loss_modification = focal_loss_modification * (p_bar + p * neg_ln_p)
        elif self.focal_gamma > 0.0:
            p = 1.0 - p_bar
            neg_ln_p = torch.nn.functional.softplus(-t_sign * x_detached)
            focal_loss_modification = focal_loss_modification * (p_bar ** self.focal_gamma + self.focal_gamma * p_bar ** (self.focal_gamma - 1.0) * p * neg_ln_p)
        elif self.focal_gamma == 0.0:
            pass
        else:
            raise NotImplementedError
        target = x_detached + t_sign * p_bar * focal_loss_modification
        d = x - target
        if self.background_clamp:
            d[(x_detached < self.background_clamp) & (t_sign == -1.0)] = 0.0
        return d


class BceL2(BceDistance):

    def forward(self, x, t):
        d = super().forward(x, t)
        return torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')


class Scale(torch.nn.Module):
    b = 1.0
    log_space = False
    relative = True
    relative_eps = 0.1
    clip = None
    soft_clamp_value = 5.0

    def __init__(self):
        super().__init__()
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Scale Loss')
        group.add_argument('--b-scale', default=cls.b, type=float, help='Laplace width b for scale loss')
        assert not cls.log_space
        group.add_argument('--scale-log', default=False, action='store_true')
        group.add_argument('--scale-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for scale')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.b = args.b_scale
        cls.log_space = args.scale_log
        if args.scale_log:
            cls.relative = False
        cls.soft_clamp_value = args.scale_soft_clamp

    def forward(self, x, t):
        assert not (self.log_space and self.relative)
        x = torch.nn.functional.softplus(x)
        d = torch.nn.functional.l1_loss(x if not self.log_space else torch.log(x), t if not self.log_space else torch.log(t), reduction='none')
        if self.clip is not None:
            d = torch.clamp(d, self.clip[0], self.clip[1])
        denominator = self.b
        if self.relative:
            denominator = self.b * (self.relative_eps + t)
        d = d / denominator
        if self.soft_clamp is not None:
            d = self.soft_clamp(d)
        loss = torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d), reduction='none')
        return loss


class ScaleDistance(Scale):

    def forward(self, x, t):
        x = torch.nn.functional.softplus(x)
        d = 1.0 / self.b * (x - t)
        d[torch.isnan(d)] = 0.0
        return d


class Laplace(torch.nn.Module):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """
    weight = None
    norm_clip = None
    soft_clamp_value = 5.0

    def __init__(self):
        super().__init__()
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = SoftClamp(self.soft_clamp_value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Laplace Loss')
        group.add_argument('--laplace-soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp for Laplace')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.laplace_soft_clamp

    def forward(self, x1, x2, logb, t1, t2, bmin):
        norm = torch.stack((x1 - t1, x2 - t2, bmin)).norm(dim=0)
        if self.norm_clip is not None:
            norm = torch.clamp(norm, self.norm_clip[0], self.norm_clip[1])
        logb = 3.0 * torch.tanh(logb / 3.0)
        scaled_norm = norm * torch.exp(-logb)
        if self.soft_clamp is not None:
            scaled_norm = self.soft_clamp(scaled_norm)
        losses = logb + scaled_norm
        if self.weight is not None:
            losses = losses * self.weight
        return losses


class CompositeLossByComponent(torch.nn.Module):
    """Default loss until v0.12"""
    prescale = 1.0
    regression_loss: Callable = components.Laplace()
    bce_total_soft_clamp = None

    def __init__(self, head_meta):
        super().__init__()
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales
        LOG.debug('%s: n_vectors = %d, n_scales = %d', head_meta.name, self.n_vectors, self.n_scales)
        self.confidence_loss = components.Bce()
        self.scale_losses = torch.nn.ModuleList([components.Scale() for _ in range(self.n_scales)])
        self.field_names = ['{}.{}.c'.format(head_meta.dataset, head_meta.name)] + ['{}.{}.vec{}'.format(head_meta.dataset, head_meta.name, i + 1) for i in range(self.n_vectors)] + ['{}.{}.scales{}'.format(head_meta.dataset, head_meta.name, i + 1) for i in range(self.n_scales)]
        w = head_meta.training_weights
        self.weights = None
        if w is not None:
            self.weights = torch.ones([1, head_meta.n_fields, 1, 1], requires_grad=False)
            self.weights[0, :, 0, 0] = torch.Tensor(w)
        LOG.debug('The weights for the keypoints are %s', self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss by Components')
        group.add_argument('--loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--regression-loss', default='laplace', choices=['smoothl1', 'l1', 'laplace'], help='type of regression loss')
        group.add_argument('--bce-total-soft-clamp', default=cls.bce_total_soft_clamp, type=float, help='per feature clamp value applied to the total')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale
        if args.regression_loss == 'smoothl1':
            cls.regression_loss = components.SmoothL1()
        elif args.regression_loss == 'l1':
            cls.regression_loss = staticmethod(components.l1_loss)
        elif args.regression_loss == 'laplace':
            cls.regression_loss = components.Laplace()
        elif args.regression_loss is None:
            cls.regression_loss = components.Laplace()
        else:
            raise Exception('unknown regression loss type {}'.format(args.regression_loss))
        cls.bce_total_soft_clamp = args.bce_total_soft_clamp

    def _confidence_loss(self, x_confidence, t_confidence):
        x_confidence = x_confidence[:, :, 0]
        t_confidence = t_confidence[:, :, 0]
        bce_masks = torch.isnan(t_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None
        batch_size = t_confidence.shape[0]
        n_fields = t_confidence.shape[1]
        n_features = t_confidence.numel()
        LOG.debug('batch size = %d, n fields = %d, n_features = %d', batch_size, n_fields, n_features)
        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            t_confidence = t_confidence[:, self.bce_blackout]
        LOG.debug('BCE: x = %s, target = %s, mask = %s', x_confidence.shape, t_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(t_confidence, bce_masks)
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        ce_loss = self.confidence_loss(x_confidence, bce_target)
        if self.prescale != 1.0:
            ce_loss = ce_loss * self.prescale
        if self.weights is not None:
            weight = torch.ones_like(t_confidence, requires_grad=False)
            weight[:] = self.weights
            weight = torch.masked_select(weight, bce_masks)
            ce_loss = ce_loss * weight
        ce_loss = ce_loss.sum()
        if self.bce_total_soft_clamp is not None:
            total_clamp_value = self.bce_total_soft_clamp * n_features / n_fields
            LOG.debug('summed ce loss = %s, soft clamp = %f', ce_loss, total_clamp_value)
            ce_loss = components.SoftClamp(total_clamp_value)(ce_loss)
        ce_loss = ce_loss / batch_size
        return ce_loss

    def _localization_loss(self, x_regs, t_regs):
        assert x_regs.shape[2] == self.n_vectors * 3
        assert t_regs.shape[2] == self.n_vectors * 3
        batch_size = t_regs.shape[0]
        reg_losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_regs[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i in range(self.n_vectors):
            reg_masks = torch.isnan(t_regs[:, :, i * 2]).bitwise_not_()
            loss = self.regression_loss(torch.masked_select(x_regs[:, :, i * 2 + 0], reg_masks), torch.masked_select(x_regs[:, :, i * 2 + 1], reg_masks), torch.masked_select(x_regs[:, :, self.n_vectors * 2 + i], reg_masks), torch.masked_select(t_regs[:, :, i * 2 + 0], reg_masks), torch.masked_select(t_regs[:, :, i * 2 + 1], reg_masks), torch.masked_select(t_regs[:, :, self.n_vectors * 2 + i], reg_masks))
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, reg_masks)
            reg_losses.append(loss.sum() / batch_size)
        return reg_losses

    def _scale_losses(self, x_scales, t_scales):
        assert x_scales.shape[2] == t_scales.shape[2] == len(self.scale_losses)
        batch_size = t_scales.shape[0]
        losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_scales[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i, sl in enumerate(self.scale_losses):
            mask = torch.isnan(t_scales[:, :, i]).bitwise_not_()
            loss = sl(torch.masked_select(x_scales[:, :, i], mask), torch.masked_select(t_scales[:, :, i], mask))
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, mask)
            losses.append(loss.sum() / batch_size)
        return losses

    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)
        if t is None:
            return [None for _ in range(1 + self.n_vectors + self.n_scales)]
        assert x.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        assert t.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        x_confidence = x[:, :, 0:1]
        x_regs = x[:, :, 1:1 + self.n_vectors * 3]
        x_scales = x[:, :, 1 + self.n_vectors * 3:]
        t_confidence = t[:, :, 0:1]
        t_regs = t[:, :, 1:1 + self.n_vectors * 3]
        t_scales = t[:, :, 1 + self.n_vectors * 3:]
        ce_loss = self._confidence_loss(x_confidence, t_confidence)
        reg_losses = self._localization_loss(x_regs, t_regs)
        scale_losses = self._scale_losses(x_scales, t_scales)
        all_losses = [ce_loss] + reg_losses + scale_losses
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'.format(all_losses, self.previous_losses))
        self.previous_losses = [(float(l.item()) if l is not None else None) for l in all_losses]
        return all_losses


class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""
    soft_clamp_value = 5.0

    def __init__(self, head_meta):
        super().__init__()
        self.n_confidences = head_meta.n_confidences
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales
        LOG.debug('%s: n_vectors = %d, n_scales = %d', head_meta.name, self.n_vectors, self.n_scales)
        self.field_names = '{}.{}.c'.format(head_meta.dataset, head_meta.name), '{}.{}.vec'.format(head_meta.dataset, head_meta.name), '{}.{}.scales'.format(head_meta.dataset, head_meta.name)
        self.bce_loss = components.BceL2()
        self.reg_loss = components.RegressionLoss()
        self.scale_loss = components.Scale()
        self.soft_clamp = None
        if self.soft_clamp_value:
            self.soft_clamp = components.SoftClamp(self.soft_clamp_value)
        self.weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            self.weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)
        LOG.debug('The weights for the keypoints are %s', self.weights)
        self.bce_blackout = None
        self.previous_losses = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Composite Loss')
        group.add_argument('--soft-clamp', default=cls.soft_clamp_value, type=float, help='soft clamp')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.soft_clamp_value = args.soft_clamp

    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)
        if t is None:
            return [None, None, None]
        assert x.shape[2] == 1 + self.n_confidences + self.n_vectors * 2 + self.n_scales
        assert t.shape[2] == self.n_confidences + self.n_vectors * 3 + self.n_scales
        t = torch.transpose(t, 2, 4)
        finite = torch.isfinite(t)
        t_confidence_raw = t[:, :, :, :, 0:self.n_confidences]
        bg_mask = torch.all(t_confidence_raw == 0.0, dim=4)
        c_mask = torch.all(t_confidence_raw > 0.0, dim=4)
        reg_mask = torch.all(finite[:, :, :, :, self.n_confidences:1 + self.n_vectors * 2], dim=4)
        scale_mask = torch.all(finite[:, :, :, :, self.n_confidences + self.n_vectors * 3:], dim=4)
        t_confidence_bg = t[bg_mask][:, 0:self.n_confidences]
        t_confidence = t[c_mask][:, 0:self.n_confidences]
        t_regs = t[reg_mask][:, self.n_confidences:1 + self.n_vectors * 2]
        t_sigma_min = t[reg_mask][:, self.n_confidences + self.n_vectors * 2:self.n_confidences + self.n_vectors * 3]
        t_scales_reg = t[reg_mask][:, self.n_confidences + self.n_vectors * 3:]
        t_scales = t[scale_mask][:, self.n_confidences + self.n_vectors * 3:]
        x = torch.transpose(x, 2, 4)
        x_confidence_bg = x[bg_mask][:, 1:1 + self.n_confidences]
        x_logs2_c = x[c_mask][:, 0:1]
        x_confidence = x[c_mask][:, 1:1 + self.n_confidences]
        x_logs2_reg = x[reg_mask][:, 0:1]
        x_regs = x[reg_mask][:, 1 + self.n_confidences:1 + self.n_confidences + self.n_vectors * 2]
        x_scales_reg = x[reg_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        x_scales = x[scale_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        t_scales_reg = t_scales_reg.clone()
        invalid_t_scales_reg = torch.isnan(t_scales_reg)
        t_scales_reg[invalid_t_scales_reg] = torch.nn.functional.softplus(x_scales_reg.detach()[invalid_t_scales_reg])
        l_confidence_bg = self.bce_loss(x_confidence_bg, t_confidence_bg)
        l_confidence = self.bce_loss(x_confidence, t_confidence)
        l_reg = self.reg_loss(x_regs, t_regs, t_sigma_min, t_scales_reg)
        l_scale = self.scale_loss(x_scales, t_scales)
        if self.soft_clamp is not None:
            l_confidence_bg = self.soft_clamp(l_confidence_bg)
            l_confidence = self.soft_clamp(l_confidence)
            l_reg = self.soft_clamp(l_reg)
            l_scale = self.soft_clamp(l_scale)
        x_logs2_c = 3.0 * torch.tanh(x_logs2_c / 3.0)
        l_confidence = 0.5 * l_confidence * torch.exp(-x_logs2_c) + 0.5 * x_logs2_c
        x_logs2_reg = 3.0 * torch.tanh(x_logs2_reg / 3.0)
        x_logb = 0.5 * x_logs2_reg + 0.69314
        reg_factor = torch.exp(-x_logb)
        x_logb = x_logb.unsqueeze(1)
        reg_factor = reg_factor.unsqueeze(1)
        if self.n_vectors > 1:
            x_logb = torch.repeat_interleave(x_logb, self.n_vectors, 1)
            reg_factor = torch.repeat_interleave(reg_factor, self.n_vectors, 1)
        l_reg = l_reg * reg_factor + x_logb
        if self.weights is not None:
            full_weights = torch.empty_like(t_confidence_raw)
            full_weights[:] = self.weights
            l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
            l_confidence = full_weights[c_mask] * l_confidence
            l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
            l_scale = full_weights[scale_mask] * l_scale
        batch_size = t.shape[0]
        losses = [(torch.sum(l_confidence_bg) + torch.sum(l_confidence)) / batch_size, torch.sum(l_reg) / batch_size, torch.sum(l_scale) / batch_size]
        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'.format(losses, self.previous_losses))
        self.previous_losses = [(float(l.item()) if l is not None else None) for l in losses]
        return losses


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.field_names = [n for l in self.losses for n in l.field_names]
        assert len(self.field_names) == len(self.lambdas)
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [(lam * l) for lam, l in zip(self.lambdas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneKendall(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None, tune=None):
        """Auto-tuning multi-head loss.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        Individual losses must not be negative for Kendall's prescription.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.tune = tune
        self.log_sigmas = torch.nn.Parameter(torch.zeros((len(lambdas),), dtype=torch.float64), requires_grad=True)
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)
        if self.tune is None:

            def tune_from_name(name):
                if '.vec' in name:
                    return 'none'
                if '.scale' in name:
                    return 'laplace'
                return 'gauss'
            self.tune = [tune_from_name(n) for l in self.losses for n in l.field_names]
        LOG.info('tune config: %s', self.tune)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, head_fields, head_targets):
        LOG.debug('losses = %d, fields = %d, targets = %d', len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.log_sigmas) == len(flat_head_losses)
        constrained_log_sigmas = 3.0 * torch.tanh(self.log_sigmas / 3.0)

        def tuned_loss(tune, log_sigma, loss):
            if tune == 'none':
                return loss
            if tune == 'laplace':
                return 0.694 + log_sigma + loss * torch.exp(-log_sigma)
            if tune == 'gauss':
                return 0.919 + log_sigma + loss * 0.5 * torch.exp(-2.0 * log_sigma)
            raise Exception('unknown tune: {}'.format(tune))
        loss_values = [(lam * tuned_loss(t, log_sigma, l)) for lam, t, log_sigma, l in zip(self.lambdas, self.tune, constrained_log_sigmas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(param.abs().max(dim=1)[0].clamp(min=1e-06).sum() for param in self.sparse_task_parameters)
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss
        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneVariance(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head loss based on loss-variance.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.epsilons = torch.ones((len(lambdas),), dtype=torch.float64)
        self.buffer = torch.full((len(lambdas), 53), float('nan'), dtype=torch.float64)
        self.buffer_index = -1
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.epsilons)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.epsilons]}

    def forward(self, head_fields, head_targets):
        LOG.debug('losses = %d, fields = %d, targets = %d', len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        self.buffer_index = (self.buffer_index + 1) % self.buffer.shape[1]
        for i, ll in enumerate(flat_head_losses):
            if not hasattr(ll, 'data'):
                continue
            self.buffer[i, self.buffer_index] = ll.data
        self.epsilons = torch.sqrt(torch.mean(self.buffer ** 2, dim=1) - torch.sum(self.buffer, dim=1) ** 2 / self.buffer.shape[1] ** 2)
        self.epsilons[torch.isnan(self.epsilons)] = 10.0
        self.epsilons = self.epsilons.clamp(0.01, 100.0)
        LOG.debug('eps before norm: %s', self.epsilons)
        self.epsilons = self.epsilons * torch.sum(1.0 / self.epsilons) / self.epsilons.shape[0]
        LOG.debug('eps after norm: %s', self.epsilons)
        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.epsilons) == len(flat_head_losses)
        loss_values = [(lam * l / eps) for lam, eps, l in zip(self.lambdas, self.epsilons, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(param.abs().max(dim=1)[0].clamp(min=1e-06).sum() for param in self.sparse_task_parameters)
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss
        return total_loss, flat_head_losses


class Shell(torch.nn.Module):

    def __init__(self, base_net, head_nets, *, process_input=None, process_heads=None):
        super().__init__()
        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads
        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)
        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride
        self.head_nets = head_nets

    def forward(self, image_batch, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)
        x = self.base_net(image_batch)
        if head_mask is not None:
            head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
        else:
            head_outputs = tuple(hn(x) for hn in self.head_nets)
        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)
        return head_outputs


class CrossTalk(torch.nn.Module):

    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, image_batch):
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


class RunningCache(torch.nn.Module):

    def __init__(self, cached_items):
        super().__init__()
        self.cached_items = cached_items
        self.duration = abs(min(cached_items)) + 1
        self.cache = [None for _ in range(self.duration)]
        self.index = 0
        LOG.debug('running cache of length %d', len(self.cache))

    def incr(self):
        self.index = (self.index + 1) % self.duration

    def get_index(self, index):
        while index < 0:
            index += self.duration
        while index >= self.duration:
            index -= self.duration
        LOG.debug('retrieving cache at index %d', index)
        v = self.cache[index]
        if v is not None:
            v = v.detach()
        return v

    def get(self):
        return [self.get_index(i + self.index) for i in self.cached_items]

    def set_next(self, data):
        self.incr()
        self.cache[self.index] = data
        LOG.debug('set new data at index %d', self.index)
        return self

    def forward(self, *args):
        LOG.debug('----------- running cache --------------')
        x = args[0]
        o = []
        for x_i in x:
            o += self.set_next(x_i).get()
        if any(oo is None for oo in o):
            o = [(oo if oo is not None else o[0]) for oo in o]
        if len(o) >= 2:
            image_sizes = [tuple(oo.shape[-2:]) for oo in o]
            if not all(ims == image_sizes[0] for ims in image_sizes[1:]):
                freq = defaultdict(int)
                for ims in image_sizes:
                    freq[ims] += 1
                max_freq = max(freq.values())
                ref_image_size = next(iter(ims for ims, f in freq.items() if f == max_freq))
                for i, ims in enumerate(image_sizes):
                    if ims == ref_image_size:
                        continue
                    for s in range(1, len(image_sizes)):
                        target_i = (i + s) % len(image_sizes)
                        if image_sizes[target_i] == ref_image_size:
                            break
                    LOG.warning('replacing %d (%s) with %d (%s) for ref %s', i, ims, target_i, image_sizes[target_i], ref_image_size)
                    o[i] = o[target_i]
        return torch.stack(o)


class Cifar10Net(openpifpaf.network.BaseNetwork):
    """Small network for Cifar10."""

    def __init__(self):
        super().__init__('cifar10net', stride=16, out_features=128)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        return x


class ModuleUsingCifHr(torch.nn.Module):

    def forward(self, x):
        cifhr = torch.classes.openpifpaf_decoder_utils.CifHr()
        with torch.no_grad():
            cifhr.reset(x.shape[1:], 8)
            cifhr.accumulate(x[1:], 8, 0.0, 1.0)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bce,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BceDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BceL2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Cifar10Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CrossTalk,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Laplace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaleDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShuffleNetV2K,
     lambda: ([], {'name': 4, 'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SoftClamp,
     lambda: ([], {'max_value': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_openpifpaf_openpifpaf(_paritybench_base):
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

