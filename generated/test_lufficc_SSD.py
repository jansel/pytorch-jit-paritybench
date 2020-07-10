import sys
_module = sys.modules[__name__]
del sys
demo = _module
ext = _module
build = _module
setup = _module
ssd = _module
config = _module
defaults = _module
path_catlog = _module
data = _module
build = _module
datasets = _module
coco = _module
evaluation = _module
voc = _module
eval_detection_voc = _module
voc = _module
samplers = _module
distributed = _module
iteration_based_batch_sampler = _module
transforms = _module
target_transform = _module
transforms = _module
engine = _module
inference = _module
trainer = _module
layers = _module
separable_conv = _module
modeling = _module
anchors = _module
prior_box = _module
backbone = _module
efficient_net = _module
efficient_net = _module
utils = _module
mobilenet = _module
vgg = _module
box_head = _module
box_head = _module
box_predictor = _module
inference = _module
loss = _module
detector = _module
ssd_detector = _module
registry = _module
solver = _module
build = _module
lr_scheduler = _module
structures = _module
container = _module
box_utils = _module
checkpoint = _module
dist_util = _module
logger = _module
metric_logger = _module
misc = _module
model_zoo = _module
nms = _module
test = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import torch


import numpy as np


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.utils.data import ConcatDataset


import torch.utils.data


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


from torchvision import transforms


import types


from numpy import random


import logging


import collections


import torch.nn as nn


import torch.nn.init as init


from torch import nn


from itertools import product


from math import sqrt


from torch.nn import functional as F


import re


import torch.nn.functional as F


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.parallel import DistributedDataParallel


from collections import deque


from collections import defaultdict


import warnings


import torchvision


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
        super().__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding), nn.BatchNorm2d(in_channels), ReLU(), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = self._block_args.se_ratio is not None and 0 < self._block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


EXTRAS = {'efficientnet-b3': [[(384, 128, 1, 1, 0), (128, 256, 3, 2, 1)], [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)], [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)]]}


INDICES = {'efficientnet-b3': [7, 17, 25]}


def add_extras(cfg, i, size=300):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3), 'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5)}
    return params_dict[model_name]


BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        assert 's' in options and len(options['s']) == 1 or len(options['s']) == 2 and options['s'][0] == options['s'][1]
        return BlockArgs(kernel_size=int(options['k']), num_repeat=int(options['r']), input_filters=int(options['i']), output_filters=int(options['o']), expand_ratio=int(options['e']), id_skip='noskip' not in block_string, se_ratio=float(options['se']) if 'se' in options else None, stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.output_filters]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


GlobalParams = collections.namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate'])


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2):
    """ Creates a efficientnet model. """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate, num_classes=1000, width_coefficient=width_coefficient, depth_coefficient=depth_coefficient, depth_divisor=8, min_depth=None)
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
       Helper function to synchronize (barrier) among all processes when
       using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def cache_url(url, model_dir=None, progress=True):
    """Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> cached_file = maskrcnn_benchmark.utils.model_zoo.cache_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == 'model_final.pkl':
        filename = parts.path.replace('/', '_')
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            if len(hash_prefix) < 6:
                hash_prefix = None
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    synchronize()
    return cached_file


def load_state_dict_from_url(url, map_location='cpu'):
    cached_file = cache_url(url)
    return torch.load(cached_file, map_location=map_location)


url_map = {'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth', 'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth', 'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth', 'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth', 'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth', 'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth'}


def load_pretrained_weights(model, model_name):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = load_state_dict_from_url(url_map[model_name])
    model.load_state_dict(state_dict, strict=False)
    None


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, model_name, blocks_args=None, global_params=None):
        super().__init__()
        self.indices = INDICES[model_name]
        self.extras = add_extras(EXTRAS[model_name])
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params), output_filters=round_filters(block_args.output_filters, self._global_params), num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        features = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate)
            if idx in self.indices:
                features.append(x)
        return x, features

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        x, features = self.extract_features(inputs)
        for layer in self.extras:
            x = layer(x)
            features.append(x)
        return tuple(features)

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(model_name, blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = [('efficientnet_b' + str(i)) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.extras = nn.ModuleList([InvertedResidual(1280, 512, 2, 0.2), InvertedResidual(512, 256, 2, 0.25), InvertedResidual(256, 256, 2, 0.5), InvertedResidual(256, 64, 2, 0.25)])
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for i in range(14):
            x = self.features[i](x)
        features.append(x)
        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)
        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)
        return tuple(features)


def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


extras_base = {'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256], '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]}


vgg_base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512], '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]}


class VGG(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]
        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm = L2Norm(512, scale=20)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)
        features.append(s)
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)
        return tuple(features)


class MultiBoxLoss(nn.Module):

    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, (0)]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[(mask), :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[(pos_mask), :].view(-1, 4)
        gt_locations = gt_locations[(pos_mask), :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class Container:
    """
    Help class for manage boxes, labels, etc...
    Not inherit dict due to `default_collate` will change dict's subclass to dict.
    """

    def __init__(self, *args, **kwargs):
        self._data_dict = dict(*args, **kwargs)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self._data_dict[key]

    def __iter__(self):
        return self._data_dict.__iter__()

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def _call(self, name, *args, **kwargs):
        keys = list(self._data_dict.keys())
        for key in keys:
            value = self._data_dict[key]
            if hasattr(value, name):
                self._data_dict[key] = getattr(value, name)(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        return self._call('to', *args, **kwargs)

    def numpy(self):
        return self._call('numpy')

    def resize(self, size):
        """resize boxes
        Args:
            size: (width, height)
        Returns:
            self
        """
        img_width = getattr(self, 'img_width', -1)
        img_height = getattr(self, 'img_height', -1)
        assert img_width > 0 and img_height > 0
        assert 'boxes' in self._data_dict
        boxes = self._data_dict['boxes']
        new_width, new_height = size
        boxes[:, 0::2] *= new_width / img_width
        boxes[:, 1::2] *= new_height / img_height
        return self

    def __repr__(self):
        return self._data_dict.__repr__()


def nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
        scores(Tensor[N]): scores
        nms_thresh(float): thresh
    Returns:
        indices kept.
    """
    keep = _nms(boxes, scores, nms_thresh)
    return keep


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, (None)]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


class PostProcessor:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]
            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]
            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height
            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            keep = keep[:self.cfg.TEST.MAX_PER_IMAGE]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
        return results


class PriorBox:

    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])
        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors


def make_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)


class SSDBoxHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss)
        detections = cls_logits, bbox_pred
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)()
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE)
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = scores, boxes
        detections = self.post_processor(detections)
        return detections, {}


class BoxPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())
        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)
        return cls_logits, bbox_pred


class SSDBoxPredictor(BoxPredictor):

    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


class SSDLiteBoxPredictor(BoxPredictor):

    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)


def build_box_head(cfg):
    return registry.BOX_HEADS[cfg.MODEL.BOX_HEAD.NAME](cfg)


class SSDDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2dSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lufficc_SSD(_paritybench_base):
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

