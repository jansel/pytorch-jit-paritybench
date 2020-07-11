import sys
_module = sys.modules[__name__]
del sys
dassl = _module
config = _module
defaults = _module
data = _module
data_manager = _module
datasets = _module
base_dataset = _module
build = _module
da = _module
cifarstl = _module
digit5 = _module
domainnet = _module
mini_domainnet = _module
office31 = _module
office_home = _module
visda17 = _module
dg = _module
digit_single = _module
digits_dg = _module
office_home_dg = _module
pacs = _module
ssl = _module
cifar = _module
stl10 = _module
svhn = _module
samplers = _module
transforms = _module
autoaugment = _module
randaugment = _module
transforms = _module
engine = _module
adabn = _module
adda = _module
dael = _module
dann = _module
m3sda = _module
mcd = _module
mme = _module
self_ensembling = _module
source_only = _module
crossgrad = _module
daeldg = _module
ddaig = _module
vanilla = _module
entmin = _module
fixmatch = _module
mean_teacher = _module
mixmatch = _module
sup_baseline = _module
trainer = _module
evaluation = _module
evaluator = _module
metrics = _module
accuracy = _module
distance = _module
modeling = _module
backbone = _module
alexnet = _module
backbone = _module
cnn_digit5_m3sda = _module
cnn_digitsdg = _module
cnn_digitsingle = _module
efficientnet = _module
model = _module
utils = _module
mobilenetv2 = _module
preact_resnet18 = _module
resnet = _module
shufflenetv2 = _module
vgg = _module
wide_resnet = _module
head = _module
mlp = _module
network = _module
ddaig_fcn = _module
ops = _module
cross_entropy = _module
dsbn = _module
mixup = _module
mmd = _module
optimal_transport = _module
reverse_grad = _module
sequential2 = _module
transnorm = _module
utils = _module
optim = _module
lr_scheduler = _module
optimizer = _module
radam = _module
logger = _module
meters = _module
registry = _module
tools = _module
torchtools = _module
cifar_stl = _module
cifar10_cifar100_svhn = _module
setup = _module
parse_test_res = _module
replace_text = _module
train = _module

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


import torchvision.transforms as T


from torch.utils.data import Dataset as TorchDataset


import copy


import random


from collections import defaultdict


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


import numpy as np


from torchvision.transforms import Resize


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.transforms import CenterCrop


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import RandomHorizontalFlip


import torch.nn as nn


from torch.nn import functional as F


import time


from collections import OrderedDict


from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import confusion_matrix


import torch.utils.model_zoo as model_zoo


from torch import nn


import re


import math


import collections


from functools import partial


from torch.utils import model_zoo


import torch.nn.functional as F


import functools


from torch.autograd import Function


import warnings


from torch.optim.optimizer import Optimizer


class Experts(nn.Module):

    def __init__(self, n_source, fdim, num_classes):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(fdim, num_classes) for _ in range(n_source)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x


class PairClassifiers(nn.Module):

    def __init__(self, fdim, num_classes):
        super().__init__()
        self.c1 = nn.Linear(fdim, num_classes)
        self.c2 = nn.Linear(fdim, num_classes)

    def forward(self, x):
        z1 = self.c1(x)
        if not self.training:
            return z1
        z2 = self.c2(x)
        return z1, z2


class Prototypes(nn.Module):

    def __init__(self, fdim, num_classes, temp=0.05):
        super().__init__()
        self.prototypes = nn.Linear(fdim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = self.prototypes(x)
        out = out / self.temp
        return out


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj):
        if name in self._obj_map:
            raise KeyError('An object named "{}" was already registered in "{}" registry'.format(name, self._name))
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:

            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class)
                return fn_or_class
            return wrapper
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError('Object name "{}" does not exist in "{}" registry'.format(name, self._name))
        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())


BACKBONE_REGISTRY = Registry('BACKBONE')


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError('The requested one is expected to belong to {}, but got [{}] (do you mean [{}]?)'.format(available, requested, psb_ans))


def build_backbone(name, verbose=True, **kwargs):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    check_availability(name, avai_backbones)
    if verbose:
        None
    return BACKBONE_REGISTRY.get(name)(**kwargs)


HEAD_REGISTRY = Registry('HEAD')


def build_head(name, verbose=True, **kwargs):
    avai_heads = HEAD_REGISTRY.registered_names()
    check_availability(name, avai_heads)
    if verbose:
        None
    return HEAD_REGISTRY.get(name)(**kwargs)


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(model_cfg.BACKBONE.NAME, verbose=cfg.VERBOSE, pretrained=model_cfg.BACKBONE.PRETRAINED, **kwargs)
        fdim = self.backbone.out_features
        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(model_cfg.HEAD.NAME, verbose=cfg.VERBOSE, in_features=fdim, hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS, activation=model_cfg.HEAD.ACTIVATION, bn=model_cfg.HEAD.BN, dropout=model_cfg.HEAD.DROPOUT, **kwargs)
            fdim = self.head.out_features
        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)
        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)
        if self.classifier is None:
            return f
        y = self.classifier(f)
        if return_feature:
            return y, f
        return y


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get('_out_features') is None:
            return None
        return self._out_features


class FeatureExtractor(Backbone):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self._out_features = 2048

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, 'Input to network must be 32x32, but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        self._out_features = 2 ** 2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, 'Input to network must be 32x32, but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)


class CNN(Backbone):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc3 = nn.Linear(5 * 5 * 128, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self._out_features = 1024

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, 'Input to network must be 32x32, but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """
    Calculates the output image size when using Conv2dSamePadding with a stride.
    Necessary for static padding. Thanks to mannatsingh for pointing this out.
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


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


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

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


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = self._block_args.se_ratio is not None and 0 < self._block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(self._project_conv(x))
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3), 'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5), 'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}
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


GlobalParams = collections.namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=dropout_rate, drop_connect_rate=drop_connect_rate, num_classes=num_classes, width_coefficient=width_coefficient, depth_coefficient=depth_coefficient, depth_divisor=8, min_depth=None, image_size=image_size)
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def load_checkpoint(fpath):
    """Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        None
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(weight_path))
    else:
        None
        if len(discarded_layers) > 0:
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


class EfficientNet(Backbone):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(block_args.input_filters, self._global_params), output_filters=round_filters(block_args.output_filters, self._global_params), num_repeat=round_repeats(block_args.num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._swish = MemoryEfficientSwish()
        self._out_features = out_channels

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        """
        Calls extract_features to extract features, applies
        final linear layer, and returns logits.
        """
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=num_classes == 1000, advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = [('efficientnet-b' + str(i)) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    def _change_in_channels(model, in_channels):
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super().__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(Backbone):

    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None):
        """
        MobileNet V2.

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super().__init__()
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self._out_features = self.last_channel
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

    def _forward_impl(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return x

    def forward(self, x):
        return self._forward_impl(x)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(Backbone):

    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self._out_features = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(Backbone):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self._out_features = 512 * block.expansion
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


class ShuffleNetV2(Backbone):

    def __init__(self, stages_repeats, stages_out_channels, **kwargs):
        super().__init__()
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._out_features = output_channels

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


class VGG(Backbone):

    def __init__(self, features, init_weights=True):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())
        self._out_features = 4096
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
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


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(Backbone):

    def __init__(self, depth, widen_factor, dropRate=0.0):
        super().__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self._out_features = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


class MLP(nn.Module):

    def __init__(self, in_features=2048, hidden_layers=[], activation='relu', bn=True, dropout=0.0):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]
        mlp = []
        if activation == 'relu':
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == 'leaky_relu':
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError
        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LocNet(nn.Module):
    """Localization network."""

    def __init__(self, input_nc, nc=32, n_blocks=3, use_dropout=False, padding_type='zero', image_size=32):
        super().__init__()
        backbone = []
        backbone += [nn.Conv2d(input_nc, nc, kernel_size=3, stride=2, padding=1, bias=False)]
        backbone += [nn.BatchNorm2d(nc)]
        backbone += [nn.ReLU(True)]
        for _ in range(n_blocks):
            backbone += [ResnetBlock(nc, padding_type=padding_type, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout, use_bias=False)]
            backbone += [nn.MaxPool2d(2, stride=2)]
        self.backbone = nn.Sequential(*backbone)
        reduced_imsize = int(image_size * 0.5 ** (n_blocks + 1))
        self.fc_loc = nn.Linear(nc * reduced_imsize ** 2, 2 * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_loc(x)
        x = torch.tanh(x)
        x = x.view(-1, 2, 2)
        theta = x.data.new_zeros(x.size(0), 2, 3)
        theta[:, :, :2] = x
        return theta


class FCN(nn.Module):
    """Fully convolutional network."""

    def __init__(self, input_nc, output_nc, nc=32, n_blocks=3, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect', gctx=True, stn=False, image_size=32):
        super().__init__()
        backbone = []
        p = 0
        if padding_type == 'reflect':
            backbone += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            backbone += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError
        backbone += [nn.Conv2d(input_nc, nc, kernel_size=3, stride=1, padding=p, bias=False)]
        backbone += [norm_layer(nc)]
        backbone += [nn.ReLU(True)]
        for _ in range(n_blocks):
            backbone += [ResnetBlock(nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=False)]
        self.backbone = nn.Sequential(*backbone)
        self.gctx_fusion = None
        if gctx:
            self.gctx_fusion = nn.Sequential(nn.Conv2d(2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(nc), nn.ReLU(True))
        self.regress = nn.Sequential(nn.Conv2d(nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True), nn.Tanh())
        self.locnet = None
        if stn:
            self.locnet = LocNet(input_nc, nc=nc, n_blocks=n_blocks, image_size=image_size)

    def init_loc_layer(self):
        """Initialize the weights/bias with identity transformation."""
        if self.locnet is not None:
            self.locnet.fc_loc.weight.data.zero_()
            self.locnet.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

    def stn(self, x):
        """Spatial transformer network."""
        theta = self.locnet(x)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid), theta

    def forward(self, x, lmda=1.0, return_p=False, return_stn_output=False):
        """
        Args:
            x (torch.Tensor): input mini-batch.
            lmda (float): multiplier for perturbation.
            return_p (bool): return perturbation.
            return_stn_output (bool): return the output of stn.
        """
        theta = None
        if self.locnet is not None:
            x, theta = self.stn(x)
        input = x
        x = self.backbone(x)
        if self.gctx_fusion is not None:
            c = F.adaptive_avg_pool2d(x, (1, 1))
            c = c.expand_as(x)
            x = torch.cat([x, c], 1)
            x = self.gctx_fusion(x)
        p = self.regress(x)
        x_p = input + lmda * p
        if return_stn_output:
            return x_p, p, input
        if return_p:
            return x_p, p
        return x_p


class _DSBN(nn.Module):
    """Domain Specific Batch Normalization.

    Args:
        num_features (int): number of features.
        n_domain (int): number of domains.
        bn_type (str): type of bn. Choices are ['1d', '2d'].
    """

    def __init__(self, num_features, n_domain, bn_type):
        super().__init__()
        if bn_type == '1d':
            BN = nn.BatchNorm1d
        elif bn_type == '2d':
            BN = nn.BatchNorm2d
        else:
            raise ValueError
        self.bn = nn.ModuleList(BN(num_features) for _ in range(n_domain))
        self.valid_domain_idxs = list(range(n_domain))
        self.n_domain = n_domain
        self.domain_idx = 0

    def select_bn(self, domain_idx=0):
        assert domain_idx in self.valid_domain_idxs
        self.domain_idx = domain_idx

    def forward(self, x):
        return self.bn[self.domain_idx](x)


class DSBN1d(_DSBN):

    def __init__(self, num_features, n_domain):
        super().__init__(num_features, n_domain, '1d')


class DSBN2d(_DSBN):

    def __init__(self, num_features, n_domain):
        super().__init__(num_features, n_domain, '2d')


class MaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernel_type='rbf', normalize=False):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.kernel_type = kernel_type
        self.normalize = normalize

    def forward(self, x, y):
        if self.normalize:
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
        if self.kernel_type == 'linear':
            return self.linear_mmd(x, y)
        elif self.kernel_type == 'poly':
            return self.poly_mmd(x, y)
        elif self.kernel_type == 'rbf':
            return self.rbf_mmd(x, y)
        else:
            raise NotImplementedError

    def linear_mmd(self, x, y):
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_xy = torch.mm(x, y.t())
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def poly_mmd(self, x, y, alpha=1.0, c=2.0, d=2):
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_xx = (alpha * k_xx + c).pow(d)
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_yy = (alpha * k_yy + c).pow(d)
        k_xy = torch.mm(x, y.t())
        k_xy = (alpha * k_xy + c).pow(d)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def rbf_mmd(self, x, y):
        d_xx = self.euclidean_squared_distance(x, x)
        d_xx = self.remove_self_distance(d_xx)
        k_xx = self.rbf_kernel_mixture(d_xx)
        d_yy = self.euclidean_squared_distance(y, y)
        d_yy = self.remove_self_distance(d_yy)
        k_yy = self.rbf_kernel_mixture(d_yy)
        d_xy = self.euclidean_squared_distance(x, y)
        k_xy = self.rbf_kernel_mixture(d_xy)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    @staticmethod
    def rbf_kernel_mixture(exponent, sigmas=[1, 5, 10]):
        K = 0
        for sigma in sigmas:
            gamma = 1.0 / (2.0 * sigma ** 2)
            K += torch.exp(-gamma * exponent)
        return K

    @staticmethod
    def remove_self_distance(distmat):
        tmp_list = []
        for i, row in enumerate(distmat):
            row1 = torch.cat([row[:i], row[i + 1:]])
            tmp_list.append(row1)
        return torch.stack(tmp_list)

    @staticmethod
    def euclidean_squared_distance(x, y):
        m, n = x.size(0), y.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, x, y.t())
        return distmat


class OptimalTransport(nn.Module):

    @staticmethod
    def distance(batch1, batch2, dist_metric='cosine'):
        if dist_metric == 'cosine':
            batch1 = F.normalize(batch1, p=2, dim=1)
            batch2 = F.normalize(batch2, p=2, dim=1)
            dist_mat = 1 - torch.mm(batch1, batch2.t())
        elif dist_metric == 'euclidean':
            m, n = batch1.size(0), batch2.size(0)
            dist_mat = torch.pow(batch1, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(batch2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_mat.addmm_(1, -2, batch1, batch2.t())
        elif dist_metric == 'fast_euclidean':
            batch1 = batch1.unsqueeze(-2)
            batch2 = batch2.unsqueeze(-3)
            dist_mat = torch.sum(torch.abs(batch1 - batch2) ** 2, -1)
        else:
            raise ValueError('Unknown cost function: {}. Expected to be one of [cosine | euclidean]'.format(dist_metric))
        return dist_mat


class SinkhornDivergence(OptimalTransport):
    thre = 0.001

    def __init__(self, dist_metric='cosine', eps=0.01, max_iter=5, bp_to_sinkhorn=False):
        super().__init__()
        self.dist_metric = dist_metric
        self.eps = eps
        self.max_iter = max_iter
        self.bp_to_sinkhorn = bp_to_sinkhorn

    def forward(self, x, y):
        W_xy = self.transport_cost(x, y)
        W_xx = self.transport_cost(x, x)
        W_yy = self.transport_cost(y, y)
        return 2 * W_xy - W_xx - W_yy

    def transport_cost(self, x, y, return_pi=False):
        C = self.distance(x, y, dist_metric=self.dist_metric)
        pi = self.sinkhorn_iterate(C, self.eps, self.max_iter, self.thre)
        if not self.bp_to_sinkhorn:
            pi = pi.detach()
        cost = torch.sum(pi * C)
        if return_pi:
            return cost, pi
        return cost

    @staticmethod
    def sinkhorn_iterate(C, eps, max_iter, thre):
        nx, ny = C.shape
        mu = torch.ones(nx, dtype=C.dtype, device=C.device) * (1.0 / nx)
        nu = torch.ones(ny, dtype=C.dtype, device=C.device) * (1.0 / ny)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        def M(_C, _u, _v):
            """Modified cost for logarithmic updates.
            Eq: M_{ij} = (-c_{ij} + u_i + v_j) / epsilon
            """
            return (-_C + _u.unsqueeze(-1) + _v.unsqueeze(-2)) / eps
        real_iter = 0
        for i in range(max_iter):
            u0 = u
            u = eps * (torch.log(mu + 1e-08) - torch.logsumexp(M(C, u, v), dim=1)) + u
            v = eps * (torch.log(nu + 1e-08) - torch.logsumexp(M(C, u, v).permute(1, 0), dim=1)) + v
            err = (u - u0).abs().sum()
            real_iter += 1
            if err.item() < thre:
                break
        return torch.exp(M(C, u, v))


class MinibatchEnergyDistance(SinkhornDivergence):

    def __init__(self, dist_metric='cosine', eps=0.01, max_iter=5, bp_to_sinkhorn=False):
        super().__init__(dist_metric=dist_metric, eps=eps, max_iter=max_iter, bp_to_sinkhorn=bp_to_sinkhorn)

    def forward(self, x, y):
        x1, x2 = torch.split(x, x.size(0) // 2, dim=0)
        y1, y2 = torch.split(y, y.size(0) // 2, dim=0)
        cost = 0
        cost += self.transport_cost(x1, y1)
        cost += self.transport_cost(x1, y2)
        cost += self.transport_cost(x2, y1)
        cost += self.transport_cost(x2, y2)
        cost -= 2 * self.transport_cost(x1, x2)
        cost -= 2 * self.transport_cost(y1, y2)
        return cost


class _ReverseGrad(Function):

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_scaling = ctx.grad_scaling
        return -grad_scaling * grad_output, None


reverse_grad = _ReverseGrad.apply


class ReverseGrad(nn.Module):
    """Gradient reversal layer.

    It acts as an identity layer in the forward,
    but reverses the sign of the gradient in
    the backward.
    """

    def forward(self, x, grad_scaling=1.0):
        assert grad_scaling >= 0, 'grad_scaling must be non-negative, but got {}'.format(grad_scaling)
        return reverse_grad(x, grad_scaling)


class Sequential2(nn.Sequential):
    """An alternative sequential container to nn.Sequential,
    which accepts an arbitrary number of input arguments.
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class _TransNorm(nn.Module):
    """Transferable normalization.

    Reference:
        - Wang et al. Transferable Normalization: Towards Improving
        Transferability of Deep Neural Networks. NeurIPS 2019.

    Args:
        num_features (int): number of features.
        eps (float): epsilon.
        momentum (float): value for updating running_mean and running_var.
        adaptive_alpha (bool): apply domain adaptive alpha.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, adaptive_alpha=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.adaptive_alpha = adaptive_alpha
        self.register_buffer('running_mean_s', torch.zeros(num_features))
        self.register_buffer('running_var_s', torch.ones(num_features))
        self.register_buffer('running_mean_t', torch.zeros(num_features))
        self.register_buffer('running_var_t', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def resnet_running_stats(self):
        self.running_mean_s.zero_()
        self.running_var_s.fill_(1)
        self.running_mean_t.zero_()
        self.running_var_t.fill_(1)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _check_input(self, x):
        raise NotImplementedError

    def _compute_alpha(self, mean_s, var_s, mean_t, var_t):
        C = self.num_features
        ratio_s = mean_s / (var_s + self.eps).sqrt()
        ratio_t = mean_t / (var_t + self.eps).sqrt()
        dist = (ratio_s - ratio_t).abs()
        dist_inv = 1 / (1 + dist)
        return C * dist_inv / dist_inv.sum()

    def forward(self, input):
        self._check_input(input)
        C = self.num_features
        if input.dim() == 2:
            new_shape = 1, C
        elif input.dim() == 4:
            new_shape = 1, C, 1, 1
        else:
            raise ValueError
        weight = self.weight.view(*new_shape)
        bias = self.bias.view(*new_shape)
        if not self.training:
            mean_t = self.running_mean_t.view(*new_shape)
            var_t = self.running_var_t.view(*new_shape)
            output = (input - mean_t) / (var_t + self.eps).sqrt()
            output = output * weight + bias
            if self.adaptive_alpha:
                mean_s = self.running_mean_s.view(*new_shape)
                var_s = self.running_var_s.view(*new_shape)
                alpha = self._compute_alpha(mean_s, var_s, mean_t, var_t)
                alpha = alpha.reshape(*new_shape)
                output = (1 + alpha.detach()) * output
            return output
        input_s, input_t = torch.split(input, input.shape[0] // 2, dim=0)
        x_s = input_s.transpose(0, 1).reshape(C, -1)
        mean_s = x_s.mean(1)
        var_s = x_s.var(1)
        self.running_mean_s.mul_(self.momentum)
        self.running_mean_s.add_((1 - self.momentum) * mean_s.data)
        self.running_var_s.mul_(self.momentum)
        self.running_var_s.add_((1 - self.momentum) * var_s.data)
        mean_s = mean_s.reshape(*new_shape)
        var_s = var_s.reshape(*new_shape)
        output_s = (input_s - mean_s) / (var_s + self.eps).sqrt()
        output_s = output_s * weight + bias
        x_t = input_t.transpose(0, 1).reshape(C, -1)
        mean_t = x_t.mean(1)
        var_t = x_t.var(1)
        self.running_mean_t.mul_(self.momentum)
        self.running_mean_t.add_((1 - self.momentum) * mean_t.data)
        self.running_var_t.mul_(self.momentum)
        self.running_var_t.add_((1 - self.momentum) * var_t.data)
        mean_t = mean_t.reshape(*new_shape)
        var_t = var_t.reshape(*new_shape)
        output_t = (input_t - mean_t) / (var_t + self.eps).sqrt()
        output_t = output_t * weight + bias
        output = torch.cat([output_s, output_t], 0)
        if self.adaptive_alpha:
            alpha = self._compute_alpha(mean_s, var_s, mean_t, var_t)
            alpha = alpha.reshape(*new_shape)
            output = (1 + alpha.detach()) * output
        return output


class TransNorm1d(_TransNorm):

    def _check_input(self, x):
        if x.dim() != 2:
            raise ValueError('Expected the input to be 2-D, but got {}-D'.format(x.dim()))


class TransNorm2d(_TransNorm):

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError('Expected the input to be 4-D, but got {}-D'.format(x.dim()))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Backbone,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dDynamicSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Convolution,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DSBN1d,
     lambda: ([], {'num_features': 4, 'n_domain': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DSBN2d,
     lambda: ([], {'num_features': 4, 'n_domain': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Experts,
     lambda: ([], {'n_source': 4, 'fdim': 4, 'num_classes': 4}),
     lambda: ([0, torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCN,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaximumMeanDiscrepancy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MinibatchEnergyDistance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (NetworkBlock,
     lambda: ([], {'nb_layers': 1, 'in_planes': 4, 'out_planes': 4, 'block': _mock_layer, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairClassifiers,
     lambda: ([], {'fdim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Prototypes,
     lambda: ([], {'fdim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReverseGrad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sequential2,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (ShuffleNetV2,
     lambda: ([], {'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SinkhornDivergence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (TransNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_KaiyangZhou_Dassl_pytorch(_paritybench_base):
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

