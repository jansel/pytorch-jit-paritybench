import sys
_module = sys.modules[__name__]
del sys
FastAutoAugment = _module
archive = _module
aug_mixup = _module
augmentations = _module
common = _module
data = _module
imagenet = _module
lr_scheduler = _module
metrics = _module
networks = _module
efficientnet_pytorch = _module
condconv = _module
model = _module
utils = _module
pyramidnet = _module
resnet = _module
shakedrop = _module
shakeshake = _module
shake_resnet = _module
shake_resnext = _module
shakeshake = _module
wideresnet = _module
safe_shell_exec = _module
search = _module
tf_port = _module
rmsprop = _module
tpu_bn = _module
train = _module
train_dist = _module
master = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import numpy as np


import torch


import copy


from collections import defaultdict


from torch import nn


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.functional as F


from torch._six import container_abcs


from itertools import repeat


from functools import partial


from typing import Union


from typing import List


from typing import Tuple


from typing import Optional


from typing import Callable


import math


from torch.nn import functional as F


import re


import collections


from torch.utils import model_zoo


from torch.autograd import Variable


import torch.nn.init as init


from collections import OrderedDict


from torch.nn import BatchNorm2d


from torch.nn.parameter import Parameter


import torch.distributed as dist


import itertools


import logging


from torch import optim


from torch.nn.parallel.data_parallel import DataParallel


class CrossEntropyMixUpLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyMixUpLabelSmooth, self).__init__()
        self.ce = CrossEntropyLabelSmooth(num_classes, epsilon, reduction=
            reduction)

    def forward(self, input, target1, target2, lam):
        return lam * self.ce(input, target1) + (1 - lam) * self.ce(input,
            target2)


class CrossEntropyLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(
            1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon
                ) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = -targets * log_probs
        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if _is_static_pad(kernel_size, **kwargs):
                padding = _get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor]=None,
    stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0),
    dilation: Tuple[int, int]=(1, 1), groups: int=1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - 
            pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


def get_condconv_initializer(initializer, num_experts, expert_shape):

    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if len(weight.shape) != 2 or weight.shape[0
            ] != num_experts or weight.shape[1] != num_params:
            raise ValueError(
                'CondConv variables must have shape [num_experts, num_params]')
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer


class CondConv2d(nn.Module):
    """ Conditional Convolution
    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        assert num_experts > 1
        if isinstance(stride, container_abcs.Iterable) and len(stride) == 1:
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding,
            kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = _pair(padding_val)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (self.out_channels, self.in_channels // self.groups
            ) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts,
            weight_num_param))
        if bias:
            self.bias_shape = self.out_channels,
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts,
                self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        num_input_fmaps = self.weight.size(1)
        num_output_fmaps = self.weight.size(0)
        receptive_field_size = 1
        if self.weight.dim() > 2:
            receptive_field_size = self.weight[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        init_weight = get_condconv_initializer(partial(nn.init.normal_,
            mean=0.0, std=np.sqrt(2.0 / fan_out)), self.num_experts, self.
            weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            init_bias = get_condconv_initializer(partial(nn.init.constant_,
                val=0), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        x_orig = x
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self
            .groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.view(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(x, weight, bias, stride=self.stride, padding=
                self.padding, dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(x, weight, bias, stride=self.stride, padding=
                self.padding, dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.
            shape[-2], out.shape[-1])
        return out

    def forward_legacy(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        x = torch.split(x, 1, 0)
        weight = torch.split(weight, 1, 0)
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = torch.split(bias, 1, 0)
        else:
            bias = [None] * B
        out = []
        if self.dynamic_padding:
            conv_fn = conv2d_same
        else:
            conv_fn = F.conv2d
        for xi, wi, bi in zip(x, weight, bias):
            wi = wi.view(*self.weight_shape)
            if bi is not None:
                bi = bi.view(*self.bias_shape)
            out.append(conv_fn(xi, wi, bi, stride=self.stride, padding=self
                .padding, dilation=self.dilation, groups=self.groups))
        out = torch.cat(out, 0)
        return out


class RoutingFn(nn.Linear):
    pass


def drop_connect(inputs, drop_p, training):
    """ Drop connect. """
    if not training:
        return inputs * (1.0 - drop_p)
    batch_size = inputs.shape[0]
    random_tensor = torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,
        device=inputs.device)
    binary_tensor = random_tensor > drop_p
    output = inputs * binary_tensor.float()
    return output


def get_same_padding_conv2d(image_size=None, condconv_num_expert=1):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if condconv_num_expert > 1:
        return partial(CondConv2d, num_experts=condconv_num_expert)
    elif image_size is None:
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

    def __init__(self, block_args, global_params, norm_layer=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None and 0 < self.
            _block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.condconv_num_expert = block_args.condconv_num_expert
        if self._is_condconv():
            self.routing_fn = RoutingFn(self._block_args.input_filters,
                self.condconv_num_expert)
        Conv2d = get_same_padding_conv2d(image_size=global_params.
            image_size, condconv_num_expert=block_args.condconv_num_expert)
        Conv2dse = get_same_padding_conv2d(image_size=global_params.image_size)
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup,
                kernel_size=1, bias=False)
            self._bn0 = norm_layer(num_features=oup, momentum=self._bn_mom,
                eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup,
            groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = norm_layer(num_features=oup, momentum=self._bn_mom, eps
            =self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.
                input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2dse(in_channels=oup, out_channels=
                num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2dse(in_channels=num_squeezed_channels,
                out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup,
            kernel_size=1, bias=False)
        self._bn2 = norm_layer(num_features=final_oup, momentum=self.
            _bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def _is_condconv(self):
        return self.condconv_num_expert > 1

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        if self._is_condconv():
            feat = F.adaptive_avg_pool2d(inputs, 1).flatten(1)
            routing_w = torch.sigmoid(self.routing_fn(feat))
            if self._block_args.expand_ratio != 1:
                _expand_conv = partial(self._expand_conv, routing_weights=
                    routing_w)
            _depthwise_conv = partial(self._depthwise_conv, routing_weights
                =routing_w)
            _project_conv = partial(self._project_conv, routing_weights=
                routing_w)
        else:
            if self._block_args.expand_ratio != 1:
                _expand_conv = self._expand_conv
            _depthwise_conv, _project_conv = (self._depthwise_conv, self.
                _project_conv)
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(_expand_conv(inputs)))
        x = self._swish(self._bn1(_depthwise_conv(x)))
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(
                x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x
        x = self._bn2(_project_conv(x))
        input_filters, output_filters = (self._block_args.input_filters,
            self._block_args.output_filters)
        if (self.id_skip and self._block_args.stride == 1 and input_filters ==
            output_filters):
            if drop_connect_rate:
                x = drop_connect(x, drop_p=drop_connect_rate, training=self
                    .training)
            x = x + inputs
        return x

    def set_swish(self):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish()


BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size',
    'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
    'id_skip', 'stride', 'se_ratio', 'condconv_num_expert'])


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
        assert 's' in options and len(options['s']) == 1 or len(options['s']
            ) == 2 and options['s'][0] == options['s'][1]
        return BlockArgs(kernel_size=int(options['k']), num_repeat=int(
            options['r']), input_filters=int(options['i']), output_filters=
            int(options['o']), expand_ratio=int(options['e']), id_skip=
            'noskip' not in block_string, se_ratio=float(options['se']) if 
            'se' in options else None, stride=[int(options['s'][0])],
            condconv_num_expert=0)

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 
            's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.
            expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.
            output_filters]
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


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])


def efficientnet(width_coefficient=None, depth_coefficient=None,
    dropout_rate=0.2, drop_connect_rate=0.2, image_size=None, num_classes=
    1000, condconv_num_expert=1):
    """ Creates a efficientnet model. """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    blocks_args_new = blocks_args[:-3]
    for blocks_arg in blocks_args[-3:]:
        blocks_arg = blocks_arg._replace(condconv_num_expert=
            condconv_num_expert)
        blocks_args_new.append(blocks_arg)
    blocks_args = blocks_args_new
    global_params = GlobalParams(batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001, dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate, num_classes=num_classes,
        width_coefficient=width_coefficient, depth_coefficient=
        depth_coefficient, depth_divisor=8, min_depth=None, image_size=
        image_size)
    return blocks_args, global_params


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 
        1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 
        2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)}
    return params_dict[model_name]


def get_model_params(model_name, override_params, condconv_num_expert=1):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w,
            depth_coefficient=d, dropout_rate=p, image_size=s,
            condconv_num_expert=condconv_num_expert)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
            model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


url_map = {'efficientnet-b0':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth'
    , 'efficientnet-b1':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth'
    , 'efficientnet-b2':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth'
    , 'efficientnet-b3':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth'
    , 'efficientnet-b4':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth'
    , 'efficientnet-b5':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth'
    , 'efficientnet-b6':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth'
    , 'efficientnet-b7':
    'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth'
    }


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']
            ), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor *
        divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, norm_layer=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3,
            stride=2, bias=False)
        self._bn0 = norm_layer(num_features=out_channels, momentum=bn_mom,
            eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for idx, block_args in enumerate(self._blocks_args):
            block_args = block_args._replace(input_filters=round_filters(
                block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters,
                self._global_params), num_repeat=round_repeats(block_args.
                num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params,
                norm_layer=norm_layer))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.
                    output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.
                    _global_params, norm_layer=norm_layer))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1,
            bias=False)
        self._bn1 = norm_layer(num_features=out_channels, momentum=bn_mom,
            eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish()
        for block in self._blocks:
            block.set_swish()

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
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, norm_layer=None,
        condconv_num_expert=1):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name,
            override_params, condconv_num_expert=condconv_num_expert)
        return cls(blocks_args, global_params, norm_layer=norm_layer)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={'num_classes':
            num_classes})
        load_pretrained_weights(model, model_name, load_fc=num_classes == 1000)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name,
        also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = [('efficientnet-b' + str(i)) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError(f'model_name={model_name} should be one of: ' +
                ', '.join(valid_models))


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]
            ] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=
        None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]
            ] * 2
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size,
            image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w //
                2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=True)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.shake_drop(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(
                batch_size, residual_channel - shortcut_channel,
                featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * 1, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.conv3 = nn.Conv2d(planes * 1, planes * Bottleneck.
            outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.shake_drop(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(
                batch_size, residual_channel - shortcut_channel,
                featuremap_size[0], featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut
        return out


class PyramidNet(nn.Module):

    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=True):
        super(PyramidNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            self.addrate = alpha / (3 * n * 1.0)
            self.ps_shakedrop = [(1.0 - (1.0 - 0.5 / (3 * n) * (i + 1))) for
                i in range(3 * n)]
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim,
                kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        elif dataset == 'imagenet':
            blocks = {(18): BasicBlock, (34): BasicBlock, (50): Bottleneck,
                (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 
                6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [
                3, 24, 36, 3]}
            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth - 2) / 12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth - 2) / 8)
                layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                None
            self.inplanes = 64
            self.addrate = alpha / (sum(layers[depth]) * 1.0)
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim,
                kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[
                depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[
                depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[
                depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[
                depth][3], stride=2)
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        assert len(self.ps_shakedrop) == 0, self.ps_shakedrop

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.
            featuremap_dim)), stride, downsample, p_shakedrop=self.
            ps_shakedrop.pop(0)))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.
                outchannel_ratio, int(round(temp_featuremap_dim)), 1,
                p_shakedrop=self.ps_shakedrop.pop(0)))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)
            ) * block.outchannel_ratio
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
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


class ResNet(nn.Module):

    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            None
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=
                1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif dataset == 'imagenet':
            blocks = {(18): BasicBlock, (34): BasicBlock, (50): Bottleneck,
                (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 
                6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [
                3, 24, 36, 3]}
            assert layers[depth
                ], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=
                2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth
                ][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth
                ][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth
                ][3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range
                    )
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.
            alpha_range)


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)
        return beta * grad_output, (1 - beta) * grad_output, None


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch,
            stride=stride)
        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(in_ch,
            out_ch, 3, padding=1, stride=stride, bias=False), nn.
            BatchNorm2d(out_ch), nn.ReLU(inplace=False), nn.Conv2d(out_ch,
            out_ch, 3, padding=1, stride=1, bias=False), nn.BatchNorm2d(out_ch)
            )


class ShakeResNet(nn.Module):

    def __init__(self, depth, w_base, label):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6
        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        self.fc_out = nn.Linear(in_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


class ShakeBottleNeck(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        super(ShakeBottleNeck, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = None if self.equal_io else Shortcut(in_ch, out_ch,
            stride=stride)
        self.branch1 = self._make_branch(in_ch, mid_ch, out_ch, cardinary,
            stride)
        self.branch2 = self._make_branch(in_ch, mid_ch, out_ch, cardinary,
            stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        return nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, padding=0, bias=
            False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False), nn.
            Conv2d(mid_ch, mid_ch, 3, padding=1, stride=stride, groups=
            cardinary, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace
            =False), nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNeXt(nn.Module):

    def __init__(self, depth, w_base, cardinary, label):
        super(ShakeResNeXt, self).__init__()
        n_units = (depth - 2) // 9
        n_chs = [64, 128, 256, 1024]
        self.n_chs = n_chs
        self.in_ch = n_chs[0]
        self.c_in = nn.Conv2d(3, n_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, n_chs[0], w_base, cardinary)
        self.layer2 = self._make_layer(n_units, n_chs[1], w_base, cardinary, 2)
        self.layer3 = self._make_layer(n_units, n_chs[2], w_base, cardinary, 2)
        self.fc_out = nn.Linear(n_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.n_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, n_ch, w_base, cardinary, stride=1):
        layers = []
        mid_ch, out_ch = n_ch * (w_base // 64) * cardinary, n_ch * 4
        for i in range(n_units):
            layers.append(ShakeBottleNeck(self.in_ch, mid_ch, out_ch,
                cardinary, stride=stride))
            self.in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)
        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)
        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)
        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class WideBasic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
            bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n,
            dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n,
            dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n,
            dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TpuBatchNormalization(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True):
        super(TpuBatchNormalization, self).__init__()
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=
            torch.long))
        self.eps = eps
        self.momentum = momentum

    def _reduce_avg(self, t):
        dist.all_reduce(t, dist.ReduceOp.SUM)
        t.mul_(1.0 / dist.get_world_size())

    def forward(self, input):
        if not self.training or not dist.is_initialized():
            bn = (input - self.running_mean.view(1, self.running_mean.shape
                [0], 1, 1)) / torch.sqrt(self.running_var.view(1, self.
                running_var.shape[0], 1, 1) + self.eps)
            return bn.mul(self.weight.view(1, self.weight.shape[0], 1, 1)).add(
                self.bias.view(1, self.bias.shape[0], 1, 1))
        shard_mean, shard_invstd = torch.batch_norm_stats(input, self.eps)
        shard_vars = (1.0 / shard_invstd) ** 2 - self.eps
        shard_square_of_mean = torch.mul(shard_mean, shard_mean)
        shard_mean_of_square = shard_vars + shard_square_of_mean
        group_mean = shard_mean.clone().detach()
        self._reduce_avg(group_mean)
        group_mean_of_square = shard_mean_of_square.clone().detach()
        self._reduce_avg(group_mean_of_square)
        group_vars = group_mean_of_square - torch.mul(group_mean, group_mean)
        group_mean = group_mean.detach()
        group_vars = group_vars.detach()
        self.running_mean.mul_(1.0 - self.momentum).add_(group_mean.mul(
            self.momentum))
        self.running_var.mul_(1.0 - self.momentum).add_(group_vars.mul(self
            .momentum))
        self.num_batches_tracked.add_(1)
        bn = (input - group_mean.view(1, group_mean.shape[0], 1, 1)
            ) / torch.sqrt(group_vars.view(1, group_vars.shape[0], 1, 1) +
            self.eps)
        return bn.mul(self.weight.view(1, self.weight.shape[0], 1, 1)).add(self
            .bias.view(1, self.bias.shape[0], 1, 1))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kakaobrain_fast_autoaugment(_paritybench_base):
    pass
    def test_000(self):
        self._check(CondConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(RoutingFn(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MemoryEfficientSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Conv2dDynamicSamePadding(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ShakeDrop(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(ShakeBlock(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ShakeResNet(*[], **{'depth': 1, 'w_base': 4, 'label': 4}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_009(self):
        self._check(ShakeBottleNeck(*[], **{'in_ch': 4, 'mid_ch': 4, 'out_ch': 4, 'cardinary': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(ShakeResNeXt(*[], **{'depth': 1, 'w_base': 4, 'cardinary': 4, 'label': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_011(self):
        self._check(Shortcut(*[], **{'in_ch': 4, 'out_ch': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(WideBasic(*[], **{'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(TpuBatchNormalization(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

